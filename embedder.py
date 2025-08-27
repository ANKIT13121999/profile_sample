import json
import base64
import io
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from typing import Dict, List, Any
import os

class ChunkEmbeddingProcessor:
    def __init__(self, 
                 text_model_name: str = "all-MiniLM-L6-v2",
                 image_caption_model: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the embedding processor
        
        Args:
            text_model_name: Sentence transformer model for text embeddings
            image_caption_model: Model for generating image captions
        """
        print("Loading models...")
        
        # Text embedding model
        self.text_model = SentenceTransformer(text_model_name)
        
        # Image captioning model
        self.caption_processor = BlipProcessor.from_pretrained(image_caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(image_caption_model)
        
        print("Models loaded successfully!")
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def save_images_from_chunks(self, chunks_data: Dict, output_dir: str = "extracted_images"):
        """Save all images from chunks to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_chunks = chunks_data.get("image_chunks", [])
        saved_images = []
        
        for i, img_chunk in enumerate(image_chunks):
            try:
                # Convert base64 to image
                image = self.base64_to_image(img_chunk["image_data"])
                
                if image is not None:
                    # Save image
                    filename = f"{img_chunk['chunk_id']}.png"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath)
                    
                    saved_images.append({
                        "chunk_id": img_chunk["chunk_id"],
                        "filepath": filepath,
                        "description": img_chunk["image_description"]
                    })
                    
                    print(f"Saved: {filepath}")
                    
            except Exception as e:
                print(f"Error saving image {i}: {e}")
        
        print(f"\nSaved {len(saved_images)} images to {output_dir}/")
        return saved_images
    
    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate caption for an image using BLIP model"""
        try:
            # Process image
            inputs = self.caption_processor(image, return_tensors="pt")
            
            # Generate caption
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=50)
            
            # Decode caption
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Image caption could not be generated"
    
    def process_text_chunks(self, text_chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks"""
        print("Processing text chunks...")
        
        processed_chunks = []
        texts = [chunk["content"] for chunk in text_chunks]
        
        if texts:
            # Generate embeddings in batch
            embeddings = self.text_model.encode(texts, convert_to_tensor=False)
            
            for i, chunk in enumerate(text_chunks):
                processed_chunk = chunk.copy()
                processed_chunk["embedding"] = embeddings[i].tolist()
                processed_chunk["embedding_model"] = "all-MiniLM-L6-v2"
                processed_chunks.append(processed_chunk)
        
        print(f"Generated embeddings for {len(processed_chunks)} text chunks")
        return processed_chunks
    
    def process_image_chunks(self, image_chunks: List[Dict]) -> List[Dict]:
        """Process image chunks - generate captions and embeddings"""
        print("Processing image chunks...")
        
        processed_chunks = []
        
        for i, chunk in enumerate(image_chunks):
            try:
                print(f"Processing image {i+1}/{len(image_chunks)}")
                
                # Convert base64 to image
                image = self.base64_to_image(chunk["image_data"])
                
                if image is not None:
                    # Generate caption
                    caption = self.generate_image_caption(image)
                    
                    # Create combined text for embedding
                    combined_text = f"{chunk['image_description']}. {caption}"
                    
                    # Generate embedding for the combined text
                    embedding = self.text_model.encode(combined_text, convert_to_tensor=False)
                    
                    # Create processed chunk
                    processed_chunk = chunk.copy()
                    processed_chunk["generated_caption"] = caption
                    processed_chunk["combined_description"] = combined_text
                    processed_chunk["embedding"] = embedding.tolist()
                    processed_chunk["embedding_model"] = "all-MiniLM-L6-v2"
                    
                    processed_chunks.append(processed_chunk)
                    
            except Exception as e:
                print(f"Error processing image chunk {i}: {e}")
                # Still add the chunk but without embedding
                processed_chunk = chunk.copy()
                processed_chunk["generated_caption"] = "Error generating caption"
                processed_chunk["error"] = str(e)
                processed_chunks.append(processed_chunk)
        
        print(f"Generated embeddings for {len(processed_chunks)} image chunks")
        return processed_chunks
    
    def process_table_chunks(self, table_chunks: List[Dict]) -> List[Dict]:
        """Process table chunks - convert to text and generate embeddings"""
        print("Processing table chunks...")
        
        processed_chunks = []
        
        for chunk in table_chunks:
            try:
                # Convert table to text representation
                table_data = chunk["table_data"]
                
                # Create multiple text representations
                text_representations = []
                
                # 1. Table description
                text_representations.append(chunk["table_description"])
                
                # 2. Column headers
                if table_data and len(table_data) > 0:
                    headers = table_data[0]
                    text_representations.append(f"Table columns: {', '.join(headers)}")
                
                # 3. Row-by-row content
                if len(table_data) > 1:
                    for i, row in enumerate(table_data[1:], 1):
                        row_text = " | ".join([f"{headers[j] if j < len(headers) else f'Col{j}'}: {cell}" 
                                             for j, cell in enumerate(row)])
                        text_representations.append(f"Row {i}: {row_text}")
                
                # Combine all text representations
                combined_text = "\n".join(text_representations)
                
                # Generate embedding
                embedding = self.text_model.encode(combined_text, convert_to_tensor=False)
                
                # Create processed chunk
                processed_chunk = chunk.copy()
                processed_chunk["text_representation"] = combined_text
                processed_chunk["embedding"] = embedding.tolist()
                processed_chunk["embedding_model"] = "all-MiniLM-L6-v2"
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                print(f"Error processing table chunk: {e}")
                # Still add the chunk but without embedding
                processed_chunk = chunk.copy()
                processed_chunk["error"] = str(e)
                processed_chunks.append(processed_chunk)
        
        print(f"Generated embeddings for {len(processed_chunks)} table chunks")
        return processed_chunks
    
    def process_all_chunks(self, json_file_path: str, output_file: str = "chunks_with_embeddings.json"):
        """Process all chunks and generate embeddings"""
        print(f"Loading chunks from {json_file_path}")
        
        # Load chunks data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        print("Chunk counts:")
        print(f"- Text chunks: {len(chunks_data.get('text_chunks', []))}")
        print(f"- Image chunks: {len(chunks_data.get('image_chunks', []))}")
        print(f"- Table chunks: {len(chunks_data.get('table_chunks', []))}")
        
        # Process each type of chunk
        processed_data = {}
        
        # Process text chunks
        if chunks_data.get("text_chunks"):
            processed_data["text_chunks"] = self.process_text_chunks(chunks_data["text_chunks"])
        
        # Process image chunks
        if chunks_data.get("image_chunks"):
            processed_data["image_chunks"] = self.process_image_chunks(chunks_data["image_chunks"])
        
        # Process table chunks
        if chunks_data.get("table_chunks"):
            processed_data["table_chunks"] = self.process_table_chunks(chunks_data["table_chunks"])
        
        # Save processed chunks
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed chunks saved to: {output_file}")
        
        # Generate summary
        self.generate_summary(processed_data)
        
        return processed_data
    
    def generate_summary(self, processed_data: Dict):
        """Generate a summary of the processed chunks"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        for chunk_type, chunks in processed_data.items():
            print(f"\n{chunk_type.upper()}:")
            print(f"  Total chunks: {len(chunks)}")
            
            if chunks:
                # Count successful embeddings
                successful_embeddings = sum(1 for chunk in chunks if "embedding" in chunk)
                print(f"  Successful embeddings: {successful_embeddings}")
                
                if chunk_type == "image_chunks":
                    captions = [chunk.get("generated_caption", "") for chunk in chunks if "generated_caption" in chunk]
                    if captions:
                        print(f"  Sample caption: {captions[0]}")
                
                elif chunk_type == "table_chunks":
                    if "text_representation" in chunks[0]:
                        print(f"  Sample table text: {chunks[0]['text_representation'][:100]}...")

# Utility functions for working with embeddings
def save_images_only(json_file_path: str, output_dir: str = "extracted_images"):
    """Simple function to just extract and save images"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_chunks = chunks_data.get("image_chunks", [])
    
    for i, img_chunk in enumerate(image_chunks):
        try:
            # Decode base64
            image_bytes = base64.b64decode(img_chunk["image_data"])
            
            # Convert to PIL Image and save
            image = Image.open(io.BytesIO(image_bytes))
            filename = f"{img_chunk['chunk_id']}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            print(f"Saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving image {i}: {e}")

# Example usage
def main():
    """Example usage of the ChunkEmbeddingProcessor"""
    
    # Initialize processor
    processor = ChunkEmbeddingProcessor()
    
    # Process chunks from JSON file
    json_file = "pdf_chunks.json"  # Replace with your JSON file path
    processed_chunks = processor.process_all_chunks(json_file)
    
    # Optionally save images separately
    processor.save_images_from_chunks(processed_chunks)
    
    return processed_chunks

if __name__ == "__main__":
    # print("Required packages:")
    # print("pip install sentence-transformers transformers torch pillow")
    # print("pip install accelerate  # For better model loading")
    
    # print("\nReplace 'pdf_chunks.json' with your actual JSON file path")
    
    # # Uncomment to run:
    processed_chunks = main() 