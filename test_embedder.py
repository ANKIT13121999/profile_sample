import json
import base64
import io
import numpy as np
from PIL import Image
from inter.core.clients.xorclient import xorclient  # Your internal client
from xyz.types import Human, Image as xyzImage  # Your custom types
from xyz import chat  # Your custom multimodal model
import pandas as pd
from typing import Dict, List, Any
import os

class ChunkEmbeddingProcessor:
    def __init__(self, 
                 embedding_model_name: str = "bembedd-1rg",
                 caption_model_name: str = "my_custom_model"):
        """
        Initialize the embedding processor
        
        Args:
            embedding_model_name: Your internal model name for embeddings
            caption_model_name: Your custom model name for image captioning
        """
        print("Loading models...")
        
        # Initialize your internal embedding client
        self.embedding_client = xorclient()
        self.embedding_model_name = embedding_model_name
        
        # Set up custom image captioning model
        self.caption_model_name = caption_model_name
        chat.model_id = caption_model_name
        
        print("Models loaded successfully!")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using your internal model"""
        try:
            response = self.embedding_client.get_embedding(
                input=text, 
                model_name=self.embedding_model_name
            )
            # Assuming the response contains the embedding in a specific format
            # You may need to adjust this based on your actual response structure
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            elif isinstance(response, list):
                return response
            else:
                # If response structure is different, adjust accordingly
                print(f"Unexpected response format: {type(response)}")
                return response
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for i, text in enumerate(texts):
            print(f"Generating embedding {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Fallback to empty list or zeros if embedding fails
                embeddings.append([])
        return embeddings
    
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
    
    def save_images_from_base64(self, image_chunks: List[Dict], output_dir: str = "extracted_images") -> Dict[str, str]:
        """Convert base64 images to files and return mapping of chunk_id to file_path"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        chunk_id_to_path = {}
        
        for i, img_chunk in enumerate(image_chunks):
            try:
                # Convert base64 to image
                image = self.base64_to_image(img_chunk["image_data"])
                
                if image is not None:
                    # Save image
                    filename = f"{img_chunk['chunk_id']}.png"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath)
                    
                    # Store mapping
                    chunk_id_to_path[img_chunk['chunk_id']] = filepath
                    print(f"Saved: {filepath}")
                    
            except Exception as e:
                print(f"Error saving image {i}: {e}")
        
        print(f"Saved {len(chunk_id_to_path)} images to {output_dir}/")
        return chunk_id_to_path
    
    def generate_image_caption(self, image_path: str, custom_prompt: str = "What do you see") -> str:
        """Generate caption for an image using your custom multimodal model"""
        try:
            # Create AFM Image object from file path
            afm_image = xyzImage.from_url(image_path)
            
            # Create multimodal message with image and text prompt
            messages = [Human(contents=[afm_image, custom_prompt])]
            
            # Get response from your custom model
            response = chat(messages)
            
            # Extract caption from response
            caption = response.content if hasattr(response, 'content') else str(response)
            return caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return "Image caption could not be generated"
    
    def process_text_chunks(self, text_chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks using internal model"""
        print("Processing text chunks...")
        
        processed_chunks = []
        texts = [chunk["content"] for chunk in text_chunks]
        
        if texts:
            # Generate embeddings using your internal model
            embeddings = self.get_embeddings_batch(texts)
            
            for i, chunk in enumerate(text_chunks):
                processed_chunk = chunk.copy()
                if i < len(embeddings) and embeddings[i]:
                    processed_chunk["embedding"] = embeddings[i]
                    processed_chunk["embedding_model"] = self.embedding_model_name
                else:
                    processed_chunk["embedding_error"] = "Failed to generate embedding"
                processed_chunks.append(processed_chunk)
        
        print(f"Generated embeddings for {len(processed_chunks)} text chunks")
        return processed_chunks
    
    def process_image_chunks(self, image_chunks: List[Dict], output_dir: str = "extracted_images") -> List[Dict]:
        """Process image chunks - save images, generate captions and embeddings"""
        print("Processing image chunks...")
        
        # First, save all base64 images to files
        print("Converting base64 images to files...")
        chunk_id_to_path = self.save_images_from_base64(image_chunks, output_dir)
        
        processed_chunks = []
        
        for i, chunk in enumerate(image_chunks):
            try:
                print(f"Processing image {i+1}/{len(image_chunks)}")
                
                chunk_id = chunk['chunk_id']
                
                # Get the saved image path
                if chunk_id in chunk_id_to_path:
                    image_path = chunk_id_to_path[chunk_id]
                    
                    # Generate caption from the saved image file
                    caption = self.generate_image_caption(image_path)
                    
                    # Create combined text for embedding
                    combined_text = f"{chunk['image_description']}. {caption}"
                    
                    # Generate embedding for the combined text using internal model
                    embedding = self.get_embedding(combined_text)
                    
                    # Create processed chunk
                    processed_chunk = chunk.copy()
                    processed_chunk["generated_caption"] = caption
                    processed_chunk["combined_description"] = combined_text
                    processed_chunk["image_path"] = image_path
                    processed_chunk["image_source"] = "saved_file"
                    
                    if embedding is not None:
                        processed_chunk["embedding"] = embedding
                        processed_chunk["embedding_model"] = self.embedding_model_name
                    else:
                        processed_chunk["embedding_error"] = "Failed to generate embedding"
                    
                    processed_chunks.append(processed_chunk)
                else:
                    # Image saving failed, add error chunk
                    processed_chunk = chunk.copy()
                    processed_chunk["generated_caption"] = "Error: Image could not be saved"
                    processed_chunk["error"] = "Image file not created"
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
                
                # Generate embedding using internal model
                embedding = self.get_embedding(combined_text)
                
                # Create processed chunk
                processed_chunk = chunk.copy()
                processed_chunk["text_representation"] = combined_text
                
                if embedding is not None:
                    processed_chunk["embedding"] = embedding
                    processed_chunk["embedding_model"] = self.embedding_model_name
                else:
                    processed_chunk["embedding_error"] = "Failed to generate embedding"
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                print(f"Error processing table chunk: {e}")
                # Still add the chunk but without embedding
                processed_chunk = chunk.copy()
                processed_chunk["error"] = str(e)
                processed_chunks.append(processed_chunk)
        
        print(f"Generated embeddings for {len(processed_chunks)} table chunks")
        return processed_chunks
    
    def process_all_chunks(self, json_file_path: str, output_file: str = "chunks_with_embeddings.json", 
                         images_output_dir: str = "extracted_images"):
        """Process all chunks and generate embeddings"""
        print(f"Loading chunks from {json_file_path}")
        
        # Load chunks data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        print("Chunk counts:")
        print(f"- Text chunks: {len(chunks_data.get('text_chunks', []))}")
        print(f"- Image chunks: {len(chunks_data.get('image_chunks', []))}")
        print(f"- Table chunks: {len(chunks_data.get('table_chunks', []))}")
        print(f"- Images will be saved to: {images_output_dir}/")
        
        # Process each type of chunk
        processed_data = {}
        
        # Process text chunks
        if chunks_data.get("text_chunks"):
            processed_data["text_chunks"] = self.process_text_chunks(chunks_data["text_chunks"])
        
        # Process image chunks (this will save images first, then generate captions)
        if chunks_data.get("image_chunks"):
            processed_data["image_chunks"] = self.process_image_chunks(
                chunks_data["image_chunks"], 
                output_dir=images_output_dir
            )
        
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
                failed_embeddings = sum(1 for chunk in chunks if "embedding_error" in chunk)
                print(f"  Successful embeddings: {successful_embeddings}")
                print(f"  Failed embeddings: {failed_embeddings}")
                print(f"  Embedding model: {self.embedding_model_name}")
                
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
    """Example usage of the ChunkEmbeddingProcessor with internal models"""
    
    # Initialize processor with your internal models
    processor = ChunkEmbeddingProcessor(
        embedding_model_name="bembedd-1rg",  # Your internal embedding model name
        caption_model_name="my_custom_model"  # Your custom captioning model name
    )
    
    # Process chunks from JSON file (this will save images and generate captions)
    json_file = "pdf_chunks.json"  # Replace with your JSON file path
    processed_chunks = processor.process_all_chunks(
        json_file,
        output_file="chunks_with_embeddings.json",
        images_output_dir="extracted_images"
    )
    
    return processed_chunks

if __name__ == "__main__":
    print("Required packages:")
    print("pip install pillow")
    print("Note: Make sure your internal packages are available:")
    print("- inter.core.clients.xorclient")
    print("- xyz.types")
    print("- xyz")
    
    print("\nReplace 'pdf_chunks.json' with your actual JSON file path")
    print("Replace model names with your actual model names")
    
    # Uncomment to run:
    # processed_chunks = main()