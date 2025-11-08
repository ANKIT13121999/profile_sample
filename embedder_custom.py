import json
import base64
import io
import numpy as np
from PIL import Image
from inter.core.clients.xorclient import xorclient
from xyz.types import HumanMultimodalMessage, Image as AFMImage
from xyz import chat
import pandas as pd
from typing import Dict, List, Any
import os
from pathlib import Path

class ChunkEmbeddingProcessor:
    def __init__(self, 
                 embedding_model_name: str = "bembedd-1rg",
                 caption_model_name: str = "my_custom_model"):
        """
        Initialize the embedding processor for multi-PDF support
        
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
            
            # Handle different response formats
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            elif isinstance(response, list):
                return response
            else:
                print(f"Unexpected response format: {type(response)}")
                return response
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for i, text in enumerate(texts):
            print(f"  Generating embedding {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
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
        """
        Save all base64 images to files organized by PDF source
        Returns mapping of chunk_id to saved file path
        """
        print("Converting base64 images to files...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        chunk_id_to_path = {}
        
        # Group images by PDF source
        images_by_pdf = {}
        for img_chunk in image_chunks:
            pdf_source = img_chunk.get("pdf_source", "unknown")
            pdf_name = Path(pdf_source).stem
            
            if pdf_name not in images_by_pdf:
                images_by_pdf[pdf_name] = []
            images_by_pdf[pdf_name].append(img_chunk)
        
        print(f"  Organizing images from {len(images_by_pdf)} PDF(s)")
        
        # Save images
        for pdf_name, img_list in images_by_pdf.items():
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            if not os.path.exists(pdf_output_dir):
                os.makedirs(pdf_output_dir)
            
            print(f"  Converting {len(img_list)} images from {pdf_name}...")
            
            for i, img_chunk in enumerate(img_list):
                try:
                    image = self.base64_to_image(img_chunk["image_data"])
                    
                    if image is not None:
                        filename = f"{img_chunk['chunk_id']}.png"
                        filepath = os.path.join(pdf_output_dir, filename)
                        image.save(filepath)
                        
                        chunk_id_to_path[img_chunk['chunk_id']] = filepath
                        
                except Exception as e:
                    print(f"    Error saving image {i}: {e}")
        
        print(f"  ✓ Saved {len(chunk_id_to_path)} images to {output_dir}/")
        return chunk_id_to_path
    
    def generate_image_caption(self, image_path: str, custom_prompt: str = "Consider yourself as someone who can see. What do you see") -> str:
        """
        Generate caption for an image using your custom multimodal model
        Takes file path as input (not PIL Image)
        """
        try:
            # Create AFM Image object directly from file path
            afm_image = AFMImage.from_url(image_path)
            
            # Create multimodal message with image and text prompt
            messages = [HumanMultimodalMessage(contents=[afm_image, custom_prompt])]
            
            # Get response from your custom model
            response = chat(messages)
            
            # Extract caption from response
            caption = response.content if hasattr(response, 'content') else str(response)
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Image caption could not be generated"
    
    def process_text_chunks(self, text_chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks using internal model"""
        if not text_chunks:
            return []
        
        print(f"\nProcessing {len(text_chunks)} text chunks...")
        
        # Group by PDF source for better tracking
        chunks_by_pdf = {}
        for chunk in text_chunks:
            pdf_source = chunk.get("pdf_source", "unknown")
            if pdf_source not in chunks_by_pdf:
                chunks_by_pdf[pdf_source] = []
            chunks_by_pdf[pdf_source].append(chunk)
        
        print(f"  Text chunks from {len(chunks_by_pdf)} PDF(s)")
        
        processed_chunks = []
        
        for pdf_source, pdf_chunks in chunks_by_pdf.items():
            pdf_name = Path(pdf_source).name
            print(f"\n  Processing {len(pdf_chunks)} text chunks from {pdf_name}")
            
            texts = [chunk["content"] for chunk in pdf_chunks]
            
            if texts:
                # Generate embeddings using your internal model
                embeddings = self.get_embeddings_batch(texts)
                
                for i, chunk in enumerate(pdf_chunks):
                    processed_chunk = chunk.copy()
                    if i < len(embeddings) and embeddings[i]:
                        processed_chunk["embedding"] = embeddings[i]
                        processed_chunk["embedding_model"] = self.embedding_model_name
                    else:
                        processed_chunk["embedding_error"] = "Failed to generate embedding"
                    processed_chunks.append(processed_chunk)
        
        print(f"\n✓ Generated embeddings for {len(processed_chunks)} text chunks")
        return processed_chunks
    
    def process_image_chunks(self, image_chunks: List[Dict], output_dir: str = "extracted_images") -> List[Dict]:
        """
        Process image chunks - save images, generate captions from saved files, and embeddings
        """
        if not image_chunks:
            return []
        
        print(f"\nProcessing {len(image_chunks)} image chunks...")
        
        # First, save all base64 images to files
        print("Converting base64 images to files...")
        chunk_id_to_path = self.save_images_from_base64(image_chunks, output_dir)
        
        # Group by PDF source
        chunks_by_pdf = {}
        for chunk in image_chunks:
            pdf_source = chunk.get("pdf_source", "unknown")
            if pdf_source not in chunks_by_pdf:
                chunks_by_pdf[pdf_source] = []
            chunks_by_pdf[pdf_source].append(chunk)
        
        print(f"\n  Generating captions for images from {len(chunks_by_pdf)} PDF(s)")
        
        processed_chunks = []
        
        for pdf_source, pdf_chunks in chunks_by_pdf.items():
            pdf_name = Path(pdf_source).name
            print(f"\n  Processing {len(pdf_chunks)} image chunks from {pdf_name}")
            
            for i, chunk in enumerate(pdf_chunks):
                try:
                    if (i + 1) % 5 == 0 or (i + 1) == len(pdf_chunks):
                        print(f"    Processing image {i+1}/{len(pdf_chunks)}")
                    
                    chunk_id = chunk["chunk_id"]
                    
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
                    print(f"    Error processing image chunk {i}: {e}")
                    # Still add the chunk but without embedding
                    processed_chunk = chunk.copy()
                    processed_chunk["generated_caption"] = "Error generating caption"
                    processed_chunk["error"] = str(e)
                    processed_chunks.append(processed_chunk)
        
        print(f"\n✓ Generated embeddings for {len(processed_chunks)} image chunks")
        return processed_chunks
    
    def process_table_chunks(self, table_chunks: List[Dict]) -> List[Dict]:
        """Process table chunks - convert to text and generate embeddings"""
        if not table_chunks:
            return []
        
        print(f"\nProcessing {len(table_chunks)} table chunks...")
        
        # Group by PDF source
        chunks_by_pdf = {}
        for chunk in table_chunks:
            pdf_source = chunk.get("pdf_source", "unknown")
            if pdf_source not in chunks_by_pdf:
                chunks_by_pdf[pdf_source] = []
            chunks_by_pdf[pdf_source].append(chunk)
        
        print(f"  Table chunks from {len(chunks_by_pdf)} PDF(s)")
        
        processed_chunks = []
        
        for pdf_source, pdf_chunks in chunks_by_pdf.items():
            pdf_name = Path(pdf_source).name
            print(f"\n  Processing {len(pdf_chunks)} table chunks from {pdf_name}")
            
            for chunk in pdf_chunks:
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
                    print(f"    Error processing table chunk: {e}")
                    # Still add the chunk but without embedding
                    processed_chunk = chunk.copy()
                    processed_chunk["error"] = str(e)
                    processed_chunks.append(processed_chunk)
        
        print(f"\n✓ Generated embeddings for {len(processed_chunks)} table chunks")
        return processed_chunks
    
    def process_all_chunks(self, json_file_path: str, output_file: str = "chunks_with_embeddings.json"):
        """Process all chunks from multiple PDFs and generate embeddings"""
        print(f"\n{'='*60}")
        print(f"Loading chunks from {json_file_path}")
        print(f"{'='*60}")
        
        # Load chunks data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Display per-PDF statistics if available
        if "per_pdf_stats" in chunks_data:
            print("\n📄 PDF FILES LOADED:")
            for pdf_path, stats in chunks_data["per_pdf_stats"].items():
                print(f"  • {stats['pdf_name']}")
                print(f"    - Text: {stats['text_chunks']}, Images: {stats['image_chunks']}, Tables: {stats['table_chunks']}")
        
        print(f"\n📊 TOTAL CHUNKS TO PROCESS:")
        print(f"  • Text chunks: {len(chunks_data.get('text_chunks', []))}")
        print(f"  • Image chunks: {len(chunks_data.get('image_chunks', []))}")
        print(f"  • Table chunks: {len(chunks_data.get('table_chunks', []))}")
        print("  • Images will be saved to files first, then processed")
        
        # Process each type of chunk
        processed_data = {
            "pdf_sources": chunks_data.get("pdf_sources", []),
            "per_pdf_stats": chunks_data.get("per_pdf_stats", {})
        }
        
        # Process text chunks
        if chunks_data.get("text_chunks"):
            processed_data["text_chunks"] = self.process_text_chunks(chunks_data["text_chunks"])
        
        # Process image chunks (saves images first, then generates captions from files)
        if chunks_data.get("image_chunks"):
            processed_data["image_chunks"] = self.process_image_chunks(
                chunks_data["image_chunks"],
                output_dir="extracted_images"
            )
        
        # Process table chunks
        if chunks_data.get("table_chunks"):
            processed_data["table_chunks"] = self.process_table_chunks(chunks_data["table_chunks"])
        
        # Save processed chunks
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Processed chunks saved to: {output_file}")
        
        # Generate summary
        self.generate_summary(processed_data)
        
        return processed_data
    
    def generate_summary(self, processed_data: Dict):
        """Generate a summary of the processed chunks"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        # Per-PDF breakdown
        if "per_pdf_stats" in processed_data:
            print("\n📄 PROCESSED FILES:")
            for pdf_path, stats in processed_data["per_pdf_stats"].items():
                print(f"  • {stats['pdf_name']}: {stats['total_chunks']} chunks")
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS:")
        for chunk_type, chunks in processed_data.items():
            if chunk_type in ["text_chunks", "image_chunks", "table_chunks"] and chunks:
                print(f"\n  {chunk_type.upper()}:")
                print(f"    Total chunks: {len(chunks)}")
                
                # Count successful embeddings
                successful_embeddings = sum(1 for chunk in chunks if "embedding" in chunk)
                failed_embeddings = sum(1 for chunk in chunks if "embedding_error" in chunk)
                print(f"    Successful embeddings: {successful_embeddings}")
                print(f"    Failed embeddings: {failed_embeddings}")
                print(f"    Embedding model: {self.embedding_model_name}")
                
                if chunk_type == "image_chunks":
                    captions = [chunk.get("generated_caption", "") for chunk in chunks if "generated_caption" in chunk]
                    if captions:
                        print(f"    Sample caption: {captions[0][:80]}...")
                    
                    # Count images with saved paths
                    with_paths = sum(1 for chunk in chunks if "image_path" in chunk)
                    print(f"    Images saved to files: {with_paths}")
                
                elif chunk_type == "table_chunks":
                    if chunks and "text_representation" in chunks[0]:
                        print(f"    Sample table text: {chunks[0]['text_representation'][:100]}...")


# Utility function
def save_images_only(json_file_path: str, output_dir: str = "extracted_images"):
    """Simple function to extract and save images from multiple PDFs"""
    processor = ChunkEmbeddingProcessor()
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    image_chunks = chunks_data.get("image_chunks", [])
    chunk_id_to_path = processor.save_images_from_base64(image_chunks, output_dir)
    
    print(f"\nSaved {len(chunk_id_to_path)} images")
    return chunk_id_to_path


# Example usage
def main():
    """Example usage of the ChunkEmbeddingProcessor with multiple PDFs"""
    
    # Initialize processor with your internal models
    processor = ChunkEmbeddingProcessor(
        embedding_model_name="bembedd-1rg",  # Your internal embedding model name
        caption_model_name="my_custom_model"  # Your custom captioning model name
    )
    
    # Process chunks from JSON file (can contain multiple PDFs)
    json_file = "multi_pdf_chunks.json"  # Replace with your JSON file path from pdf_chunk.py
    processed_chunks = processor.process_all_chunks(
        json_file, 
        output_file="multi_pdf_embeddings.json"
    )
    
    return processed_chunks


if __name__ == "__main__":
    print("Chunk Embedding Processor - Multi-PDF Support")
    print("=" * 60)
    print("Required packages: pip install pillow")
    print("\nMake sure your internal packages are available:")
    print("- inter.core.clients.xorclient")
    print("- xyz.types (HumanMultimodalMessage, Image as AFMImage)")
    print("- xyz (chat)")
    
    print("\n" + "=" * 60)
    print("WORKFLOW:")
    print("=" * 60)
    print("1. Run pdf_chunk.py to create multi_pdf_chunks.json")
    print("2. Run this script to:")
    print("   - Save all base64 images as files (organized by PDF)")
    print("   - Generate captions using saved image files")
    print("   - Generate embeddings for all content")
    print("3. Output: multi_pdf_embeddings.json with embeddings")
    print("4. Images saved in: extracted_images/<pdf_name>/*.png")
    
    print("\nKey Feature:")
    print("✓ Images are saved to files FIRST")
    print("✓ Captions generated from file paths (not base64)")
    print("✓ Uses AFMImage.from_url(image_path) as in your original code")
    
    # Uncomment to run:
    # processed_chunks = main()
