import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import base64
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class TextChunk:
    content: str
    chunk_id: str
    chunk_type: str = "text"
    page_number: int = None
    metadata: Dict[str, Any] = None

@dataclass
class ImageChunk:
    image_data: str  # base64 encoded
    image_description: str
    chunk_id: str
    chunk_type: str = "image"
    page_number: int = None
    bbox: Tuple[float, float, float, float] = None  # (x0, y0, x1, y1)
    metadata: Dict[str, Any] = None

@dataclass
class TableChunk:
    table_data: List[List[str]]
    table_html: str
    table_description: str
    chunk_id: str
    chunk_type: str = "table"
    page_number: int = None
    metadata: Dict[str, Any] = None

class PDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF Chunker
        
        Args:
            chunk_size: Maximum characters per text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
    
    def generate_chunk_id(self, content: str, chunk_type: str, page_num: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{chunk_type}_{page_num}_{content_hash}"
    
    def extract_text_chunks(self, text: str, page_num: int) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size * 0.5:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end != -1 and word_end > start + self.chunk_size * 0.5:
                        end = word_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = self.generate_chunk_id(chunk_text, "text", page_num)
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    metadata={"char_start": start, "char_end": end}
                ))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_images(self, page, page_num: int) -> List[ImageChunk]:
        """Extract images from a PDF page"""
        image_chunks = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data))
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Get image position on page
                    img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                    bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1) if img_rect else None
                    
                    # Generate description (you might want to use OCR or image captioning here)
                    description = f"Image {img_index + 1} on page {page_num + 1}"
                    
                    chunk_id = self.generate_chunk_id(f"image_{img_index}_{page_num}", "image", page_num)
                    
                    image_chunks.append(ImageChunk(
                        image_data=img_base64,
                        image_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        bbox=bbox,
                        metadata={
                            "image_index": img_index,
                            "width": pix.width,
                            "height": pix.height,
                            "colorspace": pix.colorspace.name if pix.colorspace else "unknown"
                        }
                    ))
                
                pix = None  # Clean up
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")
                continue
        
        return image_chunks
    
    def extract_tables(self, page, page_num: int) -> List[TableChunk]:
        """Extract tables from a PDF page using PyMuPDF"""
        table_chunks = []
        
        try:
            # Find tables on the page
            tabs = page.find_tables()
            
            for tab_index, tab in enumerate(tabs):
                try:
                    # Extract table data
                    table_data = tab.extract()
                    
                    if not table_data or len(table_data) < 2:  # Skip empty or single-row tables
                        continue
                    
                    # Clean the table data
                    cleaned_data = []
                    for row in table_data:
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        if any(cleaned_row):  # Skip completely empty rows
                            cleaned_data.append(cleaned_row)
                    
                    if len(cleaned_data) < 2:
                        continue
                    
                    # Convert to HTML for better representation
                    df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
                    table_html = df.to_html(index=False, escape=False)
                    
                    # Generate description
                    num_rows, num_cols = len(cleaned_data) - 1, len(cleaned_data[0])
                    description = f"Table {tab_index + 1} on page {page_num + 1} with {num_rows} rows and {num_cols} columns"
                    
                    chunk_id = self.generate_chunk_id(f"table_{tab_index}_{page_num}", "table", page_num)
                    
                    table_chunks.append(TableChunk(
                        table_data=cleaned_data,
                        table_html=table_html,
                        table_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        metadata={
                            "table_index": tab_index,
                            "num_rows": num_rows,
                            "num_cols": num_cols,
                            "bbox": tab.bbox
                        }
                    ))
                    
                except Exception as e:
                    print(f"Error processing table {tab_index} on page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error finding tables on page {page_num}: {e}")
        
        return table_chunks
    
    def chunk_pdf(self, pdf_path: str) -> Dict[str, List]:
        """
        Main method to chunk PDF into text, images, and tables
        
        Returns:
            Dict containing lists of different chunk types
        """
        try:
            doc = fitz.open(pdf_path)
            all_chunks = {
                "text_chunks": [],
                "image_chunks": [],
                "table_chunks": []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"Processing page {page_num + 1}/{len(doc)}")
                
                # Extract text
                text = page.get_text()
                text_chunks = self.extract_text_chunks(text, page_num)
                all_chunks["text_chunks"].extend(text_chunks)
                
                # Extract images
                image_chunks = self.extract_images(page, page_num)
                all_chunks["image_chunks"].extend(image_chunks)
                
                # Extract tables
                table_chunks = self.extract_tables(page, page_num)
                all_chunks["table_chunks"].extend(table_chunks)
            
            doc.close()
            
            print(f"\nExtraction completed!")
            print(f"Text chunks: {len(all_chunks['text_chunks'])}")
            print(f"Image chunks: {len(all_chunks['image_chunks'])}")
            print(f"Table chunks: {len(all_chunks['table_chunks'])}")
            
            return all_chunks
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {"text_chunks": [], "image_chunks": [], "table_chunks": []}
    
    def save_chunks_to_json(self, chunks: Dict[str, List], output_path: str):
        """Save chunks to JSON file"""
        serializable_chunks = {}
        
        for chunk_type, chunk_list in chunks.items():
            serializable_chunks[chunk_type] = [asdict(chunk) for chunk in chunk_list]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to {output_path}")
    
    def print_chunk_summary(self, chunks: Dict[str, List]):
        """Print summary of extracted chunks"""
        print("\n" + "="*50)
        print("CHUNK SUMMARY")
        print("="*50)
        
        for chunk_type, chunk_list in chunks.items():
            print(f"\n{chunk_type.upper()}: {len(chunk_list)} items")
            
            if chunk_list:
                if chunk_type == "text_chunks":
                    avg_length = sum(len(chunk.content) for chunk in chunk_list) / len(chunk_list)
                    print(f"  - Average text length: {avg_length:.0f} characters")
                    print(f"  - Sample: {chunk_list[0].content[:100]}...")
                
                elif chunk_type == "image_chunks":
                    print(f"  - Sample description: {chunk_list[0].image_description}")
                
                elif chunk_type == "table_chunks":
                    if chunk_list:
                        sample_table = chunk_list[0]
                        print(f"  - Sample: {sample_table.table_description}")
                        print(f"  - First table preview:")
                        for i, row in enumerate(sample_table.table_data[:3]):
                            print(f"    {row}")
                            if i == 2 and len(sample_table.table_data) > 3:
                                print(f"    ... ({len(sample_table.table_data)-3} more rows)")

# Example usage
def main():
    # Initialize chunker
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
    
    # Process PDF
    pdf_path = "sample_profiles_fixed.pdf"  # Replace with your PDF path
    chunks = chunker.chunk_pdf(pdf_path)
    
    # Print summary
    chunker.print_chunk_summary(chunks)
    
    # Save to JSON
    chunker.save_chunks_to_json(chunks, "pdf_chunks.json")
    
    return chunks

if __name__ == "__main__":
    # You'll need to install these packages:
    print("Required packages:")
    print("pip install PyMuPDF pandas pillow")
    print("\nReplace 'your_document.pdf' with your actual PDF path")
    
    # Uncomment to run:
    chunks = main()
