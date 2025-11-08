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
import os
from pathlib import Path

@dataclass
class TextChunk:
    content: str
    chunk_id: str
    chunk_type: str = "text"
    page_number: int = None
    pdf_source: str = None  # NEW: Track which PDF this came from
    metadata: Dict[str, Any] = None

@dataclass
class ImageChunk:
    image_data: str  # base64 encoded
    image_description: str
    chunk_id: str
    chunk_type: str = "image"
    page_number: int = None
    pdf_source: str = None  # NEW: Track which PDF this came from
    bbox: Tuple[float, float, float, float] = None
    metadata: Dict[str, Any] = None

@dataclass
class TableChunk:
    table_data: List[List[str]]
    table_html: str
    table_description: str
    chunk_id: str
    chunk_type: str = "table"
    page_number: int = None
    pdf_source: str = None  # NEW: Track which PDF this came from
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
    
    def generate_chunk_id(self, content: str, chunk_type: str, page_num: int, pdf_name: str) -> str:
        """Generate unique chunk ID including PDF source"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()[:6]
        return f"{chunk_type}_{pdf_hash}_{page_num}_{content_hash}"
    
    def extract_text_chunks(self, text: str, page_num: int, pdf_source: str) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence or word boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size * 0.5:
                    end = sentence_end + 1
                else:
                    word_end = text.rfind(' ', start, end)
                    if word_end != -1 and word_end > start + self.chunk_size * 0.5:
                        end = word_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = self.generate_chunk_id(chunk_text, "text", page_num, pdf_source)
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    pdf_source=pdf_source,
                    metadata={"char_start": start, "char_end": end}
                ))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_images(self, page, page_num: int, pdf_source: str) -> List[ImageChunk]:
        """Extract images from a PDF page"""
        image_chunks = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data))
                    
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                    bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1) if img_rect else None
                    
                    description = f"Image {img_index + 1} on page {page_num + 1} from {Path(pdf_source).name}"
                    
                    chunk_id = self.generate_chunk_id(f"image_{img_index}_{page_num}", "image", page_num, pdf_source)
                    
                    image_chunks.append(ImageChunk(
                        image_data=img_base64,
                        image_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        pdf_source=pdf_source,
                        bbox=bbox,
                        metadata={
                            "image_index": img_index,
                            "width": pix.width,
                            "height": pix.height,
                            "colorspace": pix.colorspace.name if pix.colorspace else "unknown"
                        }
                    ))
                
                pix = None
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num} of {Path(pdf_source).name}: {e}")
                continue
        
        return image_chunks
    
    def extract_tables(self, page, page_num: int, pdf_source: str) -> List[TableChunk]:
        """Extract tables from a PDF page"""
        table_chunks = []
        
        try:
            tabs = page.find_tables()
            
            for tab_index, tab in enumerate(tabs):
                try:
                    table_data = tab.extract()
                    
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    cleaned_data = []
                    for row in table_data:
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        if any(cleaned_row):
                            cleaned_data.append(cleaned_row)
                    
                    if len(cleaned_data) < 2:
                        continue
                    
                    df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
                    table_html = df.to_html(index=False, escape=False)
                    
                    num_rows, num_cols = len(cleaned_data) - 1, len(cleaned_data[0])
                    description = f"Table {tab_index + 1} on page {page_num + 1} from {Path(pdf_source).name} with {num_rows} rows and {num_cols} columns"
                    
                    chunk_id = self.generate_chunk_id(f"table_{tab_index}_{page_num}", "table", page_num, pdf_source)
                    
                    table_chunks.append(TableChunk(
                        table_data=cleaned_data,
                        table_html=table_html,
                        table_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        pdf_source=pdf_source,
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
        Chunk a single PDF into text, images, and tables
        
        Returns:
            Dict containing lists of different chunk types
        """
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            
            all_chunks = {
                "text_chunks": [],
                "image_chunks": [],
                "table_chunks": [],
                "pdf_source": pdf_path,
                "pdf_name": pdf_name
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  Processing page {page_num + 1}/{len(doc)} of {pdf_name}")
                
                # Extract text
                text = page.get_text()
                text_chunks = self.extract_text_chunks(text, page_num, pdf_path)
                all_chunks["text_chunks"].extend(text_chunks)
                
                # Extract images
                image_chunks = self.extract_images(page, page_num, pdf_path)
                all_chunks["image_chunks"].extend(image_chunks)
                
                # Extract tables
                table_chunks = self.extract_tables(page, page_num, pdf_path)
                all_chunks["table_chunks"].extend(table_chunks)
            
            doc.close()
            
            print(f"  ✓ {pdf_name}: {len(all_chunks['text_chunks'])} text, {len(all_chunks['image_chunks'])} images, {len(all_chunks['table_chunks'])} tables")
            
            return all_chunks
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {"text_chunks": [], "image_chunks": [], "table_chunks": [], "pdf_source": pdf_path, "error": str(e)}
    
    def chunk_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple PDFs at once
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Dict containing all chunks organized by type and by PDF
        """
        print(f"\n{'='*60}")
        print(f"Processing {len(pdf_paths)} PDF files")
        print(f"{'='*60}\n")
        
        all_chunks_combined = {
            "text_chunks": [],
            "image_chunks": [],
            "table_chunks": [],
            "pdf_sources": [],
            "per_pdf_stats": {}
        }
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\n[{i}/{len(pdf_paths)}] Processing: {os.path.basename(pdf_path)}")
            
            # Process single PDF
            pdf_chunks = self.chunk_pdf(pdf_path)
            
            # Combine chunks
            all_chunks_combined["text_chunks"].extend(pdf_chunks["text_chunks"])
            all_chunks_combined["image_chunks"].extend(pdf_chunks["image_chunks"])
            all_chunks_combined["table_chunks"].extend(pdf_chunks["table_chunks"])
            
            # Track PDF sources
            if pdf_path not in all_chunks_combined["pdf_sources"]:
                all_chunks_combined["pdf_sources"].append(pdf_path)
            
            # Store per-PDF statistics
            all_chunks_combined["per_pdf_stats"][pdf_path] = {
                "pdf_name": os.path.basename(pdf_path),
                "text_chunks": len(pdf_chunks["text_chunks"]),
                "image_chunks": len(pdf_chunks["image_chunks"]),
                "table_chunks": len(pdf_chunks["table_chunks"]),
                "total_chunks": len(pdf_chunks["text_chunks"]) + len(pdf_chunks["image_chunks"]) + len(pdf_chunks["table_chunks"])
            }
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETED!")
        print(f"{'='*60}")
        print(f"Total PDFs processed: {len(pdf_paths)}")
        print(f"Total text chunks: {len(all_chunks_combined['text_chunks'])}")
        print(f"Total image chunks: {len(all_chunks_combined['image_chunks'])}")
        print(f"Total table chunks: {len(all_chunks_combined['table_chunks'])}")
        print(f"Grand total: {len(all_chunks_combined['text_chunks']) + len(all_chunks_combined['image_chunks']) + len(all_chunks_combined['table_chunks'])} chunks")
        
        return all_chunks_combined
    
    def save_chunks_to_json(self, chunks: Dict[str, Any], output_path: str):
        """Save chunks to JSON file"""
        serializable_chunks = {
            "pdf_sources": chunks.get("pdf_sources", []),
            "per_pdf_stats": chunks.get("per_pdf_stats", {}),
            "text_chunks": [asdict(chunk) for chunk in chunks.get("text_chunks", [])],
            "image_chunks": [asdict(chunk) for chunk in chunks.get("image_chunks", [])],
            "table_chunks": [asdict(chunk) for chunk in chunks.get("table_chunks", [])]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Chunks saved to {output_path}")
    
    def print_chunk_summary(self, chunks: Dict[str, Any]):
        """Print summary of extracted chunks"""
        print("\n" + "="*60)
        print("DETAILED CHUNK SUMMARY")
        print("="*60)
        
        # Per-PDF breakdown
        if "per_pdf_stats" in chunks and chunks["per_pdf_stats"]:
            print("\n📄 PER-PDF BREAKDOWN:")
            for pdf_path, stats in chunks["per_pdf_stats"].items():
                print(f"\n  {stats['pdf_name']}:")
                print(f"    • Text chunks: {stats['text_chunks']}")
                print(f"    • Image chunks: {stats['image_chunks']}")
                print(f"    • Table chunks: {stats['table_chunks']}")
                print(f"    • Total: {stats['total_chunks']}")
        
        # Overall summary
        print("\n📊 OVERALL SUMMARY:")
        total_chunks = 0
        
        for chunk_type in ["text_chunks", "image_chunks", "table_chunks"]:
            chunk_list = chunks.get(chunk_type, [])
            count = len(chunk_list)
            total_chunks += count
            print(f"  • {chunk_type}: {count}")
            
            if chunk_list and count > 0:
                if chunk_type == "text_chunks":
                    avg_length = sum(len(chunk.content) for chunk in chunk_list) / len(chunk_list)
                    print(f"    - Average length: {avg_length:.0f} chars")
                    print(f"    - Sample: {chunk_list[0].content[:80]}...")
                
                elif chunk_type == "image_chunks":
                    print(f"    - Sample: {chunk_list[0].image_description}")
                
                elif chunk_type == "table_chunks":
                    print(f"    - Sample: {chunk_list[0].table_description}")
        
        print(f"\n  📦 GRAND TOTAL: {total_chunks} chunks")


def main_single_pdf():
    """Process a single PDF"""
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
    
    pdf_path = "your_document.pdf"  # Replace with your PDF path
    chunks = chunker.chunk_pdf(pdf_path)
    
    chunker.print_chunk_summary({"text_chunks": chunks["text_chunks"], 
                                  "image_chunks": chunks["image_chunks"],
                                  "table_chunks": chunks["table_chunks"]})
    
    chunker.save_chunks_to_json(chunks, "single_pdf_chunks.json")
    
    return chunks


def main_multiple_pdfs():
    """Process multiple PDFs at once"""
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
    
    # List of PDF paths to process
    pdf_paths = [
        "document1.pdf",
        "document2.pdf",
        "document3.pdf",
        # Add more PDF paths here
    ]
    
    # Process all PDFs
    all_chunks = chunker.chunk_multiple_pdfs(pdf_paths)
    
    # Print summary
    chunker.print_chunk_summary(all_chunks)
    
    # Save to JSON
    chunker.save_chunks_to_json(all_chunks, "multi_pdf_chunks.json")
    
    return all_chunks


def main_from_directory():
    """Process all PDFs in a directory"""
    chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
    
    # Directory containing PDFs
    pdf_directory = "./pdf_files"  # Replace with your directory
    
    # Find all PDFs in directory
    pdf_paths = []
    for file in os.listdir(pdf_directory):
        if file.lower().endswith('.pdf'):
            pdf_paths.append(os.path.join(pdf_directory, file))
    
    if not pdf_paths:
        print(f"No PDF files found in {pdf_directory}")
        return None
    
    print(f"Found {len(pdf_paths)} PDF files in {pdf_directory}")
    
    # Process all PDFs
    all_chunks = chunker.chunk_multiple_pdfs(pdf_paths)
    
    # Print summary
    chunker.print_chunk_summary(all_chunks)
    
    # Save to JSON
    output_file = f"all_pdfs_chunks_{len(pdf_paths)}files.json"
    chunker.save_chunks_to_json(all_chunks, output_file)
    
    return all_chunks


if __name__ == "__main__":
    print("PDF Chunker - Multi-PDF Support")
    print("Required packages: pip install PyMuPDF pandas pillow")
    print("\nChoose an option:")
    print("1. Process a single PDF: Uncomment main_single_pdf()")
    print("2. Process multiple specific PDFs: Uncomment main_multiple_pdfs()")
    print("3. Process all PDFs in a directory: Uncomment main_from_directory()")
    
    # Uncomment the one you want to use:
    
    # Option 1: Single PDF
    # chunks = main_single_pdf()
    
    # Option 2: Multiple specific PDFs
    # chunks = main_multiple_pdfs()
    
    # Option 3: All PDFs in directory
    # chunks = main_from_directory()
