import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import base64
import json
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import os
import tempfile
from datetime import datetime
import chromadb
from chromadb.config import Settings

# Import your custom modules (adjust based on your actual imports)
# from inter.core.clients.xorclient import xorclient
# from xyz import chat
# from xyz.types import HumanMultimodalMessage, Image as AFMImage

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TextChunk:
    content: str
    chunk_id: str
    chunk_type: str = "text"
    page_number: int = None
    source_file: str = None
    metadata: Dict[str, Any] = None

@dataclass
class ImageChunk:
    image_data: str
    image_description: str
    chunk_id: str
    chunk_type: str = "image"
    page_number: int = None
    source_file: str = None
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
    source_file: str = None
    metadata: Dict[str, Any] = None

# ============================================================================
# PDF CHUNKER (Modified for multiple files)
# ============================================================================

class MultiPDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.all_chunks = {
            "text_chunks": [],
            "image_chunks": [],
            "table_chunks": []
        }
    
    def generate_chunk_id(self, content: str, chunk_type: str, page_num: int, file_name: str) -> str:
        """Generate unique chunk ID with filename"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_name.encode()).hexdigest()[:6]
        return f"{chunk_type}_{file_hash}_{page_num}_{content_hash}"
    
    def extract_text_chunks(self, text: str, page_num: int, file_name: str) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
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
                chunk_id = self.generate_chunk_id(chunk_text, "text", page_num, file_name)
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    source_file=file_name,
                    metadata={"char_start": start, "char_end": end}
                ))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_images(self, page, page_num: int, file_name: str) -> List[ImageChunk]:
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
                    
                    description = f"Image {img_index + 1} on page {page_num + 1} from {file_name}"
                    
                    chunk_id = self.generate_chunk_id(f"image_{img_index}_{page_num}", "image", page_num, file_name)
                    
                    image_chunks.append(ImageChunk(
                        image_data=img_base64,
                        image_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        source_file=file_name,
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
                st.warning(f"Error extracting image {img_index} from page {page_num} in {file_name}: {e}")
        
        return image_chunks
    
    def extract_tables(self, page, page_num: int, file_name: str) -> List[TableChunk]:
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
                    description = f"Table {tab_index + 1} on page {page_num + 1} from {file_name} with {num_rows} rows and {num_cols} columns"
                    
                    chunk_id = self.generate_chunk_id(f"table_{tab_index}_{page_num}", "table", page_num, file_name)
                    
                    table_chunks.append(TableChunk(
                        table_data=cleaned_data,
                        table_html=table_html,
                        table_description=description,
                        chunk_id=chunk_id,
                        page_number=page_num,
                        source_file=file_name,
                        metadata={
                            "table_index": tab_index,
                            "num_rows": num_rows,
                            "num_cols": num_cols,
                            "bbox": tab.bbox
                        }
                    ))
                    
                except Exception as e:
                    st.warning(f"Error processing table {tab_index} on page {page_num} in {file_name}: {e}")
                    
        except Exception as e:
            st.warning(f"Error finding tables on page {page_num} in {file_name}: {e}")
        
        return table_chunks
    
    def chunk_single_pdf(self, pdf_bytes: bytes, file_name: str) -> Dict[str, List]:
        """Process a single PDF file"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            file_chunks = {
                "text_chunks": [],
                "image_chunks": [],
                "table_chunks": []
            }
            
            progress_bar = st.progress(0, text=f"Processing {file_name}...")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                text_chunks = self.extract_text_chunks(text, page_num, file_name)
                file_chunks["text_chunks"].extend(text_chunks)
                
                # Extract images
                image_chunks = self.extract_images(page, page_num, file_name)
                file_chunks["image_chunks"].extend(image_chunks)
                
                # Extract tables
                table_chunks = self.extract_tables(page, page_num, file_name)
                file_chunks["table_chunks"].extend(table_chunks)
                
                # Update progress
                progress_bar.progress((page_num + 1) / len(doc), 
                                     text=f"Processing {file_name} - Page {page_num + 1}/{len(doc)}")
            
            doc.close()
            progress_bar.empty()
            
            return file_chunks
            
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            return {"text_chunks": [], "image_chunks": [], "table_chunks": []}
    
    def chunk_multiple_pdfs(self, uploaded_files: List) -> Dict[str, List]:
        """Process multiple PDF files"""
        self.all_chunks = {
            "text_chunks": [],
            "image_chunks": [],
            "table_chunks": []
        }
        
        for uploaded_file in uploaded_files:
            st.write(f"📄 Processing: **{uploaded_file.name}**")
            
            # Read file bytes
            pdf_bytes = uploaded_file.read()
            
            # Process this PDF
            file_chunks = self.chunk_single_pdf(pdf_bytes, uploaded_file.name)
            
            # Aggregate chunks
            for chunk_type in self.all_chunks.keys():
                self.all_chunks[chunk_type].extend(file_chunks[chunk_type])
            
            st.success(f"✅ Completed: {uploaded_file.name} - "
                      f"Text: {len(file_chunks['text_chunks'])}, "
                      f"Images: {len(file_chunks['image_chunks'])}, "
                      f"Tables: {len(file_chunks['table_chunks'])}")
        
        return self.all_chunks

# ============================================================================
# EMBEDDING PROCESSOR (Simplified - Replace with your actual implementation)
# ============================================================================

class SimpleEmbeddingProcessor:
    """Simplified embedding processor - replace with your actual implementation"""
    
    def __init__(self):
        st.info("⚠️ Using simplified embedding processor. Replace with your actual xorclient implementation.")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate dummy embedding - REPLACE WITH YOUR ACTUAL MODEL"""
        # This is a placeholder - replace with your actual embedding generation
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Generate a dummy 384-dimensional embedding
        import random
        random.seed(text_hash)
        return [random.random() for _ in range(384)]
    
    def process_all_chunks(self, chunks: Dict[str, List]) -> Dict[str, List]:
        """Add embeddings to all chunks"""
        processed_chunks = {}
        
        total_chunks = sum(len(chunk_list) for chunk_list in chunks.values())
        progress_bar = st.progress(0, text="Generating embeddings...")
        current = 0
        
        for chunk_type, chunk_list in chunks.items():
            processed_list = []
            
            for chunk in chunk_list:
                chunk_dict = asdict(chunk)
                
                # Generate text for embedding
                if chunk_type == "text_chunks":
                    text = chunk_dict["content"]
                elif chunk_type == "image_chunks":
                    text = chunk_dict["image_description"]
                elif chunk_type == "table_chunks":
                    text = chunk_dict["table_description"]
                else:
                    text = str(chunk_dict)
                
                # Generate embedding
                embedding = self.get_embedding(text)
                chunk_dict["embedding"] = embedding
                chunk_dict["embedding_model"] = "simple_embedder"
                
                processed_list.append(chunk_dict)
                
                current += 1
                progress_bar.progress(current / total_chunks, 
                                    text=f"Generating embeddings... {current}/{total_chunks}")
            
            processed_chunks[chunk_type] = processed_list
        
        progress_bar.empty()
        return processed_chunks

# ============================================================================
# CHROMADB MANAGER (Simplified)
# ============================================================================

class SimplifiedChromaDBManager:
    def __init__(self, persist_directory: str = "./streamlit_chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="multi_pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    
    def insert_chunks(self, processed_chunks: Dict[str, List]) -> int:
        """Insert processed chunks into ChromaDB"""
        total_inserted = 0
        
        for chunk_type, chunk_list in processed_chunks.items():
            if not chunk_list:
                continue
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunk_list:
                try:
                    ids.append(chunk["chunk_id"])
                    embeddings.append(chunk["embedding"])
                    
                    # Prepare document content
                    if chunk_type == "text_chunks":
                        content = chunk["content"]
                    elif chunk_type == "image_chunks":
                        content = chunk["image_description"]
                    elif chunk_type == "table_chunks":
                        content = chunk["table_description"]
                    else:
                        content = str(chunk)
                    
                    documents.append(content)
                    
                    # Prepare metadata
                    metadata = {
                        "chunk_type": chunk_type.replace("_chunks", ""),
                        "page_number": chunk.get("page_number", -1),
                        "source_file": chunk.get("source_file", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                    metadatas.append(metadata)
                    
                except Exception as e:
                    st.warning(f"Error processing chunk: {e}")
                    continue
            
            if ids and embeddings:
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    total_inserted += len(ids)
                except Exception as e:
                    st.error(f"Error inserting {chunk_type}: {e}")
        
        return total_inserted
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search across all chunks"""
        # Generate query embedding (using same simple method - replace with actual)
        processor = SimpleEmbeddingProcessor()
        query_embedding = processor.get_embedding(query)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_count = self.collection.count()
        
        sample_results = self.collection.get(limit=min(100, total_count))
        
        file_counts = {}
        type_counts = {}
        
        if sample_results["metadatas"]:
            for metadata in sample_results["metadatas"]:
                source_file = metadata.get("source_file", "unknown")
                file_counts[source_file] = file_counts.get(source_file, 0) + 1
                
                chunk_type = metadata.get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        return {
            "total_chunks": total_count,
            "files": file_counts,
            "chunk_types": type_counts
        }

# ============================================================================
# STREAMLIT APP
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'chunks_processed' not in st.session_state:
        st.session_state.chunks_processed = False
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="Multi-PDF RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("📚 Multi-PDF Question Answering System")
    st.markdown("Upload multiple PDFs, process them, and ask questions across all documents")
    
    # Sidebar for settings and stats
    with st.sidebar:
        st.header("⚙️ Settings")
        
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        st.divider()
        
        if st.session_state.db_manager:
            st.header("📊 Database Stats")
            stats = st.session_state.db_manager.get_stats()
            st.metric("Total Chunks", stats["total_chunks"])
            
            st.subheader("Files Processed")
            for file, count in stats["files"].items():
                st.write(f"📄 {file}: {count} chunks")
            
            st.subheader("Chunk Types")
            for chunk_type, count in stats["chunk_types"].items():
                st.write(f"• {chunk_type}: {count}")
        
        st.divider()
        
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.db_manager:
                # Reset everything
                st.session_state.chunks_processed = False
                st.session_state.db_manager = None
                st.session_state.processed_chunks = None
                st.session_state.chat_history = []
                st.success("Database cleared!")
                st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Query & Chat", "📋 View Chunks"])
    
    # TAB 1: Upload & Process
    with tab1:
        st.header("Step 1: Upload PDF Files")
        
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size / 1024:.2f} KB)")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Process PDFs", type="primary", use_container_width=True):
                    with st.spinner("Processing PDFs..."):
                        # Step 1: Chunk PDFs
                        st.subheader("Step 1: Extracting & Chunking")
                        chunker = MultiPDFChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunks = chunker.chunk_multiple_pdfs(uploaded_files)
                        
                        # Show summary
                        st.success(f"✅ Extraction complete!")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Text Chunks", len(chunks["text_chunks"]))
                        col_b.metric("Images", len(chunks["image_chunks"]))
                        col_c.metric("Tables", len(chunks["table_chunks"]))
                        
                        # Step 2: Generate embeddings
                        st.subheader("Step 2: Generating Embeddings")
                        processor = SimpleEmbeddingProcessor()
                        processed_chunks = processor.process_all_chunks(chunks)
                        st.success("✅ Embeddings generated!")
                        
                        # Step 3: Store in ChromaDB
                        st.subheader("Step 3: Storing in Vector Database")
                        db_manager = SimplifiedChromaDBManager()
                        total_inserted = db_manager.insert_chunks(processed_chunks)
                        st.success(f"✅ Stored {total_inserted} chunks in database!")
                        
                        # Update session state
                        st.session_state.chunks_processed = True
                        st.session_state.db_manager = db_manager
                        st.session_state.processed_chunks = processed_chunks
                        
                        st.balloons()
                        st.success("🎉 All PDFs processed successfully! Go to 'Query & Chat' tab to ask questions.")
            
            with col2:
                if st.button("📊 View Processing Stats", use_container_width=True):
                    if st.session_state.processed_chunks:
                        st.json({
                            "text_chunks": len(st.session_state.processed_chunks["text_chunks"]),
                            "image_chunks": len(st.session_state.processed_chunks["image_chunks"]),
                            "table_chunks": len(st.session_state.processed_chunks["table_chunks"])
                        })
                    else:
                        st.info("Process PDFs first to see stats")
    
    # TAB 2: Query & Chat
    with tab2:
        st.header("💬 Ask Questions")
        
        if not st.session_state.chunks_processed or not st.session_state.db_manager:
            st.warning("⚠️ Please upload and process PDF files first (go to 'Upload & Process' tab)")
        else:
            # Query settings
            col1, col2 = st.columns([3, 1])
            with col1:
                num_results = st.slider("Number of results to retrieve", 3, 20, 5)
            with col2:
                filter_type = st.selectbox("Filter by type", ["All", "Text", "Image", "Table"])
            
            st.divider()
            
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat["query"])
                
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    
                    with st.expander(f"📚 View {len(chat['results'])} sources"):
                        for j, result in enumerate(chat['results'], 1):
                            st.markdown(f"**Source {j}** (Score: {result['score']:.3f})")
                            st.write(f"📄 File: {result['metadata']['source_file']}")
                            st.write(f"📃 Page: {result['metadata']['page_number'] + 1}")
                            st.write(f"🏷️ Type: {result['metadata']['chunk_type']}")
                            st.write(f"📝 Content: {result['content'][:200]}...")
                            st.divider()
            
            # Query input
            query = st.chat_input("Ask a question about your documents...")
            
            if query:
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching..."):
                        # Search in ChromaDB
                        results = st.session_state.db_manager.search(query, n_results=num_results)
                        
                        if not results:
                            st.error("No relevant documents found.")
                        else:
                            # Generate answer (simplified - replace with your actual model)
                            context = "\n\n".join([
                                f"Source {i+1} (from {r['metadata']['source_file']}, page {r['metadata']['page_number']+1}):\n{r['content']}"
                                for i, r in enumerate(results[:3])
                            ])
                            
                            answer = f"""Based on the retrieved documents, here's what I found:

{context}

**Note:** This is a simplified answer. Replace this with your actual AI model (xyz.chat) for comprehensive responses."""
                            
                            st.write(answer)
                            
                            # Show sources
                            with st.expander(f"📚 View {len(results)} sources"):
                                for i, result in enumerate(results, 1):
                                    st.markdown(f"**Source {i}** (Score: {result['score']:.3f})")
                                    st.write(f"📄 File: {result['metadata']['source_file']}")
                                    st.write(f"📃 Page: {result['metadata']['page_number'] + 1}")
                                    st.write(f"🏷️ Type: {result['metadata']['chunk_type']}")
                                    st.write(f"📝 Content: {result['content'][:200]}...")
                                    st.divider()
                            
                            # Save to chat history
                            st.session_state.chat_history.append({
                                "query": query,
                                "answer": answer,
                                "results": results
                            })
    
    # TAB 3: View Chunks
    with tab3:
        st.header("📋 Browse Processed Chunks")
        
        if not st.session_state.processed_chunks:
            st.warning("⚠️ No processed chunks available. Process some PDFs first.")
        else:
            chunk_type = st.selectbox("Select chunk type to view", 
                                     ["text_chunks", "image_chunks", "table_chunks"])
            
            chunks = st.session_state.processed_chunks[chunk_type]
            st.write(f"Total {chunk_type}: **{len(chunks)}**")
            
            if chunks:
                # Pagination
                items_per_page = 10
                total_pages = (len(chunks) - 1) // items_per_page + 1
                page = st.number_input("Page", 1, total_pages, 1)
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(chunks))
                
                st.write(f"Showing items {start_idx + 1}-{end_idx} of {len(chunks)}")
                
                for i, chunk in enumerate(chunks[start_idx:end_idx], start_idx + 1):
                    with st.expander(f"Chunk {i}: {chunk['chunk_id']}"):
                        st.write(f"**File:** {chunk['source_file']}")
                        st.write(f"**Page:** {chunk['page_number'] + 1}")
                        st.write(f"**Type:** {chunk_type.replace('_chunks', '')}")
                        
                        if chunk_type == "text_chunks":
                            st.text_area("Content", chunk['content'], height=150, key=f"text_{i}")
                        elif chunk_type == "image_chunks":
                            st.write(f"**Description:** {chunk['image_description']}")
                            # Display image if available
                            try:
                                img_data = base64.b64decode(chunk['image_data'])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img, caption=f"Image from page {chunk['page_number'] + 1}")
                            except:
                                st.warning("Could not display image")
                        elif chunk_type == "table_chunks":
                            st.write(f"**Description:** {chunk['table_description']}")
                            st.dataframe(pd.DataFrame(
                                chunk['table_data'][1:], 
                                columns=chunk['table_data'][0]
                            ))

if __name__ == "__main__":
    main()
