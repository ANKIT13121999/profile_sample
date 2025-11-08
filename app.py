import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

# Import your utility modules
from utils.pdf_chunker import PDFChunker
from utils.embedder import ChunkEmbeddingProcessor
from utils.rag_system import ChromaDBManager

# Page configuration
st.set_page_config(
    page_title="Multi-PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'chunks_file' not in st.session_state:
    st.session_state.chunks_file = None
if 'embeddings_file' not in st.session_state:
    st.session_state.embeddings_file = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    embedding_model = st.text_input(
        "Embedding Model",
        value="bembedd-1rg",
        help="Your organization's embedding model name"
    )
    
    caption_model = st.text_input(
        "Caption Model",
        value="my_custom_model",
        help="Your custom captioning model name"
    )
    
    # Chunking settings
    st.subheader("Chunking Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
    
    # Database settings
    st.subheader("Database Settings")
    persist_dir = st.text_input("ChromaDB Directory", value="./chroma_db")
    collection_name = st.text_input("Collection Name", value="multi_pdf_chunks")
    
    # Query settings
    st.subheader("Query Settings")
    n_results = st.slider("Results to Retrieve", 3, 20, 8)
    min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.0, 0.05)
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.processing_complete:
        st.success("✅ PDFs Processed")
    if st.session_state.db_initialized:
        st.success("✅ Database Ready")
    
    if st.session_state.pdf_files:
        st.info(f"📄 {len(st.session_state.pdf_files)} PDFs loaded")

# Main content
st.title("📚 Multi-PDF RAG System")
st.markdown("Upload multiple PDFs, process them, and ask questions across all documents!")

# Create tabs for different stages
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Query System", "📊 Analytics"])

# Tab 1: Upload & Process
with tab1:
    st.header("1️⃣ Upload PDF Files")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to process"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF file(s) uploaded")
        
        # Display uploaded files
        with st.expander("📄 Uploaded Files", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size / 1024:.2f} KB)")
        
        st.divider()
        
        # Processing button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                
                # Create temporary directory for PDFs
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    # Save uploaded files
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(file_path)
                    
                    st.session_state.pdf_files = [f.name for f in uploaded_files]
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Chunking
                        status_text.text("📄 Step 1/3: Extracting chunks from PDFs...")
                        progress_bar.progress(10)
                        
                        chunker = PDFChunker(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        with st.spinner("Processing PDFs..."):
                            all_chunks = chunker.chunk_multiple_pdfs(pdf_paths)
                        
                        progress_bar.progress(33)
                        
                        # Save chunks to temp file
                        chunks_file = os.path.join(temp_dir, "chunks.json")
                        chunker.save_chunks_to_json(all_chunks, chunks_file)
                        st.session_state.chunks_file = chunks_file
                        
                        # Display chunk stats
                        st.success(f"✅ Extracted {len(all_chunks['text_chunks'])} text chunks, "
                                 f"{len(all_chunks['image_chunks'])} images, "
                                 f"{len(all_chunks['table_chunks'])} tables")
                        
                        # Step 2: Embedding
                        status_text.text("🔢 Step 2/3: Generating embeddings...")
                        progress_bar.progress(40)
                        
                        processor = ChunkEmbeddingProcessor(
                            embedding_model_name=embedding_model,
                            caption_model_name=caption_model
                        )
                        
                        with st.spinner("Generating embeddings..."):
                            processed_chunks = processor.process_all_chunks(
                                chunks_file,
                                output_file=os.path.join(temp_dir, "embeddings.json")
                            )
                        
                        progress_bar.progress(66)
                        
                        embeddings_file = os.path.join(temp_dir, "embeddings.json")
                        st.session_state.embeddings_file = embeddings_file
                        
                        st.success("✅ Embeddings generated successfully")
                        
                        # Step 3: Insert into ChromaDB
                        status_text.text("💾 Step 3/3: Storing in vector database...")
                        progress_bar.progress(70)
                        
                        db = ChromaDBManager(
                            persist_directory=persist_dir,
                            collection_name=collection_name,
                            embedding_model_name=embedding_model
                        )
                        
                        with st.spinner("Inserting into database..."):
                            total_inserted = db.insert_chunks_from_json(embeddings_file)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        st.success(f"🎉 Successfully processed {len(uploaded_files)} PDFs and "
                                 f"inserted {total_inserted} chunks into the database!")
                        
                        st.session_state.processing_complete = True
                        st.session_state.db_initialized = True
                        
                        # Display summary
                        with st.expander("📊 Processing Summary", expanded=True):
                            stats = db.get_stats()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Chunks", stats['total_chunks'])
                            with col2:
                                st.metric("Unique PDFs", stats['unique_pdfs'])
                            with col3:
                                st.metric("Chunk Types", len(stats['chunk_types']))
                            
                            st.subheader("Chunks per PDF")
                            for pdf_name, count in sorted(stats['pdf_counts'].items()):
                                st.write(f"• {pdf_name}: {count} chunks")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)
                        progress_bar.progress(0)
                        status_text.text("Processing failed")

# Tab 2: Query System
with tab2:
    st.header("2️⃣ Ask Questions")
    
    if not st.session_state.db_initialized:
        st.warning("⚠️ Please upload and process PDFs first (Tab 1)")
    else:
        # Initialize database
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model
            )
            
            # PDF filter options
            st.subheader("🔍 Search Options")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                use_filter = st.checkbox("Filter by specific PDFs")
            
            pdf_filter = None
            if use_filter:
                available_pdfs = db.pdf_sources
                pdf_names = [Path(pdf).name for pdf in available_pdfs]
                selected_pdfs = st.multiselect(
                    "Select PDFs to search",
                    options=pdf_names,
                    help="Leave empty to search all PDFs"
                )
                if selected_pdfs:
                    pdf_filter = selected_pdfs
            
            st.divider()
            
            # Query interface
            st.subheader("💬 Ask Your Question")
            
            # Chat history display
            if st.session_state.chat_history:
                st.subheader("📜 Chat History")
                for i, chat in enumerate(st.session_state.chat_history):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {chat['query']}")
                        st.markdown(f"**A{i+1}:** {chat['answer'][:500]}...")
                        if st.button(f"View Full Response #{i+1}", key=f"view_{i}"):
                            with st.expander("Full Response", expanded=True):
                                st.markdown(chat['answer'])
                                if chat.get('pdfs_used'):
                                    st.write(f"**Sources:** {', '.join(chat['pdfs_used'])}")
                        st.divider()
            
            # Query input
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the main findings across all documents?"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🔎 Search", type="primary", use_container_width=True):
                    if user_query.strip():
                        with st.spinner("Searching and generating answer..."):
                            result = db.intelligent_query(
                                user_query,
                                n_results=n_results,
                                min_score=min_score,
                                pdf_filter=pdf_filter
                            )
                        
                        # Display result
                        st.divider()
                        st.subheader("💡 Answer")
                        
                        if result['success']:
                            # Answer
                            st.markdown(result['answer'])
                            
                            # Metadata
                            with st.expander("📊 Response Metadata", expanded=False):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("PDFs Used", len(result.get('pdfs_used', [])))
                                with col2:
                                    st.metric("Text Sources", result['content_summary'].get('text', 0))
                                with col3:
                                    st.metric("Images", result['content_summary'].get('images', 0))
                                with col4:
                                    st.metric("Tables", result['content_summary'].get('tables', 0))
                                
                                if result.get('pdfs_used'):
                                    st.write("**Source PDFs:**")
                                    for pdf in result['pdfs_used']:
                                        st.write(f"• {pdf}")
                            
                            # Images used
                            if result.get('images'):
                                with st.expander(f"🖼️ Visual Evidence ({len(result['images'])} images)"):
                                    for img_info in result['images']:
                                        st.write(f"**{img_info['pdf_name']} - Page {img_info['page']}**")
                                        st.write(f"*{img_info['description'][:100]}...*")
                                        st.write(f"Score: {img_info['score']:.3f}")
                                        st.divider()
                            
                            # Save to chat history
                            st.session_state.chat_history.append({
                                'query': user_query,
                                'answer': result['answer'],
                                'pdfs_used': result.get('pdfs_used', []),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                        else:
                            st.error(result['answer'])
                    else:
                        st.warning("Please enter a question")
            
            with col3:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error initializing database: {str(e)}")
            st.exception(e)

# Tab 3: Analytics
with tab3:
    st.header("3️⃣ System Analytics")
    
    if not st.session_state.db_initialized:
        st.warning("⚠️ Please upload and process PDFs first (Tab 1)")
    else:
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model
            )
            
            # Get statistics
            stats = db.get_stats()
            
            # Overall metrics
            st.subheader("📊 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
            with col2:
                st.metric("Unique PDFs", stats['unique_pdfs'])
            with col3:
                st.metric("Text Chunks", stats['chunk_types'].get('text', 0))
            with col4:
                st.metric("Image Chunks", stats['chunk_types'].get('image', 0))
            
            st.divider()
            
            # Chunk type breakdown
            st.subheader("📝 Chunk Type Breakdown")
            if stats['chunk_types']:
                import pandas as pd
                df_types = pd.DataFrame(
                    list(stats['chunk_types'].items()),
                    columns=['Type', 'Count']
                )
                st.bar_chart(df_types.set_index('Type'))
            
            st.divider()
            
            # Per-PDF statistics
            st.subheader("📄 Chunks per PDF")
            if stats['pdf_counts']:
                df_pdfs = pd.DataFrame(
                    list(stats['pdf_counts'].items()),
                    columns=['PDF', 'Chunks']
                ).sort_values('Chunks', ascending=False)
                
                st.dataframe(df_pdfs, use_container_width=True)
                st.bar_chart(df_pdfs.set_index('PDF'))
            
            st.divider()
            
            # List all PDFs
            st.subheader("📚 PDF Files in Database")
            for i, pdf_source in enumerate(db.pdf_sources, 1):
                pdf_name = Path(pdf_source).name
                chunk_count = stats['pdf_counts'].get(pdf_name, 0)
                st.write(f"{i}. **{pdf_name}** - {chunk_count} chunks")
            
            st.divider()
            
            # Database info
            st.subheader("💾 Database Information")
            st.write(f"**Location:** {persist_dir}")
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding Model:** {embedding_model}")
            
            # Export option
            if st.button("📥 Export Chat History"):
                if st.session_state.chat_history:
                    export_data = json.dumps(st.session_state.chat_history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No chat history to export")
                    
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Multi-PDF RAG System | Powered by ChromaDB & Custom Models</p>
</div>
""", unsafe_allow_html=True)
