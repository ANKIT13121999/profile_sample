import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime
import base64

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

# Custom CSS for blue buttons
st.markdown("""
<style>
    /* Make primary buttons blue */
    .stButton > button[kind="primary"] {
        background-color: #0066CC;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0052A3;
        color: white;
        border: none;
    }
    
    /* Alternative: Make all buttons blue */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565C0;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_pdf_viewer' not in st.session_state:
    st.session_state.show_pdf_viewer = False
if 'current_pdf_path' not in st.session_state:
    st.session_state.current_pdf_path = None
if 'current_pdf_page' not in st.session_state:
    st.session_state.current_pdf_page = 1

# Helper function to display PDF
def display_pdf(pdf_path, page_number=1):
    """Display PDF in Streamlit with specific page"""
    try:
        if not os.path.exists(pdf_path):
            st.error(f"PDF not found: {pdf_path}")
            return
        
        # Read PDF file
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Create iframe to display PDF at specific page
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}#page={page_number}" 
                width="100%" height="800" type="application/pdf">
        </iframe>
        '''
        
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

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
    
    # Initialize DB button
    if st.button("🔌 Initialize Database", use_container_width=True):
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model,
                streamlit_mode=True
            )
            st.session_state.db_initialized = True
            stats = db.get_stats()
            st.success(f"✅ Database loaded! {stats['total_chunks']} chunks available")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.db_initialized:
        st.success("✅ Database Ready")
    else:
        st.warning("⚠️ Click 'Initialize Database' to start")

# Main content
st.title("📚 Multi-PDF RAG System")
st.markdown("Query existing PDFs or upload new documents to expand your knowledge base!")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query PDFs", "📤 Upload New PDFs", "📊 Analytics", "📄 PDF Viewer"])

# Tab 1: Query PDFs (Main feature first!)
with tab1:
    st.header("💬 Ask Questions About Your Documents")
    
    if not st.session_state.db_initialized:
        st.info("👈 Please click **'Initialize Database'** in the sidebar to start querying!")
        st.markdown("""
        ### 🚀 Quick Start:
        1. Click **'Initialize Database'** in the sidebar
        2. Start asking questions about your documents
        3. Or upload new PDFs in the **'Upload New PDFs'** tab
        """)
    else:
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model,
                streamlit_mode=True
            )
            
            # Show database stats
            stats = db.get_stats()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
            with col2:
                st.metric("PDFs Available", stats['unique_pdfs'])
            with col3:
                st.metric("Text Chunks", stats['chunk_types'].get('text', 0))
            with col4:
                st.metric("Images", stats['chunk_types'].get('image', 0))
            
            st.divider()
            
            # PDF filter options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                use_filter = st.checkbox("🔍 Filter by specific PDFs")
            
            pdf_filter = None
            if use_filter and db.pdf_sources:
                pdf_names = [Path(pdf).name for pdf in db.pdf_sources]
                selected_pdfs = st.multiselect(
                    "Select PDFs to search",
                    options=pdf_names,
                    help="Leave empty to search all PDFs"
                )
                if selected_pdfs:
                    pdf_filter = selected_pdfs
            
            st.divider()
            
            # Query interface
            st.subheader("💭 Your Question")
            
            # Example questions
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - What are the main findings in the documents?
                - Summarize the key points from all PDFs
                - What financial data is available?
                - Show me information about [specific topic]
                - Compare the data across different documents
                """)
            
            # Query input
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the main findings across all documents?",
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_button = st.button("🔎 Search & Answer", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if search_button and user_query.strip():
                with st.spinner("🔍 Searching documents and generating answer..."):
                    result = db.intelligent_query(
                        user_query,
                        n_results=n_results,
                        min_score=min_score,
                        pdf_filter=pdf_filter
                    )
                
                # Display result
                st.divider()
                
                if result['success']:
                    # Answer
                    st.subheader("💡 Answer")
                    st.markdown(result['answer'])
                    
                    st.divider()
                    
                    # Sources with clickable links
                    st.subheader("📚 Sources")
                    
                    if result.get('pdf_links'):
                        for i, link in enumerate(result['pdf_links'][:5], 1):  # Show top 5
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                **{i}. {link['pdf_name']}** - Page {link['page_display']} 
                                ({link['chunk_type']}) | Relevance: {link['score']:.1%}
                                """)
                            
                            with col2:
                                # Create unique key for button
                                button_key = f"view_{link['chunk_id']}_{i}"
                                if st.button("📖 Open PDF", key=button_key, use_container_width=True):
                                    st.session_state.show_pdf_viewer = True
                                    st.session_state.current_pdf_path = link['pdf_path']
                                    st.session_state.current_pdf_page = link['page_display']
                                    st.switch_page("pages/3_📄_PDF_Viewer.py") if False else None
                                    # Switch to PDF Viewer tab
                                    st.info(f"📄 Opening {link['pdf_name']} at page {link['page_display']} in PDF Viewer tab →")
                        
                        if len(result['pdf_links']) > 5:
                            with st.expander(f"➕ Show all {len(result['pdf_links'])} sources"):
                                for i, link in enumerate(result['pdf_links'][5:], 6):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"""
                                        **{i}. {link['pdf_name']}** - Page {link['page_display']} 
                                        ({link['chunk_type']}) | Relevance: {link['score']:.1%}
                                        """)
                                    with col2:
                                        button_key = f"view_{link['chunk_id']}_{i}"
                                        if st.button("📖 Open", key=button_key, use_container_width=True):
                                            st.session_state.show_pdf_viewer = True
                                            st.session_state.current_pdf_path = link['pdf_path']
                                            st.session_state.current_pdf_page = link['page_display']
                                            st.info(f"📄 Go to PDF Viewer tab to see {link['pdf_name']} →")
                    
                    # Images used
                    if result.get('images'):
                        st.divider()
                        with st.expander(f"🖼️ Visual Evidence ({len(result['images'])} images)"):
                            for img_info in result['images']:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**{img_info['pdf_name']} - Page {img_info['page_display']}**")
                                    st.caption(f"{img_info['description'][:150]}...")
                                    st.write(f"Score: {img_info['score']:.1%}")
                                
                                with col2:
                                    button_key = f"img_{img_info['chunk_id']}"
                                    if st.button("📖 View", key=button_key, use_container_width=True):
                                        st.session_state.show_pdf_viewer = True
                                        st.session_state.current_pdf_path = img_info['pdf_path']
                                        st.session_state.current_pdf_page = img_info['page_display']
                                        st.info(f"📄 Go to PDF Viewer tab →")
                                
                                st.divider()
                    
                    # Metadata
                    with st.expander("📊 Response Metadata"):
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
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        'query': user_query,
                        'answer': result['answer'],
                        'pdfs_used': result.get('pdfs_used', []),
                        'pdf_links': result.get('pdf_links', []),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                else:
                    st.error(result['answer'])
            
            elif search_button:
                st.warning("⚠️ Please enter a question")
            
            # Chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("📜 Recent Queries")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['query'][:80]}..."):
                        st.markdown(f"**Question:** {chat['query']}")
                        st.markdown(f"**Answer:** {chat['answer'][:300]}...")
                        if chat.get('pdfs_used'):
                            st.write(f"**Sources:** {', '.join(chat['pdfs_used'])}")
                        st.caption(f"Asked at: {chat['timestamp']}")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Tab 2: Upload New PDFs
with tab2:
    st.header("📤 Upload New PDF Documents")
    
    st.info("💡 Upload PDFs to expand your knowledge base. Previously uploaded PDFs are already available for querying!")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the database"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF file(s) selected")
        
        # Display uploaded files
        with st.expander("📄 Selected Files", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size / 1024:.2f} KB)")
        
        st.divider()
        
        # Processing button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Process & Add to Database", use_container_width=True):
                
                # Create directories for permanent storage
                output_dir = "./pdf_processing_output"
                uploaded_pdfs_dir = "./uploaded_pdfs"
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(uploaded_pdfs_dir):
                    os.makedirs(uploaded_pdfs_dir)
                
                # Generate timestamp for unique filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save uploaded files PERMANENTLY
                pdf_paths = []
                
                status_text.text("📁 Saving uploaded PDFs...")
                for uploaded_file in uploaded_files:
                    # Create permanent file path
                    permanent_path = os.path.join(uploaded_pdfs_dir, uploaded_file.name)
                    
                    # Save the file
                    with open(permanent_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    pdf_paths.append(permanent_path)
                    st.write(f"✅ Saved: {permanent_path}")
                
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
                    
                    # Save chunks to persistent directory
                    chunks_file_saved = os.path.join(output_dir, f"chunks_{timestamp}.json")
                    chunker.save_chunks_to_json(all_chunks, chunks_file_saved)
                    
                    # Display chunk stats
                    st.success(f"✅ Extracted {len(all_chunks['text_chunks'])} text chunks, "
                             f"{len(all_chunks['image_chunks'])} images, "
                             f"{len(all_chunks['table_chunks'])} tables")
                    
                    st.info(f"📁 Chunks saved to: `{chunks_file_saved}`")
                    
                    # Step 2: Embedding
                    status_text.text("🔢 Step 2/3: Generating embeddings...")
                    progress_bar.progress(40)
                    
                    processor = ChunkEmbeddingProcessor(
                        embedding_model_name=embedding_model,
                        caption_model_name=caption_model
                    )
                    
                    embeddings_file_saved = os.path.join(output_dir, f"embeddings_{timestamp}.json")
                    
                    with st.spinner("Generating embeddings..."):
                        processed_chunks = processor.process_all_chunks(
                            chunks_file_saved,
                            output_file=embeddings_file_saved
                        )
                    
                    progress_bar.progress(66)
                    
                    if os.path.exists(embeddings_file_saved):
                        file_size = os.path.getsize(embeddings_file_saved) / (1024 * 1024)
                        st.success(f"✅ Embeddings generated successfully ({file_size:.2f} MB)")
                        st.info(f"📁 Embeddings saved to: `{embeddings_file_saved}`")
                    
                    # Step 3: Insert into ChromaDB
                    status_text.text("💾 Step 3/3: Adding to database...")
                    progress_bar.progress(70)
                    
                    db = ChromaDBManager(
                        persist_directory=persist_dir,
                        collection_name=collection_name,
                        embedding_model_name=embedding_model,
                        streamlit_mode=True
                    )
                    
                    with st.spinner("Inserting into database..."):
                        total_inserted = db.insert_chunks_from_json(embeddings_file_saved)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    st.success(f"🎉 Successfully processed {len(uploaded_files)} PDFs and "
                             f"added {total_inserted} chunks to the database!")
                    
                    st.session_state.db_initialized = True
                    
                    # Display summary with file locations
                    with st.expander("📊 Processing Summary", expanded=True):
                        st.subheader("📁 Saved Files")
                        st.write(f"**Chunks JSON:** `{chunks_file_saved}`")
                        st.write(f"**Embeddings JSON:** `{embeddings_file_saved}`")
                        st.write(f"**ChromaDB:** `{persist_dir}`")
                        
                        # Add download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with open(chunks_file_saved, 'r') as f:
                                chunks_json = f.read()
                            st.download_button(
                                label="📥 Download Chunks JSON",
                                data=chunks_json,
                                file_name=f"chunks_{timestamp}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            with open(embeddings_file_saved, 'r') as f:
                                embeddings_json = f.read()
                            st.download_button(
                                label="📥 Download Embeddings JSON",
                                data=embeddings_json,
                                file_name=f"embeddings_{timestamp}.json",
                                mime="application/json"
                            )
                        
                        st.divider()
                        
                        stats = db.get_stats()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Chunks", stats['total_chunks'])
                        with col2:
                            st.metric("Total PDFs", stats['unique_pdfs'])
                        with col3:
                            st.metric("Chunk Types", len(stats['chunk_types']))
                        
                        st.subheader("Chunks per PDF")
                        for pdf_name, count in sorted(stats['pdf_counts'].items()):
                            st.write(f"• {pdf_name}: {count} chunks")
                    
                    st.balloons()
                    st.info("💡 Go to the 'Query PDFs' tab to ask questions about your documents!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                    progress_bar.progress(0)
                    status_text.text("Processing failed")

# Tab 3: Analytics
with tab3:
    st.header("📊 System Analytics")
    
    if not st.session_state.db_initialized:
        st.info("👈 Initialize database first to see analytics")
    else:
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model,
                streamlit_mode=True
            )
            
            # Get statistics
            stats = db.get_stats()
            
            # Overall metrics
            st.subheader("📈 Overall Statistics")
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
            st.subheader("📝 Chunk Type Distribution")
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
                
                st.dataframe(df_pdfs, use_container_width=True, hide_index=True)
                st.bar_chart(df_pdfs.set_index('PDF'))
            
            st.divider()
            
            # Show uploaded PDFs
            uploaded_pdfs_dir = "./uploaded_pdfs"
            if os.path.exists(uploaded_pdfs_dir):
                st.subheader("📚 Available PDF Files")
                pdf_files = sorted([f for f in os.listdir(uploaded_pdfs_dir) if f.endswith('.pdf')])
                if pdf_files:
                    for pdf_file in pdf_files:
                        pdf_path = os.path.join(uploaded_pdfs_dir, pdf_file)
                        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📄 **{pdf_file}** ({pdf_size:.2f} MB)")
                            with st.expander("Show path"):
                                st.code(pdf_path, language=None)
                        with col2:
                            if st.button("👁️ View", key=f"view_pdf_{pdf_file}"):
                                st.session_state.show_pdf_viewer = True
                                st.session_state.current_pdf_path = pdf_path
                                st.session_state.current_pdf_page = 1
                                st.info("📄 Go to PDF Viewer tab →")
                else:
                    st.info("No PDF files uploaded yet")
            
            st.divider()
            
            # Database info
            st.subheader("💾 Database Information")
            st.write(f"**Location:** `{persist_dir}`")
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding Model:** {embedding_model}")
            
            # Show saved files
            output_dir = "./pdf_processing_output"
            if os.path.exists(output_dir):
                st.divider()
                st.subheader("📂 Processing Files")
                files = sorted([f for f in os.listdir(output_dir) if f.endswith('.json')])
                if files:
                    for file in files[:10]:  # Show last 10
                        file_path = os.path.join(output_dir, file)
                        file_size = os.path.getsize(file_path) / 1024
                        st.write(f"• {file} ({file_size:.2f} KB)")
                    
                    if len(files) > 10:
                        st.caption(f"... and {len(files) - 10} more files")
            
            # Export chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("💾 Export Data")
                export_data = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="📥 Download Chat History",
                    data=export_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                    
        except Exception as e:
            st.error(f"❌ Error loading analytics: {e}")

# Tab 4: PDF Viewer
with tab4:
    st.header("📄 PDF Viewer")
    
    if st.session_state.show_pdf_viewer and st.session_state.current_pdf_path:
        pdf_path = st.session_state.current_pdf_path
        page_num = st.session_state.current_pdf_page
        
        st.subheader(f"📖 {Path(pdf_path).name}")
        st.caption(f"Page {page_num}")
        
        # Page navigation
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First"):
                st.session_state.current_pdf_page = 1
                st.rerun()
        
        with col2:
            if st.button("◀️ Prev"):
                if st.session_state.current_pdf_page > 1:
                    st.session_state.current_pdf_page -= 1
                    st.rerun()
        
        with col3:
            new_page = st.number_input("Go to page:", min_value=1, value=page_num, step=1, key="page_input")
            if new_page != page_num:
                st.session_state.current_pdf_page = new_page
                st.rerun()
        
        with col4:
            if st.button("▶️ Next"):
                st.session_state.current_pdf_page += 1
                st.rerun()
        
        with col5:
            if st.button("⏭️ Last"):
                import fitz
                doc = fitz.open(pdf_path)
                st.session_state.current_pdf_page = len(doc)
                doc.close()
                st.rerun()
        
        st.divider()
        
        # Display PDF
        display_pdf(pdf_path, page_num)
        
        # PDF info
        with st.expander("ℹ️ PDF Information"):
            st.write(f"**File:** {Path(pdf_path).name}")
            st.write(f"**Path:** `{pdf_path}`")
            st.write(f"**Size:** {os.path.getsize(pdf_path) / (1024 * 1024):.2f} MB")
    
    else:
        st.info("👈 Click on '📖 Open PDF' buttons in the Query tab to view PDFs here!")
        
        st.markdown("""
        ### How to use PDF Viewer:
        1. Go to **Query PDFs** tab
        2. Ask a question
        3. Click **📖 Open PDF** button next to any source
        4. The PDF will open here at the exact page!
        
        You can also view PDFs from the **Analytics** tab.
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Multi-PDF RAG System | Powered by ChromaDB & Custom Models</p>
    <p>💡 Tip: Initialize database → Query existing PDFs → Upload new documents to expand</p>
</div>
""", unsafe_allow_html=True)
