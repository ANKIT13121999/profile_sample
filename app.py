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
    page_title="PDF RAG System",
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
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
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
    collection_name = st.text_input("Collection Name", value="pdf_chunks")
    
    # Query settings
    st.subheader("Query Settings")
    n_results = st.slider("Results to Retrieve", 3, 20, 8)
    min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.0, 0.05)
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.processing_complete:
        st.success("✅ PDF Processed")
    if st.session_state.db_initialized:
        st.success("✅ Database Ready")
    
    if st.session_state.pdf_path:
        st.info(f"📄 Current PDF: {Path(st.session_state.pdf_path).name}")

# Main content
st.title("📚 PDF RAG System")
st.markdown("Upload a PDF, process it, and ask questions about its content!")

# Create tabs for different stages
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Query System", "📊 Analytics"])

# Tab 1: Upload & Process
with tab1:
    st.header("1️⃣ Upload PDF File")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload one PDF file to process"
    )
    
    if uploaded_file:
        st.success(f"✅ PDF uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        st.divider()
        
        # Processing button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    # Save uploaded file
                    pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(pdf_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.pdf_path = pdf_path
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Chunking
                        status_text.text("📄 Step 1/3: Extracting chunks from PDF...")
                        progress_bar.progress(10)
                        
                        chunker = PDFChunker(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        with st.spinner("Processing PDF..."):
                            all_chunks = chunker.chunk_pdf(pdf_path)
                        
                        progress_bar.progress(33)
                        
                        # Save chunks to temp file
                        chunks_file = os.path.join(temp_dir, "chunks.json")
                        chunker.save_chunks_to_json(all_chunks, chunks_file)
                        st.session_state.chunks_file = chunks_file
                        
                        # Display chunk stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Chunks", len(all_chunks['text_chunks']))
                        with col2:
                            st.metric("Images", len(all_chunks['image_chunks']))
                        with col3:
                            st.metric("Tables", len(all_chunks['table_chunks']))
                        
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
                        
                        with st.spinner("Generating embeddings (this may take a while)..."):
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
                            embedding_model_name=embedding_model,
                            pdf_source_path=pdf_path
                        )
                        
                        with st.spinner("Inserting into database..."):
                            total_inserted = db.insert_chunks_from_json(embeddings_file)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")
                        
                        st.success(f"🎉 Successfully processed PDF and inserted {total_inserted} chunks!")
                        
                        st.session_state.processing_complete = True
                        st.session_state.db_initialized = True
                        
                        # Display processing summary
                        with st.expander("📊 Processing Summary", expanded=True):
                            stats = db.get_stats()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Chunks", stats['total_chunks'])
                            with col2:
                                st.metric("Pages with Content", stats['pages_with_content'])
                            with col3:
                                st.metric("Chunk Types", len(stats['chunk_types']))
                            
                            st.subheader("Chunk Type Distribution")
                            for chunk_type, count in sorted(stats['chunk_types'].items()):
                                st.write(f"• {chunk_type.title()}: {count} chunks")
                        
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
        st.warning("⚠️ Please upload and process a PDF first (Tab 1)")
    else:
        # Initialize database
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model,
                pdf_source_path=st.session_state.pdf_path
            )
            
            st.info("💡 Ask any question - the system will automatically find relevant text, images, and tables!")
            
            st.divider()
            
            # Chat history display
            if st.session_state.chat_history:
                st.subheader("📜 Chat History")
                for i, chat in enumerate(st.session_state.chat_history):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {chat['query']}")
                        with st.expander(f"View Answer #{i+1}", expanded=(i == len(st.session_state.chat_history) - 1)):
                            st.markdown(chat['answer'])
                            
                            # Show content summary
                            content = chat.get('content_summary', {})
                            if any(content.values()):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Text Sources", content.get('text', 0))
                                with col2:
                                    st.metric("Images", content.get('images', 0))
                                with col3:
                                    st.metric("Tables", content.get('tables', 0))
                        st.divider()
            
            # Query input
            st.subheader("💬 Ask Your Question")
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the main findings in this document?"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🔎 Search", type="primary", use_container_width=True):
                    if user_query.strip():
                        with st.spinner("Searching and generating answer..."):
                            result = db.intelligent_query(
                                user_query,
                                n_results=n_results,
                                min_score=min_score
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
                                    st.metric("Total Sources", result.get('n_docs_used', 0))
                                with col2:
                                    st.metric("Text Sources", result['content_summary'].get('text', 0))
                                with col3:
                                    st.metric("Images", result['content_summary'].get('images', 0))
                                with col4:
                                    st.metric("Tables", result['content_summary'].get('tables', 0))
                                
                                # Show source quality
                                if result.get('retrieved_docs'):
                                    scores = [doc['score'] for doc in result['retrieved_docs']]
                                    avg_score = sum(scores) / len(scores)
                                    max_score = max(scores)
                                    st.write("**Source Quality:**")
                                    st.write(f"• Average Relevance: {avg_score:.3f}")
                                    st.write(f"• Best Match Score: {max_score:.3f}")
                            
                            # Images used
                            if result.get('images'):
                                with st.expander(f"🖼️ Visual Evidence ({len(result['images'])} images)"):
                                    for img_info in result['images']:
                                        st.write(f"**Page {img_info['page']}**")
                                        st.write(f"*{img_info['description'][:150]}...*")
                                        st.write(f"Relevance Score: {img_info['score']:.3f}")
                                        if img_info.get('caption'):
                                            st.write(f"Caption: {img_info['caption'][:100]}...")
                                        st.divider()
                            
                            # PDF links
                            if result.get('pdf_links'):
                                with st.expander("📑 Source Pages"):
                                    pages = sorted(set([link['page'] for link in result['pdf_links']]))
                                    st.write(f"Referenced pages: {', '.join(map(str, pages))}")
                                    
                                    # Group by type
                                    links_by_type = {}
                                    for link in result['pdf_links']:
                                        chunk_type = link['chunk_type']
                                        if chunk_type not in links_by_type:
                                            links_by_type[chunk_type] = []
                                        links_by_type[chunk_type].append(link['page'])
                                    
                                    for chunk_type, pages in links_by_type.items():
                                        unique_pages = sorted(set(pages))
                                        st.write(f"• {chunk_type.title()}: Pages {', '.join(map(str, unique_pages))}")
                            
                            # Save to chat history
                            st.session_state.chat_history.append({
                                'query': user_query,
                                'answer': result['answer'],
                                'content_summary': result['content_summary'],
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
        st.warning("⚠️ Please upload and process a PDF first (Tab 1)")
    else:
        try:
            db = ChromaDBManager(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model_name=embedding_model,
                pdf_source_path=st.session_state.pdf_path
            )
            
            # Get statistics
            stats = db.get_stats()
            
            # Overall metrics
            st.subheader("📊 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
            with col2:
                st.metric("Pages with Content", stats['pages_with_content'])
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
                
                # Show table
                st.dataframe(df_types, use_container_width=True)
            
            st.divider()
            
            # Database info
            st.subheader("💾 Database Information")
            st.write(f"**Location:** {persist_dir}")
            st.write(f"**Collection:** {collection_name}")
            st.write(f"**Embedding Model:** {embedding_model}")
            if st.session_state.pdf_path:
                st.write(f"**Source PDF:** {Path(st.session_state.pdf_path).name}")
            
            st.divider()
            
            # Chat statistics
            if st.session_state.chat_history:
                st.subheader("💬 Chat Statistics")
                st.metric("Total Queries", len(st.session_state.chat_history))
                
                # Export option
                if st.button("📥 Export Chat History"):
                    export_data = json.dumps(st.session_state.chat_history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("No chat history yet. Ask some questions in the Query System tab!")
            
            st.divider()
            
            # Advanced operations
            st.subheader("🔧 Advanced Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Show Full Database Stats", use_container_width=True):
                    with st.expander("Full Statistics", expanded=True):
                        db.print_stats()
                        st.json(stats)
            
            with col2:
                if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
                    if st.checkbox("Confirm deletion"):
                        try:
                            # Delete collection
                            db.client.delete_collection(collection_name)
                            st.success("✅ Database cleared!")
                            st.session_state.db_initialized = False
                            st.session_state.processing_complete = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing database: {e}")
                    
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>PDF RAG System | Universal Search across Text, Images & Tables</p>
    <p style='font-size: 12px;'>Powered by ChromaDB & Custom Models</p>
</div>
""", unsafe_allow_html=True)
