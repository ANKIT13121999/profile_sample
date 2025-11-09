import json
import chromadb
from chromadb.config import Settings
from inter.core.clients.xorclient import xorclient
from xyz import chat
from xyz.types import HumanMultimodalMessage, Image as AFMImage
from langchain.schema import Human, System
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
from pathlib import Path
import base64

class ChromaDBManager:
    def __init__(self, 
                 persist_directory: str = "./chroma_db", 
                 collection_name: str = "multi_pdf_chunks",
                 embedding_model_name: str = "bembedd-1rg",
                 streamlit_mode: bool = False):
        """
        Initialize ChromaDB for multiple PDF chunks
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model_name: Your organization's embedding model name
            streamlit_mode: If True, creates special links for Streamlit
        """
        print("Initializing ChromaDB for multi-PDF RAG...")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("Loading your organization's embedding model...")
        self.embedding_client = xorclient()
        self.embedding_model_name = embedding_model_name
        self.streamlit_mode = streamlit_mode
        
        print(f"✓ ChromaDB initialized with collection: {collection_name}")
        print(f"✓ Database path: {persist_directory}")
        print(f"✓ Using embedding model: {embedding_model_name}")
        
        existing_count = self.collection.count()
        print(f"✓ Existing chunks in database: {existing_count}")
        
        # Track PDF sources in database
        self.pdf_sources = self._get_unique_pdf_sources()
        if self.pdf_sources:
            print(f"✓ PDFs in database: {len(self.pdf_sources)}")
    
    def _get_unique_pdf_sources(self) -> List[str]:
        """Get list of unique PDF sources in the database"""
        try:
            results = self.collection.get(limit=1000)
            if results["metadatas"]:
                sources = set()
                for metadata in results["metadatas"]:
                    pdf_source = metadata.get("pdf_source", "unknown")
                    if pdf_source != "unknown":
                        sources.add(pdf_source)
                return sorted(list(sources))
        except Exception as e:
            print(f"Warning: Could not retrieve PDF sources: {e}")
        return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            response = self.embedding_client.get_embedding(
                input=query, 
                model_name=self.embedding_model_name
            )
            
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            elif isinstance(response, dict) and 'data' in response:
                return response['data'][0]['embedding']
            else:
                print(f"Unexpected response format: {type(response)}")
                return response
                
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
    
    def create_pdf_link(self, page_number: int, pdf_path: str) -> str:
        """
        Create a clickable link to specific page in PDF
        Works with Streamlit by encoding PDF for browser viewing
        
        Args:
            page_number: Page number in PDF (0-indexed internally, displayed as 1-indexed)
            pdf_path: Path to PDF file
            
        Returns:
            HTML link or file URL
        """
        if not pdf_path or page_number < 0:
            return "PDF link not available"
        
        # Make sure path is absolute
        abs_path = os.path.abspath(pdf_path)
        
        if self.streamlit_mode:
            # For Streamlit: Create a viewable link
            # We'll encode the path and page for the frontend
            pdf_name = Path(pdf_path).name
            display_page = page_number + 1  # Convert to 1-indexed for display
            
            # Return a formatted string that Streamlit can display
            return f"📄 {pdf_name} - Page {display_page} | Path: {abs_path}"
        else:
            # For regular Python: file:// URL with page anchor
            display_page = page_number + 1
            return f"file://{abs_path}#page={display_page}"
    
    def get_pdf_viewer_data(self, pdf_path: str, page_number: int) -> Optional[Dict]:
        """
        Get data needed to display PDF in Streamlit
        
        Returns:
            Dict with pdf_data (base64), page_number, pdf_name
        """
        if not os.path.exists(pdf_path):
            return None
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            return {
                'pdf_data': pdf_base64,
                'page_number': page_number + 1,  # 1-indexed for display
                'pdf_name': Path(pdf_path).name,
                'pdf_path': pdf_path
            }
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    
    def insert_chunks_from_json(self, json_file_path: str) -> int:
        """
        Insert all chunks from the embeddings JSON file
        
        Args:
            json_file_path: Path to the chunks_with_embeddings.json file
            
        Returns:
            Number of chunks inserted
        """
        print(f"\n📂 Loading chunks from: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Display per-PDF statistics if available
        if "per_pdf_stats" in chunks_data:
            print(f"\n📄 Found {len(chunks_data['per_pdf_stats'])} PDF files:")
            for pdf_path, stats in chunks_data["per_pdf_stats"].items():
                print(f"  • {stats['pdf_name']}: {stats['total_chunks']} chunks")
        
        total_inserted = 0
        
        # Process each chunk type
        for chunk_type, chunk_list in chunks_data.items():
            if chunk_type not in ["text_chunks", "image_chunks", "table_chunks"]:
                continue
            
            if not chunk_list:
                continue
                
            print(f"\n📝 Processing {len(chunk_list)} {chunk_type}...")
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunk_list:
                try:
                    chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
                    ids.append(chunk_id)
                    
                    embedding = chunk.get("embedding")
                    if not embedding:
                        print(f"⚠️ No embedding found for chunk: {chunk_id}")
                        continue
                    embeddings.append(embedding)
                    
                    # Prepare document content
                    if chunk_type == "text_chunks":
                        content = chunk.get("content", "")
                    elif chunk_type == "image_chunks":
                        content = chunk.get("combined_description", 
                                          chunk.get("image_description", ""))
                        caption = chunk.get("generated_caption", "")
                        if caption:
                            content = f"{content}. Caption: {caption}"
                    elif chunk_type == "table_chunks":
                        content = chunk.get("text_representation", 
                                          chunk.get("table_description", ""))
                    else:
                        content = str(chunk)
                    
                    documents.append(content)
                    
                    # Prepare metadata with PDF source tracking
                    pdf_source = chunk.get("pdf_source", "unknown")
                    pdf_name = Path(pdf_source).name if pdf_source != "unknown" else "unknown"
                    
                    metadata = {
                        "chunk_type": chunk_type.replace("_chunks", ""),
                        "page_number": chunk.get("page_number", -1),
                        "timestamp": datetime.now().isoformat(),
                        "pdf_source": pdf_source,
                        "pdf_name": pdf_name
                    }
                    
                    # Add type-specific metadata
                    if chunk_type == "image_chunks":
                        metadata.update({
                            "generated_caption": chunk.get("generated_caption", ""),
                            "original_description": chunk.get("image_description", ""),
                            "image_width": chunk.get("metadata", {}).get("width", 0),
                            "image_height": chunk.get("metadata", {}).get("height", 0),
                            "image_path": chunk.get("image_path", ""),
                            "image_source": chunk.get("image_source", "")
                        })
                    elif chunk_type == "table_chunks":
                        metadata.update({
                            "table_description": chunk.get("table_description", ""),
                            "num_rows": chunk.get("metadata", {}).get("num_rows", 0),
                            "num_cols": chunk.get("metadata", {}).get("num_cols", 0)
                        })
                    elif chunk_type == "text_chunks":
                        metadata.update({
                            "char_start": chunk.get("metadata", {}).get("char_start", 0),
                            "char_end": chunk.get("metadata", {}).get("char_end", 0)
                        })
                    
                    metadatas.append(metadata)
                    
                except Exception as e:
                    print(f"❌ Error processing chunk: {e}")
                    continue
            
            # Insert batch into ChromaDB
            if ids and embeddings:
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    print(f"✅ Inserted {len(ids)} {chunk_type}")
                    total_inserted += len(ids)
                except Exception as e:
                    print(f"❌ Error inserting {chunk_type}: {e}")
        
        print(f"\n🎉 Total chunks inserted: {total_inserted}")
        print(f"📊 Total chunks in database: {self.collection.count()}")
        
        # Update tracked PDF sources
        self.pdf_sources = self._get_unique_pdf_sources()
        
        return total_inserted
    
    def universal_search(self, 
                        query: str, 
                        n_results: int = 10,
                        pdf_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Universal search across ALL chunk types with optional PDF filtering
        
        Args:
            query: Search query text
            n_results: Number of results to return
            pdf_filter: Optional list of PDF filenames to search within
            
        Returns:
            List of search results from all chunk types
        """
        print(f"\n🔍 Universal search for: '{query}'")
        
        if pdf_filter:
            print(f"🔧 Filtering to PDFs: {pdf_filter}")
        else:
            print("🔧 Searching across all PDFs")
        
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        try:
            # Build where clause for PDF filtering
            where_clause = None
            if pdf_filter:
                if len(pdf_filter) == 1:
                    where_clause = {"pdf_name": pdf_filter[0]}
                else:
                    where_clause = {"pdf_name": {"$in": pdf_filter}}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    similarity_score = 1 - results["distances"][0][i]
                    
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity_score,
                        "distance": results["distances"][0][i]
                    }
                    formatted_results.append(result)
            
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            print(f"✅ Found {len(formatted_results)} results")
            
            # Show breakdown by PDF and type
            pdf_counts = {}
            type_counts = {}
            for result in formatted_results:
                pdf_name = result["metadata"].get("pdf_name", "unknown")
                chunk_type = result["metadata"].get("chunk_type", "unknown")
                
                pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            print(f"📊 Results by PDF: {dict(pdf_counts)}")
            print(f"📊 Results by type: {dict(type_counts)}")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search_specific_pdf(self, query: str, pdf_name: str, n_results: int = 10) -> List[Dict]:
        """Search within a specific PDF only"""
        return self.universal_search(query, n_results, pdf_filter=[pdf_name])
    
    def create_intelligent_multimodal_prompt(self, user_query: str, retrieved_docs: List[Dict]) -> List:
        """Create an intelligent multimodal RAG prompt across multiple PDFs"""
        text_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") != "image"]
        image_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") == "image"]
        
        print(f"📝 Processing {len(text_docs)} text/table chunks and {len(image_docs)} image chunks")
        
        context_parts = []
        image_contents = []
        
        # Track which PDFs are being referenced
        pdfs_referenced = set()
        
        # Process text and table chunks
        for i, doc in enumerate(text_docs, 1):
            chunk_type = doc["metadata"].get("chunk_type", "unknown")
            page_num = doc["metadata"].get("page_number", "N/A")
            pdf_name = doc["metadata"].get("pdf_name", "unknown")
            pdf_source = doc["metadata"].get("pdf_source", "unknown")
            score = doc.get("score", 0)
            content = doc["content"]
            
            pdfs_referenced.add(pdf_name)
            
            doc_context = f"""Document {i} ({chunk_type.title()}, PDF: {pdf_name}, Page: {page_num + 1}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num, pdf_source)}
"""
            context_parts.append(doc_context)
        
        # Process image chunks
        for i, doc in enumerate(image_docs, 1):
            page_num = doc["metadata"].get("page_number", "N/A")
            pdf_name = doc["metadata"].get("pdf_name", "unknown")
            pdf_source = doc["metadata"].get("pdf_source", "unknown")
            score = doc.get("score", 0)
            content = doc["content"]
            image_path = doc["metadata"].get("image_path", "")
            
            pdfs_referenced.add(pdf_name)
            
            if image_path and os.path.exists(image_path):
                try:
                    afm_image = AFMImage.from_url(image_path)
                    image_contents.append(afm_image)
                    
                    image_context = f"""Image {len(image_contents)} (PDF: {pdf_name}, Page: {page_num + 1}, Relevance: {score:.3f}):
Description: {content}
Source: {self.create_pdf_link(page_num, pdf_source)}
"""
                    context_parts.append(image_context)
                except Exception as e:
                    print(f"⚠️ Error loading image {image_path}: {e}")
                    image_context = f"""Image Description {i} (PDF: {pdf_name}, Page: {page_num + 1}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num, pdf_source)}
"""
                    context_parts.append(image_context)
            else:
                image_context = f"""Image Description {i} (PDF: {pdf_name}, Page: {page_num + 1}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num, pdf_source)}
"""
                context_parts.append(image_context)
        
        full_context = "\n" + "-" * 50 + "\n".join(context_parts) + "-" * 50
        
        has_images = len(image_contents) > 0
        has_tables = any(doc["metadata"].get("chunk_type") == "table" for doc in retrieved_docs)
        
        system_content = f"""You are a helpful assistant that answers questions based on documents from multiple PDF files.

Content Available from {len(pdfs_referenced)} PDF(s): {', '.join(sorted(pdfs_referenced))}
- Text documents: {len(text_docs)}
- Images: {len(image_docs)} ({len(image_contents)} loaded)
- Tables: {sum(1 for doc in retrieved_docs if doc["metadata"].get("chunk_type") == "table")}

Instructions:
- Use ALL the information provided from across the different PDFs
- ALWAYS mention which PDF each piece of information comes from
- When referencing content, cite: PDF name, page number, and document/image number
{"- For images, describe what you see and relate it to the question" if has_images else ""}
{"- For tables, interpret the data and provide insights" if has_tables else ""}
- Include source links when relevant
- If information conflicts between PDFs, note the discrepancy and cite sources
- Be comprehensive but concise
- Prioritize higher relevance scores when conflicts arise

Document and Image Context:""" + full_context
        
        message_contents = [system_content]
        
        for image in image_contents:
            message_contents.append(image)
        
        message_contents.append(user_query)
        
        messages = [HumanMultimodalMessage(contents=message_contents)]
        
        return messages
    
    def intelligent_query(self, 
                         user_query: str, 
                         n_results: int = 8, 
                         min_score: float = 0.0,
                         pdf_filter: Optional[List[str]] = None) -> Dict:
        """
        Intelligent universal query across multiple PDFs
        
        Args:
            user_query: The user's question
            n_results: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            pdf_filter: Optional list of PDF names to search within
            
        Returns:
            Dictionary with answer, content info, PDF links, and metadata
        """
        print(f"\n🤖 Processing intelligent query across multiple PDFs: '{user_query}'")
        
        retrieved_docs = self.universal_search(user_query, n_results=n_results, pdf_filter=pdf_filter)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "retrieved_docs": [],
                "content_summary": {"text": 0, "images": 0, "tables": 0},
                "pdfs_used": [],
                "pdf_links": [],
                "pdf_viewers": [],
                "query": user_query,
                "success": False
            }
        
        if min_score > 0:
            retrieved_docs = [doc for doc in retrieved_docs if doc.get("score", 0) >= min_score]
            if not retrieved_docs:
                return {
                    "answer": f"No documents found with similarity score above {min_score}.",
                    "retrieved_docs": [],
                    "content_summary": {"text": 0, "images": 0, "tables": 0},
                    "pdfs_used": [],
                    "pdf_links": [],
                    "pdf_viewers": [],
                    "query": user_query,
                    "success": False
                }
        
        print(f"📚 Using {len(retrieved_docs)} most relevant documents")
        
        # Analyze content
        content_summary = {"text": 0, "images": 0, "tables": 0, "other": 0}
        images_info = []
        pdf_links = []
        pdf_viewers = []  # For Streamlit PDF viewing
        pdfs_used = set()
        
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            chunk_type = metadata.get("chunk_type", "other")
            page_num = metadata.get("page_number", -1)
            pdf_name = metadata.get("pdf_name", "unknown")
            pdf_source = metadata.get("pdf_source", "unknown")
            
            pdfs_used.add(pdf_name)
            
            if chunk_type == "text":
                content_summary["text"] += 1
            elif chunk_type == "image":
                content_summary["images"] += 1
            elif chunk_type == "table":
                content_summary["tables"] += 1
            else:
                content_summary["other"] += 1
            
            if page_num >= 0:
                pdf_link = {
                    "page": page_num,
                    "page_display": page_num + 1,  # 1-indexed for display
                    "url": self.create_pdf_link(page_num, pdf_source),
                    "chunk_id": doc["id"],
                    "chunk_type": chunk_type,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_source,
                    "score": doc["score"]
                }
                pdf_links.append(pdf_link)
                
                # Add PDF viewer data for Streamlit
                if self.streamlit_mode:
                    viewer_data = self.get_pdf_viewer_data(pdf_source, page_num)
                    if viewer_data and viewer_data not in pdf_viewers:
                        pdf_viewers.append(viewer_data)
            
            if chunk_type == "image":
                image_info = {
                    "chunk_id": doc["id"],
                    "image_path": metadata.get("image_path", ""),
                    "description": doc["content"],
                    "page": page_num,
                    "page_display": page_num + 1,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_source,
                    "pdf_link": self.create_pdf_link(page_num, pdf_source),
                    "caption": metadata.get("generated_caption", ""),
                    "dimensions": f"{metadata.get('image_width', 0)}x{metadata.get('image_height', 0)}",
                    "score": doc["score"]
                }
                images_info.append(image_info)
        
        print(f"📊 Content found: {content_summary}")
        print(f"📄 PDFs used: {sorted(list(pdfs_used))}")
        
        # Generate answer
        try:
            print("🧠 Generating comprehensive answer...")
            
            messages = self.create_intelligent_multimodal_prompt(user_query, retrieved_docs)
            response = chat(messages)
            
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            print("✅ Answer generated successfully")
            
            return {
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "content_summary": content_summary,
                "images": images_info,
                "pdf_links": pdf_links,
                "pdf_viewers": pdf_viewers,  # For Streamlit viewing
                "pdfs_used": sorted(list(pdfs_used)),
                "query": user_query,
                "n_docs_used": len(retrieved_docs),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error generating answer: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "content_summary": content_summary,
                "images": images_info,
                "pdf_links": pdf_links,
                "pdf_viewers": pdf_viewers,
                "pdfs_used": sorted(list(pdfs_used)),
                "query": user_query,
                "success": False
            }
    
    def get_stats(self) -> Dict:
        """Get database statistics including per-PDF breakdown"""
        total_count = self.collection.count()
        
        sample_results = self.collection.get(limit=min(1000, total_count))
        
        type_counts = {}
        pdf_counts = {}
        
        if sample_results["metadatas"]:
            for metadata in sample_results["metadatas"]:
                chunk_type = metadata.get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                
                pdf_name = metadata.get("pdf_name", "unknown")
                pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
        
        stats = {
            "total_chunks": total_count,
            "chunk_types": type_counts,
            "pdf_counts": pdf_counts,
            "unique_pdfs": len(pdf_counts),
            "sample_size": len(sample_results["metadatas"]) if sample_results["metadatas"] else 0
        }
        
        return stats
