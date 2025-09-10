import json
import chromadb
from chromadb.config import Settings
from inter.core.clients.xorclient import xorclient
from xyz import chat  # Your custom model interface
from xyz.types import HumanMultimodalMessage, Image as AFMImage  # Your custom types
from langchain.schema import Human, System
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os

class ChromaDBManager:
    def __init__(self, 
                 persist_directory: str = "./chroma_db", 
                 collection_name: str = "pdf_chunks",
                 embedding_model_name: str = "bembedd-1rg",
                 pdf_source_path: str = None):
        """
        Initialize ChromaDB for PDF chunks with your organization's embedding model
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model_name: Your organization's embedding model name
            pdf_source_path: Path to the original PDF file for linking
        """
        print("Initializing ChromaDB...")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize your organization's embedding client
        print("Loading your organization's embedding model...")
        self.embedding_client = xorclient()
        self.embedding_model_name = embedding_model_name
        
        # Store PDF source path for linking
        self.pdf_source_path = pdf_source_path
        
        print(f"✓ ChromaDB initialized with collection: {collection_name}")
        print(f"✓ Database path: {persist_directory}")
        print(f"✓ Using embedding model: {embedding_model_name}")
        if pdf_source_path:
            print(f"✓ PDF source: {pdf_source_path}")
        
        # Check existing data
        existing_count = self.collection.count()
        print(f"✓ Existing chunks in database: {existing_count}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query using your organization's model"""
        try:
            response = self.embedding_client.get_embedding(
                input=query, 
                model_name=self.embedding_model_name
            )
            
            # Extract embedding from response (handle different response formats)
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            elif isinstance(response, dict) and 'data' in response:
                return response['data'][0]['embedding']  # Common API format
            else:
                # Handle other response formats
                print(f"Unexpected response format: {type(response)}")
                return response
                
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
    
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
        
        total_inserted = 0
        
        # Process each chunk type
        for chunk_type, chunk_list in chunks_data.items():
            if not chunk_list:
                continue
                
            print(f"\n📝 Processing {len(chunk_list)} {chunk_type}...")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunk_list:
                try:
                    # Get chunk ID
                    chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
                    ids.append(chunk_id)
                    
                    # Get embedding
                    embedding = chunk.get("embedding")
                    if not embedding:
                        print(f"⚠️ No embedding found for chunk: {chunk_id}")
                        continue
                    embeddings.append(embedding)
                    
                    # Prepare document content based on chunk type
                    if chunk_type == "text_chunks":
                        content = chunk.get("content", "")
                    elif chunk_type == "image_chunks":
                        content = chunk.get("combined_description", 
                                          chunk.get("image_description", ""))
                        # Add generated caption if available
                        caption = chunk.get("generated_caption", "")
                        if caption:
                            content = f"{content}. Caption: {caption}"
                    elif chunk_type == "table_chunks":
                        content = chunk.get("text_representation", 
                                          chunk.get("table_description", ""))
                    else:
                        content = str(chunk)
                    
                    documents.append(content)
                    
                    # Prepare metadata with source linking information
                    metadata = {
                        "chunk_type": chunk_type.replace("_chunks", ""),
                        "page_number": chunk.get("page_number", -1),
                        "timestamp": datetime.now().isoformat(),
                        "pdf_source": self.pdf_source_path if self.pdf_source_path else "unknown"
                    }
                    
                    # Add type-specific metadata including file paths
                    if chunk_type == "image_chunks":
                        metadata.update({
                            "generated_caption": chunk.get("generated_caption", ""),
                            "original_description": chunk.get("image_description", ""),
                            "image_width": chunk.get("metadata", {}).get("width", 0),
                            "image_height": chunk.get("metadata", {}).get("height", 0),
                            "image_path": chunk.get("image_path", ""),  # Path to saved image file
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
        return total_inserted
    
    def universal_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Universal search across ALL chunk types - finds most relevant content regardless of type
        
        Args:
            query: Search query text
            n_results: Number of results to return
            
        Returns:
            List of search results with content, metadata, and scores from all chunk types
        """
        print(f"\n🔍 Universal search for: '{query}'")
        print("🔧 Searching across all chunk types (text, images, tables)")
        
        # Generate query embedding using your org's model
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Perform search WITHOUT any chunk type filtering
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                # No where filter - get all chunk types
                where=None
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity score (1 - distance)
                    similarity_score = 1 - results["distances"][0][i]
                    
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity_score,
                        "distance": results["distances"][0][i]
                    }
                    formatted_results.append(result)
            
            # Sort by similarity score (highest first)
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            print(f"✅ Found {len(formatted_results)} results across all types")
            
            # Show breakdown by type
            type_counts = {}
            for result in formatted_results:
                chunk_type = result["metadata"].get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            print(f"📊 Result breakdown: {dict(type_counts)}")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def create_pdf_link(self, page_number: int, pdf_path: str = None) -> str:
        """
        Create a clickable link to specific page in PDF
        
        Args:
            page_number: Page number in PDF
            pdf_path: Path to PDF file (uses self.pdf_source_path if None)
            
        Returns:
            Formatted link string
        """
        pdf_file = pdf_path or self.pdf_source_path
        if not pdf_file or page_number < 0:
            return "PDF link not available"
        
        # Create file URL that can be clicked to open PDF at specific page
        # This format depends on your frontend/system
        return f"file://{os.path.abspath(pdf_file)}#page={page_number}"
    
    def create_intelligent_multimodal_prompt(self, user_query: str, retrieved_docs: List[Dict]) -> List:
        """
        Create an intelligent multimodal RAG prompt that automatically includes relevant content
        
        Args:
            user_query: The user's question
            retrieved_docs: List of retrieved documents from ChromaDB (all types)
            
        Returns:
            List of multimodal messages for your xyz chat model
        """
        # Automatically separate content types
        text_contents = []
        image_contents = []
        context_parts = []
        
        # Analyze and categorize retrieved documents
        text_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") != "image"]
        image_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") == "image"]
        
        print(f"📝 Processing {len(text_docs)} text/table chunks and {len(image_docs)} image chunks")
        
        # Process text and table chunks
        for i, doc in enumerate(text_docs, 1):
            chunk_type = doc["metadata"].get("chunk_type", "unknown")
            page_num = doc["metadata"].get("page_number", "N/A")
            score = doc.get("score", 0)
            content = doc["content"]
            
            doc_context = f"""Document {i} ({chunk_type.title()}, Page: {page_num}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num)}
"""
            context_parts.append(doc_context)
        
        # Process image chunks
        for i, doc in enumerate(image_docs, 1):
            page_num = doc["metadata"].get("page_number", "N/A")
            score = doc.get("score", 0)
            content = doc["content"]
            image_path = doc["metadata"].get("image_path", "")
            
            if image_path and os.path.exists(image_path):
                try:
                    # Create AFM Image object for your model
                    afm_image = AFMImage.from_url(image_path)
                    image_contents.append(afm_image)
                    
                    # Add image description to context
                    image_context = f"""Image {len(image_contents)} (Page: {page_num}, Relevance: {score:.3f}):
Description: {content}
Source: {self.create_pdf_link(page_num)}
"""
                    context_parts.append(image_context)
                except Exception as e:
                    print(f"⚠️ Error loading image {image_path}: {e}")
                    # Fallback to text description
                    image_context = f"""Image Description {i} (Page: {page_num}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num)}
"""
                    context_parts.append(image_context)
            else:
                # No image file, just use description
                image_context = f"""Image Description {i} (Page: {page_num}, Relevance: {score:.3f}):
{content}
Source: {self.create_pdf_link(page_num)}
"""
                context_parts.append(image_context)
        
        # Combine all context
        full_context = "\n" + "-" * 50 + "\n".join(context_parts) + "-" * 50
        
        # Create adaptive system instruction
        has_images = len(image_contents) > 0
        has_tables = any(doc["metadata"].get("chunk_type") == "table" for doc in retrieved_docs)
        
        system_content = f"""You are a helpful assistant that answers questions based on provided documents from a PDF.

Content Available:
- Text documents: {len(text_docs)}
- Images: {len(image_docs)} ({len(image_contents)} loaded)
- Tables: {sum(1 for doc in retrieved_docs if doc["metadata"].get("chunk_type") == "table")}

Instructions:
- Use ALL the information provided in the documents {"and images " if has_images else ""}below
- When referencing content, mention the document/image number and page
{"- For images, describe what you see and relate it to the question" if has_images else ""}
{"- For tables, interpret the data and provide insights" if has_tables else ""}
- Include source links when relevant for user navigation
- Be comprehensive but concise
- If insufficient information, state clearly what's missing
- Prioritize higher relevance scores when conflicts arise

Document and Image Context:""" + full_context
        
        # Create multimodal message contents
        message_contents = [system_content]
        
        # Add all images to the message if any
        for image in image_contents:
            message_contents.append(image)
        
        # Add user query
        message_contents.append(user_query)
        
        # Create the multimodal message
        messages = [HumanMultimodalMessage(contents=message_contents)]
        
        return messages
    
    def intelligent_query(self, user_query: str, n_results: int = 8, min_score: float = 0.0) -> Dict:
        """
        Intelligent universal query that automatically handles all content types
        
        Args:
            user_query: The user's question (no need to specify content type)
            n_results: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            Dictionary with answer, content info, PDF links, and metadata
        """
        print(f"\n🤖 Processing intelligent universal query: '{user_query}'")
        
        # Step 1: Universal search across ALL content types
        retrieved_docs = self.universal_search(user_query, n_results=n_results)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "retrieved_docs": [],
                "content_summary": {"text": 0, "images": 0, "tables": 0},
                "pdf_links": [],
                "query": user_query,
                "success": False
            }
        
        # Step 2: Filter by minimum score if specified
        if min_score > 0:
            retrieved_docs = [doc for doc in retrieved_docs if doc.get("score", 0) >= min_score]
            if not retrieved_docs:
                return {
                    "answer": f"No documents found with similarity score above {min_score}.",
                    "retrieved_docs": [],
                    "content_summary": {"text": 0, "images": 0, "tables": 0},
                    "pdf_links": [],
                    "query": user_query,
                    "success": False
                }
        
        print(f"📚 Using {len(retrieved_docs)} most relevant documents")
        
        # Step 3: Analyze content types found
        content_summary = {"text": 0, "images": 0, "tables": 0, "other": 0}
        images_info = []
        pdf_links = []
        
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            chunk_type = metadata.get("chunk_type", "other")
            page_num = metadata.get("page_number", -1)
            
            # Count content types
            if chunk_type == "text":
                content_summary["text"] += 1
            elif chunk_type == "image":
                content_summary["images"] += 1
            elif chunk_type == "table":
                content_summary["tables"] += 1
            else:
                content_summary["other"] += 1
            
            # Add PDF link for each document
            if page_num >= 0:
                pdf_link = {
                    "page": page_num,
                    "url": self.create_pdf_link(page_num),
                    "chunk_id": doc["id"],
                    "chunk_type": chunk_type,
                    "score": doc["score"]
                }
                pdf_links.append(pdf_link)
            
            # Collect image information
            if chunk_type == "image":
                image_info = {
                    "chunk_id": doc["id"],
                    "image_path": metadata.get("image_path", ""),
                    "description": doc["content"],
                    "page": page_num,
                    "pdf_link": self.create_pdf_link(page_num),
                    "caption": metadata.get("generated_caption", ""),
                    "dimensions": f"{metadata.get('image_width', 0)}x{metadata.get('image_height', 0)}",
                    "score": doc["score"]
                }
                images_info.append(image_info)
        
        print(f"📊 Content found: {content_summary}")
        
        # Step 4: Generate intelligent response using all available content
        try:
            print("🧠 Generating comprehensive answer...")
            
            # Use intelligent multimodal prompt that adapts to available content
            messages = self.create_intelligent_multimodal_prompt(user_query, retrieved_docs)
            
            # Call your custom model
            response = chat(messages)
            
            # Extract answer
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
                "query": user_query,
                "n_docs_used": len(retrieved_docs),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "content_summary": content_summary,
                "images": images_info,
                "pdf_links": pdf_links,
                "query": user_query,
                "success": False
            }
    
    def print_intelligent_result(self, result: Dict):
        """Pretty print intelligent query result emphasizing unified response"""
        print("\n" + "="*80)
        print("🤖 UNIFIED RAG RESPONSE")
        print("="*80)
        
        print(f"❓ Query: {result['query']}")
        print(f"✅ Success: {result['success']}")
        print(f"🎯 Response Type: {result.get('response_type', 'unified').title()}")
        
        if result['success']:
            content_summary = result['content_summary']
            total_content = sum(content_summary.values())
            print(f"📚 Sources Synthesized: {total_content}")
            print(f"📊 Content Mix:")
            for content_type, count in content_summary.items():
                if count > 0:
                    print(f"   • {content_type.title()}: {count} sources")
        
        print(f"\n💬 UNIFIED ANSWER:")
        print("=" * 60)
        print(result['answer'])
        print("=" * 60)
        
        # Show supporting evidence breakdown
        if result.get('images'):
            print(f"\n🖼️  Visual Evidence Used ({len(result['images'])} images):")
            for i, img_info in enumerate(result['images'], 1):
                print(f"   {i}. Page {img_info['page']} - {img_info['description'][:80]}...")
                print(f"      📁 {img_info['image_path']}")
        
        if result.get('pdf_links'):
            # Group by content type for organized display
            links_by_type = {}
            for link in result['pdf_links']:
                content_type = link['chunk_type']
                if content_type not in links_by_type:
                    links_by_type[content_type] = []
                links_by_type[content_type].append(link)
            
            print(f"\n📑 Source Pages Referenced:")
            for content_type, links in links_by_type.items():
                pages = [f"Page {link['page']}" for link in sorted(links, key=lambda x: x['page'])]
                print(f"   • {content_type.title()}: {', '.join(pages[:5])}")
                if len(pages) > 5:
                    print(f"     ... and {len(pages) - 5} more pages")
        
        # Show quality metrics
        if result.get('retrieved_docs'):
            scores = [doc['score'] for doc in result['retrieved_docs']]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"\n📈 Source Quality:")
            print(f"   • Average Relevance: {avg_score:.3f}")
            print(f"   • Best Match Score: {max_score:.3f}")
            print(f"   • Total Sources Used: {len(result['retrieved_docs'])}")
        
        print(f"\n✨ This response synthesizes information from multiple sources into one comprehensive answer.")
        print(f"💡 All relevant text, images, and tables were considered together.")3:
                    print(f"     ... and {len(pages) - 3} more")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_count = self.collection.count()
        
        # Get samples to analyze types
        sample_results = self.collection.get(limit=min(100, total_count))
        
        type_counts = {}
        page_counts = {}
        
        if sample_results["metadatas"]:
            for metadata in sample_results["metadatas"]:
                chunk_type = metadata.get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                
                page_num = metadata.get("page_number", -1)
                if page_num >= 0:
                    page_counts[page_num] = page_counts.get(page_num, 0) + 1
        
        stats = {
            "total_chunks": total_count,
            "chunk_types": type_counts,
            "pages_with_content": len(page_counts),
            "sample_size": len(sample_results["metadatas"]) if sample_results["metadatas"] else 0
        }
        
        return stats
    
    def print_stats(self):
        """Print database statistics"""
        stats = self.get_stats()
        
        print("\n📊 DATABASE STATISTICS")
        print("=" * 50)
        print(f"📦 Total Chunks: {stats['total_chunks']}")
        print(f"📄 Pages with Content: {stats['pages_with_content']}")
        
        print(f"\n📝 Chunk Types:")
        for chunk_type, count in stats["chunk_types"].items():
            print(f"   • {chunk_type}: {count}")
        
        if stats["sample_size"] < stats["total_chunks"]:
            print(f"\n⚠️  Statistics based on sample of {stats['sample_size']} chunks")

# Simplified usage functions for universal querying
def main_universal_rag_example():
    """Complete example of universal RAG - user just asks questions"""
    
    # Initialize ChromaDB with PDF source
    db = ChromaDBManager(
        persist_directory="./my_pdf_db",
        embedding_model_name="bembedd-1rg",
        pdf_source_path="./original_document.pdf"  # Your PDF path
    )
    
    # Show database stats
    db.print_stats()
    
    # Universal RAG Query Examples - NO chunk type specification needed
    print("\n" + "="*80)
    print("🤖 UNIVERSAL RAG EXAMPLES - JUST ASK QUESTIONS!")
    print("="*80)
    
    # Example 1: Any question - system finds relevant content automatically
    result1 = db.intelligent_query("What are the main findings in this document?")
    db.print_intelligent_result(result1)
    
    # Example 2: Another question - system adapts automatically
    result2 = db.intelligent_query("Show me organizational information and leadership details")
    db.print_intelligent_result(result2)
    
    # Example 3: Technical question - system finds best matches
    result3 = db.intelligent_query("What are the financial results and performance metrics?")
    db.print_intelligent_result(result3)
    
    print("\n🎉 Universal RAG system is working!")
    print("💡 Key features:")
    print("   ✅ No chunk type specification needed")
    print("   ✅ Automatic content type detection")
    print("   ✅ Intelligent multimodal responses") 
    print("   ✅ PDF source linking")
    print("   ✅ Relevance-based ranking")
    
    return db

def interactive_universal_rag():
    """Interactive universal RAG session - just ask any question"""
    db = ChromaDBManager(
        embedding_model_name="bembedd-1rg",
        pdf_source_path="./original_document.pdf"  # Update with your PDF path
    )
    
    print("\n🤖 Universal RAG - Just Ask Anything!")
    print("The system will automatically find the most relevant content (text, images, tables)")
    print("Commands: 'quit' to exit, 'stats' for database info")
    print("-" * 80)
    
    while True:
        user_query = input("\n❓ Ask me anything: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if user_query.lower() == 'stats':
            db.print_stats()
            continue
        
        if not user_query:
            continue
        
        # Process with intelligent universal RAG
        result = db.intelligent_query(user_query)
        
        # Show full result
        db.print_intelligent_result(result)

if __name__ == "__main__":
    print("🤖 Universal ChromaDB PDF RAG System")
    print("Features:")
    print("✅ Ask any question - no content type specification needed")
    print("✅ Automatic text + image + table retrieval")
    print("✅ Intelligent multimodal AI responses")
    print("✅ PDF source linking")
    print("✅ Relevance-based content ranking")
    
    print("\nHow to use:")
    print("1. Just ask any question about your PDF")
    print("2. System automatically finds most relevant content")
    print("3. Get comprehensive answers with sources")
    
    # Run interactive session
    interactive_universal_rag()
