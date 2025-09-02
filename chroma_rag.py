import json
import chromadb
from chromadb.config import Settings
from inter.core.clients.xorclient import xorclient
from xyz import chat  # Your custom model interface
from langchain.schema import Human, System
from typing import List, Dict, Any
import uuid
from datetime import datetime

class ChromaDBManager:
    def __init__(self, 
                 persist_directory: str = "./chroma_db", 
                 collection_name: str = "pdf_chunks",
                 embedding_model_name: str = "bembedd-1rg"):
        """
        Initialize ChromaDB for PDF chunks with your organization's embedding model
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model_name: Your organization's embedding model name
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
        
        print(f"✓ ChromaDB initialized with collection: {collection_name}")
        print(f"✓ Database path: {persist_directory}")
        print(f"✓ Using embedding model: {embedding_model_name}")
        
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
                    
                    # Prepare metadata
                    metadata = {
                        "chunk_type": chunk_type.replace("_chunks", ""),
                        "page_number": chunk.get("page_number", -1),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add type-specific metadata
                    if chunk_type == "image_chunks":
                        metadata.update({
                            "generated_caption": chunk.get("generated_caption", ""),
                            "original_description": chunk.get("image_description", ""),
                            "image_width": chunk.get("metadata", {}).get("width", 0),
                            "image_height": chunk.get("metadata", {}).get("height", 0)
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
    
    def search(self, 
               query: str, 
               n_results: int = 5, 
               chunk_type: str = None) -> List[Dict]:
        """
        Search for similar chunks using your organization's embedding model
        
        Args:
            query: Search query text
            n_results: Number of results to return
            chunk_type: Filter by chunk type ("text", "image", "table")
            
        Returns:
            List of search results with content, metadata, and scores
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding using your org's model
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Prepare filter
        where_filter = None
        if chunk_type:
            where_filter = {"chunk_type": chunk_type}
            print(f"🔧 Filtering by type: {chunk_type}")
        
        # Perform search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
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
            
            print(f"✅ Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def create_rag_prompt(self, user_query: str, retrieved_docs: List[Dict]) -> List:
        """
        Create a RAG prompt for your custom model using the langchain schema format
        
        Args:
            user_query: The user's question
            retrieved_docs: List of retrieved documents from ChromaDB
            
        Returns:
            List of messages in the format expected by your xyz chat model
        """
        # Create context from retrieved documents
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            chunk_type = doc["metadata"].get("chunk_type", "unknown")
            page_num = doc["metadata"].get("page_number", "N/A")
            score = doc.get("score", 0)
            content = doc["content"]
            
            # Format each document with metadata
            doc_context = f"""Document {i} (Type: {chunk_type}, Page: {page_num}, Relevance: {score:.3f}):
{content}
"""
            context_parts.append(doc_context)
        
        # Combine all context
        full_context = "\n" + "-" * 50 + "\n".join(context_parts) + "-" * 50
        
        # Create the system message with instructions
        system_content = """You are a helpful assistant that answers questions based on the provided document chunks from a PDF. 

Instructions:
- Use ONLY the information provided in the document chunks below to answer the question
- If the documents don't contain enough information to answer the question, say so clearly
- When referencing information, mention which document number you're citing
- Be concise but comprehensive in your answers
- If the question asks about images, tables, or specific content types, prioritize those document types
- Maintain accuracy and don't make assumptions beyond what's in the documents

Document Context:""" + full_context
        
        # Create messages list in the format your model expects
        messages = [
            System(content=system_content),
            Human(content=user_query)
        ]
        
        return messages
    
    def query_with_rag(self, 
                       user_query: str, 
                       n_results: int = 5, 
                       chunk_type: str = None,
                       min_score: float = 0.0) -> Dict:
        """
        Query the ChromaDB and get an AI-generated answer using RAG
        
        Args:
            user_query: The user's question
            n_results: Number of documents to retrieve for context
            chunk_type: Filter by chunk type ("text", "image", "table")
            min_score: Minimum similarity score threshold
            
        Returns:
            Dictionary containing the answer, retrieved docs, and metadata
        """
        print(f"\n🤖 Processing RAG query: '{user_query}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.search(user_query, n_results=n_results, chunk_type=chunk_type)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "retrieved_docs": [],
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
                    "query": user_query,
                    "success": False
                }
        
        print(f"📚 Using {len(retrieved_docs)} documents for context")
        
        # Step 3: Create RAG prompt
        messages = self.create_rag_prompt(user_query, retrieved_docs)
        
        # Step 4: Call your custom model
        try:
            print("🧠 Generating answer with your custom model...")
            response = chat(messages)
            
            # Extract answer from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            print("✅ Answer generated successfully")
            
            return {
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "query": user_query,
                "n_docs_used": len(retrieved_docs),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "query": user_query,
                "success": False
            }
    
    def print_rag_result(self, result: Dict):
        """Pretty print RAG query result"""
        print("\n" + "="*80)
        print("🤖 RAG QUERY RESULT")
        print("="*80)
        
        print(f"❓ Query: {result['query']}")
        print(f"✅ Success: {result['success']}")
        
        if result['success']:
            print(f"📚 Documents Used: {result['n_docs_used']}")
        
        print(f"\n💬 Answer:")
        print("-" * 40)
        print(result['answer'])
        print("-" * 40)
        
        # Show retrieved documents
        if result['retrieved_docs']:
            print(f"\n📋 Retrieved Documents ({len(result['retrieved_docs'])}):")
            for i, doc in enumerate(result['retrieved_docs'], 1):
                metadata = doc['metadata']
                print(f"\n{i}. 📄 Chunk ID: {doc['id']}")
                print(f"   🎯 Score: {doc['score']:.3f}")
                print(f"   📑 Type: {metadata.get('chunk_type', 'unknown')}")
                print(f"   📖 Page: {metadata.get('page_number', 'N/A')}")
                
                # Content preview
                content = doc['content']
                if len(content) > 150:
                    print(f"   💬 Content: {content[:150]}...")
                else:
                    print(f"   💬 Content: {content}")
    
    def print_results(self, results: List[Dict], query: str = ""):
        """Pretty print search results"""
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n📋 Search Results for: '{query}'")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            
            print(f"\n{i}. 📄 Chunk ID: {result['id']}")
            print(f"   🎯 Similarity Score: {result['score']:.3f}")
            print(f"   📑 Type: {metadata.get('chunk_type', 'unknown')}")
            print(f"   📖 Page: {metadata.get('page_number', 'N/A')}")
            
            # Type-specific information
            if metadata.get('chunk_type') == 'image':
                print(f"   🖼️  Generated Caption: {metadata.get('generated_caption', 'N/A')}")
                print(f"   📐 Dimensions: {metadata.get('image_width', 0)}x{metadata.get('image_height', 0)}")
            elif metadata.get('chunk_type') == 'table':
                print(f"   📊 Table Size: {metadata.get('num_rows', 0)} rows × {metadata.get('num_cols', 0)} cols")
                print(f"   📝 Description: {metadata.get('table_description', 'N/A')}")
            
            # Content preview
            content = result["content"]
            if len(content) > 200:
                print(f"   💬 Content: {content[:200]}...")
            else:
                print(f"   💬 Content: {content}")
            
            print("-" * 40)
    
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

# Example usage functions with RAG
def main_rag_example():
    """Complete example of using ChromaDB with RAG querying"""
    
    # Initialize ChromaDB with your organization's model
    db = ChromaDBManager(
        persist_directory="./my_pdf_db",
        embedding_model_name="bembedd-1rg"  # Your org's model
    )
    
    # Insert chunks from JSON file (if needed)
    # chunks_inserted = db.insert_chunks_from_json("chunks_with_embeddings.json")
    
    # Show database stats
    db.print_stats()
    
    # RAG Query Examples
    print("\n" + "="*80)
    print("🤖 RAG QUERY EXAMPLES WITH YOUR CUSTOM MODEL")
    print("="*80)
    
    # Example 1: General business question
    result1 = db.query_with_rag(
        "What is the company's financial performance this quarter?",
        n_results=5
    )
    db.print_rag_result(result1)
    
    # Example 2: Ask about specific people/executives
    result2 = db.query_with_rag(
        "Who are the key executives and what are their roles?",
        n_results=4,
        min_score=0.3
    )
    db.print_rag_result(result2)
    
    # Example 3: Focus on tables only
    result3 = db.query_with_rag(
        "Show me the budget allocation by department",
        n_results=3,
        chunk_type="table"
    )
    db.print_rag_result(result3)
    
    # Example 4: Focus on images
    result4 = db.query_with_rag(
        "What images are available and what do they show?",
        n_results=5,
        chunk_type="image"
    )
    db.print_rag_result(result4)
    
    print("\n🎉 RAG system with your custom model is working!")
    print("💡 You can now ask natural language questions about your PDF content!")
    
    return db

def interactive_rag_session():
    """Interactive RAG session for asking questions"""
    db = ChromaDBManager(embedding_model_name="bembedd-1rg")
    
    print("\n🤖 Interactive RAG Session Started!")
    print("Ask questions about your PDF content. Type 'quit' to exit.")
    print("-" * 60)
    
    while True:
        user_query = input("\n❓ Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            continue
        
        # Process the query
        result = db.query_with_rag(user_query, n_results=5)
        
        # Show just the answer (compact format)
        print(f"\n🤖 Answer: {result['answer']}")
        
        # Optionally show sources
        show_sources = input("\n📚 Show source documents? (y/n): ").strip().lower()
        if show_sources == 'y':
            db.print_rag_result(result)

if __name__ == "__main__":
    print("🤖 Enhanced ChromaDB PDF RAG System")
    print("Required packages:")
    print("- pip install chromadb langchain")
    print("- Your internal packages: inter.core.clients.xorclient, xyz")
    
    print("\n🔧 Make sure your xyz chat model is properly configured")
    print("Replace file paths and model names with your actual values")
    
    # Uncomment to run examples:
    # db = main_rag_example()
    
    # For interactive session:
    # interactive_rag_session()