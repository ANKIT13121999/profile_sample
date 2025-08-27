import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
from datetime import datetime

class ChromaDBManager:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "pdf_chunks"):
        """
        Initialize ChromaDB for PDF chunks
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
        """
        print("Initializing ChromaDB...")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize text encoder for query embeddings
        print("Loading sentence transformer model...")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"✓ ChromaDB initialized with collection: {collection_name}")
        print(f"✓ Database path: {persist_directory}")
        
        # Check existing data
        existing_count = self.collection.count()
        print(f"✓ Existing chunks in database: {existing_count}")
    
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
        Search for similar chunks
        
        Args:
            query: Search query text
            n_results: Number of results to return
            chunk_type: Filter by chunk type ("text", "image", "table")
            
        Returns:
            List of search results with content, metadata, and scores
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.text_model.encode(query, convert_to_tensor=False).tolist()
        
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

# Example usage functions
def main_example():
    """Complete example of using ChromaDB with PDF chunks"""
    
    # Initialize ChromaDB
    db = ChromaDBManager(persist_directory="./my_pdf_db")
    
    # Insert chunks from JSON file
    chunks_inserted = db.insert_chunks_from_json("chunks_with_embeddings.json")
    
    if chunks_inserted > 0:
        # Show database stats
        db.print_stats()
        
        # Example searches
        print("\n" + "="*60)
        print("EXAMPLE SEARCHES")
        print("="*60)
        
        # Search 1: General query
        # results = db.search("", n_results=3)
        # db.print_results(results, "CEO leadership team")
        
        # Search 2: Financial data
        results = db.search("what is a data incident in humanitarian response?", n_results=3)
        db.print_results(results, "what is a data incident in humanitarian response?")
        
        # Search 3: Images only
        results = db.search("Risk managment strategy or approach", n_results=3, chunk_type="image")
        db.print_results(results, "Risk managment strategy or approach")
        
        # Search 4: Tables only
        # results = db.search("budget department allocation", n_results=3, chunk_type="table")
        # db.print_results(results, "budget department allocation")
        
        print("\n🎉 ChromaDB setup and search examples completed!")
        print("💡 You can now build Q&A systems, chatbots, or semantic search applications!")
    
    return db

def quick_search_example():
    """Quick search example for existing database"""
    db = ChromaDBManager()
    
    # Custom searches
    queries = [
        "Who are the executives?",
        "What is the company revenue?", 
        "Show me financial tables",
        "Employee photos"
    ]
    
    for query in queries:
        results = db.search(query, n_results=2)
        db.print_results(results, query)

if __name__ == "__main__":
    print("📚 ChromaDB PDF Chunk Manager")
    print("Required: pip install chromadb sentence-transformers")
    print("\n🔧 Replace 'chunks_with_embeddings.json' with your actual file path")
    
    # Uncomment to run:
    db = main_example()
    
    # For quick searches on existing database:
    # quick_search_example()
