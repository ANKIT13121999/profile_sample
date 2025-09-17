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

class ChromaDBRAGTool:
    """
    ChromaDB RAG Tool that can be registered as a tool for MCP integration
    Provides universal PDF document search and intelligent multimodal responses
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db", 
                 collection_name: str = "pdf_chunks",
                 embedding_model_name: str = "bembedd-1rg",
                 pdf_source_path: str = None,
                 client=None,
                 project_id: str = None):
        """
        Initialize ChromaDB RAG Tool
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model_name: Your organization's embedding model name
            pdf_source_path: Path to the original PDF file for linking
            client: MCP client instance (for tool registration)
            project_id: Project ID for tool registration
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.pdf_source_path = pdf_source_path
        self.client = client
        self.project_id = project_id
        
        # Initialize ChromaDB
        self._initialize_chromadb()
        
        # Initialize embedding client
        self.embedding_client = xorclient()
        
        # Tool metadata for registration
        self.tool_metadata = {
            "name": "ChromaDBRAGTool",
            "function_name": "search_documents",
            "description": "Universal PDF document search with intelligent multimodal responses. Searches across text, images, and tables to provide comprehensive answers.",
            "category": "documentation",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Search query - can be any question about the PDF content",
                    "required": True
                },
                "n_results": {
                    "type": "integer", 
                    "description": "Number of results to return (default: 8)",
                    "required": False,
                    "default": 8
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score threshold (default: 0.0)",
                    "required": False,
                    "default": 0.0
                }
            }
        }
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        print(f"Initializing ChromaDB at {self.persist_directory}...")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        existing_count = self.collection.count()
        print(f"ChromaDB initialized. Existing chunks: {existing_count}")
    
    def register_as_tool(self) -> Optional[str]:
        """
        Register this ChromaDB RAG system as a tool that can be used in agentic workflows
        
        Returns:
            Tool ID if successful, None if failed
        """
        if not self.client or not self.project_id:
            print("Error: Client and project_id required for tool registration")
            return None
            
        try:
            # Use the new import_workflow_as_tool function from the Python SDK
            response = self.client.agents.import_workflow_as_tool(
                project_id=self.project_id,
                workflow_id="chromadb_rag_workflow",
                function_name=self.tool_metadata["function_name"],
                function_description=self.tool_metadata["description"],
                category=self.tool_metadata["category"]
            )
            
            print(f"ChromaDB RAG Tool successfully registered using import_workflow_as_tool!")
            print(f"Tool ID: {response.id}")
            return response.id
            
        except Exception as e:
            print(f"Error registering ChromaDB RAG Tool: {e}")
            return None
    
    def create_workflow_function(self) -> callable:
        """
        Create the main workflow function that will be called by the tool system
        
        Returns:
            Callable function for the workflow
        """
        def search_documents(query: str, n_results: int = 8, min_score: float = 0.0) -> Dict[str, Any]:
            """
            Universal document search function - main tool entry point
            
            Args:
                query: Search query about PDF content
                n_results: Number of results to return
                min_score: Minimum similarity score threshold
                
            Returns:
                Dictionary with search results and AI-generated answer
            """
            try:
                print(f"ChromaDB RAG Tool processing query: '{query}'")
                
                # Perform universal search and generate intelligent response
                result = self.intelligent_query(query, n_results, min_score)
                
                # Format for tool output
                tool_output = {
                    "success": result.get("success", False),
                    "answer": result.get("answer", "No answer generated"),
                    "query": query,
                    "sources_found": result.get("n_docs_used", 0),
                    "content_types": result.get("content_summary", {}),
                    "pdf_source": self.pdf_source_path,
                    "tool_name": "ChromaDBRAGTool"
                }
                
                # Add source information for transparency
                if result.get("pdf_links"):
                    pages = list(set([link["page"] for link in result["pdf_links"] if link["page"] >= 0]))
                    tool_output["source_pages"] = sorted(pages)
                
                return tool_output
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "query": query,
                    "tool_name": "ChromaDBRAGTool"
                }
        
        return search_documents
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query using organization's model"""
        try:
            response = self.embedding_client.get_embedding(
                input=query, 
                model_name=self.embedding_model_name
            )
            
            # Extract embedding from response
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            elif isinstance(response, dict) and 'data' in response:
                return response['data'][0]['embedding']
            else:
                return response
                
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
    
    def universal_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Universal search across ALL chunk types"""
        print(f"Universal search for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            print("Failed to generate query embedding")
            return []
        
        # Search without chunk type filtering
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=None  # No filtering - get all types
            )
            
            # Format results
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
            
            # Sort by similarity score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Show type breakdown
            type_counts = {}
            for result in formatted_results:
                chunk_type = result["metadata"].get("chunk_type", "unknown")
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            print(f"Found {len(formatted_results)} results. Types: {dict(type_counts)}")
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def create_intelligent_multimodal_prompt(self, user_query: str, retrieved_docs: List[Dict]) -> List:
        """Create intelligent multimodal RAG prompt for unified answer generation"""
        
        # Separate content types
        text_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") != "image"]
        image_docs = [doc for doc in retrieved_docs if doc["metadata"].get("chunk_type") == "image"]
        
        print(f"Processing {len(text_docs)} text/table chunks and {len(image_docs)} image chunks")
        
        # Build context sections
        context_parts = []
        image_contents = []
        
        # Process text chunks
        text_chunks = [doc for doc in text_docs if doc["metadata"].get("chunk_type") == "text"]
        if text_chunks:
            context_parts.append("TEXT SOURCES:")
            for i, doc in enumerate(text_chunks, 1):
                page_num = doc["metadata"].get("page_number", "N/A")
                score = doc.get("score", 0)
                content = doc["content"]
                context_parts.append(f"[T{i}] Page {page_num} (Score: {score:.3f}): {content}")
        
        # Process table chunks
        table_chunks = [doc for doc in text_docs if doc["metadata"].get("chunk_type") == "table"]
        if table_chunks:
            if context_parts:
                context_parts.append("")  # Spacing
            context_parts.append("TABLE SOURCES:")
            for i, doc in enumerate(table_chunks, 1):
                page_num = doc["metadata"].get("page_number", "N/A")
                score = doc.get("score", 0)
                content = doc["content"]
                context_parts.append(f"[TABLE{i}] Page {page_num} (Score: {score:.3f}): {content}")
        
        # Process image chunks
        if image_docs:
            if context_parts:
                context_parts.append("")  # Spacing
            context_parts.append("IMAGE SOURCES:")
            for i, doc in enumerate(image_docs, 1):
                page_num = doc["metadata"].get("page_number", "N/A")
                score = doc.get("score", 0)
                content = doc["content"]
                image_path = doc["metadata"].get("image_path", "")
                
                # Load actual images if available
                if image_path and os.path.exists(image_path):
                    try:
                        afm_image = AFMImage.from_url(image_path)
                        image_contents.append(afm_image)
                        context_parts.append(f"[IMG{i}] Page {page_num} (Score: {score:.3f}): {content}")
                    except Exception as e:
                        print(f"Warning: Error loading image {image_path}: {e}")
                        context_parts.append(f"[IMG{i}] Page {page_num} (Score: {score:.3f}): {content}")
                else:
                    context_parts.append(f"[IMG{i}] Page {page_num} (Score: {score:.3f}): {content}")
        
        full_context = "\n".join(context_parts)
        
        # Create system prompt
        has_images = len(image_contents) > 0
        has_tables = len(table_chunks) > 0
        
        system_content = f"""You are an expert assistant providing comprehensive answers based on PDF content.

INSTRUCTION: Generate ONE unified answer that synthesizes ALL provided sources.

Available Sources:
- Text sources: [T1] to [T{len(text_chunks)}] (if any)
- Table sources: [TABLE1] to [TABLE{len(table_chunks)}] (if any) 
- Image sources: [IMG1] to [IMG{len(image_docs)}] (if any)

REQUIREMENTS:
1. Create ONE comprehensive answer addressing the user's question
2. Synthesize information from multiple sources naturally
3. Reference sources using provided IDs (e.g., "According to [T1] and [IMG2]...")
4. Include page numbers when referencing content
5. Provide a unified conclusion

{"Note: Images are attached for visual analysis." if has_images else ""}

CONTEXT:
{full_context}"""
        
        # Create multimodal message
        message_contents = [system_content]
        
        # Add images if any
        for image in image_contents:
            message_contents.append(image)
        
        # Add user query
        message_contents.append(f"USER QUESTION: {user_query}\n\nProvide ONE comprehensive answer synthesizing all sources.")
        
        messages = [HumanMultimodalMessage(contents=message_contents)]
        
        return messages
    
    def intelligent_query(self, user_query: str, n_results: int = 8, min_score: float = 0.0) -> Dict:
        """
        Main intelligent query function - generates unified answers from all content types
        """
        print(f"Processing intelligent query: '{user_query}'")
        
        # Step 1: Universal search
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
        
        # Step 2: Filter by minimum score
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
        
        print(f"Synthesizing {len(retrieved_docs)} documents into unified response")
        
        # Step 3: Analyze content types
        content_summary = {"text": 0, "images": 0, "tables": 0, "other": 0}
        pdf_links = []
        images_info = []
        
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            chunk_type = metadata.get("chunk_type", "other")
            page_num = metadata.get("page_number", -1)
            
            # Count types
            if chunk_type == "text":
                content_summary["text"] += 1
            elif chunk_type == "image":
                content_summary["images"] += 1
            elif chunk_type == "table":
                content_summary["tables"] += 1
            else:
                content_summary["other"] += 1
            
            # Add PDF links
            if page_num >= 0:
                pdf_links.append({
                    "page": page_num,
                    "chunk_id": doc["id"],
                    "chunk_type": chunk_type,
                    "score": doc["score"]
                })
            
            # Collect image info
            if chunk_type == "image":
                images_info.append({
                    "chunk_id": doc["id"],
                    "image_path": metadata.get("image_path", ""),
                    "description": doc["content"],
                    "page": page_num,
                    "score": doc["score"]
                })
        
        print(f"Content found: {content_summary}")
        
        # Step 4: Generate unified answer
        try:
            print("Generating unified answer...")
            
            # Create multimodal prompt
            messages = self.create_intelligent_multimodal_prompt(user_query, retrieved_docs)
            
            # Call model
            response = chat(messages)
            
            # Extract answer
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            print("Unified answer generated successfully")
            
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
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "retrieved_docs": retrieved_docs,
                "content_summary": content_summary,
                "images": images_info,
                "pdf_links": pdf_links,
                "query": user_query,
                "success": False
            }
    
    def load_chunks_from_json(self, json_file_path: str) -> int:
        """Load chunks from JSON file into ChromaDB"""
        print(f"Loading chunks from: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        total_inserted = 0
        
        for chunk_type, chunk_list in chunks_data.items():
            if not chunk_list:
                continue
                
            print(f"Processing {len(chunk_list)} {chunk_type}...")
            
            ids, embeddings, documents, metadatas = [], [], [], []
            
            for chunk in chunk_list:
                try:
                    chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
                    ids.append(chunk_id)
                    
                    embedding = chunk.get("embedding")
                    if not embedding:
                        print(f"Warning: No embedding for chunk: {chunk_id}")
                        continue
                    embeddings.append(embedding)
                    
                    # Prepare content based on type
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
                    
                    # Prepare metadata
                    metadata = {
                        "chunk_type": chunk_type.replace("_chunks", ""),
                        "page_number": chunk.get("page_number", -1),
                        "timestamp": datetime.now().isoformat(),
                        "pdf_source": self.pdf_source_path or "unknown"
                    }
                    
                    # Add type-specific metadata
                    if chunk_type == "image_chunks":
                        metadata.update({
                            "generated_caption": chunk.get("generated_caption", ""),
                            "image_path": chunk.get("image_path", ""),
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
                    print(f"Error processing chunk: {e}")
                    continue
            
            # Insert batch
            if ids and embeddings:
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    print(f"Inserted {len(ids)} {chunk_type}")
                    total_inserted += len(ids)
                except Exception as e:
                    print(f"Error inserting {chunk_type}: {e}")
        
        print(f"Total chunks inserted: {total_inserted}")
        return total_inserted
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_count = self.collection.count()
        return {"total_chunks": total_count}


def create_chromadb_rag_tool(client, project_id: str, workflow_id: str = "doc_search_workflow") -> Optional[str]:
    """
    Create and register ChromaDB RAG workflow as a tool
    
    Args:
        client: MCP client instance
        project_id: Project ID for registration
        workflow_id: Workflow ID for the tool
        
    Returns:
        Tool ID if successful, None if failed
    """
    print("Creating ChromaDB RAG workflow for tool registration...")
    
    # Configure the language model for RAG
    language_model = {
        "model_id": "gemini-2.5-flash",  # Use appropriate model
        "prompt": """SYSTEM: You are a documentation assistant. Your task is to provide helpful information from the documentation based on user queries.

Guidelines:
- Be concise but comprehensive
- Cite sources when possible
- If uncertain, express your uncertainty
- Focus on being helpful and accurate

USER: Context: {context}
Question: {query}
Answer:""",
        "tokens": 800,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    # Configure data retrieval for ChromaDB
    data_retrieval = {
        "embedding_model": "bge-large-en-v1.5",  # Advanced embedding model
        "num_retrieved_documents": 8,
        "similarity_score_threshold": 0.75,
        "filters": []
    }
    
    # Configure guardrails
    guardrails = {
        "enabled": False,
        "guardrails": []
    }
    
    try:
        # Create the workflow
        workflow = client.workflows.create_workflow(
            project_id=project_id,
            workflow_id=workflow_id,
            language_model=language_model,
            data_retrieval=data_retrieval,
            pre_guardrail=guardrails,
            post_guardrail=guardrails,
            workflow_type="smart_answer"  # Use smart_answer for RAG
        )
        
        print(f"ChromaDB RAG workflow created: {workflow_id}")
        
        # Register as tool
        tool_response = client.agents.import_workflow_as_tool(
            project_id=project_id,
            workflow_id=workflow_id,
            function_name="search_documents", 
            function_description="Search and analyze PDF documents using ChromaDB with intelligent multimodal responses",
            category="documentation"
        )
        
        if tool_response:
            actual_tool_id = tool_response.id
            print(f"ChromaDB RAG tool registered successfully with ID: {actual_tool_id}")
            print("This tool can now be used in agentic workflows.")
            return actual_tool_id
        else:
            print("Failed to register ChromaDB RAG tool. Check the error message above.")
            return None
            
    except Exception as e:
        print(f"Error creating ChromaDB RAG workflow: {e}")
        return None


# Example usage for tool registration
def main_tool_registration_example():
    """Example of how to register ChromaDB RAG as a tool"""
    
    # This would be your actual client and project setup
    # client = your_client_instance
    # project_id = "your_project_id"
    
    print("ChromaDB RAG Tool Registration Example")
    print("=" * 50)
    
    # Step 1: Create ChromaDB RAG Tool instance
    rag_tool = ChromaDBRAGTool(
        persist_directory="./my_pdf_db",
        collection_name="pdf_chunks", 
        embedding_model_name="bembedd-1rg",
        pdf_source_path="./original_document.pdf",
        # client=client,  # Uncomment when you have client
        # project_id=project_id  # Uncomment when you have project_id
    )
    
    # Step 2: Load your PDF chunks (if not already loaded)
    # rag_tool.load_chunks_from_json("chunks_with_embeddings.json")
    
    # Step 3: Register as tool (uncomment when ready)
    # tool_id = rag_tool.register_as_tool()
    
    # Step 4: Alternative - Create workflow and register
    # tool_id = create_chromadb_rag_tool(client, project_id, "doc_search_workflow")
    
    print("\nNext steps:")
    print("1. Uncomment client and project_id lines")
    print("2. Load your PDF chunks using load_chunks_from_json()")
    print("3. Call register_as_tool() or create_chromadb_rag_tool()")
    print("4. Use the tool ID in your MCP server integration")
    
    # Test the core functionality
    print("\nTesting core RAG functionality...")
    test_result = rag_tool.intelligent_query("What are the main topics covered?")
    print(f"Test query success: {test_result.get('success', False)}")
    
    return rag_tool

if __name__ == "__main__":
    main_tool_registration_example()