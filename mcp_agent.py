from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from lox_agent import ChromaDBManager
from xyz import chat   # your org’s chat model (used for routing)

app = FastAPI(title="MCP Server", description="Multi-Agent Orchestrator")

# ------------------------------
# Input schema
# ------------------------------
class QueryInput(BaseModel):
    query: str
    n_results: int = 5
    chunk_type: str = None
    min_score: float = 0.0

# ------------------------------
# Initialize lox Agent
# ------------------------------
lox_agent = ChromaDBManager(
    persist_directory="./my_pdf_db",
    embedding_model_name="bembedd-1rg"
)

# ------------------------------
# Router (LLM-based agent selector)
# ------------------------------
def route_query(query: str) -> str:
    """
    Use your custom LLM to decide which agent to call.
    Returns: "lox" | "ladar"
    """
    messages = [
        {"role": "system", "content": "You are a router. Decide if the user query is about PDF documents (lox) or tickets (ladar). Respond only 'lox' or 'ladar'."},
        {"role": "user", "content": query}
    ]
    response = chat(messages)  # using your org’s LLM
    if hasattr(response, "content"):
        decision = response.content.strip().lower()
    else:
        decision = str(response).strip().lower()
    return "lox" if "lox" in decision else "ladar"

# ------------------------------
# Endpoints
# ------------------------------
@app.post("/query")
def handle_query(req: QueryInput) -> Dict[str, Any]:
    # Step 1: Route query
    agent = route_query(req.query)

    if agent == "lox":
        # Step 2a: Handle with lox agent
        result = lox_agent.query_with_rag(
            user_query=req.query,
            n_results=req.n_results,
            chunk_type=req.chunk_type,
            min_score=req.min_score
        )
        return {"agent": "lox", "query": req.query, "result": result}

    elif agent == "ladar":
        # Step 2b: ladar stub for now
        return {"agent": "ladar", "query": req.query, "result": "🚧 ladar agent under construction"}

