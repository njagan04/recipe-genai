from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings before importing heavy libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from src.graph.graph import build_graph
from src.llm.generator import get_chat_response

app = FastAPI(title="Recipe GenAI API")

# Configure CORS so the React frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LangGraph
graph = build_graph()

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    user_input: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    recipe: Dict[str, Any]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Backend is running"}

@app.post("/api/search")
def search_recipes(request: SearchRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="Ingredients input cannot be empty")
        
    try:
        # Run the LangGraph state machine
        result = graph.invoke({"user_input": request.user_input})
        filtered_recipes = result.get("filtered_recipes", [])
        return {"recipes": filtered_recipes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_chef(request: ChatRequest):
    try:
        # Convert Pydantic messages to list of dicts expected by generator.py
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response = get_chat_response(messages_dict, request.recipe)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
