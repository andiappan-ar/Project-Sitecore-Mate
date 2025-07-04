# src/python_server/main.py

import os
import json
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio # Import asyncio for running blocking code in a thread pool
import httpx # For making asynchronous HTTP requests to the Gemini API

# Import ChromaDB
import chromadb
# Corrected import for Collection type
from chromadb import Collection # Changed from chromadb.api.models.collections import Collection

# Import SentenceTransformer for actual embeddings
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Path where ChromaDB will store its data files
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), ".chroma_db_data")
# Base name for ChromaDB collections (will be suffixed with environment ID)
CHROMA_COLLECTION_BASE_NAME = "sitecore_content"

# Gemini API Configuration (hardcoded for POC as requested)
# IMPORTANT: For production, always use environment variables or a secrets management system.
GEMINI_API_KEY = "" # Hardcoded API Key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# New: Define a maximum acceptable distance for retrieved results.
# This value is highly dependent on your embedding model and data.
# You'll need to experiment with this. A good starting point might be 1.0 or 1.5.
# If you get too few results, increase it. If you get too many irrelevant, decrease it.
MAX_DISTANCE_THRESHOLD = 1.5 # Adjust this value based on experimentation

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ChromaDB Indexing Service",
    description="A FastAPI service to receive content from Next.js and index it into ChromaDB.",
    version="1.0.0"
)

# --- ChromaDB Client and Embedding Model Initialization ---
chroma_client: Optional[chromadb.PersistentClient] = None
embedding_model: Optional[SentenceTransformer] = None

def initialize_chromadb_and_model():
    """Initializes the ChromaDB client and embedding model."""
    global chroma_client, embedding_model

    if chroma_client and embedding_model:
        print("ChromaDB Client and Embedding Model already initialized.", file=sys.stderr)
        return

    try:
        # 1. Initialize ChromaDB PersistentClient
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        print(f"ChromaDB: Initializing PersistentClient at {CHROMA_PERSIST_DIRECTORY}", file=sys.stderr)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

        # 2. Load the SentenceTransformer embedding model
        print("Loading SentenceTransformer embedding model 'all-MiniLM-L6-v2'...", file=sys.stderr)
        # This will download the model if not already cached
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully.", file=sys.stderr)

    except Exception as e:
        print(f"ERROR: Failed to initialize ChromaDB Client or Embedding Model: {e}", file=sys.stderr)
        raise # Re-raise to prevent the app from starting if initialization fails

# Helper to get or create a ChromaDB collection for a specific environment
def get_chroma_collection(environment_id: str) -> Collection:
    """
    Gets or creates a ChromaDB collection for a given environment ID.
    Each environment will have its own collection to segregate data.
    """
    if not chroma_client:
        raise HTTPException(status_code=500, detail="ChromaDB Client not initialized.")
    
    collection_name = f"{CHROMA_COLLECTION_BASE_NAME}_{environment_id}"
    print(f"ChromaDB: Getting/Creating collection '{collection_name}'...", file=sys.stderr)
    # IMPORTANT: Set embedding_function=None because we are providing embeddings directly
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=None
    )
    print(f"ChromaDB: Collection '{collection_name}' ready.", file=sys.stderr)
    return collection


# Initialize ChromaDB client and embedding model on startup
@app.on_event("startup")
async def startup_event():
    try:
        # Run the blocking initialization in a separate thread to avoid blocking the Uvicorn event loop
        await asyncio.to_thread(initialize_chromadb_and_model)
    except Exception as e:
        print(f"CRITICAL ERROR: Application startup failed due to initialization error: {e}", file=sys.stderr)
        raise

# --- Data Model for Incoming Content ---
class ScrapedItem(BaseModel):
    id: str
    name: str
    path: str
    url: str
    language: str
    content: str
    childrenPaths: Optional[List[Dict[str, str]]] = None
    environmentId: str # New field to associate content with an environment

# --- Data Model for Incoming Query ---
class QueryRequest(BaseModel):
    query: str
    environmentId: str # New field to specify which environment to query
    n_results: int = 5 # Default to top 5 results

# --- Endpoint to receive and index content ---
@app.post("/index-content")
async def index_content(item: ScrapedItem):
    """
    Receives a scraped content item, generates embeddings, and indexes it into
    the ChromaDB collection associated with its environmentId.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")
    
    # Get the specific collection for this environment
    chroma_collection = get_chroma_collection(item.environmentId)

    print(f"Received item for indexing: {item.id} ({item.language}) - {item.path} into collection {chroma_collection.name}", file=sys.stderr)

    try:
        # Generate actual embedding for the content using the loaded model
        embedding = await asyncio.to_thread(embedding_model.encode, item.content, convert_to_tensor=False)
        embedding_list = embedding.tolist() # Convert numpy array to list for JSON serialization

        # --- Prepare metadata ---
        metadata = {
            "name": item.name,
            "path": item.path,
            "url": item.url,
            "language": item.language,
            "environmentId": item.environmentId # Store environment ID in metadata
        }

        # --- Add/Update document in ChromaDB ---
        chroma_collection.add(
            documents=[item.content],
            metadatas=[metadata],
            ids=[f"{item.id}-{item.language}"], # Use a combined ID for uniqueness across languages
            embeddings=[embedding_list] # Provide the generated numerical embedding
        )
        print(f"Successfully indexed item {item.id} ({item.language}) into ChromaDB collection {chroma_collection.name}.", file=sys.stderr)
        return {"status": "success", "message": f"Item {item.id} ({item.language}) indexed successfully."}

    except Exception as e:
        print(f"ERROR indexing item {item.id} ({item.language}): {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to index content: {e}")

# --- Endpoint to query content ---
@app.post("/query-content")
async def query_content(request: QueryRequest):
    """
    Receives a query string and environmentId, generates its embedding,
    performs a similarity search in the specified ChromaDB collection,
    and returns relevant content.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")

    # Get the specific collection for this environment
    chroma_collection = get_chroma_collection(request.environmentId)

    print(f"Received query: '{request.query}' for environment '{request.environmentId}' (n_results: {request.n_results})", file=sys.stderr)

    try:
        # Generate actual embedding for the incoming query
        query_embedding = await asyncio.to_thread(embedding_model.encode, request.query, convert_to_tensor=False)
        query_embedding_list = query_embedding.tolist()

        # --- Perform similarity search in ChromaDB ---
        results = chroma_collection.query(
            query_embeddings=[query_embedding_list], # Provide the generated query embedding
            n_results=request.n_results,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results for sending back to Next.js
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Apply distance threshold filter
                if distance <= MAX_DISTANCE_THRESHOLD:
                    formatted_results.append({
                        "content": doc_content,
                        "metadata": metadata,
                        "distance": distance
                    })
                else:
                    print(f"DEBUG: Skipping result due to high distance ({distance}): {metadata.get('path', 'N/A')}", file=sys.stderr)
        
        print(f"ChromaDB: Found {len(formatted_results)} results for query in collection {chroma_collection.name} (after distance filter).", file=sys.stderr)
        return {"status": "success", "results": formatted_results}

    except Exception as e:
        print(f"ERROR querying ChromaDB collection {chroma_collection.name}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to query content: {e}")

# --- Endpoint to generate answer using LLM ---
class GenerateAnswerRequest(BaseModel):
    query: str
    environmentId: str # New field to specify which environment to query
    n_results: int = 5 # Number of documents to retrieve for context

@app.post("/generate-answer")
async def generate_answer(request: GenerateAnswerRequest):
    """
    Receives a query and environmentId, retrieves relevant context from the
    specified ChromaDB collection, and generates an answer using the Gemini 2.0 Flash LLM.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")

    # Get the specific collection for this environment
    chroma_collection = get_chroma_collection(request.environmentId)

    print(f"Received request to generate answer for query: '{request.query}' from environment '{request.environmentId}'", file=sys.stderr)

    try:
        # 1. Retrieve relevant context from ChromaDB
        query_embedding = await asyncio.to_thread(embedding_model.encode, request.query, convert_to_tensor=False)
        query_embedding_list = query_embedding.tolist()

        retrieval_results = chroma_collection.query(
            query_embeddings=[query_embedding_list],
            n_results=request.n_results,
            include=['documents', 'metadatas', 'distances']
        )

        # Format retrieved context for sending back to Next.js and for LLM prompt
        retrieved_context_formatted = []
        context_documents_for_llm = []
        if retrieval_results and retrieval_results['documents'] and retrieval_results['documents'][0]:
            for i, doc_content in enumerate(retrieval_results['documents'][0]):
                metadata = retrieval_results['metadatas'][0][i]
                distance = retrieval_results['distances'][0][i]
                
                # Apply distance threshold filter for LLM context
                if distance <= MAX_DISTANCE_THRESHOLD:
                    retrieved_context_formatted.append({
                        "content": doc_content,
                        "metadata": metadata,
                        "distance": distance
                    })

                    # Modified to exclude 'Path' and focus on URL and content
                    context_documents_for_llm.append(
                        f"--- Document Title: {metadata.get('name', 'unknown')} --- "
                        f"URL: {metadata.get('url', 'unknown')}\n"
                        f"{doc_content}\n"
                    )
                else:
                    print(f"DEBUG: Skipping LLM context document due to high distance ({distance}): {metadata.get('path', 'N/A')}", file=sys.stderr)
        
        context_str = "\n\n".join(context_documents_for_llm)
        print(f"Retrieved {len(context_documents_for_llm)} documents for context (after distance filter).", file=sys.stderr)

        # 2. Construct prompt for LLM
        if not context_str:
            llm_prompt = (
                f"RAG: No related results found. Based on the query: '{request.query}', "
                f"please provide a general knowledge answer if possible. If you cannot provide a general knowledge answer, "
                f"simply state that you cannot answer the question.\n\n"
                f"Query: {request.query}"
            )
        else:
            llm_prompt = (
                f"You are a helpful assistant. Use the following retrieved information to answer the question. "
                f"If the answer is not available in the provided context, state that you cannot answer based on the given information. "
                f"For each piece of information you use, please cite the 'Document Title' and 'URL' from which it came.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {request.query}\n\n"
                f"Answer:"
            )

        # --- DEBUGGING: Print the full prompt sent to LLM ---
        print("\n--- Full LLM Prompt Sent to Gemini ---", file=sys.stderr)
        print(llm_prompt, file=sys.stderr)
        print("--- End LLM Prompt ---\n", file=sys.stderr)
        # --- END DEBUGGING ---

        print("Sending prompt to Gemini LLM...", file=sys.stderr)
        # 3. Call Gemini API
        async with httpx.AsyncClient() as client:
            payload = {
                "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 1024,
                }
            }
            headers = {
                "Content-Type": "application/json"
            }
            gemini_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

            response = await client.post(gemini_url_with_key, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            gemini_result = response.json()
            
            generated_text = "No answer generated."
            if gemini_result and gemini_result.get('candidates') and gemini_result['candidates'][0].get('content') and gemini_result['candidates'][0]['content'].get('parts'):
                generated_text = gemini_result['candidates'][0]['content']['parts'][0].get('text', "No text part found in Gemini response.")
            
            print("Gemini LLM response received.", file=sys.stderr)
            return {"status": "success", "answer": generated_text, "retrieved_context": retrieved_context_formatted}

    except httpx.RequestError as exc:
        print(f"ERROR: Network or request error during Gemini API call: {exc}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Network error with LLM: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"ERROR: HTTP error from Gemini API ({exc.response.status_code}): {exc.response.text}", file=sys.stderr)
        raise HTTPException(status_code=exc.response.status_code, detail=f"LLM API error: {exc.response.text}")
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON response from Gemini API: {exc}. Response: {response.text}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Invalid LLM response: {exc}")
    except Exception as e:
        print(f"ERROR generating answer: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")


# --- CORS Configuration (Crucial for frontend calls) ---
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware to allow your Next.js frontend to call this service
# Adjust origins to your Next.js development server URL (e.g., http://localhost:3000)
origins = [
    "http://localhost",
    "http://localhost:3000", # Your Next.js dev server
    "http://127.0.0.1:3000",
    # Add other origins if your Next.js app runs on a different host/port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (POST, GET, etc.)
    allow_headers=["*"], # Allow all headers
)

# To run this server:
# 1. Install dependencies: pip install fastapi uvicorn "chromadb[uvicorn]" sentence-transformers httpx
# 2. Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
#    (Using port 8001 to avoid conflict with Next.js dev server on 3000)
