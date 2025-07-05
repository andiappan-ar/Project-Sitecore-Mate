# src/python_server/main.py

import os
import json
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Set
import asyncio 
import httpx 
import xml.etree.ElementTree as ET

# Import ChromaDB
import chromadb
from chromadb import Collection 

# Import SentenceTransformer for actual embeddings
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), ".chroma_db_data")
CHROMA_COLLECTION_BASE_NAME = "sitecore_content"
GEMINI_API_KEY = "" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
MAX_DISTANCE_THRESHOLD = 1.5 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ChromaDB Indexing Service",
    description="A service to index Sitecore content, including component datasources.",
    version="1.1.0"
)

# --- ChromaDB Client and Embedding Model Initialization ---
chroma_client: Optional[chromadb.PersistentClient] = None
embedding_model: Optional[SentenceTransformer] = None

def initialize_chromadb_and_model():
    """Initializes the ChromaDB client and embedding model."""
    global chroma_client, embedding_model
    if chroma_client and embedding_model:
        return
    try:
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("ChromaDB Client and SentenceTransformer model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to initialize services: {e}", file=sys.stderr)
        raise

def get_chroma_collection(environment_id: str) -> Collection:
    """Gets or creates a ChromaDB collection for a given environment ID."""
    if not chroma_client:
        raise HTTPException(status_code=500, detail="ChromaDB Client not initialized.")
    collection_name = f"{CHROMA_COLLECTION_BASE_NAME}_{environment_id}"
    return chroma_client.get_or_create_collection(name=collection_name)

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(initialize_chromadb_and_model)

# --- Pydantic Models ---
class IndexRequestItem(BaseModel):
    id: str
    name: str
    path: str
    url: str
    language: str
    content: str
    environmentId: str
    graphql_endpoint: str
    api_key: str
    sharedLayout: Optional[Dict[str, Any]] = None
    finalLayout: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str
    environmentId: str
    n_results: int = 5

class GenerateAnswerRequest(BaseModel):
    query: str
    environmentId: str
    n_results: int = 5

# --- Helper Functions ---

def _extract_datasource_ids_from_layout(layout_xml: str) -> Set[str]:
    """Parses layout XML and extracts a unique set of datasource IDs."""
    if not layout_xml:
        return set()
    try:
        sanitized_xml = layout_xml.replace('&', '&amp;')
        root = ET.fromstring(sanitized_xml)
        namespaces = {'s': 's'} 
        datasource_ids = {
            r.get('{s}ds') for r in root.findall('.//r[@s:ds]', namespaces=namespaces) if r.get('{s}ds')
        }
        return datasource_ids
    except ET.ParseError as e:
        print(f"Warning: Could not parse layout XML. Error: {e}", file=sys.stderr)
        return set()

async def fetch_datasource_content(datasource_id: str, language: str, graphql_endpoint: str, api_key: str) -> str:
    """Fetches and concatenates all text fields from a given datasource item."""
    query = """
    query GetDataSourceContent($id: String!, $language: String!) {
      item(path: $id, language: $language) {
        ownFields: fields(ownFields: false, excludeStandardFields: true) {
          name
          value
        }
      }
    }
    """
    try:
        # FIX: Added verify=False to disable SSL certificate verification for local development.
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                graphql_endpoint,
                headers={"Content-Type": "application/json", "sc_apikey": api_key},
                json={"query": query, "variables": {"id": datasource_id, "language": language}},
                timeout=20.0
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("data") and data["data"].get("item"):
                fields = data["data"]["item"].get("ownFields", [])
                return "\n".join([f"{field['name']}: {field['value']}" for field in fields if field.get('value')])
            return ""
    except Exception as e:
        print(f"Error fetching datasource {datasource_id}: {e}", file=sys.stderr)
        return ""

# --- API Endpoints ---

@app.post("/index-content")
async def index_content(item: IndexRequestItem):
    """Receives content, fetches unique datasource content from layouts, and indexes it."""
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")
    
    collection = get_chroma_collection(item.environmentId)
    print(f"Aggregating content for item: {item.path}", file=sys.stderr)

    shared_ds_ids = _extract_datasource_ids_from_layout(item.sharedLayout.get("value") if item.sharedLayout else "")
    final_ds_ids = _extract_datasource_ids_from_layout(item.finalLayout.get("value") if item.finalLayout else "")
    
    unique_datasource_ids = shared_ds_ids.union(final_ds_ids)
    
    print(f"Found {len(unique_datasource_ids)} unique datasources for this page.", file=sys.stderr)

    tasks = [
        fetch_datasource_content(ds_id, item.language, item.graphql_endpoint, item.api_key) 
        for ds_id in unique_datasource_ids
    ]
    datasource_contents = await asyncio.gather(*tasks)
    
    aggregated_content = item.content
    if datasource_contents:
        valid_contents = filter(None, datasource_contents)
        aggregated_content += "\n\n--- Components Content ---\n" + "\n\n".join(valid_contents)

    try:
        embedding = await asyncio.to_thread(embedding_model.encode, aggregated_content)
        metadata = {"name": item.name, "path": item.path, "url": item.url, "language": item.language}
        
        collection.add(
            documents=[aggregated_content],
            metadatas=[metadata],
            ids=[f"{item.id}-{item.language}"],
            embeddings=[embedding.tolist()]
        )
        return {"status": "success", "message": f"Item {item.id} indexed with aggregated content."}
    except Exception as e:
        print(f"ERROR indexing item {item.id}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-content")
async def query_content(request: QueryRequest):
    """Performs a vector similarity search."""
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")
    collection = get_chroma_collection(request.environmentId)
    try:
        query_embedding = await asyncio.to_thread(embedding_model.encode, request.query)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=request.n_results,
            include=['documents', 'metadatas', 'distances']
        )
        formatted_results = [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
            if dist <= MAX_DISTANCE_THRESHOLD
        ]
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-answer")
async def generate_answer(request: GenerateAnswerRequest):
    """Generates an answer using RAG."""
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding Model not initialized.")
    collection = get_chroma_collection(request.environmentId)
    try:
        query_embedding = await asyncio.to_thread(embedding_model.encode, request.query)
        retrieval_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=request.n_results,
            include=['documents', 'metadatas', 'distances']
        )
        context_documents = [
            {"content": doc, "metadata": meta}
            for doc, meta, dist in zip(retrieval_results['documents'][0], retrieval_results['metadatas'][0], retrieval_results['distances'][0])
            if dist <= MAX_DISTANCE_THRESHOLD
        ]
        if not context_documents:
            return {"status": "success", "answer": "I could not find any relevant information to answer your question.", "context": []}
        
        context_str = "\n\n".join([f"--- Document URL: {doc['metadata'].get('url', 'N/A')} ---\n{doc['content']}" for doc in context_documents])
        llm_prompt = (
            "You are an assistant that answers questions based *only* on the provided context. "
            "Do not use any external knowledge. If the answer is not in the context, "
            "state that you cannot answer the question based on the provided information. "
            "After your answer, list the URLs of the documents you used as references.\n\n"
            f"--- CONTEXT ---\n{context_str}\n\n"
            f"--- QUESTION ---\n{request.query}\n\n"
            "--- ANSWER ---"
        )
        
        async with httpx.AsyncClient() as client:
            payload = {"contents": [{"parts": [{"text": llm_prompt}]}]}
            headers = {"Content-Type": "application/json"}
            gemini_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            response = await client.post(gemini_url_with_key, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            gemini_result = response.json()
            generated_text = gemini_result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Could not extract text from LLM response.")
            return {"status": "success", "answer": generated_text, "context": context_documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

# --- CORS Configuration ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
