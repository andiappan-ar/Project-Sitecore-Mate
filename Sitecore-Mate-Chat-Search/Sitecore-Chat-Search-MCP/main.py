# Sitecore-Chat-Search-MCP/main.py

import os
import hashlib
import datetime
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv # Re-import load_dotenv
import asyncio
from fastapi.responses import StreamingResponse

# Import prompts
from prompts import RAG_PROMPT_TEMPLATE

# --- Configuration ---
load_dotenv() # Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# New: Load chunking and query parameters from environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) # Default to 1000 if not set
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) # Default to 100 if not set
N_RESULTS = int(os.getenv("N_RESULTS", 8)) # Default to 8 if not set

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or .env file is missing.")
genai.configure(api_key=GEMINI_API_KEY)
print("Gemini API configured successfully.")

# --- Models ---
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")

# --- Global Log Queue for SSE ---
log_queue = asyncio.Queue()

# --- Pydantic Models for API Payload ---
class Field(BaseModel):
    fieldName: str
    fieldValue: str
    componentId: Optional[str] = None

class Component(BaseModel):
    componentId: str
    componentName: str
    fields: List[Field]

class Page(BaseModel):
    pageId: str
    pagePath: str
    pageTitle: str
    language: str
    fields: List[Field]
    components: List[Component]
    itemType: str = "page"
    url: Optional[str] = None

class ContentPayload(BaseModel):
    pages: List[Page]
    environment: str

class QueryPayload(BaseModel):
    query: str
    environment: str

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Helper Functions for Chunking & Indexing ---

def sanitize_chroma_db_name(name: str) -> str:
    """
    Sanitizes a string to be a valid ChromaDB collection name.
    Follows ChromaDB's naming rules:
    - Must be between 3 and 63 characters long.
    - Must start and end with a lowercase letter or a digit.
    - Must contain only lowercase letters, digits, or hyphens.
    - Must not contain two consecutive hyphens.
    """
    # Replace spaces with hyphens
    sanitized = name.replace(r'\s+', '-')
    # Remove any characters not allowed by ChromaDB (keep a-z, 0-9, -)
    sanitized = re.sub(r'[^a-z0-9-]', '', sanitized.lower())
    # Ensure it starts and ends with an alphanumeric character
    sanitized = re.sub(r'^[^a-z0-9]+', '', sanitized) # Remove non-alphanumeric from start
    sanitized = re.sub(r'[^a-z0-9]+$', '', sanitized) # Remove non-alphanumeric from end
    # Replace multiple hyphens with a single hyphen
    sanitized = re.sub(r'-+', '-', sanitized)

    # Ensure minimum length (ChromaDB requires at least 3 chars)
    if len(sanitized) < 3:
        # If too short after sanitization, append 'id' or a hash
        # For simplicity, we'll just append 'id' if it's too short.
        # In a real app, you might want a more robust unique suffix.
        sanitized = sanitized + 'id'
    
    # Trim to max length if necessary (ChromaDB max 63 chars)
    sanitized = sanitized[:63]

    return sanitized


def generate_deterministic_id(page_id: str, field_name: str, chunk_index: int, component_id: Optional[str] = None) -> str:
    """Creates a unique and deterministic ID for a chunk."""
    base_string = f"{page_id}-{component_id or ''}-{field_name}-{chunk_index}"
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initializes and returns a text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, # Use CHUNK_SIZE from environment
        chunk_overlap=CHUNK_OVERLAP, # Use CHUNK_OVERLAP from environment
        length_function=len,
        is_separator_regex=False,
    )

# --- SSE Log Stream Endpoint ---
@app.get("/log-stream")
async def log_stream():
    """
    Streams real-time logs to the client using Server-Sent Events (SSE).
    """
    async def event_generator():
        while True:
            try:
                message = await log_queue.get()
                yield f"data: {message}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in log stream event_generator: {e}")
                yield f"data: Error: {e}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"message": "Sitecore Chat Search MCP is running"}

@app.post("/index-content")
async def index_content(payload: ContentPayload):
    """
    Receives structured content from Sitecore, chunks it, and indexes it in ChromaDB.
    This endpoint implements the logic from our agreed plan.
    """
    try:
        print("\n--- Incoming Payload to Python Indexing Service ---")
        print(payload.model_dump_json(indent=2))
        print("---------------------------------------------------\n")

        # Sanitize the environment name for ChromaDB
        sanitized_environment_name = sanitize_chroma_db_name(payload.environment)
        await log_queue.put(f"Sanitized ChromaDB collection name: {sanitized_environment_name}")

        # Get or create a collection in ChromaDB using the sanitized name
        collection = client.get_or_create_collection(name=sanitized_environment_name)
        text_splitter = get_text_splitter()

        all_chunks = []
        all_metadatas = []
        all_ids = []

        await log_queue.put(f"--- Starting indexing for environment: {payload.environment} (ChromaDB Collection: {sanitized_environment_name}) ---")
        await log_queue.put(f"Total pages to process: {len(payload.pages)}")

        for page_idx, page in enumerate(payload.pages):
            page_type_label = "page"
            if page.itemType and page.itemType.lower() == "component":
                page_type_label = "component page"
            
            await log_queue.put(f"Processing {page_type_label} {page_idx + 1}/{len(payload.pages)}: {page.pageTitle} (ID: {page.pageId})")
            
            for field_idx, field in enumerate(page.fields):
                field_log_label = "page field"
                if field.fieldName in ["__Renderings", "__Final Renderings"]:
                    field_log_label = "rendering field"
                elif field.componentId:
                    field_log_label = "component field (flattened)"
                
                await log_queue.put(f"  - Processing {field_log_label} {field_idx + 1}: {field.fieldName} (Page ID: {page.pageId})")
                
                soup = BeautifulSoup(field.fieldValue, "html.parser")
                text = soup.get_text(separator=" ", strip=True)

                if not text:
                    await log_queue.put(f"    (Skipping empty {field_log_label}: {field.fieldName} for Page ID: {page.pageId})")
                    continue

                chunks = text_splitter.split_text(text)
                await log_queue.put(f"    Split into {len(chunks)} chunks.")
                for i, chunk_text in enumerate(chunks):
                    chunk_id = generate_deterministic_id(page.pageId, field.fieldName, i, field.componentId)
                    metadata = {
                        "page_id": page.pageId,
                        "page_path": page.pagePath,
                        "page_title": page.pageTitle,
                        "page_url": page.url,
                        "component_id": field.componentId or "",
                        "field_name": field.fieldName,
                        "chunk_index": i,
                        "language": page.language,
                        "created_at": datetime.datetime.utcnow().isoformat()
                    }
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

        if all_chunks:
            await log_queue.put(f"--- Embedding and adding {len(all_chunks)} total chunks to ChromaDB ---")
            embeddings = model.encode(all_chunks).tolist()
            collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            await log_queue.put("--- Finished adding chunks to ChromaDB ---")
            return {"status": "success", "message": f"Indexed {len(all_chunks)} chunks for environment '{payload.environment}'."}
        else:
            await log_queue.put("--- No new content to index after processing all pages ---")
            return {"status": "success", "message": "No new content to index."}

    except Exception as e:
        await log_queue.put(f"Error indexing content: {e}")
        print(f"Error indexing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-content")
def query_content(payload: QueryPayload):
    """
    Receives a query, embeds it, and performs a similarity search in ChromaDB.
    """
    try:
        # Sanitize the environment name for ChromaDB
        sanitized_environment_name = sanitize_chroma_db_name(payload.environment)
        collection = client.get_collection(name=sanitized_environment_name)
        
        query_embedding = model.encode([payload.query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS, # Use N_RESULTS from environment
            include=['documents', 'metadatas', 'distances'] 
        )
        
        print("\n--- Raw ChromaDB Query Results ---")
        print(results)
        print("----------------------------------\n")

        formatted_results = []
        if results and results.get('documents'):
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                doc_metadata = results['metadatas'][0][i]
                doc_distance = results['distances'][0][i]

                formatted_results.append({
                    "content": doc_content,
                    "metadata": {
                        "id": doc_metadata.get("page_id", ""),
                        "name": doc_metadata.get("page_title", ""),
                        "path": doc_metadata.get("page_path", ""),
                        "url": doc_metadata.get("page_url", ""),
                        "language": doc_metadata.get("language", ""),
                        "environmentId": payload.environment,
                        "componentId": doc_metadata.get("component_id", ""),
                        "fieldName": doc_metadata.get("field_name", ""),
                    },
                    "distance": doc_distance,
                })
        
        return {"status": "success", "results": formatted_results}
    except Exception as e:
        print(f"Error querying content: {e}")
        raise HTTPException(status_code=500, detail=f"Could not query environment '{payload.environment}'. It might not exist or an error occurred.")


@app.post("/generate-answer")
async def generate_answer(payload: QueryPayload):
    """
    Generates a conversational answer using a RAG (Retrieval-Augmented Generation) approach.
    """
    try:
        # Sanitize the environment name for ChromaDB
        sanitized_environment_name = sanitize_chroma_db_name(payload.environment)
        collection = client.get_collection(name=sanitized_environment_name)
        query_embedding = model.encode([payload.query]).tolist()
        context_results = collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS, # Use N_RESULTS from environment
            include=['documents', 'metadatas', 'distances'] 
        )

        if not context_results or not context_results.get('documents'):
            return {"answer": "I could not find any relevant information to answer your question.", "sources": []}

        context = "\n".join(context_results['documents'][0])
        
        # Use the imported prompt template
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=payload.query)

        llm = genai.GenerativeModel('gemini-2.0-flash')
        response = await llm.generate_content_async(prompt)
        
        sources = []
        if context_results.get('metadatas'):
            seen_urls = set()
            for meta in context_results['metadatas'][0]:
                if meta.get('page_url') and meta['page_url'] not in seen_urls:
                    sources.append({
                        "title": meta.get('page_title', ''),
                        "path": meta.get('page_path', ''),
                        "url": meta['page_url']
                    })
                    seen_urls.add(meta['page_url'])

        return {"answer": response.text, "sources": sources}

    except Exception as e:
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

