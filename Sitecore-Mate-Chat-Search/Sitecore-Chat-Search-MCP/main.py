# Sitecore-Chat-Search-MCP/main.py

import os
import hashlib
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Keep Field if needed for other aliases, otherwise remove
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
# Import the dotenv library
from dotenv import load_dotenv
import asyncio # Import asyncio for the queue
from fastapi.responses import StreamingResponse # Import StreamingResponse

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Get the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available and configure Gemini
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or .env file is missing.")
genai.configure(api_key=GEMINI_API_KEY)


# --- Models ---
# Initialize the sentence transformer model for creating embeddings
# This model is optimized for semantic search.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
# This will create a persistent database in the './chroma_db' directory
client = chromadb.PersistentClient(path="./chroma_db")

# --- Global Log Queue for SSE ---
# This queue will hold log messages to be streamed to the frontend.
log_queue = asyncio.Queue()

# --- Pydantic Models for API Payload ---

# Represents a single field from Sitecore (e.g., 'Title', 'Body')
class Field(BaseModel):
    fieldName: str
    fieldValue: str
    componentId: Optional[str] = None # Added to receive component ID from frontend

# Represents a Sitecore component/rendering
class Component(BaseModel):
    componentId: str
    componentName: str
    fields: List[Field]

# Represents a Sitecore page
class Page(BaseModel):
    pageId: str
    pagePath: str
    pageTitle: str
    language: str
    fields: List[Field] # This will now contain both page's own fields and component fields
    components: List[Component] # This array will now always be empty
    itemType: str = "page" 

# This is the main payload the frontend will send for indexing
class ContentPayload(BaseModel):
    pages: List[Page]
    environment: str # e.g., 'production', 'staging'

# Payload for querying the indexed content
class QueryPayload(BaseModel):
    query: str
    environment: str

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Helper Functions for Chunking & Indexing ---

def generate_deterministic_id(page_id: str, field_name: str, chunk_index: int, component_id: Optional[str] = None) -> str:
    """Creates a unique and deterministic ID for a chunk."""
    base_string = f"{page_id}-{component_id or ''}-{field_name}-{chunk_index}"
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initializes and returns a text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Corresponds to ~250 tokens, a good size for context
        chunk_overlap=100, # Provides context overlap between chunks
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
                # Wait for a log message to be put into the queue
                message = await log_queue.get()
                yield f"data: {message}\n\n"
            except asyncio.CancelledError:
                # This exception is raised when the client disconnects
                break
            except Exception as e:
                print(f"Error in log stream event_generator: {e}")
                yield f"data: Error: {e}\n\n"
                # Optionally, re-raise the exception or break if a critical error occurs

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"message": "Sitecore Chat Search MCP is running"}

@app.post("/index-content")
async def index_content(payload: ContentPayload): # Changed to async
    """
    Receives structured content from Sitecore, chunks it, and indexes it in ChromaDB.
    This endpoint implements the logic from our agreed plan.
    """
    try:
        # Get or create a collection in ChromaDB for the specified environment
        collection = client.get_or_create_collection(name=payload.environment)
        text_splitter = get_text_splitter()

        all_chunks = []
        all_metadatas = []
        all_ids = []

        await log_queue.put(f"--- Starting indexing for environment: {payload.environment} ---")
        await log_queue.put(f"Total pages to process: {len(payload.pages)}")

        # Process each page from the payload
        for page_idx, page in enumerate(payload.pages):
            # Determine if it's a "page" or "component page" based on itemType
            # Now, page.itemType should accurately reflect if it's a primary content page or a component data source
            page_type_label = "page"
            if page.itemType and page.itemType.lower() == "component": # Access page.itemType
                page_type_label = "component page"
            
            await log_queue.put(f"Processing {page_type_label} {page_idx + 1}/{len(payload.pages)}: {page.pageTitle} (ID: {page.pageId})")
            
            # Process ALL fields (both page's own and flattened component fields)
            for field_idx, field in enumerate(page.fields):
                field_log_label = "page field"
                if field.fieldName in ["__Renderings", "__Final Renderings"]:
                    field_log_label = "rendering field"
                elif field.componentId: # If field has a componentId, it's from a component data source
                    field_log_label = "component field (flattened)"
                
                # Added page.pageId to the log message
                await log_queue.put(f"  - Processing {field_log_label} {field_idx + 1}: {field.fieldName} (Page ID: {page.pageId})")
                
                # Clean HTML from rich text fields
                soup = BeautifulSoup(field.fieldValue, "html.parser")
                text = soup.get_text(separator=" ", strip=True)

                if not text:
                    await log_queue.put(f"    (Skipping empty {field_log_label}: {field.fieldName} for Page ID: {page.pageId})")
                    continue

                # Split the cleaned text into chunks
                chunks = text_splitter.split_text(text)
                await log_queue.put(f"    Split into {len(chunks)} chunks.")
                for i, chunk_text in enumerate(chunks):
                    chunk_id = generate_deterministic_id(page.pageId, field.fieldName, i, field.componentId) # Pass field.componentId
                    metadata = {
                        "page_id": page.pageId,
                        "page_path": page.pagePath,
                        "page_title": page.pageTitle,
                        "component_id": field.componentId or "", # Use field.componentId or empty string
                        "field_name": field.fieldName,
                        "chunk_index": i,
                        "language": page.language,
                        "created_at": datetime.datetime.utcnow().isoformat()
                    }
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

            # The 'components' array in Page is now expected to be empty, as fields are flattened.
            # So, the original loop for page.components is no longer needed here for processing.
            # It was primarily for logging component names and IDs, which are now handled by field.componentId.
            # If you still need to log component names/IDs without processing their fields here,
            # you would need a separate mechanism or modify the frontend to send a list of component metadata.
            # For now, this part is effectively removed as all relevant content is in page.fields.

        # If there are chunks to add, embed and store them in ChromaDB
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
        # Log the error and return a meaningful response
        await log_queue.put(f"Error indexing content: {e}")
        print(f"Error indexing content: {e}") # Keep for server-side debugging
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-content")
def query_content(payload: QueryPayload):
    """
    Receives a query, embeds it, and performs a similarity search in ChromaDB.
    """
    try:
        collection = client.get_collection(name=payload.environment)
        
        # Create an embedding for the user's query
        query_embedding = model.encode([payload.query]).tolist()
        
        # Query the collection to find the 5 most relevant chunks
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )
        return results
    except Exception as e:
        print(f"Error querying content: {e}")
        raise HTTPException(status_code=500, detail=f"Could not query environment '{payload.environment}'. It might not exist or an error occurred.")


@app.post("/generate-answer")
async def generate_answer(payload: QueryPayload):
    """
    Generates a conversational answer using a RAG (Retrieval-Augmented Generation) approach.
    """
    try:
        # 1. Retrieve Context (same as /query-content)
        collection = client.get_collection(name=payload.environment)
        query_embedding = model.encode([payload.query]).tolist()
        context_results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        # Check if we got any documents back
        if not context_results or not context_results.get('documents'):
            return {"answer": "I could not find any relevant information to answer your question.", "sources": []}

        # 2. Augment the Prompt
        # Combine the retrieved documents into a single context string
        context = "\n".join(context_results['documents'][0])
        
        # Create a prompt for the Gemini model
        prompt = f"""
        You are a helpful assistant for a website.
        Based on the following context, please answer the user's question.
        If the context does not contain the answer, say that you don't know.

        Context:
        ---
        {context}
        ---

        User Question: {payload.query}

        Answer:
        """

        # 3. Generate the Answer
        llm = genai.GenerativeModel('gemini-pro')
        response = await llm.generate_content_async(prompt)
        
        # Extract unique sources from the metadata
        sources = []
        if context_results.get('metadatas'):
            seen_paths = set()
            for meta in context_results['metadatas'][0]:
                if meta['page_path'] not in seen_paths:
                    sources.append({
                        "title": meta['page_title'],
                        "path": meta['page_path']
                    })
                    seen_paths.add(meta['page_path'])

        return {"answer": response.text, "sources": sources}

    except Exception as e:
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))
