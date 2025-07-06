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
from dotenv import load_dotenv
import asyncio
from fastapi.responses import StreamingResponse
from collections import defaultdict # Import defaultdict

# Import prompts
from prompts import RAG_PROMPT_TEMPLATE, PAGE_SUMMARY_PROMPT_TEMPLATE, COMPONENT_SUMMARY_PROMPT_TEMPLATE

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# New: Load chunking and query parameters from environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
N_RESULTS = int(os.getenv("N_RESULTS", 8))
ADVANCED_RAG_ENABLED = os.getenv("ADVANCED_RAG_ENABLED", "false").lower() == "true"
print(f"DEBUG: ADVANCED_RAG_ENABLED is set to: {ADVANCED_RAG_ENABLED}") # DEBUG line

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
    sanitized = name.replace(r'\s+', '-')
    sanitized = re.sub(r'[^a-z0-9-]', '', sanitized.lower())
    sanitized = re.sub(r'^[^a-z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-z0-9]+$', '', sanitized)
    sanitized = re.sub(r'-+', '-', sanitized)

    if len(sanitized) < 3:
        sanitized = sanitized + 'id'
    
    sanitized = sanitized[:63]
    return sanitized


def generate_deterministic_id(page_id: str, field_name: str, chunk_index: int, component_id: Optional[str] = None) -> str:
    """Creates a unique and deterministic ID for a chunk."""
    base_string = f"{page_id}-{component_id or ''}-{field_name}-{chunk_index}"
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Initializes and returns a text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

# --- Dynamic Prompt Functions (using templates from prompts.py) ---
def build_page_summary_prompt(page_fields: dict) -> str:
    """
    Given a dict of {field_name: field_value}, build a dynamic LLM prompt for summarization.
    """
    field_lines = "\n".join([f"{k}: {v}" for k, v in page_fields.items()])
    return PAGE_SUMMARY_PROMPT_TEMPLATE.format(field_lines=field_lines)

def build_component_summary_prompt(component_name: str, component_fields: dict) -> str:
    """
    Given component name and a dict of {field_name: field_value}, build a dynamic LLM prompt.
    """
    field_lines = "\n".join([f"{k}: {v}" for k, v in component_fields.items()])
    return COMPONENT_SUMMARY_PROMPT_TEMPLATE.format(component_name=component_name, field_lines=field_lines)

async def enhance_text_with_llm(prompt: str, log_label: str) -> str:
    """
    Uses an LLM to enhance/summarize text based on the provided prompt.
    """
    try:
        llm = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = await llm.generate_content_async(prompt)
        await log_queue.put(f"    (LLM enhanced {log_label}.)")
        return response.text
    except Exception as e:
        await log_queue.put(f"    (Error enhancing {log_label} with LLM: {e}. Using original text or skipping.)")
        print(f"Error enhancing {log_label} with LLM: {e}")
        return "" # Return empty string if LLM fails, leading to skipping this enhanced text

async def advanced_rag_indexing(payload: ContentPayload, collection, text_splitter):
    """
    Handles advanced RAG indexing, enhancing content with LLM summaries.
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []

    await log_queue.put(f"--- Starting Advanced RAG indexing for environment: {payload.environment} ---")
    await log_queue.put(f"Total pages to process: {len(payload.pages)}")


    for page_idx, page in enumerate(payload.pages):
        page_type_label = "page"
        if page.itemType and page.itemType.lower() == "component":
            page_type_label = "component page"
        
        await log_queue.put(f"Processing {page_type_label} {page_idx + 1}/{len(payload.pages)}: {page.pageTitle} (ID: {page.pageId})")

        # Process Page-Level Fields for summary
        # Includes fields where componentId is None OR componentId is equal to pageId,
        # and excludes rendering fields.
        page_field_values_for_summary = {}
        for field in page.fields:
            if field.fieldName in ["__Renderings", "__Final Renderings"]:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: it's a rendering field.)")
                continue
            
            if not field.fieldValue:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: field value is empty.)")
                continue

            # This is the core logic for page-level summary eligibility
            if field.componentId is None or field.componentId == page.pageId:
                page_field_values_for_summary[field.fieldName] = BeautifulSoup(field.fieldValue, "html.parser").get_text(separator=" ", strip=True)
            else:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: componentId '{field.componentId}' does not match pageId '{page.pageId}' and is not None.)")

        
        if page_field_values_for_summary:
            await log_queue.put(f"    Raw page fields for summary: {page_field_values_for_summary}")
            page_summary_prompt = build_page_summary_prompt(page_field_values_for_summary)
            await log_queue.put(f"    Generated page summary prompt:\n{page_summary_prompt[:500]}...") # Truncate for log readability

            enhanced_page_text = await enhance_text_with_llm(page_summary_prompt, f"page '{page.pageTitle}' fields")
            if enhanced_page_text:
                await log_queue.put(f"    LLM-generated page summary: {enhanced_page_text[:500]}...") # Truncate for log readability
                page_chunks = text_splitter.split_text(enhanced_page_text)
                await log_queue.put(f"    Page-level fields (enhanced summary) split into {len(page_chunks)} chunks.")
                for i, chunk_text in enumerate(page_chunks):
                    chunk_id = generate_deterministic_id(page.pageId, "page_summary", i)
                    metadata = {
                        "page_id": page.pageId,
                        "page_path": page.pagePath,
                        "page_title": page.pageTitle,
                        "page_url": page.url,
                        "component_id": "", # Page summaries don't belong to a specific component ID
                        "field_name": "page_summary",
                        "chunk_index": i,
                        "language": page.language,
                        "created_at": datetime.datetime.utcnow().isoformat(),
                        "source_type": "page_summary"
                    }
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)
        else:
            await log_queue.put(f"    (No significant page-level fields found for summary for page '{page.pageTitle}')")


        # --- NEW LOGIC FOR VIRTUAL COMPONENTS IF page.components IS EMPTY ---
        if not page.components:
            await log_queue.put(f"    (No explicit components found in payload for page '{page.pageTitle}'. Checking for virtual components from page fields based on fieldName prefix.)")
            
            # --- START NEW LOGGING HERE ---
            await log_queue.put(f"    All fields for page '{page.pageTitle}' before virtual component grouping:")
            for field in page.fields:
                field_val_preview = (field.fieldValue[:100] + '...') if len(field.fieldValue) > 100 else field.fieldValue
                await log_queue.put(f"      - fieldName: '{field.fieldName}', componentId: '{field.componentId}', fieldValue: '{field_val_preview}'")
            # --- END NEW LOGGING HERE ---

            # Virtual grouping by component name prefix in fieldName, excluding 'Page.AllFields'
            virtual_components = defaultdict(list)
            for field in page.fields:
                if "." in field.fieldName:
                    comp_prefix = field.fieldName.split(".")[0]
                    if comp_prefix.lower() != "page": # Exclude "Page.AllFields" and similar
                        virtual_components[comp_prefix].append(field)

            if not virtual_components: # Add a log if no virtual components were found after this logic
                await log_queue.put(f"    (No virtual components identified from fieldName prefixes for page '{page.pageTitle}'.)")

            for comp_name, fields_list in virtual_components.items(): # Changed comp_id to comp_name to match the key
                # Use the comp_name directly for component name
                # For unique ID, use componentId if available from a field, or create a hash from comp_name
                # If your fields consistently have a componentId that refers to the instance of the component, use that.
                # If they don't, you might need to synthesize one.
                # Given your payload, field.componentId currently matches page.pageId, so we'll use a synthesized one.
                virtual_comp_instance_id = hashlib.sha256(f"{page.pageId}-{comp_name}".encode('utf-8')).hexdigest()[:16] # Generate a unique ID for virtual component instance
                
                await log_queue.put(f"  - Processing Virtual Component: {comp_name} (Generated ID: {virtual_comp_instance_id}) on Page ID: {page.pageId}")

                component_field_values_for_summary = {}
                for field in fields_list: # Iterate through fields in this virtual component
                    if field.fieldName in ["__Renderings", "__Final Renderings"]:
                        await log_queue.put(f"    (Skipping virtual component field '{field.fieldName}' for summary: it's a rendering field.)")
                        continue
                    if not field.fieldValue:
                        await log_queue.put(f"    (Skipping virtual component field '{field.fieldName}' for summary: field value is empty.)")
                        continue
                    component_field_values_for_summary[field.fieldName] = BeautifulSoup(field.fieldValue, "html.parser").get_text(separator=" ", strip=True)

                if component_field_values_for_summary:
                    await log_queue.put(f"    Raw virtual component fields for summary ('{comp_name}'): {component_field_values_for_summary}")
                    component_summary_prompt = build_component_summary_prompt(comp_name, component_field_values_for_summary)
                    await log_queue.put(f"    Generated virtual component summary prompt:\n{component_summary_prompt[:500]}...")
                    enhanced_component_text = await enhance_text_with_llm(component_summary_prompt, f"virtual component '{comp_name}' fields")
                    if enhanced_component_text:
                        await log_queue.put(f"    LLM-generated virtual component summary: {enhanced_component_text[:500]}...")
                        component_chunks = text_splitter.split_text(enhanced_component_text)
                        await log_queue.put(f"    Virtual Component fields (enhanced summary) split into {len(component_chunks)} chunks.")
                        for i, chunk_text in enumerate(component_chunks):
                            # Use the generated virtual_comp_instance_id
                            chunk_id = generate_deterministic_id(page.pageId, f"component_summary_{comp_name}", i, virtual_comp_instance_id)
                            metadata = {
                                "page_id": page.pageId,
                                "page_path": page.pagePath,
                                "page_title": page.pageTitle,
                                "page_url": page.url,
                                "component_id": virtual_comp_instance_id, # Use generated ID
                                "field_name": f"component_summary_{comp_name}",
                                "chunk_index": i,
                                "language": page.language,
                                "created_at": datetime.datetime.utcnow().isoformat(),
                                "source_type": "component_summary_virtual" # Differentiate source
                            }
                            all_chunks.append(chunk_text)
                            all_metadatas.append(metadata)
                            all_ids.append(chunk_id)
                else:
                    await log_queue.put(f"    (No significant fields found for summary in virtual component '{comp_name}')")
        # --- END NEW LOGIC FOR VIRTUAL COMPONENTS ---


        # Original processing for explicit components (this block will run if page.components is NOT empty)
        # Note: This loop will be skipped if page.components is empty, so no explicit component logs
        # will appear in that case.
        for comp_idx, component in enumerate(page.components):
            await log_queue.put(f"  - Processing Explicit Component {comp_idx + 1}: {component.componentName} (ID: {component.componentId}) on Page ID: {page.pageId}")
            
            component_field_values_for_summary = {}
            for field in component.fields:
                if field.fieldName in ["__Renderings", "__Final Renderings"]:
                    await log_queue.put(f"    (Skipping explicit component field '{field.fieldName}' for summary: it's a rendering field.)")
                    continue
                if not field.fieldValue:
                    await log_queue.put(f"    (Skipping explicit component field '{field.fieldName}' for summary: field value is empty.)")
                    continue
                component_field_values_for_summary[field.fieldName] = BeautifulSoup(field.fieldValue, "html.parser").get_text(separator=" ", strip=True)

            if component_field_values_for_summary:
                await log_queue.put(f"    Raw explicit component fields for summary ('{component.componentName}'): {component_field_values_for_summary}")
                component_summary_prompt = build_component_summary_prompt(component.componentName, component_field_values_for_summary)
                await log_queue.put(f"    Generated explicit component summary prompt:\n{component_summary_prompt[:500]}...") # Truncate for log readability

                enhanced_component_text = await enhance_text_with_llm(component_summary_prompt, f"explicit component '{component.componentName}' fields")
                if enhanced_component_text:
                    await log_queue.put(f"    LLM-generated explicit component summary: {enhanced_component_text[:500]}...")
                    component_chunks = text_splitter.split_text(enhanced_component_text)
                    await log_queue.put(f"    Explicit Component fields (enhanced summary) split into {len(component_chunks)} chunks.")
                    for i, chunk_text in enumerate(component_chunks):
                        chunk_id = generate_deterministic_id(page.pageId, f"component_summary_{component.componentName}", i, component.componentId)
                        metadata = {
                            "page_id": page.pageId,
                            "page_path": page.pagePath,
                            "page_title": page.pageTitle,
                            "page_url": page.url,
                            "component_id": component.componentId,
                            "field_name": f"component_summary_{component.componentName}",
                            "chunk_index": i,
                            "language": page.language,
                            "created_at": datetime.datetime.utcnow().isoformat(),
                            "source_type": "component_summary"
                        }
                        all_chunks.append(chunk_text)
                        all_metadatas.append(metadata)
                        all_ids.append(chunk_id)
            else:
                await log_queue.put(f"    (No significant explicit component fields found for summary for component '{component.componentName}')")
        
        # Determine which fields have been summarized to avoid re-indexing them as individual fields
        summarized_field_names_on_page = set(page_field_values_for_summary.keys())
        
        summarized_component_field_ids = set()
        
        # Capture fields from explicit components that were summarized
        for comp in page.components:
            comp_fields_for_summary = {
                f.fieldName: BeautifulSoup(f.fieldValue, "html.parser").get_text(separator=" ", strip=True)
                for f in comp.fields
                if f.fieldValue and f.fieldName not in ["__Renderings", "__Final Renderings"]
            }
            if comp_fields_for_summary:
                for f in comp.fields:
                    if f.fieldValue and f.fieldName not in ["__Renderings", "__Final Renderings"]:
                        summarized_component_field_ids.add(f"{f.componentId}_{f.fieldName}")

        # Capture fields from virtual components that were summarized
        # This needs to run whether page.components was empty or not, to correctly track all summarized fields
        # Re-create virtual_components based on the same logic used for processing
        virtual_components_for_summary_tracking = defaultdict(list)
        for field in page.fields:
            if "." in field.fieldName:
                comp_prefix = field.fieldName.split(".")[0]
                if comp_prefix.lower() != "page":
                    virtual_components_for_summary_tracking[field.componentId if field.componentId else comp_prefix].append(field)
        
        for comp_key, fields_list in virtual_components_for_summary_tracking.items():
            comp_fields_for_summary = {
                f.fieldName: BeautifulSoup(f.fieldValue, "html.parser").get_text(separator=" ", strip=True)
                for f in fields_list
                if f.fieldValue and f.fieldName not in ["__Renderings", "__Final Renderings"]
            }
            if comp_fields_for_summary:
                # Use the actual componentId from the field if available, otherwise the derived comp_key
                for f in fields_list:
                    if f.fieldValue and f.fieldName not in ["__Renderings", "__Final Renderings"]:
                        # Correctly use f.componentId if present, falling back to a derived ID if needed
                        # The key in summarized_component_field_ids should match how you generated the chunk_id
                        # For virtual components, we used a generated ID earlier: virtual_comp_instance_id
                        # We need to ensure consistency. Let's use the field's componentId as the base for the key.
                        # If field.componentId is pageId for these, that's what will be used in the key.
                        summarized_component_field_ids.add(f"{f.componentId}_{f.fieldName}")


        for field_idx, field in enumerate(page.fields):
            # Skip rendering fields as they are usually technical content
            if field.fieldName in ["__Renderings", "__Final Renderings"]:
                await log_queue.put(f"    (Skipping rendering field: {field.fieldName} for Page ID: {page.pageId})")
                continue

            # Check if this field was covered by a page-level summary
            is_part_of_page_summary = (field.componentId is None or field.componentId == page.pageId) and \
                                     field.fieldName in summarized_field_names_on_page
            
            if is_part_of_page_summary:
                await log_queue.put(f"    (Skipping individual page field '{field.fieldName}' as it contributed to the page summary.)")
                continue

            # Check if this field was covered by a component summary (explicit or virtual)
            is_part_of_component_summary = field.componentId and \
                                           f"{field.componentId}_{field.fieldName}" in summarized_component_field_ids

            if is_part_of_component_summary:
                await log_queue.put(f"    (Skipping individual component field '{field.fieldName}' as its component '{field.componentId}' was summarized.)")
                continue

            # If not skipped, process as an individual field
            field_log_label = "individual page field"
            if field.componentId:
                field_log_label = "individual component field (not summarized)"
            
            await log_queue.put(f"  - Processing {field_log_label} {field_idx + 1}: {field.fieldName} (Page ID: {page.pageId})")
            
            soup = BeautifulSoup(field.fieldValue, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            if not text:
                await log_queue.put(f"    (Skipping empty {field_log_label}: {field.fieldName} for Page ID: {page.pageId})")
                continue

            chunks = text_splitter.split_text(text)
            await log_queue.put(f"    Split into {len(chunks)} chunks. Original text length: {len(text)}")
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
                    "created_at": datetime.datetime.utcnow().isoformat(),
                    "source_type": "individual_field"
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
        return {"status": "success", "message": f"Indexed {len(all_chunks)} chunks for environment '{payload.environment}' using Advanced RAG."}
    else:
        await log_queue.put("--- No new content to index after processing all pages ---")
        return {"status": "success", "message": "No new content to index."}


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

        if ADVANCED_RAG_ENABLED:
            await log_queue.put("--- Advanced RAG indexing is ENABLED. ---")
            return await advanced_rag_indexing(payload, collection, text_splitter)
        else:
            await log_queue.put("--- Advanced RAG indexing is DISABLED. Performing normal indexing. ---")
            all_chunks = []
            all_metadatas = []
            all_ids = []

            await log_queue.put(f"--- Starting normal indexing for environment: {payload.environment} (ChromaDB Collection: {sanitized_environment_name}) ---")
            await log_queue.put(f"Total pages to process: {len(payload.pages)}")

            for page_idx, page in enumerate(payload.pages):
                page_type_label = "page"
                if page.itemType and page.itemType.lower() == "component":
                    page_type_label = "component page"
                
                await log_queue.put(f"Processing {page_type_label} {page_idx + 1}/{len(payload.pages)}: {page.pageTitle} (ID: {page.pageId})")
                
                # Combine page-level fields and component fields for normal indexing
                all_fields_to_process = page.fields + [f for comp in page.components for f in comp.fields]
                
                for field_idx, field in enumerate(all_fields_to_process):
                    field_log_label = "page field"
                    if field.fieldName in ["__Renderings", "__Final Renderings"]:
                        field_log_label = "rendering field"
                        await log_queue.put(f"    (Skipping rendering field: {field.fieldName} for Page ID: {page.pageId})")
                        continue 
                    elif field.componentId:
                        field_log_label = "component field"
                    
                    await log_queue.put(f"  - Processing {field_log_label} {field_idx + 1}: {field.fieldName} (Page ID: {page.pageId})")
                    
                    soup = BeautifulSoup(field.fieldValue, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)

                    if not text:
                        await log_queue.put(f"    (Skipping empty {field_log_label}: {field.fieldName} for Page ID: {page.pageId})")
                        continue

                    chunks = text_splitter.split_text(text)
                    await log_queue.put(f"    Split into {len(chunks)} chunks. Original text length: {len(text)}")
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
                            "created_at": datetime.datetime.utcnow().isoformat(),
                            "source_type": "normal_field"
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
                return {"status": "success", "message": f"Indexed {len(all_chunks)} chunks for environment '{payload.environment}' using normal indexing."}
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
            n_results=N_RESULTS,
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
                        "sourceType": doc_metadata.get("source_type", "unknown")
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
            n_results=N_RESULTS,
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