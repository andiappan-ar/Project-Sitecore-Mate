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
    renderingUid: Optional[str] = None  # Optional: to capture the rendering UID if needed

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

# New helper function to get the most reliable rendering UID
def get_effective_rendering_uid(field: Field) -> Optional[str]:
    """
    Determines the most appropriate rendering UID for a field.
    Prioritizes the field's 'renderinguid' property.
    If not available, attempts to extract it from 'fieldName' if it matches the expected format:
    'componentName.AllFields.itemId.dsId.renderingUid'.
    """
    if field.renderingUid:
        return field.renderingUid
    
    # Attempt to extract from fieldName: componentName.AllFields.itemId.dsId.renderingUid
    parts = field.fieldName.split('.')
    # Check for at least 5 parts and that the second part is 'AllFields' (case-insensitive)
    if len(parts) >= 5 and parts[1].lower() == 'allfields':
        potential_rendering_uid = parts[-1]
        # Basic check if it looks like a GUID (common for Sitecore rendering UIDs)
        if re.match(r'^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$', potential_rendering_uid):
            return potential_rendering_uid
        # If it's not a GUID but you still want to use it as a UID from fieldName, return it.
        # This makes it more flexible if renderingUids are not strictly GUIDs.
        # return potential_rendering_uid # Uncomment this if renderingUids are not always GUIDs
        
    return None


def generate_deterministic_id(page_id: str, field_name: str, chunk_index: int, component_id: Optional[str] = None, rendering_uid: Optional[str] = None) -> str:
    """Creates a unique and deterministic ID for a chunk, including rendering_uid if present."""
    # Ensure all components are strings for consistent hashing
    component_id_str = component_id if component_id is not None else ''
    rendering_uid_str = rendering_uid if rendering_uid is not None else ''

    base_string = f"{page_id}-{component_id_str}-{rendering_uid_str}-{field_name}-{chunk_index}"
    
    # --- ADD THIS DEBUG LOG ---
    print(f"DEBUG: Generating ID for base_string: '{base_string}'")
    # --- END DEBUG LOG ---

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
        page_field_values_for_summary = {}
        for field in page.fields:
            if field.fieldName in ["__Renderings", "__Final Renderings"]:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: it's a rendering field.)")
                continue
            
            if not field.fieldValue:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: field value is empty.)")
                continue

            if field.componentId is None or field.componentId == page.pageId:
                page_field_values_for_summary[field.fieldName] = BeautifulSoup(field.fieldValue, "html.parser").get_text(separator=" ", strip=True)
            else:
                await log_queue.put(f"    (Skipping field '{field.fieldName}' for page summary: componentId '{field.componentId}' does not match pageId '{page.pageId}' and is not None.)")

        
        if page_field_values_for_summary:
            await log_queue.put(f"    Raw page fields for summary: {page_field_values_for_summary}")
            page_summary_prompt = build_page_summary_prompt(page_field_values_for_summary)
            await log_queue.put(f"    Generated page summary prompt:\n{page_summary_prompt[:500]}...")
            enhanced_page_text = await enhance_text_with_llm(page_summary_prompt, f"page '{page.pageTitle}' fields")
            if enhanced_page_text:
                await log_queue.put(f"    LLM-generated page summary: {enhanced_page_text[:500]}...")
                page_chunks = text_splitter.split_text(enhanced_page_text)
                await log_queue.put(f"    Page-level fields (enhanced summary) split into {len(page_chunks)} chunks.")
                for i, chunk_text in enumerate(page_chunks):
                    # No renderingUid for page summary as it's not tied to a specific rendering instance
                    chunk_id = generate_deterministic_id(page.pageId, "page_summary", i)
                    metadata = {
                        "page_id": page.pageId,
                        "page_path": page.pagePath,
                        "page_title": page.pageTitle,
                        "page_url": page.url,
                        "component_id": "",
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
            
            await log_queue.put(f"    All fields for page '{page.pageTitle}' before virtual component grouping:")
            for field in page.fields:
                field_val_preview = (field.fieldValue[:100] + '...') if len(field.fieldValue) > 100 else field.fieldValue
                await log_queue.put(f"      - fieldName: '{field.fieldName}', componentId: '{field.componentId}', fieldValue: '{field_val_preview}', renderinguid (property): '{field.renderingUid}'")

            # Virtual grouping by (component name prefix, effective renderinguid or fallback)
            # This ensures that if the same component prefix appears multiple times (e.g., 'Hero Banner')
            # but is associated with different rendering UIDs (either from property or fieldName),
            # they are treated as distinct virtual component instances.
            virtual_components = defaultdict(list)
            for field in page.fields:
                if "." in field.fieldName:
                    comp_parts = field.fieldName.split(".")
                    comp_prefix = comp_parts[0]

                    if comp_prefix.lower() != "page": # Exclude "Page.AllFields" and similar
                        effective_rendering_uid_for_grouping = get_effective_rendering_uid(field)

                        # Use "NO_RENDER_UID" as a placeholder for the grouping key if no effective UID is found
                        grouping_key_render_uid = effective_rendering_uid_for_grouping if effective_rendering_uid_for_grouping else "NO_RENDER_UID"
                        grouping_key = (comp_prefix, grouping_key_render_uid)
                        virtual_components[grouping_key].append(field)

            if not virtual_components:
                await log_queue.put(f"    (No virtual components identified from fieldName prefixes for page '{page.pageTitle}'.)")

            for (comp_name_prefix, grouping_render_uid), fields_list in virtual_components.items():
                # The actual rendering UID used for the deterministic ID will be empty if "NO_RENDER_UID" was used for grouping
                actual_rendering_uid_for_id = grouping_render_uid if grouping_render_uid != "NO_RENDER_UID" else ''
                
                # Create a unique key for this virtual component instance that includes the rendering UID
                virtual_comp_instance_key = f"{comp_name_prefix}{'_' + actual_rendering_uid_for_id if actual_rendering_uid_for_id else ''}"

                # Generate a unique ID for this virtual component instance
                virtual_comp_instance_id = hashlib.sha256(f"{page.pageId}-{virtual_comp_instance_key}".encode('utf-8')).hexdigest()[:16]
                
                await log_queue.put(f"  - Processing Virtual Component: {comp_name_prefix} (Generated ID: {virtual_comp_instance_id}) on Page ID: {page.pageId}, Grouping Render UID: {actual_rendering_uid_for_id if actual_rendering_uid_for_id else 'None'}")

                component_field_values_for_summary = {}
                for field in fields_list:
                    if field.fieldName in ["__Renderings", "__Final Renderings"]:
                        await log_queue.put(f"    (Skipping virtual component field '{field.fieldName}' for summary: it's a rendering field.)")
                        continue
                    if not field.fieldValue:
                        await log_queue.put(f"    (Skipping virtual component field '{field.fieldName}' for summary: field value is empty.)")
                        continue
                    component_field_values_for_summary[field.fieldName] = BeautifulSoup(field.fieldValue, "html.parser").get_text(separator=" ", strip=True)

                if component_field_values_for_summary:
                    await log_queue.put(f"    Raw virtual component fields for summary ('{comp_name_prefix}'): {component_field_values_for_summary}")
                    component_summary_prompt = build_component_summary_prompt(comp_name_prefix, component_field_values_for_summary)
                    await log_queue.put(f"    Generated virtual component summary prompt:\n{component_summary_prompt[:500]}...")

                    enhanced_component_text = await enhance_text_with_llm(component_summary_prompt, f"virtual component '{comp_name_prefix}' fields")
                    if enhanced_component_text:
                        await log_queue.put(f"    LLM-generated virtual component summary: {enhanced_component_text[:500]}...")
                        component_chunks = text_splitter.split_text(enhanced_component_text)
                        await log_queue.put(f"    Virtual Component fields (enhanced summary) split into {len(component_chunks)} chunks.")
                        for i, chunk_text in enumerate(component_chunks):
                            # The field_name for the summary itself should reflect its uniqueness
                            summary_field_name = f"component_summary_{virtual_comp_instance_key}"
                            chunk_id = generate_deterministic_id(page.pageId, summary_field_name, i, virtual_comp_instance_id, actual_rendering_uid_for_id)
                            metadata = {
                                "page_id": page.pageId,
                                "page_path": page.pagePath,
                                "page_title": page.pageTitle,
                                "page_url": page.url,
                                "component_id": virtual_comp_instance_id, # Use generated ID for metadata
                                "field_name": summary_field_name, # Use the more specific summary field name
                                "chunk_index": i,
                                "language": page.language,
                                "created_at": datetime.datetime.utcnow().isoformat(),
                                "source_type": "component_summary_virtual"
                            }
                            all_chunks.append(chunk_text)
                            all_metadatas.append(metadata)
                            all_ids.append(chunk_id)
                else:
                    await log_queue.put(f"    (No significant fields found for summary in virtual component '{comp_name_prefix}')")
        # --- END NEW LOGIC FOR VIRTUAL COMPONENTS ---


        # Original processing for explicit components
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
                await log_queue.put(f"    Generated explicit component summary prompt:\n{component_summary_prompt[:500]}...")

                enhanced_component_text = await enhance_text_with_llm(component_summary_prompt, f"explicit component '{component.componentName}' fields")
                if enhanced_component_text:
                    await log_queue.put(f"    LLM-generated explicit component summary: {enhanced_component_text[:500]}...")
                    component_chunks = text_splitter.split_text(enhanced_component_text)
                    await log_queue.put(f"    Explicit Component fields (enhanced summary) split into {len(component_chunks)} chunks.")
                    for i, chunk_text in enumerate(component_chunks):
                        # Collect all unique effective rendering UIDs from the component's fields for its summary
                        comp_relevant_rendering_uids = sorted(list(set(get_effective_rendering_uid(f) for f in component.fields if get_effective_rendering_uid(f))))
                        comp_rendering_uid_string = "-".join(comp_relevant_rendering_uids)

                        # Make the summary field name unique by including rendering UIDs if present
                        explicit_comp_summary_field_name = f"component_summary_{component.componentName}"
                        if comp_rendering_uid_string:
                            explicit_comp_summary_field_name += f"_{comp_rendering_uid_string}"

                        chunk_id = generate_deterministic_id(page.pageId, explicit_comp_summary_field_name, i, component.componentId, comp_rendering_uid_string)
                        metadata = {
                            "page_id": page.pageId,
                            "page_path": page.pagePath,
                            "page_title": page.pageTitle,
                            "page_url": page.url,
                            "component_id": component.componentId,
                            "field_name": explicit_comp_summary_field_name, # Update metadata field_name
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
        
        # Using a set of (page_id, effective_component_id, field_name) to track summarized fields
        summarized_field_identifier_tuples = set()

        # Capture fields from explicit components that were summarized
        for comp in page.components:
            # Check if this component had fields that contributed to its summary
            component_had_fields_summarized = False
            for f_check in comp.fields:
                if f_check.fieldValue and f_check.fieldName not in ["__Renderings", "__Final Renderings"]:
                    component_had_fields_summarized = True
                    break
            
            if component_had_fields_summarized:
                for f in comp.fields:
                    if f.fieldValue and f.fieldName not in ["__Renderings", "__Final Renderings"]:
                        summarized_field_identifier_tuples.add((page.pageId, comp.componentId, f.fieldName))

        # Capture fields from virtual components that were summarized
        virtual_components_for_summary_tracking = defaultdict(list)
        for field in page.fields:
            if "." in field.fieldName:
                comp_parts = field.fieldName.split(".")
                comp_prefix = comp_parts[0]
                if comp_prefix.lower() != "page":
                    effective_rendering_uid_for_grouping_tracking = get_effective_rendering_uid(field)
                    grouping_key_render_uid_tracking = effective_rendering_uid_for_grouping_tracking if effective_rendering_uid_for_grouping_tracking else "NO_RENDER_UID"
                    grouping_key_tracking = (comp_prefix, grouping_key_render_uid_tracking)
                    virtual_components_for_summary_tracking[grouping_key_tracking].append(field)
        
        for (comp_name_prefix_tracking, grouping_render_uid_tracking), fields_list_tracking in virtual_components_for_summary_tracking.items():
            # Check if this virtual component had fields that contributed to its summary
            virtual_component_had_fields_summarized = False
            for f_check in fields_list_tracking:
                if f_check.fieldValue and f_check.fieldName not in ["__Renderings", "__Final Renderings"]:
                    virtual_component_had_fields_summarized = True
                    break

            if virtual_component_had_fields_summarized:
                actual_rendering_uid_for_id_tracking = grouping_render_uid_tracking if grouping_render_uid_tracking != "NO_RENDER_UID" else ''
                virtual_comp_instance_key_tracking = f"{comp_name_prefix_tracking}{'_' + actual_rendering_uid_for_id_tracking if actual_rendering_uid_for_id_tracking else ''}"
                virtual_comp_instance_id_tracking = hashlib.sha256(f"{page.pageId}-{virtual_comp_instance_key_tracking}".encode('utf-8')).hexdigest()[:16]

                for f_track in fields_list_tracking:
                    if f_track.fieldValue and f_track.fieldName not in ["__Renderings", "__Final Renderings"]:
                        summarized_field_identifier_tuples.add((page.pageId, virtual_comp_instance_id_tracking, f_track.fieldName))


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
            is_part_of_component_summary = False
            
            # 1. Check if it contributed to an EXPLICIT component summary
            if (page.pageId, field.componentId, field.fieldName) in summarized_field_identifier_tuples:
                is_part_of_component_summary = True
            
            # 2. Check if it contributed to a VIRTUAL component summary (only if not already covered by explicit)
            if not is_part_of_component_summary and "." in field.fieldName:
                comp_parts = field.fieldName.split(".")
                comp_prefix = comp_parts[0]
                if comp_prefix.lower() != "page":
                    effective_rendering_uid_for_field_check = get_effective_rendering_uid(field)
                    grouping_key_render_uid_for_field_check = effective_rendering_uid_for_field_check if effective_rendering_uid_for_field_check else "NO_RENDER_UID"
                    
                    virtual_comp_instance_key_for_field_check = f"{comp_prefix}{'_' + grouping_key_render_uid_for_field_check if grouping_key_render_uid_for_field_check != 'NO_RENDER_UID' else ''}"
                    potential_virtual_comp_instance_id_for_field = hashlib.sha256(f"{page.pageId}-{virtual_comp_instance_key_for_field_check}".encode('utf-8')).hexdigest()[:16]

                    if (page.pageId, potential_virtual_comp_instance_id_for_field, field.fieldName) in summarized_field_identifier_tuples:
                        is_part_of_component_summary = True

            if is_part_of_component_summary:
                await log_queue.put(f"    (Skipping individual component field '{field.fieldName}' as its component '{field.componentId or 'virtual'}' was summarized.)")
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
                field_specific_rendering_uid = get_effective_rendering_uid(field)
                chunk_id = generate_deterministic_id(page.pageId, field.fieldName, i, field.componentId, field_specific_rendering_uid)
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
        # --- NEW LOG ADDED HERE ---
        print(f"Full Incoming Payload: {payload.model_dump_json(indent=2)}")
        # --- END NEW LOG ---
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
                    
                    await log_queue.put(f"  - Processing {field_log_label} {field_idx + 1}: {field.fieldName} (Page ID: {page.pageId}), renderinguid (property): '{field.renderingUid}'")
                    
                    soup = BeautifulSoup(field.fieldValue, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)

                    if not text:
                        await log_queue.put(f"    (Skipping empty {field_log_label}: {field.fieldName} for Page ID: {page.pageId})")
                        continue

                    chunks = text_splitter.split_text(text)
                    await log_queue.put(f"    Split into {len(chunks)} chunks. Original text length: {len(text)}")
                    for i, chunk_text in enumerate(chunks):
                        field_specific_rendering_uid = get_effective_rendering_uid(field)
                        chunk_id = generate_deterministic_id(page.pageId, field.fieldName, i, field.componentId, field_specific_rendering_uid)
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