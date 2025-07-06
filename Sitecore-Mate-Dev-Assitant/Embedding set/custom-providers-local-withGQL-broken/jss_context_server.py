import os
import sys
import json
import urllib.parse
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add the directory containing jss_context_provider.py and sitecore_graphql_context_provider.py
# to the Python path. We also expect sitecore_graphql_client.py to be here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your custom context provider logic
from jss_context_provider import JSSContextProvider, ContextItem, ContextItemType
from sitecore_graphql_context_provider import SitecoreGraphQLContextProvider # NEW import for GraphQL provider
# Import the separate Sitecore GraphQL Client
from sitecore_graphql_client import SitecoreGraphQLClient

# Initialize FastAPI app
app = FastAPI()

# --- IMPORTANT CONFIGURATION: READ FROM ENVIRONMENT VARIABLES ---
# All environment variables now have the MATE_ prefix and default fallbacks where appropriate.
# These variables should ideally be set by the VS Code extension or your environment.

# JSS Project Root: CRITICAL - This MUST be set to your actual JSS project root.
# Using a generic placeholder as a default. For Windows, consider something like "C:\\your\\jss\\project\\root"
# CORRECTED: Providing a direct default absolute path string to avoid concatenation issues.
MATE_JSS_PROJECT_ROOT = os.getenv("MATE_JSS_PROJECT_ROOT", "D:\\ARC\\Code\\AI\\Project Mate\\DEV ASSISTANT\\Demo-Mate-D-JSS\\XP") 

# Ollama Embedding Model: Fallback for local embedding service
MATE_OLLAMA_EMBEDDING_MODEL = os.getenv("MATE_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text") 

# Ollama Embedding URL: Fallback for local embedding service endpoint
MATE_OLLAMA_EMBEDDING_URL = os.getenv("MATE_OLLAMA_EMBEDDING_URL", "http://localhost:11434/api/embeddings")

# Google API Key for Gemini Embeddings: Cannot have a safe or meaningful default.
# If not provided, Ollama will be used as a fallback if configured.
MATE_GOOGLE_API_KEY = os.getenv("MATE_GOOGLE_API_KEY", "YOUR_GOOGLE_GEMINI_API_KEY_HERE") # Recommend not hardcoding your actual key here for production

# Sitecore GraphQL Endpoint: Default to a common local Authoring GraphQL URL.
# This MUST be set to your actual Sitecore instance's GraphQL endpoint.
MATE_SITECORE_GRAPHQL_ENDPOINT = os.getenv("MATE_SITECORE_GRAPHQL_ENDPOINT", "https://sc104sc.dev.local/sitecore/api/authoring/graphql/v1")

# Sitecore GraphQL SSL Verification: Defaults to True for security, set to "0" to disable.
MATE_SITECORE_GRAPHQL_VERIFY_SSL = os.getenv("MATE_SITECORE_GRAPHQL_VERIFY_SSL", "0").lower() in ('true', '1', 't')

# Sitecore GraphQL Authorization Token: Cannot have a safe or meaningful default.
# This MUST be set with a valid token for Sitecore GraphQL queries to work.
MATE_SITECORE_GRAPHQL_AUTH_TOKEN = os.getenv("MATE_SITECORE_GRAPHQL_AUTH_TOKEN", "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjhBNjZGMEYwMEY5OUNBNjczMTY3ODgzMjA4QjVDNjk5RkE1QTY0MjZSUzI1NiIsIng1dCI6ImltYnc4QS1aeW1jeFo0Z3lDTFhHbWZwYVpDWSIsInR5cCI6ImF0K2p3dCJ9.eyJpc3MiOiJodHRwczovL3NjMTA0aWRlbnRpdHlzZXJ2ZXIuZGV2LmxvY2FsIiwibmJmIjoxNzUxMjA1NjkwLCJpYXQiOjE3NTEyMDU2OTAsImV4cCI6MTc1MTIwOTI5MCwiYXVkIjpbInNpdGVjb3JlLnByb2ZpbGUuYXBpIiwiaHR0cHM6Ly9zYzEwNGlkZW50aXR5c2VydmVyLmRldi5sb2NhbC9yZXNvdXJjZXMiXSwic2NvcGUiOlsib3BlbmlkIiwic2l0ZWNvcmUucHJvZmlsZSIsInNpdGVjb3JlLnByb2ZpbGUuYXBpIl0sImFtciI6WyJwd2QiXSwiY2xpZW50X2lkIjoicG9zdG1hbi1hcGkiLCJzdWIiOiI5MzE3OTYyZTBjMjA0OTIyYTBlMGNhOWIxZmQ2MDk1NyIsImF1dGhfdGltZSI6MTc1MTIwNTY5MCwiaWRwIjoibG9jYWwiLCJuYW1lIjoic2l0ZWNvcmVcXEFkbWluIiwiZW1haWwiOiIiLCJodHRwOi8vd3d3LnNpdGVjb3JlLm5ldC9pZGVudGl0eS9jbGFpbXMvaXNBZG1pbiI6IlRydWUiLCJqdGkiOiJERTBCNjFGQkRDQjM3M0NEM0I1NTYxMzYxOTE1RjdCMyJ9.siUz6KyRmLvwgAR7ZoDRP39OFvErrd1tanNQjJZdfqgFcS6PE7terCD__eM7wiPkuB345_4qp8kOrwLE_7T0KNa_NF9VmgIeOm2CUEur3Hnge2YigevzVbegfiOYbzj4SxV1j6S1JJuDabXJKwtGshsCwbOxh81gXc81YhlHmePn2EeUT8Jz5YFz-xKpk7Lt1YGruvh_9yURPp5MRC5lEoyiagDMhhlmMvm5_z2dxedgTt08QkV8IXwzFsPLuCK1Y7yf2H_8jRQpjPHUfOeK1VycgXM83EZsMjDS_QIKo197yTMRQqE1AD3CjZCEySVGgwMl-EU55HpdeHFZo4xR1Q") 
# --- END IMPORTANT CONFIGURATION ---

# Initialize your JSSContextProvider for code context
jss_context_provider_instance = JSSContextProvider(
    ollama_embedding_model=MATE_OLLAMA_EMBEDDING_MODEL,
    ollama_embedding_url=MATE_OLLAMA_EMBEDDING_URL,
    google_api_key=MATE_GOOGLE_API_KEY 
)

# Initialize the Sitecore GraphQL Client - This will be used by the GraphQL Context Provider
sitecore_graphql_client_instance = SitecoreGraphQLClient(
    sitecore_graphql_endpoint=MATE_SITECORE_GRAPHQL_ENDPOINT,
    verify_ssl=MATE_SITECORE_GRAPHQL_VERIFY_SSL
)

# Initialize the SitecoreGraphQLContextProvider for GraphQL data context
sitecore_graphql_context_provider_instance = SitecoreGraphQLContextProvider(
    sitecore_graphql_server_url="http://localhost:8000/graphql-query", # This proxy endpoint is exposed by THIS server
    sitecore_graphql_auth_token=MATE_SITECORE_GRAPHQL_AUTH_TOKEN # Pass the token from env to the GraphQL provider
)

class ContextRequest(BaseModel):
    query: Optional[str] = None
    fullInput: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    workspacePath: Optional[str] = None

class GraphQLRequest(BaseModel):
    query: str
    variables: Optional[Dict[str, Any]] = None

@app.post("/jss-code-context") # Dedicated endpoint for JSS code context
async def get_jss_code_context(request_body: ContextRequest):
    """
    This endpoint serves the custom context related to JSS project files (code, definitions, etc.) to continue.dev.
    """
    print(f"\n--- Received JSS Code Context Request (from continue.dev) ---")
    print(json.dumps(request_body.dict(), indent=2))
    print(f"--- End JSS Code Context Request ---\n")

    print(f"Received context request for query: '{request_body.query}'")
    print(f"Full Input: '{request_body.fullInput}'")

    determined_workspace_dir = MATE_JSS_PROJECT_ROOT
    print(f"DEBUG (Server): Attempting to use JSS project root from environment variable or default: {determined_workspace_dir}")

    if not determined_workspace_dir or not os.path.isdir(determined_workspace_dir): 
        error_msg = f"ERROR (Server): MATE_JSS_PROJECT_ROOT environment variable is not set or the path does not exist: '{determined_workspace_dir}'. Please set it to the actual root of your Sitecore JSS project. This is typically configured by the VS Code extension."
        print(error_msg, file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        # Pass the full input query and workspace_dir to the JSS code provider
        context_items = await jss_context_provider_instance.provide_context_items(determined_workspace_dir, request_body.fullInput)
        
        response_items = [
            {"name": item.name, "description": item.description, "content": item.content}
            for item in context_items
        ]
        
        print(f"Returning {len(response_items)} JSS code context items from: {determined_workspace_dir}")
        return response_items
    except Exception as e:
        print(f"ERROR (Server): Error providing JSS code context: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error providing JSS code context: {e}")

@app.post("/sitecore-graphql-context") # Dedicated endpoint for Sitecore GraphQL context
async def get_sitecore_graphql_context(request_body: ContextRequest):
    """
    This endpoint serves Sitecore GraphQL context to continue.dev.
    It receives the context request, calls the SitecoreGraphQLContextProvider,
    and returns the context items.
    """
    print(f"\n--- Received Sitecore GraphQL Context Request (from continue.dev) ---")
    print(json.dumps(request_body.dict(), indent=2))
    print(f"--- End Sitecore GraphQL Context Request ---\n")

    print(f"Received GraphQL context request for query: '{request_body.query}'")
    print(f"Full Input: '{request_body.fullInput}'")

    try:
        # Pass only the full input query to the Sitecore GraphQL provider
        context_items = await sitecore_graphql_context_provider_instance.provide_context_items(request_body.fullInput)
        
        response_items = [
            {"name": item.name, "description": item.description, "content": item.content}
            for item in context_items
        ]
        
        print(f"Returning {len(response_items)} Sitecore GraphQL context items.")
        return response_items
    except Exception as e:
        print(f"ERROR (Server): Error providing Sitecore GraphQL context: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error providing Sitecore GraphQL context: {e}")


@app.post("/graphql-query") # This endpoint is a proxy for the SitecoreGraphQLClient
async def execute_sitecore_graphql_query_proxy(request: Request, request_body: GraphQLRequest):
    """
    This endpoint acts as a proxy for the SitecoreGraphQLClient.
    It allows context providers (like SitecoreGraphQLContextProvider) to send
    GraphQL queries to Sitecore by calling this local endpoint, which then
    handles the actual request to the Sitecore GraphQL API.
    """
    print(f"\n--- Received GraphQL Proxy Query Request ---")
    print(json.dumps(request_body.dict(), indent=2))
    print(f"--- End GraphQL Proxy Query Request ---\n")

    if not sitecore_graphql_client_instance.sitecore_graphql_endpoint:
        error_msg = "ERROR (Server): Sitecore GraphQL endpoint is not configured for the GraphQL client. Cannot execute query."
        print(error_msg, file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg)

    # Extract Authorization header from incoming request (from context provider)
    authorization_header = request.headers.get("Authorization")
    if authorization_header:
        print(f"DEBUG (Server): Forwarding Authorization header: {authorization_header[:30]}...", file=sys.stderr)
    else:
        print("WARNING (Server): No Authorization header received by proxy from context provider. Ensure MATE_SITECORE_GRAPHQL_AUTH_TOKEN is set and passed.", file=sys.stderr)

    try:
        graphql_results = await sitecore_graphql_client_instance.execute_query(
            request_body.query,
            request_body.variables,
            authorization_header=authorization_header
        )
        print("DEBUG (Server): GraphQL proxy query executed and results returned from Sitecore.", file=sys.stderr)
        return graphql_results
    except ValueError as e:
        print(f"ERROR (Server): GraphQL client configuration error during proxy call: {e}", file=sys.stderr)
        raise HTTPException(status_code=400, detail=f"GraphQL Client Error during proxy: {e}")
    except ConnectionError as e:
        print(f"ERROR (Server): Connection error to Sitecore GraphQL during proxy call: {e}", file=sys.stderr)
        raise HTTPException(status_code=503, detail=f"Failed to connect to Sitecore GraphQL via proxy: {e}")
    except HTTPException:
        raise # Re-raise existing HTTPExceptions
    except Exception as e:
        print(f"ERROR (Server): Unexpected error during GraphQL proxy query execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Unexpected error during GraphQL proxy query: {e}")


@app.post("/clear-index")
async def clear_jss_index_endpoint(): # Renamed to avoid conflict with provider method if any
    print("DEBUG (Server): Received request to clear ChromaDB index.", file=sys.stderr)
    determined_workspace_dir = MATE_JSS_PROJECT_ROOT

    if not determined_workspace_dir:
        error_msg = "ERROR (Server): MATE_JSS_PROJECT_ROOT environment variable is not set. Cannot clear index."
        print(error_msg, file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        # Call the clear method on the JSSContextProvider instance
        await jss_context_provider_instance._clear_chroma_index(determined_workspace_dir)
        print("DEBUG (Server): Index clear command sent to JSS code provider.", file=sys.stderr)
        return {"status": "success", "message": "ChromaDB index clear initiated for JSS code context."}
    except Exception as e:
        error_msg = f"ERROR (Server): Failed to clear index: {e}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg)
