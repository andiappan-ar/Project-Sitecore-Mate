import os
import sys
import json
import urllib.parse
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add the directory containing jss_context_provider.py to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your custom context provider logic
# Ensure jss_context_provider.py is in the same directory or its path is correctly set above
from jss_context_provider import JSSContextProvider, ContextItem, ContextItemType

# Initialize FastAPI app - THIS IS THE 'app' THAT UVICORN LOOKS FOR
app = FastAPI()

# Initialize your JSSContextProvider
jss_provider = JSSContextProvider()

class ContextRequest(BaseModel):
    query: Optional[str] = None
    fullInput: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    workspacePath: Optional[str] = None # This will still be in the request, but not relied upon for the root

# --- IMPORTANT CONFIGURATION: YOU MUST SET THIS TO YOUR JSS PROJECT ROOT ---
# This path is required because the 'http' context provider in continue.dev
# does not automatically send the workspacePath.
# Replace "D:\\YOUR\\JSS\\PROJECT\\ROOT" with the actual root path of your Sitecore JSS project.
# Ensure correct backslashes (escaped with \\) for Windows paths or use forward slashes.
JSS_PROJECT_ROOT = "D:\\ARC\\Code\\AI\\Project Mate\\DEV ASSISTANT\\Demo-Mate-D-JSS\\XP"
# --- END IMPORTANT CONFIGURATION ---


def _uri_to_os_path(uri: str) -> str:
    """
    Converts a file:// URI to an OS-specific file system path.
    (This function is still kept for completeness if workspacePath was to be used directly in other contexts,
    but it's not the primary mechanism for determining JSS_PROJECT_ROOT anymore.)
    """
    if not uri.startswith("file:///"):
        return uri
    path = uri[len("file:///"):]
    path = urllib.parse.unquote(path)
    if sys.platform == "win32" and len(path) > 2 and path[0] == '/' and path[2] == ':':
        path = path[1:]
    return os.path.normpath(path)

@app.post("/context")
async def get_jss_context(request_body: ContextRequest):
    """
    This endpoint serves the custom context to continue.dev.
    It receives the context request, calls the JSSContextProvider,
    and returns the context items.
    """
    print(f"\n--- Received Raw Context Request (from continue.dev) ---")
    print(json.dumps(request_body.dict(), indent=2))
    print(f"--- End Raw Context Request ---\n")

    print(f"Received context request for query: '{request_body.query}'")
    print(f"Full Input: '{request_body.fullInput}'")
    print(f"Options: {request_body.options}")
    print(f"Workspace Path from request (if any): {request_body.workspacePath}")

    # --- Use the explicitly configured JSS_PROJECT_ROOT ---
    determined_workspace_dir = JSS_PROJECT_ROOT
    print(f"DEBUG (Server): Using configured JSS project root: {determined_workspace_dir}")

    if not determined_workspace_dir:
        print("ERROR (Server): JSS_PROJECT_ROOT is not configured. Please set it in jss_context_server.py", file=sys.stderr)
        return []

    try:
        # Pass the full input query to the provider for intelligent context sending
        context_items = await jss_provider.provide_context_items(determined_workspace_dir, request_body.fullInput)
        
        response_items = [
            {"name": item.name, "description": item.description, "content": item.content}
            for item in context_items
        ]
        
        print(f"Returning {len(response_items)} context items from: {determined_workspace_dir}")
        return response_items
    except Exception as e:
        print(f"ERROR (Server): Error providing JSS context: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []

# To run this server:
# 1. Open your terminal/command prompt.
# 2. Navigate to the directory where you saved this file: C:\Users\Andi\.continue\custom-providers\
# 3. Ensure packages are installed: pip install fastapi uvicorn
# 4. Run the server: uvicorn jss_context_server:app --reload --port 8000
