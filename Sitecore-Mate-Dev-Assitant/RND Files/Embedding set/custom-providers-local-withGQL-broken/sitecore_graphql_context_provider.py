import os
import json
from typing import List, Dict, Any, Optional
import sys
import httpx # Required for making HTTP requests to our local proxy

# continue.dev specific imports for custom context providers
try:
    from continuedev.src.meadow.context_provider import ContextProvider
    from continuedev.src.meadow.context_provider import ContextItem, ContextItemType
    # from continuedev.src.meadow.schema import RangeInFileWith # Not directly used in this provider for ranges
    # from continuedev.src.meadow.schema import RangeInFile, Position # Not directly used in this provider for ranges
except ImportError:
    # Fallback for local testing outside of continue.dev's runtime environment
    class ContextProvider:
        pass
    class ContextItem:
        def __init__(self, name, content, description="", start_line=0, end_line=0):
            self.name = name
            self.content = content
            self.description = description
            self.start_line = start_line
            self.end_line = end_line
    class ContextItemType:
        FILE = "file"
        FOLDER = "folder"
        CODE = "code"
        MARKDOWN = "markdown"
        TEXT = "text"
    # Dummy classes if not in continue.dev environment
    class LoadContextItemsArgs:
        pass
    class RangeInFile:
        pass
    class Position:
        pass


class SitecoreGraphQLContextProvider(ContextProvider):
    """
    A custom context provider specifically designed to interact with
    Sitecore's GraphQL API via a local proxy server.
    It provides AI with real-time Sitecore content data based on user queries.
    """
    
    name: str = "Sitecore GraphQL Context" # Unique name for this provider

    def __init__(self, 
                 sitecore_graphql_server_url: Optional[str] = None, 
                 sitecore_graphql_auth_token: Optional[str] = None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sitecore_graphql_server_url = sitecore_graphql_server_url
        self.sitecore_graphql_auth_token = sitecore_graphql_auth_token

        print(f"DEBUG (SitecoreGraphQLContextProvider.__init__): Sitecore GraphQL proxy URL: {self.sitecore_graphql_server_url}", file=sys.stderr)
        if self.sitecore_graphql_auth_token:
            # Print only a snippet for security
            print(f"DEBUG (SitecoreGraphQLContextProvider.__init__): Sitecore GraphQL auth token loaded for proxy (starts with: {self.sitecore_graphql_auth_token[:10]}...)", file=sys.stderr)
        else:
            print("WARNING (SitecoreGraphQLContextProvider.__init__): No Sitecore GraphQL auth token provided. GraphQL queries via this context provider may fail. Ensure MATE_SITECORE_GRAPHQL_AUTH_TOKEN is set as an environment variable in jss_context_server.py.", file=sys.stderr)

    async def _execute_graphql_via_server_proxy(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """
        Executes a GraphQL query by calling our local FastAPI /graphql-query endpoint.
        This acts as a proxy, ensuring the request goes through our server's auth and SSL handling.
        """
        if not self.sitecore_graphql_server_url:
            raise ValueError("Sitecore GraphQL proxy server URL is not configured.")

        print(f"DEBUG (SitecoreGraphQLContextProvider): Calling local GraphQL proxy at {self.sitecore_graphql_server_url}", file=sys.stderr)
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.sitecore_graphql_auth_token:
            headers["Authorization"] = self.sitecore_graphql_auth_token
            print("DEBUG (SitecoreGraphQLContextProvider): Added Authorization header to local GraphQL proxy call.", file=sys.stderr)
        else:
            print("WARNING (SitecoreGraphQLContextProvider): No Authorization header provided to local GraphQL proxy call. May lead to auth errors.", file=sys.stderr)

        request_payload = {"query": query}
        if variables:
            request_payload["variables"] = variables

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.sitecore_graphql_server_url,
                    json=request_payload,
                    headers=headers,
                    timeout=60.0 
                )
                response.raise_for_status()
                json_response = response.json()
                print("DEBUG (SitecoreGraphQLContextProvider): Local GraphQL proxy call successful.", file=sys.stderr)
                
                # --- NEW: Log the full GraphQL response ---
                print(f"DEBUG (SitecoreGraphQLContextProvider): Full GraphQL response received:\n{json.dumps(json_response, indent=2)}", file=sys.stderr)
                # --- END NEW ---

                return json_response
        except httpx.RequestError as exc:
            print(f"ERROR (SitecoreGraphQLContextProvider): Network or request error while calling local GraphQL proxy: {exc}", file=sys.stderr)
            raise ConnectionError(f"Failed to reach local GraphQL proxy server: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"ERROR (SitecoreGraphQLContextProvider): HTTP error from local GraphQL proxy ({exc.response.status_code}): {exc.response.text}", file=sys.stderr)
            raise ValueError(f"Local GraphQL proxy responded with error status {exc.response.status_code}: {exc.response.text}")
        except json.JSONDecodeError as exc:
            print(f"ERROR (SitecoreGraphQLContextProvider): Invalid JSON response from local GraphQL proxy: {exc}. Response: {response.text}", file=sys.stderr)
            raise ValueError(f"Invalid JSON response from local GraphQL proxy: {exc}")
        except Exception as e:
            print(f"ERROR (SitecoreGraphQLContextProvider): Unexpected error during local GraphQL proxy call: {e}", file=sys.stderr)
            raise


    async def provide_context_items(self, full_input_query: str = "") -> List[ContextItem]:
        """
        Provides Sitecore GraphQL content based on the user's query.
        """
        print("DEBUG (SitecoreGraphQLContextProvider): Entering provide_context_items...", file=sys.stderr)
        context_items: List[ContextItem] = []

        # --- Check for GraphQL query intent in full_input_query ---
        lower_query = full_input_query.lower()
        print(f"DEBUG (SitecoreGraphQLContextProvider): Full input query (lowercased): '{lower_query}'", file=sys.stderr) # Added debug

        # Define keywords that suggest a GraphQL query is needed
        graphql_keywords = [
            "sitecore item", "sitecore content", "sitecore template", 
            "graphql item", "get item details", "query item", "item details for", 
            "details for the home item" # Added a more specific phrase from your example
        ]
        
        # Added granular debug for keyword matching
        intent_detected = False
        for keyword in graphql_keywords:
            if keyword in lower_query:
                print(f"DEBUG (SitecoreGraphQLContextProvider): Matched keyword: '{keyword}' - TRUE", file=sys.stderr)
                intent_detected = True
                # No 'break' here to ensure all keywords are logged even if one matches
            else:
                print(f"DEBUG (SitecoreGraphQLContextProvider): Keyword '{keyword}' - FALSE", file=sys.stderr)


        # Only proceed if the query is not empty and contains a GraphQL keyword
        if full_input_query and intent_detected: # Changed 'any(...)' to 'intent_detected'
            print("DEBUG (SitecoreGraphQLContextProvider): Detected potential Sitecore GraphQL query intent. Proceeding with GraphQL query.", file=sys.stderr) # Added debug
            
            if self.sitecore_graphql_server_url and self.sitecore_graphql_auth_token:
                # IMPORTANT: For now, we're using a hardcoded example query.
                # In a more advanced scenario, the LLM would dynamically generate the
                # GraphQL query based on the user's natural language input.
                # You might parse the 'full_input_query' here to extract specific
                # item paths, databases, or languages if you want to make it dynamic.
                
                # Example: try to extract a path or name from the query
                # For simplicity, we'll keep the hardcoded example that worked previously.
                target_path = "/sitecore/content/demo-mate-d-jss-xp/home" 
                target_db = "master"
                target_lang = "en"

                # Corrected GraphQL query based on your feedback
                graphql_query = f"""
                    query {{
                      item(where: {{
                        path: "{target_path}"
                        database: "{target_db}"
                        language: "{target_lang}"
                      }}) {{
                        name
                        itemId
                        displayName
                        hasChildren
                        hasPresentation
                        # Removed id, path, displayName as per your example of a proper query
                      }}
                    }}
                """
                try:
                    graphql_results = await self._execute_graphql_via_server_proxy(graphql_query)
                    context_items.append(
                        ContextItem(
                            name="Sitecore_GraphQL_Query_Result",
                            content=f"Sitecore GraphQL Query Result for '{target_path}' ({target_db}/{target_lang}):\n```json\n{json.dumps(graphql_results, indent=2)}\n```",
                            description="Results from an automatically executed Sitecore GraphQL query based on user intent."
                        )
                    )
                    print("DEBUG (SitecoreGraphQLContextProvider): Added Sitecore GraphQL query results to context.", file=sys.stderr)
                except Exception as e:
                    context_items.append(
                        ContextItem(
                            name="Sitecore_GraphQL_Query_Error",
                            content=f"Error executing Sitecore GraphQL query: {e}",
                            description="Error details for an attempted Sitecore GraphQL query."
                        )
                    )
                    print(f"ERROR (SitecoreGraphQLContextProvider): Failed to execute Sitecore GraphQL query automatically: {e}", file=sys.stderr)
            else:
                context_items.append(
                    ContextItem(
                        name="Sitecore_GraphQL_Setup_Warning",
                        content="Sitecore GraphQL proxy URL or auth token not configured. Automatic GraphQL queries are unavailable. Please set MATE_SITECORE_GRAPHQL_ENDPOINT and MATE_SITECORE_GRAPHQL_AUTH_TOKEN environment variables.",
                        description="Warning about incomplete Sitecore GraphQL setup."
                    )
                )
        else:
            print("DEBUG (SitecoreGraphQLContextProvider): No Sitecore GraphQL query intent detected in the input. Skipping GraphQL query.", file=sys.stderr)
            # You might add a default context item here if no intent is detected but the provider is active
            context_items.append(
                ContextItem(
                    name="Sitecore_GraphQL_Provider_Status",
                    content="Sitecore GraphQL Context Provider is active. Ask questions about Sitecore items (e.g., 'What are the details for the home item?') to trigger GraphQL queries.",
                    description="Provides guidance on how to use this context provider."
                )
            )

        return context_items
