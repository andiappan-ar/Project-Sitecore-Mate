import httpx
import json
from typing import Dict, Optional, Any
import sys

class SitecoreGraphQLClient:
    """
    A client for interacting with the Sitecore Authoring Management GraphQL API.
    Handles GraphQL query execution, basic error reporting, and authentication.
    """
    def __init__(self, sitecore_graphql_endpoint: Optional[str] = None, verify_ssl: bool = True):
        """
        Initializes the SitecoreGraphQLClient.

        Args:
            sitecore_graphql_endpoint (str, optional): The URL of the Sitecore
                Authoring Management GraphQL endpoint. If None, operations will fail.
            verify_ssl (bool): Whether to verify SSL certificates. Set to False for
                development environments with self-signed certificates. Defaults to True.
        """
        self.sitecore_graphql_endpoint = sitecore_graphql_endpoint
        self.verify_ssl = verify_ssl # Store the SSL verification setting
        print(f"DEBUG (SitecoreGraphQLClient.__init__): Initialized with endpoint: {self.sitecore_graphql_endpoint}", file=sys.stderr)
        print(f"DEBUG (SitecoreGraphQLClient.__init__): SSL Verification: {'Enabled' if self.verify_ssl else 'Disabled'}", file=sys.stderr)


        if not self.sitecore_graphql_endpoint:
            print("WARNING (SitecoreGraphQLClient.__init__): Sitecore GraphQL endpoint is not configured. GraphQL operations will not work.", file=sys.stderr)

    async def execute_query(self, query: str, variables: Optional[Dict] = None, authorization_header: Optional[str] = None) -> Dict:
        """
        Executes a GraphQL query against the configured Sitecore GraphQL endpoint.

        Args:
            query (str): The GraphQL query string.
            variables (Dict, optional): A dictionary of variables for the GraphQL query.
            authorization_header (str, optional): The full Authorization header string
                                                  (e.g., "Bearer <token>").

        Returns:
            Dict: The JSON response from the GraphQL endpoint.

        Raises:
            ValueError: If the Sitecore GraphQL endpoint is not configured.
            ConnectionError: If there's a network or connection issue.
            httpx.HTTPStatusError: If the GraphQL endpoint returns an HTTP error.
            json.JSONDecodeError: If the response is not valid JSON.
            Exception: For any other unexpected errors.
        """
        if not self.sitecore_graphql_endpoint:
            raise ValueError("Sitecore GraphQL endpoint is not configured. Cannot execute query.")

        print(f"DEBUG (SitecoreGraphQLClient.execute_query): Executing GraphQL query to {self.sitecore_graphql_endpoint}", file=sys.stderr)
        print(f"Query snippet: {query[:100]}...", file=sys.stderr) # Log first 100 chars of query

        headers = {
            "Content-Type": "application/json",
        }
        
        # --- IMPORTANT: Dynamically set Authorization header if provided ---
        if authorization_header:
            headers["Authorization"] = authorization_header
            print("DEBUG (SitecoreGraphQLClient.execute_query): Using provided Authorization header.", file=sys.stderr)
        else:
            print("WARNING (SitecoreGraphQLClient.execute_query): No Authorization header provided for GraphQL request.", file=sys.stderr)
        # --- End Dynamic Authorization ---

        request_payload = {"query": query}
        if variables:
            request_payload["variables"] = variables

        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.post(
                    self.sitecore_graphql_endpoint,
                    json=request_payload,
                    headers=headers, # <-- NOW USES THE DYNAMIC HEADERS
                    timeout=30.0 # Increased timeout for network requests
                )
                response.raise_for_status() # Raise for 4xx/5xx HTTP errors

                graphql_response = response.json()

                # Check for GraphQL-specific errors in the response body (even if HTTP status is 200 OK)
                if "errors" in graphql_response:
                    print(f"ERROR (SitecoreGraphQLClient.execute_query): GraphQL errors in response: {json.dumps(graphql_response['errors'], indent=2)}", file=sys.stderr)
                    return {"data": graphql_response.get("data"), "errors": graphql_response["errors"]}
                
                print("DEBUG (SitecoreGraphQLClient.execute_query): GraphQL query executed successfully.", file=sys.stderr)
                return graphql_response

        except httpx.RequestError as exc:
            print(f"ERROR (SitecoreGraphQLClient.execute_query): Network or request error while connecting to Sitecore GraphQL: {exc}", file=sys.stderr)
            raise ConnectionError(f"Failed to connect to Sitecore GraphQL endpoint: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"ERROR (SitecoreGraphQLClient.execute_query): HTTP error from Sitecore GraphQL ({exc.response.status_code}): {exc.response.text}", file=sys.stderr)
            raise ValueError(f"Sitecore GraphQL responded with error status {exc.response.status_code}: {exc.response.text}")
        except json.JSONDecodeError as exc:
            print(f"ERROR (SitecoreGraphQLClient.execute_query): Invalid JSON response from Sitecore GraphQL: {exc}. Response: {response.text}", file=sys.stderr)
            raise ValueError(f"Invalid JSON response from Sitecore GraphQL: {exc}")
        except Exception as e:
            print(f"ERROR (SitecoreGraphQLClient.execute_query): Unexpected error during GraphQL execution: {e}", file=sys.stderr)
            raise
