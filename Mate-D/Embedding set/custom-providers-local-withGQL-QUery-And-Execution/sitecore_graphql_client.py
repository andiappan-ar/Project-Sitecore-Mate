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

    async def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None, authorization_header: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes a GraphQL query against the Sitecore GraphQL endpoint.

        Args:
            query (str): The GraphQL query string.
            variables (Optional[Dict[str, Any]]): A dictionary of variables for the GraphQL query.
            authorization_header (Optional[str]): The full Authorization header value (e.g., "Bearer YOUR_TOKEN").

        Returns:
            Dict[str, Any]: The JSON response from the GraphQL endpoint.

        Raises:
            ConnectionError: If there's a network or connection issue.
            ValueError: If the GraphQL endpoint returns an HTTP error or invalid JSON.
            Exception: For other unexpected errors.
        """
        if not self.sitecore_graphql_endpoint:
            raise ValueError("Sitecore GraphQL endpoint is not configured.")

        headers = {
            "Content-Type": "application/json"
        }
        if authorization_header:
            headers["Authorization"] = authorization_header
        
        request_body = {"query": query}
        if variables:
            request_body["variables"] = variables

        try:
            print(f"DEBUG (SitecoreGraphQLClient.execute_query): Sending GraphQL request to {self.sitecore_graphql_endpoint} with query excerpt: {query[:100]}...", file=sys.stderr)
            async with httpx.AsyncClient(verify=self.verify_ssl) as client:
                response = await client.post(self.sitecore_graphql_endpoint, headers=headers, json=request_body)
                response.raise_for_status() # Raise an exception for 4xx or 5xx responses

                graphql_response = response.json()

                # Sitecore GraphQL often returns 200 OK even with GraphQL errors in the body
                if 'errors' in graphql_response and graphql_response['errors']:
                    print(f"WARNING (SitecoreGraphQLClient.execute_query): GraphQL errors in response: {json.dumps(graphql_response['errors'], indent=2)}", file=sys.stderr)
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
