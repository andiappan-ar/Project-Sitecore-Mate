import os
import json
import re
from typing import List, Dict, Any, Optional
import sys
import httpx # Required for making HTTP requests to our local proxy
import asyncio # For running blocking operations in a thread

# --- Imports for Local Embeddings and Vector DB ---
from sentence_transformers import SentenceTransformer 
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter # Still imported, but used differently
# --- End Imports for Local Embeddings and Vector DB ---

# continue.dev specific imports for custom context providers
try:
    from continuedev.src.meadow.context_provider import ContextProvider
    from continuedev.src.meadow.context_provider import ContextItem, ContextItemType
    from continuedev.src.meadow.schema import RangeInFileWith
    from continuedev.src.meadow.schema import RangeInFile, Position
except ImportError:
    # Fallback for local testing outside of continue.dev's runtime environment
    class ContextProvider:
        def __init__(self, *args, **kwargs):
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
    It provides AI with real-time Sitecore content data based on user queries,
    and now includes semantic search over sample GraphQL queries.
    """
    
    name: str = "Sitecore GraphQL Context" # Unique name for this provider

    def __init__(self, 
                 sitecore_graphql_server_url: Optional[str] = None, 
                 sitecore_graphql_auth_token: Optional[str] = None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sitecore_graphql_server_url = sitecore_graphql_server_url
        self.sitecore_graphql_auth_token = sitecore_graphql_auth_token

        # --- ChromaDB and Embedding Model Initialization ---
        self.chroma_client = None
        self.chroma_collection = None
        self.embedding_model = None 
        self.embedding_model_name = "jinaai/jina-embeddings-v2-base-code" # Optimized for code/text
        self.initialized_db = False # Flag to track if DB and model are initialized
        self.use_fastembed = False # Set to False to leverage SentenceTransformer/GPU
        # --- End ChromaDB and Embedding Model Initialization ---

        print(f"DEBUG (SitecoreGraphQLContextProvider.__init__): Sitecore GraphQL proxy URL: {self.sitecore_graphql_server_url}", file=sys.stderr)
        if self.sitecore_graphql_auth_token:
            # Print only a snippet for security
            print(f"DEBUG (SitecoreGraphQLContextProvider.__init__): Sitecore GraphQL auth token loaded for proxy (starts with: {self.sitecore_graphql_auth_token[:10]}...)", file=sys.stderr)
        else:
            print("WARNING (SitecoreGraphQLContextProvider.__init__): No Sitecore GraphQL auth token provided. GraphQL queries via this context provider may fail. Ensure MATE_SITECORE_GRAPHQL_AUTH_TOKEN is set as an environment variable in jss_context_server.py.", file=sys.stderr)

    async def _init_vector_db_and_model(self, workspace_dir: str):
        """Initializes the embedding model and ChromaDB client/collection for GraphQL samples."""
        print("DEBUG (GraphQLContextProvider): Entering _init_vector_db_and_model...", file=sys.stderr)
        if self.initialized_db:
            print("DEBUG (GraphQLContextProvider): Vector DB already initialized. Skipping.", file=sys.stderr)
            return

        try:
            db_path = os.path.join(workspace_dir, ".chroma_db_graphql") # Separate DB for GraphQL samples
            os.makedirs(db_path, exist_ok=True)
            print(f"DEBUG (GraphQLContextProvider): ChromaDB client attempting to connect to: {db_path}", file=sys.stderr)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            try:
                self.chroma_collection = self.chroma_client.get_collection("graphql_sample_queries")
                print("DEBUG (GraphQLContextProvider): Existing ChromaDB collection 'graphql_sample_queries' found.", file=sys.stderr)
            except Exception as e:
                print(f"DEBUG (GraphQLContextProvider): Collection 'graphql_sample_queries' not found or error accessing: {e}. Creating new one.", file=sys.stderr)
                self.chroma_collection = self.chroma_client.create_collection("graphql_sample_queries")
                print("DEBUG (GraphQLContextProvider): New ChromaDB collection 'graphql_sample_queries' created.", file=sys.stderr)

            if not self.use_fastembed:
                print(f"DEBUG (GraphQLContextProvider): Attempting to load '{self.embedding_model_name}' using SentenceTransformer...", file=sys.stderr)
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"DEBUG (GraphQLContextProvider): SentenceTransformer model '{self.embedding_model_name}' loaded successfully.", file=sys.stderr)

            self.initialized_db = True
            print("DEBUG (GraphQLContextProvider): Vector DB and embedding model initialization COMPLETE.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR (GraphQLContextProvider): Failed to initialize vector DB or embedding model: {e}", file=sys.stderr)
            import traceback 
            traceback.print_exc(file=sys.stderr) 
            self.initialized_db = False 

    async def process_single_chunk_embedding(self, chunk_content: str, details: Dict[str, Any]):
        """
        Helper function to embed a single chunk (GraphQL query part) asynchronously.
        """
        if not self.embedding_model:
            print("ERROR (GraphQLContextProvider): Embedding model not initialized in process_single_chunk_embedding.", file=sys.stderr)
            return None, None, None

        try:
            print(f"DEBUG (GraphQLContextProvider): Embedding chunk with Jina ({self.embedding_model_name}): {chunk_content[:50]}...", file=sys.stderr)
            embedding = await asyncio.to_thread(self.embedding_model.encode, chunk_content, convert_to_tensor=False)
            embedding = embedding.tolist()
            
            return chunk_content, embedding, details
        except Exception as e:
            print(f"ERROR (GraphQLContextProvider): Failed to embed chunk from {details.get('source', 'N/A')} (chunk {details.get('chunk_index', 'N/A')}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None, None, None

    async def _index_sample_graphql_queries(self, sample_file_path: str):
        """
        Reads the sample GraphQL queries file, extracts queries based on separators,
        and indexes each full query as a single document into ChromaDB.
        """
        if not self.initialized_db or not self.chroma_collection or not self.embedding_model:
            print("ERROR (GraphQLContextProvider): DB or embedding model not initialized. Cannot index sample queries.", file=sys.stderr)
            return

        print(f"DEBUG (GraphQLContextProvider): Starting indexing of sample GraphQL queries from: {sample_file_path}", file=sys.stderr)
        
        # We no longer use RecursiveCharacterTextSplitter to split *within* the query blocks.
        # Each query block itself will be treated as a single "document" for embedding.
        
        indexed_count = 0
        chunks_for_embedding_tasks = []

        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                
                # Regex to extract content between the custom separators
                # This pattern matches the description and the content of each query block.
                query_blocks = re.findall(r'#\*\*\*\*-------------- \[start\] (.*?) --------------\*\*\*\*\s*(.*?)\s*#\*\*\*\*-------------- \[end\] .*? --------------\*\*\*\*', full_content, re.DOTALL)
                
                if not query_blocks:
                    print(f"WARNING (GraphQLContextProvider): No query blocks found using the specified separators in {sample_file_path}", file=sys.stderr)
                    # If no specific blocks are found, consider the whole file as one segment (fallback)
                    if full_content.strip():
                        # Use a simpler description if it's the full file content
                        query_blocks = [("Full Sample File Content", full_content)] 
                    else:
                        print(f"WARNING (GraphQLContextProvider): Sample file {sample_file_path} is empty or contains no detectable queries.", file=sys.stderr)
                        return # Nothing to index

                for i, (description, query_content) in enumerate(query_blocks):
                    # Remove comments from the query content for cleaner embedding
                    # and reduce multiple spaces to single spaces, stripping leading/trailing whitespace.
                    cleaned_query = re.sub(r'#.*?\n', '\n', query_content) # Remove single-line comments
                    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip() 

                    if not cleaned_query:
                        print(f"WARNING (GraphQLContextProvider): Skipped empty or whitespace-only query block {i} (Description: '{description[:50]}...') from {sample_file_path}", file=sys.stderr)
                        continue

                    # Treat the entire cleaned_query as a single document/chunk
                    # We are NOT splitting it further with text_splitter here.
                    print(f"DEBUG (GraphQLContextProvider): Preparing to index full query block '{description[:50]}...' as one document.", file=sys.stderr)

                    details = {
                        "chunk_content": cleaned_query, # The full query block is the "chunk"
                        "source_file": os.path.basename(sample_file_path),
                        "query_description": description.strip(), # Store the description from the start marker
                        "query_index": i,
                        "chunk_index": 0, # Since it's a single chunk per query block
                        "source_type": "graphql_sample",
                        "id": f"graphql_sample_full_query_{i}" # Unique ID for the full query block
                    }
                    # Append the coroutine object (result of calling async method)
                    chunks_for_embedding_tasks.append(self.process_single_chunk_embedding(cleaned_query, details))
            except Exception as e:
                print(f"ERROR (GraphQLContextProvider): Failed to read or process sample GraphQL file {sample_file_path}: {e}", file=sys.stderr)
        else:
            print(f"WARNING (GraphQLContextProvider): Sample GraphQL queries file not found at: {sample_file_path}", file=sys.stderr)
            return

        print(f"DEBUG (GraphQLContextProvider): Gathering {len(chunks_for_embedding_tasks)} embedding tasks concurrently for samples...", file=sys.stderr)
        
        processed_results_from_embedding = []
        # Await all coroutines here. batch_results will contain the actual results (not coroutines).
        batch_results = await asyncio.gather(*chunks_for_embedding_tasks, return_exceptions=True) 
        
        for res in batch_results: # Iterate directly over batch_results
            if isinstance(res, Exception): # Check if the result was an exception
                print(f"ERROR (GraphQLContextProvider): An async embedding task for sample failed: {res}", file=sys.stderr)
            else:
                processed_results_from_embedding.append(res) # Append the already-awaited result

        all_documents_for_chroma = []
        all_embeddings_for_chroma = []
        all_metadatas_for_chroma = []
        all_ids_for_chroma = []

        for chunk_content, embedding, details in processed_results_from_embedding:
            if chunk_content and embedding is not None and details: 
                all_documents_for_chroma.append(chunk_content)
                all_embeddings_for_chroma.append(embedding)
                all_metadatas_for_chroma.append({k: v for k, v in details.items() if k not in ['chunk_content', 'id']}) 
                all_ids_for_chroma.append(details['id'])
        
        print(f"DEBUG (GraphQLContextProvider): Total sample chunks prepared for ChromaDB addition: {len(all_documents_for_chroma)}", file=sys.stderr)

        CHROMA_ADD_BATCH_SIZE = 1000 
        for i in range(0, len(all_documents_for_chroma), CHROMA_ADD_BATCH_SIZE):
            docs_batch = all_documents_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            embeddings_batch = all_embeddings_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            metadatas_batch = all_metadatas_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            ids_batch = all_ids_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]

            try:
                print(f"DEBUG (GraphQLContextProvider): Adding ChromaDB batch {i // CHROMA_ADD_BATCH_SIZE + 1} of {len(docs_batch)} sample chunks.", file=sys.stderr)
                self.chroma_collection.add(
                    documents=docs_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch,
                    ids=ids_batch
                )
                indexed_count += len(docs_batch)
            except Exception as e:
                print(f"ERROR (GraphQLContextProvider): Failed to add ChromaDB batch for samples: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue 

        print(f"DEBUG (GraphQLContextProvider): Sample query indexing finished. Total chunks added: {indexed_count}.", file=sys.stderr)


    async def _query_vector_db(self, query_text: str) -> List[str]:
        """
        Queries the vector database for semantically similar sample GraphQL queries.
        """
        if not self.initialized_db or not self.chroma_collection or not self.embedding_model:
            print("ERROR (GraphQLContextProvider): DB or embedding model not initialized. Cannot query samples.", file=sys.stderr)
            return []

        try:
            print(f"DEBUG (GraphQLContextProvider): Generating query embedding for sample search: {query_text[:50]}...", file=sys.stderr)
            query_embedding = await asyncio.to_thread(self.embedding_model.encode, query_text, convert_to_tensor=False)
            query_embedding = query_embedding.tolist()
            
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding], 
                n_results=3, # Retrieve top 3 most relevant sample queries
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"\n--- DEBUG: Sample GraphQL Query Search Results for '{query_text}' ---", file=sys.stderr)
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    query_description = metadata.get('query_description', 'N/A')
                    print(f"DEBUG Result {i+1}: Distance={distance:.4f}, Source={metadata.get('source_file', 'N/A')}, Desc='{query_description}'", file=sys.stderr)
                    print(f"DEBUG Query Snippet:\n{doc_content[:300]}...\n", file=sys.stderr)

                    if distance < 1000.0: # ADJUSTED THRESHOLD (keeping it high as per previous discussion)
                        formatted_results.append(
                            f"Relevant Sample GraphQL Query (Description: '{query_description}', Similarity Distance: {distance:.4f}):\n"
                            f"```graphql\n{doc_content}\n```"
                        )
            print("--- END DEBUG Sample GraphQL Query Search Results ---\n", file=sys.stderr)
            return formatted_results
        except Exception as e:
            print(f"ERROR (GraphQLContextProvider): Failed to query sample GraphQL vector DB: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return []

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
                
                print(f"DEBUG (SitecoreGraphQLContextProvider): Full GraphQL response received:\n{json.dumps(json_response, indent=2)}", file=sys.stderr)

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
        Now includes semantic search over sample GraphQL queries.
        """
        print("DEBUG (SitecoreGraphQLContextProvider): Entering provide_context_items...", file=sys.stderr)
        context_items: List[ContextItem] = []
        
        # Determine the workspace directory. Assuming it's the directory where this script resides for samples.
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        sample_graphql_file = os.path.join(workspace_dir, "sitecore_auhoring_gql_samples.txt")

        # --- Initialize Vector DB and Embedding Model ---
        await self._init_vector_db_and_model(workspace_dir)
        
        if not self.initialized_db:
            context_items.append(
                ContextItem(
                    name="GraphQL_Vector_DB_Status",
                    content="GraphQL sample vector database and embedding model failed to initialize. Semantic search for GraphQL queries unavailable.",
                    description="Indicates a problem with GraphQL sample vector database setup (check terminal logs)."
                )
            )
            return context_items # Cannot proceed without DB

        # Check current chunk count to decide if initial indexing is needed for samples
        current_sample_chunk_count = self.chroma_collection.count()
        if current_sample_chunk_count == 0:
            print("DEBUG (GraphQLContextProvider): ChromaDB collection for GraphQL samples is empty. Performing initial indexing...", file=sys.stderr)
            await self._index_sample_graphql_queries(sample_graphql_file)
            current_sample_chunk_count = self.chroma_collection.count()
            if current_sample_chunk_count > 0:
                print(f"DEBUG (GraphQLContextProvider): Initial indexing of GraphQL samples complete. Total chunks: {current_sample_chunk_count}", file=sys.stderr)
            else:
                print("DEBUG (GraphQLContextProvider): Initial indexing of GraphQL samples attempted, but no chunks were added.", file=sys.stderr)
        else:
            print(f"DEBUG (GraphQLContextProvider): ChromaDB collection for GraphQL samples already contains {current_sample_chunk_count} chunks. Skipping indexing.", file=sys.stderr)
        
        context_items.append(
            ContextItem(
                name="GraphQL_Vector_DB_Status",
                content=f"GraphQL sample vector database initialized with {current_sample_chunk_count} chunks.",
                description="Status of the vector database for sample GraphQL queries."
            )
        )

        # --- New: Handle direct GraphQL execution command ---
        EXECUTE_COMMAND = "execute-graphql-cmd"
        if full_input_query.lower().startswith(EXECUTE_COMMAND):
            print(f"DEBUG (GraphQLContextProvider): Detected '{EXECUTE_COMMAND}' command. Attempting direct GraphQL execution.", file=sys.stderr)
            graphql_query_to_execute = full_input_query[len(EXECUTE_COMMAND):].strip()

            if not graphql_query_to_execute:
                context_items.append(
                    ContextItem(
                        name="GraphQL_Execution_Error",
                        content=f"Error: No GraphQL query provided after '{EXECUTE_COMMAND}'. Please provide the query.",
                        description="Error: Missing GraphQL query for direct execution command."
                    )
                )
                print(f"ERROR (GraphQLContextProvider): No GraphQL query provided after '{EXECUTE_COMMAND}'.", file=sys.stderr)
                return context_items # Return with error message

            try:
                print(f"DEBUG (GraphQLContextProvider): Executing GraphQL query via proxy: {graphql_query_to_execute[:200]}...", file=sys.stderr)
                sitecore_response = await self._execute_graphql_via_server_proxy(graphql_query_to_execute)
                
                context_items.append(
                    ContextItem(
                        name="Sitecore_GraphQL_Execution_Result",
                        content=f"Successfully executed GraphQL query. Sitecore Response:\n```json\n{json.dumps(sitecore_response, indent=2)}\n```",
                        description="Result of direct GraphQL query execution against Sitecore."
                    )
                )
                print("DEBUG (GraphQLContextProvider): Successfully executed GraphQL query and added response to context.", file=sys.stderr)
            except Exception as e:
                context_items.append(
                    ContextItem(
                        name="GraphQL_Execution_Error",
                        content=f"Failed to execute GraphQL query: {e}",
                        description="Error during direct GraphQL query execution."
                    )
                )
                print(f"ERROR (GraphQLContextProvider): Failed to execute GraphQL query via proxy: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
            return context_items # End of execution path for direct command
        # --- End New: Handle direct GraphQL execution command ---


        # --- Existing: Check for GraphQL query intent for sample provision ---
        lower_query = full_input_query.lower()
        graphql_keywords = [
            "sitecore item", "sitecore content", 
            "graphql item", "get item details", "query item", "item details for", 
            "get me data for", "home item", "master database", "get me the", 
            "find item", "item in", "item data", "item info", "name", "id", "path", 
            "component", "rendering", "component details", "rendering details",
            "component info", "rendering info", "find component", "get component",
            "details from sitecore", "get me the details for",
            "template", "template details", "gimme the template details for", "details for this template",
            "component name", "rendering name", "details by component name"
        ]
        
        intent_detected = any(keyword in lower_query for keyword in graphql_keywords)
        
        print(f"DEBUG (GraphQLContextProvider): full_input_query received: '{full_input_query}'", file=sys.stderr)
        print(f"DEBUG (GraphQLContextProvider): lower_query: '{lower_query}'", file=sys.stderr)
        print(f"DEBUG (GraphQLContextProvider): Keywords matched (intent_detected): {intent_detected}", file=sys.stderr)

        if full_input_query and intent_detected:
            print("DEBUG (GraphQLContextProvider): Detected potential Sitecore GraphQL query intent. Performing semantic search for sample queries.", file=sys.stderr)
            
            if self.sitecore_graphql_server_url and self.sitecore_graphql_auth_token:
                # Perform semantic search on sample queries
                relevant_sample_queries = await self._query_vector_db(full_input_query)
                
                print(f"DEBUG (GraphQLContextProvider): Results from _query_vector_db (relevant_sample_queries count): {len(relevant_sample_queries)}", file=sys.stderr)

                if relevant_sample_queries:
                    context_items.append(
                        ContextItem(
                            name="Relevant_GraphQL_Query_Samples",
                            content="Here are some relevant Sitecore GraphQL query examples:\n\n" + "\n\n".join(relevant_sample_queries),
                            description="Semantically similar sample GraphQL queries from the project, to guide LLM in query generation."
                        )
                    )
                    print(f"DEBUG (GraphQLContextProvider): Added {len(relevant_sample_queries)} relevant GraphQL query samples to context.", file=sys.stderr)
                else:
                    context_items.append(
                        ContextItem(
                            name="Relevant_GraphQL_Query_Samples_Status",
                            content="No highly relevant sample GraphQL queries found for your request.",
                            description="Indicates that semantic search did not yield highly relevant GraphQL query examples."
                        )
                    )
                    print("DEBUG (GraphQLContextProvider): No highly relevant GraphQL query samples found.", file=sys.stderr)
                
                context_items.append(
                    ContextItem(
                        name="GraphQL_Query_Generation_Instruction",
                        content=(
                            "User wants to get Sitecore item/component/template data using GraphQL. " 
                            "Based on the user's request and the `Relevant_GraphQL_Query_Samples` (if provided), "
                            "please generate a suitable GraphQL query. "
                            "Once generated, show the query to the user and **ask for confirmation** before suggesting its execution. "
                            "If the user then types `execute-graphql-cmd YOUR_GRAPHQL_QUERY`, the provider will directly execute it and provide the raw response. "
                            "**After receiving a `Sitecore_GraphQL_Execution_Result` context item, please analyze the JSON response and provide a human-readable explanation or summary of the data, in addition to displaying the raw JSON if appropriate.**" # Clarified instruction for explanation
                        ),
                        description="Instructions for the main LLM to generate and confirm a GraphQL query."
                    )
                )

            else:
                context_items.append(
                    ContextItem(
                        name="Sitecore_GraphQL_Setup_Warning",
                        content="Sitecore GraphQL proxy URL or auth token not configured. Automatic GraphQL queries are unavailable. Please set MATE_SITECORE_GRAPHQL_ENDPOINT and MATE_SITECORE_GRAPHQL_AUTH_TOKEN environment variables.",
                        description="Warning about incomplete Sitecore GraphQL setup."
                    )
                )
        else:
            print("DEBUG (SitecoreGraphQLContextProvider): No explicit Sitecore GraphQL query intent detected in the input. Skipping semantic search.", file=sys.stderr)
            context_items.append(
                ContextItem(
                    name="Sitecore_GraphQL_Provider_Status",
                    content="Sitecore GraphQL Context Provider is active. Ask questions about Sitecore items, components, or templates (e.g., 'What are the details for the home item?', 'Get HeroBanner component details', 'Show template details for X') to trigger relevant GraphQL query samples. You can also directly execute a GraphQL query by typing `execute-graphql-cmd YOUR_GRAPHQL_QUERY`.", # Clarified guidance
                    description="Provides guidance on how to use this context provider."
                )
            )
        
        # --- Logging the final context items being sent to the LLM ---
        print("\n--- Final Context Items Sent to LLM by SitecoreGraphQLContextProvider ---", file=sys.stderr)
        for item in context_items:
            print(f"Context Item Name: {item.name}", file=sys.stderr)
            print(f"Context Item Content (first 500 chars):\n{item.content[:500]}{'...' if len(item.content) > 500 else ''}\n", file=sys.stderr)
        print("--- End Final Context Items ---\n", file=sys.stderr)

        return context_items
