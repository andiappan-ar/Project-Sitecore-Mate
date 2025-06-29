import os
import json
import re
from typing import List, Dict, Any, Optional
import sys
import shutil
import asyncio 

# --- Imports for Local Embeddings ---
# CHANGED: Reverted to SentenceTransformer for Jina V2 Embeddings
from sentence_transformers import SentenceTransformer 
# If you decide to use fastembed for potentially faster CPU inference, uncomment the line below:
# from fastembed import TextEmbedding

# Removed nomic as Jina is now the chosen embedding method
# import nomic 

# --- Existing Imports for Vector DB and Text Splitters ---
import chromadb
# FIX: Corrected import statement for langchain_text_splitters (added 's' to splitter)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- End Existing Imports ---


# continue.dev specific imports for custom context providers
try:
    from continuedev.src.meadow.context_provider import ContextProvider
    from continuedev.src.meadow.context_provider import ContextItem, ContextItemType
    from continuedev.src.meadow.schema import RangeInFileWith
    from continuedev.src.meadow.schema import RangeInFile, Position
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
    class LoadContextItemsArgs:
        pass
    class RangeInFile:
        pass
    class Position:
        pass


class JSSContextProvider(ContextProvider):
    """
    A custom context provider for Sitecore JSS projects.
    It detects project-level conventions like JSS version, styling framework (Tailwind),
    and now intelligently sends specific component code content based on the query.
    """
    
    name: str = "Sitecore JSS Project Context"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chroma_client = None
        self.chroma_collection = None
        self.embedding_model = None # To store our local embedding model (SentenceTransformer instance)
        # CHANGED: Using Jina's code-specific embedding model
        self.embedding_model_name = "jinaai/jina-embeddings-v2-base-code" 
        self.initialized_db = False # Flag to track if DB and model are initialized

        # Ensure use_fastembed is False to leverage SentenceTransformer/GPU
        self.use_fastembed = False 

    async def _init_vector_db_and_model(self, workspace_dir: str):
        """Initializes the embedding model and ChromaDB client/collection."""
        print("DEBUG (ContextProvider): Entering _init_vector_db_and_model...", file=sys.stderr)
        if self.initialized_db:
            print("DEBUG (ContextProvider): Vector DB already initialized. Skipping.", file=sys.stderr)
            return

        try:
            # ChromaDB will persist to a .chroma_db directory in your workspace
            db_path = os.path.join(workspace_dir, ".chroma_db")
            os.makedirs(db_path, exist_ok=True) # Ensure directory exists
            print(f"DEBUG (ContextProvider): ChromaDB client attempting to connect to: {db_path}", file=sys.stderr)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Check if collection exists. If not, it's truly a fresh start.
            try:
                self.chroma_collection = self.chroma_client.get_collection("jss_code_chunks")
                print("DEBUG (ContextProvider): Existing ChromaDB collection 'jss_code_chunks' found.", file=sys.stderr)
            except Exception as e:
                print(f"DEBUG (ContextProvider): Collection 'jss_code_chunks' not found or error accessing: {e}. Creating new one.", file=sys.stderr)
                self.chroma_collection = self.chroma_client.create_collection("jss_code_chunks")
                print("DEBUG (ContextProvider): New ChromaDB collection 'jss_code_chunks' created.", file=sys.stderr)

            # CHANGED: Load Jina embeddings using SentenceTransformer
            if self.use_fastembed:
                 # This branch is currently not active as use_fastembed is False
                print("DEBUG (ContextProvider): Attempting to load using fastembed (not active)...", file=sys.stderr)
                # self.embedding_model = TextEmbedding(model_name=self.embedding_model_name)
            else:
                print(f"DEBUG (ContextProvider): Attempting to load '{self.embedding_model_name}' using SentenceTransformer (will use GPU if CUDA is active)...", file=sys.stderr)
                # SentenceTransformer automatically tries to use a GPU (CUDA) if available
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"DEBUG (ContextProvider): SentenceTransformer model '{self.embedding_model_name}' loaded successfully (should use GPU).", file=sys.stderr)

            self.initialized_db = True
            print("DEBUG (ContextProvider): Vector DB and embedding model initialization COMPLETE.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR (ContextProvider): Failed to initialize vector DB or embedding model: {e}", file=sys.stderr)
            import traceback 
            traceback.print_exc(file=sys.stderr) 
            self.initialized_db = False 


    async def process_single_chunk_embedding(self, chunk_content: str, details: Dict[str, Any]):
        """
        Helper function to embed a single chunk asynchronously using the Jina model via SentenceTransformer.
        This uses asyncio.to_thread to run the blocking embedding model call
        in a separate thread, preventing the main event loop from blocking.
        """
        if not self.embedding_model:
            print("ERROR (ContextProvider): Embedding model not initialized in process_single_chunk_embedding.", file=sys.stderr)
            return None, None, None

        try:
            print(f"DEBUG (ContextProvider): Embedding chunk with Jina ({self.embedding_model_name}): {chunk_content[:50]}...", file=sys.stderr)
            
            # SentenceTransformer.encode handles single strings directly and will use GPU if CUDA is active
            embedding = await asyncio.to_thread(self.embedding_model.encode, chunk_content, convert_to_tensor=False)
            embedding = embedding.tolist() # Ensure it's a standard Python list for ChromaDB
            
            return chunk_content, embedding, details
        except Exception as e:
            print(f"ERROR (ContextProvider): Failed to embed chunk from {details.get('filepath', 'N/A')} (chunk {details.get('chunk_index', 'N/A')}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None, None, None


    async def provide_context_items(self, workspace_dir: str, full_input_query: str = "") -> List[ContextItem]:
        print("DEBUG (ContextProvider): Entering provide_context_items...", file=sys.stderr)
        print(f"DEBUG (ContextProvider): Received workspace_dir: {workspace_dir}", file=sys.stderr)

        context_items: List[ContextItem] = []
        
        if not workspace_dir:
            print("CRITICAL ERROR (ContextProvider): Workspace directory is missing or empty. Cannot provide JSS context.", file=sys.stderr)
            return []

        print(f"Scanning JSS project at: {workspace_dir}")

        # --- Initialize Vector DB and Embedding Model ---
        await self._init_vector_db_and_model(workspace_dir)
        
        # Check initialization status AFTER attempting initialization
        if not self.initialized_db:
            context_items.append(
                ContextItem(
                    name="Vector_DB_Status",
                    content="Vector database and embedding model failed to initialize. Semantic search unavailable.",
                    description="Indicates a problem with vector database setup (check terminal logs)."
                )
            )
        else:
            # Check current chunk count to decide if initial indexing is needed
            current_chunk_count = self.chroma_collection.count()
            if current_chunk_count == 0:
                print("DEBUG (ContextProvider): ChromaDB collection is empty. Performing initial full indexing...", file=sys.stderr)
                await self._index_codebase(workspace_dir)
                # Update chunk count after indexing
                current_chunk_count = self.chroma_collection.count()
                if current_chunk_count > 0:
                    print(f"DEBUG (ContextProvider): Initial full indexing complete. Total chunks in DB: {current_chunk_count}", file=sys.stderr)
                else:
                    print("DEBUG (ContextProvider): Initial full indexing attempted, but no chunks were added to ChromaDB. (Final count: 0)", file=sys.stderr)
            else:
                print(f"DEBUG (ContextProvider): ChromaDB collection already contains {current_chunk_count} chunks. Skipping indexing.", file=sys.stderr)
            
            context_items.append(
                ContextItem(
                    name="Vector_DB_Status",
                    content=f"Vector database initialized with {current_chunk_count} chunks.",
                    description="Status of the vector database."
                )
            )

        # --- Detect Project Context (JSS Version, Tailwind) ---
        jss_version = self._detect_jss_version(workspace_dir)
        if jss_version:
            context_items.append(
                ContextItem(
                    name="JSS_Version",
                    content=f"Sitecore JSS Version: {jss_version}",
                    description=f"Detected Sitecore JSS version: {jss_version}"
                )
            )
        else:
            context_items.append(
                ContextItem(
                    name="JSS_Version",
                    content="Sitecore JSS Version: Not detected. Check package.json in project root.",
                    description="Could not detect Sitecore JSS version in package.json."
                )
            )

        uses_tailwind = self._detect_tailwind(workspace_dir)
        if uses_tailwind:
            context_items.append(
                ContextItem(
                    name="Styling_Framework",
                    content="Project uses Tailwind CSS for styling.",
                    description="Detected Tailwind CSS configuration (tailwind.config.js)"
                )
            )
        else:
            context_items.append(
                ContextItem(
                    name="Styling_Framework",
                    content="Tailwind CSS not detected. Assuming other CSS approach or no specific framework.",
                    description="Tailwind CSS configuration not found."
                )
            )

        # --- Scan all components to get their names and paths ---
        all_component_paths = self._get_all_component_file_paths(workspace_dir)
        component_names = sorted(list(all_component_paths.keys()))

        if component_names:
            context_items.append(
                ContextItem(
                    name="All_JSS_Component_Names",
                    content="Known JSS Components in project:\n- " + "\n- ".join(component_names),
                    description="List of all detected JSS component names."
                )
            )
        else:
            context_items.append(
                ContextItem(
                    name="All_JSS_Component_Names_Status",
                    content="No JSS component code files found in standard directories (src/components).",
                    description="Indicates that no React/Vue/Angular JSS component code files were found."
                )
            )

        # --- Determine if a specific component's code needs to be sent ---
        targeted_component_name = self._find_component_in_query(full_input_query, component_names)
        
        if targeted_component_name and targeted_component_name in all_component_paths:
            component_filepath = all_component_paths[targeted_component_name]
            try:
                with open(component_filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    context_items.append(
                        ContextItem(
                            name=f"Code_for_{targeted_component_name}",
                            content=f"Component Name: {targeted_component_name}\nComponent File Content:\n```typescript\n{file_content}\n```",
                            description=f"Full code content for the {targeted_component_name} component, requested by the query."
                        )
                    )
                print(f"DEBUG (ContextProvider): Added full code for specific component: {targeted_component_name}", file=sys.stderr)
            except Exception as e:
                print(f"ERROR (ContextProvider): Could not read file content for {targeted_component_name} at {component_filepath}: {e}", file=sys.stderr)
        elif targeted_component_name:
            print(f"DEBUG (ContextProvider): Targeted component '{targeted_component_name}' not found in scanned paths.", file=sys.stderr)
        else:
            print(f"DEBUG (ContextProvider): No specific component detected in query to send full code. Attempting codebase search.", file=sys.stderr)
            # --- Perform semantic search if no specific component is targeted ---
            if self.initialized_db and self.embedding_model and full_input_query and self.chroma_collection.count() > 0: 
                print(f"DEBUG (ContextProvider): Performing semantic search for query: '{full_input_query}'", file=sys.stderr)
                search_results = await self._query_vector_db(full_input_query)
                if search_results:
                    context_items.append(
                        ContextItem(
                            name="Codebase_Search_Results",
                            content="Relevant Codebase Snippets (from semantic search):\n" + "\n---\n".join(search_results),
                            description="Code snippets semantically related to the user's query from the project codebase."
                        )
                    )
                    print(f"DEBUG (ContextProvider): Added {len(search_results)} codebase search results.", file=sys.stderr)
                else:
                    print("DEBUG (ContextProvider): Semantic search found no relevant results.", file=sys.stderr)


        # --- Standard package.json context ---
        package_json_path = os.path.join(workspace_dir, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    context_items.append(
                        ContextItem(
                            name="package.json",
                            content=content,
                            description="Contents of package.json for project dependencies and scripts."
                        )
                    )
            except Exception as e:
                print(f"ERROR (ContextProvider): Could not read package.json at {package_json_path}: {e}", file=sys.stderr)
        else:
            print(f"DEBUG (ContextProvider): package.json not found at {package_json_path}", file=sys.stderr)
            context_items.append(
                ContextItem(
                    name="package.json_status",
                    content=f"package.json not found in the detected workspace directory: {workspace_dir}.",
                    description="Indicates that package.json was not found, which is essential for JSS context."
                )
            )

        return context_items

    def _detect_jss_version(self, workspace_dir: str) -> Optional[str]:
        """Detects the Sitecore JSS version from package.json."""
        package_json_path = os.path.join(workspace_dir, "package.json")
        if not os.path.exists(package_json_path):
            print(f"DEBUG (ContextProvider): package.json not found by _detect_jss_version at {package_json_path}", file=sys.stderr)
            return None
        
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
                dependencies = package_data.get("dependencies", {})
                dev_dependencies = package_data.get("devDependencies", {})
                
                jss_deps_to_check = [
                    "@sitecore-jss/sitecore-jss-nextjs",
                    "@sitecore-jss/sitecore-jss-react",
                    "@sitecore-jss/sitecore-jss",
                    "@sitecore-jss/sitecore-jss-angular",
                    "@sitecore-jss/sitecore-jss-vue"
                ]
                for dep_name in jss_deps_to_check:
                    if dep_name in dependencies:
                        version = dependencies[dep_name]
                        print(f"DEBUG (ContextProvider): Found JSS dependency '{dep_name}' version '{version}' in {package_json_path}")
                        return version.lstrip('^~=')
                    if dep_name in dev_dependencies:
                        version = dev_dependencies[dep_name]
                        print(f"DEBUG (ContextProvider): Found JSS devDependency '{dep_name}' version '{version}' in {package_json_path}")
                        return version.lstrip('^~=')
                print(f"DEBUG (ContextProvider): No known JSS dependencies found in package.json at {package_json_path}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"ERROR (ContextProvider): Invalid JSON in package.json at {package_json_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR (ContextProvider): Unexpected error reading package.json for JSS version at {package_json_path}: {e}", file=sys.stderr)
        return None

    def _detect_tailwind(self, workspace_dir: str) -> bool:
        """Detects if Tailwind CSS is used by checking for tailwind.config.js or tailwind.config.ts."""
        tailwind_config_path_js = os.path.join(workspace_dir, "tailwind.config.js")
        tailwind_config_path_ts = os.path.join(workspace_dir, "tailwind.config.ts")
        
        uses_tailwind = os.path.exists(tailwind_config_path_js) or os.path.exists(tailwind_config_path_ts)
        if not uses_tailwind:
            print(f"DEBUG (ContextProvider): Tailwind config not found by _detect_tailwind in {workspace_dir}", file=sys.stderr)
        return uses_tailwind

    def _get_all_component_file_paths(self, workspace_dir: str) -> Dict[str, str]:
        """
        Scans standard JSS component code directories (src/components) and
        returns a dictionary mapping clean component names to their full file paths.
        Does NOT read file content here.
        """
        component_code_dir = os.path.join(workspace_dir, "src", "components")
        component_name_to_path = {}

        print(f"DEBUG (ContextProvider): Scanning for all component files in: {component_code_dir}", file=sys.stderr)

        if not os.path.isdir(component_code_dir):
            print(f"DEBUG (ContextProvider): Component code directory does NOT exist or is not a directory: {component_code_dir}", file=sys.stderr)
            return {}

        for root, _, files in os.walk(component_code_dir):
            for file in files:
                if file.endswith((".tsx", ".jsx")):
                    filepath = os.path.join(root, file)
                    component_name = os.path.splitext(file)[0] # Default to filename

                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Attempt to refine component name from export default, if cleaner
                            match_export_default = re.search(r'export default\s+(.*?)(?:;|\n|$)', content)
                            if match_export_default:
                                name_candidate = match_export_default.group(1).strip()
                                # Handle `export default withDatasourceCheck()(MyComponent);`
                                inner_match = re.search(r'\(\)\((.*?)\)', name_candidate)
                                if inner_match:
                                    component_name_refined = inner_match.group(1).strip()
                                    component_name = component_name_refined
                                # Handle `export default withDatasourceCheck<Props>(MyComponent);`
                                elif re.search(r'^[A-Za-z0-9_]+\s*<[^>]+>\s*\(', name_candidate):
                                    inner_match = re.search(r'\(([^)]+)\)$', name_candidate)
                                    if inner_match:
                                        component_name_refined = inner_match.group(1).strip()
                                        component_name = component_name_refined
                                else:
                                    # If it's a simple export default, ensure it's a valid component name (starts with uppercase)
                                    if name_candidate and name_candidate[0].isupper():
                                        component_name = name_candidate
                            
                            component_name_to_path[component_name] = filepath

                    except Exception as e:
                        print(f"ERROR (ContextProvider): Error processing component file {filepath} for name extraction: {e}", file=sys.stderr)
        
        return component_name_to_path

    def _find_component_in_query(self, query: str, known_components: List[str]) -> Optional[str]:
        """
        Attempts to find a known component name within the user's query.
        Prioritizes exact matches and longer matches.
        """
        if not query:
            return None

        # Sort known_components by length (longest first) for better matching
        # e.g., 'StyleguideLayout' before 'Layout'
        sorted_components = sorted(known_components, key=len, reverse=True)

        # Create a regex pattern to find any of the component names as whole words
        # using non-capturing groups and word boundaries
        pattern_parts = [re.escape(name) for name in sorted_components]
        # Use case-insensitive matching
        pattern = r'\b(' + '|'.join(pattern_parts) + r')\b'
        
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Return the exact matched component name from the list, preserving its casing
            matched_name = match.group(1)
            for component_name in known_components:
                if component_name.lower() == matched_name.lower():
                    print(f"DEBUG (ContextProvider): Detected specific component '{component_name}' in query.", file=sys.stderr)
                    return component_name
        
        return None

    async def _index_codebase(self, workspace_dir: str):
        """
        Performs a full indexing of the codebase. This function is only called
        when the ChromaDB collection is initially empty.
        """
        if not self.initialized_db or not self.chroma_collection or not self.embedding_model: # Now check embedding_model as it's an object again
            print("ERROR (ContextProvider): Vector DB, Chroma collection, or embedding model not initialized. Cannot index codebase.", file=sys.stderr)
            return

        print(f"DEBUG (ContextProvider): Starting full codebase indexing (ChromaDB was empty) using Jina model: {self.embedding_model_name}...", file=sys.stderr)
        
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="ts", 
            chunk_size=256, 
            chunk_overlap=20
        )

        files_to_scan = []
        src_dir = os.path.join(workspace_dir, "src")
        
        excluded_dirs = [
            os.path.join(workspace_dir, 'node_modules'),
            os.path.join(workspace_dir, '.next'),
            os.path.join(workspace_dir, 'dist'),
            os.path.join(workspace_dir, 'build'),
            os.path.join(workspace_dir, '.chroma_db'), 
            os.path.join(workspace_dir, '.git'),
            os.path.join(workspace_dir, '.vscode'),
            os.path.join(workspace_dir, 'out'), 
            os.path.join(workspace_dir, 'coverage'), 
            os.path.join(workspace_dir, 'storybook-static'), 
        ]

        if os.path.isdir(src_dir):
            for root, dirs, files in os.walk(src_dir, topdown=True):
                dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(excluded_dir) for excluded_dir in excluded_dirs)]

                for file in files:
                    filepath = os.path.join(root, file)
                    if any(excluded_dir in filepath for excluded_dir in excluded_dirs):
                        print(f"DEBUG (ContextProvider): Skipping excluded file: {filepath}", file=sys.stderr)
                        continue

                    if file.endswith((".tsx", ".jsx", ".ts", ".js", ".css", ".scss", ".json", ".graphql", ".gql", ".yml", ".yaml", ".md")):
                        files_to_scan.append(filepath)
        
        indexed_count = 0
        
        chunks_for_embedding_tasks = []

        # In a full index scenario (ChromaDB was empty), all scanned files are considered "new"
        # and need to be chunked and embedded.
        for filepath in files_to_scan:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                component_name = os.path.splitext(os.path.basename(filepath))[0]
                if "src" + os.sep + "components" in filepath:
                    refined_name_match = re.search(r'export default\s+(.*?)(?:;|\n|$)', content)
                    if refined_name_match:
                        name_candidate = refined_name_match.group(1).strip()
                        inner_match = re.search(r'\(\)\((.*?)\)', name_candidate)
                        if inner_match:
                            component_name_refined = inner_match.group(1).strip()
                            component_name = component_name_refined
                        elif re.search(r'^[A-Za-z0-9_]+\s*<[^>]+>\s*\(', name_candidate):
                            inner_match = re.search(r'\(([^)]+)\)$', name_candidate)
                            if inner_match:
                                component_name_refined = inner_match.group(1).strip()
                                component_name = component_name_refined
                        else:
                            if name_candidate and name_candidate[0].isupper():
                                component_name = name_candidate

                chunks = text_splitter.split_text(content)
                current_mtime = os.path.getmtime(filepath) 
                
                print(f"DEBUG (ContextProvider): Chunked {filepath} into {len(chunks)} parts for initial indexing.", file=sys.stderr)

                for i, chunk in enumerate(chunks):
                    details = {
                        "chunk_content": chunk, 
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "component_name": component_name,
                        "chunk_index": i,
                        "source_type": "code",
                        "last_modified_time": current_mtime, 
                        "id": f"{filepath}_{i}"
                    }
                    # Calling process_single_chunk_embedding as a method of self
                    chunks_for_embedding_tasks.append(self.process_single_chunk_embedding(chunk, details))

            except Exception as e:
                print(f"ERROR (ContextProvider): Failed to read or process file {filepath} for indexing: {e}", file=sys.stderr)
        
        print(f"DEBUG (ContextProvider): Gathering {len(chunks_for_embedding_tasks)} embedding tasks concurrently...", file=sys.stderr)
        
        processed_results_from_embedding = []
        ASYNC_EMBEDDING_BATCH_SIZE = 50 
        for i in range(0, len(chunks_for_embedding_tasks), ASYNC_EMBEDDING_BATCH_SIZE):
            batch_tasks = chunks_for_embedding_tasks[i:i + ASYNC_EMBEDDING_BATCH_SIZE]
            print(f"DEBUG (ContextProvider): Processing embedding batch {i // ASYNC_EMBEDDING_BATCH_SIZE + 1} of {len(batch_tasks)} chunks.", file=sys.stderr)
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True) 
            for res in batch_results:
                if isinstance(res, Exception):
                    print(f"ERROR (ContextProvider): An async embedding task failed: {res}", file=sys.stderr)
                else:
                    processed_results_from_embedding.append(res)

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
        
        print(f"DEBUG (ContextProvider): Total chunks prepared for ChromaDB addition (after embedding): {len(all_documents_for_chroma)}", file=sys.stderr)

        CHROMA_ADD_BATCH_SIZE = 5000 
        print(f"DEBUG (ContextProvider): Preparing to add {len(all_documents_for_chroma)} chunks to ChromaDB in batches of {CHROMA_ADD_BATCH_SIZE}...", file=sys.stderr)

        for i in range(0, len(all_documents_for_chroma), CHROMA_ADD_BATCH_SIZE):
            docs_batch = all_documents_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            embeddings_batch = all_embeddings_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            metadatas_batch = all_metadatas_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]
            ids_batch = all_ids_for_chroma[i:i + CHROMA_ADD_BATCH_SIZE]

            try:
                print(f"DEBUG (ContextProvider): Adding ChromaDB batch {i // CHROMA_ADD_BATCH_SIZE + 1} of {len(docs_batch)} chunks.", file=sys.stderr)
                self.chroma_collection.add(
                    documents=docs_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch,
                    ids=ids_batch
                )
                indexed_count += len(docs_batch)
            except Exception as e:
                print(f"ERROR (ContextProvider): Failed to add ChromaDB batch starting with ID {ids_batch[0] if ids_batch else 'N/A'}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue 

        print(f"DEBUG (ContextProvider): Full indexing finished. Total chunks added: {indexed_count}.", file=sys.stderr)


    async def _query_vector_db(self, query_text: str) -> List[str]:
        """
        Queries the vector database for semantically similar code snippets.
        """
        if not self.initialized_db or not self.chroma_collection or not self.embedding_model: # Re-added check for self.embedding_model
            print("ERROR (ContextProvider): Vector DB, Chroma collection, or embedding model not initialized. Cannot query codebase.", file=sys.stderr)
            return []

        try:
            # Query embedding using the Jina model via SentenceTransformer
            print(f"DEBUG (ContextProvider): Generating query embedding with Jina ({self.embedding_model_name}): {query_text[:50]}...", file=sys.stderr)
            query_embedding = await asyncio.to_thread(self.embedding_model.encode, query_text, convert_to_tensor=False)
            query_embedding = query_embedding.tolist() # Ensure it's a standard Python list for ChromaDB
            
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding], 
                n_results=5, # Retrieve top 5 most relevant results
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"\n--- DEBUG: Query Results for '{query_text}' ---", file=sys.stderr)
            if results and results['documents'] and results['documents'][0]:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    print(f"DEBUG Result {i+1}: Distance={distance:.4f}, File={metadata.get('filepath', 'N/A')}, Component={metadata.get('component_name', 'N/A')}", file=sys.stderr)
                    print(f"DEBUG Chunk Content:\n{doc_content[:300]}...\n", file=sys.stderr) # Print first 300 chars of chunk

            formatted_results = []
            if results and results['documents']: 
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]

                    if distance < 0.6: 
                        formatted_results.append(
                            f"File: {metadata.get('filepath', 'N/A')}\n"
                            f"Component: {metadata.get('component_name', 'N/A')}\n"
                            f"Chunk (Distance: {distance:.4f}):\n"
                            f"```typescript\n{doc_content}\n```"
                        )
            print("--- END DEBUG Query Results ---\n", file=sys.stderr)
            return formatted_results
        except Exception as e:
            print(f"ERROR (ContextProvider): Failed to query vector DB: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return []

    async def _analyze_component_details(self, component_path: str) -> Dict[str, Any]:
        print(f"Analyzing component details for {component_path} (placeholder)...")
        return {}

    async def _clear_chroma_index(self, workspace_dir: str):
        """
        Clears the persistent ChromaDB index from the workspace directory.
        This forces a full re-index on the next query.
        """
        db_path = os.path.join(workspace_dir, ".chroma_db")
        if os.path.exists(db_path) and os.path.isdir(db_path):
            print(f"DEBUG (ContextProvider): Clearing ChromaDB index at: {db_path}", file=sys.stderr)
            try:
                shutil.rmtree(db_path)
                self.initialized_db = False 
                self.chroma_client = None 
                self.chroma_collection = None 
                self.embedding_model = None # Reset the model as well
                print("DEBUG (ContextProvider): ChromaDB index cleared successfully.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR (ContextProvider): Failed to clear ChromaDB index: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"DEBUG (ContextProvider): No ChromaDB index found at {db_path} to clear.", file=sys.stderr)
