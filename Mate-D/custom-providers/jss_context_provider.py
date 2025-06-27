import os
import json
import re
from typing import List, Dict, Any, Optional
import sys
import shutil # Added for directory removal

# --- New Imports for Vector DB and Embeddings ---
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- End New Imports ---

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
        self.initialized_db = False # Flag to track if DB and model are initialized
        # Retrieve API key from environment variable (recommended for security)
        self.api_key = os.getenv("GOOGLE_API_KEY") 
        
        # --- NEW DEBUG PRINT: Check API Key status at initialization ---
        if self.api_key:
            print("DEBUG (ContextProvider.__init__): GOOGLE_API_KEY is set.", file=sys.stderr)
        else:
            print("WARNING (ContextProvider.__init__): GOOGLE_API_KEY environment variable NOT set.", file=sys.stderr)


    async def _init_vector_db_and_model(self, workspace_dir: str):
        """Initializes the embedding model and ChromaDB client/collection."""
        print("DEBUG (ContextProvider): Entering _init_vector_db_and_model...", file=sys.stderr)
        if self.initialized_db:
            print("DEBUG (ContextProvider): Vector DB already initialized. Skipping.", file=sys.stderr)
            return

        if not self.api_key:
            print("ERROR (ContextProvider): Google API Key is missing. Cannot initialize embedding model. Exiting _init_vector_db_and_model.", file=sys.stderr) # More explicit error
            return 

        try:
            print("DEBUG (ContextProvider): Attempting genai.configure() and model configuration...", file=sys.stderr) # NEW DEBUG PRINT
            genai.configure(api_key=self.api_key)
            print("DEBUG (ContextProvider): Google Embedding model configured (via genai.configure).", file=sys.stderr)


            # ChromaDB will persist to a .chroma_db directory in your workspace
            db_path = os.path.join(workspace_dir, ".chroma_db")
            os.makedirs(db_path, exist_ok=True) # Ensure directory exists
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection("jss_code_chunks")
            print(f"DEBUG (ContextProvider): ChromaDB initialized at: {db_path}", file=sys.stderr)
            self.initialized_db = True
            print("DEBUG (ContextProvider): Vector DB and embedding model initialization COMPLETE.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR (ContextProvider): Failed to initialize vector DB or embedding model: {e}", file=sys.stderr)
            import traceback # Import for detailed error
            traceback.print_exc(file=sys.stderr) # Print full traceback
            self.initialized_db = False # Ensure flag is reset on failure


    async def provide_context_items(self, workspace_dir: str, full_input_query: str = "") -> List[ContextItem]:
        print("DEBUG (ContextProvider): Entering provide_context_items...", file=sys.stderr) # New trace point
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
                    description="Indicates a problem with vector database setup (check GOOGLE_API_KEY and terminal logs)."
                )
            )
            # Continue without semantic search capabilities if DB init fails
        else:
            # --- Index Codebase if empty ---
            if self.chroma_collection.count() == 0:
                print("DEBUG (ContextProvider): ChromaDB collection empty, starting indexing...", file=sys.stderr)
                await self._index_codebase(workspace_dir)
                if self.chroma_collection.count() > 0:
                    print(f"DEBUG (ContextProvider): Indexing complete. Total chunks: {self.chroma_collection.count()}", file=sys.stderr)
                else:
                    print("DEBUG (ContextProvider): Indexing attempted, but no chunks were added to ChromaDB.", file=sys.stderr)
            else:
                print(f"DEBUG (ContextProvider): ChromaDB collection already contains {self.chroma_collection.count()} chunks. Skipping indexing.", file=sys.stderr)
            
            context_items.append(
                ContextItem(
                    name="Vector_DB_Status",
                    content=f"Vector database initialized with {self.chroma_collection.count()} chunks.",
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
        # This will now return a map of component_name -> file_path
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
            if self.initialized_db and full_input_query and self.chroma_collection.count() > 0:
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
        Indexes the codebase by chunking files and storing their embeddings in ChromaDB.
        """
        if not self.initialized_db or not self.chroma_collection:
            print("ERROR (ContextProvider): Vector DB or embedding model not initialized. Cannot index codebase.", file=sys.stderr)
            return

        print("DEBUG (ContextProvider): Starting codebase indexing...", file=sys.stderr)
        
        # Configure text splitter for code
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="ts", # Corrected from "typescript" to "ts"
            chunk_size=1000,
            chunk_overlap=100
        )

        files_to_index = []
        src_dir = os.path.join(workspace_dir, "src")
        
        # List of directories to skip during indexing
        # Make sure these are relative to the 'src' directory or absolute if preferred
        excluded_dirs = [
            os.path.join(workspace_dir, 'node_modules'),
            os.path.join(workspace_dir, '.next'),
            os.path.join(workspace_dir, 'dist'),
            os.path.join(workspace_dir, 'build'),
            os.path.join(workspace_dir, '.chroma_db'), # Exclude our own DB folder
            os.path.join(workspace_dir, '.git'),
            os.path.join(workspace_dir, '.vscode'),
            os.path.join(workspace_dir, 'out'), # Common Next.js export folder
            os.path.join(workspace_dir, 'coverage'), # Test coverage reports
            os.path.join(workspace_dir, 'storybook-static'), # Storybook build output
        ]

        if os.path.isdir(src_dir):
            for root, dirs, files in os.walk(src_dir, topdown=True):
                # Modify 'dirs' in-place to skip directories
                # This prevents os.walk from descending into excluded directories
                # Filter out directories whose full path starts with an excluded_dir
                dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(excluded_dir) for excluded_dir in excluded_dirs)]

                for file in files:
                    filepath = os.path.join(root, file)
                    # Also check if the file path itself is within an excluded directory
                    # This is a redundant check if dirs[:] filtering works, but good as a fallback
                    if any(excluded_dir in filepath for excluded_dir in excluded_dirs):
                        print(f"DEBUG (ContextProvider): Skipping excluded file: {filepath}", file=sys.stderr)
                        continue

                    # Filter for common code/config file extensions
                    if file.endswith((".tsx", ".jsx", ".ts", ".js", ".css", ".scss", ".json", ".graphql", ".gql", ".yml", ".yaml", ".md")): # Added .md for documentation
                        files_to_index.append(filepath)
        
        indexed_count = 0
        for filepath in files_to_index:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                component_name = os.path.splitext(os.path.basename(filepath))[0]
                # The heuristic for component name extraction currently relies on "src\\components"
                # This may need adjustment if you have components outside this directory or
                # if you want to extract names from other file types (e.g., GraphQL files).
                if "src\\components" in filepath:
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
                print(f"DEBUG (ContextProvider): Chunked {filepath} into {len(chunks)} parts.", file=sys.stderr)

                chunk_embeddings = []
                chunk_metadatas = []
                chunk_ids = []

                for i, chunk in enumerate(chunks):
                    try:
                        embedding_response = genai.embed_content( # Removed 'await'
                            model='text-embedding-004', # Specify the model explicitly
                            content=chunk
                        )
                        embedding = embedding_response['embedding'] # Access the 'embedding' key from the response
                        
                        chunk_embeddings.append(embedding)
                        chunk_metadatas.append({
                            "filepath": filepath,
                            "filename": os.path.basename(filepath),
                            "component_name": component_name,
                            "chunk_index": i,
                            "source_type": "code"
                        })
                        chunk_ids.append(f"{filepath}_{i}")
                    except Exception as e:
                        print(f"ERROR (ContextProvider): Failed to embed chunk {i} from {filepath}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        continue

                if chunk_embeddings:
                    self.chroma_collection.add(
                        documents=chunks,
                        embeddings=chunk_embeddings,
                        metadatas=chunk_metadatas,
                        ids=chunk_ids
                    )
                    indexed_count += len(chunk_embeddings)
                    print(f"DEBUG (ContextProvider): Added {len(chunk_embeddings)} chunks from {filepath} to ChromaDB.", file=sys.stderr)

            except Exception as e:
                print(f"ERROR (ContextProvider): Failed to read or process file {filepath} for indexing: {e}", file=sys.stderr)
        
        print(f"DEBUG (ContextProvider): Finished indexing. Total new chunks added: {indexed_count}", file=sys.stderr)


    async def _query_vector_db(self, query_text: str) -> List[str]:
        """
        Queries the vector database for semantically similar code snippets.
        """
        if not self.initialized_db or not self.chroma_collection: # Removed self.embedding_model from this check
            print("ERROR (ContextProvider): Vector DB or embedding model not initialized. Cannot query codebase.", file=sys.stderr)
            return []

        try:
            query_embedding_response = genai.embed_content( # Removed 'await'
                model='text-embedding-004', # Specify the model explicitly
                content=query_text
            )
            query_embedding = query_embedding_response['embedding'] # Access the 'embedding' key from the response

            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5, # Retrieve top 5 most relevant results
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                # Filter out less relevant results based on a threshold (adjust as needed)
                # Lower distance means higher similarity. Max distance is 1 (cosine similarity)
                if distance < 0.5: # Example threshold, adjust based on experiment
                    formatted_results.append(
                        f"File: {metadata.get('filepath', 'N/A')}\n"
                        f"Component: {metadata.get('component_name', 'N/A')}\n"
                        f"Chunk (Distance: {distance:.4f}):\n"
                        f"```typescript\n{doc_content}\n```"
                    )
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
                self.initialized_db = False # Reset flag so it re-initializes
                self.chroma_client = None # Clear client to ensure new one is created
                self.chroma_collection = None # Clear collection
                print("DEBUG (ContextProvider): ChromaDB index cleared successfully.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR (ContextProvider): Failed to clear ChromaDB index: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"DEBUG (ContextProvider): No ChromaDB index found at {db_path} to clear.", file=sys.stderr)
