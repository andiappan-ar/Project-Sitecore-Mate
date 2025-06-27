import os
import json
import re
from typing import List, Dict, Any, Optional
import sys

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
    
    async def provide_context_items(self, workspace_dir: str, full_input_query: str = "") -> List[ContextItem]:
        context_items: List[ContextItem] = []
        
        if not workspace_dir:
            print("CRITICAL ERROR (ContextProvider): Workspace directory is missing or empty. Cannot provide JSS context.", file=sys.stderr)
            return []

        print(f"Scanning JSS project at: {workspace_dir}")

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
            print(f"DEBUG (ContextProvider): No specific component detected in query to send full code.", file=sys.stderr)

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

    # --- Placeholder for future methods ---
    async def _index_codebase(self, workspace_dir: str):
        print("Indexing codebase for vector DB (placeholder)...")

    async def _query_vector_db(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        print("Querying vector DB (placeholder)...")
        return []

    async def _analyze_component_details(self, component_path: str) -> Dict[str, Any]:
        print(f"Analyzing component details for {component_path} (placeholder)...")
        return {}

