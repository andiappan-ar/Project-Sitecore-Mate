# llm_config.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import AsyncOpenAI # Changed from OpenAI to AsyncOpenAI for async operations
import ollama # For Ollama API interaction
import asyncio # For running synchronous ollama.pull in an async context

# Load environment variables once at the top
load_dotenv()

# --- Global LLM Client Instance ---
# This will hold the initialized LLM client (Gemini, OpenAI, or Ollama)
_llm_client = None
_llm_provider = None

async def _initialize_llm_client():
    """
    Initializes the appropriate LLM client based on LLM_PROVIDER environment variable.
    This function should be called once at application startup.
    It's now async because ollama.pull can be blocking.
    """
    global _llm_client, _llm_provider

    _llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    print(f"Configured LLM Provider: {_llm_provider.upper()}")

    if _llm_provider == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini provider.")
        genai.configure(api_key=gemini_api_key)
        _llm_client = "configured" # We don't need a client object, just configured the API
        print("Gemini API configured successfully.")
    elif _llm_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not set. OpenAI API calls might fail if authentication is required.")
        _llm_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url) # Changed to AsyncOpenAI
        print(f"OpenAI client initialized. Base URL: {openai_base_url}")
    elif _llm_provider == "ollama":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _llm_client = ollama.Client(host=ollama_base_url)
        print(f"Ollama client initialized. Base URL: {ollama_base_url}")
        # Test Ollama connection and ensure models are pulled
        try:
            ollama_default_model_general = os.getenv("OLLAMA_DEFAULT_MODEL_GENERAL")
            ollama_default_model_summary = os.getenv("OLLAMA_DEFAULT_MODEL_SUMMARY")

            if ollama_default_model_general:
                print(f"Attempting to ensure Ollama general model '{ollama_default_model_general}' is pulled...")
                await asyncio.to_thread(ollama.pull, ollama_default_model_general)
                print(f"Ollama general model '{ollama_default_model_general}' ensured.")
            else:
                print("Warning: OLLAMA_DEFAULT_MODEL_GENERAL not set. Cannot ensure Ollama general model is pulled.")
            
            # Only pull summary model if it's different or explicitly set
            if ollama_default_model_summary and ollama_default_model_summary != ollama_default_model_general:
                print(f"Attempting to ensure Ollama summary model '{ollama_default_model_summary}' is pulled...")
                await asyncio.to_thread(ollama.pull, ollama_default_model_summary)
                print(f"Ollama summary model '{ollama_default_model_summary}' ensured.")
            elif not ollama_default_model_summary:
                print("Warning: OLLAMA_DEFAULT_MODEL_SUMMARY not set. Cannot ensure Ollama summary model is pulled.")

        except Exception as e:
            print(f"Warning: Could not connect to Ollama server or pull default model(s): {e}")
            print("Please ensure Ollama server is running and the model(s) are available.")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {_llm_provider}. Please choose 'gemini', 'openai', or 'ollama'.")

# This is a critical change: _initialize_llm_client needs to be awaited.
# For a FastAPI app, this typically happens during startup.
# We'll make get_llm_model handle the initial await if _llm_client is None.

async def get_llm_model(task_type: str = "general"):
    """
    Returns an LLM model instance/client and the active provider string
    based on the configured provider and task type.
    Model names are strictly loaded from environment variables.
    """
    global _llm_provider, _llm_client

    # Ensure client is initialized (it should be by module import, but defensive check)
    if _llm_client is None:
        # If not initialized, await the initialization
        await _initialize_llm_client()

    if _llm_provider == "gemini":
        # Retrieve Gemini model directly from environment variable based on task_type
        model_name = os.getenv(f"GEMINI_DEFAULT_MODEL_{task_type.upper()}")
        if not model_name:
            raise ValueError(f"GEMINI_DEFAULT_MODEL_{task_type.upper()} environment variable not set for Gemini provider.")
        print(f"Using Gemini model: {model_name} for task: {task_type}")
        return genai.GenerativeModel(model_name), _llm_provider
    elif _llm_provider == "openai":
        # Retrieve OpenAI model directly from environment variable based on task_type
        model_name = os.getenv(f"OPENAI_DEFAULT_MODEL_{task_type.upper()}")
        if not model_name:
            raise ValueError(f"OPENAI_DEFAULT_MODEL_{task_type.upper()} environment variable not set for OpenAI provider.")
        print(f"Using OpenAI model: {model_name} for task: {task_type}")
        # OpenAI's client.chat.completions.create needs model during the call.
        # We return the client itself, and the model_name will be used in main.py.
        return _llm_client, _llm_provider
    elif _llm_provider == "ollama":
        # Retrieve Ollama model directly from environment variable based on task_type
        model_name = os.getenv(f"OLLAMA_DEFAULT_MODEL_{task_type.upper()}")
        if not model_name:
            raise ValueError(f"OLLAMA_DEFAULT_MODEL_{task_type.upper()} environment variable not set for Ollama provider.")
        print(f"Using Ollama model: {model_name} for task: {task_type}")
        return _llm_client, _llm_provider # Ollama client object is directly used for chat/generate
    else:
        raise ValueError(f"Unknown LLM Provider: {_llm_provider}")
