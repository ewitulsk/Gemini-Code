import logging
import functools
import os
import json
import threading
import time
import sys # Import sys for exit
from typing import Optional
from google.adk.agents import Agent, LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.cloud import aiplatform
from dotenv import load_dotenv

# Import necessary components from other modules
from .rag_setup import setup_rag_for_directory, _validate_bucket_name_component
from .rag_tool import create_rag_tool_closure, initialize_internal_rag_model
from .background_state import BackgroundRAGState
# Remove unused RagState import
# from .state import RagState 

load_dotenv()

# Remove STATE_FILE constant
# STATE_FILE = ".rag_state.json" 

# --- Environment Variables --- 
# Required (Check and exit if missing)
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
TARGET_DIRECTORY_RAW = os.environ.get("TARGET_DIRECTORY") # Get raw path
PROJECT_IDENTIFIER = os.environ.get("PROJECT_IDENTIFIER") # e.g., 'my-proj'

# Optional with defaults
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
# Use Gemini 1.5 Flash as a default if not specified
RAG_MODEL_ID = os.environ.get("RAG_MODEL_ID", "gemini-1.5-flash-001") # Internal model for RAG tool
MAIN_MODEL_ID = os.environ.get("MAIN_MODEL_ID", "gemini-1.5-flash-001") # Agent's main model

# --- Validate required Environment Variables ---
missing_vars = []
if not PROJECT_ID:
    missing_vars.append("GOOGLE_CLOUD_PROJECT")
if not TARGET_DIRECTORY_RAW:
    missing_vars.append("TARGET_DIRECTORY")
if not PROJECT_IDENTIFIER:
    missing_vars.append("PROJECT_IDENTIFIER")

if missing_vars:
    logging.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
    logging.critical("Please set these variables (e.g., in a .env file) and restart.")
    # Exit here prevents the agent from being defined incorrectly
    # Note: This exit works when running `adk run`, but might behave differently if imported otherwise
    sys.exit(f"Error: Missing environment variables: {', '.join(missing_vars)}") 

# --- Derived Configuration & Validation ---
TARGET_DIRECTORY = os.path.abspath(TARGET_DIRECTORY_RAW)
if not os.path.isdir(TARGET_DIRECTORY):
     logging.critical(f"TARGET_DIRECTORY path is not a valid directory: {TARGET_DIRECTORY}")
     sys.exit(f"Error: Invalid TARGET_DIRECTORY: {TARGET_DIRECTORY}")

if not _validate_bucket_name_component(PROJECT_IDENTIFIER) or not (3 <= len(PROJECT_IDENTIFIER) <= 20):
     logging.critical(f"Invalid PROJECT_IDENTIFIER: '{PROJECT_IDENTIFIER}'. Must be 3-20 lowercase letters, numbers, hyphens.")
     sys.exit(f"Error: Invalid PROJECT_IDENTIFIER: '{PROJECT_IDENTIFIER}'")

# Construct bucket name (using first 8 chars of project_id for uniqueness)
GCS_BUCKET_NAME = f"{PROJECT_IDENTIFIER}-{PROJECT_ID[:8]}-rag-store"
logging.info(f"Using GCS Bucket Name: {GCS_BUCKET_NAME}")

# --- Background Setup Worker --- 
def _run_rag_background_setup(
    background_state: BackgroundRAGState,
    target_dir: str,
    bucket_name: str,
    rag_model_id: str
):
    """Worker function to run RAG setup and internal model initialization."""
    try:
        logging.info(f"Background thread started for RAG setup (Dir: {target_dir}, Bucket: {bucket_name})")
        
        # Phase 1: Setup GCS and RAG Corpus
        # setup_rag_for_directory updates state to 'running_setup' internally
        corpus_name = setup_rag_for_directory(
            local_directory_path=target_dir,
            gcs_bucket_name=bucket_name,
            background_state=background_state # Pass state object
        )

        if not corpus_name:
            # setup_rag_for_directory should have set state to 'failed'
            logging.error("Background thread: RAG setup failed (corpus_name is None).")
            # Ensure state is failed if not already set by the setup function
            if background_state.setup_status != "failed":
                 background_state.set_status_failed("Corpus setup returned None without explicit error.")
            return

        logging.info(f"Background thread: Corpus setup phase complete. Corpus Name: {corpus_name}")
        # Update state to indicate moving to model initialization
        background_state.set_status_running_model_init(corpus_name)

        # Phase 2: Initialize the internal RAG model
        logging.info(f"Background thread: Initializing internal RAG model ({rag_model_id})...")
        internal_model = initialize_internal_rag_model(
            project_id=background_state.project_id,
            location=background_state.location,
            rag_corpus_name=corpus_name, # Use the obtained corpus name
            rag_model_id=rag_model_id
        )

        if internal_model:
            background_state.set_status_complete(internal_model)
            logging.info("Background thread: RAG setup and model initialization complete.")
        else:
            errmsg = "Failed to initialize internal RAG model."
            logging.error(f"Background thread: {errmsg}")
            background_state.set_status_failed(errmsg)

    except Exception as e:
        # Catch any unexpected errors during the whole process
        errmsg = f"Unexpected error in background RAG setup thread: {e}"
        logging.exception(errmsg) # Log traceback
        # Ensure state reflects failure even if previous steps seemed successful
        background_state.set_status_failed(errmsg)


# --- Shared State Instance ---
# This instance will be created in the callback and shared with the tool closure
# and the background thread.
shared_background_state: Optional[BackgroundRAGState] = None

# --- Agent Callback --- 
def before_agent_starts(callback_context: CallbackContext):
    """Callback executed before the main agent starts running."""
    global shared_background_state # Allow modification of the global variable
    agent: LlmAgent = callback_context._invocation_context.agent 
    
    logging.info("Executing before_agent_callback...")
    logging.info(f"Main Model: {MAIN_MODEL_ID}, Internal RAG Model: {RAG_MODEL_ID}")
    logging.info(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    logging.info(f"Target Directory: {TARGET_DIRECTORY}")
    logging.info(f"GCS Bucket: {GCS_BUCKET_NAME}")

    # --- Initialize Vertex AI SDK ---
    # This needs to happen in the main thread before background thread or tool use Vertex AI
    try:
        logging.info(f"Initializing Vertex AI SDK (Project: {PROJECT_ID}, Location: {LOCATION})...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        logging.info("Vertex AI SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Fatal: Failed to initialize Vertex AI SDK: {e}")
        # Prevent agent from starting without SDK
        agent.tools = [] 
        # Optionally raise an exception or exit if critical
        raise RuntimeError(f"Vertex AI SDK initialization failed: {e}") from e

    # --- Create Shared State ---
    shared_background_state = BackgroundRAGState(project_id=PROJECT_ID, location=LOCATION)
    logging.info("BackgroundRAGState instance created.")

    # --- Start Background RAG Setup ---
    # Check if state was successfully created before starting thread
    if shared_background_state:
        setup_thread = threading.Thread(
            target=_run_rag_background_setup,
            args=(shared_background_state, TARGET_DIRECTORY, GCS_BUCKET_NAME, RAG_MODEL_ID),
            daemon=True # Allows main program to exit even if thread is running
        )
        setup_thread.start()
        logging.info("Background RAG setup thread started.")
    else:
        # Should not happen based on current logic, but safeguard
        logging.error("Failed to create shared background state. Cannot start setup thread.")
        agent.tools = []
        return # Exit callback if state creation failed

    # --- Create and Assign RAG Tool ---
    # The tool uses the shared_background_state to check status and access the model
    try:
        rag_tool_function = create_rag_tool_closure(shared_background_state)
        agent.tools = [rag_tool_function]
        logging.info(f"RAG tool '{rag_tool_function.__name__}' dynamically added to agent '{agent.name}'.")
    except Exception as e:
        logging.error(f"Failed to create RAG tool closure: {e}")
        # Proceed without the RAG tool if creation fails
        agent.tools = []


# --- Agent Definition ---
main_agent = LlmAgent(
    name="coding_assistant",
    model=MAIN_MODEL_ID, 
    instruction="""You are a helpful coding assistant.
You have access to a tool that can search a specific codebase that is being indexed in the background.
Use the 'query_rag_codebase_impl' tool ONLY if the user asks a question specifically about the indexed codebase.
If the tool responds with a 'loading' status, inform the user that the codebase indexing is still in progress and they should try again shortly.
If the tool responds with an 'error' status, inform the user that there was a problem accessing the codebase information.
Otherwise, answer general programming questions directly or provide the information from the successful tool call.
If the user asks if the tool is ready or active, simply ask it a question and if the response is not loading or error, then it is read.""",
    description="A coding assistant that can answer general questions and search an indexed codebase (setup runs in background).",
    tools=[], # Tools are added dynamically via the callback
    before_agent_callback=before_agent_starts # Assign the callback
)

# --- ADK Runner Integration ---
# No longer need the __main__ block that checked for the state file
# The necessary checks and setup happen in the callback or at module level

# Ensure logging is configured when imported by `adk run`
# Basic config, level can be overridden by env var ADK_LOG_LEVEL potentially
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(), format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Agent '{main_agent.name}' module loaded, ready for ADK runner.") 