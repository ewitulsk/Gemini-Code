import logging
import functools
import os # Import os
import json # Import json
from google.adk.agents import Agent, LlmAgent # Import base Agent/LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.cloud import aiplatform # Needed for init in callback
from dotenv import load_dotenv

# Import the factory function, not the old implementation
from .rag_tool import initialize_rag_tool
from .state import RagState # Keep RagState if needed for type hints or instantiation

load_dotenv()

# Define the state file path (must match main.py)
STATE_FILE = ".rag_state.json" 

# --- Environment Variables for Models ---
# Get model names from environment variables with defaults
RAG_MODEL_ID = os.environ.get("RAG_MODEL", "gemini-2.0-flash") 
MAIN_MODEL_ID = os.environ.get("MAIN_MODEL", "gemini-2.0-flash")

# --- Agent Callback --- 
def before_agent_starts(callback_context: CallbackContext):
    """Callback executed before the main agent starts running."""
    # Access the agent via the _invocation_context (leading underscore)
    agent: LlmAgent = callback_context._invocation_context.agent 
    
    logging.info("Executing before_agent_callback...")
    logging.info(f"Using RAG Model: {RAG_MODEL_ID}, Main Model: {MAIN_MODEL_ID}") # Log models being used

    # --- Load state from file ---
    loaded_state_data = None
    if not os.path.exists(STATE_FILE):
        logging.error(f"State file '{STATE_FILE}' not found. Cannot configure RAG tool.")
        logging.error("Please run 'python src/main.py' first to perform the setup.")
        agent.tools = []
        return
    try:
        with open(STATE_FILE, 'r') as f:
            loaded_state_data = json.load(f)
        logging.info(f"Successfully loaded state from {STATE_FILE}")
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load or parse state file '{STATE_FILE}': {e}")
        agent.tools = []
        return

    # Extract necessary info
    project_id = loaded_state_data.get("project_id")
    location = loaded_state_data.get("location")
    rag_corpus_name = loaded_state_data.get("rag_corpus_name")

    if not all([project_id, location, rag_corpus_name]):
        logging.error("State file is missing required fields (project_id, location, rag_corpus_name).")
        agent.tools = []
        return
        
    # --- Proceed with initialization using loaded state ---

    # 1. Ensure Vertex AI is initialized 
    try:
        logging.info(f"Initializing Vertex AI SDK in callback (Project: {project_id})...")
        aiplatform.init(project=project_id, location=location)
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI in callback: {e}")
        agent.tools = []
        return

    # 2. Check for RAG corpus name (already done during loading, but good practice)
    if not rag_corpus_name:
        # This case should be caught earlier, but safeguard
        logging.warning("RAG corpus name missing after loading state. Skipping RAG tool setup.")
        agent.tools = []
        return

    # 3. Initialize the RAG tool using the dedicated function
    # The RAG_MODEL_ID is needed for the internal model used by the tool
    rag_tool_function = initialize_rag_tool(
        project_id=project_id, 
        location=location, 
        rag_corpus_name=rag_corpus_name, 
        rag_model_id=RAG_MODEL_ID # Pass the correct model ID
    )

    # 4. Add the tool to the agent if initialization was successful
    if rag_tool_function:
        agent.tools = [rag_tool_function]
        logging.info(f"RAG tool '{rag_tool_function.__name__}' dynamically added to agent '{agent.name}'.")
    else:
        logging.warning("Failed to initialize RAG tool. No RAG tool added to agent.")
        agent.tools = []

# --- Agent Definition ---
# Use LlmAgent directly or keep Agent alias if preferred
main_agent = LlmAgent(
    name="coding_assistant",
    model=MAIN_MODEL_ID, # Use MAIN_MODEL_ID here
    instruction="""You are a helpful coding assistant.
You have access to a tool that can search a specific codebase that has been indexed.
Use the 'query_rag_codebase_impl' tool ONLY if the user asks a question specifically about the indexed codebase.
Otherwise, answer general programming questions directly.""",
    description="A coding assistant that can answer general questions and search an indexed codebase.",
    tools=[], # Tools are now added dynamically via the callback
    before_agent_callback=before_agent_starts # Assign the callback
)

# --- ADK Runner Integration (if running with python -m src.coding_agent) ---
# This part allows running the agent directly using Python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting agent via python -m src.coding_agent...")
    
    # Check if the state file exists as an indicator of setup
    if not os.path.exists(STATE_FILE):
        logging.error(f"RAG state file '{STATE_FILE}' not found.")
        logging.error("Please run 'python src/main.py' first to perform the setup.")
    else:
        logging.info(f"Found state file '{STATE_FILE}'. Proceeding to run agent.")
        print("Agent defined. Use 'adk run src.coding_agent' to start the interactive session.")
else:
    # This branch is executed when imported by `adk run`
    # Ensure logging is configured when imported
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(), format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Agent '{main_agent.name}' module loaded, ready for ADK runner.")


# Old print statement - remove or keep for debugging if needed
# print("Basic coding agent defined in src/coding_agent.py") 