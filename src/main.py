import os
import sys
import logging
import json # Import json
# Remove functools, Agent/tool imports as they are not needed here
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview import rag # Keep rag for setup
# from vertexai.preview.generative_models import GenerativeModel, Tool # Not needed here
from .rag_setup import setup_rag_for_directory, _validate_bucket_name_component
# from .coding_agent import main_agent # Not needed here
# from .rag_tool import query_rag_codebase # Not needed here
from .state import global_rag_state # Import the global state instance

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the state file path (relative to workspace root ideally)
STATE_FILE = ".rag_state.json"

def initialize_vertex_ai(project_id: str, location: str):
    try:
        logging.info(
            f"Initializing Vertex AI SDK for project '{project_id}' in location '{location}'..."
        )
        aiplatform.init(project=project_id, location=location)
        logging.info("Vertex AI SDK initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI SDK: {e}")
        print(
            f"Error: Could not connect to Vertex AI. Please ensure your environment is configured correctly."
        )
        return False

# Update function to modify the global_rag_state
def get_user_inputs():
    state = global_rag_state # Work with the global instance
    user_identifier = ""
    while not user_identifier:
        user_input = input(
            "Enter a unique identifier for this project (e.g., 'my-proj', used for GCS bucket): "
        ).strip()
        user_identifier = user_input.lower().replace(" ", "-")
        if not _validate_bucket_name_component(user_identifier):
            print(
                "Invalid identifier. Use only lowercase letters, numbers, hyphens."
            )
            user_identifier = ""
        elif len(user_identifier) < 3 or len(user_identifier) > 20:
            print("Identifier must be between 3 and 20 characters.")
            user_identifier = ""
    state.gcs_bucket_name = f"{user_identifier}-{state.project_id[:8]}-rag-store"
    print(f"Using GCS Bucket: gs://{state.gcs_bucket_name}")
    while not state.target_directory:
        dir_input = input(
            "Enter the full path to the code directory you want to work on: "
        ).strip()
        abs_path = os.path.abspath(dir_input)
        if os.path.isdir(abs_path):
            state.target_directory = abs_path
            logging.info(f"Target directory set to: {state.target_directory}")
        else:
            print(f"Error: Directory not found or invalid: '{abs_path}'")

# Update function to modify the global_rag_state
def run_rag_setup_and_save_state() -> bool:
    state = global_rag_state # Work with the global instance
    if not state.is_setup_complete():
        logging.error("Cannot run RAG setup, initial state is incomplete.")
        print("Error: Missing project ID, bucket name, or target directory.")
        return False
    print(f"\nStarting RAG setup for directory: {state.target_directory}")
    print(f"This will upload files to gs://{state.gcs_bucket_name} and index them.")
    print("This process can take some time, please wait...")
    try:
        corpus_name = setup_rag_for_directory(state.target_directory,
                                              state.gcs_bucket_name)
        if corpus_name:
            state.rag_corpus_name = corpus_name
            logging.info(
                f"RAG setup successful. Corpus Name stored: {state.rag_corpus_name}"
            )
            print(f"RAG setup complete! Corpus Name: {state.rag_corpus_name}")
            
            # --- Save state to file ---
            state_data = {
                "project_id": state.project_id,
                "location": state.location,
                "gcs_bucket_name": state.gcs_bucket_name,
                "target_directory": state.target_directory,
                "rag_corpus_name": state.rag_corpus_name
            }
            try:
                with open(STATE_FILE, 'w') as f:
                    json.dump(state_data, f, indent=4)
                logging.info(f"Saved RAG state to {STATE_FILE}")
                print(f"Successfully saved state to {STATE_FILE}")
            except IOError as e:
                logging.error(f"Failed to save state to {STATE_FILE}: {e}")
                print(f"Error: Could not write state file {STATE_FILE}. Agent may not work correctly.")
                # Still return True as RAG setup itself succeeded, but warn user
                return True 
            
            return True
        else:
            logging.error("RAG setup function returned failure (no corpus name).")
            print("RAG setup failed. Please check the logs above for details.")
            return False
    except ValueError as e:
        logging.error(f"RAG Setup Error: {e}")
        print(f"RAG Setup Error: {e}")
        return False
    except Exception as e:
        logging.exception("An unexpected error occurred during RAG setup:")
        print(f"An unexpected error occurred during RAG setup: {e}")
        return False

# Remove agent_interaction_loop function
# def agent_interaction_loop(state: RagState):
#     ...

def main():
    print("--- RAG Setup Utility ---")
    state = global_rag_state # Use global state
    if not state.project_id:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable must be set.")
        sys.exit(1)
    print("Ensure you have authenticated with Google Cloud: gcloud auth application-default login")
    print("Ensure dependencies are installed: pip install -r requirements.txt google-adk")
    
    if not initialize_vertex_ai(state.project_id, state.location):
        sys.exit(1)
        
    get_user_inputs() # Modifies global_rag_state directly
    
    if not run_rag_setup_and_save_state(): # Modifies global_rag_state
        print("RAG setup failed. Cannot proceed.")
        sys.exit(1)

    print(f"\nRAG Setup is complete and state saved to {STATE_FILE}.") # Updated message
    # Display saved state details for confirmation
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                saved_state = json.load(f)
                print("  State details from file:")
                for key, value in saved_state.items():
                    print(f"    {key}: {value}")
        except Exception as e:
             print(f"  (Could not read back state file for verification: {e})")
    else:
         print("  (State file was not created successfully)")

    print("\nYou can now run the agent using:")
    # Removed python -m option as adk run is the standard
    print(f"  adk run src.coding_agent") 
    print("--- RAG Setup Utility Finished ---")


if __name__ == "__main__":
    main()
