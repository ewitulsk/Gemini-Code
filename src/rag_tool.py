import logging
from .state import RagState
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from typing import Callable


def create_rag_tool_closure(state: RagState):
    """Factory function that creates the RAG query tool closure.

    Args:
        state: The RagState object containing the initialized RAG model.

    Returns:
        A function that takes a query string and returns the RAG result dictionary.
    """
    
    # This is the actual function that will be registered as a tool
    # It only takes the query parameter, which the LLM needs to provide.
    def query_rag_codebase_impl(query: str) -> dict:
        """Queries the indexed codebase using the configured RAG model.

        Use this tool ONLY when the user asks a question specifically about the codebase
        that has been indexed (e.g., "How is authentication handled?", "Where is the database connection defined?").
        Do not use this for general programming questions or questions about code not in the index.

        Args:
            query: The user's question or search query about the codebase.

        Returns:
            A dictionary containing the status and the RAG model's response.
            Example success: {'status': 'success', 'response': 'The code defines authentication in auth.py...'}
            Example failure: {'status': 'error', 'error_message': 'RAG model not initialized.'}
            Example no_answer: {'status': 'no_answer', 'message': 'The RAG model could not find relevant information.'}
        """
        logging.info(f"RAG Tool closure triggered with query: '{query}'")
        
        # Access state from the enclosing scope (closure)
        if not state or not state.is_rag_ready() or not state.rag_model:
            logging.error("RAG tool closure called but state or RAG model is not ready.")
            return {"status": "error", "error_message": "RAG model not initialized or ready."}

        try:
            logging.info("Sending query to internal RAG model...")
            response = state.rag_model.generate_content(query)

            if response and response.text:
                logging.info("RAG model returned a response.")
                return {"status": "success", "response": response.text}
            else:
                logging.warning("RAG model did not return a text response.")
                return {"status": "no_answer", "message": "The RAG model could not find relevant information for the query."}

        except Exception as e:
            logging.exception(f"Error during RAG model query execution in closure: {e}")
            return {"status": "error", "error_message": f"An error occurred while querying the RAG model: {e}"}

    # Return the inner function (the closure)
    return query_rag_codebase_impl

# New function to initialize the RAG model and create the tool
def initialize_rag_tool(project_id: str, location: str, rag_corpus_name: str, rag_model_id: str) -> Callable | None:
    """Initializes the internal RAG model and creates the RAG tool function.

    Args:
        project_id: Google Cloud Project ID.
        location: Google Cloud resource location (e.g., 'us-central1').
        rag_corpus_name: The full resource name of the RAG Corpus.
        rag_model_id: The identifier for the Gemini model to use internally for RAG.

    Returns:
        The callable RAG tool function (closure) if successful, otherwise None.
    """
    rag_model_instance = None
    try:
        logging.info(f"Initializing internal RAG model for tool (Corpus: {rag_corpus_name}, Model: {rag_model_id})...")
        # Ensure the corpus name is the full resource path if needed, or just the ID if API handles it.
        # Assuming rag.RagResource expects the full name based on current coding_agent.py usage
        rag_resource = rag.RagResource(rag_corpus=rag_corpus_name) 
        rag_retrieval_tool = Tool.from_retrieval(retrieval=rag.Retrieval(
            source=rag.VertexRagStore(rag_resources=[rag_resource],
                                      similarity_top_k=5,
                                      vector_distance_threshold=0.5)))
        
        rag_model_instance = GenerativeModel(rag_model_id, tools=[rag_retrieval_tool])
        logging.info(f"Internal RAG model ({rag_model_id}) for tool initialized successfully.")
    except Exception as e:
        logging.exception(f"Failed to initialize RAG model for tool: {e}")
        return None # Return None on failure

    if rag_model_instance:
        # Create a temporary state object JUST for the closure factory
        # It holds the model and necessary identifiers needed by the closure's logic
        tool_state = RagState() 
        tool_state.rag_model = rag_model_instance
        tool_state.project_id = project_id # Pass necessary info
        tool_state.rag_corpus_name = rag_corpus_name # Pass necessary info
        
        # Call the factory to get the actual tool function (closure)
        rag_tool_function = create_rag_tool_closure(tool_state)
        logging.info(f"RAG tool closure '{rag_tool_function.__name__}' created successfully.")
        return rag_tool_function
    else:
        logging.warning("RAG model instance not created. Cannot create tool.")
        return None

# The original function is no longer needed directly as a tool
# def query_rag_codebase(query: str, state: RagState) -> dict:
#    ... 