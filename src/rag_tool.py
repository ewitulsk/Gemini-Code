import logging
from .background_state import BackgroundRAGState
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from typing import Callable


def create_rag_tool_closure(background_state: BackgroundRAGState):
    """Factory function that creates the RAG query tool closure using BackgroundRAGState.

    Args:
        background_state: The BackgroundRAGState object tracking the setup process.

    Returns:
        A function that takes a query string and returns the RAG result dictionary.
    """
    
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
            Example loading: {'status': 'loading', 'message': 'RAG setup is still in progress. Please try again shortly.'}
            Example failure: {'status': 'error', 'error_message': 'RAG setup failed: [Reason]'}
            Example no_answer: {'status': 'no_answer', 'message': 'The RAG model could not find relevant information.'}
        """
        logging.info(f"RAG Tool closure triggered with query: '{query}'")
        
        # Access state from the enclosing scope (closure)
        status = background_state.setup_status
        internal_model = background_state.internal_rag_model

        if status == "pending" or status == "running_setup" or status == "running_model_init":
            logging.info(f"RAG tool called but setup is still in progress (Status: {status}).")
            return {"status": "loading", "message": f"RAG setup is still in progress (Status: {status}). Please try again shortly."}
        elif status == "failed":
            error_msg = background_state.error_message or "Unknown error during RAG setup."
            logging.error(f"RAG tool called but setup failed: {error_msg}")
            return {"status": "error", "error_message": f"RAG setup failed: {error_msg}"}
        elif status == "complete":
            if not internal_model:
                # This case should ideally not happen if status is complete
                logging.error("RAG tool called. Status is complete, but internal model is missing.")
                return {"status": "error", "error_message": "Internal error: RAG setup complete but model unavailable."}
            
            # Proceed with query execution
            try:
                logging.info("Sending query to internal RAG model...")
                response = internal_model.generate_content(query)

                if response and response.text:
                    logging.info("RAG model returned a response.")
                    return {"status": "success", "response": response.text}
                else:
                    logging.warning("RAG model did not return a text response.")
                    return {"status": "no_answer", "message": "The RAG model could not find relevant information for the query."}

            except Exception as e:
                logging.exception(f"Error during RAG model query execution in closure: {e}")
                return {"status": "error", "error_message": f"An error occurred while querying the RAG model: {e}"}
        else:
             # Should not happen with defined states
             logging.error(f"RAG tool called with unexpected status: {status}")
             return {"status": "error", "error_message": f"Unexpected internal state: {status}"}

    # Return the inner function (the closure)
    return query_rag_codebase_impl

# New function specifically for initializing the internal model *after* corpus is ready
def initialize_internal_rag_model(project_id: str, location: str, rag_corpus_name: str, rag_model_id: str) -> GenerativeModel | None:
    """Initializes the internal Gemini model with RAG retrieval tool.

    Args:
        project_id: Google Cloud Project ID.
        location: Google Cloud resource location.
        rag_corpus_name: The full resource name of the *ready* RAG Corpus.
        rag_model_id: The identifier for the Gemini model to use internally.

    Returns:
        An initialized GenerativeModel instance if successful, otherwise None.
    """
    try:
        logging.info(f"Initializing internal RAG model for tool (Corpus: {rag_corpus_name}, Model: {rag_model_id})...")
        # Assuming rag.RagResource expects the full corpus resource name
        rag_resource = rag.RagResource(rag_corpus=rag_corpus_name) 
        rag_retrieval_tool = Tool.from_retrieval(retrieval=rag.Retrieval(
            source=rag.VertexRagStore(rag_resources=[rag_resource],
                                      similarity_top_k=5, # Configurable?
                                      vector_distance_threshold=0.5) # Configurable?
                                      ))
        
        # Use the specified model ID (e.g., gemini-1.5-flash)
        internal_model_instance = GenerativeModel(rag_model_id, tools=[rag_retrieval_tool])
        logging.info(f"Internal RAG model ({rag_model_id}) for tool initialized successfully.")
        return internal_model_instance
    except Exception as e:
        logging.exception(f"Failed to initialize internal RAG model for tool: {e}")
        return None

# The original function is no longer needed directly as a tool
# def query_rag_codebase(query: str, state: RagState) -> dict:
#    ... 