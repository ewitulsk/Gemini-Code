import logging
from .state import RagState


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

# The original function is no longer needed directly as a tool
# def query_rag_codebase(query: str, state: RagState) -> dict:
#    ... 