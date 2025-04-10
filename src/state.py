import os
from vertexai.preview.generative_models import GenerativeModel

class RagState:

    def __init__(self):
        self.target_directory: str | None = None
        self.rag_corpus_name: str | None = None
        self.gcs_bucket_name: str | None = None
        self.project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location: str = "us-central1"
        # This model is used INTERNALLY by the RAG tool function
        self.rag_model: GenerativeModel | None = None 

    def is_rag_ready(self) -> bool:
        # Check if the internal RAG model needed by the tool is ready
        return bool(self.rag_model and self.project_id and self.rag_corpus_name)

    def is_setup_complete(self) -> bool:
        # Check if initial user inputs are gathered
        return bool(self.target_directory and self.gcs_bucket_name and self.project_id)

# Global instance to hold the state
global_rag_state = RagState()

print("RagState class and global_rag_state instance defined in src/state.py") 