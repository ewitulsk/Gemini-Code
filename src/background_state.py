import threading
from typing import Literal, Optional
from vertexai.preview.generative_models import GenerativeModel

# Define possible states for the background setup process
SetupStatus = Literal["pending", "running_setup", "running_model_init", "complete", "failed"]

class BackgroundRAGState:
    """
    Manages the state of the background RAG setup process in a thread-safe manner.
    """
    def __init__(self, project_id: str, location: str):
        self._lock = threading.Lock()
        self._setup_status: SetupStatus = "pending"
        self._error_message: Optional[str] = None
        self._rag_corpus_name: Optional[str] = None
        self._internal_rag_model: Optional[GenerativeModel] = None
        self.project_id: str = project_id
        self.location: str = location
        # Store other config if needed by the tool/setup later
        # self.rag_model_id: str = rag_model_id # Example

    @property
    def setup_status(self) -> SetupStatus:
        with self._lock:
            return self._setup_status

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error_message

    @property
    def rag_corpus_name(self) -> Optional[str]:
        with self._lock:
            return self._rag_corpus_name

    @property
    def internal_rag_model(self) -> Optional[GenerativeModel]:
        with self._lock:
            return self._internal_rag_model

    def set_status_running_setup(self):
        with self._lock:
            self._setup_status = "running_setup"
            self._error_message = None

    def set_status_running_model_init(self, corpus_name: str):
        with self._lock:
            if self._setup_status == "running_setup": # Ensure setup finished first
                 self._setup_status = "running_model_init"
                 self._rag_corpus_name = corpus_name
                 self._error_message = None

    def set_status_complete(self, model: GenerativeModel):
        with self._lock:
             if self._setup_status == "running_model_init": # Ensure model init was running
                self._setup_status = "complete"
                self._internal_rag_model = model
                self._error_message = None

    def set_status_failed(self, error: str):
        with self._lock:
            self._setup_status = "failed"
            self._error_message = error
            self._internal_rag_model = None # Ensure no model is set on failure

print("BackgroundRAGState class defined in src/background_state.py") 