from abc import ABC, abstractmethod
import os
from app.config import settings

class KnowledgeProvider(ABC):
    """Abstract interface for knowledge retrieval.
    Allows swapping static KB for full RAG (embeddings, hybrid search, reranker)
    later without modifying the agent graph.
    """

    @abstractmethod
    def get_context(self, query: str) -> str:
        """Retrieve relevant context for the query."""
        pass


class StaticKBProvider(KnowledgeProvider):
    """Demo implementation loading data/knowledge_base.md at startup.
    Returns the complete structured markdown within <knowledge_base> tags.
    """

    def __init__(self, kb_path: str = None):
        self.kb_path = kb_path or settings.KB_PATH
        self._content = self._load_kb()

    def _load_kb(self) -> str:
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def get_context(self, query: str = "") -> str:
        return f"<knowledge_base>\n{self._content}\n</knowledge_base>"

    def reload(self):
        """Hot-reload KB if file was updated."""
        self._content = self._load_kb()
