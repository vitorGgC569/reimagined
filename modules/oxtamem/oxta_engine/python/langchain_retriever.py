from typing import List, Any
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    # Mock classes for environment without langchain installed
    class BaseRetriever: pass
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    class CallbackManagerForRetrieverRun: pass

from .geodesic_sdk import GeodesicClient

class GeodesicCausalRetriever(BaseRetriever):
    """
    A Time-Traveling Retriever for LangChain.
    Instead of searching by semantic similarity, it retrieves the
    causal history of a specific entity (Variable/Key).
    """
    client: GeodesicClient
    depth: int = 10

    def __init__(self, client: GeodesicClient, depth: int = 10):
        super().__init__()
        self.client = client
        self.depth = depth

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Query format expected: "KEY_NAME"
        (e.g., "Sensor_X" or "Conversation_ID_123")

        Returns the history of that key as a sequence of documents.
        """
        key = query.strip()

        # Use the native driver or redis to fetch history
        # Note: If using Redis driver, ensure the server supports RECALL or fetch manually iteratively
        # For this prototype, we assume native driver or that 'recall_history' works.

        try:
            history = self.client.recall_history(key, self.depth)
        except NotImplementedError:
            # Fallback for Redis: just get latest
            latest = self.client.load_latest(key)
            history = [latest] if latest else []

        docs = []
        for i, state in enumerate(history):
            # Format the state into text if it's not already
            content = str(state)

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "geodesic_memory",
                    "key": key,
                    "causal_step": i, # 0 = Latest
                    "is_historical": i > 0
                }
            ))

        return docs
