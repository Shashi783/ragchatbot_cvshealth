import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
import logging

logger = logging.getLogger(__name__)

class StorageContextManager:
    """Handles setup of Qdrant vector store and index creation."""

    def __init__(self, config, secret):
        """
        :param config: Callable or dict for configuration values
        :param secret: Callable for fetching secrets
        """
        self._config = config
        self._secret = secret

    def setup_storage_context(self, collection_name: str) -> VectorStoreIndex:
        """Setup storage context with Qdrant vector store."""
        try:
            client = qdrant_client.QdrantClient(
                url=self._secret("QDRANT_URL"),
                api_key=self._secret("QDRANT_API_KEY")
            )

            aclient = qdrant_client.AsyncQdrantClient(
                url=self._secret("QDRANT_URL"),
                api_key=self._secret("QDRANT_API_KEY")
            )

            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,
                collection_name=collection_name,
                enable_hybrid=self._config("QDRANT_ENABLE_HYBRID"),
                fastembed_sparse_model=self._config("FASTEMBED_SPARSE_MODEL"),
                prefer_grpc=False,
            )

            logger.info("Qdrant clients and vector store created successfully.")

            collections = client.get_collections()
            logger.debug(f"Collections in Vector DB: {collections}")

            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            logger.info("VectorStoreIndex created.")

            return index

        except Exception as e:
            logger.error(f"Error setting up storage context: {e}")
            raise
