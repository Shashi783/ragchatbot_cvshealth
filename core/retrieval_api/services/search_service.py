import logging
from typing import List, Dict, Any
from datetime import datetime
import time
from pathlib import Path
import json
from ..models.base import Document, SearchResult
from ..schemas.api import SearchRequest, SearchResponse
from ..managers.storage_manager import LocalStorageManager
from ..managers.embedding_manager import EmbeddingManager
from ..managers.quadrant_manager import QuadrantManager

logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling document search operations."""
    
    def __init__(self, storage_manager: LocalStorageManager, embedding_manager: EmbeddingManager, config: Dict[str, Any]):
        self._storage = storage_manager
        self._embedding = embedding_manager
        self._quadrant = QuadrantManager(config)
        
    async def search_documents(self, request: SearchRequest) -> SearchResponse:
        """Search for documents using semantic search.
        
        Args:
            request: Search request containing query and parameters
            
        Returns:
            SearchResponse containing the search results
        """
        try:
            start_time = time.time()
            
            # Get collection path
            collection_path = self._storage.load_persist_dir(request.collection_id)
            
            # Load quadrant data if not already loaded
            quadrant_path = Path(collection_path) / "quadrants"
            if quadrant_path.exists():
                self._quadrant.load_from_disk(str(quadrant_path))
            
            # Generate query embedding
            query_embedding = await self._embedding.get_embedding(request.query)
            
            # Search using quadrant manager
            results = self._quadrant.search_vectors(
                query_vector=query_embedding,
                top_k=request.top_k
            )
            
            # Convert results to SearchResult objects
            search_results = []
            for result in results:
                doc_id = result["doc_id"]
                doc_file = Path(collection_path) / "documents" / f"{doc_id}.json"
                
                if doc_file.exists():
                    with open(doc_file, 'r') as f:
                        doc_data = json.load(f)
                        document = Document(**doc_data)
                        search_results.append(SearchResult(
                            document=document,
                            score=result["score"],
                            metadata=result["metadata"]
                        ))
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                results=[result.dict() for result in search_results],
                total_results=len(search_results),
                processing_time=processing_time,
                metadata={"collection_id": request.collection_id}
            )
            
        except Exception as e:
            logger.error(f"Error performing document search: {e}")
            raise
            
    def _load_documents(self, collection_path: str) -> List[Document]:
        """Load documents from a collection.
        
        Args:
            collection_path: Path to the collection directory
            
        Returns:
            List of Document objects
        """
        try:
            # Implementation depends on how documents are stored
            # This is a placeholder that should be implemented based on your storage format
            documents = []
            # TODO: Implement document loading logic
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise 