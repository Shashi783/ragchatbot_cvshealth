from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    """Standardized search result format."""
    content: str
    document_id: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any]
    file_path: str
    file_name: str
    owner_id: str
    created_at: datetime
    updated_at: datetime

class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations.
    All vector database implementations must inherit from this class
    and implement its abstract methods.
    """
    
    @abstractmethod
    def add_documents(
        self,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Add document chunks to vector database.
        
        Args:
            chunks: List of text chunks to be embedded and stored
            metadata: Document metadata containing:
                - document_id: Unique identifier for the document
                - file_path: Path to the original file
                - file_name: Name of the original file
                - owner_id: ID of the document owner
                - created_at: Document creation timestamp
                - updated_at: Document update timestamp
                - custom_metadata: Dict of additional metadata
                
        Returns:
            document_id: ID of the added document
            
        Raises:
            ValueError: If chunks is empty or metadata is invalid
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents with optional filters.
        
        Args:
            query: Search query text to be embedded and compared
            limit: Maximum number of results to return
            filters: Optional filters for metadata fields, e.g.:
                {
                    "owner_id": "user123",
                    "department": "Finance",
                    "created_at": {"$gte": "2024-01-01"}
                }
            min_score: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects sorted by relevance
            
        Raises:
            ValueError: If query is empty or filters are invalid
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if successful, False if document not found
            
        Raises:
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def update_metadata(
        self,
        document_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for an existing document.
        
        Args:
            document_id: ID of the document to update
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            bool: True if successful, False if document not found
            
        Raises:
            ValueError: If metadata_updates is invalid
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def get_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document and its metadata by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Dict containing document content and metadata,
            or None if document not found
            
        Raises:
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List documents matching the given filters.
        
        Args:
            filters: Optional metadata filters
            offset: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of documents with their metadata
            
        Raises:
            ValueError: If filters are invalid
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the vector database connection.
        
        Returns:
            Dict containing:
                - status: "healthy" or "unhealthy"
                - details: Additional health information
                - timestamp: Time of the check
                
        Raises:
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dict containing:
                - total_documents: Total number of documents
                - total_chunks: Total number of chunks
                - storage_size: Size of the database in bytes
                - index_size: Size of the index in bytes
                - last_updated: Timestamp of last update
                
        Raises:
            ConnectionError: If database connection fails
        """
        raise NotImplementedError

    def batch_add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents in batch.
        Can be overridden by implementations for better performance.
        
        Args:
            documents: List of documents, each containing:
                - chunks: List of text chunks
                - metadata: Document metadata
                
        Returns:
            List of document IDs in the same order
            
        Raises:
            ValueError: If any document is invalid
            ConnectionError: If database connection fails
        """
        document_ids = []
        for doc in documents:
            doc_id = self.add_documents(
                chunks=doc['chunks'],
                metadata=doc['metadata']
            )
            document_ids.append(doc_id)
        return document_ids

    def batch_delete_documents(
        self,
        document_ids: List[str]
    ) -> Dict[str, bool]:
        """
        Delete multiple documents in batch.
        Can be overridden by implementations for better performance.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Dict mapping document IDs to deletion success status
            
        Raises:
            ConnectionError: If database connection fails
        """
        results = {}
        for doc_id in document_ids:
            results[doc_id] = self.delete_document(doc_id)
        return results