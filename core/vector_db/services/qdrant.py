from typing import List, Dict, Any, Optional
import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from .base import BaseVectorDB, SearchResult
from datetime import datetime

logger = logging.getLogger(__name__)

class QdrantDB(BaseVectorDB):
    """Qdrant vector database implementation."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_size: int = 384,  # Default size for all-MiniLM-L6-v2
    ):
        """
        Initialize Qdrant client and ensure collection exists.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant host
            port: Qdrant port
            embedding_model: SentenceTransformer model name
            vector_size: Size of the embedding vectors
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(embedding_model)
        
        # Create collection if it doesn't exist
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int):
        """Ensure the collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="keyword"
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="owner_id",
                    field_schema="keyword"
                )
                
                logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {str(e)}")
            raise

    def add_documents(
        self,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Add document chunks to Qdrant.
        
        Args:
            chunks: List of text chunks
            metadata: Document metadata
            
        Returns:
            document_id: ID of the added document
        """
        try:
            # Generate embeddings for chunks
            embeddings = self.encoder.encode(chunks)
            
            # Prepare points for batch upload
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=f"{metadata['document_id']}_{i}",
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "document_id": metadata["document_id"],
                        "file_path": metadata["file_path"],
                        "file_name": metadata["file_name"],
                        "owner_id": metadata["owner_id"],
                        "created_at": metadata["created_at"],
                        "updated_at": metadata["updated_at"],
                        "chunk_index": i,
                        "custom_metadata": json.dumps(metadata.get("custom_metadata", {}))
                    }
                )
                points.append(point)
            
            # Upload points in batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added document {metadata['document_id']} with {len(chunks)} chunks")
            return metadata["document_id"]
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents with optional filters.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters for metadata fields
            
        Returns:
            List of SearchResult objects with their metadata
        """
        try:
            # Generate query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            # Prepare filter conditions if any
            filter_conditions = None
            if filters:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filters.items()
                    ]
                )
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
            
            # Format results into SearchResult objects
            formatted_results = []
            for result in results:
                search_result = SearchResult(
                    content=result.payload["content"],
                    document_id=result.payload["document_id"],
                    chunk_index=result.payload["chunk_index"],
                    score=result.score,
                    metadata=json.loads(result.payload["custom_metadata"]),
                    file_path=result.payload["file_path"],
                    file_name=result.payload["file_name"],
                    owner_id=result.payload["owner_id"],
                    created_at=result.payload["created_at"],
                    updated_at=result.payload["updated_at"]
                )
                formatted_results.append(search_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if successful
        """
        try:
            # Delete all points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise

    def update_metadata(
        self,
        document_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for an existing document."""
        try:
            # Update the document's metadata in Qdrant
            self.client.update(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                payload=metadata_updates
            )
            logger.info(f"Updated metadata for document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            return False

    def get_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document and its metadata by ID."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0] * 384,  # Dummy vector for searching by ID
                limit=1,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
            if results:
                return results[0].payload
            return None
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            return None

    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List documents matching the given filters."""
        try:
            filter_conditions = None
            if filters:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filters.items()
                    ]
                )
            results = self.client.scroll(
                collection_name=self.collection_name,
                query_filter=filter_conditions,
                offset=offset,
                limit=limit
            )
            return [result.payload for result in results]
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector database connection."""
        try:
            # Assuming a simple check by getting collection stats
            stats = self.client.get_collection(collection_name=self.collection_name)
            return {
                "status": "healthy",
                "details": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            stats = self.client.get_collection(collection_name=self.collection_name)
            return {
                "total_documents": stats.total_points,
                "total_chunks": stats.total_points,  # Assuming chunks are equivalent to points
                "storage_size": stats.storage_size,
                "index_size": stats.index_size,
                "last_updated": stats.last_updated.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}