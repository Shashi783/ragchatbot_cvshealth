import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path
from ..models.base import Document, Collection
from ..schemas.api import CollectionCreateRequest, CollectionResponse, DocumentUploadRequest, DocumentResponse
from ..managers.storage_manager import LocalStorageManager
from ..managers.embedding_manager import EmbeddingManager
from ..managers.quadrant_manager import QuadrantManager

logger = logging.getLogger(__name__)

class CollectionService:
    """Service for handling document collections."""
    
    def __init__(self, storage_manager: LocalStorageManager, embedding_manager: EmbeddingManager, config: Dict[str, Any]):
        self._storage = storage_manager
        self._embedding = embedding_manager
        self._quadrant = QuadrantManager(config)
        
    async def create_collection(self, request: CollectionCreateRequest) -> CollectionResponse:
        """Create a new document collection.
        
        Args:
            request: Collection creation request
            
        Returns:
            CollectionResponse containing the created collection details
        """
        try:
            collection_id = str(uuid.uuid4())
            
            # Create collection directory
            collection_path = self._storage.load_persist_dir(collection_id)
            
            # Create collection object
            collection = Collection(
                collection_id=collection_id,
                name=request.name,
                description=request.description,
                metadata=request.metadata
            )
            
            # Save collection metadata
            self._save_collection_metadata(collection_path, collection)
            
            return CollectionResponse(
                collection_id=collection_id,
                name=request.name,
                description=request.description,
                document_count=0,
                created_at=collection.created_at,
                updated_at=collection.updated_at,
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
            
    async def add_document(self, request: DocumentUploadRequest) -> DocumentResponse:
        """Add a document to a collection.
        
        Args:
            request: Document upload request
            
        Returns:
            DocumentResponse containing the added document details
        """
        try:
            if not request.collection_id:
                raise ValueError("Collection ID is required")
                
            # Get collection path
            collection_path = self._storage.load_persist_dir(request.collection_id)
            
            # Create document
            doc_id = str(uuid.uuid4())
            document = Document(
                doc_id=doc_id,
                content=request.content,
                metadata=request.metadata
            )
            
            # Generate embedding
            document.embedding = await self._embedding.get_embedding(request.content)
            
            # Add vector to quadrant manager
            self._quadrant.add_vector(
                vector=document.embedding,
                doc_id=doc_id,
                metadata={
                    "content": document.content,
                    "collection_id": request.collection_id,
                    "metadata": document.metadata,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat()
                }
            )
            
            # Save document to disk
            self._save_document(collection_path, document)
            
            # Update collection metadata
            self._update_collection_document_count(request.collection_id)
            
            # Save quadrant data
            self._quadrant.save_to_disk(str(collection_path / "quadrants"))
            
            return DocumentResponse(
                doc_id=doc_id,
                content=request.content,
                metadata=request.metadata,
                created_at=document.created_at,
                updated_at=document.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
            
    def get_collection(self, collection_id: str) -> CollectionResponse:
        """Get collection details.
        
        Args:
            collection_id: ID of the collection to retrieve
            
        Returns:
            CollectionResponse containing the collection details
        """
        try:
            collection_path = self._storage.load_persist_dir(collection_id)
            collection = self._load_collection_metadata(collection_path)
            
            return CollectionResponse(
                collection_id=collection.collection_id,
                name=collection.name,
                description=collection.description,
                document_count=len(collection.documents),
                created_at=collection.created_at,
                updated_at=collection.updated_at,
                metadata=collection.metadata
            )
            
        except Exception as e:
            logger.error(f"Error retrieving collection: {e}")
            raise
            
    def _save_collection_metadata(self, collection_path: str, collection: Collection):
        """Save collection metadata to storage."""
        try:
            metadata_file = Path(collection_path) / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(collection.dict(), f, default=str)
            logger.info(f"Saved collection metadata to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving collection metadata: {e}")
            raise
            
    def _save_document(self, collection_path: str, document: Document):
        """Save document to storage."""
        try:
            documents_dir = Path(collection_path) / "documents"
            documents_dir.mkdir(exist_ok=True)
            
            doc_file = documents_dir / f"{document.doc_id}.json"
            with open(doc_file, 'w') as f:
                json.dump(document.dict(), f, default=str)
            logger.info(f"Saved document to {doc_file}")
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise
            
    def _load_collection_metadata(self, collection_path: str) -> Collection:
        """Load collection metadata from storage."""
        try:
            metadata_file = Path(collection_path) / "metadata.json"
            if not metadata_file.exists():
                raise ValueError(f"Collection metadata not found at {metadata_file}")
                
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return Collection(**data)
            
        except Exception as e:
            logger.error(f"Error loading collection metadata: {e}")
            raise
            
    def _update_collection_document_count(self, collection_id: str):
        """Update document count in collection metadata."""
        try:
            collection_path = self._storage.load_persist_dir(collection_id)
            collection = self._load_collection_metadata(collection_path)
            
            # Update document count
            collection.documents = []  # We don't need to load all documents
            collection.updated_at = datetime.utcnow()
            
            # Save updated metadata
            self._save_collection_metadata(collection_path, collection)
            
        except Exception as e:
            logger.error(f"Error updating collection document count: {e}")
            raise 