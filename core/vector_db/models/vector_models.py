from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from dataclasses import dataclass

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

class VectorSearchRequest(BaseModel):
    """Request model for vector search operations."""
    query: str
    collection_name: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    min_score: float = 0.0

class VectorSearchResponse(BaseModel):
    """Response model for vector search operations."""
    results: List[SearchResult]
    metadata: Optional[Dict[str, Any]] = {}

class VectorDocument(BaseModel):
    """Model for vector document storage."""
    document_id: str
    file_path: str
    file_name: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    custom_metadata: Dict[str, Any] = {}
    chunks: List[str]

class VectorCollection(BaseModel):
    """Model for vector collection configuration."""
    name: str
    dimension: int
    distance_metric: str = "cosine"
    metadata: Optional[Dict[str, Any]] = {}

class VectorDocumentMetadata(BaseModel):
    """Model for document metadata."""
    document_id: str
    file_path: str
    file_name: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    custom_metadata: Dict[str, Any] = {}
    chunk_index: int

class VectorHealthCheck(BaseModel):
    """Model for vector database health check response."""
    status: str
    details: Dict[str, Any]
    timestamp: datetime

class VectorStats(BaseModel):
    """Model for vector database statistics."""
    total_documents: int
    total_chunks: int
    storage_size: int
    index_size: int
    last_updated: datetime 