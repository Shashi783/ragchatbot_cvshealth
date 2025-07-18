from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Request Schemas
class SearchRequest(BaseModel):
    """Schema for document search request."""
    query: str = Field(..., description="Search query")
    collection_id: str = Field(..., description="ID of the collection to search in")
    top_k: int = Field(default=5, description="Number of results to return")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class DocumentUploadRequest(BaseModel):
    """Schema for document upload request."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_id: Optional[str] = Field(default=None, description="ID of the collection to add the document to")

class CollectionCreateRequest(BaseModel):
    """Schema for collection creation request."""
    name: str = Field(..., description="Name of the collection")
    description: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

# Response Schemas
class SearchResponse(BaseModel):
    """Schema for search response."""
    results: List[Dict[str, Any]] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time: float = Field(..., description="Time taken to process the search")
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class DocumentResponse(BaseModel):
    """Schema for document response."""
    doc_id: str = Field(..., description="ID of the document")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class CollectionResponse(BaseModel):
    """Schema for collection response."""
    collection_id: str = Field(..., description="ID of the collection")
    name: str = Field(..., description="Name of the collection")
    description: Optional[str] = Field(default=None)
    document_count: int = Field(..., description="Number of documents in the collection")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None) 