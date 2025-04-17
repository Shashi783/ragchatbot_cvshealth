from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from core.vector_db.models.vector_models import VectorSearchRequest, VectorSearchResponse

class GenerateRequest(BaseModel):
    """Request model for generate endpoint."""
    chat_id: str
    query: str
    category_name: str
    persist_dir: str
    collection_name: str
    llm_model_type: str
    is_web_search: str
    metadata: Optional[Dict[str, Any]] = {}

class GenerateResponse(BaseModel):
    """Response model for generate endpoint."""
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str
    content: str

class ChatHistory(BaseModel):
    """Model for chat history."""
    messages: List[ChatMessage]

class MetadataFilter(BaseModel):
    """Model for metadata filters."""
    key: str
    value: Any
    operator: str = "eq"

class MetadataFilters(BaseModel):
    """Model for metadata filters collection.""" 