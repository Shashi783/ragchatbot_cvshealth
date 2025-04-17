from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from core.vector_db.models.vector_models import SearchResult, VectorDocument, VectorCollection

from .chat import ChatMessage, ChatHistory
from .error import ErrorResponse

class Message(BaseModel):
    """Base class for all message types."""
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class ChatRAGResponse(BaseModel):
    """Represents a RAG (Retrieval-Augmented Generation) response."""
    answer: str = Field(..., description="Generated answer")
    sources: List[VectorDocument] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    processing_time: float = Field(..., description="Time taken to process the request")

# # Request flow
# ChatRequest -> ChatMessage (via to_chat_message)
# ChatMessage -> ChatRequest (via from_chat_message)

# # Response flow
# ChatResponse -> ChatMessage (via to_chat_message)
# ChatMessage -> ChatResponse (via from_chat_message)

# # Base model flow
# ChatMessage.from_request(ChatRequest) -> ChatMessage
# ChatMessage.from_response(ChatResponse) -> ChatMessage

__all__ = ['ChatMessage', 'ChatHistory', 'ErrorResponse', 'Message', 'ChatRAGResponse']