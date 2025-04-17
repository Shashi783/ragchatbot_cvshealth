from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging
import re
from core.vector_db.schemas import (
    SearchRequest, SearchResponse,
    DocumentUploadRequest, DocumentResponse,
    CollectionCreateRequest, CollectionResponse
)
from core.retrieval_api.models.chat import ChatMessage
from core.retrieval_api.models.error import ErrorResponse

logger = logging.getLogger(__name__)

# Request Schemas
class ChatRequest(BaseModel):
    """Schema for chat request.
    
    Example:
        {
            "message": "What is machine learning?",
            "chat_id": "chat_123",
            "metadata": {...}
        }
    """
    message: str = Field(..., description="User's message")
    chat_id: str = Field(..., description="Unique identifier for the chat session")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the request"
    )

    @validator("chat_id")
    def validate_chat_id(cls, value):
        """Validate chat_id can contain alphanumeric characters, hyphens, and underscores."""
        if not all(c.isalnum() or c in "-_" for c in value):
            logger.warning(f"Invalid chat_id format: {value}")
            raise ValueError("Chat ID must be alphanumeric and can include hyphens and underscores.")
        return value

    @validator("message")
    def validate_message(cls, value):
        """Validate message is not empty."""
        if value.strip() == "":
            logger.warning("Message is empty")
            raise ValueError("Message cannot be empty.")
        return value.strip()

    @classmethod
    def from_chat_message(cls, message: ChatMessage, chat_id: Optional[str] = None) -> "ChatRequest":
        """Create request from ChatMessage model."""
        return cls(
            message=message.content,
            chat_id=chat_id or "new_chat",  # Provide default for new chats
            metadata=message.metadata
        )

    def to_chat_message(self) -> ChatMessage:
        """Convert request to ChatMessage model."""
        return ChatMessage(
            role="user",
            content=self.message,
            timestamp=datetime.utcnow(),
            metadata=self.metadata
        )

# Response Schemas
class ChatResponse(BaseModel):
    """Schema for chat response.
    
    Example:
        {
            "chat_id": "chat_123",
            "message": "Here's what I found...",
            "sources": [...],
            "processing_time": 0.5,
            "timestamp": "2024-03-26T12:00:00Z",
            "metadata": {...}
        }
    """
    chat_id: str = Field(..., description="ID of the chat session")
    message: str = Field(..., description="Assistant's response message")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sources used for generation (if any)"
    )
    processing_time: float = Field(
        default=0.0,
        description="Time taken to process the request in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the response"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the response"
    )

    @classmethod
    def from_chat_message(cls, chat_id: str, message: ChatMessage) -> "ChatResponse":
        """Create response from ChatMessage model."""
        return cls(
            chat_id=chat_id,
            message=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata
        )

    def to_chat_message(self) -> ChatMessage:
        """Convert response to ChatMessage model."""
        return ChatMessage(
            role="assistant",
            content=self.message,
            timestamp=self.timestamp,
            metadata=self.metadata
        )

# Error Schemas
class ChatErrorResponse(ErrorResponse):
    """Schema for chat-specific error responses."""
    pass 