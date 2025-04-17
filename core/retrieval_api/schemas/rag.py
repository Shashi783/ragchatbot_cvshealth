from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import logging
import re

from .category import CategoryConfig

logger = logging.getLogger(__name__)

class LLMModelType(str, Enum):
    """Supported LLM model types."""
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GOOGLE = "GOOGLE"

class ChatRAGRequest(BaseModel):
    """Schema for RAG (Retrieval-Augmented Generation) API requests.
    
    Example:
        {
            "message": "What is machine learning?",
            "chat_id": "chat_123",
            "category": "general",
            "llm_model_type": "OPENAI",
            "is_web_search": true,
            "collection_name": "rag_llm",
            "persist_dir": "persist",
            "metadata": {...}
        }
    """
    message: str = Field(..., description="User's message for RAG processing")
    chat_id: str = Field(..., description="Unique identifier for the chat session")
    category: str = Field(..., description="Category of the query for specialized processing")
    llm_model_type: LLMModelType = Field(
        default=LLMModelType.OPENAI,
        description="Type of LLM model to use for generation"
    )
    is_web_search: bool = Field(
        default=True,
        description="Whether to include web search results in RAG"
    )
    collection_name: str = Field(
        default="rag_llm",
        description="Name of the vector collection to search in"
    )
    persist_dir: str = Field(
        default="persist",
        description="Directory for persisting vector data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the request"
    )

    @validator("chat_id")
    def validate_chat_id(cls, value):
        """Validate chat_id can contain alphanumeric characters, hyphens, and underscores."""
        if not all(c.isalnum() or c in "-_" for c in value):
            logger.warning(f"Invalid chat_id format: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid chat_id. Chat ID must be alphanumeric and can include hyphens and underscores.",
            )
        return value

    @validator("message")
    def validate_query(cls, value):
        """Validate query is not empty and not profane."""
        if not value.strip():
            logger.warning("Query is empty")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Query cannot be empty."
            )
        return value.strip()

    @validator("category")
    def validate_category(cls, value):
        """Validate category name is correct."""
        if value not in CategoryConfig.model_fields.keys():
            logger.warning(f"Invalid category: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid category. Category must be either {', '.join(CategoryConfig.model_fields.keys())}.",
            )
        return value

    @validator("collection_name")
    def validate_collection_name(cls, value):
        """Validate collection name format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            logger.warning(f"Invalid collection name format: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Collection name must contain only alphanumeric characters, hyphens, and underscores."
            )
        return value

    @validator("persist_dir")
    def validate_persist_dir(cls, value):
        """Validate persist directory path."""
        if not re.match(r'^[a-zA-Z0-9/_-]+$', value):
            logger.warning(f"Invalid persist directory format: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Persist directory must contain only alphanumeric characters, hyphens, underscores, and forward slashes."
            )
        return value

class ChatRAGResponse(BaseModel):
    """Schema for RAG (Retrieval-Augmented Generation) API responses.
    
    Example:
        {
            "chat_id": "chat_123",
            "message": "Machine learning is a field of study...",
            "sources": [...],
            "processing_time": 1.23,
            "timestamp": "2024-03-26T12:00:00Z",
            "metadata": {...}
        }
    """
    chat_id: str = Field(..., description="ID of the chat session")
    message: str = Field(..., description="Generated answer from RAG")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sources used for generation"
    )
    processing_time: float = Field(
        ...,
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