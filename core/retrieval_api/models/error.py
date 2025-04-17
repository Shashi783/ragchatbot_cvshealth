from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ErrorResponse(BaseModel):
    """Base model for error responses."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow) 