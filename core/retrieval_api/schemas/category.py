from pydantic import BaseModel

class CategoryConfig(BaseModel):
    """Configuration for category names."""
    oil_gas: str = "Oil and Gas"
    finance: str = "Finance"
    healthcare: str = "Healthcare" 