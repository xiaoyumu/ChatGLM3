from typing import Optional, List
from pydantic import BaseModel


class TextEmbeddingRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class TextEmbeddingResponse(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    took: Optional[float] = None  # Seconds
    embeddings: Optional[List[List[float]]] = None
    token: Optional[int] = 0
