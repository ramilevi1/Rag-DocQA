from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None
    top_k: int = 8
    session_id: Optional[str] = Field(default=None, description="Logical namespace for retrieval")

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict]
