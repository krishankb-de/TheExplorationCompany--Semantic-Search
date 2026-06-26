from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)


class DocumentRead(BaseModel):
    # from_attributes lets Pydantic read straight off the SQLAlchemy object.
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime


class DocumentSearchResult(DocumentRead):
    score: float
