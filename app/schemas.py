from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class DocumentCreate(BaseModel):
    # strip_whitespace + min_length rejects blank or whitespace-only values (422)
    # instead of silently storing an empty document and embedding it.
    title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
    ]
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DocumentRead(BaseModel):
    # from_attributes lets Pydantic read straight off the SQLAlchemy object.
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime


class DocumentSearchResult(DocumentRead):
    score: float
