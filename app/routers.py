from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AfterValidator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.embeddings import embed, rank
from app.models import Document
from app.schemas import DocumentCreate, DocumentRead, DocumentSearchResult

router = APIRouter()


def _strip_nonblank(value: str | None) -> str | None:
    """Strip a query-string value and reject it (422) if it is blank.

    Used for `q`/`filter_title`: FastAPI honours `min_length` on query params but
    does not strip first, so a whitespace-only value would otherwise slip through.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("must not be blank or whitespace-only")
    return value


@router.post("/documents", response_model=DocumentRead, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    # Fast path: reject an already-stored duplicate before paying for an embedding.
    # Values are already whitespace-stripped by the schema.
    duplicate = db.scalar(
        select(Document).where(
            Document.title == payload.title, Document.content == payload.content
        )
    )
    if duplicate is not None:
        raise HTTPException(status_code=409, detail="Document already exists")

    document = Document(
        title=payload.title,
        content=payload.content,
        # Embed title + content so the title's keywords contribute to recall.
        embedding=embed(f"{payload.title}. {payload.content}"),
    )
    db.add(document)
    try:
        db.commit()
    except IntegrityError:
        # The pre-check above races: a concurrent POST may insert the same
        # (title, content) between our SELECT and COMMIT. The unique constraint
        # is the authoritative guard — the loser rolls back and gets a 409.
        db.rollback()
        raise HTTPException(status_code=409, detail="Document already exists")
    db.refresh(document)
    return document


@router.get("/documents/search", response_model=list[DocumentSearchResult])
def search_documents(
    q: Annotated[
        str,
        Query(min_length=1, description="Natural language search query"),
        AfterValidator(_strip_nonblank),
    ],
    top_k: Annotated[int, Query(ge=1, le=100)] = 5,
    filter_title: Annotated[
        str | None, Query(min_length=1), AfterValidator(_strip_nonblank)
    ] = None,
    db: Session = Depends(get_db),
) -> list[DocumentSearchResult]:
    statement = select(Document)
    if filter_title:
        # Case-insensitive substring filter applied before ranking.
        statement = statement.where(Document.title.ilike(f"%{filter_title}%"))
    documents = db.scalars(statement).all()

    ranked = rank(embed(q), [(d.id, d.embedding) for d in documents], top_k)
    by_id = {d.id: d for d in documents}
    return [
        DocumentSearchResult(
            id=by_id[doc_id].id,
            title=by_id[doc_id].title,
            content=by_id[doc_id].content,
            created_at=by_id[doc_id].created_at,
            score=round(score, 4),
        )
        for doc_id, score in ranked
    ]


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db)) -> None:
    document = db.get(Document, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()
