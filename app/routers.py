from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.embeddings import embed, rank
from app.models import Document
from app.schemas import DocumentCreate, DocumentRead, DocumentSearchResult

router = APIRouter()


@router.post("/documents", response_model=DocumentRead, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    document = Document(
        title=payload.title,
        content=payload.content,
        # Embed title + content so the title's keywords contribute to recall.
        embedding=embed(f"{payload.title}. {payload.content}"),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.get("/documents/search", response_model=list[DocumentSearchResult])
def search_documents(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=100),
    filter_title: str | None = Query(None, min_length=1),
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
