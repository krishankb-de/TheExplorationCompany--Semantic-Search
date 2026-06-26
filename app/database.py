import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Default to a local file DB; tests override this with an in-memory DB.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./documents.db")

# check_same_thread=False is required because FastAPI runs sync endpoints in a
# thread pool, so a connection may be touched by a thread other than its creator.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """Yield a session and guarantee it is closed once the request finishes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  register the Document table on Base

    Base.metadata.create_all(bind=engine)
