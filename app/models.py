import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.database import Base


class EmbeddingType(TypeDecorator):
    """Persist a list[float] embedding as a JSON string in a TEXT column.

    SQLite has no native vector/array type, so we serialise on write and
    deserialise on read. Storing the vector in the same row as the text keeps
    a document and its embedding atomic.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else json.dumps(value)

    def process_result_value(self, value, dialect):
        return None if value is None else json.loads(value)


class UTCDateTime(TypeDecorator):
    """Keep datetimes timezone-aware UTC across a SQLite round-trip.

    SQLite has no tz-aware datetime type and SQLAlchemy's SQLite storage format
    drops the offset, so a plain ``DateTime(timezone=True)`` column reads back
    *naive* — the API would emit ``...T15:55:57`` with no ``+00:00`` and a client
    could not tell it is UTC. We normalise to UTC on write and re-attach UTC on
    read so the value is always aware.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:  # assume naive input is already UTC
            value = value.replace(tzinfo=timezone.utc)
        # Store UTC components; SQLite drops the offset either way.
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Document(Base):
    __tablename__ = "documents"
    # The unique constraint is the real guard against duplicate rows: the SELECT
    # pre-check in the router is best-effort and races under concurrent POSTs.
    __table_args__ = (
        UniqueConstraint("title", "content", name="uq_documents_title_content"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType, nullable=False)
    # NOTE: default must be a callable, otherwise the timestamp is frozen at
    # import time and every row gets the same created_at.
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
