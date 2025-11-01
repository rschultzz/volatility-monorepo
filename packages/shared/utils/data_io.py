import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy-friendly driver. Examples:
      postgres://user:pass@host/db  -> postgresql+psycopg://user:pass@host/db
      postgresql://...              -> postgresql+psycopg://...
      postgresql+psycopg://...      -> unchanged
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def get_engine(url: Optional[str] = None, *, pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Create a pooled SQLAlchemy Engine using DATABASE_URL or provided url.
    """
    raw = (url or os.getenv("DATABASE_URL", "")).strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    norm = _normalize_db_url(raw)
    engine = create_engine(
        norm,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        future=True,
    )
    return engine
