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
# Add to utils/data_io.py
from contextlib import contextmanager
from typing import Optional
from sqlalchemy import text

@contextmanager
def tx(url: Optional[str] = None):
    """
    Begin a transaction and yield a connection. Auto-commits/rollbacks.
    Usage:
        with tx() as conn:
            conn.execute(text("SELECT 1"))
    """
    eng = get_engine(url)
    with eng.begin() as conn:
        # Keep timestamps consistent (DB stores UTC)
        try:
            conn.execute(text("SET TIME ZONE 'UTC'"))
        except Exception:
            pass  # harmless on engines that don't support this
        yield conn

def init_orats_monies_schema(url: Optional[str] = None) -> None:
    """
    Create the minute-level ORATS monies table + indexes if they don't exist.
    Safe to call anytime.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS orats_monies_minute (
      ticker           text        NOT NULL,
      trade_date       date        NOT NULL,
      minute_ts        timestamptz NOT NULL,
      expiry_date      date        NOT NULL,
      dte              int         NOT NULL,
      underlying       numeric(14,6),
      forward          numeric(14,6),
      rf_rate          double precision,
      div_yield        double precision,
      atm_iv           double precision,
      smile            jsonb       NOT NULL,
      inserted_at      timestamptz NOT NULL DEFAULT now(),
      updated_at       timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (ticker, expiry_date, minute_ts)
    );
    CREATE INDEX IF NOT EXISTS idx_orats_monies_minute_td
      ON orats_monies_minute (ticker, trade_date);
    CREATE INDEX IF NOT EXISTS idx_orats_monies_minute_exp
      ON orats_monies_minute (ticker, expiry_date);
    CREATE INDEX IF NOT EXISTS idx_orats_monies_minute_minute
      ON orats_monies_minute (ticker, minute_ts DESC);
    """
    with tx(url) as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

