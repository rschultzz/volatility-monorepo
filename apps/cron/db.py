import os
import psycopg

def _normalize_db_url(url: str) -> str:
    # Render often provides postgres:// — psycopg prefers postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Ensure sslmode=require if not present
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url

def get_conn():
    """
    Return a psycopg v3 connection using DATABASE_URL.
    Matches the 'from db import get_conn' import your script expects.
    """
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    dsn = _normalize_db_url(raw)
    return psycopg.connect(dsn)
