"""CR-DV Step 0 — read-only DB helper (DATABASE_URL, session set read only)."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import psycopg

def conn():
    c = psycopg.connect(os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://"), autocommit=True)
    c.execute("SET default_transaction_read_only = on")
    c.execute("SET statement_timeout = '120s'")
    return c
