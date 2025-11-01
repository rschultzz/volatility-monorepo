from __future__ import annotations
from typing import Optional
from sqlalchemy import text
from shared.utils.data_io import get_engine

def run(date: Optional[str] = None) -> None:
    """
    Entry point for EOD ingest. Replace the stub with your real job.
    """
    eng = get_engine()
    # Example: prove connectivity; replace with your upsert logic
    with eng.begin() as cx:
        cx.execute(text("select 1"))
    print(f"[ingest] ORATS EOD stub ran (date={date or 'auto'})")
