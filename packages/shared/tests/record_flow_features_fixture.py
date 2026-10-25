"""Record the 1-second flow fixture for test_flow_features.py (CR-CD step 1). READ-ONLY.

Three train sessions of `es_flow_1s_databento`, [06:15, 13:00) PT, the session's highest-volume
contract. Run once:
    apps/web/.venv/bin/python packages/shared/tests/record_flow_features_fixture.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402
from packages.shared.flow_features import pt  # noqa: E402

SESSIONS = (date(2026, 1, 6), date(2026, 3, 2), date(2026, 5, 1))
OUT = Path(__file__).resolve().parent / "fixtures" / "cr_cd_flow_1s_3sessions.npz"


def main():
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    conn.execute("SET default_transaction_read_only = on")
    arrays = {}
    for d in SESSIONS:
        a, b = pt(d, 6, 15), pt(d, 13, 0)
        sym = conn.execute("""SELECT symbol FROM es_flow_1s_databento WHERE ts_utc >= %s AND ts_utc < %s
                              GROUP BY symbol ORDER BY sum(volume) DESC LIMIT 1""", (a, b)).fetchone()[0]
        rows = conn.execute("""SELECT extract(epoch FROM ts_utc)::bigint, buy_vol, sell_vol, volume,
                                      open, high, low, close
                               FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
                               ORDER BY ts_utc""", (sym, a, b)).fetchall()
        x = np.array(rows, dtype=float)
        key = d.isoformat()
        arrays[f"{key}/t"] = x[:, 0].astype(np.int64)
        for j, k in enumerate(("buy", "sell", "vol"), start=1):
            arrays[f"{key}/{k}"] = x[:, j].astype(np.int64)
        for j, k in enumerate(("o", "h", "l", "c"), start=4):
            arrays[f"{key}/{k}"] = x[:, j]
        print(key, sym, len(rows))
    np.savez_compressed(OUT, **arrays)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
