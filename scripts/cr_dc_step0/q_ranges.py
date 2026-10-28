"""CR-DC Step 0 — 1-s range > 1 tick share; trades aggressor values; exact vs 1-s approximation.
READ-ONLY. Value reads: Databento train only (< 2026-06-05 21:00 UTC); Ironbeam 2026 train for the
1-s range share; the exact-vs-approx comparison uses the retained trades (mechanical, price assignment only)."""
import json
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

conn = get_backfill_db_conn()
assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
conn.execute("SET statement_timeout = '1200s'")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
out = {}
out["aggressor_side"] = [list(map(str, r)) for r in conn.execute(
    "SELECT aggressor_side, count(*), sum(size) FROM ironbeam_es_trades GROUP BY 1 ORDER BY 2 DESC").fetchall()]

RTH = "(ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30' AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'"
def share(table, a, b, extra=""):
    r = conn.execute(f"""
        SELECT count(*) n, sum((high - low > 0.25 + 1e-9)::int) n_gt1,
               sum(volume) v, sum(CASE WHEN high - low > 0.25 + 1e-9 THEN volume ELSE 0 END) v_gt1,
               sum((high - low > 0.5 + 1e-9)::int) n_gt2,
               sum(CASE WHEN high - low > 0.5 + 1e-9 THEN volume ELSE 0 END) v_gt2,
               sum((high = low)::int) n_eq
        FROM {table} WHERE ts_utc >= %s AND ts_utc < %s AND {RTH} {extra}""", (a, b)).fetchone()
    n, n1, v, v1, n2, v2, neq = [float(x or 0) for x in r]
    return {"seconds": int(n), "sec_gt_1tick": n1 / n, "vol_in_gt_1tick": v1 / v,
            "sec_gt_2tick": n2 / n, "vol_in_gt_2tick": v2 / v, "sec_single_price": neq / n}
out["databento_2026_train_rth"] = share("es_flow_1s_databento", "2026-01-02", "2026-06-05 21:00+00")
out["databento_2024R_a_rth"] = share("es_flow_1s_databento", "2024-04-03", "2025-01-01")
out["databento_2026-05-01_rth"] = share("es_flow_1s_databento", "2026-05-01 13:00+00", "2026-05-01 21:00+00")
out["ironbeam_2026_train_rth"] = share("ironbeam_es_flow_1s", "2026-01-02", "2026-06-05 21:00+00")
print(json.dumps(out, indent=1))
