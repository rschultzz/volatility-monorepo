"""CR-DC smoke (DC11 / A1.2): absorption marks on exact vs approximate (1-s low/high) cells, same session,
same rule (K 3, M 3). READ-ONLY. Mechanical display comparison: counts and position agreement of a display
mark; no direction, outcome or forward statistic. Train sessions (Databento: es_flow_price_1m vs
es_flow_1s_databento) and one retained Ironbeam session (ironbeam_es_trades vs ironbeam_es_flow_1s)."""
import json
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from packages.shared import flow_heat as fh  # noqa: E402
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

conn = get_backfill_db_conn(); assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"


def df(sql, params, cols):
    return pd.DataFrame(conn.execute(sql, params).fetchall(), columns=cols)


def run(d, source):
    a, b = f"{d} 06:30 America/Los_Angeles", f"{d} 13:00 America/Los_Angeles"
    t1s = "es_flow_1s_databento" if source == "databento" else "ironbeam_es_flow_1s"
    sym = conn.execute(f"SELECT symbol FROM {t1s} WHERE ts_utc >= %s::timestamptz AND ts_utc < %s::timestamptz "
                       "GROUP BY 1 ORDER BY sum(volume) DESC LIMIT 1", (a, b)).fetchone()[0]
    s = df(f"SELECT extract(epoch FROM ts_utc)::bigint, high, low, buy_vol, sell_vol FROM {t1s} "
           "WHERE symbol=%s AND ts_utc >= %s::timestamptz AND ts_utc < %s::timestamptz", (sym, a, b), ["ts", "h", "l", "buy", "sell"])
    s[["h", "l", "buy", "sell"]] = s[["h", "l", "buy", "sell"]].astype(float)
    approx = fh.cells_from_seconds(s["ts"], s["h"], s["l"], s["buy"], s["sell"])
    if source == "databento":
        r = df("SELECT extract(epoch FROM minute_utc)::bigint, price, buy_vol, sell_vol FROM es_flow_price_1m "
               "WHERE source='databento' AND symbol=%s AND minute_utc >= %s::timestamptz AND minute_utc < %s::timestamptz",
               (sym, a, b), ["t", "price", "buy", "sell"])
        exact = fh.cells_from_price_1m(r["t"], r["price"], r["buy"], r["sell"])
        tier = "es_flow_price_1m"
    else:
        r = df("SELECT (floor(extract(epoch FROM ts_utc)/60)*60)::bigint, price, aggressor_side, sum(size)::float8 FROM ironbeam_es_trades "
               "WHERE symbol=%s AND ts_utc >= %s::timestamptz AND ts_utc < %s::timestamptz GROUP BY 1,2,3", (sym, a, b),
               ["t", "price", "side", "size"])
        exact = fh.cells_from_trades(r["t"], r["price"], r["size"], r["side"])
        tier = "trades"
    bars = s.assign(t=fh.minute_of(s["ts"])).groupby("t").agg(h=("h", "max"), l=("l", "min"))
    start = int(pd.Timestamp(a.replace(" America/Los_Angeles", ""), tz="America/Los_Angeles").timestamp())
    end = int(pd.Timestamp(b.replace(" America/Los_Angeles", ""), tz="America/Los_Angeles").timestamp())
    me = fh.absorption_marks(exact, bars, start, end)
    ma = fh.absorption_marks(approx, bars, start, end)
    vol_e = float(exact[["buy", "sell"]].sum().sum()); vol_a = float(approx[["buy", "sell"]].sum().sum())

    def near(x, ys):
        return any(y["kind"] == x["kind"] and abs(y["time"] - x["time"]) <= 60 and abs(y["p"] - x["p"]) <= 1 for y in ys)
    by = lambda m, k: sum(1 for x in m if x["kind"] == k)  # noqa: E731
    return {"session": d, "source": source, "exact_tier": tier, "symbol": sym,
            "cell_volume_exact_over_1s": round(vol_e / vol_a, 4) if vol_a else None,
            "marks_exact": len(me), "marks_approx": len(ma),
            "sellers_exact": by(me, "sellers_absorbed"), "sellers_approx": by(ma, "sellers_absorbed"),
            "buyers_exact": by(me, "buyers_absorbed"), "buyers_approx": by(ma, "buyers_absorbed"),
            "approx_marks_matching_an_exact_mark": sum(near(x, me) for x in ma),
            "exact_marks_matching_an_approx_mark": sum(near(x, ma) for x in me)}


out = [run(d, "databento") for d in ("2026-01-06", "2026-03-02", "2026-04-01", "2026-05-01")]
out.append(run("2026-10-27", "ironbeam"))
print(json.dumps(out, indent=1))
