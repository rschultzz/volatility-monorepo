"""CR-DC Step 0 — exact (trade-level) vs 1-s approximate aggressor price assignment, one retained
Ironbeam session (RTH). READ-ONLY. Mechanical only: where volume lands on the price grid; no
direction, outcome or feature statistics are computed (the session is forward-log, B4)."""
import json
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

D = sys.argv[2]
conn = get_backfill_db_conn()
assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
a = f"{D} 06:30 America/Los_Angeles"; b = f"{D} 13:00 America/Los_Angeles"
tr = pd.DataFrame(conn.execute("""SELECT ts_utc, price, size, aggressor_side FROM ironbeam_es_trades
    WHERE ts_utc >= %s::timestamptz AND ts_utc < %s::timestamptz AND symbol='XCME:ES.Z26'""", (a, b)).fetchall(),
    columns=["ts", "price", "size", "side"])
s1 = pd.DataFrame(conn.execute("""SELECT ts_utc, buy_vol, sell_vol, high, low, close, notional, volume
    FROM ironbeam_es_flow_1s WHERE ts_utc >= %s::timestamptz AND ts_utc < %s::timestamptz AND symbol='XCME:ES.Z26'""",
    (a, b)).fetchall(), columns=["ts", "buy", "sell", "high", "low", "close", "notional", "vol"])
for c in ["buy", "sell", "high", "low", "close", "notional", "vol"]:
    s1[c] = s1[c].astype(float)
tr["sec"] = tr["ts"].dt.floor("s")
out = {"session": D, "trades": len(tr), "seconds": len(s1),
       "trade_vol": float(tr["size"].sum()), "flow_1s_vol": float(s1[["buy", "sell"]].sum().sum())}

def cell(p, w):
    return np.floor(np.asarray(p, float) / w + 1e-9) * w

res = {}
for side, col, ext in (("SELL", "sell", "low"), ("BUY", "buy", "high")):
    t = tr[tr["side"] == side]
    for w in (0.25, 0.5):
        ex = t.assign(m=t["ts"].dt.floor("min"), c=cell(t["price"], w)).groupby(["m", "c"])["size"].sum()
        for name, px in (("extreme", s1[ext]), ("close", s1["close"]),
                         ("vwap", np.round(s1["notional"] / s1["vol"].where(s1["vol"] > 0) / 0.25) * 0.25)):
            ap = s1.assign(m=s1["ts"].dt.floor("min"), c=cell(px, w))
            ap = ap[ap[col] > 0].groupby(["m", "c"])[col].sum()
            j = pd.concat([ex.rename("ex"), ap.rename("ap")], axis=1).fillna(0)
            overlap = float(np.minimum(j["ex"], j["ap"]).sum() / j["ex"].sum())
            res[f"{side}_{w}_{name}"] = round(overlap, 4)
    # volume-weighted signed price error of the extreme rule, per trade (trade price vs its second's extreme)
    m = t.merge(s1[["ts", ext]].rename(columns={"ts": "sec"}), on="sec", how="left")
    err = (m[ext] - m["price"]).astype(float)
    res[f"{side}_extreme_err_mean_pts"] = round(float(np.average(err.fillna(0), weights=m["size"])), 4)
    res[f"{side}_extreme_err_share_exact"] = round(float(m.loc[err.abs() < 1e-9, "size"].sum() / m["size"].sum()), 4)
out["cell_overlap_share"] = res
print(json.dumps(out, indent=1))
