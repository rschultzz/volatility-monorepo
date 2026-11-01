"""CR-DU Step 0 Q5 — contradiction-bar percentiles per feed and per 30-min RTH bucket (read-only, train only).

Train sessions 2026-01-02 -> 2026-06-05 (B4), FOMC decision days and the March 2026 expiry week left out (the CR-DQ
baseline's exclusions, scripts/cr_dq_step0.py). Per session the symbol with the most RTH volume. Per feed (B5: never
pooled): Databento `es_flow_1m_databento`, Ironbeam `ironbeam_es_flow_1m`. Per minute: volume, signed delta
(buy - sell), range (high - low of that feed's own minute). p10 / p50 / p80 / p90 per 30-min bucket from 06:30 PT.
Ratio = Ironbeam / Databento of each percentile; plus the paired per-minute ratio on minutes both feeds have.
No outcome is read. Usage: apps/web/.venv/bin/python scripts/cr_du_step0/q5_contradiction_pcts.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cr_dq_step0 as s0  # noqa: E402  (loads .env, read-only PGOPTIONS)

import psycopg  # noqa: E402

from packages.shared.backfill_safety import connect_redacted  # noqa: E402

SQL = """
SELECT symbol, (minute_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, extract(epoch FROM minute_utc)::bigint AS t,
       volume::float8, (buy_vol - sell_vol)::float8, (high - low)::float8
FROM {table}
WHERE minute_utc >= %s AND minute_utc < %s
  AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
  AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
"""
QS = [10, 50, 80, 90]
NB = 13  # 30-min buckets, 06:30 -> 13:00


def load(conn, src, days):
    import datetime as dt
    lo = dt.datetime.combine(s0.FIRST, dt.time(0), s0.PT)
    hi = dt.datetime.combine(s0.LAST + dt.timedelta(days=1), dt.time(0), s0.PT)
    by = {}
    for sym, d, t, v, dl, rg in conn.execute(SQL.format(table=s0.TABLES[src]), (lo, hi)).fetchall():
        by.setdefault(d, {}).setdefault(sym, []).append((t, v, dl, rg))
    out = {}
    for d in days:
        syms = by.get(d)
        if not syms:
            continue
        sym = max(syms, key=lambda s: sum(r[1] for r in syms[s]))
        t0 = s0.rth_open_utc(d)
        out[d] = {(t - t0) // 60: (v, dl, rg) for t, v, dl, rg in syms[sym] if 0 <= (t - t0) // 60 < s0.RTH_MIN}
    return out


def main():
    days = s0.sessions()
    try:
        with connect_redacted(os.environ["BACKFILL_DATABASE_URL"], connect_timeout=20) as conn:
            data = {src: load(conn, src, days) for src in ("databento", "ironbeam")}
    except Exception as e:  # never print the connection string
        print("DB error:", type(e).__name__)
        sys.exit(1)
    print(f"sessions: databento {len(data['databento'])}, ironbeam {len(data['ironbeam'])} "
          f"({s0.FIRST} -> {s0.LAST}, excl. FOMC + March expiry week)")
    P = {}
    for src, per in data.items():
        for b in range(NB):
            rows = np.array([x for d in per.values() for i, x in per[d].items() if i // 30 == b] if False else
                            [x for m in per.values() for i, x in m.items() if i // 30 == b])
            P[(src, b)] = {k: np.percentile(rows[:, j], QS) for j, k in enumerate(("vol", "delta", "range"))} | {"n": len(rows)}
    hh = lambda b: f"{6 + (30 + 30 * b) // 60:02d}:{(30 + 30 * b) % 60:02d}"
    for k, unit in (("vol", "contracts"), ("delta", "contracts, signed"), ("range", "pts")):
        print(f"\n## minute {k} ({unit}) — p10 / p50 / p80 / p90")
        print("| bucket PT | n DB | Databento | n IB | Ironbeam | IB / DB |")
        print("|---|---|---|---|---|---|")
        for b in range(NB):
            db, ib = P[("databento", b)], P[("ironbeam", b)]
            f = (lambda a: " / ".join(f"{x:.2f}" for x in a)) if k == "range" else (lambda a: " / ".join(f"{x:.0f}" for x in a))
            ratio = " / ".join("–" if abs(y) < 1e-9 else f"{x / y:.2f}" for x, y in zip(ib[k], db[k]))
            print(f"| {hh(b)}–{hh(b + 1)} | {db['n']} | {f(db[k])} | {ib['n']} | {f(ib[k])} | {ratio} |")
    # paired minutes (same session, same RTH minute, both feeds)
    pv, pd_, pr = [], [], []
    for d, m in data["databento"].items():
        mi = data["ironbeam"].get(d, {})
        for i, (v, dl, rg) in m.items():
            if i in mi and v > 0:
                v2, dl2, rg2 = mi[i]
                pv.append(v2 / v)
                pd_.append((dl, dl2))
                pr.append(rg2 - rg)
    pd_ = np.array(pd_)
    slope = float(np.sum(pd_[:, 0] * pd_[:, 1]) / np.sum(pd_[:, 0] ** 2))
    print(f"\npaired minutes {len(pv)}: IB/DB volume median {np.median(pv):.3f} (p5 {np.percentile(pv, 5):.3f}, "
          f"p95 {np.percentile(pv, 95):.3f}); delta slope IB on DB (through 0) {slope:.3f}, corr "
          f"{np.corrcoef(pd_[:, 0], pd_[:, 1])[0, 1]:.3f}; range IB - DB median {np.median(pr):+.2f} "
          f"(p5 {np.percentile(pr, 5):+.2f}, p95 {np.percentile(pr, 95):+.2f})")


if __name__ == "__main__":
    main()
