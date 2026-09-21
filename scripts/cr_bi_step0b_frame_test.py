#!/usr/bin/env python3
"""CR-BI Step 0b (D follow-up) — read-only. Touch/close under three frame pairings:

  ES  vs B   canonical v0.6.0-openiv            (price C, target B)
  SPX vs B   CR-BH shadow v0.6.0-openiv-spxcash (price A, target B)
  SPX vs A   SPX cash vs the wall converted to cash: target − carry(trade_date), where carry is the
             kernel-weighted (discounted_level − strike) of the rows that build the wall
             (scripts/logs/cr_bi_step0b_frame_b_minus_a.csv).

Implemented by adding carry(trade_date) to the SPX daily frame for that row — identical to lowering
the target for touch / close / distance. No DB writes; nothing persisted.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")
import logging; logging.disable(logging.WARNING)
import pandas as pd
from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.outcomes_runner import compute_outcome_for_date
from scripts.cr_b_backfill_outcomes import _fetch_landscape

conn = get_backfill_db_conn(); conn.execute("SET default_transaction_read_only = on")
cache = pd.read_pickle(REPO_ROOT / "scripts/.cache/cr_bh_rows.pkl")
spx = cache["spx_daily"][["open", "high", "low", "close"]]
carry = pd.read_csv(REPO_ROOT / "scripts/logs/cr_bi_step0b_frame_b_minus_a.csv", parse_dates=["trade_date"])
carry = dict(zip(carry.trade_date.dt.date, carry.b_minus_a))
diff = pd.read_csv(REPO_ROOT / "scripts/logs/cr_bh_diff_rows.csv", parse_dates=["trade_date"])
diff["trade_date"] = diff.trade_date.dt.date
src = conn.execute("""SELECT o.trade_date, f.regime_at_classification, f.feature_vector FROM bt_daily_outcomes o
  JOIN bt_daily_features f ON f.ticker=o.ticker AND f.trade_date=o.trade_date AND f.feature_version=o.feature_version AND f.active
  WHERE o.ticker='SPX' AND o.feature_version=%s AND o.active ORDER BY 1""", (CANONICAL_FEATURE_VERSION,)).fetchall()
ls = _fetch_landscape(conn, "SPX", [r[0] for r in src])
rows = []
for d, regime, fv in src:
    c = carry.get(d)
    if c is None:
        continue
    o, _ = compute_outcome_for_date(d, regime, fv or {}, ls.get(d, {}), spx + c)
    rows.append({"trade_date": d, "status_aa": o["outcome_status"], "touch_aa": o["reached_touch"],
                 "close_aa": o["reached_close"], "dtr_aa": o["days_to_reach"], "carry": c})
j = diff.merge(pd.DataFrame(rows), on="trade_date")
c = j[(j.outcome_status_old == "computed") & (j.outcome_status_new == "computed") & (j.status_aa == "computed")]
def p2(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return math.erfc(abs((k2 / n2 - k1 / n1) / se) / math.sqrt(2)) if se else 1.0
print(f"rows computed under all three: {len(c)} (same-calendar subset: {(c.horizon_end_date_old == c.horizon_end_date_new).sum()})")
print(f"mean carry {c.carry.mean():.1f}, mean ES basis {c.basis.mean():.1f}, mean (basis − carry) {(c.basis - c.carry).mean():+.1f}, "
      f"sd {(c.basis - c.carry).std():.1f}")
print("\n| subset | n | touch ES-vs-B | SPX-vs-B | SPX-vs-A | close ES-vs-B | SPX-vs-B | SPX-vs-A |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
def line(name, s):
    f = lambda col: f"{100 * s[col].astype(bool).mean():.1f} %"
    print(f"| {name} | {len(s)} | {f('reached_touch_old')} | {f('reached_touch_new')} | {f('touch_aa')} | "
          f"{f('reached_close_old')} | {f('reached_close_new')} | {f('close_aa')} |")
line("all", c)
same = c[c.horizon_end_date_old == c.horizon_end_date_new]
line("all, same calendar", same)
for hz in (5, 20, 60):
    line(f"magnet-above {hz}", c[(c.regime == "magnet-above") & (c.horizon == hz)])
    line(f"magnet-above {hz}, same calendar", same[(same.regime == "magnet-above") & (same.horizon == hz)])
line("magnetic-pin (all horizons)", c[c.regime == "magnetic-pin"])
print("\nmagnet-above 5-session touch by ES-basis bucket (<20 · 20-35 · 35+):")
g = c[(c.regime == "magnet-above") & (c.horizon == 5)]
for name, col in (("ES-vs-B", "reached_touch_old"), ("SPX-vs-B", "reached_touch_new"), ("SPX-vs-A", "touch_aa")):
    ks = [(int(g[g.bucket == b][col].astype(bool).sum()), int((g.bucket == b).sum())) for b in ("<20", "20-35", "35+")]
    print(f"  {name}: " + " · ".join(f"{k}/{n} {100 * k / n:.0f}%" for k, n in ks) + f"  p(<20 vs 35+) {p2(*ks[0], *ks[2]):.3f}")
m = c[c.regime == "magnet-above"]
print(f"\nmagnet-above flips ES-vs-B → SPX-vs-A: touch T→F {((m.reached_touch_old == True) & (m.touch_aa == False)).sum()}, "
      f"F→T {((m.reached_touch_old == False) & (m.touch_aa == True)).sum()}; "
      f"same-calendar only: T→F {((same.regime == 'magnet-above') & (same.reached_touch_old == True) & (same.touch_aa == False)).sum()}, "
      f"F→T {((same.regime == 'magnet-above') & (same.reached_touch_old == False) & (same.touch_aa == True)).sum()}")
