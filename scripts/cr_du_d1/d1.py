#!/usr/bin/env python3
"""CR-DU diagnosis D1 — does the v1 engine see the same setup as Ryan's CR-CZ cards? READ-ONLY, no outcomes, no tuning.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_d1/d1.py

For each CR-CZ card (card 31 dropped) by Ryan's first pre-reveal label: (a) the engine's box at the card's decision
minute (06:30 anchor, the day's engine state) vs the card's box (meta.range: closes over range start .. qualifying bar):
overlap in pts and minutes; (b) where they overlap, the first and every failed v1 rule; (c) the same with the engine
anchored at the card's range start. Rules on the card's side are listed even when the engine has no read (`forceSide`).
Reads setup_candidates id / session_date / direction / entry_ts_pt / meta only (never `outcome`) and setup_reviews labels.
Sessions before 2026-02-03 have no prior-sessions baseline: box geometry only, rules not evaluated.
Writes scripts/logs/cr-du/d1/ (git-ignored).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cr_du_backtest_v1 as BT  # noqa: E402

OUT = BT.LOG / "d1"
WARM = BT.LOG / "sessions_warmup"
ENGINE_DIR = os.environ.get("CRDU_D1_ENGINE", str(BT.C.REPO_ROOT / "react_price_preview" / "src"))   # evaluateAt's forceSide


def cards():
    with BT.db() as c:
        return c.execute("""SELECT x.id, x.session_date, x.direction, x.entry_ts_pt, x.meta->'range',
                                   (SELECT r.label FROM setup_reviews r WHERE r.candidate_id = x.id AND NOT coalesce(r.post_reveal, false)
                                    ORDER BY r.reviewed_at, r.id LIMIT 1)
                            FROM setup_candidates x WHERE x.set_name = 'cr-cz:cp-ranges' AND x.id <> 'CR-CZ:cr-cz:cp-ranges:31'
                            ORDER BY x.session_date, x.entry_ts_pt""").fetchall()


def session_file(d: dt.date) -> Path:
    f = BT.SES / f"{d}.json"
    if f.exists():
        return f
    w = WARM / f"{d}.json"
    if not w.exists():
        IB, flow_v3, V = BT._ib()
        assert d <= BT.TRAIN_END
        p, _ = flow_v3.build_cells_payload(IB.engine, d.isoformat(), session="RTH", until_pt=None,
                                           params=flow_v3.parse_absorb_params({}), session_date=d, session_note=None)
        inp = V.build_inputs(IB.engine, IB, d, p, check_blocked=False)
        WARM.mkdir(parents=True, exist_ok=True)
        w.write_text(json.dumps({"trade_date": d.isoformat(), "payload": p, "inputs": inp}))
    return w


def pt_epoch(x) -> int:
    t = x if isinstance(x, dt.datetime) else dt.datetime.fromisoformat(str(x))
    return int(t.replace(tzinfo=BT.PT).timestamp())


def main():
    rows = cards()
    job = []
    for cid, d, direction, entry, rg, label in rows:
        f = session_file(d)
        job.append({"id": cid, "date": str(d), "label": label, "side": direction, "file": str(f), "warm": f.parent == WARM,
                    "t": pt_epoch(entry) - 60, "r0": pt_epoch(rg["formed_ts_pt"]), "top": rg["top"], "bottom": rg["bottom"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "job.json").write_text(json.dumps(job))
    subprocess.run(["node", str(Path(__file__).with_name("d1.mjs")), str(OUT / "job.json"), str(OUT / "out.json"), ENGINE_DIR], check=True)
    res = json.loads((OUT / "out.json").read_text())
    lines = ["# CR-DU D1 — engine box vs Ryan's CR-CZ cards (train, learning set; no outcomes)", ""]
    for lab in ("take", "skip"):
        rs = [r for r in res if r["label"] == lab]
        warm = [r for r in rs if r["warm"]]
        ev = [r for r in rs if not r["warm"]]
        a = Counter()
        for r in rs:
            b = r["b"]
            a["engine box formed (≥ 8 min, ≥ 2 pts)" if b["formed"] else "engine box not formed / none"] += 1
            a["boxes overlap in price and time" if b["overlap_pts"] > 0 and b["overlap_min"] > 0 else "no overlap"] += 1
        ovp = sorted(r["b"]["overlap_pts"] for r in rs)
        ovm = sorted(r["b"]["overlap_min"] for r in rs)
        med = lambda v: v[len(v) // 2] if v else None
        lines += [f"## {lab}s: {len(rs)} cards ({len(warm)} on warm-up sessions: geometry only)", "",
                  "(a) " + " · ".join(f"{k} {v}" for k, v in a.items()),
                  f"    overlap pts median {med(ovp)} (card height median {med(sorted(r['card_h'] for r in rs))}); "
                  f"overlap min median {med(ovm)} (card box median {med(sorted(r['card_min'] for r in rs))} min, engine box median "
                  f"{med(sorted(r['b']['minutes'] or 0 for r in rs))} min); engine box starts before the card's: "
                  f"{sum(r['b']['from'] < r['r0'] for r in rs if r['b']['from'])}, after: {sum(r['b']['from'] > r['r0'] for r in rs if r['b']['from'])}", ""]
        for key, title in (("b", "(b) engine's own box (06:30 anchor), cards whose boxes overlap"), ("c", "(c) engine anchored at the card's range start")):
            sel = [r for r in ev if key == "c" or (r["b"]["overlap_pts"] > 0 and r["b"]["overlap_min"] > 0)]
            first = Counter(r[key]["failed"][0]["code"] if r[key]["failed"] else f"verdict {r[key]['verdict']}" for r in sel)
            every = Counter(f["code"] for r in sel for f in r[key]["failed"])
            lines += [f"{title}: {len(sel)} cards (rules evaluated, warm-up excluded)", "",
                      "| First failed rule | Cards |", "|---|---|", *[f"| {k} | {v} |" for k, v in first.most_common()], "",
                      "Every failed rule (cards failing it): " + ", ".join(f"{k} {v}" for k, v in every.most_common()), ""]
            if key == "c":
                lines += ["(c) box vs card: engine box = card box (±0.25 pts both edges) on "
                          f"{sum(abs(r['c']['top'] - r['top']) <= 0.25 and abs(r['c']['bottom'] - r['bottom']) <= 0.25 for r in sel if r['c']['top'] is not None)}"
                          f" / {len(sel)}; engine box restarted after the card's start (a break) on {sum(r['c']['from'] > r['r0'] for r in sel if r['c']['from'])}", ""]
    (OUT / "d1_report.md").write_text("\n".join(lines))
    (OUT / "d1_cards.json").write_text(json.dumps(res))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
