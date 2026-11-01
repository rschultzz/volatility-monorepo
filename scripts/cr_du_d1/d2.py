#!/usr/bin/env python3
"""CR-DU diagnosis D2 — summary of scripts/cr_du_d1/d2.mjs (+ D1's engine-box reads). Read-only files, no outcomes.

    node scripts/cr_du_d1/d2.mjs scripts/logs/cr-du/d1/job.json scripts/logs/cr-du/d1/d2_cards.json
    python3 scripts/cr_du_d1/d2.py

Evaluated sessions only (2026-02-03 → 06-05, prior-sessions baseline); card side; takes vs skips; counts and
distributions only. Writes scripts/logs/cr-du/d1/d2_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

D = Path(__file__).resolve().parent.parent / "logs" / "cr-du" / "d1"


def q(v, nd=1):
    v = sorted(x for x in v if x is not None)
    if not v:
        return "–"
    f = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return f"{f(.25):.{nd}f} / {f(.5):.{nd}f} / {f(.75):.{nd}f} (n {len(v)})"


def signed(m):
    """Δ share percentile signed to the card side: +p when the lean is the card's (buying for a short), −p otherwise."""
    return None if m is None or m.get("p") is None else (m["p"] if m["cardSide"] else -m["p"])


def main():
    d1 = {r["id"]: r for r in json.loads((D / "d1_cards.json").read_text())}
    d2 = [r for r in json.loads((D / "d2_cards.json").read_text()) if not r["warm"]]
    out = ["# CR-DU D2 — takes vs skips at the decision minute (evaluated sessions; card side; no outcomes)", ""]
    rows2, rows3, rows4, rows5 = [], [], [], []
    for lab in ("take", "skip"):
        rs = [r for r in d2 if r["label"] == lab]
        eng = [d1[r["id"]]["b"]["read"] for r in rs]
        eng_m = [{"p": e["sharePct"], "cardSide": e["share"] is not None and ((e["share"] > 0) == (d1[r["id"]]["side"] == "short"))} if e["sharePct"] is not None else None
                 for e, r in zip(eng, rs)]
        for nm, ms in (("(a) engine's box", eng_m), ("(b) card's box", [r["cardBox"] for r in rs]), ("(c) last 10 closed minutes", [r["last10"] for r in rs])):
            rows2.append([lab, nm, q([m["p"] if m and m["p"] is not None else None for m in ms]),
                          q([signed(m) for m in ms]), f"{sum(1 for m in ms if m and m['p'] is not None and m['cardSide'] and m['p'] >= 80)} / {len(ms)}"])
        ts = [r["cvd"]["cardSideT"] for r in rs]
        rows3.append([lab, len(rs), f"{sum(1 for t in ts if t and t > 0)}", q(ts, 2), f"{sum(1 for t in ts if t and t >= 2)}", f"{sum(1 for t in ts if t and t <= -2)}"])
        rows4.append([lab, len(rs), sum(r["win"]["any"] > 0 for r in rs), sum(r["win"]["inRange"] > 0 for r in rs), sum(r["win"]["atEdge"] > 0 for r in rs), sum(r["win"]["inTop8"] > 0 for r in rs)])
        for key, nm in (("tick", "1 tick (D2.5 first run)"), ("pts2", "2 pts (box break threshold)")):
            hs = [r["edge"][key] for r in rs]
            near = [min(h, key=lambda x: (x["dist"], x["before"])) for h in hs if h]
            allh = [x for h in hs for x in h]
            rows5.append([lab, nm, f"{sum(bool(h) for h in hs)} / {len(rs)}", f"{median([x['dist'] for x in near]):.2f}" if near else "–",
                          f"{median([x['before'] for x in near]):.0f}" if near else "–", f"{sum(x['inBox'] for x in allh)} / {sum(not x['inBox'] for x in allh)}"])
    # item 1: takes that failed the lean (engine's own box) — in-box absorption on the card side, and what stops them
    item1 = []
    for lab in ("take", "skip"):
        rs = [r for r in d2 if r["label"] == lab]
        failed = []
        for r in rs:
            e = d1[r["id"]]["b"]["read"]
            short = d1[r["id"]]["side"] == "short"
            lean_ok = e["sharePct"] is not None and e["sharePct"] >= 80 and e["share"] is not None and ((e["share"] > 0) == short)
            if not lean_ok:
                failed.append(r)
        side = lambda r: "buyers" if d1[r["id"]]["side"] == "short" else "sellers"
        edge = lambda r: "top" if d1[r["id"]]["side"] == "short" else "bottom"
        abs_any = [r for r in failed if any(x.startswith(side(r)) for x in d1[r["id"]]["b"]["read"]["inBoxAbs"])]
        abs_edge = [r for r in failed if f"{side(r)}@{edge(r)}" in d1[r["id"]]["b"]["read"]["inBoxAbs"]]
        why = {}
        for r in abs_edge:
            f = d1[r["id"]]["b"]["failed"]
            why[f[0]["text"] if f else d1[r["id"]]["b"]["verdict"]] = why.get(f[0]["text"] if f else d1[r["id"]]["b"]["verdict"], 0) + 1
        stall = lambda r: (d1[r["id"]]["b"]["read"]["cppPct"] or 0) >= 80 or (d1[r["id"]]["b"]["read"]["noExtend"] or {}).get("up" if d1[r["id"]]["side"] == "short" else "down")
        win_edge = [r for r in failed if r["win"]["atEdge"] > 0]
        item1.append([lab, len(rs), len(failed), len(abs_any), len(abs_edge), "; ".join(f"{k} ({v})" for k, v in why.items()) or "–",
                      len(win_edge), sum(bool(stall(r)) for r in win_edge)])
    T = lambda h, rows: ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]
    out += ["## 1. Absorption alone (lean failed, engine's own box)", ""] + T(
        ["Label", "Cards", "Lean failed", "…with card-side absorption the verdict saw (in the engine box)", "…at the faded edge", "First failed rule on those",
         "…with a card-side mark at the edge in the card's window (any of the session's marks)", "…of those, stall leg passes"], item1)
    out += ["## 2. |Δ share| percentile at the decision minute (p25 / p50 / p75)", ""] + T(
        ["Label", "Measured over", "Percentile", "Signed to the card side", "≥ p80 on the card side"], rows2)
    out += ["## 3. CVD inside the card's box (OLS of cumulative delta on minutes; t signed to the card side)", ""] + T(
        ["Label", "Cards", "Slope on the card side (t > 0)", "t p25 / p50 / p75", "t ≥ 2", "t ≤ −2"], rows3)
    out += ["## 4. Card-side absorption inside the card's box window (mark minute ≥ range start, confirmed by the decision)", ""] + T(
        ["Label", "Cards", "Any price", "Within the box high–low ± 2", "At the faded edge (≤ 2 pts)", "Among the chart's top 8, unbroken"], rows4)
    out += ["## 5. Absorption at the faded edge (≤ 2 pts, any earlier mark), unbroken by closes: two thresholds", ""] + T(
        ["Label", "Unbroken =", "Cards with ≥ 1", "Nearest distance (median)", "Minutes before (median)", "Marks in window / before"], rows5)
    (D / "d2_report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
