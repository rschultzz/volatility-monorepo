#!/usr/bin/env python3
"""CR-DU diagnosis D2 item 5 — absorption at the level, not just in the box. READ-ONLY (cached session files), no outcomes.

    python3 scripts/cr_du_d1/d2_absorption.py      (after scripts/cr_du_d1/d1.py)

Per CR-CZ card, on the card's side (short: buyers absorbed; long: sellers absorbed): the session's absorption marks
(all exact RTH marks of the cells payload, not only the chart's top 8) with
  - price within 2 pts of the faded edge (short: the card box's top by closes; long: its bottom);
  - the mark at an earlier minute and confirmed by the decision bar's close (no lookahead);
  - not broken by closes since: no 1-min close more than 1 tick beyond the mark's price (above it for buyers absorbed,
    below for sellers absorbed) from the minute after the mark through the decision bar.
Reports takes vs skips: share of cards with ≥ 1 mark, the nearest mark's distance to the edge and minutes before the
decision, marks inside the card's box window (mark minute ≥ range start) vs before it, and takes that failed the lean
(D1 (b): Δ share below p80 or against the card side) with one. Writes scripts/logs/cr-du/d1/d2_item5.md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

D1 = Path(__file__).resolve().parent.parent / "logs" / "cr-du" / "d1"
TICK, NEAR = 0.25, 2.0


def main():
    job = json.loads((D1 / "job.json").read_text())
    d1 = {r["id"]: r for r in json.loads((D1 / "d1_cards.json").read_text())}
    cache, rows = {}, []
    for c in job:
        if c["file"] not in cache:
            cache[c["file"]] = json.loads(Path(c["file"]).read_text())
        rth = next((s for s in cache[c["file"]]["payload"]["segments"] if s["name"] == "rth"), None)
        exact = bool(rth and rth.get("exact") and rth.get("marks") is not None)
        bt, bc = rth["bars"]["t"], rth["bars"]["c"]
        short = c["side"] == "short"
        edge = c["top"] if short else c["bottom"]
        kind = "buyers_absorbed" if short else "sellers_absorbed"
        dec_close = c["t"] + 60
        hits = []
        for mk in (rth.get("marks") or []) if exact else []:
            if mk["kind"] != kind or not (mk["time"] < c["t"] and mk["confirm_time"] <= dec_close):
                continue
            p = float(mk["price"])
            if abs(p - edge) > NEAR + 1e-9:
                continue
            broken = any((cl > p + TICK + 1e-9) if short else (cl < p - TICK - 1e-9)
                         for t, cl in zip(bt, bc) if mk["time"] < t <= c["t"])
            if broken:
                continue
            hits.append({"price": p, "dist": round(abs(p - edge), 2), "before_min": round((dec_close - mk["time"]) / 60),
                         "in_box": mk["time"] >= c["r0"]})
        r = d1[c["id"]]
        rd = r["b"].get("read") or {}
        lean_ok = (rd.get("sharePct") is not None and rd["sharePct"] >= 80 and rd.get("share") is not None
                   and ((rd["share"] > 0) == short))
        rows.append({"id": c["id"], "label": c["label"], "warm": c["warm"], "exact": exact, "hits": hits,
                     "evaluated": not c["warm"], "lean_ok": lean_ok})
    out = ["# CR-DU D2 item 5 — absorption at the faded edge (train, learning set; no outcomes)", ""]
    for lab in ("take", "skip"):
        rs = [r for r in rows if r["label"] == lab]
        ex = [r for r in rs if r["exact"]]
        w = [r for r in ex if r["hits"]]
        near = [min(r["hits"], key=lambda h: (h["dist"], h["before_min"])) for r in w]
        allh = [h for r in w for h in r["hits"]]
        failed = [r for r in rs if r["evaluated"] and not r["lean_ok"]]
        out += [f"## {lab}s: {len(rs)} cards ({len(rs) - len(ex)} without exact marks)",
                f"- ≥ 1 mark at the faded edge: {len(w)} / {len(ex)} ({100 * len(w) / max(1, len(ex)):.0f} %)",
                f"- nearest mark per card: distance to the edge median {median([h['dist'] for h in near]) if near else '–'} pts; "
                f"minutes before the decision median {median([h['before_min'] for h in near]) if near else '–'}"
                + (f" (p25 {sorted(h['before_min'] for h in near)[len(near) // 4]}, p75 {sorted(h['before_min'] for h in near)[3 * len(near) // 4]})" if near else ""),
                f"- marks: {len(allh)} — inside the card's box window {sum(h['in_box'] for h in allh)}, before it {sum(not h['in_box'] for h in allh)}; "
                f"cards with one inside the window {sum(any(h['in_box'] for h in r['hits']) for r in w)}, only before it {sum(not any(h['in_box'] for h in r['hits']) for r in w)}",
                f"- cards that failed the lean (D1 (b), rules evaluated): {len(failed)}; with ≥ 1 such mark: {sum(bool(r['hits']) for r in failed)}"
                f" (inside the window {sum(any(h['in_box'] for h in r['hits']) for r in failed)})", ""]
    (D1 / "d2_item5.md").write_text("\n".join(out))
    (D1 / "d2_item5.json").write_text(json.dumps(rows))
    print("\n".join(out))


if __name__ == "__main__":
    main()
