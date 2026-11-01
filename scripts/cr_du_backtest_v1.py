#!/usr/bin/env python3
"""CR-DU item 3 — train backtest of the v1 rules engine. READ-ONLY (read-only transactions), train only.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_backtest_v1.py build   # session inputs
    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_backtest_v1.py run     # engine + report

Spec: specs/CR-DU-v1-rules-engine.md (Step 0 + Amendment 1). Every number is **train, in-sample**.
- Sessions: train trading days with a prior-sessions baseline (2026-02-03 → 2026-06-05; the 21 earlier sessions are the
  20-session warm-up). Nothing after 2026-06-05 is read (asserted); `bt_derived_repair_archive` is never queried.
- Data (B5): the chart's cells payload (`flow_v3.build_cells_payload`, RTH, Databento: `es_flow_1m_databento` bars and
  flow, exact `es_flow_price_1m` cells, absorption marks) and the engine inputs (`v1_engine.build_inputs`: prior-only
  baseline and contradiction percentiles, A7 swing, strong corrected GEX levels per minute, `es_flow_1s_databento`
  seconds). Cached as JSON in scripts/logs/cr-du/sessions/ (git-ignored).
- Engine: react_price_preview/src/v1Engine.js through scripts/v1EngineCli.mjs (node) — the same code as the panel and
  the overlay. Variants: primary + the pre-declared sensitivity rows (box min 10; no 12:15 cutoff; no loss cap; D4
  entry; fixed 10-pt stop), report-only.
- Set 2 (Amendment 1): per-trade rows and monthly splits are withheld for sessions with a `set-2:cp-calls` card until
  set 2 is reviewed; totals are shown with and without those sessions. Side / hour splits use non-set-2 sessions only.
- CR-CZ overlap: the engine's verdict at each CR-CZ card's decision minute (its own trades that day before it) vs Ryan's
  first pre-reveal label. Counts only; no card outcome is read.
Writes scripts/logs/cr-du/backtest_report.md and backtest_results.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cp_lib as C  # noqa: E402  (loads .env, repo + scripts on sys.path)

sys.path.insert(0, str(C.REPO_ROOT / "apps" / "web"))
import numpy as np  # noqa: E402

TRAIN_END = dt.date(2026, 6, 5)
LOG = C.REPO_ROOT / "scripts" / "logs" / "cr-du"
SES = LOG / "sessions"
CLI = C.REPO_ROOT / "react_price_preview" / "scripts" / "v1EngineCli.mjs"
PT = C.PT
VARIANTS = {
    "primary": {},
    "box min 10": {"boxMin": 10},
    "no 12:15 cutoff": {"cutoff": False},
    "no loss cap": {"lossCap": False},
    "D4 entry (1 pt inside)": {"entry": "d4"},
    "fixed 10-pt stop": {"stop": "fixed10"},
}
LABEL = "train, in-sample"


def _ib():
    from modules.Ironbeam import callbacks as IB, flow_v3, v1_engine as V
    return IB, flow_v3, V


def sessions() -> list[dt.date]:
    from modules.Ironbeam import v1_engine as V
    out = sorted(dt.date.fromisoformat(d) for d in V.prior_baselines()["sessions"])
    assert out and max(out) <= TRAIN_END                                       # B4
    return out


def build():
    IB, flow_v3, V = _ib()
    SES.mkdir(parents=True, exist_ok=True)
    for d in sessions():
        f = SES / f"{d}.json"
        if f.exists():
            continue
        try:
            p, st = flow_v3.build_cells_payload(IB.engine, d.isoformat(), session="RTH", until_pt=None,
                                                params=flow_v3.parse_absorb_params({}), session_date=d, session_note=None)
            inp = V.build_inputs(IB.engine, IB, d, p, check_blocked=False)
        except Exception as e:                                                 # error type only (no URLs)
            print(d, "error", type(e).__name__)
            continue
        rth = next((s for s in p.get("segments", []) if s["name"] == "rth"), None)
        f.write_text(json.dumps({"trade_date": d.isoformat(), "payload": p, "inputs": inp}))
        print(d, st, rth and rth["source"], rth and rth["tier"], rth and len(rth["bars"]["t"]), "gex", inp["gex"]["reason"],
              len(inp["gex"]["schedule"]), "secs", len((inp.get("seconds") or {}).get("t", [])))


def db():
    import psycopg
    return psycopg.connect(os.environ["DATABASE_URL"].replace("+psycopg", ""), connect_timeout=20)


def set2_state():
    with db() as c:
        rows = c.execute("""SELECT x.session_date, EXISTS (SELECT 1 FROM setup_reviews r WHERE r.candidate_id = x.id)
                            FROM setup_candidates x WHERE x.set_name = 'set-2:cp-calls'""").fetchall()
    dates = {r[0] for r in rows}
    done = bool(rows) and all(r[1] for r in rows)
    return dates, done


def cz_cards():
    """CR-CZ cards with Ryan's first pre-reveal label (no outcome column read)."""
    with db() as c:
        rows = c.execute("""SELECT x.id, x.session_date, x.direction, x.entry_ts_pt,
                                   (SELECT r.label FROM setup_reviews r WHERE r.candidate_id = x.id AND NOT coalesce(r.post_reveal, false)
                                    ORDER BY r.reviewed_at, r.id LIMIT 1)
                            FROM setup_candidates x WHERE x.set_name = 'cr-cz:cp-ranges' AND x.id <> 'CR-CZ:cr-cz:cp-ranges:31'
                            ORDER BY x.id""").fetchall()
    return rows


def decision_bar(entry_ts_pt) -> int:
    """The card's decision bar (entry_ts_pt = its close, naive PT) as the bar's UTC start epoch."""
    t = entry_ts_pt if isinstance(entry_ts_pt, dt.datetime) else dt.datetime.fromisoformat(str(entry_ts_pt))
    t = t.replace(tzinfo=PT) - dt.timedelta(minutes=1)
    return int(t.timestamp())


def node(job: dict) -> dict:
    jp, op = LOG / "engine_job.json", LOG / "engine_out.json"
    jp.write_text(json.dumps(job))
    subprocess.run(["node", str(CLI), str(jp), str(op)], check=True)
    return json.loads(op.read_text())


def pt(t: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(t), PT)


def summary(trades: list[dict], n_sessions: int, cost=0.25) -> dict:
    if not trades:
        return {"trades": 0, "sessions": n_sessions}
    tr = sorted(trades, key=lambda x: (x["date"], x["entryT"]))
    pts = np.array([x["pts"] for x in tr])
    net = pts - cost
    wins = [x for x in tr if x["pts"] > 0]
    losses = [x for x in tr if x["pts"] <= 0]
    cum = np.cumsum(net)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])))
    streak = best = 0
    for x in tr:
        streak = streak + 1 if x["pts"] <= 0 else 0
        best = max(best, streak)
    return {
        "trades": len(tr), "sessions": n_sessions, "per_day": round(len(tr) / max(1, n_sessions), 2),
        "win_pct": round(100 * len(wins) / len(tr), 1), "net_gross": round(float(pts.sum()), 2), "net_after_cost": round(float(net.sum()), 2),
        "avg_win": round(float(np.mean([x["pts"] for x in wins])), 2) if wins else None,
        "avg_loss": round(float(np.mean([x["pts"] for x in losses])), 2) if losses else None,
        "max_drawdown": round(dd, 2), "longest_losing_streak": best,
        "exits": dict(Counter(x["reason"] for x in tr)),
    }


def race10(bars: dict, i0: int, entry: float, side: str) -> str:
    """Readout (A15): stop-free ±10 race from the fill minute's close onward on 1-min bars; same-minute both = loss."""
    h = -1 if side == "short" else 1
    for j in range(i0 + 1, len(bars["t"])):
        fav = (bars["h"][j] - entry) if h == 1 else (entry - bars["l"][j])
        adv = (entry - bars["l"][j]) if h == 1 else (bars["h"][j] - entry)
        if adv >= 10:
            return "loss"
        if fav >= 10:
            return "win"
    return "unresolved"


def fmt_table(rows: list[list], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("–" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def run():
    days = [d for d in sessions() if (SES / f"{d}.json").exists()]
    missing = [str(d) for d in sessions() if d not in days]
    files = [str(SES / f"{d}.json") for d in days]
    set2_dates, set2_done = set2_state()
    withheld = set() if set2_done else {d for d in days if d in set2_dates}
    cards = cz_cards()
    at = [{"id": cid, "file": str(SES / f"{d}.json"), "t": decision_bar(e)} for cid, d, _, e, _ in cards if (SES / f"{d}.json").exists()]
    warm_cards = [(cid, lb) for cid, d, _, e, lb in cards if not (SES / f"{d}.json").exists()]
    out = node({"variants": VARIANTS, "sessions": files, "minutes": True, "at": []})
    # CR-CZ overlap: the verdict at the card's decision minute with the engine's own state that day (simulate to there)
    ov = node({"variants": {"primary": {}}, "sessions": [], "at": at})
    res = {}
    for name in VARIANTS:
        allt = []
        for d in days:
            for x in out["sessions"][str(d)][name]["trades"]:
                allt.append({**x, "date": str(d)})
        res[name] = allt
    prim = res["primary"]
    lines = [f"# CR-DU v1 rules engine — train backtest ({LABEL})", "",
             f"Engine {out['version']} · {len(days)} sessions {days[0]} → {days[-1]} (Databento) · built {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"Missing session files: {missing or 'none'}. Set 2 {'reviewed' if set2_done else 'in progress'}: "
             f"{len(withheld)} sessions withheld from per-trade rows and monthly splits.", ""]
    # totals per variant, with and without set-2 sessions
    rows = []
    keep = [d for d in days if d not in withheld]
    for name, tr in res.items():
        for scope, ds in (("all sessions", days), ("without set-2 sessions", keep)):
            s = summary([x for x in tr if dt.date.fromisoformat(x["date"]) in set(ds)], len(ds))
            rows.append([name, scope, s["trades"], s.get("per_day"), s.get("win_pct"), s.get("net_gross"), s.get("net_after_cost"),
                         s.get("avg_win"), s.get("avg_loss"), s.get("max_drawdown"), s.get("longest_losing_streak"),
                         ", ".join(f"{k} {v}" for k, v in sorted((s.get("exits") or {}).items()))])
    lines += [f"## Totals ({LABEL})", "", fmt_table(rows, ["Variant", "Sessions", "Trades", "Trades / day", "Win %", "Net pts gross",
                                                         "Net pts after 0.25 / RT", "Avg win", "Avg loss", "Max DD", "Longest losing streak", "Exits"]), ""]
    kp = [x for x in prim if x["date"] not in {str(d) for d in withheld}]
    # splits (non-set-2 sessions)
    def split(key, title):
        g = defaultdict(list)
        for x in kp:
            g[key(x)].append(x)
        r = [[k, *(lambda s: [s["trades"], s.get("win_pct"), s.get("net_gross"), s.get("net_after_cost")])(summary(v, 0))] for k, v in sorted(g.items())]
        return [f"### By {title} (primary, non-set-2 sessions, {LABEL})", "", fmt_table(r, [title.capitalize(), "Trades", "Win %", "Net gross", "Net after cost"]), ""]
    lines += split(lambda x: x["side"], "side")
    lines += split(lambda x: f"{pt(x['entryT']):%H}:00", "hour (PT, entry)")
    lines += split(lambda x: x["date"][:7], "month")
    # readouts
    if kp:
        dist = np.array([abs(x["stop"] - x["entry"]) for x in kp])
        mfe = np.array([x["mfe"] for x in kp])
        lines += [f"### Readouts (primary, non-set-2 sessions, {LABEL})", "",
                  f"- Stop distance (rule 15) pts: p10 {np.percentile(dist, 10):.2f} · p50 {np.median(dist):.2f} · p90 {np.percentile(dist, 90):.2f} · max {dist.max():.2f}",
                  f"- MFE (from fill to exit, minute bars) pts: p50 {np.median(mfe):.2f} · p90 {np.percentile(mfe, 90):.2f}", ""]
        # stop-free ±10 race of the same entries (A15 readout)
        rc = Counter()
        for x in kp:
            b = json.loads((SES / f"{x['date']}.json").read_text())["payload"]
            bars = next(s for s in b["segments"] if s["name"] == "rth")["bars"]
            rc[race10(bars, int(x["fillI"]), x["entry"], x["side"])] += 1
        lines += [f"- Stop-free ±10 race of the same entries (A15 readout): {dict(rc)}", ""]
    # item 4 — contradiction bars at the decision (primary trades): with ≥ 1 vs none, primary and secondary definitions
    def contra_cmp(trades, key):
        a = [x for x in trades if x["contra"][key] >= 1]
        b = [x for x in trades if x["contra"][key] == 0]
        wa, wb = np.array([x["pts"] > 0 for x in a], float), np.array([x["pts"] > 0 for x in b], float)
        row = [len(a), round(100 * wa.mean(), 1) if len(a) else None, round(sum(x["pts"] - 0.25 for x in a), 2),
               len(b), round(100 * wb.mean(), 1) if len(b) else None, round(sum(x["pts"] - 0.25 for x in b), 2)]
        if len(a) >= 2 and len(b) >= 2:
            rng = np.random.default_rng(20261031)
            diffs = [rng.choice(wa, len(wa)).mean() - rng.choice(wb, len(wb)).mean() for _ in range(10_000)]
            lo, hi = np.percentile(diffs, [5, 95])
            verdict = "pass" if lo > 0 else "refuted" if hi < 0 else "inconclusive"
            row += [f"{100 * (wa.mean() - wb.mean()):+.1f} ({100 * lo:+.1f}, {100 * hi:+.1f})", verdict]
        else:
            row += ["n too small", "inconclusive"]
        return row
    crow = []
    for scope, tr in (("all sessions", prim), ("without set-2 sessions", kp)):
        for key, nm in (("primary", "primary (p10 / p90 delta, wick ≥ 50 %, vol ≥ p80, range ≥ 2)"),
                        ("secondary", "secondary (p20 / p80 delta, wick ≥ 40 %, no volume floor)")):
            crow.append([scope, nm, *contra_cmp(tr, key)])
    lines += [f"## Contradiction bars in the box at the faded edge (item 4; primary engine trades, {LABEL})", "",
              fmt_table(crow, ["Sessions", "Definition", "n with", "Win % with", "Net with (after cost)", "n none", "Win % none",
                               "Net none (after cost)", "Win-rate diff, pts (90 % bootstrap CI)", "B1"]),
              "", "Small n: a difference this size can't be separated from chance here; the secondary definition is report-only. "
              "Bars are counted from the box start through the decision minute, on the edge being faded (bar low / high within 2 pts "
              "of the box's wick low / high); percentiles per 30-min RTH bucket from prior sessions (Databento).", ""]
    # no-trade reasons (all sessions; no outcomes)
    rc = Counter()
    for d in days:
        for m in out["sessions"][str(d)]["primary"]["minutes"]:
            if m["failed"]:
                rc[m["failed"][0]] += 1
            else:
                rc[f"verdict {m['verdict']}"] += 1
    lines += [f"## Minute verdicts: first failed rule (primary, all sessions, no outcomes)", "",
              fmt_table([[k, v] for k, v in rc.most_common()], ["First failed rule / verdict", "Minutes"]), ""]
    # CR-CZ overlap (counts only)
    lab = {cid: (lb or "unlabelled", dr) for cid, _, dr, _, lb in cards}
    oc = defaultdict(Counter)
    for a in ov["at"]:
        lb, dr = lab[a["id"]]
        same = a.get("side") == dr and a.get("verdict") in ("SHORT", "LONG")
        oc[lb]["engine takes (same side)" if same else "engine: NO ACTION" if a.get("verdict") == "NO ACTION" else "engine: other side" if a.get("verdict") else "no bar"] += 1
    for _, lb in warm_cards:
        oc[lb or "unlabelled"]["warm-up session (not evaluated)"] += 1
    cols = ("engine takes (same side)", "engine: other side", "engine: NO ACTION", "no bar", "warm-up session (not evaluated)")
    lines += ["## Overlap with CR-CZ cards (verdict at the card's decision minute; counts only, no outcomes)", "",
              fmt_table([[k, sum(v.values()), *[v.get(c, 0) for c in cols]] for k, v in sorted(oc.items())],
                        ["Ryan's label", "Cards", "Engine takes (same side)", "Engine: other side", "Engine: NO ACTION", "No bar",
                         "Warm-up session (not evaluated)"]), ""]
    na = Counter()
    for a in ov["at"]:
        if a.get("verdict") == "NO ACTION" and a.get("failed"):
            na[a["failed"][0]["code"]] += 1
    lines += ["First failed rule on those NO ACTION cards: " + ", ".join(f"{k} {v}" for k, v in na.most_common()), ""]
    # per-trade rows (non-set-2 sessions)
    tr_rows = [[x["date"], x["side"], f"{pt(x['box']['from']):%H:%M} {x['box']['bottom']:.2f}–{x['box']['top']:.2f}",
                f"{pt(x['decisionT']):%H:%M}", f"{pt(x['entryT']):%H:%M:%S}", f"{x['entry']:.2f}", f"{x['stop']:.2f}", f"{x['target']:.2f}",
                f"{pt(x['exitT']):%H:%M:%S}", f"{x['exit']:.2f}", x["reason"], f"{x['pts']:+.2f}",
                f"contra {x['contra']['primary']}/{x['contra']['secondary']} · loc {x['locationSupport']} · r12 {'lighter' if x['rule12'].get('lighter') else 'seen' if x['rule12'].get('seen') else '–'}"]
               for x in sorted(kp, key=lambda x: (x["date"], x["entryT"]))]
    lines += [f"## Trades (primary, non-set-2 sessions, {LABEL})", "",
              fmt_table(tr_rows, ["Date", "Side", "Box (start, bottom–top)", "Decision", "Entry time", "Entry", "Stop", "Target", "Exit time", "Exit", "Exit by", "Pts", "Evidence"]), ""]
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "backtest_report.md").write_text("\n".join(lines))
    (LOG / "backtest_results.json").write_text(json.dumps({"label": LABEL, "version": out["version"], "days": [str(d) for d in days],
                                                           "withheld": sorted(str(d) for d in withheld), "trades": res}, default=str))
    print("\n".join(lines[:4 + len(rows) + 6]))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
