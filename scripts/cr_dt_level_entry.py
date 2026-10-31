#!/usr/bin/env python3
"""CR-DT — would a resting limit at a nearby GEX level have beaten the card's edge limit? READ-ONLY, learning set.

Spec: specs/CR-DT-gex-level-entry.md (T1-T15). Cards = CR-CZ set `cr-cz:cp-ranges` (119, card 31 dropped).

  step0  level function per card session, sessions without corrected levels, coverage per variant. No outcome read.
  run    fills and +-10 races for the level variants and the edge baseline; baseline asserted equal to the cards.

Bars, fill loop and race are cr_cp_lib's (the cards' own); levels are the React chart's corrected lines
(_corrected_gex_tables -> _select_levels at 40 $bn gross, band cut at the card's entry_ts_pt). Writes only
scripts/logs/cr_dt/ (git-ignored). DB errors print the error type only.

  apps/web/.venv/bin/python scripts/cr_dt_level_entry.py step0|run
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cp_lib as C  # noqa: E402  (loads .env, repo + scripts on sys.path)

sys.path.insert(0, str(C.REPO_ROOT / "apps" / "web"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SET_NAME, DROP_ID = "cr-cz:cp-ranges", "CR-CZ:cr-cz:cp-ranges:31"
GROSS_MIN = 40e9
VARIANTS = ("D2", "D4", "D6", "inside2")
LEVEL_SETS = ("gross40", "a12")
GROUPS = ("takes", "skips", "all")
X, SEED, N_BOOT = 10.0, 20261031, 10_000
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_dt"
OUT.mkdir(parents=True, exist_ok=True)
H = {"short": -1, "long": 1}


def load_cards(with_outcome: bool) -> pd.DataFrame:
    cols = "c.id, c.session_date, c.direction, c.entry_ts_pt, c.entry_price, c.meta" + (", c.outcome" if with_outcome else "")
    cards = C.q(f"SELECT {cols} FROM setup_candidates c WHERE c.set_name = %s ORDER BY c.id", (SET_NAME,))
    lab = C.q("""
        SELECT r.candidate_id AS id,
               (array_agg(r.label ORDER BY r.reviewed_at, r.id) FILTER (WHERE NOT r.post_reveal))[1] AS label,
               (array_agg(array_to_string(r.reasons, '|') ORDER BY r.reviewed_at, r.id) FILTER (WHERE NOT r.post_reveal))[1] AS reasons
        FROM setup_reviews r JOIN setup_candidates c ON c.id = r.candidate_id
        WHERE c.set_name = %s GROUP BY r.candidate_id
    """, (SET_NAME,))
    cards = cards.merge(lab, on="id", how="left")
    assert len(cards) == 120 and DROP_ID in set(cards["id"]), len(cards)
    cards = cards[cards["id"] != DROP_ID].reset_index(drop=True)
    cards["n"] = cards["id"].str.rsplit(":", n=1).str[1].astype(int)
    cards["d"] = pd.to_datetime(cards["session_date"]).dt.date
    assert cards["d"].max() <= date(2026, 6, 5)                                  # B4
    cards["h"] = cards["direction"].map(H)
    cards["top"] = cards["meta"].map(lambda m: float(m["range"]["top"]))
    cards["bottom"] = cards["meta"].map(lambda m: float(m["range"]["bottom"]))
    et = pd.to_datetime(cards["entry_ts_pt"])                                    # close of decision bar t, naive PT
    cards["t"] = ((et.dt.hour * 60 + et.dt.minute) - 390 - 1).astype(int)
    cards["until_pt"] = et
    return cards.sort_values("n").reset_index(drop=True)


def drawn_levels(cards: pd.DataFrame):
    """T4-T6: per card, the chart's drawn corrected levels [(L, net, gross, strong)], or the session's reason."""
    from modules.Ironbeam import callbacks as IB
    pt_tz = C.PT
    tables, reasons = {}, {}
    for d in sorted(set(cards["d"])):
        try:
            df_c, _, why = IB._corrected_gex_tables(d)
        except Exception as e:                                                   # error type only (no URLs)
            df_c, why = None, f"error: {type(e).__name__}"
        tables[d], reasons[d] = df_c, why
    LV = {}
    for r in cards.itertuples():
        if tables[r.d] is None:
            LV[r.n] = None
            continue
        start, end = IB._session_window_pt(r.d, pt_tz)
        end = min(end, r.until_pt.to_pydatetime().replace(tzinfo=pt_tz))
        bars = IB._fetch_bars_pt(start, end, "1m", pt_tz)
        lo = float(bars["low"].min()) - float(IB.GEX_LEVEL_PADDING)
        hi = float(bars["high"].max()) + float(IB.GEX_LEVEL_PADDING)
        df = IB._select_levels(tables[r.d], lo, hi, GROSS_MIN)
        if df is None or df.empty:
            LV[r.n] = []
            continue
        _, span = IB._compute_color_span(df["net_gamma"].to_numpy(dtype=float))
        LV[r.n] = [dict(L=float(x.level), net=float(x.net_gamma),
                        gross=abs(float(x.call_gamma)) + abs(float(x.put_gamma)),
                        strong=bool(abs(float(x.net_gamma)) >= span))
                   for x in df.itertuples()]
    return LV, reasons


def pick(levels, r, variant: str, strong_only: bool):
    """T7-T8: the level the variant enters at, or None."""
    if not levels:
        return None
    Ls = [x["L"] for x in levels if (x["strong"] or not strong_only)]
    if r.h == -1:
        E = r.top
        if variant == "inside2":
            c = [L for L in Ls if E - 2 <= L < E]
            return max(c) if c else None
        D = float(variant[1:])
        c = [L for L in Ls if E <= L <= E + D]
        return min(c) if c else None
    E = r.bottom
    if variant == "inside2":
        c = [L for L in Ls if E < L <= E + 2]
        return min(c) if c else None
    D = float(variant[1:])
    c = [L for L in Ls if E - D <= L <= E]
    return max(c) if c else None


def group_of(label) -> list[str]:
    g = ["all"]
    if label == "take":
        g.append("takes")
    elif label == "skip":
        g.append("skips")
    return g


def step0():
    cards = load_cards(with_outcome=False)
    LV, reasons = drawn_levels(cards)
    no_lv = {str(d): w for d, w in reasons.items() if w is not None}
    cov = {}
    for ls in LEVEL_SETS:
        for v in VARIANTS:
            for g in GROUPS:
                sub = cards[cards["label"].map(lambda x: g in group_of(x))]
                cov[f"{ls}/{v}/{g}"] = dict(
                    n_cards=int(len(sub)),
                    n_with_level=int(sum(pick(LV[r.n], r, v, ls == "a12") is not None for r in sub.itertuples())))
    doc = dict(
        n_cards=int(len(cards)), n_sessions=int(cards["d"].nunique()),
        sessions=[str(d) for d in sorted(set(cards["d"]))],
        labels=cards["label"].fillna("none").value_counts().to_dict(),
        directions=cards["direction"].value_counts().to_dict(),
        level_function_reasons={str(d): (w or "ok") for d, w in reasons.items()},
        sessions_without_corrected_levels=no_lv,
        cards_excluded_no_levels=int(sum(LV[n] is None for n in cards["n"])),
        cards_zero_drawn_levels=int(sum(LV[n] == [] for n in cards["n"])),
        drawn_levels_per_card=dict(pd.Series([len(LV[n]) for n in cards["n"] if LV[n]]).describe().round(1)),
        strong_per_card=dict(pd.Series([sum(x["strong"] for x in LV[n]) for n in cards["n"] if LV[n]]).describe().round(1)),
        coverage=cov,
    )
    (OUT / "step0.json").write_text(json.dumps(doc, indent=1, default=str))
    (OUT / "levels.json").write_text(json.dumps({str(k): v for k, v in LV.items()}, indent=0))
    print(json.dumps(doc, indent=1, default=str))


# ------------------------------------------------------------------------------------------------------------ run
def ts(d, i) -> str:
    """Naive PT ISO of RTH bar i's start (as cr_cz_build_gallery.ts)."""
    return (datetime.combine(d, time(6, 30)) + timedelta(minutes=int(i))).isoformat()


def entry(B, r, px):
    """T9 / T10: cr_cp_lib.fill with the limit at px (same loop, same far-break cancel)."""
    rg = dict(hi=r.top, lo=r.bottom)
    p = C.PRIMARY
    for j in range(r.t + 1, C.NB):
        touch = B["l"][j] <= px if r.h == 1 else B["h"][j] >= px
        far_break = (B["c"][j] > rg["hi"] + p.void) if r.h == 1 else (B["c"][j] < rg["lo"] - p.void)
        if far_break:
            return "no fill: box broke", None
        if touch:
            return "fill", j
    return "no fill: 13:00", None


def trade(B, r, px):
    st, j = entry(B, r, px)
    if st != "fill":
        return dict(status=st, px=px, res=None, net=0.0, net_mtc=0.0)
    res = C.race_from(B, j, px, r.h, X)
    net = {"win": X, "loss": -X, "tie": -X}.get(res, 0.0)
    mtc = net if res != "unresolved" else r.h * (float(B["c"][C.NB - 1]) - px)
    return dict(status=st, px=px, j=int(j), res=res, net=net, net_mtc=round(mtc, 2))


def wilson(w, n):
    if n == 0:
        return (None, None)
    z, p = 1.96, w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - hw), 1), round(100 * (c + hw), 1))


def summ(T):
    fills = [t for t in T if t["status"] == "fill"]
    w = sum(t["res"] == "win" for t in fills)
    lo = sum(t["res"] in ("loss", "tie") for t in fills)
    return dict(n=len(T), fills=len(fills), won=w, lost=lo, ties=sum(t["res"] == "tie" for t in fills),
                unresolved=sum(t["res"] == "unresolved" for t in fills),
                win_pct=round(100 * w / (w + lo), 1) if w + lo else None, ci95=wilson(w, w + lo),
                net=sum(t["net"] for t in T), net_mtc=round(sum(t["net_mtc"] for t in T), 2))


def verdict(diff, rng):
    if len(diff) == 0:
        return dict(n=0, mean=None, ci95=None, verdict="no cards")
    a = np.asarray(diff, float)
    bs = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    v = "pass" if lo > 0 else "refuted" if hi < 0 else "inconclusive"
    return dict(n=len(a), mean=round(float(a.mean()), 2), ci95=(round(float(lo), 2), round(float(hi), 2)), verdict=v)


def run():
    cards = load_cards(with_outcome=True)
    LV = {int(k): v for k, v in json.loads((OUT / "levels.json").read_text()).items()}
    D = C.build(C.load_minutes(), C.load_im())
    rows = []
    for r in cards.itertuples():
        assert r.d in D["BARS"], r.d
        B = D["BARS"][r.d]
        edge_px = r.top - 1.0 if r.h == -1 else r.bottom + 1.0
        assert abs(edge_px - float(r.entry_price)) < 1e-6, (r.n, edge_px, r.entry_price)
        base = trade(B, r, edge_px)
        oc = r.outcome["called"]                                                  # T10: the card's stored outcome
        assert (base["status"] == "fill") == (oc["fill_status"] == "fill"), (r.n, base, oc)
        if base["status"] == "fill":
            assert abs(oc["fill_price"] - edge_px) < 0.011 and base["res"] == oc["race_10"]["result"], (r.n, base, oc)
            assert ts(r.d, base["j"]) == oc["fill_ts_pt"], (r.n, base, oc)
        row = dict(n=r.n, d=str(r.d), dir=r.direction, label=r.label, top=r.top, bottom=r.bottom, t=r.t,
                   edge=base, lv=LV[r.n] is not None)
        for ls in LEVEL_SETS:
            for v in VARIANTS:
                L = pick(LV[r.n], r, v, ls == "a12") if LV[r.n] is not None else None
                row[f"{ls}/{v}"] = None if L is None else trade(B, r, L)
        rows.append(row)
    (OUT / "trades.json").write_text(json.dumps(rows, indent=0, default=str))

    rng = np.random.default_rng(SEED)
    res = dict(baseline={}, cells={})
    for g in GROUPS:
        sub = [x for x in rows if g in group_of(x["label"])]
        res["baseline"][g] = summ([x["edge"] for x in sub])
    for ls in LEVEL_SETS:
        for v in VARIANTS:
            for g in GROUPS:
                sub = [x for x in rows if g in group_of(x["label"]) and x[f"{ls}/{v}"] is not None]
                lvl, edg = [x[f"{ls}/{v}"] for x in sub], [x["edge"] for x in sub]
                pair = dict(both=0, better=0, same=0, worse=0, level_only=0, edge_only=0, neither=0,
                            level_only_net=0.0, edge_only_net=0.0)
                for a, b in zip(lvl, edg):
                    fa, fb = a["status"] == "fill", b["status"] == "fill"
                    if fa and fb:
                        pair["both"] += 1
                        k = "better" if a["net"] > b["net"] else "worse" if a["net"] < b["net"] else "same"
                        pair[k] += 1
                    elif fa:
                        pair["level_only"] += 1
                        pair["level_only_net"] += a["net"]
                    elif fb:
                        pair["edge_only"] += 1
                        pair["edge_only_net"] += b["net"]
                    else:
                        pair["neither"] += 1
                res["cells"][f"{ls}/{v}/{g}"] = dict(
                    n_cards_group=len([x for x in rows if g in group_of(x["label"])]),
                    level=summ(lvl), edge_same_cards=summ(edg), paired=pair,
                    b1=verdict([a["net"] - b["net"] for a, b in zip(lvl, edg)], rng))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    try:
        {"step0": step0, "run": run}[sys.argv[1]]()
    except Exception as e:
        if type(e).__module__.split(".")[0] in ("psycopg", "sqlalchemy", "psycopg2"):
            print(f"DB error: {type(e).__name__}")
            sys.exit(2)
        raise
