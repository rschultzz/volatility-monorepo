#!/usr/bin/env python3
"""CR-BR extraction — per-day term structure, slopes, quintiles and outcomes (READ-ONLY, train only).

Spec: specs/CR-BR-term-structure-slope.md (L1–L4, Step 0, amendment A1). No statistic across quintiles here.
Output: scripts/output/cr_br/days.parquet — one row per x-complete train day (778): IV at 7/14/30/60/90 d,
  pc2_log (A1.1 primary), s_t (secondary), pc2_lvl (frozen, failed sign check), quintiles of each, inversion
  indicator, log IV7², universe flags, h = 1 / h = 5 outcomes (raw + level-neutral), the shifted-outcome placebo.
Gate (A1.1): PC2_log(2024-05-17) > 0 and PC2_log(2025-04-08) < 0, else abort.

Usage: apps/web/.venv/bin/python scripts/cr_br_extract.py
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import cr_br_common as R

GATE_UP, GATE_INV = dt.date(2024, 5, 17), dt.date(2025, 4, 8)


def pca2(M: np.ndarray):
    """PC2 scores, loadings, variance shares; sign fixed: loading(90 d) − loading(7 d) > 0 (positive = upward)."""
    m0 = M.mean(0)
    e, W = np.linalg.eigh(np.cov(M - m0, rowvar=False))
    o = np.argsort(e)[::-1]
    e, W = e[o], W[:, o]
    if W[-1, 1] - W[0, 1] < 0:
        W[:, 1] *= -1
    return (M - m0) @ W[:, 1], W[:, 1], e / e.sum()


def resid(y: pd.Series, x: pd.Series) -> pd.Series:
    m = y.notna() & x.notna()
    X = np.column_stack([np.ones(m.sum()), x[m].to_numpy()])
    b = np.linalg.lstsq(X, y[m].to_numpy(), rcond=None)[0]
    out = pd.Series(np.nan, index=y.index)
    out[m] = y[m].to_numpy() - X @ b
    return out, b


def main() -> None:
    conn = R.connect()
    days = R.sessions()
    snaps = R.load_snapshots(conn, days)
    rows = []
    for d in days:
        sp, an, _ = snaps[d]
        r = {"t": d, "snapshot_pt": sp}
        for T in R.TENORS:
            a = R.cm_iv(an, T)
            r[f"iv{T}"] = a[0] if a else np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    D = D[D[[f"iv{T}" for T in R.TENORS]].notna().all(axis=1)].reset_index(drop=True)
    assert len(D) == 778, len(D)
    IV = D[[f"iv{T}" for T in R.TENORS]].to_numpy()

    D["pc2_log"], w_log, sh_log = pca2(np.log(IV ** 2))
    D["pc2_lvl"], w_lvl, sh_lvl = pca2(IV ** 2)
    D["s"] = 100 * (D.iv30 - D.iv7)
    up = float(D.loc[D.t == GATE_UP, "pc2_log"].iloc[0]); inv = float(D.loc[D.t == GATE_INV, "pc2_log"].iloc[0])
    print(f"sign gate: PC2_log {GATE_UP} = {up:+.5f} (must be > 0); {GATE_INV} = {inv:+.5f} (must be < 0)")
    assert up > 0 and inv < 0, "A1.1 sign-check gate FAILED — stop"
    for c in ("pc2_log", "s", "pc2_lvl"):
        D[f"q_{c}"] = pd.qcut(D[c], 5, labels=False) + 1
    D["inv"] = (D.iv90 < D.iv7).astype(int)
    D["log_iv7sq"] = np.log(D.iv7 ** 2)
    D["year"] = [t.year for t in D.t]

    # ── ES outcomes ──────────────────────────────────────────────────────────
    es = R.load_es(conn)
    nights, switches = R.es_nights(es, R.load_switch_px(conn))
    pos = {d: i for i, d in enumerate(days)}

    def ok(d):
        return d in nights and nights[d]["ok"]

    def win5(d):
        i = pos[d]
        ws = days[i:i + 5]
        return ws if len(ws) == 5 and all(ok(w) for w in ws) else None

    def win5_prev(d):
        i = pos[d]
        ws = days[i - 5:i] if i >= 5 else []
        return ws if len(ws) == 5 and all(ok(w) for w in ws) else None

    iv7sq = D.iv7 ** 2
    D["in_h1"] = [ok(d) for d in D.t]
    D["RV1"] = [nights[d]["RV"] if ok(d) else np.nan for d in D.t]
    D["y1"] = np.log(D.RV1 / (iv7sq / 365))
    W5 = [win5(d) for d in D.t]
    D["in_h5"] = [w is not None for w in W5]
    D["RV5"] = [sum(nights[x]["RV"] for x in w) if w else np.nan for w in W5]
    D["y5"] = np.log(D.RV5 / (iv7sq * 5 / 365))
    # placebo (ii): outcome shifted one day earlier, slope from t
    prev = [days[pos[d] - 1] if pos[d] > 0 else None for d in D.t]
    D["RV1_prev"] = [nights[p]["RV"] if (p is not None and ok(p)) else np.nan for p in prev]
    D["y1_prev"] = np.log(D.RV1_prev / (iv7sq / 365))
    W5p = [win5_prev(d) for d in D.t]
    D["RV5_prev"] = [sum(nights[x]["RV"] for x in w) if w else np.nan for w in W5p]
    D["y5_prev"] = np.log(D.RV5_prev / (iv7sq * 5 / 365))
    D.loc[~D.in_h1, ["y1_prev"]] = np.nan          # placebo on the h = 1 universe
    D.loc[~D.in_h5, ["y5_prev"]] = np.nan          # placebo on the h = 5 universe
    D.loc[~D.in_h1, ["y1"]] = np.nan
    D.loc[~D.in_h5, ["y5"]] = np.nan
    coefs = {}
    for c in ("y1", "y5", "y1_prev", "y5_prev"):
        D[f"{c}_ln"], b = resid(D[c], D.log_iv7sq)
        coefs[c] = [float(x) for x in b]

    assert int(D.in_h1.sum()) == 621 and int(D.in_h5.sum()) == 515, (D.in_h1.sum(), D.in_h5.sum())
    R.OUT_DIR.mkdir(parents=True, exist_ok=True)
    D.to_parquet(R.OUT_DIR / "days.parquet")
    meta = {"loadings_log": dict(zip(map(str, R.TENORS), map(float, w_log))), "var_share_log": [float(x) for x in sh_log[:3]],
            "loadings_lvl": dict(zip(map(str, R.TENORS), map(float, w_lvl))), "var_share_lvl": [float(x) for x in sh_lvl[:3]],
            "gate": {str(GATE_UP): up, str(GATE_INV): inv}, "level_neutral_coefs": coefs,
            "n": {"x_complete": len(D), "h1": int(D.in_h1.sum()), "h5": int(D.in_h5.sum()),
                  "h1_prev": int(D.y1_prev.notna().sum()), "h5_prev": int(D.y5_prev.notna().sum())}}
    (R.OUT_DIR / "extract_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta["n"]), "| PC2_log loadings", {k: round(v, 3) for k, v in meta["loadings_log"].items()},
          "share", [round(x, 4) for x in meta["var_share_log"]])
    print(f"y1 finite {int(np.isfinite(D.y1).sum())}, y5 finite {int(np.isfinite(D.y5).sum())}; saved {R.OUT_DIR / 'days.parquet'}")


if __name__ == "__main__":
    main()
