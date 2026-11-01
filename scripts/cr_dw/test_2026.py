"""CR-DW test (amendment 1): close-pin on 2026 train days, primary + secondaries + readouts.

Levels / opens / IM come from Step 0 (scripts/logs/cr_dw/step0_2026_days.json, pre-open band). This is the first
script that reads settlements. B4 guard: every date in 2026-01-01 .. 2026-06-05.
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "apps/web"), str(ROOT / "scripts/cr_dw")]
from step0_2026 import D0, D1, OUT, cb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared import spx_cash  # noqa: E402

FOMC = {"2026-01-28", "2026-03-18", "2026-04-29"}
SEED, N_PERM, N_BOOT, N_BOOT_PERM = 20261031, 10_000, 2_000, 1_000


def wilson(k, n, z=1.6449):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def dist(lv, x):
    """|x - nearest level| for x of any shape; lv a 1-d array."""
    x = np.asarray(x, float)
    return np.min(np.abs(x[..., None] - lv), axis=-1)


def move_swap(days, z_pool, rng, n):
    """(n, len(days)) matrix of |settle* - nearest| / IM with settle* = open + IM * z, z from z_pool."""
    Z = rng.choice(z_pool, (n, len(days)))
    out = np.empty((n, len(days)))
    for j, d in enumerate(days):
        out[:, j] = dist(d["lv"], d["open"] + d["im"] * Z[:, j]) / d["im"]
    return out, Z


def analyse(days, rng, boot=False):
    if not days:
        return {"n_days": 0}
    im = np.array([d["im"] for d in days])
    o = np.array([d["open"] for d in days])
    s = np.array([d["settle"] for d in days])
    ds = np.array([dist(d["lv"], d["settle"]) for d in days])
    do = np.array([dist(d["lv"], d["open"]) for d in days])
    z = (s - o) / im
    M, M_pts = np.median(ds / im), np.median(ds)
    null, Z = move_swap(days, z, rng, N_PERM)
    null_M = np.median(null, axis=1)
    null_M_pts = np.median(null * im, axis=1)
    p = (np.sum(null_M <= M) + 1) / (N_PERM + 1)
    F = float(np.mean(ds < do))
    null_F = np.mean(null * im < do, axis=1)
    p_b = (np.sum(null_F >= F) + 1) / (N_PERM + 1)
    # null (a) readout: whole level set shifted by U(-IM, IM)
    null_a = np.empty((N_PERM, len(days)))
    for j, d in enumerate(days):
        sh = rng.uniform(-d["im"], d["im"], N_PERM)
        null_a[:, j] = np.min(np.abs(d["settle"] - (d["lv"][None, :] + sh[:, None])), axis=1) / d["im"]
    res = {
        "n_days": len(days),
        "M_obs_IM": M, "M_null_median_IM": float(np.median(null_M)), "effect_IM": M - float(np.median(null_M)),
        "M_obs_pts": M_pts, "M_null_median_pts": float(np.median(null_M_pts)),
        "effect_pts": M_pts - float(np.median(null_M_pts)), "p_move_swap": p,
        "b_fraction": F, "b_wilson90": wilson(int(round(F * len(days))), len(days)),
        "b_null_median": float(np.median(null_F)), "b_null_p05_p95": [float(np.percentile(null_F, 5)), float(np.percentile(null_F, 95))],
        "b_p_vs_null": p_b, "b_vs_50pct_readout": "above" if F > 0.5 else "not above",
        "readout_null_a": {"M_null_median_IM": float(np.median(np.median(null_a, axis=1))),
                           "p": (np.sum(np.median(null_a, axis=1) <= M) + 1) / (N_PERM + 1),
                           "note": "null (a) false-pin rate 23 % under no pin (Step 0); not a valid test"},
        "median_dist_open_pts": float(np.median(do)), "median_abs_move_IM": float(np.median(np.abs(z))),
    }
    if boot:
        eff = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, len(days), len(days))
            sub = [days[i] for i in idx]
            nb, _ = move_swap(sub, z[idx], rng, N_BOOT_PERM)
            eff[b] = np.median((ds / im)[idx]) - np.median(np.median(nb, axis=1))
        lo, hi = np.percentile(eff, [5, 95])
        res["effect_IM_boot90"] = [float(lo), float(hi)]
        res["verdict"] = ("PASS" if p < 0.05 and hi < 0 else "REFUTED" if lo >= 0 else "INCONCLUSIVE")
    return res


def main():
    recs = json.loads((OUT / "step0_2026_days.json").read_text())
    assert all(D0 <= dt.date.fromisoformat(r["date"]) <= D1 for r in recs)
    raw = cb.engine.raw_connection()
    try:
        clean, _ = spx_cash.build_series(spx_cash.fetch_minutes(raw, D0, D1))
    finally:
        raw.close()
    settle = spx_cash.settlement_prints(clean)
    ohlc = spx_cash.daily_ohlc(clean)
    rng = np.random.default_rng(SEED)
    out = {}
    for pl in ("cash_corrected|pre", "strike|pre"):
        days = []
        for r in recs:
            d = dt.date.fromisoformat(r["date"])
            lv = np.array([l for l, _ in r[pl]], float)
            if lv.size == 0:
                continue
            near = [(l, k) for l, k in r[pl] if abs(l - r["open"]) <= 0.5 * r["im"]]
            kinds = {k for _, k in near}
            days.append({"date": r["date"], "im": r["im"], "open": r["open"], "settle": float(settle[d]), "lv": lv,
                         "cond": bool(near), "kind": "/".join(sorted(kinds)), "fomc": r["date"] in FOMC,
                         "range_IM": float((ohlc.loc[d, "high"] - ohlc.loc[d, "low"]) / r["im"])})
        cond = [d for d in days if d["cond"] and not d["fomc"]]
        other = [d for d in days if not d["cond"] and not d["fomc"]]
        primary = pl == "cash_corrected|pre"
        out[pl] = {
            "condition": analyse(cond, rng, boot=primary),
            "non_condition": analyse(other, rng),
            "fomc_cell": {"n": sum(d["fomc"] for d in days), "condition": sum(d["fomc"] and d["cond"] for d in days),
                          "dist_settle_IM": [round(float(dist(d["lv"], d["settle"]) / d["im"]), 3) for d in days if d["fomc"]]},
            "a8_yellow_only": analyse([d for d in cond if d["kind"] == "yellow"], rng),
            "a8_darkest_only": analyse([d for d in cond if d["kind"] == "darkest"], rng),
            "range_IM": {"condition_median": float(np.median([d["range_IM"] for d in cond])),
                         "other_median": float(np.median([d["range_IM"] for d in other])),
                         "condition_mean": float(np.mean([d["range_IM"] for d in cond])),
                         "other_mean": float(np.mean([d["range_IM"] for d in other])),
                         "n": [len(cond), len(other)]},
            "no_levels_days": len(recs) - len(days),
        }
        pd.DataFrame([{k: v for k, v in d.items() if k != "lv"} | {"lv": list(d["lv"])} for d in days]).to_csv(
            OUT / f"test_days_{pl.split('|')[0]}.csv", index=False)
    (OUT / "test_2026.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
