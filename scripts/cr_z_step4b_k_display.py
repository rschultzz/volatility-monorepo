"""CR-031 Step 4b — verify compute_structural_probability result["k"] is real count.

The fix: result["k"] = len(rows), not the safety bound (200).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
import psycopg
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.probability import compute_structural_probability

def _conn():
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return psycopg.connect(url)

def main():
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector FROM bt_daily_features_active "
            "WHERE ticker='SPX' AND feature_version=%s "
            "AND trade_date IN ('2026-04-29', '2026-05-07')",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()
    corpus = {d.isoformat(): v for (d, v) in rows}

    for anchor_date, desc in [("2026-04-29", "common magnet-above"), ("2026-05-07", "rare K=1")]:
        fv = corpus.get(anchor_date)
        if fv is None:
            print(f"{anchor_date}: not in corpus"); continue
        result = compute_structural_probability(fv, conn, exclude_date=anchor_date)
        k_displayed = result["k"]
        k_outcomes  = result["k_with_outcomes"]
        ci_lo = result.get("touch_ci_lower")
        ci_hi = result.get("touch_ci_upper")
        ci_w  = round(ci_hi - ci_lo, 4) if ci_lo is not None and ci_hi is not None else None
        print(f"\n{anchor_date} ({desc})")
        print(f"  result['k']              = {k_displayed}  ← should be real count, not 200")
        print(f"  result['k_with_outcomes']= {k_outcomes}")
        print(f"  CI=[{ci_lo},{ci_hi}]  width={ci_w}")
        assert k_displayed != 200 or k_outcomes == 200, (
            f"result['k'] is still 200 — fix not applied!"
        )
        if k_displayed == k_outcomes:
            print(f"  ✓ k == k_with_outcomes (no pending/na attrition on this date)")
        else:
            print(f"  ✓ k ({k_displayed}) > k_with_outcomes ({k_outcomes}) "
                  f"— attrition from pending_history/na_data shown in note")

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
