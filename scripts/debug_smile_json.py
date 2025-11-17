# scripts/debug_smile_json.py

import json
import pandas as pd
from sqlalchemy import text

from packages.shared.utils.data_io import tx
from packages.shared.options_orats import pt_minute_to_et, ET_TZ

trade_date = "2025-11-17"
expiry = "2025-12-19"
minutes_pt = ["12:00", "12:17"]


def main() -> None:
    for hhmm in minutes_pt:
        print(f"\n=== RAW DB ROWS FOR {trade_date} {hhmm} PT ===")

        # Convert PT time → ET → UTC window [floor, floor+60s)
        ts_et = pt_minute_to_et(trade_date, hhmm)
        ts = pd.Timestamp(ts_et)
        if ts.tz is None:
            ts = ts.tz_localize(ET_TZ)
        ts_floor_utc = ts.tz_convert("UTC").floor("min")
        ts_end_utc = ts_floor_utc + pd.Timedelta(minutes=1)

        params = {
            "t": "SPX",
            "d": trade_date,
            "e": expiry,
            "s": ts_floor_utc.to_pydatetime(),
            "u": ts_end_utc.to_pydatetime(),
        }

        sql = """
          SELECT minute_ts, trade_date, expiry_date, grid, atm_iv, smile
            FROM orats_monies_minute
           WHERE ticker = :t
             AND trade_date = :d
             AND expiry_date = :e
             AND minute_ts >= :s
             AND minute_ts <  :u
           ORDER BY minute_ts;
        """

        with tx() as conn:
            recs = conn.execute(text(sql), params).mappings().all()

        if not recs:
            print("  <no rows found in orats_monies_minute for this minute>")
            continue

        print(f"  rows found: {len(recs)}")

        for i, r in enumerate(recs, start=1):
            print(f"\n  --- row {i} ---")
            print("  minute_ts:", r["minute_ts"])
            print("  trade_date:", r["trade_date"], " expiry_date:", r["expiry_date"])
            print("  grid:", r["grid"], " atm_iv:", r["atm_iv"])

            smile = r["smile"]
            if isinstance(smile, str):
                try:
                    smile = json.loads(smile)
                except Exception as e:
                    print("  ERROR decoding JSON:", e)
                    print("  raw smile:", r["smile"])
                    continue

            if not isinstance(smile, dict):
                print("  smile is not a dict:", smile)
                continue

            keys = sorted(smile.keys())
            print("  smile keys:", keys[:12], ("..." if len(keys) > 12 else ""))

            def safe_get(k):
                v = smile.get(k)
                return None if v is None else float(v)

            print("  smile['atm']  :", safe_get("atm"))
            print("  smile['vol90']:", safe_get("vol90"))
            print("  smile['vol50']:", safe_get("vol50"))
            print("  smile['vol10']:", safe_get("vol10"))


if __name__ == "__main__":
    main()
