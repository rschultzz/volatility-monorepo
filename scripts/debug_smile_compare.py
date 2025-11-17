# scripts/debug_smile_compare.py

from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et
from packages.shared.ingest.monies_ingest import (
    upsert_from_dashboard_minute,
    read_minute_expiry_df_from_db,
)

trade_date = "2025-11-17"
expiry = "2025-12-19"
minutes_pt = ["12:00", "12:17"]


def main() -> None:
    for hhmm in minutes_pt:
        print(f"\n=== {trade_date} {hhmm} PT ===")

        ts_et = pt_minute_to_et(trade_date, hhmm)

        # 1) Live ORATS data for this minute
        df_api = fetch_one_minute_monies(ts_et, "SPX", expiry)
        if df_api is None or df_api.empty:
            print("No API data for this minute")
            continue

        print("API vols (vol90, vol50, vol10):")
        print(df_api[["vol90", "vol50", "vol10"]].head(1).T)

        # 2) Upsert using the NEW pipeline (overwrites existing DB row)
        n = upsert_from_dashboard_minute(df_api, ticker="SPX", store_volxx=True)
        print("Upserted rows:", n)

        # 3) Read back via the same function the Smile callback uses
        df_db = read_minute_expiry_df_from_db("SPX", trade_date, expiry, hhmm)

        print("DB vols (vol90, vol50, vol10):")
        if df_db is None or df_db.empty:
            print("  <no DB data>")
        else:
            print(df_db[["vol90", "vol50", "vol10"]].T)


if __name__ == "__main__":
    main()
