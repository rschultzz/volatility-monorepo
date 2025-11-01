
import os
import argparse
from datetime import datetime, timezone

def main(date: str | None):
    token = os.getenv("ORATS_API_KEY", "")
    if not token:
        print("[cron] WARNING: ORATS_API_KEY is not set.")
    print(f"[cron] Running EOD ingest for date={date or 'auto'} with token len={len(token)}")
    # TODO: import your real job here; this is just a stub.
    # from shared.utils.data_io import upsert_orats_batch
    # upsert_orats_batch(...)
    print("[cron] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Trade date YYYY-MM-DD (optional)", default=None)
    args = parser.parse_args()
    main(args.date)
