import os, argparse
from shared.ingest.orats_eod import run as run_orats_eod

def main(date: str | None):
    token = os.getenv("ORATS_API_KEY", "")
    print(f"[cron] date={date or 'auto'} token_len={len(token)}")
    run_orats_eod(date)
    print("[cron] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (optional)", default=None)
    args = parser.parse_args()
    main(args.date)
