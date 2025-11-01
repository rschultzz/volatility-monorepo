import os, argparse
from shared.utils.data_io import get_engine
from sqlalchemy import text

def main(date: str | None):
    token = os.getenv("ORATS_API_KEY", "")
    print(f"[cron] date={date or 'auto'} token_len={len(token)}")

    # DB sanity check
    eng = get_engine()
    with eng.connect() as cx:
        res = cx.execute(text("select 1")).scalar_one()
    print(f"[cron] DB check ok (select 1 -> {res})")

    print("[cron] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (optional)", default=None)
    args = parser.parse_args()
    main(args.date)
