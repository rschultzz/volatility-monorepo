# pragma: no cover
# Finds which ORATS day/minute CSV endpoint returns rows for your account.
import os, io, datetime as dt
import pandas as pd, requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

ORATS_API_KEY = os.getenv("ORATS_API_KEY")
BASE = "https://api.orats.io"

def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.2,
                                                      status_forcelist=[429,500,502,503,504])))
    return s

def _try_url(ses, url, params):
    try:
        r = ses.get(url, params=params, headers={"User-Agent":"debug-orats/0.1"}, timeout=45)
        code = r.status_code
        txt = (r.text or "").strip()
        if code != 200 or not txt or txt.startswith("<"):
            return code, None, txt[:120]
        df = pd.read_csv(io.StringIO(txt))
        return code, df, txt.splitlines()[0][:140]
    except Exception as e:
        return -1, None, str(e)[:140]

def main():
    if not ORATS_API_KEY:
        print("Set ORATS_API_KEY"); return
    ticker = os.getenv("TICKER","SPX")
    td_env = os.getenv("TEST_TRADE_DATE")
    trade_date = dt.date.fromisoformat(td_env) if td_env else dt.date.today()

    params = {"ticker": ticker, "tradeDate": trade_date.strftime("%Y%m%d"), "token": ORATS_API_KEY}

    candidates = [
        f"{BASE}/datav2/hist/intraday/monies/implied.csv",
        f"{BASE}/datav2/hist/monies/implied.csv",
        f"{BASE}/datav2/live/intraday/monies/implied.csv",
        f"{BASE}/datav2/intraday/monies/implied.csv",
        # add any you suspect here:
        f"{BASE}/datav2/hist/intraday/monies/smile.csv",
        f"{BASE}/datav2/hist/intraday/monies/implieds.csv",
    ]

    ses = _session()
    print(f"Testing {len(candidates)} endpoints for {ticker} {params['tradeDate']}")
    for url in candidates:
        code, df, hint = _try_url(ses, url, params)
        status = "OK" if (df is not None and not df.empty) else "NO DATA"
        cols = list(df.columns)[:12] if df is not None else []
        print(f"- {url}\n   status={code} {status}; header={hint}\n   cols={cols}")
        if df is not None and not df.empty:
            # success → suggest setting env var for the test/ingest
            print(f"\n[FOUND] Working endpoint:\n  export ORATS_MONIES_DAY_URL='{url}'\n")
            break

if __name__ == "__main__":
    main()
