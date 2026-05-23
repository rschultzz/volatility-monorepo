"""
Standalone ORATS endpoint diagnostic.

Bypasses the cache module entirely. Just hits ORATS endpoints directly
with `requests` and shows what comes back. Use this to figure out which
URL paths your subscription actually allows.

Usage (from project root):
    python orats_diagnostic.py

Reads ORATS_API_KEY from environment (or .env via python-dotenv if installed).
"""
from __future__ import annotations

import os
import sys
from urllib.parse import urlencode

import requests

# Try to load .env if available — same fallback as the cache module
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


BASE = "https://api.orats.io"


def get_token() -> str:
    token = os.environ.get("ORATS_API_KEY")
    if not token:
        print("ERROR: ORATS_API_KEY is not set.")
        print("Set it in your .env file or export it in your shell.")
        sys.exit(1)
    return token


def try_endpoint(label: str, path: str, params: dict, token: str) -> None:
    """Hit one endpoint and print a concise result summary."""
    full_params = dict(params)
    full_params["token"] = token

    url = f"{BASE}{path}"
    # Reconstruct the URL the way requests will send it, but redact token
    safe_params = {k: v for k, v in full_params.items() if k != "token"}
    safe_url = f"{url}?{urlencode(safe_params)}&token=***"

    print()
    print("=" * 78)
    print(f"TEST: {label}")
    print(f"URL:  {safe_url}")
    print("-" * 78)

    try:
        r = requests.get(url, params=full_params, timeout=30)
    except requests.RequestException as e:
        print(f"REQUEST FAILED: {e}")
        return

    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type', '?')}")

    body = r.text
    if not body:
        print("Body: (empty)")
        return

    # Show first 3 lines of body. For CSV, that's header + 2 data rows.
    lines = body.splitlines()
    print(f"Body lines: {len(lines)}")
    print(f"First 3 lines:")
    for i, line in enumerate(lines[:3]):
        # Truncate very long lines so the output stays readable
        if len(line) > 200:
            line = line[:200] + "...(truncated)"
        print(f"  [{i}] {line}")

    if r.status_code == 200 and len(lines) >= 2:
        print(f"✓ SUCCESS — got {len(lines) - 1} data rows")
    elif r.status_code != 200:
        print(f"✗ FAILED — HTTP {r.status_code}")


def main() -> int:
    token = get_token()

    print(f"Using token starting with: {token[:8]}...")
    print(f"Hitting base URL: {BASE}")

    # ─── Test 1: Reproduce the user's known-working AAPL test ────────────
    # If this fails, something is fundamentally wrong with environment/auth.
    # If it works, our auth/network layer is fine.
    try_endpoint(
        "Known-working: AAPL strikes/option via /historical/",
        "/datav2/historical/one-minute/strikes/option",
        {
            "ticker": "AAPL23091500175000",
            "tradeDate": "202208081000",
        },
        token,
    )

    # ─── Test 2: Same AAPL contract via /hist/live/ ──────────────────────
    # Verifies whether /hist/live/ works for this user at all.
    try_endpoint(
        "AAPL strikes/option via /hist/live/",
        "/datav2/hist/live/one-minute/strikes/option",
        {
            "ticker": "AAPL23091500175000",
            "tradeDate": "202208081000",
        },
        token,
    )

    # ─── Test 3: Our SPXW contract via /historical/, single timestamp ───
    # Single timestamp matches the working AAPL test pattern.
    try_endpoint(
        "SPXW 0DTE put via /historical/, single timestamp",
        "/datav2/historical/one-minute/strikes/option",
        {
            "ticker": "SPXW24020204935000",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 4: Our SPXW contract via /hist/live/, single timestamp ────
    try_endpoint(
        "SPXW 0DTE put via /hist/live/, single timestamp",
        "/datav2/hist/live/one-minute/strikes/option",
        {
            "ticker": "SPXW24020204935000",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 5: SPXW with comma-separated range, /historical/ ──────────
    # Verifies whether ranges work for SPXW specifically.
    try_endpoint(
        "SPXW 0DTE put via /historical/, range tradeDate",
        "/datav2/historical/one-minute/strikes/option",
        {
            "ticker": "SPXW24020204935000",
            "tradeDate": "202402021214,202402021230",
        },
        token,
    )

    # ─── Test 6: SPXW chain endpoint ─────────────────────────────────────
    # The chain endpoint (different from option) — verifies SPXW data
    # exists at all on Feb 2 2024.
    try_endpoint(
        "SPXW chain via /historical/ on 2024-02-02",
        "/datav2/historical/one-minute/strikes/chain",
        {
            "ticker": "SPXW",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 7: Same SPXW chain, /hist/live/ ────────────────────────────
    try_endpoint(
        "SPXW chain via /hist/live/ on 2024-02-02",
        "/datav2/hist/live/one-minute/strikes/chain",
        {
            "ticker": "SPXW",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 8: SPY chain via /hist/live/ ───────────────────────────────
    # SPY is the ETF tracking SPX, with strikes ~1/10 the size. If SPX
    # isn't in ORATS' coverage, SPY almost certainly is — it's standard
    # OPRA-tape equity-options data like AAPL.
    try_endpoint(
        "SPY chain via /hist/live/ on 2024-02-02",
        "/datav2/hist/live/one-minute/strikes/chain",
        {
            "ticker": "SPY",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 9: SPY chain via /historical/ ──────────────────────────────
    try_endpoint(
        "SPY chain via /historical/ on 2024-02-02",
        "/datav2/historical/one-minute/strikes/chain",
        {
            "ticker": "SPY",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 10: SPY 0DTE put via /hist/live/ ───────────────────────────
    # Same setup as the SPX test, but for SPY: 4935 SPX ≈ 493.5 SPY.
    # If this works, we have a viable path forward.
    try_endpoint(
        "SPY 0DTE put via /hist/live/, single timestamp",
        "/datav2/hist/live/one-minute/strikes/option",
        {
            "ticker": "SPY24020200493500",  # SPY, 240202, 493.5 strike
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 11: SPY 0DTE put via /historical/ ──────────────────────────
    try_endpoint(
        "SPY 0DTE put via /historical/, single timestamp",
        "/datav2/historical/one-minute/strikes/option",
        {
            "ticker": "SPY24020200493500",
            "tradeDate": "202402021214",
        },
        token,
    )

    # ─── Test 12: Verify SPX exists in EOD strikes (different endpoint) ─
    # The original Data API (not Intraday) is on a different URL family
    # entirely. If SPX has EOD coverage but not intraday, that's a useful
    # data point for the support email.
    try_endpoint(
        "SPX EOD strikes (Data API, not Intraday) on 2024-02-02",
        "/data/hist/strikes",
        {
            "tickers": "SPX",
            "tradeDate": "2024-02-02",
        },
        token,
    )

    print()
    print("=" * 78)
    print("Diagnostic complete. Look for ✓ SUCCESS lines above.")
    print("Whichever combination(s) work tell us exactly what to use in the cache.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
