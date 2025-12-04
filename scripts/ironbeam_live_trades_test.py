#!/usr/bin/env python
"""
Minimal Ironbeam LIVE /auth test.

Goal:
- Use live account ID + live API key
- Get a 200 + token back
- No websockets, no trades, no bars

Required env vars:

  IRONBEAM_USERNAME   = your LIVE account ID (numbers)
  IRONBEAM_API_KEY    = the LIVE API key they gave you
"""

import os
import requests

BASE_URL = "https://live.ironbeamapi.com/v2"
AUTH_URL = f"{BASE_URL}/auth"


def main() -> None:
    # username = os.getenv("IRONBEAM_USERNAME")
    # api_key = os.getenv("IRONBEAM_API_KEY")
    username = "23233577"
    api_key = "f20ae06eb7184cd8999321af363024ab"

    if not username or not api_key:
        raise SystemExit(
            "Set IRONBEAM_USERNAME (live account id) and IRONBEAM_API_KEY (live API key) first."
        )

    # Non-tenant mode: password field is the API key
    payload = {
        "username": username,
        "password": api_key,
    }

    headers = {"Content-Type": "application/json"}

    print(f"POST {AUTH_URL}")
    print("Payload (sanitized):", {"username": username, "password": "***"})

    resp = requests.post(AUTH_URL, json=payload, headers=headers, timeout=10)

    print("Status:", resp.status_code)
    print("Raw body:", resp.text)

    # Don't raise immediately; we want to see the body even on error
    if resp.ok:
        try:
            data = resp.json()
        except Exception:
            print("Response is not valid JSON")
            return

        token = data.get("token")
        print("\nParsed JSON:", data)
        print("\nToken:", token)
    else:
        print("\nAuth failed; see status + raw body above.")


if __name__ == "__main__":
    main()
