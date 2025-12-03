#!/usr/bin/env python
"""
Minimal Ironbeam LIVE /auth test using *only* username + apiKey.

Required env vars:

  IRONBEAM_USERNAME      -> your live account id (e.g. 51395669)
  IRONBEAM_LIVE_API_KEY  -> your LIVE API key from the email
"""

import os
import requests

BASE_URL = "https://live.ironbeamapi.com/v2"
AUTH_URL = f"{BASE_URL}/auth"


def main() -> None:
    username = os.getenv("IRONBEAM_USERNAME")
    api_key = os.getenv("IRONBEAM_LIVE_API_KEY")

    if not username or not api_key:
        raise SystemExit(
            "Set IRONBEAM_USERNAME and IRONBEAM_LIVE_API_KEY first."
        )

    # Helpful: show a redacted preview so you know what’s being used
    redacted_key = api_key[:4] + "..." + api_key[-4:]

    payload = {
        "username": username,
        "apiKey": api_key,  # NOTE: no 'password' field at all
    }

    print(f"POST {AUTH_URL}")
    print("Payload (sanitized):", {
        "username": username,
        "apiKey": redacted_key,
    })

    resp = requests.post(
        AUTH_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    print("Status:", resp.status_code)
    print("Raw body:", resp.text)

    if not resp.ok:
        print("\nAuth failed; see status + raw body above.")
        return

    try:
        data = resp.json()
    except Exception:
        print("\nResponse is not valid JSON")
        return

    print("\nParsed JSON:", data)
    token = data.get("token")
    print("\nToken:", token)


if __name__ == "__main__":
    main()
