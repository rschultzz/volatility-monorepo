def auth() -> str:
    """
    POST /auth to obtain a bearer token.

    - Uses username/password and optional TENANT_API_KEY ("apikey" field).
    - Tries JSON first (matches your original ingest).
    - If the server returns 415 (Unsupported Media Type), it retries
      as form-encoded. Any non-415 status is treated as final.
    """
    if not IRONBEAM_USERNAME or not IRONBEAM_PASSWORD:
        raise RuntimeError("IRONBEAM_USERNAME and IRONBEAM_PASSWORD must be set")

    url = f"{API_BASE}/auth"
    payload: Dict[str, Any] = {
        "username": IRONBEAM_USERNAME,
        "password": IRONBEAM_PASSWORD,
    }
    if IRONBEAM_TENANT_API_KEY:
        payload["apikey"] = IRONBEAM_TENANT_API_KEY

    print(f"[AUTH] POST {url}")
    sanitized = {"username": IRONBEAM_USERNAME, "password": "***"}
    if IRONBEAM_TENANT_API_KEY:
        sanitized["apikey"] = "***"
    print(f"[AUTH] Payload (sanitized): {sanitized}")

    # Try JSON first, then fall back to form-encoded if we see a 415
    attempts = [
        ("json", dict(json=payload, headers={"Content-Type": "application/json"})),
        ("form", dict(data=payload)),  # application/x-www-form-urlencoded
    ]

    last_body = None
    last_status = None

    for mode, kwargs in attempts:
        print(f"[AUTH] Attempt mode={mode}")
        resp = requests.post(url, timeout=HTTP_TIMEOUT, **kwargs)
        last_status = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        last_body = body
        print(f"[AUTH] Status ({mode}): {resp.status_code}")
        print(f"[AUTH] Body ({mode}): {body!r}")

        # If the server doesn't like this media type, try the next mode
        if resp.status_code == 415:
            print(f"[AUTH] Got 415 with mode={mode}, trying next mode...")
            continue

        # For anything else, treat it as final (success or other error)
        resp.raise_for_status()
        if isinstance(body, dict) and "token" in body:
            token = body["token"]
            print("[AUTH] Authenticated OK.")
            return token

        raise RuntimeError(f"[AUTH] No token in response ({mode}): {body!r}")

    # If we got here, both attempts failed with 415 or similar
    raise RuntimeError(
        f"[AUTH] All attempts failed. Last status={last_status}, body={last_body!r}"
    )
