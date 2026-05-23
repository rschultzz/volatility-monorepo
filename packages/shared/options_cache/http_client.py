"""
ORATS HTTP client.

Low-level wrapper around requests for the ORATS API. Handles:
    - Authentication (token from env)
    - Retries with exponential backoff on transient errors
    - Error classification (transient vs permanent)
    - Connection pooling via a shared Session

This module deliberately knows nothing about OptionMinuteBar, the cache,
or specific endpoints. It just makes authenticated GET requests and returns
response text. Higher layers handle parsing.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Base URL for ORATS API. Same as in apps/cron/orats_monies_today_ingest.py.
BASE_URL = "https://api.orats.io"

# Default timeouts. (connect, read) — connect can be tight, read needs to
# accommodate large CSVs.
DEFAULT_TIMEOUT: tuple[float, float] = (10.0, 60.0)

# Retry policy: 5 attempts, exponential backoff with jitter.
# Schedule: ~1s, ~2s, ~4s, ~8s, ~16s between attempts.
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_S = 1.0
DEFAULT_MAX_BACKOFF_S = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0


class OratsError(Exception):
    """Base class for all ORATS client errors."""


class OratsTransientError(OratsError):
    """Network errors, 5xx, 429 — should be retried."""


class OratsPermanentError(OratsError):
    """Auth errors, 4xx (non-429), malformed responses — do not retry."""


# ────────────────────────────────────────────────────────────────────────
#  Session and token management
# ────────────────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Lazily-initialized shared session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def reset_session() -> None:
    """For tests: dispose and reset the cached session."""
    global _session
    if _session is not None:
        _session.close()
    _session = None


def _get_api_token() -> str:
    """Read ORATS_API_KEY from env. Raises if not set."""
    # Reuse the dotenv-loading path from repository — same pattern, same
    # behavior across env sources.
    from .repository import _load_env_file_if_present

    token = os.environ.get("ORATS_API_KEY")
    if not token:
        _load_env_file_if_present()
        token = os.environ.get("ORATS_API_KEY")
    if not token:
        raise OratsPermanentError(
            "ORATS_API_KEY is not set in the environment. "
            "Set it directly, or place a .env file at the project root."
        )
    return token


# ────────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────────

def get_csv(
    path: str,
    params: dict,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> str:
    """
    Make an authenticated GET request to ORATS, returning response body as text.

    The token is added to params automatically — callers should NOT include
    it in the params dict passed in.

    Args:
        path: Path under BASE_URL (e.g., '/datav2/hist/one-minute/strikes/option').
        params: Query parameters.
        max_retries: Number of retry attempts on transient errors. 0 = no retries.
        timeout: (connect, read) timeout in seconds.

    Returns:
        Response body as text (typically CSV).

    Raises:
        OratsTransientError if all retries are exhausted on transient errors.
        OratsPermanentError on auth failures, 4xx responses (non-429), or
        malformed responses.
    """
    token = _get_api_token()
    url = f"{BASE_URL}{path}"

    # Don't mutate caller's dict; add token to a copy
    final_params = dict(params)
    final_params["token"] = token

    session = _get_session()
    last_error: Optional[Exception] = None
    backoff = DEFAULT_INITIAL_BACKOFF_S

    for attempt in range(max_retries + 1):
        try:
            return _do_request(session, url, final_params, timeout)
        except OratsPermanentError:
            # Don't retry permanent errors
            raise
        except OratsTransientError as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff with jitter (full jitter strategy)
                sleep_s = min(backoff, DEFAULT_MAX_BACKOFF_S)
                sleep_s = random.uniform(0, sleep_s)
                logger.warning(
                    "ORATS request failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, e, sleep_s,
                )
                time.sleep(sleep_s)
                backoff *= DEFAULT_BACKOFF_FACTOR
            else:
                logger.error(
                    "ORATS request failed after %d attempts: %s",
                    max_retries + 1, e,
                )

    # Exhausted all retries
    assert last_error is not None
    raise last_error


def _do_request(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: tuple[float, float],
) -> str:
    """
    Execute one HTTP request and classify the response.

    Returns the response text on 2xx with non-empty body. Raises
    OratsTransientError or OratsPermanentError otherwise.
    """
    # Log the URL without the token, for debugging without leaking creds
    safe_params = {k: v for k, v in params.items() if k != "token"}
    logger.info("GET %s params=%s", url, safe_params)

    try:
        resp = session.get(url, params=params, timeout=timeout)
    except requests.Timeout as e:
        raise OratsTransientError(f"timeout: {e}") from e
    except requests.ConnectionError as e:
        raise OratsTransientError(f"connection error: {e}") from e
    except requests.RequestException as e:
        # Other unexpected requests-level errors — treat as transient
        raise OratsTransientError(f"request error: {e}") from e

    # Classify the response status
    if resp.status_code == 200:
        body = resp.text
        if body.lstrip().startswith("<"):
            # ORATS sometimes returns 200 with an HTML error page on bad
            # auth or invalid endpoints. Treat as permanent.
            raise OratsPermanentError(
                f"got 200 but response looks like HTML/XML (likely auth or "
                f"endpoint error). First 200 chars: {body[:200]!r}"
            )
        return body

    if resp.status_code == 429:
        raise OratsTransientError(
            f"rate limited (429). Body: {resp.text[:200]!r}"
        )

    if 500 <= resp.status_code < 600:
        raise OratsTransientError(
            f"server error {resp.status_code}. Body: {resp.text[:200]!r}"
        )

    if 400 <= resp.status_code < 500:
        raise OratsPermanentError(
            f"client error {resp.status_code}. Body: {resp.text[:200]!r}"
        )

    # Some other status code — treat as permanent
    raise OratsPermanentError(
        f"unexpected status {resp.status_code}. Body: {resp.text[:200]!r}"
    )
