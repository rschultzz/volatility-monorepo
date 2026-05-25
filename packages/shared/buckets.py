"""Outcome horizon bucket helpers (CR-B).

Distinct from strategy_templates._bucket_dte, which maps cluster dicts to
strategy entry DTE targets. This module maps dominant-bucket labels to
outcome measurement horizon lengths in RTH sessions.

The label format matches orats_gex_landscape.peaks_by_bucket keys and the
dominance_* fields in bt_daily_features.feature_vector.
"""

# Outcome horizon sessions by dominant bucket.
# These are distinct from strategy_templates.DTE_TARGET_BY_BUCKET:
#   - strategy targets: how many DTE to size the trade for (3, 15, 45)
#   - outcome horizons: how many sessions to wait before measuring result
# The two sets of constants serve different purposes and must not be conflated.
_OUTCOME_SESSIONS: dict[str, int] = {
    "0DTE":     1,   # intraday: check same session's close
    "1-7 DTE":  5,   # one trading week
    "8-30 DTE": 20,  # one trading month
    "30+ DTE":  60,  # one trading quarter
}


def bucket_sessions(label: str) -> int:
    """Return RTH session count for an outcome horizon given a dominant-bucket label.

    Accepts the label format used in orats_gex_landscape.peaks_by_bucket and
    derived from bt_daily_features.feature_vector dominance keys:
        '0DTE', '1-7 DTE', '8-30 DTE', '30+ DTE'

    Raises KeyError on unknown labels — callers must validate the label before
    passing here. Fail-fast is intentional: a silent default would produce
    wrong horizon lengths that are hard to detect in downstream smoke tests.
    """
    try:
        return _OUTCOME_SESSIONS[label]
    except KeyError:
        raise KeyError(
            f"Unknown dominant_bucket label {label!r}. "
            f"Expected one of: {sorted(_OUTCOME_SESSIONS)}"
        ) from None
