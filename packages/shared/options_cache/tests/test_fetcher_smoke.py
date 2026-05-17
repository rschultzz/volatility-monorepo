"""
Smoke test against the real ORATS API. Hits one OPRA over a one-day range
to verify _fetch_option_bars_from_orats's URL construction and parser
integration end-to-end. Disabled by default — see gating below.

Activation:
    RUN_ORATS_SMOKE=1 ORATS_API_KEY=... python -m unittest \
      packages.shared.options_cache.tests.test_fetcher_smoke

pytest.mark.smoke is set as a module-level pytestmark guarded by import
success — forward-compat for any future pytest infra without imposing a
runtime dep today.
"""
from __future__ import annotations

import os
import unittest
from datetime import datetime

try:
    import pytest
    pytestmark = pytest.mark.smoke
except ImportError:
    pass

from packages.shared.options_cache.fetcher import _fetch_option_bars_from_orats


@unittest.skipUnless(
    os.environ.get("RUN_ORATS_SMOKE") == "1",
    "smoke test disabled (set RUN_ORATS_SMOKE=1 to enable)",
)
@unittest.skipUnless(
    os.environ.get("ORATS_API_KEY"),
    "ORATS_API_KEY not set",
)
class TestFetchOptionBarsFromOratsSmoke(unittest.TestCase):
    def test_one_spx_opra_one_day_range(self):
        # SPX 4935 put, 0DTE 2024-02-02. Per the conventions doc's verified
        # reference points: bid=1.9, ask=2.0, delta=-0.177 at 09:44 PT.
        # Fetch the full regular session (06:30 PT == 09:30 ET to
        # 13:00 PT == 16:00 ET).
        bars = _fetch_option_bars_from_orats(
            "SPX240202P04935000",
            datetime(2024, 2, 2, 6, 30),
            datetime(2024, 2, 2, 13, 0),
        )

        self.assertGreater(
            len(bars), 0,
            "expected at least one bar from ORATS for a known-good OPRA",
        )

        bar = bars[0]
        self.assertIsNotNone(bar.bid_price)
        self.assertIsNotNone(bar.ask_price)
        self.assertIsNotNone(bar.delta)


if __name__ == "__main__":
    unittest.main()
