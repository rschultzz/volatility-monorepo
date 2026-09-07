"""0DTE iron-condor box construction, contract dedupe, entry credit and settlement
payoff (CR-AS, decisions 2–5 and 8).

Pure functions shared by scripts/cr_as_capture_0dte_condor_legs.py (which
fetches the legs) and scripts/cr_as_condor_analysis.py (which prices them), so
the two scripts cannot disagree about which contracts a box is made of.

Geometry (decision 2, amended in Step 0 to centre on the SPX open, not the ES
open): shorts at round5(open ± k·IM), long wings WING_WIDTH points further out.
Puts below, calls above. Two boxes per date: k = 0.5 and k = 1.0.

Payoff (decision 5): SPXW PM settlement is intrinsic, so the condor's value at
the close S is the intrinsic of the four legs. P&L = credit − loss.

Fees (decision 8): FEE_PER_CONTRACT_SIDE dollars per contract per side; a
four-leg condor opened and settled is 4 legs × 2 sides. In SPX points that is
/ 100 (the multiplier). Settlement of an expiring leg is not a commissioned
trade at Schwab, but the locked decision counts both sides; the gross figure
is reported alongside net so either convention can be read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional, Sequence

from packages.shared.backtest.quote_validity import leg_quote_is_valid

WING_WIDTH = 10.0
BOX_HALF_WIDTHS_IM: tuple[float, ...] = (0.5, 1.0)
FEE_PER_CONTRACT_SIDE = 1.30            # dollars, decision 8
CONDOR_LEGS = 4
CONDOR_ROUND_TRIP_FEE_USD = FEE_PER_CONTRACT_SIDE * CONDOR_LEGS * 2   # 10.40
CONDOR_ROUND_TRIP_FEE_PTS = CONDOR_ROUND_TRIP_FEE_USD / 100.0         # 0.104 SPX points


def round5(p: float) -> float:
    """Nearest multiple of 5 (same rule as cr_ah_step4_analysis.round5)."""
    return float(round(p / 5.0) * 5)


@dataclass(frozen=True)
class CondorBox:
    """One iron condor: four strikes on one expiry (the trade date itself)."""
    trade_date: date
    k_im: float                 # half-width in IM units (0.5 or 1.0)
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    open_px: float
    implied_move: float

    @property
    def label(self) -> str:
        return f"pm{self.k_im:g}"

    @property
    def wing_width_put(self) -> float:
        return self.short_put - self.long_put

    @property
    def wing_width_call(self) -> float:
        return self.long_call - self.short_call

    @property
    def min_wing_width(self) -> float:
        return min(self.wing_width_put, self.wing_width_call)

    @property
    def strikes(self) -> tuple[float, float, float, float]:
        return (self.long_put, self.short_put, self.short_call, self.long_call)

    def legs(self) -> list[tuple[float, str, str]]:
        """(strike, 'P'|'C', 'short'|'long') for the four legs."""
        return [
            (self.long_put, "P", "long"),
            (self.short_put, "P", "short"),
            (self.short_call, "C", "short"),
            (self.long_call, "C", "long"),
        ]

    def contracts(self) -> list[tuple[float, str]]:
        """(strike, option_type) for the four legs, in strike order."""
        return [(k, t) for k, t, _ in self.legs()]


def build_box(trade_date: date, open_px: float, implied_move: float, k_im: float,
              wing_width: float = WING_WIDTH) -> CondorBox:
    """Shorts at round5(open ± k·IM); wings `wing_width` further out (decision 2)."""
    if implied_move <= 0:
        raise ValueError(f"implied_move must be > 0, got {implied_move}")
    sp = round5(open_px - k_im * implied_move)
    sc = round5(open_px + k_im * implied_move)
    if sp >= sc:
        raise ValueError(f"degenerate box: short put {sp} >= short call {sc} (open {open_px}, IM {implied_move}, k {k_im})")
    return CondorBox(
        trade_date=trade_date, k_im=k_im,
        short_put=sp, long_put=sp - wing_width,
        short_call=sc, long_call=sc + wing_width,
        open_px=open_px, implied_move=implied_move,
    )


def build_boxes(trade_date: date, open_px: float, implied_move: float,
                half_widths: Sequence[float] = BOX_HALF_WIDTHS_IM) -> list[CondorBox]:
    """Both boxes for a date (decision 2)."""
    return [build_box(trade_date, open_px, implied_move, k) for k in half_widths]


def dedupe_contracts(boxes: Iterable[CondorBox]) -> list[tuple[float, str]]:
    """Distinct (strike, option_type) across boxes, sorted by (type, strike).
    Wings may coincide across boxes at small IM (decision 2) — fetch each
    contract once."""
    seen: set[tuple[float, str]] = set()
    for b in boxes:
        seen.update(b.contracts())
    return sorted(seen, key=lambda c: (c[1], c[0]))


def unlisted_legs(box: CondorBox, listed: Iterable[float]) -> list[float]:
    """Strikes of the box absent from the listed set (decision 3)."""
    ls = {float(s) for s in listed}
    return [k for k in box.strikes if float(k) not in ls]


# ── Pricing ──────────────────────────────────────────────────────────────────

def condor_credit(box: CondorBox, mids: dict[tuple[float, str], float]) -> Optional[float]:
    """Net credit = (short put mid − long put mid) + (short call mid − long call mid).
    None if any leg is missing from `mids`."""
    try:
        return ((mids[(box.short_put, "P")] - mids[(box.long_put, "P")])
                + (mids[(box.short_call, "C")] - mids[(box.long_call, "C")]))
    except KeyError:
        return None


def credit_is_valid(credit: Optional[float], box: CondorBox) -> bool:
    """Decision 4: credit ∈ (0, min wing width]."""
    return credit is not None and credit > 0.0 and credit <= box.min_wing_width + 1e-9


def minute_mids(quotes: Iterable[tuple[float, str, object, object]]) -> tuple[dict[tuple[float, str], float], int]:
    """(strike, type, bid, ask) rows for one minute → mids for the legs whose
    quote passes the CR-AN leg rule; returns the count rejected."""
    mids: dict[tuple[float, str], float] = {}
    n_bad = 0
    for strike, typ, bid, ask in quotes:
        if leg_quote_is_valid(bid, ask):
            mids[(float(strike), typ)] = (float(bid) + float(ask)) / 2.0
        else:
            n_bad += 1
    return mids, n_bad


def settlement_loss(box: CondorBox, close_px: float) -> tuple[float, float]:
    """Intrinsic loss of the condor at the close: (put-side loss, call-side loss),
    each in [0, wing width]."""
    put_loss = max(0.0, box.short_put - close_px) - max(0.0, box.long_put - close_px)
    call_loss = max(0.0, close_px - box.short_call) - max(0.0, close_px - box.long_call)
    return put_loss, call_loss


def breach_side(box: CondorBox, close_px: float) -> Optional[str]:
    if close_px < box.short_put:
        return "below"
    if close_px > box.short_call:
        return "above"
    return None


@dataclass
class CondorResult:
    box: CondorBox
    credit: float
    close_px: float
    put_loss: float
    call_loss: float
    gross_pnl_pts: float
    net_pnl_pts: float
    breach: Optional[str]
    max_loss_pts: float
    extra: dict = field(default_factory=dict)

    @property
    def gross_pnl_im(self) -> float:
        return self.gross_pnl_pts / self.box.implied_move

    @property
    def net_pnl_im(self) -> float:
        return self.net_pnl_pts / self.box.implied_move

    @property
    def credit_im(self) -> float:
        return self.credit / self.box.implied_move

    @property
    def loss_im(self) -> float:
        return (self.put_loss + self.call_loss) / self.box.implied_move


def settle_condor(box: CondorBox, credit: float, close_px: float,
                  fee_pts: float = CONDOR_ROUND_TRIP_FEE_PTS) -> CondorResult:
    """P&L at settlement: gross = credit − intrinsic loss; net = gross − fees."""
    put_loss, call_loss = settlement_loss(box, close_px)
    gross = credit - put_loss - call_loss
    return CondorResult(
        box=box, credit=credit, close_px=close_px,
        put_loss=put_loss, call_loss=call_loss,
        gross_pnl_pts=gross, net_pnl_pts=gross - fee_pts,
        breach=breach_side(box, close_px),
        max_loss_pts=max(box.wing_width_put, box.wing_width_call) - credit,
    )


# ── Sampling (decision 6) ────────────────────────────────────────────────────

REGIME_ORDER: tuple[str, ...] = ("magnetic-pin", "magnet-above", "amplification", "untethered")
FIRST_REGIME = "bounded"
TARGET_PER_REGIME = 40


def stride_select(items: list, n: int) -> list:
    """n evenly spaced items (all if len ≤ n); deterministic."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def sample_order(dates_by_regime: dict[str, list[date]], target: int = TARGET_PER_REGIME,
                 first: str = FIRST_REGIME, regimes: tuple[str, ...] = REGIME_ORDER) -> list[date]:
    """All `first`; then round-robin over stride-selected dates of the other
    regimes until `target` each; then round-robin over the remainder — so a
    budget cut leaves balanced groups."""
    order: list[date] = sorted(dates_by_regime.get(first, []))
    picked = {r: stride_select(sorted(dates_by_regime.get(r, [])), target) for r in regimes}
    rest = {r: [d for d in sorted(dates_by_regime.get(r, [])) if d not in set(picked[r])] for r in regimes}
    for group in (picked, rest):
        queues = {r: list(group[r]) for r in regimes}
        while any(queues.values()):
            for r in regimes:
                if queues[r]:
                    order.append(queues[r].pop(0))
    return order


def is_third_friday(d: date) -> bool:
    """Monthly SPX expiry (AM-settled) shares the date with the PM SPXW; ORATS
    keys both under root SPX, so the option endpoint is ambiguous there."""
    return d.weekday() == 4 and 15 <= d.day <= 21


# ── Entry selection (decision 4) ─────────────────────────────────────────────

def first_valid_entry(box: CondorBox, minutes: Sequence[tuple], floor=None) -> Optional[dict]:
    """First minute ≥ `floor` where all four legs pass the leg rule and the
    credit is in (0, min wing width]. `minutes` is [(snapshot_pt, quotes)] in
    time order, quotes = [(strike, type, bid, ask), ...] for that minute.
    Returns {"snapshot_pt", "credit", "mids", "n_minutes_seen", "n_minutes_invalid"} or None."""
    seen = invalid = 0
    for snap, quotes in minutes:
        if floor is not None and (snap.time() if hasattr(snap, "time") else snap) < floor:
            continue
        seen += 1
        mids, _ = minute_mids(quotes)
        credit = condor_credit(box, mids)
        if credit_is_valid(credit, box):
            return {"snapshot_pt": snap, "credit": credit, "mids": mids,
                    "n_minutes_seen": seen, "n_minutes_invalid": invalid}
        invalid += 1
    return None
