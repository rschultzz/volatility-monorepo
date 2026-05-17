"""
OPRA symbol formatting and parsing.

The OPRA Options Symbol format encodes a unique listed-option contract as
a string of up to 21 characters:

    [ROOT(1-6)] [YYMMDD(6)] [C|P(1)] [STRIKE(8)]

    ROOT     - underlying root symbol, left-justified, padded with spaces
               to 6 chars in the formal spec. In practice, ORATS and most
               vendors return the root without trailing spaces.
    YYMMDD   - expiration date
    C|P      - 'C' for call, 'P' for put
    STRIKE   - 8-digit strike price: 5 digits dollars + 3 digits decimals
               (i.e., strike * 1000, zero-padded)

Examples:
    SPXW   2026-01-17  Put   5800.0  ->  SPXW260117P05800000
    SPX    2026-03-20  Call  5750.0  ->  SPX260320C05750000
    AAPL   2026-06-19  Call   210.5  ->  AAPL260619C00210500

ORATS uses root `SPX` for all SPX expirations, including PM-settled
weeklies (colloquially called SPXW). The `expiryTod` column on chain
rows distinguishes AM vs PM settlement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

OptionType = Literal["C", "P"]

# Regex matching a well-formed OPRA symbol. We intentionally allow root
# lengths from 1 to 6 chars (alphabetic) to support equity options later.
_OPRA_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})"
    r"(?P<yymmdd>\d{6})"
    r"(?P<cp>[CP])"
    r"(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OpraSymbol:
    """A parsed OPRA contract symbol."""
    root: str
    expir: date
    option_type: OptionType
    strike: float

    @property
    def symbol(self) -> str:
        """The canonical OPRA string."""
        return format_opra(self.root, self.expir, self.option_type, self.strike)

    def __str__(self) -> str:
        return self.symbol


def format_opra(
    root: str,
    expir: date,
    option_type: OptionType,
    strike: float,
) -> str:
    """
    Format a contract specification into an OPRA symbol string.

    Args:
        root: Underlying root symbol (e.g., 'SPX', 'SPXW', 'AAPL'). Will be
              upper-cased. Whitespace stripped.
        expir: Expiration date.
        option_type: 'C' or 'P'.
        strike: Strike price as a float (e.g., 5800.0, 210.5).

    Returns:
        OPRA symbol string (e.g., 'SPXW260117P05800000').

    Raises:
        ValueError if root is empty or too long, option_type isn't C/P, or
        strike doesn't fit in 8 digits (i.e., >= 100000).
    """
    root = root.strip().upper()
    if not root or len(root) > 6:
        raise ValueError(f"OPRA root must be 1-6 chars, got: {root!r}")
    if not root.isalpha():
        raise ValueError(f"OPRA root must be alphabetic, got: {root!r}")
    if option_type not in ("C", "P"):
        raise ValueError(f"option_type must be 'C' or 'P', got: {option_type!r}")
    if strike < 0:
        raise ValueError(f"strike must be non-negative, got: {strike}")

    # Strike: 5 dollar digits + 3 decimal digits = 8 total
    # strike * 1000 gives us the integer encoding
    strike_int = round(strike * 1000)
    if strike_int >= 10**8:
        raise ValueError(
            f"strike too large for 8-digit OPRA encoding: {strike} "
            f"(max ~99999.999)"
        )
    strike_str = f"{strike_int:08d}"

    yymmdd = expir.strftime("%y%m%d")

    return f"{root}{yymmdd}{option_type}{strike_str}"


def parse_opra(symbol: str) -> OpraSymbol:
    """
    Parse an OPRA symbol string into its components.

    Args:
        symbol: OPRA symbol (e.g., 'SPXW260117P05800000'). Whitespace
                stripped. Case-insensitive on input but normalized to upper.

    Returns:
        OpraSymbol dataclass.

    Raises:
        ValueError if the symbol is malformed.
    """
    s = symbol.strip().upper()
    m = _OPRA_RE.match(s)
    if not m:
        raise ValueError(f"Malformed OPRA symbol: {symbol!r}")

    yy = int(m.group("yymmdd")[0:2])
    mm = int(m.group("yymmdd")[2:4])
    dd = int(m.group("yymmdd")[4:6])

    # OPRA YY is 2-digit. Pivot at 70 like most vendors:
    # YY < 70 -> 20YY, YY >= 70 -> 19YY. Practical for options data which
    # only goes back to ~1973 anyway, and good through 2069.
    full_year = 2000 + yy if yy < 70 else 1900 + yy

    try:
        expir = date(full_year, mm, dd)
    except ValueError as e:
        raise ValueError(f"Malformed OPRA expiration in {symbol!r}: {e}") from e

    strike = int(m.group("strike")) / 1000.0

    return OpraSymbol(
        root=m.group("root"),
        expir=expir,
        option_type=m.group("cp"),  # type: ignore[arg-type]
        strike=strike,
    )


def opra_to_orats_ticker(opra: str) -> str:
    """
    Convert a canonical OPRA symbol (with C|P side character) to the
    side-stripped form ORATS' /strikes/option endpoint expects as its
    `ticker` URL parameter.

    The option endpoint returns one row containing both call and put
    data for the requested strike+expir, so side isn't part of the
    query — sending the canonical 18-char OPRA returns 404. Internally
    we keep the side-bearing form as canonical; this helper bridges to
    the API at the request boundary.

    Both put and call inputs at the same strike+expir map to the same
    ORATS ticker, by design.

    Example:
        opra_to_orats_ticker("SPX240202P04935000") -> "SPX24020204935000"
        opra_to_orats_ticker("SPX240202C04935000") -> "SPX24020204935000"

    Raises ValueError if opra is malformed (delegated to parse_opra).
    """
    parsed = parse_opra(opra)
    yymmdd = parsed.expir.strftime("%y%m%d")
    strike_int = round(parsed.strike * 1000)
    return f"{parsed.root}{yymmdd}{strike_int:08d}"
