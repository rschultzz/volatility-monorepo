"""
Minimal command-line interface for the options cache.

Run from the project root:

    # Fetch the full chain for a ticker over a time range
    python -m packages.shared.options_cache.cli fetch-chain \\
        --ticker SPY \\
        --start "2024-02-02 09:14:00" \\
        --end   "2024-02-02 09:30:00"

    # Fetch a specific contract (uses chain endpoint internally)
    python -m packages.shared.options_cache.cli fetch \\
        --opra SPY240202P00493500 \\
        --start "2024-02-02 09:14:00" \\
        --end   "2024-02-02 09:30:00"

    # Inspect what's cached for a contract
    python -m packages.shared.options_cache.cli inspect \\
        --opra SPY240202P00493500

All times are interpreted as Pacific Time. Use 'YYYY-MM-DD HH:MM' format.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from . import repository as repo
from . import windows as W
from .fetcher import fetch_chain, fetch_contract
from .models import ChainFilter, TimeRange


def _parse_pt(s: str) -> datetime:
    """Parse a user-provided timestamp string as naive PT."""
    s = s.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"could not parse {s!r} as a datetime. "
        f"Use 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD'."
    )


def _build_filter(args: argparse.Namespace) -> ChainFilter | None:
    """Construct a ChainFilter from CLI args, or None if --no-filter set."""
    if args.no_filter:
        return None
    return ChainFilter(
        min_strike_pct_of_spot=args.min_strike_pct,
        max_strike_pct_of_spot=args.max_strike_pct,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
    )


def _cmd_fetch_chain(args: argparse.Namespace) -> int:
    chain_filter = _build_filter(args)
    print(
        f"Fetching {args.ticker} chain from {args.start} to {args.end} (PT)"
        f"{' [no filter]' if chain_filter is None else f' [filter: {chain_filter}]'}"
    )

    summary = fetch_chain(
        ticker=args.ticker,
        start_pt=args.start,
        end_pt=args.end,
        chain_filter=chain_filter,
    )

    print()
    print(f"✓ done")
    print(f"  API calls:        {summary.api_calls}")
    print(f"  Rows received:    {summary.rows_received:,}")
    print(f"  Rows kept:        {summary.rows_kept:,}")
    print(f"  Bars built:       {summary.bars_total:,}")
    print(f"  Bars inserted:    {summary.bars_inserted:,} (rest were dups)")
    print(f"  Unique contracts: {summary.contracts_touched:,}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    chain_filter = _build_filter(args)
    print(f"Fetching {args.opra} from {args.start} to {args.end} (PT)...")
    bars = fetch_contract(
        opra_symbol=args.opra,
        start_pt=args.start,
        end_pt=args.end,
        chain_filter=chain_filter,
    )
    print(f"\n✓ {len(bars)} bars in cache for {args.opra}")
    if bars:
        first, last = bars[0], bars[-1]
        print(
            f"  First: {first.snapshot_pt} bid={first.bid_price} "
            f"ask={first.ask_price} delta={first.delta:.3f}"
        )
        print(
            f"  Last:  {last.snapshot_pt} bid={last.bid_price} "
            f"ask={last.ask_price} delta={last.delta:.3f}"
        )
    return 0


def _cmd_fetch_for_scan(args: argparse.Namespace) -> int:
    """
    Read scan results from a JSON file (one row per object, or a top-level
    list/dict containing a 'rows' key), apply optional filters, and run
    the orchestrator to fetch pricing for matching rows.

    This is a manual-testing convenience for Phase 4a. The real UI flow
    (Phase 4c) will POST scan rows to a backend route.
    """
    import json
    from .orchestrator import fetch_for_rows

    with open(args.rows_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = data.get("rows", [])
    elif isinstance(data, list):
        rows = data
    else:
        raise SystemExit("rows_json must be a list or {rows: [...]} dict")

    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} row(s) from {args.rows_json}")

    result = fetch_for_rows(
        rows,
        strategy=args.strategy,
    )

    print(f"\n✓ done")
    print(f"  Rows attempted:        {result.rows_attempted}")
    print(f"  Rows with legs:        {result.rows_with_legs}")
    print(f"  Unique OPRAs fetched:  {result.unique_opras_fetched}")
    print(f"  Cache hits:            {result.cache_hits}")
    print(f"  Total API calls:       {result.total_api_calls}")
    print(f"  Total bars inserted:   {result.total_bars_inserted:,}")

    failed = [r for r in result.rows if r.error]
    if failed:
        print(f"\n⚠ {len(failed)} row(s) had errors:")
        for r in failed[:10]:
            print(f"    {r.row_key}: {r.error}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Show what's cached for a contract without making any API calls."""
    existing = repo.get_windows_for_contract(args.opra)

    print(f"Contract: {args.opra}")
    print(f"Fetched windows: {len(existing)}")
    for w in existing:
        print(
            f"  [{w.window_start_pt} -> {w.window_end_pt}]  "
            f"rows={w.row_count}  source={w.source}  "
            f"fetched_at={w.fetched_at}"
        )

    if args.start and args.end:
        request = TimeRange(start_pt=args.start, end_pt=args.end)
        gaps = W.find_gaps(request, existing)
        if gaps:
            print(f"\nFor [{args.start} -> {args.end}]: {len(gaps)} gap(s):")
            for g in gaps:
                print(f"    [{g.start_pt} -> {g.end_pt}]")
        else:
            print(f"\nFor [{args.start} -> {args.end}]: fully cached ✓")

        n = repo.count_bars_for_contract(args.opra, args.start, args.end)
        print(f"  Bars currently in cache for that range: {n}")

    return 0


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add --min-strike-pct, --max-strike-pct, --min-dte, --max-dte, --no-filter."""
    parser.add_argument(
        "--min-strike-pct", type=float, default=0.90,
        help="min strike as fraction of spot (default 0.90 = 10%% below spot)",
    )
    parser.add_argument(
        "--max-strike-pct", type=float, default=1.10,
        help="max strike as fraction of spot (default 1.10 = 10%% above spot)",
    )
    parser.add_argument(
        "--min-dte", type=int, default=0,
        help="min DTE (default 0)",
    )
    parser.add_argument(
        "--max-dte", type=int, default=60,
        help="max DTE (default 60)",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="ignore other filter args, keep all chain rows",
    )


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO-level logging")
    common.add_argument("-vv", "--debug", action="store_true",
                        help="enable DEBUG-level logging")

    parser = argparse.ArgumentParser(
        prog="python -m packages.shared.options_cache.cli",
        description="Options cache management CLI.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fetch-chain
    p_chain = sub.add_parser(
        "fetch-chain", parents=[common],
        help="fetch the full chain for a ticker over a time range",
    )
    p_chain.add_argument("--ticker", required=True,
                         help="underlying ticker (e.g., SPY)")
    p_chain.add_argument("--start", required=True, type=_parse_pt,
                         help="start time, naive PT")
    p_chain.add_argument("--end", required=True, type=_parse_pt,
                         help="end time, naive PT")
    _add_filter_args(p_chain)
    p_chain.set_defaults(func=_cmd_fetch_chain)

    # fetch (single contract via chain endpoint internally)
    p_fetch = sub.add_parser(
        "fetch", parents=[common],
        help="fetch pricing for a single contract (gap-aware)",
    )
    p_fetch.add_argument("--opra", required=True,
                         help="OPRA symbol (e.g. SPY240202P00493500)")
    p_fetch.add_argument("--start", required=True, type=_parse_pt,
                         help="start time, naive PT")
    p_fetch.add_argument("--end", required=True, type=_parse_pt,
                         help="end time, naive PT")
    _add_filter_args(p_fetch)
    p_fetch.set_defaults(func=_cmd_fetch)

    # fetch-for-scan
    p_scan = sub.add_parser(
        "fetch-for-scan", parents=[common],
        help="orchestrate pricing fetches for rows from a saved scan "
             "(reads scan rows from a JSON file)",
    )
    p_scan.add_argument(
        "--rows-json", required=True,
        help="path to JSON file containing scan rows (a list or "
             "an object with 'rows' key)",
    )
    p_scan.add_argument(
        "--strategy", default="condor",
        help="strategy name (default: condor)",
    )
    p_scan.add_argument(
        "--limit", type=int, default=None,
        help="optional: only process the first N rows",
    )
    p_scan.set_defaults(func=_cmd_fetch_for_scan)

    # inspect
    p_inspect = sub.add_parser(
        "inspect", parents=[common],
        help="show cached windows and gaps for a contract (no API calls)",
    )
    p_inspect.add_argument("--opra", required=True, help="OPRA symbol")
    p_inspect.add_argument("--start", type=_parse_pt, default=None,
                           help="optional: start of range to check coverage")
    p_inspect.add_argument("--end", type=_parse_pt, default=None,
                           help="optional: end of range to check coverage")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
