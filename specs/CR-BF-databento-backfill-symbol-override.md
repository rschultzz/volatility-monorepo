# CR-BF — Databento backfill: symbol override + fetch-before-delete

> Branch: `feat/CR-BF-databento-backfill-symbol-override` (off `origin/main` aa1e005)
> Scope: `scripts/backfill_databento_es.py` only. Attended script (owner role, legitimately needs DELETE) — **not** converted to the unattended backfill protocol. No Step-0 diagnosis gate. **The backfill itself is not run in this CR.**

## Problem

The ES futures roll U26 → Z26 (due Mon 2026-09-14) was missed. `ironbeam_es_1m_bars` and `es_minutes` hold dying-contract ES U26 bars from 2026-09-13 22:00 UTC through 2026-09-18 13:29 UTC (`IRONBEAM_SYMBOL` is already fixed on Render; live ingest is on Z26). The backfill script hardcodes `ES.c.0` / `continuous`, a calendar-roll continuous contract that resolves to U26 for that week, so it cannot repair the window. It also deletes before fetching, so a failed fetch leaves a hole.

## Changes

1. CLI args `--symbol` (default `ES.c.0`) and `--stype-in` (default `continuous`), threaded through `get_cost` and `get_range`. Default behavior unchanged.
2. Fetch all Databento chunks into memory first; abort with no DB changes if the fetch raises or the result is empty. Only then: disable trigger → delete → insert → populate `es_minutes` → re-enable trigger.
3. Print resolved symbol, stype, first/last bar datetime (UTC) and total volume of the fetched frame before the confirmation prompt, and in `--dry-run` output.
4. Docstring usage block updated.

## Out of scope

Ingest scripts, `discover_es_front_month()` (separate CR), the unattended backfill protocol.

## Verify

`python scripts/backfill_databento_es.py --start-date 2026-09-13 --end-date 2026-09-19 --symbol ESZ6 --stype-in raw_symbol --dry-run` — expect cost, first bar ~2026-09-13 22:00 UTC, last bar 2026-09-18 20:59 UTC, ~1M+ contracts/day.
