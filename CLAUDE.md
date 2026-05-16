# Dash project — Claude Code memory

Options analytics and algorithmic trading platform (ES/SPX). Python monorepo, PostgreSQL on Render, Flask backend (`service.py`), React frontend. Two GEX-wall-based backtest strategies (short = mature, long = in development) plus an exploratory volatility-compression iron condor. Live signal panel (`SignalPanel.jsx`) persists to the `bt_signals` table.

## Vault-wide conventions

Vault-wide conventions for note types, naming, frontmatter schemas, and indexing live in `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Obsidian Setup/claude-vault-instructions.md` (referenced from notes as `[[claude-vault-instructions]]`). When writing to the vault, follow those conventions. The sections below describe Dash-specific overrides and code-side conventions that apply on top of them.

## Vault location

Project notes live in an Obsidian vault at `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/`. Treat it as the durable memory layer across sessions — it's where you read prior context from and where you write session output to.

The path contains a space; quote it in any shell commands you run (e.g. `ls "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/sessions/"`).

Vault subfolders:

- `_Dash MOC.md` — project home / Map of Content (read first for project overview)
- `sessions/` — chronological journal, one note per work session
- `strategies/` — strategy logic, parameters, current status
- `components/` — modules, tables, views, frontend pieces
- `decisions/` — ADR-style records of architectural choices
- `open-questions/` — unresolved bugs and investigations
- `_templates/session-template.md` — template for new session notes

## Start of session

Before doing any work, every session:

1. Skim `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/_Dash MOC.md` for project state — strategies in flight, key data tables, current focus.
2. List the 3 most recent files in `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/sessions/` (sorted by filename — they're `YYYY-MM-DD-<slug>` prefixed).
3. Read at least the most recent in full. Pay particular attention to `## Open questions` and `## Next session`.
4. If today's task overlaps with anything in `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/open-questions/`, read those notes too.
5. Briefly summarize to me what's outstanding and confirm what we're tackling today before writing any code.

## End of session

When I say "wrap up", "log this session", or run `/log-session`:

1. Create `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/sessions/YYYY-MM-DD-<slug>.md` from `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/_templates/session-template.md`. Slug = 2–4 hyphenated words capturing the session's primary topic.
2. Fill in the frontmatter completely. Use `[]` for empty lists rather than leaving fields blank.
3. Body sections: **Goal · What changed · Decisions · Open questions · Next session**.
4. If a strategy parameter changed, mirror the change into the relevant file under `/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/strategies/` so the strategy note always reflects current state.
5. If we resolved an item in `open-questions/`, set its frontmatter `status: complete` and link to today's session.
6. If we made an architectural decision, write a short note in `decisions/` (one decision per file, ADR-style: context · decision · consequences).
7. Print the absolute path to the new session note when done.

## Conventions

- Postgres tables and views are snake_case. Reference them as inline code (e.g. `es_minutes_with_features_bt`).
- Always annotate ES vs SPX — never assume.
- Times UTC unless explicitly tagged otherwise.
- Databento → Ironbeam cutover is the 2025-12-31 / 2026-01-01 UTC boundary. The `source` column on `ironbeam_es_1m_bars` distinguishes the two.
- "Walls" = high-|GEX| strikes used as price targets in the backtest strategies.
- Short vs long strategy parameters differ: `zoneMergeDistancePts` (short=10, long=5); `maxStartPctOfRange` semantics invert; long entries search from the signal bar itself.
- `target_level_gex_bn` carries signed GEX at target levels; color-coded in the results table.

## Frontmatter conventions

The `status` field uses **exactly one** of these values. Never invent new ones — Dataview queries depend on consistent strings.

- `created` — note exists but no work has started
- `in-progress` — actively being worked on
- `future-project` — parked for later; not in active rotation
- `complete` — done
- `cancelled` — explicitly stopped without completing

Status is most useful on notes that represent work in flight (sessions, open-questions, one-off task notes like migration plans or refactors). Reference material (strategy docs, component docs, glossary entries) typically doesn't need a status — the file's existence is its own state.

When updating a note's status, also update its modification context if relevant (link to the session that resolved it, note why it was cancelled, etc.) so the status change tells a story rather than just flipping a flag.

## What NOT to write to the vault

- Secrets, API keys, broker credentials, database connection strings.
- Raw market data (CSV/parquet). Reference paths only.
- Anything sensitive enough you wouldn't want indexed by Obsidian search.
