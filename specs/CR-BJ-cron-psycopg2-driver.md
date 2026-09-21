# CR-BJ — Cron build: Postgres driver for the options_cache engine (hotfix)

> Authority: kickoff prompt, 2026-09-21 (hotfix; no prior vault session note)
> Branch: `feat/CR-BJ-cron-psycopg2-driver` (off `origin/main` 2288dcb)
> Scope: `apps/cron/requirements.txt` + one smoke test. **No deploy, no env var change.**

## Problem

`daily-leg-capture` and the CR-AU stage of `sweep_pending_outcomes` have failed on every run since ~2026-09-08 with `ModuleNotFoundError: No module named 'psycopg2'`. `packages/shared/options_cache/repository.py` `get_engine()` calls `create_engine(url)` on a plain `postgresql://` URL, which selects SQLAlchemy's default psycopg2 dialect; the cron build installs `apps/cron/requirements.txt`, which does not include it. Last write to `orats_options_minute`: 2026-09-08 14:03 UTC.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Fix | Least-risk change that makes `get_engine()` work under the cron build. Prefer adding `psycopg2-binary` to `apps/cron/requirements.txt` over rewriting the URL scheme, unless Step 0 shows a reason not to. |
| 2 | Env | `DATABASE_URL` / `BACKFILL_DATABASE_URL` and every other env var unchanged. |
| 3 | Test | A smoke test that imports the repository and builds the engine using only the cron requirements. |
| 4 | Stop | Open the PR and stop. Merge + deploy of `daily-leg-capture` and `sweep_pending_outcomes` happen outside this CR, before the 13:50 UTC run. |

## Step 0 (read-only)

1. Which Postgres driver does `apps/cron/requirements.txt` install?
2. Does any other cron-run script import the options_cache repository?

## Follow-up (separate step, after the 13:50 UTC run, on explicit go)

List the post-2026-06-05 magnet-above entry days from 2026-09-09 through 2026-09-18 with no entry-day leg quotes; recapture them with the existing CR-AO path; report how far back ORATS actually serves minute quotes.
