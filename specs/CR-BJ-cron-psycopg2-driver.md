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

## Step-0 findings (2026-09-21, read-only)

1. **Driver installed by the cron build.** `[Certain]` `apps/cron/requirements.txt` installs psycopg **3** only (`psycopg[binary]==3.2.3`, `psycopg_pool==3.2.3`). psycopg2 is present only as a commented-out line (`## psycopg2`, commented since the early Ironbeam-ingest commits 5f5d0e8 / 627b372 — long before CR-AU). `apps/web/requirements.txt` carries both `psycopg[binary,pool]` and `psycopg2-binary`, which is why the same code works under the web build and locally. `get_engine()` does `create_engine(url, pool_pre_ping=True)` with no scheme rewrite, so a plain `postgresql://` URL resolves to the `postgresql+psycopg2` dialect and imports `psycopg2` at engine construction.
2. **Why "since ~2026-09-08".** `[Likely]` Nothing regressed: `daily-leg-capture` was first defined by CR-AU on 2026-09-07 (727cc49 / b548eaf) with `buildCommand: pip install -r apps/cron/requirements.txt`. It has never had psycopg2 under the cron build; the 2026-09-08 14:03 UTC write to `orats_options_minute` is consistent with a run outside the cron build (CR-BD repair of 09-08), not a successful cron run.
3. **Other cron-run importers of the repository.** `[Certain]` Only two cron entry points import `packages.shared.options_cache.repository`, both lazily, both after overriding `DATABASE_URL` in-process from `BACKFILL_DATABASE_URL`:
   - `scripts/cron_daily_leg_capture.py:337` (`daily-leg-capture`)
   - `scripts/cr_aa_sweep_pending_outcomes.py:392` (CR-AU matured-capture stage of `sweep_pending_outcomes`)

   `scripts/run_reference_rerun.py` and everything under `apps/cron/*.py` do not import options_cache. The other importers are one-off `scripts/cr_*` backfills (run from the laptop under `apps/web/.venv`, which has psycopg2 2.9.12) and web modules (web build has psycopg2-binary). `packages/shared/utils/data_io.py` also calls `create_engine`, but is not reached from a cron entry point.
4. **Reason not to add psycopg2-binary?** None found. psycopg2-binary and psycopg 3 coexist already in the web build; psycopg2-binary ships cp313 wheels from 2.9.10 (cron `PYTHON_VERSION` is 3.13.4), so the pin is `>=2.9.10`. Rewriting the scheme to `postgresql+psycopg://` would change the driver under code that has only ever run on psycopg2 (parameter binding, `executemany` behaviour in the upsert paths) — higher risk for a hotfix. **Decision 1 stands: add `psycopg2-binary`.**
