# CR-034 — Today's Setup Data Cache (Instant Tab-Return)

**Source:** vault session note `sessions/2026-06-02 - CR-034 — Today's Setup Data Cache and Q2 DB Perf Diagnostic.md`

## Scope

Frontend-only, **one file**: `react_today_setup/src/App.jsx`. No backend, no other components, no layout/style change. Adds a cache layer behind the existing load path; rendered output for a given date is byte-identical to today's, just sometimes served from `sessionStorage`.

Explicitly untouched: `DayView.jsx` and all other components, all backend, the CR-032 date-persistence logic (extended, not rewritten).

## Design

**Cache store.** A second `sessionStorage` key namespace, `today-setup-cache:v1:<ticker>:<date>`, value = JSON `{ savedAt: <epoch_ms>, proposals, analogues, anchorLandscape, flags }`. Per-date entries (not one blob) so each day is independently valid/evictable and writes stay small. Version prefix `v1` so a future shape change can invalidate cleanly.

**What "today" means.** Reuse the CR-032 `mostRecentTradingDay()` helper (now local-time). `isCurrentTradingDay(d) === (d === mostRecentTradingDay())`. The 90s TTL applies only when this is true.

**Cache validity on read:**
- Entry missing → miss (fetch).
- Entry present, `date` is **historical** (`!isCurrentTradingDay`) → **valid** (hydrate, no fetch).
- Entry present, `date` **is today** → valid only if `Date.now() - savedAt <= TODAY_TTL_MS` (90_000). Else miss (fetch).

**Write path.** After `loadAnchor()` completes a load where `proposals` resolved successfully (i.e. not the 404/no-data path), write the assembled four-tuple to the cache key with `savedAt = Date.now()`. Only cache a *complete, successful* anchor load — do not cache partial/errored states. Wrap in try/catch (storage can throw / quota).

**Read/hydrate path.** `loadAnchor()` currently clears state and sets `loading=true` immediately. Restructure the analogue-mode load effect:
1. On mount / date change in analogue mode, abort any in-flight request (from prior date change), then check the cache for `(ticker, date)`.
2. **Hit** → synchronously set `proposals`, `analogues`, `anchorLandscape`, `flags` from the cached tuple; set `loading=false`, `error=null`; reset `selectedDate`, `selectedLandscape`, `selectedAnalogue` to null; do **not** call `loadAnchor()`.
3. **Miss** → existing behavior: call `loadAnchor()` (clears + fetches, AbortController semantics preserved).

**Invalidation / clearing:**
- **On hard reload** (`isPageReload()` true): clear **all** `today-setup-cache:v1:*` entries (iterate `sessionStorage` keys by prefix) in addition to the existing date-key clear.
- **On manual date change via the picker:** no special clearing needed — different cache key; valid entries hydrate, invalid entries fetch.
- **On flag mutations:** invalidate (delete) the cache key for `(ticker, date)` (the anchor's cache key) after any successful flag write. Covers all five mutation handlers — see Step-0 amendment below.

**Constants:**
```
const TODAY_SETUP_CACHE_PREFIX = 'today-setup-cache:v1:';
const TODAY_TTL_MS = 90_000;  // current trading day only
```

## Edge Cases

- **sessionStorage disabled / quota exceeded:** all access in try/catch; on any failure, degrade to current behavior (always fetch). Never throw into render.
- **Corrupt/unparseable cache entry:** treat as miss; best-effort delete the bad key.
- **Browse mode:** out of scope — caching applies to analogue-mode anchor loads only.
- **Selected-day data** (`selectedLandscape`, `selectedAnalogue`): not cached — re-derived on selection from the (cached or fetched) analogue list. Only the four-tuple is cached.
- **Stale-after-TTL for today:** when today's entry expires, the miss path re-fetches and overwrites with a fresh `savedAt` — no special expiry sweep needed.

## Acceptance Criteria

1. Load a **historical** day in analogue mode; click Price Chart pill; return to Today's Setup → instant render, no `Loading…` flash, no network calls to `/api/setup/proposals` etc.
2. Same round-trip for **today** within 90s → instant, no refetch.
3. Same round-trip for **today** after >90s → refetches (Loading appears), data refreshes, cache `savedAt` updates.
4. Hard refresh (Cmd-R) on any day → CR-032 reset-to-current-trading-day still holds **and** data cache is cleared.
5. `?date=` URL param still wins (CR-032 unregressed); a `?date=` historical day hydrates from cache on return.
6. Pick date A, pick date B, pick date A again → A hydrates instantly from cache.
7. Add or remove a flag on a day, leave, and return → flags shown are correct (cache invalidated; no stale flag state).
8. sessionStorage disabled → page still works, just always fetches; no console errors thrown into render.
9. `npm run build` clean; `git diff --stat` shows only `App.jsx`.

## Implementation Order

1. **Spec freeze** (commit 1): this file.
2. **Step-0 contradiction findings** (commit 2): append findings; amend before implementing.
3. **Cache core** (commit 3): constants, `cacheKey`, `readCache`, `writeCache`, `invalidateCache`, `clearAllCache`, `isCurrentTradingDay`; wire hydrate-or-load guard into the analogue-mode effect; write-on-success in `loadAnchor`; extend `isPageReload()` clear path to also `clearAllCache()`.
4. **Flag-mutation invalidation** (commit 4): delete the affected date's cache key in the **five** flag handlers after a successful mutation (see Step-0 amendment).
5. **Smoke + wrap** (commit 5): `npm run build`; manual browser smoke of ACs #1–#8; PR via merge commit.

---

## Step-0 Contradiction Check

**Finding — fifth flag handler (`handlePairFlag`):**

The spec (vault note) says "four handlers" and lists four API calls (`postFlag/deleteFlag/promote/demote`). The actual `App.jsx` has **five** mutation handlers:

| Handler | API call | Mutates cached state |
|---|---|---|
| `handleRegimeFlag` | `postFlag` | `flags` |
| `handleDeleteFlag` | `deleteFlag` | `flags` |
| `handlePromote` | `promoteFlag` | `flags` |
| `handleDemote` | `demoteFlag` | `flags` |
| **`handlePairFlag`** | `postFlag` | **`analogues`** (removes entry) |

`handlePairFlag` (App.jsx:331) calls `postFlag` to write a `not_a_true_analogue` flag and then calls `setAnalogues` to remove the analogue from the in-memory list. The `analogues` field is part of the cached four-tuple. If we don't invalidate on pair, a subsequent cache hydration would restore the removed analogue — failing AC #7.

**Resolution:** Add cache invalidation (`invalidateCache(ticker, date)`) to `handlePairFlag` as a fifth handler in commit 4. No other spec changes needed — the intent "invalidate on any flag write" already covers this; the "four" count was a miss.

**All other assumptions check out:**
- `loadAnchor` structure (useCallback, clears state, Promise.allSettled, AbortController) ✅
- Four state setters (`setProposals`, `setAnalogues`, `setAnchorLandscape`, `setFlags`) ✅
- `mostRecentTradingDay()` present (App.jsx:9) ✅
- `isPageReload()` present (App.jsx:20) ✅
- AbortController + `anchorAbortRef` semantics ✅

**Amendment to Implementation Order:** commit 4 invalidates **five** handlers, not four.

---

## Outcome

**Build:** `npm run build` clean (1.43s, no warnings).

**Diff:** `git diff --stat origin/main` = `react_today_setup/src/App.jsx` (+94/-9) + `specs/CR-034-today-setup-data-cache.md` (new). Scope held — no other files.

**Smoke (preview environment, no backend):**
- AC #4 ✅ Hard reload clears all `today-setup-cache:v1:*` keys and the date key — verified via `sessionStorage` inspection after `window.location.reload()`.
- AC #2/#3 logic ✅ TTL validation: historical entry (200s-old timestamp) → HIT; today's entry (200s-old) → MISS; today's entry (10s-old) → HIT.
- AC #8 ✅ App renders without console errors (root has content); all storage access in try/catch.

**ACs requiring live backend (manual smoke against production):** #1 (historical instant return), #2/#3 (today TTL round-trip), #5 (`?date=` + cache composes), #6 (date A→B→A hydrates A), #7 (flag invalidation → correct flags on return).

**PR:** #36 — `feat/CR-034-today-setup-data-cache` → `main` (merge commit).
