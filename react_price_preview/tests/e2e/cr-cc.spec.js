// CR-CC step 6 — setup review mode smoke (specs/CR-CC-setup-review-mode.md).
//
// Servers (started outside Playwright):
//   backend   apps/web/.venv/bin/python -u scripts/cr_cc_smoke_backend.py   (port 8061; review routes write)
//   frontend  npm --prefix react_price_preview run dev -- --port 5176 --strictPort
// then:      CRBV_BASE_URL=http://localhost:5176 npx playwright test cr-cc   (from react_price_preview/)
//
// "replay" (read-only, repeatable): activating a cc:fixture candidate replays its session up to
//   entry_ts_pt — no candle at or after the cutoff, Corrected levels, SMILE/SIGNALS hidden, banner,
//   candidate price lines — and the list/detail responses carry no outcome before a review.
// "label → reveal" (writes setup_reviews, insert-only): on the first two UNREVIEWED cc:fixture
//   candidates, label (Take; Skip with reasons + a custom reason), Save, Reveal; the full session
//   comes back, the outcome is shown; then the Summary tab counts them. Each run consumes two of
//   the five fixture candidates.
// "revealed view" (read-only, repeatable): reopens revealed candidates and frames test → exit.
// Screenshots go to specs/assets/cr-cc/.
import { test, expect } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCC_API_BASE || 'http://127.0.0.1:8061'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-cc')
const SET = 'cc:fixture'

function shifted(ptIso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}

async function api(request, p) {
  const r = await request.get(`${API_BASE}/api/setup-review${p}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

async function openReview(page) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
      window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
    } catch (e) { /* ignore */ }
  })
  await page.goto(`/?trade_date=2026-10-23&interval=1min&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').waitFor()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption(SET)
}

// Candle series data, and framing (test-only, via React's fiber tree): `frame` is the number of
// last bars to show, or { from, to } shifted times to show (plus a margin of bars).
async function chartState(page, frame = 0) {
  return page.evaluate((n) => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    const find = (pred) => {
      let f = chartEl.parentElement[key]
      for (let d = 0; f && d < 30; d += 1, f = f.return) {
        for (let h = f.memoizedState; h; h = h.next) {
          const v = h.memoizedState
          const c = v && typeof v === 'object' && 'current' in v ? v.current : null
          if (c && pred(c)) return c
        }
      }
      return null
    }
    const chart = find((c) => typeof c.timeScale === 'function')
    const series = find((c) => typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick')
    const data = series.data().filter((b) => b.close != null)
    if (typeof n === 'number' && n > 0) {
      chart.timeScale().setVisibleLogicalRange({ from: data.length - n, to: data.length + 4 })
    } else if (n && typeof n === 'object') {
      const i0 = data.findIndex((b) => b.time >= n.from)
      let i1 = data.findIndex((b) => b.time >= n.to)
      if (i1 === -1) i1 = data.length - 1
      chart.timeScale().setVisibleLogicalRange({ from: i0 - n.margin, to: i1 + n.margin })
    }
    return {
      n: data.length,
      lastTime: data.at(-1).time,
      priceLines: series.priceLines().map((l) => l.options().title),
    }
  }, frame)
}

// Resolves when the center bars response for `sessionDate` arrives (with or without the cutoff).
function centerBars(page, sessionDate, { withUntil }) {
  return page.waitForResponse((r) => {
    const u = r.url()
    return u.includes('/api/ironbeam/bars') && u.includes(`trade_date=${sessionDate}`)
      && u.includes('phase=center') && u.includes('until=') === withUntil
  }, { timeout: 60_000 })
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
  await page.locator('.tv-lightweight-charts').first().waitFor()
  // the surrounding-sessions request (phase=multi) has finished too
  await page.waitForFunction(() => !document.querySelector('.top-pane-status'), null, { timeout: 60_000 })
  // flow and ATM IV requests start after the surrounding sessions load
  await page.waitForLoadState('networkidle', { timeout: 20_000 }).catch(() => {})
  await page.waitForTimeout(1200)
}

async function activate(page, id, sessionDate, { revealed = false } = {}) {
  const bars = centerBars(page, sessionDate, { withUntil: !revealed })
  await page.locator(`[data-testid="review-row"][data-candidate-id="${id}"]`).click()
  await bars
  await expect(page.getByTestId('review-banner')).toContainText(`Review: ${sessionDate}`)
  await expect(page.getByTestId('review-active')).toContainText(id)
  await settle(page)
}

test('replay: candidate session up to entry, no outcome, panels hidden', async ({ page, request }) => {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}`)
  expect(list.candidates.length).toBe(5)
  expect(JSON.stringify(list)).not.toContain('pnl_pts')
  const cand = list.candidates.find((c) => !c.reviewed) || list.candidates[0]
  const detail = (await api(request, `/candidates/${encodeURIComponent(cand.id)}`)).candidate
  if (!detail.reviewed) expect(detail.outcome).toBeUndefined()

  const outcomeSeen = []
  page.on('response', async (r) => {
    if (r.url().includes('/api/setup-review/candidates') && r.request().method() === 'GET') {
      try {
        const body = await r.text()
        if (body.includes('pnl_pts')) outcomeSeen.push(r.url())
      } catch { /* ignore */ }
    }
  })
  const barRequests = []
  page.on('request', (r) => {
    if (r.url().includes('/api/ironbeam/bars') && r.url().includes(`trade_date=${cand.session_date}`)) barRequests.push(r.url())
  })

  await openReview(page)
  await activate(page, cand.id, cand.session_date, { revealed: detail.revealed })

  if (!detail.revealed) {
    await expect(page.getByTestId('review-banner')).toContainText(`replay to ${cand.entry_ts_pt.slice(11, 16)} PT`)
    const st = await chartState(page, 100)
    expect(st.lastTime).toBeLessThan(shifted(cand.entry_ts_pt))
    expect(st.lastTime).toBe(shifted(cand.entry_ts_pt) - 60) // the signal bar is the last one shown
    expect(st.priceLines.some((t) => t.startsWith('level '))).toBeTruthy()
    for (const t of ['entry', 'stop']) expect(st.priceLines).toContain(t)
    expect(barRequests.length).toBeGreaterThan(0)
    for (const u of barRequests) {
      expect(u).toContain('levels=corrected')
      expect(u).toContain(`until=${encodeURIComponent(cand.entry_ts_pt)}`)
    }
    await expect(page.getByRole('button', { name: 'Show SMILE' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Show SIGNALS' })).toHaveCount(0)
    await expect(page.getByTestId('review-outcome')).toHaveCount(0)
    if (!detail.reviewed) expect(outcomeSeen).toEqual([])
    await page.waitForTimeout(500)
    await page.screenshot({ path: path.join(ASSETS, `replay-${cand.session_date}.png`) })
  }

  // Exit review restores the URL date and the SMILE pill.
  const urlBars = centerBars(page, '2026-10-23', { withUntil: false })
  await page.getByRole('button', { name: 'Exit review' }).click()
  await urlBars
  await expect(page.getByTestId('review-banner')).toHaveCount(0)
  await settle(page)
  await expect(page.getByRole('button', { name: 'Show SMILE' })).toHaveCount(1)
})

test('label → reveal on two candidates, then summary', async ({ page, request }) => {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}&status=unreviewed`)
  test.skip(list.candidates.length < 2, 'fewer than two unreviewed fixture candidates left')
  const [a, b] = list.candidates
  const before = (await api(request, `/summary?set=${encodeURIComponent(SET)}`)).groups[0]

  await openReview(page)

  // A: Take via the T key, comment, Save, Reveal via the R key.
  await activate(page, a.id, a.session_date)
  await expect(page.getByTestId('review-reveal')).toBeDisabled() // save is required first
  await page.evaluate(() => document.activeElement?.blur())
  await page.keyboard.press('t')
  await page.getByTestId('review-panel').getByLabel('Comment', { exact: true }).fill('CR-CC smoke: take')
  await page.getByTestId('review-save').click()
  await expect(page.getByTestId('review-active')).toContainText('Saved:')
  await expect(page.getByTestId('review-reveal')).toBeEnabled()
  await page.screenshot({ path: path.join(ASSETS, `saved-${a.session_date}.png`) })
  await page.evaluate(() => document.activeElement?.blur())
  const fullA = centerBars(page, a.session_date, { withUntil: false })
  await page.keyboard.press('r')
  await fullA
  await expect(page.getByTestId('review-outcome')).toContainText('pts')
  await expect(page.getByTestId('review-banner')).toContainText('revealed')
  await settle(page)
  let st = await chartState(page, 200)
  expect(st.lastTime).toBeGreaterThan(shifted(a.entry_ts_pt)) // full session after reveal
  expect(st.priceLines).toContain('exit')
  await expect(page.getByRole('button', { name: 'Show SMILE' })).toHaveCount(1)
  await page.waitForTimeout(500)

  // B: Skip with a chip and a custom reason, Save, Reveal button.
  await activate(page, b.id, b.session_date)
  await page.getByTestId('review-label-skip').click()
  await page.getByTestId('review-panel').getByRole('button', { name: 'trend day', exact: true }).click()
  await page.getByTestId('review-panel').getByLabel('Add reason', { exact: true }).fill('fixture smoke reason')
  await page.getByTestId('review-panel').getByLabel('Add reason', { exact: true }).press('Enter')
  await page.getByTestId('review-save').click()
  await expect(page.getByTestId('review-active')).toContainText('fixture smoke reason')
  const fullB = centerBars(page, b.session_date, { withUntil: false })
  await page.getByTestId('review-reveal').click()
  await fullB
  await expect(page.getByTestId('review-outcome')).toContainText('pts')
  await settle(page)
  st = await chartState(page, 200)
  expect(st.lastTime).toBeGreaterThan(shifted(b.entry_ts_pt))

  // DB state through the API: both revealed, outcomes now returned, labels counted.
  for (const c of [a, b]) {
    const d = (await api(request, `/candidates/${encodeURIComponent(c.id)}`)).candidate
    expect(d.revealed).toBeTruthy()
    expect(d.outcome.pnl_pts).toBeDefined()
  }
  const after = (await api(request, `/summary?set=${encodeURIComponent(SET)}`)).groups[0]
  expect(after.n_revealed).toBe(before.n_revealed + 2)
  expect((after.by_label.take?.n || 0)).toBe((before.by_label.take?.n || 0) + 1)
  expect((after.by_label.skip?.n || 0)).toBe((before.by_label.skip?.n || 0) + 1)
  expect(after.skip_reasons.map((r) => r.reason)).toContain('fixture smoke reason')

  await page.getByTestId('review-panel').getByRole('button', { name: 'Summary', exact: true }).click()
  await expect(page.getByTestId('review-summary')).toContainText(`${after.n_revealed} revealed`)
  await page.screenshot({ path: path.join(ASSETS, 'summary.png') })
})

test('revealed view: entry → exit on the full session (read-only)', async ({ page, request }) => {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}&status=reviewed`)
  const revealed = list.candidates.filter((c) => c.revealed).slice(0, 2)
  test.skip(revealed.length === 0, 'no revealed fixture candidates yet')
  await openReview(page)
  for (const c of revealed) {
    const d = (await api(request, `/candidates/${encodeURIComponent(c.id)}`)).candidate
    await activate(page, c.id, c.session_date, { revealed: true })
    await expect(page.getByTestId('review-banner')).toContainText('revealed')
    await expect(page.getByTestId('review-outcome')).toContainText(`${d.outcome.pnl_pts > 0 ? '+' : ''}${d.outcome.pnl_pts.toFixed(2)} pts`)
    const st = await chartState(page, { from: shifted(d.test_ts_pt), to: shifted(d.outcome.exit_ts_pt), margin: 45 })
    expect(st.lastTime).toBeGreaterThan(shifted(d.outcome.exit_ts_pt))
    expect(st.priceLines).toContain('exit')
    await page.mouse.move(5, 5)
    await page.waitForTimeout(600)
    await page.screenshot({ path: path.join(ASSETS, `revealed-${c.session_date}.png`) })
  }
})
