// CR-CD A5 — swing effort labels smoke (specs/CR-CD-order-flow-pane-v2.md, Amendment A5). Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_cd_smoke_backend.py   (port 8062, read-only DB)
//   review     apps/web/.venv/bin/python -u scripts/cr_cc_smoke_backend.py   (port 8061, review routes)
//   frontend   npm --prefix react_price_preview run dev -- --port 5177 --strictPort
//   baseline   the same frontend at origin/main before A5 (a detached worktree), --port 5178
// then:       CRBV_BASE_URL=http://localhost:5177 npx playwright test cr-cd-a5   (from react_price_preview/)
// CRCD_BASELINE_URL overrides the baseline address.
//
// "unchanged when off": Current (default) and Flow v2 with Effort off render pixel-identical to the
//   pre-A5 frontend, with no `effort` on any flow request; so does the CR-BV landscape view, and v2
//   after Effort was turned on and off.
// "effort on": the toggle sends effort=1, persists across a reload, and draws one label per leg in
//   the A5 format; the settings threshold changes which are highlighted, client-side.
// "replay": in a review replay the request carries until + effort=1 and every label ends before until.
// "screenshots": 2026-05-01 and 2026-04-01, with per-session leg counts in a5-effort-counts.json.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCD_API_BASE || 'http://127.0.0.1:8062'
const REVIEW_API = process.env.CRCD_REVIEW_API_BASE || 'http://127.0.0.1:8061'
const BASELINE = process.env.CRCD_BASELINE_URL || 'http://localhost:5178'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-cd')
const LABEL_RE = /^[+-]\d+\.\d · \d+(\.\d)?k? · (\d+(\.\d)?k?|–)\/pt( · ×\d+\.\d)?$/

function shifted(ptIso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}

function utcOf(ptIso) {
  // 2026-04/05 sessions are PDT (UTC-7)
  return shifted(ptIso) + 7 * 3600
}

async function open(page, day, { base = '', storage = {}, api = API_BASE } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript((init) => {
    try {
      if (!window.sessionStorage.getItem('crcd-a5-init')) {
        for (const k of ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort']) {
          window.localStorage.removeItem(k)
        }
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        for (const [k, v] of Object.entries(init)) window.localStorage.setItem(k, v)
        window.sessionStorage.setItem('crcd-a5-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, storage)
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(api)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
  await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The price chart's markers primitives (test-only, through React's fiber tree): every useRef whose
// current is { series, api } with a markers() API. Returns the markers with '/pt' in their text.
async function effortMarkers(page) {
  return page.evaluate(() => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    const out = []
    let f = chartEl.parentElement[key]
    for (let d = 0; f && d < 30; d += 1, f = f.return) {
      for (let h = f.memoizedState; h; h = h.next) {
        const v = h.memoizedState
        const c = v && typeof v === 'object' && 'current' in v ? v.current : null
        if (c && c.api && typeof c.api.markers === 'function') {
          for (const m of c.api.markers()) if (String(m.text).includes('/pt')) out.push(m)
        }
      }
    }
    return out
  })
}

async function frame(page, from, to) {
  await page.evaluate(({ a, b }) => {
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
    const data = series.data().filter((x) => x.close != null)
    const i0 = data.findIndex((x) => x.time >= a)
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to })
  await page.waitForTimeout(800)
}

function trackFlow(page) {
  const urls = []
  page.on('request', (r) => {
    if (r.url().includes('/api/ironbeam/flow')) urls.push(r.url())
  })
  return urls
}

test('unchanged when off: Current and v2 (Effort off) are pixel-identical to the pre-A5 frontend', async ({ browser }) => {
  for (const storage of [{}, { 'ib-react-flow-mode': 'v2' }]) {
    const shots = {}
    for (const [name, base] of [['a5', ''], ['baseline', BASELINE]]) {
      const ctx = await browser.newContext()
      const page = await ctx.newPage()
      const urls = trackFlow(page)
      await open(page, '2026-05-01', { base, storage })
      await settle(page)
      expect(urls.length).toBeGreaterThan(0)
      expect(urls.every((u) => !u.includes('effort'))).toBeTruthy()
      if (name === 'a5' && storage['ib-react-flow-mode'] === 'v2') {
        await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveAttribute('aria-pressed', 'false')
        expect(await effortMarkers(page)).toEqual([])
      }
      if (name === 'a5' && !storage['ib-react-flow-mode']) {
        await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveCount(0)   // Current header unchanged
      }
      // the price chart only: the v2 header gains the Effort chip by design (A5.4)
      shots[name] = await page.locator('.react-top-pane').screenshot()
      if (!storage['ib-react-flow-mode']) shots[`${name}-pane`] = await page.locator('.react-bottom-pane').screenshot()
      await ctx.close()
    }
    expect(Buffer.compare(shots.a5, shots.baseline), `price chart, ${JSON.stringify(storage)}`).toBe(0)
    if (shots['a5-pane']) expect(Buffer.compare(shots['a5-pane'], shots['baseline-pane']), 'Current flow pane').toBe(0)
  }

  // a CR-BV view (GEX landscape panel docked beside the chart, 2026-10-23): whole page identical
  const pages = {}
  for (const [name, base] of [['a5', ''], ['baseline', BASELINE]]) {
    const c = await browser.newContext()
    const pg = await c.newPage()
    await open(pg, '2026-10-23', { base })
    await settle(pg)
    await pg.getByRole('button', { name: 'Show GEX landscape panel' }).click()
    await pg.waitForFunction(() => [...document.querySelectorAll('svg')]
      .some((x) => x.querySelector('polyline[stroke="#f1f5f9"]')), null, { timeout: 60_000 })
    await settle(pg)
    pages[name] = await pg.screenshot()
    await c.close()
  }
  expect(Buffer.compare(pages.a5, pages.baseline), 'CR-BV landscape view').toBe(0)

  // Effort on, then off again: back to the baseline pixels
  const ctx = await browser.newContext()
  const page = await ctx.newPage()
  await open(page, '2026-05-01', { storage: { 'ib-react-flow-mode': 'v2' } })
  await settle(page)
  const before = await page.locator('.react-top-pane').screenshot()
  await page.getByTestId('flow-v2-effort-toggle').click()
  await settle(page)
  expect((await effortMarkers(page)).length).toBeGreaterThan(0)
  await page.getByTestId('flow-v2-effort-toggle').click()
  await settle(page)
  expect(await effortMarkers(page)).toEqual([])
  expect(Buffer.compare(await page.locator('.react-top-pane').screenshot(), before)).toBe(0)
  await ctx.close()
})

test('effort on: effort=1, persisted, one label per leg, threshold in settings', async ({ page }) => {
  const urls = trackFlow(page)
  await open(page, '2026-05-01', { storage: { 'ib-react-flow-mode': 'v2' } })
  await settle(page)
  const resp = page.waitForResponse((r) => r.url().includes('mode=v2') && r.url().includes('effort=1'))
  await page.getByTestId('flow-v2-effort-toggle').click()
  const payload = await (await resp).json()
  await settle(page)
  await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveAttribute('aria-pressed', 'true')
  const legs = payload.legs
  expect(legs.length).toBe(13)                                    // A5 step 2 figure, defaults
  let marks = await effortMarkers(page)
  expect(marks.length).toBe(legs.length)
  for (const m of marks) expect(m.text).toMatch(LABEL_RE)
  const ends = legs.map((l) => l.end_time - 7 * 3600).sort((a, b) => a - b)   // UTC -> shifted PDT
  expect(marks.map((m) => m.time)).toEqual(ends)
  const hot = (t) => legs.filter((l) => l.ratio != null && l.ratio >= t).length
  expect(marks.filter((m) => m.size === 1).length).toBe(hot(1.5))
  expect(marks.filter((m) => m.position === 'aboveBar').length).toBe(legs.filter((l) => l.dir === 'up').length)

  // threshold in settings: client-side, no new request
  const n = urls.length
  await page.getByRole('button', { name: 'Open Flow v2 marker settings' }).click()
  await page.getByLabel('Effort: highlight at × previous contracts/pt').fill('1.8')
  await page.getByRole('button', { name: 'Apply' }).click()
  await page.waitForTimeout(800)
  marks = await effortMarkers(page)
  expect(marks.filter((m) => m.size === 1).length).toBe(hot(1.8))
  expect(urls.length).toBe(n)

  // remembered across a reload
  await page.reload()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await settle(page)
  await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveAttribute('aria-pressed', 'true')
  expect((await effortMarkers(page)).length).toBe(legs.length)

  // the guide explains it
  await page.getByRole('button', { name: 'How to read Flow v2' }).click()
  await expect(page.getByRole('dialog', { name: 'How to read Flow v2' })).toContainText('contracts per point')
})

test('replay: until + effort=1 on the request, every label ends before until', async ({ page }) => {
  const asUnreviewed = (c) => {
    const { outcome, reviews, ...rest } = c
    return { ...rest, reviewed: false, revealed: false, n_reviews: 0, label: null, revealed_at: null, reviews: [] }
  }
  await page.route('**/api/setup-review/candidates?*', async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    body.candidates = body.candidates.map((c) => (c.id === 'CR-CC:cc:fixture:5' ? asUnreviewed(c) : c))
    await route.fulfill({ response: r, json: body })
  })
  await page.route('**/api/setup-review/candidates/CR-CC%3Acc%3Afixture%3A5', async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    body.candidate = asUnreviewed(body.candidate)
    await route.fulfill({ response: r, json: body })
  })
  await open(page, '2026-10-23', { api: REVIEW_API, storage: { 'ib-react-flow-mode': 'v2', 'ib-react-flow-v2-effort': '1' } })
  await settle(page)
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption('cc:fixture')
  const urls = trackFlow(page)
  await page.locator('[data-testid="review-row"][data-candidate-id="CR-CC:cc:fixture:5"]').click()
  await expect(page.getByTestId('review-banner')).toContainText('Review: 2026-05-01')
  // the first replay request can still carry the previous trade_date and is superseded once the
  // 2026-05-01 bars load; wait for the 2026-05-01 one and read it again after the page settles
  const want = (u) => u.includes('mode=v2') && u.includes('until=') && u.includes('effort=1') &&
    u.includes('trade_date=2026-05-01')
  await expect.poll(() => urls.some(want), { timeout: 90_000 }).toBeTruthy()
  await settle(page)
  const last = urls.filter(want).pop()
  const p = await (await page.request.get(last, { timeout: 120_000 })).json()
  const until = utcOf(p.until)
  expect(p.legs.length).toBeGreaterThan(0)
  for (const l of p.legs) expect(l.confirm_time).toBeLessThan(until)
  const marks = await effortMarkers(page)
  expect(marks.length).toBe(p.legs.length)
  for (const m of marks) expect(m.time).toBeLessThan(shifted(p.until))
  await page.getByRole('button', { name: 'Exit review' }).click()
  await expect(page.getByTestId('review-banner')).toHaveCount(0)
})

test('screenshots and leg counts', async ({ page, request }) => {
  const counts = {}
  for (const day of ['2026-05-01', '2026-04-01']) {
    const r = await request.get(`${API_BASE}/api/ironbeam/flow?trade_date=${day}&session=FULL&mode=v2&effort=1`, { timeout: 120_000 })
    expect(r.ok()).toBeTruthy()
    const p = await r.json()
    const ratios = p.legs.map((l) => l.ratio).filter((x) => x != null)
    counts[day] = {
      source: p.source,
      legs: p.legs.length,
      up: p.legs.filter((l) => l.dir === 'up').length,
      highlighted_1_5: ratios.filter((x) => x >= 1.5).length,
      ratio_min: Math.min(...ratios),
      ratio_max: Math.max(...ratios),
      cpp_median: p.legs.map((l) => l.cpp).sort((a, b) => a - b)[Math.floor(p.legs.length / 2)],
    }
    await page.context().clearCookies()
    await open(page, day, { storage: { 'ib-react-flow-mode': 'v2', 'ib-react-flow-v2-effort': '1' } })
    await settle(page)
    await frame(page, shifted(`${day}T06:30`), shifted(`${day}T12:59`))
    await page.screenshot({ path: path.join(ASSETS, `a5-effort-${day}-rth.png`) })
    await frame(page, shifted(`${day}T07:00`), shifted(`${day}T08:45`))
    await page.screenshot({ path: path.join(ASSETS, `a5-effort-${day}-0700-0845.png`) })
  }
  fs.writeFileSync(path.join(ASSETS, 'a5-effort-counts.json'), JSON.stringify(counts, null, 2) + '\n')
})
