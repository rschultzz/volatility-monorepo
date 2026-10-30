// CR-DC — Flow v3 smoke (specs/CR-DC-flow-v3.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_dc_smoke_backend.py   (port 8063, read-only DB; review GETs)
//   frontend   npm --prefix react_price_preview run dev -- --port 5179 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5178
// then:       CRBV_BASE_URL=http://localhost:5179 npx playwright test cr-dc   (from react_price_preview/)
// CRDC_BASELINE_URL / CRDC_API_BASE / CRDC_LIVE_DATE override the defaults.
//
// "unchanged": Current and v2 render pixel-identical to origin/main — price chart, flow plot, and the
//   header on both sides of the mode toggle (DC1: the toggle's new "v3" button is the only difference) —
//   with the same flow requests and no cells request.
// "v3 defaults": the toggle opens v3 (new key, DC2); Heatmap / Profile / Absorbed are off, no cells
//   request, the price chart is unchanged; remembered across a reload; back to Current clears the key.
// "v3 on 2026-05-01": layers on → cells request, approximate badges, no marks (A1.2); anchor pick.
// "exact marks": 2026-10-27 (trades retained) draws marks with a hover.
// "live": the live session renders v3 with exact ETH cells.
// "replay": a review replay sends `until`; no cell, bar or mark after it.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
// CR-DF: Heatmap and Absorbed moved from the v3 header into ⚙ ("Heatmap (cells)", "Absorbed (marks)"); the
// tests below switch them there. The 'unchanged' test compares against a pre-v3 origin/main (DC1) and is
// superseded by cr-df.spec.js's identity test once origin/main has v3.
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRDC_API_BASE || 'http://127.0.0.1:8063'
const BASELINE = process.env.CRDC_BASELINE_URL || 'http://localhost:5178'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-dc')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers']

function liveDate() {
  if (process.env.CRDC_LIVE_DATE) return process.env.CRDC_LIVE_DATE
  const now = new Date()
  const pt = new Date(now.toLocaleString('en-US', { timeZone: 'America/Los_Angeles' }))
  if (pt.getHours() >= 15) pt.setDate(pt.getDate() + 1)
  while (pt.getDay() === 0 || pt.getDay() === 6) pt.setDate(pt.getDate() + 1)
  return `${pt.getFullYear()}-${String(pt.getMonth() + 1).padStart(2, '0')}-${String(pt.getDate()).padStart(2, '0')}`
}

function shifted(ptIso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}

async function open(page, day, { base = '', storage = {} } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdc-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        // CR-DH: origin/main defaults Levels to Current; compare like with like
        window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
        window.localStorage.setItem('ib-react-gex-levels-mode-v2', 'current')
        window.sessionStorage.setItem('crdc-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function gearLayers(page, want) {
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  const dlg = page.getByRole('dialog', { name: 'Flow v3 settings' })
  for (const [id, on] of Object.entries(want)) await dlg.getByTestId(id).setChecked(on)
  await dlg.getByRole('button', { name: 'Apply' }).click()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 90_000 })
  await page.waitForLoadState('networkidle', { timeout: 90_000 }).catch(() => {})
  await page.waitForTimeout(1500)
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

function track(page) {
  const urls = []
  page.on('request', (r) => {
    if (r.url().includes('/api/ironbeam/flow')) urls.push(r.url())
  })
  return urls
}

const pathOf = (u) => {
  const x = new URL(u)
  return `${x.pathname}?${[...x.searchParams.entries()].sort().map(([k, v]) => `${k}=${v}`).join('&')}`
}

// The flow pane's header split at the mode toggle: [left of it] and [right of it, same width on both builds].
async function headerParts(page) {
  const header = page.locator('.react-bottom-pane .flow-panel-header')
  const hb = await header.boundingBox()
  const tb = await page.locator('.react-bottom-pane [role=group][aria-label="Flow pane mode"]').boundingBox()
  const left = await page.screenshot({ clip: { x: hb.x, y: hb.y, width: Math.floor(tb.x) - 1 - hb.x, height: hb.height } })
  // start past the toggle's (fractional) right border so its anti-aliased edge is not compared
  const right = { x: Math.ceil(tb.x + tb.width) + 2, y: hb.y, height: hb.height, end: hb.x + hb.width }
  return { left, right, toggle: tb }
}

// CR-DP: retired — tests the Flow Current / v2 pane or the mode toggle, removed by CR-DP item 1 (Flow is v3 only)
test.skip('unchanged: Current and v2 are pixel-identical to origin/main (toggle aside), same requests', async ({ browser }) => {
  for (const storage of [{}, { 'ib-react-flow-mode': 'v2' }]) {
    const out = {}
    for (const [name, base] of [['branch', ''], ['main', BASELINE]]) {
      const ctx = await browser.newContext()
      const page = await ctx.newPage()
      const urls = track(page)
      await open(page, '2026-05-01', { base, storage })
      await settle(page)
      const parts = await headerParts(page)
      out[name] = {
        urls: [...new Set(urls.map(pathOf))].sort(),
        top: await page.locator('.react-top-pane').screenshot(),
        plot: await page.locator('.react-bottom-pane .flow-stage').screenshot(),
        left: parts.left,
        rightBox: parts.right,
        toggleW: parts.toggle.width,
        page,
        ctx,
      }
    }
    const tag = JSON.stringify(storage)
    const b = out.branch
    const m = out.main
    expect(b.urls, `flow requests ${tag}`).toEqual(m.urls)
    expect(b.urls.some((u) => u.includes('/flow/cells')), `no cells request ${tag}`).toBeFalsy()
    expect(Buffer.compare(b.top, m.top), `price chart ${tag}`).toBe(0)
    expect(Buffer.compare(b.plot, m.plot), `flow plot ${tag}`).toBe(0)
    expect(Buffer.compare(b.left, m.left), `header left of the toggle ${tag}`).toBe(0)
    // the same page region on both builds: from the (wider) branch toggle's right edge to the header end
    const x0 = Math.max(b.rightBox.x, m.rightBox.x)
    const clip = { x: x0, y: b.rightBox.y, width: Math.min(b.rightBox.end, m.rightBox.end) - x0, height: b.rightBox.height }
    const rb = await b.page.screenshot({ clip })
    const rm = await m.page.screenshot({ clip })
    if (!storage['ib-react-flow-mode']) {
      expect(Buffer.compare(rb, rm), `header right of the toggle ${tag}`).toBe(0)
    } else {
      // v2: the chips right of the toggle move right by the new button's width (DC1). Each one is the
      // same element (DOM) and draws the same pixels (element screenshot).
      const after = (pg) => pg.locator('.react-bottom-pane .flow-panel-header > [role=group] ~ *')
      const nb = await after(b.page).count()
      expect(nb).toBe(await after(m.page).count())
      for (let i = 0; i < nb; i += 1) {
        const eb = after(b.page).nth(i)
        const em = after(m.page).nth(i)
        expect(await eb.evaluate((e) => e.outerHTML)).toBe(await em.evaluate((e) => e.outerHTML))
        const sb = await eb.screenshot()
        const sm = await em.screenshot()
        expect(Buffer.compare(sb, sm), `v2 header element ${i} right of the toggle`).toBe(0)
      }
    }
    // DC1 added the "v3" button; since CR-DC merged, origin/main has it too, so the toggles are the same width
    // (CR-DG: was toBeGreaterThan, a known failure on every branch off main after CR-DC).
    expect(b.toggleW).toBe(m.toggleW)
    await expect(b.page.getByTestId('flow-mode-v3')).toBeVisible()
    await expect(m.page.getByTestId('flow-mode-v3')).toBeVisible()
    expect(await b.page.evaluate(() => window.localStorage.getItem('ib-react-flow-mode-v3'))).toBeNull()
    await b.ctx.close()
    await m.ctx.close()
  }
})

// CR-DP: retired — tests the Flow Current / v2 pane or the mode toggle, removed by CR-DP item 1 (Flow is v3 only) (its "back to Current" and toggle steps)
test.skip('v3 defaults: layers off, no cells request, price chart unchanged, remembered, back to Current', async ({ page }) => {
  const urls = track(page)
  await open(page, '2026-05-01')
  await settle(page)
  const before = await page.locator('.react-top-pane').screenshot()
  urls.length = 0
  await page.getByTestId('flow-mode-v3').click()
  await expect(page.getByTestId('flow-v3-panel')).toBeVisible()
  await settle(page)
  for (const id of ['flow-v3-levels-toggle', 'flow-v3-profile-toggle', 'flow-v3-edges-toggle']) {
    await expect(page.getByTestId(id)).toHaveAttribute('aria-pressed', 'false')
  }
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  for (const id of ['flow-v3-heatmap-toggle', 'flow-v3-absorbed-toggle']) await expect(page.getByTestId(id)).not.toBeChecked()
  await page.keyboard.press('Escape')
  expect(urls.some((u) => u.includes('mode=v2') && u.includes('effort=1'))).toBeTruthy()
  expect(urls.some((u) => u.includes('/flow/cells'))).toBeFalsy()
  expect(Buffer.compare(await page.locator('.react-top-pane').screenshot(), before), 'price chart with v3, layers off').toBe(0)
  expect(await page.evaluate(() => [localStorage.getItem('ib-react-flow-mode-v3'), localStorage.getItem('ib-react-flow-mode')]))
    .toEqual(['1', 'current'])
  await page.reload()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await settle(page)
  await expect(page.getByTestId('flow-v3-panel')).toBeVisible()
  await page.getByRole('button', { name: 'How to read Flow v3' }).click()
  const guide = page.getByRole('dialog', { name: 'How to read Flow v3' })
  await expect(guide).toContainText('Visual aid — untested.')
  expect(await guide.getByText('Visual aid — untested.').count()).toBeGreaterThanOrEqual(5)
  await page.keyboard.press('Escape')
  await page.getByTestId('flow-mode-current').first().click()
  await expect(page.getByTestId('flow-v3-panel')).toHaveCount(0)
  expect(await page.evaluate(() => localStorage.getItem('ib-react-flow-mode-v3'))).toBeNull()
})

test('v3 on 2026-05-01: approximate cells, no marks, anchor pick and readout', async ({ page }) => {
  const urls = track(page)
  await open(page, '2026-05-01', { storage: { 'ib-react-flow-mode-v3': '1' } })
  await settle(page)
  const off = await page.locator('.react-top-pane').screenshot()
  const cellsResp = page.waitForResponse((r) => r.url().includes('/flow/cells'))
  await gearLayers(page, { 'flow-v3-heatmap-toggle': true, 'flow-v3-absorbed-toggle': true })
  const cells = await (await cellsResp).json()
  await page.getByTestId('flow-v3-profile-toggle').click()
  await settle(page)
  const cu = urls.filter((u) => u.includes('/flow/cells')).pop()
  expect(cu).toContain('a_k=3')
  expect(cu).toContain('a_m=3')
  // RTH is exact once CR-DD's es_flow_price_1m covers the session (DC17), else approximate with no marks (A1.2)
  const badge = { price_1m: 'exact: es_flow_price_1m', trades: 'exact: trades', seconds: 'approximate: 1-s data' }
  const [eth, rth] = cells.segments
  expect([eth.name, eth.tier, eth.marks]).toEqual(['eth', 'seconds', null])
  expect(rth.name).toBe('rth')
  expect(rth.source).toBe('databento')
  // CR-DP item 2: the ETH / RTH exact / approximate pills are one "approx data" note when either segment is approximate
  void badge
  await expect(page.getByTestId('flow-v3-cells-badge')).toHaveCount(0)
  await expect(page.getByTestId('flow-v3-approx-note')).toHaveText('approx data — absorption & edges off')
  if (rth.tier === 'seconds') {
    expect(rth.marks).toBeNull()
    await expect(page.getByTestId('flow-v3-marks-note')).toHaveCount(0) // CR-DP item 2: the approx note says it
  } else {
    expect(Array.isArray(rth.marks)).toBeTruthy()
    await expect(page.getByTestId('flow-v3-marks-note')).toHaveCount(0)
  }
  fs.mkdirSync(ASSETS, { recursive: true })
  fs.writeFileSync(path.join(ASSETS, 'cells-2026-05-01.json'), JSON.stringify({ rth_tier: rth.tier, rth_cells: rth.cells.t.length,
    marks: rth.marks ? rth.marks.length : null }, null, 1))
  expect(Buffer.compare(await page.locator('.react-top-pane').screenshot(), off), 'layers draw').not.toBe(0)

  await frame(page, shifted('2026-05-01T07:20'), shifted('2026-05-01T08:20'))
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.screenshot({ path: path.join(ASSETS, 'v3-2026-05-01-0720-0820.png') })

  // anchor: arm, click a price bar, readout; the click does not toggle a timeslice
  const timesBefore = await page.evaluate(() => document.querySelectorAll('[data-testid^=timeslice]').length)
  await page.getByTestId('flow-v3-anchor-pick').click()
  const box = await page.locator('.react-top-pane .tv-lightweight-charts').first().boundingBox()
  await page.mouse.click(box.x + box.width * 0.35, box.y + box.height * 0.5)
  await page.waitForTimeout(600)
  await expect(page.getByTestId('flow-v3-readout')).toContainText('since ')
  await expect(page.getByTestId('flow-v3-anchor-line')).toHaveCount(1)
  await expect(page.getByTestId('flow-v3-anchor-pick')).toHaveAttribute('aria-pressed', 'false')
  expect(await page.evaluate(() => document.querySelectorAll('[data-testid^=timeslice]').length)).toBe(timesBefore)
  await page.screenshot({ path: path.join(ASSETS, 'v3-2026-05-01-anchored.png') })
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('flow-v3-readout')).toContainText('CVD from session open')
  await page.getByTestId('flow-v3-anchor-swing').click()
  await expect(page.getByTestId('flow-v3-readout')).toContainText('since ')
})

test('exact marks: 2026-10-27 (retained trades) draws absorption marks with a hover', async ({ page }) => {
  const resp = page.waitForResponse((r) => r.url().includes('/flow/cells'), { timeout: 90_000 })
  await open(page, '2026-10-27', { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ heat: true, profile: true, marks: true }) } })
  await settle(page)
  const cells = await (await resp).json()
  const rth = cells.segments.find((s) => s.name === 'rth')
  test.skip(rth.tier === 'seconds', '2026-10-27 is no longer in the trades table (2-day retention)')
  expect(rth.exact).toBeTruthy()
  expect(rth.marks.length).toBeGreaterThan(0)
  await expect(page.getByTestId('flow-v3-approx-note')).toHaveCount(0) // CR-DP item 2: exact sessions show no note (was the RTH exact pill)
  const mk = rth.marks[0]
  const t = mk.time - 7 * 3600 // PDT
  await frame(page, t - 20 * 60, t + 40 * 60)
  const pos = await page.evaluate(({ time, price }) => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    let f = chartEl.parentElement[key]
    let chart = null
    let series = null
    for (let d = 0; f && d < 30; d += 1, f = f.return) {
      for (let h = f.memoizedState; h; h = h.next) {
        const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
        if (c && typeof c.timeScale === 'function') chart = chart || c
        if (c && typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick') series = series || c
      }
    }
    const rect = chartEl.getBoundingClientRect()
    return { x: rect.x + chart.timeScale().timeToCoordinate(time), y: rect.y + series.priceToCoordinate(price) }
  }, { time: t - (t % 60), price: mk.price })
  const dy = mk.kind === 'sellers_absorbed' ? 9 : -9
  await page.mouse.move(pos.x, pos.y + dy)
  await expect(page.getByTestId('flow-v3-mark-tooltip')).toContainText('visual aid — untested', { timeout: 5000 })
  await expect(page.getByTestId('flow-v3-mark-tooltip')).toContainText(mk.kind === 'sellers_absorbed' ? 'Sellers absorbed' : 'Buyers absorbed')
  await page.screenshot({ path: path.join(ASSETS, 'v3-2026-10-27-marks.png') })
  fs.writeFileSync(path.join(ASSETS, 'marks-2026-10-27.json'), JSON.stringify({ k: 3, m: 3, n: rth.marks.length,
    sellers: rth.marks.filter((x) => x.kind === 'sellers_absorbed').length, buyers: rth.marks.filter((x) => x.kind === 'buyers_absorbed').length }, null, 1))
})

test('live: the live session renders v3 with exact cells', async ({ page }) => {
  const day = liveDate()
  const resp = page.waitForResponse((r) => r.url().includes('/flow/cells'), { timeout: 90_000 })
  await open(page, day, { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ heat: true, profile: true, marks: false }) } })
  await settle(page)
  const cells = await (await resp).json()
  await expect(page.getByTestId('flow-v3-panel')).toBeVisible()
  expect(cells.segments.length).toBeGreaterThan(0)
  for (const s of cells.segments) expect(['trades', 'price_1m']).toContain(s.tier)   // live = exact
  await expect(page.getByTestId('flow-v3-approx-note')).toHaveCount(0) // CR-DP item 2: exact sessions show no note (was the exact pill)
  await page.screenshot({ path: path.join(ASSETS, `v3-live-${day}.png`) })
})

test('replay: until on the cells request, nothing after it', async ({ page }) => {
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
  await open(page, '2026-10-23', { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ heat: true, profile: true, marks: true }) } })
  await settle(page)
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption('cc:fixture')
  const urls = track(page)
  await page.locator('[data-testid="review-row"][data-candidate-id="CR-CC:cc:fixture:5"]').click()
  await expect(page.getByTestId('review-banner')).toContainText('Review: 2026-05-01', { timeout: 30_000 }) // candidate detail is several DB round trips
  const want = (u) => u.includes('/flow/cells') && u.includes('until=') && u.includes('trade_date=2026-05-01')
  await expect.poll(() => urls.some(want), { timeout: 90_000 }).toBeTruthy()
  await settle(page)
  const cells = await (await page.request.get(urls.filter(want).pop(), { timeout: 120_000 })).json()
  const v2 = await (await page.request.get(urls.filter((u) => u.includes('mode=v2') && u.includes('until=') && u.includes('trade_date=2026-05-01')).pop(), { timeout: 120_000 })).json()
  const until = shifted(cells.until) + 7 * 3600
  for (const s of cells.segments) {
    expect(s.end).toBeLessThanOrEqual(until)
    for (const t of s.cells.t) expect(t).toBeLessThan(until)
    for (const mk of s.marks || []) expect(mk.confirm_time).toBeLessThanOrEqual(until)
  }
  for (const b of v2.bars) expect(b.time).toBeLessThan(until)
  await page.screenshot({ path: path.join(ASSETS, 'v3-replay-2026-05-01.png') })
  // (CR-DP item 1: the mode toggle this checked in the narrow header is removed)
  await page.getByRole('button', { name: 'Exit review' }).click()
  await expect(page.getByTestId('review-banner')).toHaveCount(0)
})
