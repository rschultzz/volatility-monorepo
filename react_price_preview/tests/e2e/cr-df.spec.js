// CR-DF — absorption levels and the edge readout (specs/CR-DF-absorption-levels.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_df_smoke_backend.py   (port 8064, read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5181 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5180
// then:       CRBV_BASE_URL=http://localhost:5181 npx playwright test cr-df   (from react_price_preview/)
// CRDF_BASELINE_URL / CRDF_API_BASE / CRDF_LIVE_DATE override the defaults.
//
// "unchanged": Current and v2 render pixel-identical to origin/main (price chart, flow plot, whole header) with
//   the same flow requests and no cells request; v3 with every layer off leaves the price chart as Current's.
// "defaults": Absorption levels / Profile / Edges off in the header; Heatmap (cells) / Absorbed (marks) off in ⚙;
//   a stored CR-DC layers object keeps its heat / marks values there.
// "2026-05-01" and "live": levels and the readout on — the levels drawn are the payload's (flowV3.absorptionLevels),
//   the header text is flowV3.edgeText of the payload and anchor, the dashed edges draw, a level hovers.
// "replay": a review replay sends `until`; levels and edges read nothing after it.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { absorptionLevels, edgeReadout, edgeText, flattenBars, makeShifter } from '../../src/flowV3.js'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8064'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5180'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-df')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers']
const ALL_NEW = { levels: true, edges: true, profile: false, heat: false, marks: false }

function liveDate() {
  if (process.env.CRDF_LIVE_DATE) return process.env.CRDF_LIVE_DATE
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

// utc -> shifted PT wall clock, as the app (Intl, once per UTC hour)
const ptShift = makeShifter((t) => {
  const d = new Date(t * 1000)
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hourCycle: 'h23',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
    .formatToParts(d).map((x) => [x.type, x.value]))
  return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second) / 1000
})
const hhmm = (shiftedT) => new Date(shiftedT * 1000).toISOString().slice(11, 16)

async function open(page, day, { base = '', storage = {}, session = null } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdf-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crdf-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
  const s = session ? `&flow_session=${session}` : ''
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0${s}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 90_000 })
  await page.waitForLoadState('networkidle', { timeout: 90_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The price chart's lightweight-charts objects, from the React fiber (as cr-dc.spec.js).
const CHART_FN = `(() => {
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
  return { chartEl, chart, series }
})()`

async function frame(page, from, to) {
  await page.evaluate(({ a, b, fn }) => {
    const { chart, series } = eval(fn) // eslint-disable-line no-eval
    const data = series.data().filter((x) => x.close != null)
    const i0 = data.findIndex((x) => x.time >= a)
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to, fn: CHART_FN })
  await page.waitForTimeout(800)
}

async function point(page, time, price) {
  return page.evaluate(({ time: t, price: p, fn }) => {
    const { chartEl, chart, series } = eval(fn) // eslint-disable-line no-eval
    const rect = chartEl.getBoundingClientRect()
    return { x: rect.x + chart.timeScale().timeToCoordinate(t), y: rect.y + series.priceToCoordinate(p) }
  }, { time, price, fn: CHART_FN })
}

async function candleTimes(page) {
  return page.evaluate((fn) => {
    const { series } = eval(fn) // eslint-disable-line no-eval
    return series.data().filter((x) => x.close != null).map((x) => x.time)
  }, CHART_FN)
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

// "last swing" when a swing is confirmed, else "session open"; waits for the readout to show the anchor.
async function setAnchor(page) {
  const swing = page.getByTestId('flow-v3-anchor-swing')
  await page.getByTestId(await swing.isEnabled() ? 'flow-v3-anchor-swing' : 'flow-v3-anchor-open').click()
  await expect(page.getByTestId('flow-v3-readout')).toContainText('since ', { timeout: 20_000 })
}

async function anchorFromReadout(page, day) {
  const txt = await page.getByTestId('flow-v3-readout').innerText()
  const m = /since (\d{2}):(\d{2})/.exec(txt)
  if (!m) return null
  // ETH starts 15:00 PT the day before the session (RTH ends 13:00), so a time >= 15:00 is on the prior day
  return shifted(`${day}T${m[1]}:${m[2]}`) - (Number(m[1]) >= 15 ? 86400 : 0)
}

// Levels and readout on one date: the drawn levels and the header text are the payload's.
async function levelsAndReadout(page, day, tag) {
  const resp = page.waitForResponse((r) => r.url().includes('/flow/cells'), { timeout: 120_000 })
  // live sessions re-poll cells every 15 s: compare the header with the last payload the page received
  let latest = null
  page.on('response', async (r) => {
    if (r.url().includes('/flow/cells') && r.ok()) latest = await r.json().catch(() => latest)
  })
  const textFor = (a) => {
    const r = edgeReadout(latest, flattenBars(latest), a, ptShift)
    return edgeText(r, (t) => hhmm(ptShift(t)))
  }
  await open(page, day, { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify(ALL_NEW) } })
  await settle(page)
  const cells = await (await resp).json()
  await expect(page.getByTestId('flow-v3-levels-toggle')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('flow-v3-edge-readout')).toHaveText('edges: set an anchor')
  const rth = cells.segments.find((s) => s.name === 'rth')
  const times = await candleTimes(page)
  const levels = absorptionLevels(cells, times, ptShift)
  const out = { day, rth_tier: rth?.tier, rth_exact: rth?.exact, marks: rth?.marks?.length ?? null, levels_drawn: levels.length }
  if (!rth) {
    expect(levels).toEqual([]) // live before the RTH open: no RTH segment yet, so no marks (DC9: RTH only)
  } else if (!rth.exact) {
    expect(levels).toEqual([])
    await expect(page.getByTestId('flow-v3-marks-note')).toHaveText('Absorbed: exact data only')
  } else {
    expect(levels.length).toBe(Math.min(8, rth.marks.filter((mk) => mk.confirm_time <= rth.bars.t[rth.bars.t.length - 1] + 60).length))
  }
  // anchor: last swing, else session open; the header text is edgeText of the payload and that anchor
  await setAnchor(page)
  const anchor = await anchorFromReadout(page, day)
  expect(anchor).not.toBeNull()
  await expect.poll(async () => (await page.getByTestId('flow-v3-edge-readout').innerText()) === textFor(anchor), { timeout: 20_000 }).toBeTruthy()
  const want = textFor(anchor)
  expect(want).toMatch(/^Top \d+\.\d{2} · (\S+ buyers|exact data only)( · held \d+ m)?( · broke \d{2}:\d{2})? \| Bottom \d+\.\d{2} · (\S+ sellers|exact data only)( · held \d+ m)?( · broke \d{2}:\d{2})?$/)
  out.anchor = hhmm(anchor)
  out.readout = want
  // an anchor inside RTH on an exact session gives contracts (no "exact data only")
  if (rth?.exact && rth.bars.t.length > 30) {
    await page.getByTestId('flow-v3-anchor-pick').click()
    const t = rth.bars.t[Math.floor(rth.bars.t.length / 2)]
    await frame(page, ptShift(t) - 30 * 60, ptShift(t) + 30 * 60)
    const pos = await point(page, ptShift(t), rth.bars.c[Math.floor(rth.bars.t.length / 2)])
    await page.mouse.click(pos.x, pos.y)
    await page.waitForTimeout(600)
    const a2 = await anchorFromReadout(page, day)
    const r2 = edgeReadout(cells, flattenBars(cells), a2, ptShift)
    expect(r2.exact).toBeTruthy()
    expect(r2.top.contracts).not.toBeNull()
    await expect.poll(async () => (await page.getByTestId('flow-v3-edge-readout').innerText()) === textFor(a2), { timeout: 20_000 }).toBeTruthy()
    out.anchor_rth = hhmm(a2)
    out.readout_rth = edgeText(r2, (x) => hhmm(ptShift(x)))
    await frame(page, a2 - 10 * 60, ptShift(r2.to) + 5 * 60)
    fs.mkdirSync(ASSETS, { recursive: true })
    await page.screenshot({ path: path.join(ASSETS, `levels-edges-${tag}.png`) })
  }
  // hover the largest level
  if (levels.length) {
    const lv = [...levels].sort((a, b) => b.contracts - a.contracts)[0]
    await frame(page, lv.startShifted - 15 * 60, Math.max(lv.endShifted, lv.startShifted) + 20 * 60)
    const t = times[Math.min(lv.endIdx, lv.idx + 1)]
    const pos = await point(page, t, lv.price)
    await page.mouse.move(pos.x, pos.y)
    const tip = page.getByTestId('flow-v3-level-tooltip')
    await expect(tip).toContainText('visual aid — untested', { timeout: 5000 })
    await expect(tip).toContainText(lv.kind === 'sellers_absorbed' ? 'Sellers absorbed' : 'Buyers absorbed')
    await expect(tip).toContainText(`held ${lv.heldMin} min`)
    await expect(tip).toContainText(lv.broken ? 'broke at ' : 'not broken')
    fs.mkdirSync(ASSETS, { recursive: true })
    await page.screenshot({ path: path.join(ASSETS, `levels-hover-${tag}.png`) })
    out.hovered = lv.label
  }
  fs.mkdirSync(ASSETS, { recursive: true })
  fs.writeFileSync(path.join(ASSETS, `levels-${tag}.json`), JSON.stringify(out, null, 1))
  return { cells, levels }
}

test('unchanged: Current and v2 are pixel-identical to origin/main, same requests', async ({ browser }) => {
  test.setTimeout(480_000) // four cold page loads
  for (const storage of [{}, { 'ib-react-flow-mode': 'v2' }]) {
    const out = {}
    for (const [name, base] of [['branch', ''], ['main', BASELINE]]) {
      const ctx = await browser.newContext()
      const page = await ctx.newPage()
      const urls = track(page)
      await open(page, '2026-05-01', { base, storage })
      await settle(page)
      out[name] = {
        urls: [...new Set(urls.map(pathOf))].sort(),
        top: await page.locator('.react-top-pane').screenshot(),
        plot: await page.locator('.react-bottom-pane .flow-stage').screenshot(),
        header: await page.locator('.react-bottom-pane .flow-panel-header').screenshot(),
        headerHtml: await page.locator('.react-bottom-pane .flow-panel-header').evaluate((e) => e.outerHTML),
      }
      await ctx.close()
    }
    const tag = JSON.stringify(storage)
    expect(out.branch.urls, `flow requests ${tag}`).toEqual(out.main.urls)
    expect(out.branch.urls.some((u) => u.includes('/flow/cells')), `no cells request ${tag}`).toBeFalsy()
    expect(Buffer.compare(out.branch.top, out.main.top), `price chart ${tag}`).toBe(0)
    expect(Buffer.compare(out.branch.plot, out.main.plot), `flow plot ${tag}`).toBe(0)
    expect(out.branch.headerHtml, `header DOM ${tag}`).toBe(out.main.headerHtml)
    expect(Buffer.compare(out.branch.header, out.main.header), `header ${tag}`).toBe(0)
  }
})

test('defaults: new layers off, Heatmap / Absorbed off in ⚙, no cells request; stored CR-DC layers carry over', async ({ page }) => {
  const urls = track(page)
  await open(page, '2026-05-01')
  await settle(page)
  const current = await page.locator('.react-top-pane').screenshot()
  urls.length = 0
  await page.getByTestId('flow-mode-v3').click()
  await expect(page.getByTestId('flow-v3-panel')).toBeVisible()
  await settle(page)
  for (const id of ['flow-v3-levels-toggle', 'flow-v3-profile-toggle', 'flow-v3-edges-toggle']) {
    await expect(page.getByTestId(id)).toHaveAttribute('aria-pressed', 'false')
  }
  await expect(page.getByTestId('flow-v3-edge-readout')).toHaveCount(0)
  expect(urls.some((u) => u.includes('/flow/cells'))).toBeFalsy()
  expect(Buffer.compare(await page.locator('.react-top-pane').screenshot(), current), 'v3 with layers off = Current').toBe(0)
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  const dlg = page.getByRole('dialog', { name: 'Flow v3 settings' })
  await expect(dlg.getByTestId('flow-v3-heatmap-toggle')).not.toBeChecked()
  await expect(dlg.getByTestId('flow-v3-absorbed-toggle')).not.toBeChecked()
  await expect(dlg).toContainText('Heatmap (cells)')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'How to read Flow v3' }).click()
  const guide = page.getByRole('dialog', { name: 'How to read Flow v3' })
  for (const t of ['Absorption levels', 'Edge readout', 'Heatmap (cells)', 'Absorbed (marks)']) await expect(guide).toContainText(t)
  expect(await guide.getByText('Visual aid — untested.').count()).toBeGreaterThanOrEqual(7)
  await page.keyboard.press('Escape')
})

test('defaults: a stored CR-DC layers object keeps heat / marks (now in ⚙), levels off', async ({ page }) => {
  await open(page, '2026-05-01', { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ heat: true, profile: false, marks: true }) } })
  await settle(page)
  await expect(page.getByTestId('flow-v3-levels-toggle')).toHaveAttribute('aria-pressed', 'false')
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  const dlg = page.getByRole('dialog', { name: 'Flow v3 settings' })
  await expect(dlg.getByTestId('flow-v3-heatmap-toggle')).toBeChecked()
  await expect(dlg.getByTestId('flow-v3-absorbed-toggle')).toBeChecked()
  await dlg.getByTestId('flow-v3-heatmap-toggle').setChecked(false)
  await dlg.getByRole('button', { name: 'Apply' }).click()
  expect(JSON.parse(await page.evaluate(() => localStorage.getItem('ib-react-flow-v3-layers'))))
    .toMatchObject({ heat: false, marks: true, levels: false, edges: false })
})

test('2026-05-01: absorption levels and the edge readout', async ({ page }) => {
  await levelsAndReadout(page, '2026-05-01', '2026-05-01')
})

test('live: absorption levels and the edge readout on the live date', async ({ page }) => {
  const day = liveDate()
  const { cells } = await levelsAndReadout(page, day, `live-${day}`)
  for (const s of cells.segments) expect(['trades', 'price_1m']).toContain(s.tier) // live = exact
})

test('replay: levels and edges read nothing after until', async ({ page }) => {
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
  await open(page, '2026-10-23', { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify(ALL_NEW) } })
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
  const until = shifted(cells.until) + 7 * 3600 // PDT -> UTC
  for (const s of cells.segments) {
    for (const t of s.bars.t) expect(t).toBeLessThan(until)
    for (const mk of s.marks || []) expect(mk.confirm_time).toBeLessThanOrEqual(until)
  }
  const levels = absorptionLevels(cells, await candleTimes(page), ptShift)
  for (const lv of levels) {
    expect(lv.end).toBeLessThanOrEqual(until)
    if (lv.broken) expect(lv.end).toBeLessThan(until)
  }
  await setAnchor(page)
  const anchor = await anchorFromReadout(page, '2026-05-01')
  expect(anchor).not.toBeNull()
  const r = edgeReadout(cells, flattenBars(cells), anchor, ptShift)
  expect(r.to).toBeLessThan(until)
  for (const e of [r.top, r.bottom]) if (e.brokeAt != null) expect(e.brokeAt).toBeLessThan(until)
  await expect(page.getByTestId('flow-v3-edge-readout')).toHaveText(edgeText(r, (t) => hhmm(ptShift(t))))
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.screenshot({ path: path.join(ASSETS, 'replay-2026-05-01.png') })
  fs.writeFileSync(path.join(ASSETS, 'replay-2026-05-01.json'), JSON.stringify({ until: cells.until, levels: levels.length,
    readout: edgeText(r, (t) => hhmm(ptShift(t))) }, null, 1))
  await page.getByRole('button', { name: 'Exit review' }).click()
})
