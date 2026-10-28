// CR-DG — absorption colour by side, held minutes to the payload's last bar, cursor time snaps to the bar
// (specs/CR-DG-absorption-side-colours.md). Read-only.
//
// Servers (started outside Playwright), as cr-df.spec.js:
//   backend    apps/web/.venv/bin/python -u scripts/cr_dg_smoke_backend.py   (port 8064, read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5181 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5180
// then:       CRBV_BASE_URL=http://localhost:5181 npx playwright test cr-dg   (from react_price_preview/)
//
// "2026-10-27": FULL session, levels on — every drawn level is coloured by side (pixels on the chart canvas), the
//   12:30–12:50 PT buyers-absorbed stack is magenta, held minutes of unbroken levels run to the payload's end, and the
//   level / mark hovers say so. "replay": the cells request carries until=12:55 (what a review replay sends).
// "cursor": the tooltip time on the 07:22 PT candle reads 07:22 anywhere across the bar (branch; origin/main logged).
// "live": an unbroken level's held minutes grow between two refreshes, in the label data and in an open hover.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { absorptionLevels, chartMarks, makeShifter, payloadNow } from '../../src/flowV3.js'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8064'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5180'
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-dg/smoke')
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-dg')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers']
const LEVELS_ON = { levels: true, edges: false, profile: false, heat: false, marks: false }
const DAY = '2026-10-27'
const MAGENTA = [232, 121, 249]
const ORANGE = [251, 146, 60]

function liveDate() {
  if (process.env.CRDF_LIVE_DATE) return process.env.CRDF_LIVE_DATE
  const pt = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/Los_Angeles' }))
  if (pt.getHours() >= 15) pt.setDate(pt.getDate() + 1)
  while (pt.getDay() === 0 || pt.getDay() === 6) pt.setDate(pt.getDate() + 1)
  return `${pt.getFullYear()}-${String(pt.getMonth() + 1).padStart(2, '0')}-${String(pt.getDate()).padStart(2, '0')}`
}

const shifted = (ptIso) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}
const ptShift = makeShifter((t) => {
  const d = new Date(t * 1000)
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hourCycle: 'h23',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
    .formatToParts(d).map((x) => [x.type, x.value]))
  return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second) / 1000
})
const hhmm = (shiftedT) => new Date(shiftedT * 1000).toISOString().slice(11, 16)
const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}

async function open(page, day, { base = '', storage = {}, session = 'FULL', until = null, interval = '1min', either = 0 } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdg-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        // CR-DH: origin/main defaults Levels to Current; compare like with like
        window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
        window.localStorage.setItem('ib-react-gex-levels-mode-v2', 'current')
        window.sessionStorage.setItem('crdg-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
  if (until) {
    await page.route('**/api/ironbeam/flow/cells?*', (route) => {
      const u = new URL(route.request().url())
      u.searchParams.set('until', until)
      return route.continue({ url: u.toString() })
    })
  }
  await page.goto(`${base}/?trade_date=${day}&interval=${interval}&days_either_side=${either}&flow_session=${session}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

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
    const x = chart.timeScale().timeToCoordinate(t)
    return { x: rect.x + x, y: rect.y + series.priceToCoordinate(p), spacing: chart.timeScale().options().barSpacing }
  }, { time, price, fn: CHART_FN })
}

async function candleTimes(page) {
  return page.evaluate((fn) => {
    const { series } = eval(fn) // eslint-disable-line no-eval
    return series.data().filter((x) => x.close != null).map((x) => x.time)
  }, CHART_FN)
}

// Composited RGB of the price chart at css points (all of the chart's canvases drawn in order).
async function pixels(page, pts) {
  return page.evaluate(({ pts: ps, fn }) => {
    const { chartEl } = eval(fn) // eslint-disable-line no-eval
    const rect = chartEl.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const out = document.createElement('canvas')
    out.width = Math.round(rect.width * dpr)
    out.height = Math.round(rect.height * dpr)
    const ctx = out.getContext('2d')
    for (const c of chartEl.querySelectorAll('canvas')) {
      const r = c.getBoundingClientRect()
      if (!r.width || !r.height) continue
      ctx.drawImage(c, (r.x - rect.x) * dpr, (r.y - rect.y) * dpr, r.width * dpr, r.height * dpr)
    }
    return ps.map((p) => [...ctx.getImageData(Math.round((p.x - rect.x) * dpr), Math.round((p.y - rect.y) * dpr), 1, 1).data].slice(0, 3))
  }, { pts, fn: CHART_FN })
}

// Which absorption colour a pixel is nearest after alpha over the session background (ETH / RTH shading differs);
// null when the background (or anything else) is nearer.
const BGS = [[31, 41, 55], [75, 85, 99]]
function sideOf(rgb, alpha) {
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
  let best = null
  let bestD = 30
  for (const bg of BGS) {
    if (d(rgb, bg) < bestD) return null
    for (const [side, c] of [['buyers_absorbed', MAGENTA], ['sellers_absorbed', ORANGE]]) {
      const dd = d(rgb, c.map((v, i) => alpha * v + (1 - alpha) * bg[i]))
      if (dd < bestD) [best, bestD] = [side, dd]
    }
  }
  return best
}

async function cellsOnLoad(page, run) {
  const resp = page.waitForResponse((r) => r.url().includes('/flow/cells') && r.ok(), { timeout: 180_000 })
  await run()
  return (await resp).json()
}

async function levelColours(page, levels, times) {
  const rows = []
  for (const lv of levels) {
    if (lv.endIdx <= lv.idx + 1) continue // too short to sample clear of the candles
    await frame(page, lv.startShifted - 10 * 60, lv.endShifted + 10 * 60)
    const xs = []
    for (let i = lv.idx; i <= lv.endIdx; i += 1) xs.push(times[i])
    const pts = []
    for (const t of xs) {
      const p = await point(page, t, lv.price)
      for (const dy of [-1, 0, 1]) pts.push({ x: p.x + p.spacing / 2, y: p.y + dy }) // between two candles, on the line
    }
    const px = await pixels(page, pts)
    const sides = px.map((c) => sideOf(c, lv.alpha))
    const hit = sides.filter(Boolean)
    rows.push({ kind: lv.kind, price: lv.price, from: hhmm(lv.startShifted), alpha: Number(lv.alpha.toFixed(2)),
      samples: px.length, matched: hit.length, sides: [...new Set(hit)], px: px.slice(0, 9) })
  }
  return rows
}

test('2026-10-27: levels coloured by side; held minutes run to the payload end; hovers agree', async ({ page }) => {
  test.setTimeout(300_000)
  const cells = await cellsOnLoad(page, async () => {
    await open(page, DAY, { storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ ...LEVELS_ON, marks: true }) } })
  })
  await settle(page)
  const times = await candleTimes(page)
  const now = payloadNow(cells)
  const levels = absorptionLevels(cells, times, ptShift)
  expect(levels.length).toBeGreaterThan(0)
  // held: unbroken to the payload's last bar, broken to the break
  for (const lv of levels) expect(lv.heldMin).toBe(Math.round(((lv.broken ? lv.end : now) - lv.start) / 60))
  const stack = levels.filter((lv) => hhmm(lv.startShifted) >= '12:30' && hhmm(lv.startShifted) <= '12:50')
  expect(stack.some((lv) => lv.kind === 'buyers_absorbed')).toBeTruthy()
  // colours on the canvas
  const colours = await levelColours(page, levels, times)
  log('2026-10-27-colours.json', colours)
  for (const r of colours) {
    expect(r.matched, `${r.kind} ${r.price} from ${r.from}`).toBeGreaterThan(0)
    expect(r.sides, `${r.kind} ${r.price} from ${r.from}`).toEqual([r.kind])
  }
  // screenshot of the 12:15–13:00 stack
  await frame(page, shifted(`${DAY}T12:10`), shifted(`${DAY}T13:00`))
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.screenshot({ path: path.join(ASSETS, 'levels-2026-10-27.png') })
  // level hover on the newest unbroken level: held = start to the payload's last bar
  const lv = levels.filter((x) => !x.broken).sort((a, b) => b.start - a.start)[0]
  const pos = await point(page, times[lv.idx], lv.price)
  await page.mouse.move(pos.x + pos.spacing / 2, pos.y)
  const tip = page.getByTestId('flow-v3-level-tooltip')
  await expect(tip).toContainText(`held ${Math.round((now - lv.start) / 60)} min · not broken`, { timeout: 5000 })
  const levelTip = await tip.innerText()
  // mark hover: client heldMin (to the payload end), not the server's held_min
  const marks = chartMarks(cells, times, ptShift)
  const mk = marks.filter((m) => m.break_time == null).sort((a, b) => b.time - a.time)[0]
  let markTip = null
  if (mk) {
    await page.mouse.move(10, 10)
    await frame(page, mk.candleTime - 15 * 60, mk.candleTime + 10 * 60)
    const mp = await page.evaluate(({ t, fn }) => {
      const { chartEl, chart } = eval(fn) // eslint-disable-line no-eval
      return { x: chartEl.getBoundingClientRect().x + chart.timeScale().timeToCoordinate(t) }
    }, { t: mk.candleTime, fn: CHART_FN })
    const mtip = page.getByTestId('flow-v3-mark-tooltip')
    // the marker sits above / below its candle: sweep vertically at the candle's x until the hover opens
    const rect = await page.locator('.tv-lightweight-charts').first().boundingBox()
    for (let y = rect.y + 10; y < rect.y + rect.height - 30 && !(await mtip.count()); y += 4) await page.mouse.move(mp.x, y)
    await expect(mtip).toContainText(`held ${mk.heldMin} min · not broken`, { timeout: 5000 })
    expect(mk.heldMin).toBe(Math.round((now - mk.time) / 60))
    markTip = await mtip.innerText()
  }
  log('2026-10-27.json', {
    payload_now: hhmm(ptShift(now)),
    levels: levels.map((x) => ({ kind: x.kind, price: x.price, from: hhmm(x.startShifted), to: hhmm(x.endShifted), broken: x.broken, label: x.label })),
    colours, levelTip, markTip, mark_server_held_min: mk?.held_min ?? null, mark_client_held_min: mk?.heldMin ?? null,
  })
})

test('replay (until 12:55 PT): unbroken levels held to until', async ({ page }) => {
  test.setTimeout(300_000)
  const cells = await cellsOnLoad(page, async () => {
    await open(page, DAY, { until: `${DAY}T12:55:00`, storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify(LEVELS_ON) } })
  })
  await settle(page)
  expect(cells.until).toBe(`${DAY}T12:55:00`)
  const now = payloadNow(cells)
  expect(hhmm(ptShift(now))).toBe('12:55')
  const times = await candleTimes(page)
  const levels = absorptionLevels(cells, times, ptShift)
  const unbroken = levels.filter((x) => !x.broken)
  expect(unbroken.length).toBeGreaterThan(0)
  for (const lv of unbroken) expect(lv.end).toBe(now)
  const colours = await levelColours(page, levels, times)
  for (const r of colours) expect(r.sides, `${r.kind} ${r.price} from ${r.from}`).toEqual([r.kind])
  log('replay-1255.json', { payload_now: '12:55', levels: levels.map((x) => x.label), colours })
})

// The tooltip time across the 07:22 candle (07:20 on 5-minute bars): left quarter, centre, right quarter of the bar,
// GEX lines on. Variants: one session, and a day either side / 5-minute bars, where other series' time points (GEX
// segment ends between sessions) sit inside the data.
const CURSOR_VARIANTS = [
  { name: '1min-0d', interval: '1min', either: 0, bar: '07:22' },
  { name: '1min-1d', interval: '1min', either: 1, bar: '07:22' },
  { name: '5min-1d', interval: '5min', either: 1, bar: '07:20' },
]

async function cursorAt(page, base, v) {
  await open(page, DAY, { base, interval: v.interval, either: v.either })
  await settle(page)
  const t = shifted(`${DAY}T${v.bar}`)
  await frame(page, t - 20 * 60, t + 20 * 60)
  const p = await point(page, t, 0)
  const rect = await page.locator('.tv-lightweight-charts').first().boundingBox()
  const y = rect.y + rect.height * 0.4
  const time = page.locator('div', { hasText: /^Cursor$/ }).locator('xpath=following-sibling::div[1]')
  const out = []
  for (const dx of [-0.3, 0, 0.3]) {
    await page.mouse.move(p.x + dx * p.spacing, y)
    await page.waitForTimeout(150)
    out.push(await time.innerText())
  }
  return out
}

test('cursor: tooltip time on the 07:22 PT candle reads 07:22', async ({ browser }) => {
  test.setTimeout(600_000)
  const res = {}
  for (const v of CURSOR_VARIANTS) {
    for (const [name, base] of [['branch', ''], ['main', BASELINE]]) {
      const ctx = await browser.newContext()
      res[`${v.name}:${name}`] = await cursorAt(await ctx.newPage(), base, v)
      await ctx.close()
    }
  }
  log('cursor-0722.json', res)
  for (const v of CURSOR_VARIANTS) for (const txt of res[`${v.name}:branch`]) expect(txt, v.name).toBe(`10/27 ${v.bar}:00 PT`)
})

test('live: an unbroken level\'s held minutes grow between two refreshes (labels and an open hover)', async ({ page }) => {
  test.setTimeout(480_000)
  const day = liveDate()
  const payloads = []
  page.on('response', async (r) => {
    if (r.url().includes('/flow/cells') && r.ok()) payloads.push(await r.json().catch(() => null))
  })
  await open(page, day, { session: 'RTH', storage: { 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify(LEVELS_ON) } })
  await settle(page)
  const times = await candleTimes(page)
  const first = payloads.at(-1)
  const lv0 = absorptionLevels(first, times, ptShift).filter((x) => !x.broken).sort((a, b) => b.start - a.start)[0]
  test.skip(!lv0, `no unbroken level on ${day} yet (RTH marks only)`)
  await frame(page, lv0.startShifted - 10 * 60, lv0.endShifted + 5 * 60)
  const pos = await point(page, times[lv0.idx], lv0.price)
  await page.mouse.move(pos.x + pos.spacing / 2, pos.y)
  const tip = page.getByTestId('flow-v3-level-tooltip')
  await expect(tip).toContainText(`held ${lv0.heldMin} min`, { timeout: 5000 })
  const tip0 = await tip.innerText()
  // pointer still; wait for a refresh whose last bar is at least two minutes later
  const n0 = payloads.length
  await expect.poll(() => payloads.length > n0 && payloadNow(payloads.at(-1)) >= payloadNow(first) + 120, { timeout: 300_000, intervals: [5000] }).toBeTruthy()
  const last = payloads.at(-1)
  const lv1 = absorptionLevels(last, await candleTimes(page), ptShift).find((x) => x.kind === lv0.kind && x.start === lv0.start && x.price === lv0.price)
  expect(lv1).toBeTruthy()
  if (!lv1.broken) {
    expect(lv1.heldMin).toBeGreaterThan(lv0.heldMin)
    await expect(tip).toContainText(`held ${lv1.heldMin} min`, { timeout: 20_000 })
  }
  log(`live-${day}.json`, { level: lv0.label, first_now: hhmm(ptShift(payloadNow(first))), last_now: hhmm(ptShift(payloadNow(last))),
    held_first: lv0.heldMin, held_last: lv1.heldMin, broken_last: lv1.broken, tip0, tip1: await tip.innerText().catch(() => null) })
})

// Item 4 — Effort in Flow v3, the same switch as v2. Price chart only (the v3 header gains the chip, by spec).
async function priceChart(page, base, storage) {
  await open(page, DAY, { base, storage })
  await settle(page)
  await frame(page, shifted(`${DAY}T06:30`), shifted(`${DAY}T09:30`))
  await page.mouse.move(5, 5)
  await page.waitForTimeout(500)
  return page.locator('.react-top-pane').screenshot()
}

test('effort in v3: off = chart unchanged, on = the v2 labels, one switch for both modes', async ({ browser }) => {
  test.setTimeout(600_000)
  const V3 = { 'ib-react-flow-mode-v3': '1' }
  const V2 = { 'ib-react-flow-mode': 'v2' }
  const ON = { 'ib-react-flow-v2-effort': '1' }
  const shot = async (base, storage) => {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    const png = await priceChart(page, base, storage)
    await ctx.close()
    return png
  }
  const v3offBranch = await shot('', V3)
  const v3offMain = await shot(BASELINE, V3)
  const v3on = await shot('', { ...V3, ...ON })
  const v2on = await shot('', { ...V2, ...ON })
  fs.mkdirSync(ASSETS, { recursive: true })
  fs.writeFileSync(path.join(ASSETS, 'effort-v3-on-2026-10-27.png'), v3on)
  fs.writeFileSync(path.join(ASSETS, 'effort-v3-off-2026-10-27.png'), v3offBranch)
  const res = {
    v3_off_equals_main_v3: Buffer.compare(v3offBranch, v3offMain) === 0,
    v3_on_equals_v2_on: Buffer.compare(v3on, v2on) === 0,
    v3_on_differs_from_off: Buffer.compare(v3on, v3offBranch) !== 0,
  }
  // the switch: default off; on in v2 -> on in v3; off in v3 -> off in v2; stored once
  const ctx = await browser.newContext()
  const page = await ctx.newPage()
  await open(page, DAY, { storage: V2 })
  await settle(page)
  await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveAttribute('aria-pressed', 'false')
  await page.getByTestId('flow-v2-effort-toggle').click()
  await page.getByTestId('flow-mode-v3').click()
  await expect(page.getByTestId('flow-v3-effort-toggle')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('flow-v3-effort-toggle')).toHaveText('Effort on')
  res.stored_after_v2_on = await page.evaluate(() => localStorage.getItem('ib-react-flow-v2-effort'))
  await page.getByTestId('flow-v3-effort-toggle').click()
  await expect(page.getByTestId('flow-v3-effort-toggle')).toHaveAttribute('aria-pressed', 'false')
  await page.getByTestId('flow-mode-v2').click()
  await expect(page.getByTestId('flow-v2-effort-toggle')).toHaveAttribute('aria-pressed', 'false')
  res.stored_after_v3_off = await page.evaluate(() => localStorage.getItem('ib-react-flow-v2-effort'))
  await ctx.close()
  log('effort-v3.json', res)
  expect(res.v3_off_equals_main_v3, 'v3, Effort off: price chart = origin/main').toBeTruthy()
  expect(res.v3_on_equals_v2_on, 'v3, Effort on: price chart = v2 with Effort on').toBeTruthy()
  expect(res.v3_on_differs_from_off, 'v3, Effort on draws labels').toBeTruthy()
  expect(res.stored_after_v2_on).toBe('1')
  expect(res.stored_after_v3_off).toBe('0')
})
