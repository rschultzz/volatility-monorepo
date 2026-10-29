// CR-DN — Flow v3 ranges as segments: set end, breaks need acceptance, CVD and stats per box, readable absorption
// levels (specs/CR-DN-range-segments.md). Read-only.
//
// Servers (started outside Playwright), as cr-di.spec.js:
//   backend    CRDF_PORT=8066 apps/web/.venv/bin/python -u scripts/cr_dm_smoke_backend.py   (read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5188 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5189
// then:       CRBV_BASE_URL=http://localhost:5188 CRDF_API_BASE=http://127.0.0.1:8066 CRDF_BASELINE_URL=http://localhost:5189 \
//             npx playwright test cr-dn   (from react_price_preview/)
//
// Expected values: the shipped flowV3.js run in Node on the payloads the page itself loaded, and hand sums straight from
// the v2 minute bars (delta, vol) and the price chart's candle closes.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  absorptionLevels, boxLabelText, drawnBoxes, edgeReadout, edgeText, flattenBars, flattenCells, kfmt, makeShifter, profileTotals,
} from '../../src/flowV3.js'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8066'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5189'
const SIDE_A = process.env.CRDN_SIDE_A || ''
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-dn/smoke')
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-dn')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height']
const V3 = { 'ib-react-flow-mode-v3': '1' }
const layers = (o) => ({ 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: false, profile: false, heat: false, marks: false, ...o }) })
const EDGES = { ...V3, ...layers({ edges: true }) }
const settings = (o) => ({ 'ib-react-flow-v3-settings': JSON.stringify({ rollingMin: 10, k: 3, m: 3, sellLayer: true, buyLayer: true, boxHoldMin: 15, rollingOn: false, liveMinMove: 4, breakHoldMin: 5, ...o }) })
const D29 = '2026-10-29'
const D28 = '2026-10-28'
// The live 2026-10-29 session gains a bar a minute and its two flow payloads refresh apart; the hand-sum checks replay it
// to a fixed time so both payloads hold the same minutes.
const D29_UNTIL = process.env.CRDN_D29_UNTIL || `${D29}T09:30`

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
const pt = (utc) => hhmm(ptShift(utc))
const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}

async function open(page, day, { base = '', storage = {}, session = 'RTH', until = null } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdn-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crdn-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
  if (until) {
    // replay: every flow request (v2 and cells) is cut at `until`, as the review mode does
    const cut = (route) => {
      const u = new URL(route.request().url())
      u.searchParams.set('until', until)
      return route.continue({ url: u.toString() })
    }
    await page.route('**/api/ironbeam/flow/cells?*', cut)
    await page.route('**/api/ironbeam/flow?*', cut)
  }
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0&flow_session=${session}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// Collect the payloads the page loads: the cells payload and every v2 flow payload (one per session).
function capture(page) {
  const got = { cells: null, v2: [] }
  page.on('response', async (r) => {
    const u = r.url()
    if (!r.ok() || !u.includes('/api/ironbeam/flow')) return
    try {
      if (u.includes('/flow/cells')) got.cells = await r.json()
      else if (u.includes('mode=v2')) got.v2.push(await r.json())
    } catch (e) { /* ignore */ }
  })
  return got
}

// The price chart (first chart), its candle series and the Flow v3 primitives registry; the Flow v3 pane's chart.
const CHARTS_FN = `(() => {
  const walk = (el, pick) => {
    const key = Object.keys(el.parentElement).find((k) => k.startsWith('__reactFiber'))
    let f = el.parentElement[key]
    const out = {}
    for (let d = 0; f && d < 40; d += 1, f = f.return) {
      for (let h = f.memoizedState; h; h = h.next) {
        const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
        if (c) pick(c, out)
      }
    }
    return out
  }
  const priceEl = document.querySelector('.tv-lightweight-charts')
  const price = walk(priceEl, (c, o) => {
    if (typeof c.timeScale === 'function') o.chart = o.chart || c
    if (typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick') o.series = o.series || c
    if (c && typeof c === 'object' && 'edges' in c && 'levels' in c && 'profile' in c && 'series' in c) o.prims = o.prims || c
  })
  const flowEl = document.querySelector('[data-testid=flow-v3-panel] .tv-lightweight-charts')
  const flow = flowEl ? walk(flowEl, (c, o) => {
    if (typeof c.timeScale === 'function') o.chart = o.chart || c
    if (c.delta && c.rolling && c.cvd) o.reg = o.reg || c
  }) : {}
  return { priceEl, flowEl, price, flow }
})()`

async function frame(page, from, to) {
  await page.evaluate(({ a, b, fn }) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    const data = price.series.data().filter((x) => x.close != null)
    const i0 = data.findIndex((x) => x.time >= a)
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    price.chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to, fn: CHARTS_FN })
  await page.waitForTimeout(800)
}

// Arm a pick (📍 or set end) and click the candle at `shiftedT` on the price chart.
async function pickAt(page, testId, shiftedT, frameFrom = shiftedT - 30 * 60) {
  await frame(page, frameFrom, shiftedT + 30 * 60)
  await page.getByTestId(testId).click()
  const p = await page.evaluate(({ t, fn }) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    const bar = price.series.data().find((x) => x.time === t)
    return { x: rect.x + price.chart.timeScale().timeToCoordinate(t), y: rect.y + price.series.priceToCoordinate(bar.close) }
  }, { t: shiftedT, fn: CHARTS_FN })
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(800)
}

// What the page draws: the CVD series data, the edge lines' data, the levels' data, the profile rows, the window.
async function drawn(page) {
  return page.evaluate((fn) => {
    const { price, flow } = eval(fn) // eslint-disable-line no-eval
    const cvd = flow.reg.cvd.data().map((p) => ({ time: p.time, value: p.value ?? null, gap: p.topLineColor === 'rgba(0,0,0,0)' }))
    const e = price.prims?.edges?._data || null
    const lv = price.prims?.levels
    return {
      cvd,
      // to = the drawn end (a frozen box: its break minute); lastUtc = the box's own last bar (the bar before its break)
      boxes: e ? e.boxes.map((b) => ({ from: b.from, to: b.to, lastUtc: b.box?.to ?? null, top: b.top, bottom: b.bottom, frozen: b.frozen, label: b.label })) : null,
      probes: e ? (e.probes || []).map((q) => ({ time: q.time, price: q.price, label: q.label })) : null,
      levels: lv?._data ? lv._data.levels.map((l) => ({ price: l.price, kind: l.kind, broken: l.broken, contracts: l.contracts, alpha: l.alpha, drawAlpha: l.drawAlpha, start: l.startShifted, end: l.endShifted })) : null,
      tags: lv?._tags ? lv._tags.map((l) => l.price) : null,
      tagViews: lv?.priceAxisViews ? lv.priceAxisViews().filter((v) => v.visible()).map((v) => ({ text: v.text(), back: v.backColor(), y: v.coordinate() })) : null,
      window: price.prims?.window?._data ? { from: price.prims.window._data.from, to: price.prims.window._data.to } : null,
      profile: price.prims?.profile?._data ? price.prims.profile._data.rows[1] : null,
    }
  }, CHARTS_FN)
}

// Hand sums from the minute data: v2 bars (delta, vol) by shifted minute; closes from the price chart's candles.
async function handBox(page, v2, from, to) {
  const closes = await page.evaluate((fn) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    return price.series.data().filter((x) => x.close != null).map((x) => [x.time, x.close])
  }, CHARTS_FN)
  const close = new Map(closes)
  const bars = new Map()
  for (const p of v2) for (const b of p.bars || []) if (Number.isFinite(Number(b.delta))) bars.set(ptShift(Number(b.time)), b)
  let delta = 0
  let vol = 0
  let last = from
  for (const [t, b] of [...bars.entries()].sort((a, c) => a[0] - c[0])) {
    if (t <= from || t > to) continue
    delta += Number(b.delta)
    vol += Number(b.vol)
    last = t
  }
  const dPrice = close.get(last) - close.get(from)
  return { delta, dPrice, minutes: (last - from) / 60, vol, cpp: Math.abs(dPrice) < 1e-9 ? null : vol / Math.abs(dPrice), last: hhmm(last) }
}

// Composited fingerprint of a chart element (all its canvases in order) and its pixels.
const COMPOSITE_FN = `((sel) => {
  const el = document.querySelector(sel)
  const rect = el.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const out = document.createElement('canvas')
  out.width = Math.round(rect.width * dpr)
  out.height = Math.round(rect.height * dpr)
  const ctx = out.getContext('2d')
  for (const c of el.querySelectorAll('canvas')) {
    const r = c.getBoundingClientRect()
    if (!r.width || !r.height) continue
    ctx.drawImage(c, (r.x - rect.x) * dpr, (r.y - rect.y) * dpr, r.width * dpr, r.height * dpr)
  }
  return { out, ctx, dpr }
})`
async function canvasHash(page, selector) {
  return page.evaluate(({ sel, fn }) => eval(fn)(sel).out.toDataURL(), { sel: selector, fn: COMPOSITE_FN }) // eslint-disable-line no-eval
}
async function canvasPixels(page, selector) {
  return page.evaluate(({ sel, fn }) => {
    const { out, ctx, dpr } = eval(fn)(sel) // eslint-disable-line no-eval
    return { w: out.width, h: out.height, dpr, data: Array.from(ctx.getImageData(0, 0, out.width, out.height).data) }
  }, { sel: selector, fn: COMPOSITE_FN })
}

// The GEX price lines' prices on both pages (live snapshots refresh between two loads).
async function gexLines(pa, pb) {
  const get = (page) => page.evaluate((fn) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    return price.chart.panes()[0].getSeries().filter((s) => s.seriesType() === 'Line').map((s) => JSON.stringify(s.data()))
  }, CHARTS_FN)
  const [x, y] = [await get(pa), await get(pb)]
  const differ = x.map((v, i) => (v === y[i] ? null : i)).filter((i) => i != null)
  return { n: x.length, differ, equal: differ.length === 0 && x.length === y.length,
    sample: differ.slice(0, 2).map((i) => ({ branch: x[i].slice(-300), main: y[i].slice(-300) })) }
}

// Item 6 (amendment 3) changes the volume bars on purpose, so the Flow v3 pane is compared by its other series' data and
// its pixels are recorded; the price chart stays pixel-identical.
test('no end, Edges off: price chart pixel-identical to origin/main; Flow v3 CVD / delta / rolling unchanged (2026-10-28, 2026-10-29)', async ({ browser }) => {
  test.setTimeout(600_000)
  const shot = async (base, day) => {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    await open(page, day, { base, storage: V3 })
    await settle(page)
    await frame(page, shifted(`${day}T06:30`), shifted(`${day}T13:00`))
    await page.mouse.move(5, 5)
    await page.waitForTimeout(600)
    const lastBars = () => page.evaluate((fn) => JSON.stringify(eval(fn).price.series.data().slice(-3)), CHARTS_FN) // eslint-disable-line no-eval
    const before = await lastBars()
    const res = {
      price: await canvasHash(page, '.react-top-pane .tv-lightweight-charts'),
      flow: await canvasHash(page, '[data-testid=flow-v3-panel] .flow-stage'),
      readout: await page.getByTestId('flow-v3-readout').innerText(),
      // the live session's forming bar ticks between two loads; the comparison only counts when both saw the same bars
      // read before and after the capture: a tick in between (the last-price label) makes the pair not a comparison
      bars: before,
      barsStable: before === await lastBars(),
      flowBars: await page.evaluate((fn) => JSON.stringify(eval(fn).flow.reg.delta.data().slice(-3)), CHARTS_FN), // eslint-disable-line no-eval
      series: await page.evaluate((fn) => { const r = eval(fn).flow.reg; return JSON.stringify([r.cvd.data(), r.delta.data(), r.rolling.data()]) }, CHARTS_FN), // eslint-disable-line no-eval
    }
    await ctx.close()
    return res
  }
  const rows = []
  for (const day of [D28, D29]) {
    let a
    let b
    let tries = 0
    do {
      ;[a, b] = await Promise.all([shot(SIDE_A, day), shot(BASELINE, day)])
      tries += 1
    } while (day === D29 && (a.price !== b.price || a.series !== b.series) && tries < 8)
    rows.push({ day, tries, price_equal: a.price === b.price, flow_equal: a.flow === b.flow, readout_equal: a.readout === b.readout,
      same_price_bars: a.bars === b.bars && a.barsStable && b.barsStable, same_flow_bars: a.flowBars === b.flowBars, series_equal: a.series === b.series })
    if (a.price !== b.price) {
      fs.mkdirSync(LOGS, { recursive: true })
      for (const [who, r] of [['branch', a], ['main', b]]) fs.writeFileSync(path.join(LOGS, `diff-${day}-price-${who}.png`), Buffer.from(r.price.split(',')[1], 'base64'))
    }
  }
  log('pixel-no-end-edges-off.json', rows)
  for (const r of rows) {
    // a pane whose two loads saw different last bars (live) is not a comparison; the complete session always is
    if (r.day === D28 || r.same_price_bars) expect(r.price_equal, `${r.day}: price chart = origin/main`).toBeTruthy()
    if (r.day === D28 || r.same_flow_bars) expect(r.series_equal, `${r.day}: Flow v3 CVD / delta / rolling = origin/main`).toBeTruthy()
    expect(r.readout_equal).toBeTruthy()
  }
})

test('2026-10-29 anchor 06:49, Edges on, A = 5: 07:34 is a probe; one box from 06:49; label = hand sums', async ({ page }) => {
  test.setTimeout(300_000)
  const got = capture(page)
  await open(page, D29, { storage: EDGES, until: D29_UNTIL })
  await settle(page)
  const anchor = shifted(`${D29}T06:49`)
  await pickAt(page, 'flow-v3-anchor-pick', anchor)
  await page.waitForTimeout(1500)
  const r = edgeReadout(got.cells, flattenBars(got.cells), anchor, ptShift, 15, null, 5)
  expect(ptShift(flattenBars(got.cells).t.at(-1)), 'cells cut at until').toBeLessThan(shifted(D29_UNTIL))
  await expect.poll(async () => page.getByTestId('flow-v3-edge-readout').innerText(), { timeout: 20_000 }).toBe(edgeText(r, pt))
  const d = await drawn(page)
  const allProbes = [...r.frozen, r].flatMap((b) => b.probes)
  const probe0734 = allProbes.find((p) => pt(p.at) === '07:34')
  const res = {
    last_bar: pt(got.cells.segments.at(-1).bars.t.at(-1)), header: edgeText(r, pt),
    frozen: r.frozen.map((f) => ({ from: pt(f.from), broke: pt(f.breakAt), dir: f.dir })),
    current: { from: pt(r.from), bottom: r.bottom.price, top: r.top.price, pending: r.pending ? pt(r.pending.at) : null },
    probes: allProbes.map((p) => ({ at: pt(p.at), ext: p.extreme, min: p.minutesOutside, back: pt(p.back), buy: p.buy, sell: p.sell })),
    drawn_probes: d.probes.map((q) => q.label), drawn_boxes: d.boxes,
  }
  expect(probe0734, '07:34 is a probe').toBeTruthy()
  expect(probe0734).toMatchObject({ minutesOutside: 4, extreme: 7761, buy: 8025, sell: 7432 }) // Step 0d
  expect(d.probes.map((q) => q.label)).toEqual(drawnBoxes(r).boxes.flatMap((b) => b.probes.map((p) => `probe ${pt(p.at)}`)))
  // one box from 06:49 unless a later break held (then the 06:49 box is frozen at that break, not at 07:34)
  if (r.frozen.length === 0) expect(res.current.from).toBe('06:49')
  else expect(res.frozen[0].broke).not.toBe('07:34')
  // per-box labels = hand sums from the minute data
  const hand = []
  for (const b of d.boxes) {
    const h = await handBox(page, got.v2, b.from, ptShift(b.lastUtc))
    hand.push({ box: `${hhmm(b.from)}–${hhmm(ptShift(b.lastUtc))}`, label: b.label, hand: boxLabelText(h), ...h })
    expect(b.label, `box ${hhmm(b.from)} label = hand sums`).toBe(boxLabelText(h))
  }
  res.labels = hand
  // the header readout is the current box
  const cur = hand.at(-1)
  await expect(page.getByTestId('flow-v3-readout')).toContainText(`box from ${hhmm(d.boxes.at(-1).from)}: Δ`)
  res.readout = await page.getByTestId('flow-v3-readout').innerText()
  expect(res.readout).toContain(`${cur.minutes} min`)
  // CVD: one segment from 06:49, 0 there, last value = the box's hand delta; nothing before the anchor
  const cvd = d.cvd.filter((p) => p.value != null)
  expect(cvd[0].time).toBe(d.boxes[0].from)
  expect(cvd[0].value).toBe(0)
  if (r.frozen.length === 0) expect(cvd.at(-1).value).toBe(cur.delta)
  // hover the 07:34 probe tick
  await frame(page, shifted(`${D29}T07:00`), shifted(`${D29}T08:00`))
  const tick = await page.evaluate(({ t, price, fn }) => {
    const { priceEl, price: pc } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    return { x: rect.x + pc.chart.timeScale().timeToCoordinate(t), y: rect.y + pc.series.priceToCoordinate(price) - 3.5 }
  }, { t: shifted(`${D29}T07:34`), price: probe0734.edge, fn: CHARTS_FN })
  await page.mouse.move(tick.x, tick.y)
  const tip = page.getByTestId('flow-v3-probe-tooltip')
  await expect(tip).toContainText('Probe 07:34 ↑ · top 7756.25')
  await expect(tip).toContainText('extreme 7761.00 (4.75 pts beyond) · 4 min outside · back inside 07:38')
  await expect(tip).toContainText('beyond the edge: buyers 8.0k / sellers 7.4k')
  res.probe_hover = await tip.innerText()
  fs.mkdirSync(ASSETS, { recursive: true })
  fs.writeFileSync(path.join(ASSETS, 'anchor-0649-A5.png'), await page.screenshot())
  log('anchor-0649-A5.json', res)
})

test('2026-10-29 anchor 06:49, A = 1: the old split at 07:34, CVD restarts at 0 there with a gap', async ({ page }) => {
  test.setTimeout(300_000)
  const got = capture(page)
  await open(page, D29, { storage: { ...EDGES, ...settings({ breakHoldMin: 1 }) }, until: D29_UNTIL })
  await settle(page)
  const anchor = shifted(`${D29}T06:49`)
  await pickAt(page, 'flow-v3-anchor-pick', anchor)
  await page.waitForTimeout(1500)
  const r = edgeReadout(got.cells, flattenBars(got.cells), anchor, ptShift, 15, null, 1)
  await expect.poll(async () => page.getByTestId('flow-v3-edge-readout').innerText(), { timeout: 20_000 }).toBe(edgeText(r, pt))
  expect(r.frozen.length).toBeGreaterThanOrEqual(1)
  expect({ from: pt(r.frozen[0].from), broke: pt(r.frozen[0].breakAt), dir: r.frozen[0].dir, bottom: r.frozen[0].bottom.price, top: r.frozen[0].top.price })
    .toEqual({ from: '06:49', broke: '07:34', dir: 'up', bottom: 7736, top: 7756.25 })
  const d = await drawn(page)
  const at = (hm) => d.cvd.find((p) => p.time === shifted(`${D29}T${hm}`))
  expect(at('07:34').value, 'CVD restarts at 0 on the break bar').toBe(0)
  expect(at('07:33').gap, 'the line from 07:33 into the next box is not drawn').toBe(true)
  expect(at('07:33').value).not.toBe(null)
  expect(d.cvd.filter((p) => p.gap).length).toBe(Math.min(r.frozen.length, 4))
  const hand = []
  for (const b of d.boxes) {
    const h = await handBox(page, got.v2, b.from, ptShift(b.lastUtc))
    hand.push({ box: `${hhmm(b.from)}–${hhmm(ptShift(b.lastUtc))}`, label: b.label, hand: boxLabelText(h) })
    expect(b.label).toBe(boxLabelText(h))
  }
  expect(at('07:33').value).toBe((await handBox(page, got.v2, anchor, shifted(`${D29}T07:33`))).delta)
  fs.mkdirSync(ASSETS, { recursive: true })
  await frame(page, shifted(`${D29}T06:40`), shifted(`${D29}T08:10`))
  fs.writeFileSync(path.join(ASSETS, 'anchor-0649-A1.png'), await page.screenshot())
  log('anchor-0649-A1.json', { header: edgeText(r, pt), frozen: r.frozen.map((f) => `${pt(f.from)}–${pt(f.breakAt)} ${f.dir}`), labels: hand,
    cvd_0733: at('07:33'), cvd_0734: at('07:34') })
})

test('2026-10-29 set end 06:57: one box 06:49–06:57; readout, strip and profile use only those minutes', async ({ page }) => {
  test.setTimeout(300_000)
  const got = capture(page)
  await open(page, D29, { storage: { ...V3, ...layers({ edges: true, profile: true }), 'ib-react-flow-v3-edge-effort': '1' }, until: D29_UNTIL })
  await settle(page)
  const anchor = shifted(`${D29}T06:49`)
  const end = shifted(`${D29}T06:57`)
  await expect(page.getByTestId('flow-v3-end-pick')).toBeDisabled() // needs an anchor
  await pickAt(page, 'flow-v3-anchor-pick', anchor)
  // a bar at the anchor is ignored (still armed); Esc cancels the pick
  await pickAt(page, 'flow-v3-end-pick', anchor)
  await expect(page.getByTestId('flow-v3-end-pick')).toHaveText('click a bar…')
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('flow-v3-end-pick')).toHaveText('set end')
  await expect(page.getByTestId('flow-v3-window')).toHaveCount(0)
  await pickAt(page, 'flow-v3-end-pick', end, anchor - 10 * 60)
  await expect(page.getByTestId('flow-v3-window')).toHaveText('06:49 → 06:57 · 8 min')
  await page.waitForTimeout(1500)
  const r = edgeReadout(got.cells, flattenBars(got.cells), anchor, ptShift, 15, end, 5)
  await expect.poll(async () => page.getByTestId('flow-v3-edge-readout').innerText(), { timeout: 20_000 }).toBe(edgeText(r, pt))
  expect(r.frozen.length).toBe(0)
  expect([pt(r.from), pt(r.to)]).toEqual(['06:49', '06:57'])
  const d = await drawn(page)
  const h = await handBox(page, got.v2, anchor, end)
  expect(d.boxes.length).toBe(1)
  expect([hhmm(d.boxes[0].from), hhmm(d.boxes[0].to)]).toEqual(['06:49', '06:57'])
  expect(d.boxes[0].label).toBe(boxLabelText(h))
  expect(h.minutes).toBe(8)
  const readout = await page.getByTestId('flow-v3-readout').innerText()
  expect(readout).toContain(`box from 06:49: Δ`)
  expect(readout).toContain('8 min')
  // CVD drawn only 06:49 … 06:57
  const cvd = d.cvd.filter((p) => p.value != null)
  expect([hhmm(cvd[0].time), hhmm(cvd.at(-1).time)]).toEqual(['06:49', '06:57'])
  expect(cvd.at(-1).value).toBe(h.delta)
  // strip: only those minutes, sums = the box's edge counts
  const strip = await page.evaluate((fn) => {
    const { flow } = eval(fn) // eslint-disable-line no-eval
    const sd = flow.reg?.stripPrim?._data
    return sd ? [...sd.cols.entries()].map(([i, c]) => [sd.times[i], ...c]) : null
  }, CHARTS_FN)
  expect(strip.every(([t]) => t >= anchor && t <= end), 'strip minutes inside 06:49–06:57').toBeTruthy()
  expect(strip.reduce((a, c) => a + c[1], 0)).toBe(r.top.contracts)
  expect(strip.reduce((a, c) => a + c[3], 0)).toBe(r.bottom.contracts)
  // profile: cells of [06:49, 06:58) only
  const flat = flattenCells(got.cells, ptShift)
  expect(d.profile).toEqual(profileTotals(flat, anchor, end + 60))
  expect(d.window).toEqual({ from: anchor, to: end })
  // a pick after the payload's last bar is ignored; clear end keeps the anchor; clear removes both
  await page.getByTestId('flow-v3-end-clear').click()
  await expect(page.getByTestId('flow-v3-window')).toHaveCount(0)
  // the end is gone: the boxes run on to until again and the readout is the current (latest) box
  await expect.poll(async () => (await drawn(page)).boxes.length, { timeout: 20_000 }).toBeGreaterThan(1)
  await expect(page.getByTestId('flow-v3-readout')).toContainText('box from ')
  fs.mkdirSync(ASSETS, { recursive: true })
  await pickAt(page, 'flow-v3-end-pick', end, anchor - 10 * 60)
  await frame(page, anchor - 15 * 60, end + 20 * 60)
  fs.writeFileSync(path.join(ASSETS, 'end-0657.png'), await page.screenshot())
  await page.getByTestId('flow-v3-anchor-clear').click()
  await expect(page.getByTestId('flow-v3-window')).toHaveCount(0)
  expect((await drawn(page)).window).toBe(null)
  log('end-0657.json', { header: edgeText(r, pt), readout, label: d.boxes[0].label, hand: h, strip_minutes: strip.length,
    profile_rows: d.profile.length, window: '06:49 → 06:57 · 8 min' })
})

test('2026-10-28 anchor 07:00: breaks that hold still freeze and restart; replay never uses bars after until', async ({ page }) => {
  test.setTimeout(300_000)
  const got = capture(page)
  await open(page, D28, { storage: EDGES })
  await settle(page)
  const anchor = shifted(`${D28}T07:00`)
  await pickAt(page, 'flow-v3-anchor-pick', anchor)
  await page.waitForTimeout(1500)
  const r = edgeReadout(got.cells, flattenBars(got.cells), anchor, ptShift, 15, null, 5)
  await expect.poll(async () => page.getByTestId('flow-v3-edge-readout').innerText(), { timeout: 20_000 }).toBe(edgeText(r, pt))
  const broke = r.frozen.map((f) => `${pt(f.breakAt)} ${f.dir}`)
  expect(broke.slice(0, 4)).toEqual(['08:16 up', '08:39 up', '09:54 up', '11:05 down']) // Step 0d
  const d = await drawn(page)
  for (const b of d.boxes) expect(b.label).toBe(boxLabelText(await handBox(page, got.v2, b.from, ptShift(b.lastUtc))))
  log('held-1028.json', { header: edgeText(r, pt), frozen: broke, probes: r.probes.map((p) => pt(p.at)), drawn: d.boxes.map((b) => `${hhmm(b.from)}–${hhmm(b.to)} ${b.label}`) })
})

test('replay until 2026-10-28 08:09: no bar after until in the boxes or CVD; an end after until is ignored', async ({ page }) => {
  test.setTimeout(300_000)
  const until = `${D28}T08:09`
  const got = capture(page)
  await open(page, D28, { storage: EDGES, until })
  await settle(page)
  const anchor = shifted(`${D28}T07:00`)
  await pickAt(page, 'flow-v3-anchor-pick', anchor)
  await page.waitForTimeout(1500)
  const lastCells = Math.max(...got.cells.segments.map((s) => s.bars.t.at(-1) ?? 0))
  expect(ptShift(lastCells)).toBeLessThan(shifted(until))
  for (const p of got.v2) for (const b of p.bars || []) expect(ptShift(Number(b.time))).toBeLessThan(shifted(until))
  const d = await drawn(page)
  expect(d.boxes.every((b) => ptShift(b.lastUtc) < shifted(until))).toBeTruthy()
  expect(d.cvd.filter((p) => p.value != null).every((p) => p.time < shifted(until))).toBeTruthy()
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D28}T08:30`), anchor)
  await expect(page.getByTestId('flow-v3-window')).toHaveCount(0)
  await page.keyboard.press('Escape')
  log('replay-0809.json', { last_cells_bar: pt(lastCells), boxes: d.boxes.map((b) => `${hhmm(b.from)}–${hhmm(b.to)}`) })
})

// The 330-contract buyers-absorbed level (7760.25, from 08:14 PT) broke at 08:27: the full item 5 check replays to 08:26,
// while it is unbroken; the broken-level check replays to 09:30.
const L330_UNTIL = `${D29}T08:26`
test('item 5, 2026-10-29 until 08:26: the 330-contract buyers-absorbed level over candles, pill label, axis tag; nothing else moved', async ({ browser }) => {
  test.setTimeout(600_000)
  const LV = { ...V3, ...layers({ levels: true }) }
  const load = async (base) => {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    const got = capture(page)
    await open(page, D29, { base, storage: LV, until: L330_UNTIL })
    await settle(page)
    await frame(page, shifted(`${D29}T07:40`), shifted(`${D29}T08:26`))
    await page.mouse.move(5, 5)
    await page.waitForTimeout(800)
    return { ctx, page, got }
  }
  let a
  let b
  let pa
  let pb
  const tried = []
  for (let tries = 0; tries < 4; tries += 1) {
    if (a) {
      await a.ctx.close()
      await b.ctx.close()
    }
    ;[a, b] = await Promise.all([load(SIDE_A), load(BASELINE)])
    const da = await drawn(a.page)
    const db = await drawn(b.page)
    pa = await canvasPixels(a.page, '.react-top-pane .tv-lightweight-charts')
    pb = await canvasPixels(b.page, '.react-top-pane .tv-lightweight-charts')
    const same = JSON.stringify(da.levels?.map((l) => [l.price, l.contracts, l.broken])) === JSON.stringify(db.levels?.map((l) => [l.price, l.contracts, l.broken]))
    tried.push({ same_levels: same, size: [pa.w, pa.h, pb.w, pb.h], branch: da.levels?.length, main: db.levels?.length })
    if (same && pa.w === pb.w && pa.h === pb.h) break
  }
  log('levels-1029-tries.json', tried)
  for (const [who, pg] of [['branch', a.page], ['main', b.page]]) {
    const url = await canvasHash(pg, '.react-top-pane .tv-lightweight-charts')
    fs.mkdirSync(LOGS, { recursive: true })
    fs.writeFileSync(path.join(LOGS, `levels-0826-composite-${who}.png`), Buffer.from(url.split(',')[1], 'base64'))
  }
  const page = a.page
  const d = await drawn(page)
  const want = absorptionLevels(a.got.cells, [], ptShift)
  const lv330 = d.levels.find((l) => l.kind === 'buyers_absorbed' && l.contracts === 330)
  log('levels-1029-drawn.json', { levels: d.levels, tags: d.tagViews })
  expect(lv330, 'the 330-contract buyers-absorbed level is drawn').toBeTruthy()
  // 5c: unbroken drawn at the 0.5 floor or above; broken as before; selection = origin/main's (same prices)
  for (const l of d.levels) {
    if (l.broken) expect(l.drawAlpha).toBe(l.alpha)
    else expect(l.drawAlpha).toBeGreaterThanOrEqual(0.5)
  }
  expect(d.levels.map((l) => l.price)).toEqual((await drawn(b.page)).levels.map((l) => l.price))
  // 5d: an axis tag per unbroken drawn level in view, price only, side colour
  const unbroken = d.levels.filter((l) => !l.broken).map((l) => l.price.toFixed(2))
  expect(d.tagViews.map((v) => v.text).sort()).toEqual(unbroken.sort())
  expect(d.tagViews.find((v) => v.text === lv330.price.toFixed(2)).back).toBe('rgb(232,121,249)')
  // 5a: over a candle body on the level's row, the level's magenta is on top
  const overOn = (pg) => pg.evaluate(({ price, a0, a1, fn, cfn }) => {
    const { price: pc } = eval(fn) // eslint-disable-line no-eval
    const { ctx, dpr } = eval(cfn)('.react-top-pane .tv-lightweight-charts') // eslint-disable-line no-eval
    const y = pc.series.priceToCoordinate(price)
      // a ceiling: in its span candles reach the level with their highs (within a tick) without crossing it
    const lv = pc.series.data().filter((x) => x.close != null && x.time >= a0 && x.time < a1 && x.high >= price - 0.25 && x.low < price - 0.5)
    const hits = []
    for (const bar of lv) {
      const x = pc.chart.timeScale().timeToCoordinate(bar.time)
      if (x == null) continue
      const [r, g, b2] = ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data
      hits.push({ t: bar.time, rgb: [r, g, b2] })
    }
    return hits
  }, { price: lv330.price, a0: lv330.start, a1: lv330.end, fn: CHARTS_FN, cfn: COMPOSITE_FN })
  const over = await overOn(page)
  // 5a on origin/main: the same sample (the levels primitive's zOrder was already 'top')
  const overMain = await overOn(b.page)
  // nothing else moved: every pixel that differs from origin/main lies within a level's row band, its label / pill, or
  // the price axis tags (rows of drawn levels ± 9 px)
  const rows = await page.evaluate(({ prices, fn }) => {
    const { price: pc } = eval(fn) // eslint-disable-line no-eval
    return prices.map((p) => pc.series.priceToCoordinate(p))
  }, { prices: d.levels.map((l) => l.price), fn: CHARTS_FN })
  let diff = 0
  let outside = 0
  const outsideSample = []
  const outsideRows = {}
  for (let y = 0; y < pa.h; y += 1) {
    for (let x = 0; x < pa.w; x += 1) {
      const k = (y * pa.w + x) * 4
      if (pa.data[k] === pb.data[k] && pa.data[k + 1] === pb.data[k + 1] && pa.data[k + 2] === pb.data[k + 2]) continue
      diff += 1
      const yc = y / pa.dpr
      if (!rows.some((ry) => ry != null && Math.abs(yc - ry) <= 30)) {
        outside += 1
        const ry = Math.round(yc)
        const o = outsideRows[ry] || (outsideRows[ry] = { n: 0, x0: x / pa.dpr, branch: [pa.data[k], pa.data[k + 1], pa.data[k + 2]], main: [pb.data[k], pb.data[k + 1], pb.data[k + 2]] })
        o.n += 1
        o.x1 = x / pa.dpr
        if (outsideSample.length < 20) outsideSample.push([x / pa.dpr, yc])
      }
    }
  }
  const magentaOnTop = over.filter(({ rgb: [r, g, bb] }) => r - g > 25 && bb - g > 25).length
  fs.mkdirSync(ASSETS, { recursive: true })
  fs.writeFileSync(path.join(ASSETS, 'levels-1029-0826.png'), await page.screenshot())
  fs.writeFileSync(path.join(ASSETS, 'levels-1029-0826-main.png'), await b.page.screenshot())
  log('levels-1029.json', { levels: d.levels, want_prices: want.map((l) => l.price), tags: d.tagViews, lv330, over_candles: over.length, magenta_on_top: magentaOnTop, main_over: overMain, diff_px: diff, outside_px: outside, outside_sample: outsideSample, outside_rows: outsideRows, gex_lines: await gexLines(a.page, b.page) })
  expect(over.length, 'candles reach the 330 level').toBeGreaterThan(0)
  expect(magentaOnTop, 'the level line paints over the candle bodies').toBe(over.length)
  expect(outside, 'no pixel outside the level rows differs from origin/main').toBe(0)
  await a.ctx.close()
  await b.ctx.close()
})

test('item 5, 2026-10-29 until 09:30: the broken 330 level keeps today\'s fade and has no axis tag', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D29, { storage: { ...V3, ...layers({ levels: true }) }, until: D29_UNTIL })
  await settle(page)
  await frame(page, shifted(`${D29}T07:40`), shifted(`${D29}T09:30`))
  const d = await drawn(page)
  const lv = d.levels.find((l) => l.kind === 'buyers_absorbed' && l.contracts === 330)
  expect(lv.broken).toBe(true)
  expect(lv.drawAlpha).toBe(lv.alpha)
  expect(lv.alpha).toBe(0.25) // 76 min old: today's floor
  expect(d.tagViews.map((v) => v.text)).not.toContain(lv.price.toFixed(2))
  expect(d.tagViews.map((v) => v.text).sort()).toEqual(d.levels.filter((l) => !l.broken).map((l) => l.price.toFixed(2)).sort())
  log('levels-1029-0930.json', { levels: d.levels, tags: d.tagViews })
})

// Item 6: the volume bars' scale tops out at the session's p98 minute; the 57k closing minute of 2026-10-27 is capped.
const D27 = '2026-10-27'
async function volState(page) {
  return page.evaluate((fn) => {
    const { flow } = eval(fn) // eslint-disable-line no-eval
    const s = flow.reg.vol
    const data = s.data().filter((p) => p.value != null)
    const cap = flow.reg.volCap?._data || null
    const sid = s.options().priceScaleId
    // the top of the visible scale = the price at the slot's top edge
    const range = s.priceScale().getVisibleRange?.() || null
    return { n: data.length, maxDrawn: Math.max(...data.map((p) => p.value)), cap: cap?.cap ?? null, capped: cap?.times ?? [], sid, range }
  }, CHARTS_FN)
}

test('item 6, 2026-10-27: the 57k closing minute is capped at the p98; hover shows its true volume; a normal stretch has origin/main\'s bars and scale', async ({ browser }) => {
  test.setTimeout(600_000)
  const load = async (base) => {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    const got = capture(page)
    await open(page, D27, { base, storage: V3 })
    await settle(page)
    return { ctx, page, got }
  }
  const [a, b] = await Promise.all([load(''), load(BASELINE)])
  const page = a.page
  // full day
  await frame(page, shifted(`${D27}T06:30`), shifted(`${D27}T13:00`))
  await frame(b.page, shifted(`${D27}T06:30`), shifted(`${D27}T13:00`))
  const v = await volState(page)
  const rows = []
  for (const p of a.got.v2) for (const bar of p.bars || []) if (Number.isFinite(Number(bar.vol))) rows.push({ t: ptShift(Number(bar.time)), vol: Number(bar.vol) })
  const rth = rows.filter((r) => hhmm(r.t) >= '06:30' && hhmm(r.t) < '13:00')
  const top = rth.reduce((m, r) => (r.vol > m.vol ? r : m), rth[0])
  const vMain = await volState(b.page)
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.mouse.move(5, 5)
  fs.writeFileSync(path.join(ASSETS, 'volume-1027-fullday.png'), await page.locator('[data-testid=flow-v3-panel]').screenshot())
  fs.writeFileSync(path.join(ASSETS, 'volume-1027-fullday-main.png'), await b.page.locator('[data-testid=flow-v3-panel]').screenshot())
  expect(top.vol, 'the closing minute is the largest').toBeGreaterThan(50_000)
  expect(v.cap).toBeLessThan(top.vol / 5)
  expect(v.maxDrawn).toBe(v.cap)
  expect(v.capped).toContain(top.t)
  expect(vMain.maxDrawn).toBe(top.vol) // origin/main scales to the largest minute
  // hover the capped bar: the true volume
  const h = await page.evaluate(({ t, fn }) => {
    const { flowEl, flow } = eval(fn) // eslint-disable-line no-eval
    const rect = flowEl.getBoundingClientRect()
    return { x: rect.x + flow.chart.timeScale().timeToCoordinate(t), y: rect.y + flow.chart.paneSize(0).height * 0.55 }
  }, { t: top.t, fn: CHARTS_FN })
  await page.mouse.move(h.x, h.y)
  await expect(page.getByTestId('flow-v3-tooltip')).toContainText(`${hhmm(top.t)} · vol ${kfmt(top.vol)}`)
  const hover = await page.getByTestId('flow-v3-tooltip').innerText()
  // zoomed in on a stretch with no clipped minute: the pane draws exactly as origin/main
  const z0 = shifted(`${D27}T10:00`)
  const z1 = shifted(`${D27}T11:00`)
  const clippedIn = v.capped.filter((t) => t >= z0 - 120 && t <= z1 + 120)
  // fresh loads framed straight on the stretch (after the full-day view, lightweight-charts keeps axis tick labels laid
  // out for the previous range, which differs between the builds: 7.3k vs 57k)
  const [za0, zb0] = await Promise.all([load(''), load(BASELINE)])
  for (const x of [za0, zb0]) {
    await frame(x.page, z0, z1)
    await x.page.mouse.move(5, 5)
    await x.page.waitForTimeout(800)
  }
  const za = await canvasHash(za0.page, '[data-testid=flow-v3-panel] .flow-stage')
  const zb = await canvasHash(zb0.page, '[data-testid=flow-v3-panel] .flow-stage')
  const zr = async (pg) => pg.evaluate((fn) => { const { flow } = eval(fn); return { vis: flow.chart.timeScale().getVisibleLogicalRange(), vol: flow.reg.vol.priceScale().options().scaleMargins, ar: flow.chart.priceScale('right').getVisibleRange?.() } }, CHARTS_FN) // eslint-disable-line no-eval
  const zr2 = async (pg) => pg.evaluate((fn) => { const { flow, price } = eval(fn); const ts = flow.chart.timeScale(); const pts = price.chart.timeScale(); const t1030 = price.series.data().find((d) => new Date(d.time * 1000).toISOString().slice(11, 16) === '10:30' && d.close != null)?.time; return { priceW: pts.width(), priceAxis: price.chart.priceScale('right').width(), x1030flow: ts.timeToCoordinate(t1030), x1030price: pts.timeToCoordinate(t1030), vis: ts.getVisibleLogicalRange(), w: ts.width(), rw: flow.chart.priceScale('right').width(), ar: flow.chart.priceScale('right').getVisibleRange?.(), cvd: flow.chart.priceScale('cvd').getVisibleRange?.(), delta: flow.chart.priceScale('delta').getVisibleRange?.(), volMax: Math.max(...flow.reg.vol.data().filter((p) => p.value != null).map((p) => p.value)), volN: flow.reg.vol.data().length } }, CHARTS_FN) // eslint-disable-line no-eval
  const zinfo = { branch: await zr(page), main: await zr(b.page), freshBranch: await zr2(za0.page), freshMain: await zr2(zb0.page) }
  const pxA = await canvasPixels(za0.page, '[data-testid=flow-v3-panel] .flow-stage')
  const pxB = await canvasPixels(zb0.page, '[data-testid=flow-v3-panel] .flow-stage')
  const bbox = { x0: 1e9, y0: 1e9, x1: -1, y1: -1, n: 0, size: [pxA.w, pxA.h, pxB.w, pxB.h] }
  if (pxA.w === pxB.w && pxA.h === pxB.h) {
    for (let y = 0; y < pxA.h; y += 1) for (let x = 0; x < pxA.w; x += 1) {
      const k = (y * pxA.w + x) * 4
      if (pxA.data[k] !== pxB.data[k] || pxA.data[k + 1] !== pxB.data[k + 1] || pxA.data[k + 2] !== pxB.data[k + 2]) {
        bbox.n += 1
        bbox.x0 = Math.min(bbox.x0, x / pxA.dpr); bbox.x1 = Math.max(bbox.x1, x / pxA.dpr)
        bbox.y0 = Math.min(bbox.y0, y / pxA.dpr); bbox.y1 = Math.max(bbox.y1, y / pxA.dpr)
      }
    }
  }
  // and after the full-day view (recorded, not asserted)
  await frame(page, z0, z1)
  await frame(b.page, z0, z1)
  await page.mouse.move(5, 5)
  await b.page.mouse.move(5, 5)
  await page.waitForTimeout(600)
  const afterFullDayEqual = (await canvasHash(page, '[data-testid=flow-v3-panel] .flow-stage')) === (await canvasHash(b.page, '[data-testid=flow-v3-panel] .flow-stage'))
  fs.writeFileSync(path.join(ASSETS, 'volume-1027-1000-1100.png'), await page.locator('[data-testid=flow-v3-panel]').screenshot())
  fs.writeFileSync(path.join(LOGS, 'volume-1027-1000-1100-main.png'), await b.page.locator('[data-testid=flow-v3-panel]').screenshot())
  log('volume-1027.json', { cap: v.cap, capped: v.capped.map(hhmm), top: { t: hhmm(top.t), vol: top.vol }, main_max: vMain.maxDrawn,
    hover, zoom: '10:00–11:00', clipped_in_zoom: clippedIn.map(hhmm), zoom_equal_main: za === zb, zoom_after_full_day_equal_main: afterFullDayEqual, zinfo, zoom_diff_bbox: bbox })
  expect(clippedIn.length, '10:00–11:00 holds no clipped minute').toBe(0)
  // the stretch's bars are the same values on the same scale as origin/main; the pixels differ only by the volume axis
  // width (lightweight-charts sizes it from the tick labels in view: 52 vs 60 px here), which shifts the plot (recorded)
  const barsIn = (pg) => pg.evaluate(({ a0, a1, fn }) => eval(fn).flow.reg.vol.data().filter((p) => p.time >= a0 && p.time <= a1).map((p) => [p.time, p.value ?? null]), { a0: z0, a1: z1, fn: CHARTS_FN }) // eslint-disable-line no-eval
  expect(await barsIn(za0.page)).toEqual(await barsIn(zb0.page))
  expect(zinfo.freshBranch.ar).toEqual(zinfo.freshMain.ar)
  for (const x of [a, b, za0, zb0]) await x.ctx.close()
})
