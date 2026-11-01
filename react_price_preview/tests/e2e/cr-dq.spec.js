// CR-DQ — box panel, verdict line, percentile badges, faint box shading, no in-chart box labels
// (specs/CR-DQ-box-panel.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    CRDP_PORT=8068 apps/web/.venv/bin/python -u scripts/cr_dp_smoke_backend.py   (read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5193 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5194
// then:       CRBV_BASE_URL=http://localhost:5193 CRDQ_API_BASE=http://127.0.0.1:8068 CRDQ_BASELINE_URL=http://localhost:5194 \
//             npx playwright test cr-dq   (from react_price_preview/)
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { boxLabelText, boxPercentiles, kfmt, makeShifter, pctTier } from '../../src/flowV3.js'

const API_BASE = process.env.CRDQ_API_BASE || 'http://127.0.0.1:8068'
const BASELINE = process.env.CRDQ_BASELINE_URL || 'http://localhost:5194'
const HERE = path.dirname(fileURLToPath(import.meta.url))
const LOGS = path.resolve(HERE, '../../../scripts/logs/cr-dq/smoke')
const ASSETS = path.resolve(HERE, '../../../specs/assets/cr-dq')
const BASE_JSON = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../src/data/boxBaseline.json')))
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height',
  'ib-react-flow-v3-slots', 'ib-react-price-zoom', 'ib-react-box-panel-pos']
const layers = (o) => ({ 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: false, profile: false, heat: false, marks: false, ...o }) })
const EDGES = layers({ edges: true })
const EDGES_LEVELS = layers({ edges: true, levels: true })
const D28 = '2026-10-28'
const D29 = '2026-10-29'
const D30 = '2026-10-30'

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
const ptMinute = (utc) => Math.floor((ptShift(utc) % 86400) / 60)
const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}
const shot = async (page, name) => {
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.screenshot({ path: path.join(ASSETS, name) })
}
const noBadges = (s) => s.replace(/ · p\d{1,2}(?= ·|$)/g, '')

async function open(page, day, { base = '', storage = {}, session = 'RTH', until = null, keepStorage = false } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  if (!keepStorage) {
    await page.addInitScript(({ keys, values }) => {
      try {
        if (!window.sessionStorage.getItem('crdq-init')) {
          for (const k of keys) window.localStorage.removeItem(k)
          for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
          window.localStorage.setItem('ib-react-flow-mode-v3', '1')
          // no GEX lines: an armed anchor click on a GEX line opens its popup instead (not this CR's subject)
          window.localStorage.setItem('ib-react-gex-min-abs-b', '9999')
          window.sessionStorage.setItem('crdq-init', '1')
        }
      } catch (e) { /* ignore */ }
    }, { keys: FLOW_KEYS, values: storage })
  }
  if (until) {
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
  await page.getByTestId('flow-v3-panel').waitFor()
  await settle(page)
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(2000)
}

// The price chart / Flow v3 objects, through React's fiber tree (test-only, as cr-dn.spec.js).
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
  return { priceEl, price }
})()`

async function frame(page, from, to) {
  await page.evaluate(({ a, b, fn }) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    const data = price.series.data().filter((x) => x.close != null)
    const i0 = Math.max(0, data.findIndex((x) => x.time >= a))
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    price.chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to, fn: CHARTS_FN })
  await page.waitForTimeout(800)
}

async function pickAt(page, testId, shiftedT, frameFrom = shiftedT - 30 * 60, frameTo = shiftedT + 3 * 3600) {
  await frame(page, frameFrom, frameTo)
  await page.getByTestId(testId).click()
  const p = await page.evaluate(({ t, fn }) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    const data = price.series.data().filter((x) => x.close != null)
    const bar = data.find((x) => x.time >= t) || data[data.length - 1]
    return { x: rect.x + price.chart.timeScale().timeToCoordinate(bar.time), y: rect.y + price.series.priceToCoordinate(bar.close) }
  }, { t: shiftedT, fn: CHARTS_FN })
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(1500)
  return p
}
async function anchorAt(page, t, from, to) {
  await pickAt(page, 'flow-v3-anchor-pick', t, from, to)
  const got = await page.locator('[data-anchor]').first().getAttribute('data-anchor')
  expect(Number(got), `anchor set at ${hhmm(t)}`).toBe(t)
}

// Drawn boxes as the edges primitive holds them (plus whether any label is set), the shade primitive's data, the levels.
const drawn = (page) => page.evaluate((fn) => {
  const { price } = eval(fn) // eslint-disable-line no-eval
  const e = price.prims?.edges?._data || null
  const strip = (b) => JSON.parse(JSON.stringify(b))
  return {
    boxes: e ? e.boxes.map((b) => ({ from: b.from, to: b.to, top: b.top, bottom: b.bottom, frozen: b.frozen, label: b.label ?? null, box: strip(b.box) })) : null,
    shade: price.prims?.boxShade?._data ? price.prims.boxShade._data.boxes.map((b) => ({ from: b.from, to: b.to, top: b.top, bottom: b.bottom, frozen: b.frozen })) : null,
    levels: price.prims?.levels?._data ? strip(price.prims.levels._data.levels) : null,
  }
}, CHARTS_FN)

// The plot rect (page px): the price pane's row, the time scale's width.
const plotRect = (page) => page.evaluate((fn) => {
  const { price } = eval(fn) // eslint-disable-line no-eval
  const r = price.chart.panes()[0].getHTMLElement().getBoundingClientRect()
  return { x: r.x, y: r.y, w: price.chart.timeScale().width(), h: r.height }
}, CHARTS_FN)
const panelRect = (page) => page.getByTestId('flow-v3-box-popup').evaluate((el) => {
  const r = el.getBoundingClientRect()
  return { x: r.x, y: r.y, w: r.width, h: r.height, dragged: el.dataset.dragged }
})

// The panel as text: every section expanded.
async function panelText(page) {
  const panel = page.getByTestId('flow-v3-box-popup')
  const sums = panel.getByTestId('flow-v3-box-panel-summary')
  for (let i = 0; i < await sums.count(); i += 1) {
    if ((await sums.nth(i).getAttribute('aria-expanded')) !== 'true') await sums.nth(i).click()
  }
  return panel.evaluate((el) => {
    const secs = [...el.querySelectorAll('[data-testid=flow-v3-box-panel-current],[data-testid=flow-v3-box-panel-frozen]')]
    return {
      dropped: el.querySelector('[data-testid=flow-v3-box-panel-dropped]')?.innerText ?? null,
      sections: secs.map((s) => ({
        current: s.dataset.testid === 'flow-v3-box-panel-current',
        first: s.firstElementChild?.dataset?.testid ?? null,
        summary: s.querySelector('[data-testid=flow-v3-box-panel-summary]').innerText.replace(/^[▸▾]/, ''),
        // CR-DS: the stats block's rows (expanded frozen sections now show a verdict table above them)
        lines: [...s.querySelectorAll('[data-testid$=-stats] > div')].map((d) => d.innerText),
        badges: [...s.querySelectorAll('[data-testid^=flow-v3-box-pct-]')].map((b) => ({ kind: b.dataset.testid.slice(16), p: Number(b.dataset.p), tier: b.dataset.tier, text: b.innerText, title: b.title, weight: getComputedStyle(b).fontWeight, color: getComputedStyle(b).color })),
        verdict: s.querySelector('[data-testid=flow-v3-box-verdict]') ? {
          kind: s.querySelector('[data-testid=flow-v3-box-verdict]').dataset.kind,
          pill: s.querySelector('[data-testid=flow-v3-box-verdict-pill]').innerText,
          // CR-DS: the evidence line became the table (kept as data-evidence); the hover moved to the pill
          evidence: s.querySelector('[data-testid=flow-v3-box-verdict]').dataset.evidence,
          title: s.querySelector('[data-testid=flow-v3-box-verdict-pill]').title,
        } : null,
      })),
    }
  })
}

// Same box objects + payload segments → the panel's percentiles, recomputed here from the JSON (hand check).
async function cellsSegments(page) {
  return page.evaluate((fn) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    return null
  }, CHARTS_FN)
}

test('items 1–3: 2026-10-30 anchor 11:57 — no box text, panel bottom-left, one section per box (current last, highlighted, verdict first); numbers = origin/main', async ({ browser }) => {
  test.setTimeout(300_000)
  const ctxA = await browser.newContext()
  const ctxB = await browser.newContext()
  const a = await ctxA.newPage()
  const b = await ctxB.newPage()
  await open(a, D30, { storage: EDGES })
  await open(b, D30, { base: BASELINE, storage: EDGES })
  const anchor = shifted(`${D30}T11:57`)
  await anchorAt(a, anchor, anchor - 30 * 60, shifted(`${D30}T13:00`))
  await anchorAt(b, anchor, anchor - 30 * 60, shifted(`${D30}T13:00`))
  const da = await drawn(a)
  const db = await drawn(b)
  const res = { boxes: da.boxes.length, frozen: da.boxes.filter((x) => x.frozen).length }
  expect(da.boxes.length, '2+ boxes after 11:57').toBeGreaterThanOrEqual(2)
  // item 1: no label on any drawn box. CR-DS: the baseline is now origin/main after CR-DQ, which has no labels either, so
  // the old "origin/main had one per box" check is stale; check that both sides show the box panel (popup) instead.
  expect(da.boxes.every((x) => x.label == null)).toBeTruthy()
  expect(db.boxes.every((x) => x.label == null)).toBeTruthy()
  await expect(a.getByTestId('flow-v3-box-popup')).toBeVisible()
  await expect(b.getByTestId('flow-v3-box-popup')).toBeVisible()
  await expect(a.getByTestId('flow-v3-box-tooltip')).toHaveCount(0)
  // one source: the branch's box objects equal origin/main's (the branch adds lastClose only)
  // CR-DS item 5: frozen boxes (also the current box's nested `frozen` list) gain lastClose too
  const noLast = ({ lastClose, ...rest }) => rest // eslint-disable-line no-unused-vars
  const clean = (x) => { const rest = noLast(x.box); if (Array.isArray(rest.frozen)) rest.frozen = rest.frozen.map(noLast); return { ...x, label: null, box: rest } }
  expect(da.boxes.map(clean)).toEqual(db.boxes.map(clean))
  // item 4 data: the shade is fed the same boxes
  expect(da.shade).toEqual(da.boxes.map(({ from, to, top, bottom, frozen }) => ({ from, to, top, bottom, frozen })))

  // item 3: default bottom-left inside the plot
  const plot = await plotRect(a)
  const pr = await panelRect(a)
  res.plot = plot
  res.panel = pr
  expect(Math.abs(pr.x - (plot.x + 4))).toBeLessThanOrEqual(2)
  expect(Math.abs(pr.y + pr.h - (plot.y + plot.h - 4))).toBeLessThanOrEqual(2)
  expect(pr.h).toBeLessThanOrEqual(plot.h * 0.5 + 2)
  await shot(a, 'panel-1030-1157-collapsed.png')

  // item 2: sections = drawn boxes, oldest first, current last + highlighted, verdict first in it
  const t = await panelText(a)
  res.panelText = t
  expect(t.sections.length).toBe(da.boxes.length)
  expect(t.sections.map((s) => s.current)).toEqual(da.boxes.map((x) => !x.frozen))
  const cur = t.sections[t.sections.length - 1]
  expect(cur.first).toBe('flow-v3-box-verdict')
  expect(cur.verdict).toBeTruthy()
  // numbers: the current section's lines = origin/main's popup lines; frozen sections carry origin/main's label numbers
  const mainPopup = await b.getByTestId('flow-v3-box-popup').evaluate((el) => ({
    head: el.querySelector('[data-testid=flow-v3-box-popup-head]').innerText,
    top: el.querySelector('[data-testid=flow-v3-box-popup-top]').innerText,
    bottom: el.querySelector('[data-testid=flow-v3-box-popup-bottom]').innerText,
  }))
  res.mainPopup = mainPopup
  expect(noBadges(cur.lines[0])).toBe(noBadges(mainPopup.head)) // CR-DS: origin/main (post CR-DQ) has badges too
  expect(cur.lines[1]).toBe(mainPopup.top)
  expect(cur.lines[2]).toBe(mainPopup.bottom)
  for (let k = 0; k < db.boxes.length; k += 1) {
    const m = db.boxes[k]
    const s = t.sections[k]
    // CR-DS: origin/main (post CR-DQ) draws no labels; the label numbers from the same function, on its readout
    const [d, pts, mins, cpp] = boxLabelText(m.box.readout).split(' · ')
    for (const piece of [d, pts, mins, cpp]) expect(noBadges(s.lines[0])).toContain(piece)
    expect(s.lines[1]).toContain(`Top ${m.top.toFixed(2)} · buyers ${m.box.top.contracts == null ? 'exact data only' : kfmt(m.box.top.contracts)}`)
    expect(s.lines[2]).toContain(`Bottom ${m.bottom.toFixed(2)} · sellers ${m.box.bottom.contracts == null ? 'exact data only' : kfmt(m.box.bottom.contracts)}`)
    // CR-DS item 5: the frozen row is "hh:mm–hh:mm · broke ↓ hh:mm · share pNN · c/pt pNN"
    if (m.frozen) expect(s.summary).toContain(`${pt(m.box.from)}–${pt(m.box.breakAt)} · broke ${m.box.dir === 'up' ? '↑' : '↓'} ${pt(m.box.breakAt)} · share `)
  }
  await shot(a, 'panel-1030-1157-expanded.png')
  log('items-1-3.json', res)
  await ctxA.close()
  await ctxB.close()
})

test('item 3: drag by the header, kept across a reload and a new anchor; double-click returns bottom-left; × until next anchor; Esc clear removes', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D30, { storage: EDGES })
  const anchor = shifted(`${D30}T11:57`)
  await anchorAt(page, anchor, anchor - 30 * 60, shifted(`${D30}T13:00`))
  const p0 = await panelRect(page)
  const hdr = page.getByTestId('flow-v3-box-panel-header')
  const hb = await hdr.boundingBox()
  await page.mouse.move(hb.x + 40, hb.y + hb.height / 2)
  await page.mouse.down()
  await page.mouse.move(hb.x + 240, hb.y + hb.height / 2 - 150, { steps: 8 })
  await page.mouse.up()
  const p1 = await panelRect(page)
  const stored = await page.evaluate(() => window.localStorage.getItem('ib-react-box-panel-pos'))
  const res = { p0, p1, stored }
  expect(Math.round(p1.x - p0.x)).toBe(200)
  expect(Math.round(p0.y - p1.y)).toBe(150)
  expect(stored).toBeTruthy()
  // reload: same place (the anchor is stored per date by App)
  await page.reload()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await settle(page)
  let p2 = null
  if (await page.getByTestId('flow-v3-box-popup').count()) p2 = await panelRect(page)
  else {
    await anchorAt(page, anchor, anchor - 30 * 60, shifted(`${D30}T13:00`))
    p2 = await panelRect(page)
  }
  res.p2 = p2
  expect(Math.abs(p2.x - p1.x)).toBeLessThanOrEqual(1)
  expect(Math.abs(p2.y + p2.h - (p1.y + p1.h))).toBeLessThanOrEqual(1)
  // a new anchor reuses the place
  await anchorAt(page, shifted(`${D30}T10:00`), shifted(`${D30}T09:30`), shifted(`${D30}T13:00`))
  const p3 = await panelRect(page)
  res.p3 = p3
  expect(Math.abs(p3.x - p1.x)).toBeLessThanOrEqual(1)
  await shot(page, 'panel-dragged.png')
  // double-click the header: back to bottom-left, key removed
  await page.getByTestId('flow-v3-box-panel-header').dblclick()
  await page.waitForTimeout(300)
  const p4 = await panelRect(page)
  const plot = await plotRect(page)
  res.p4 = p4
  expect(Math.abs(p4.x - (plot.x + 4))).toBeLessThanOrEqual(2)
  expect(Math.abs(p4.y + p4.h - (plot.y + plot.h - 4))).toBeLessThanOrEqual(2)
  expect(await page.evaluate(() => window.localStorage.getItem('ib-react-box-panel-pos'))).toBeNull()
  // resize: stays inside the plot
  await page.setViewportSize({ width: 900, height: 700 })
  await page.waitForTimeout(800)
  const p5 = await panelRect(page)
  const plot5 = await plotRect(page)
  res.p5 = { p5, plot5 }
  expect(p5.x).toBeGreaterThanOrEqual(plot5.x - 1)
  expect(p5.y + p5.h).toBeLessThanOrEqual(plot5.y + plot5.h + 1)
  await page.setViewportSize({ width: 1440, height: 900 })
  // × hides until the next anchor
  await page.getByTestId('flow-v3-box-popup-close').click()
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  // (the same anchor picked again is not a new anchor, as origin/main's popup)
  await anchorAt(page, shifted(`${D30}T11:30`), anchor - 60 * 60, shifted(`${D30}T13:00`))
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(1)
  // clear removes
  await page.getByTestId('flow-v3-anchor-clear').click()
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  log('item-3-drag.json', res)
})

// Composite of the price chart's canvases.
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
  return { out, ctx, dpr, rect }
})`

test('item 4: boxes shaded (current stronger than frozen), candles on top; the set-end window stays slate and full height', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D30, { storage: EDGES })
  const anchor = shifted(`${D30}T11:57`)
  await anchorAt(page, anchor, anchor - 10 * 60, shifted(`${D30}T13:00`))
  await page.getByTestId('flow-v3-box-popup-close').click()
  // pixel deltas with the shade on vs its data removed, at sample points inside each box away from candles
  const sample = () => page.evaluate(({ fn, comp }) => {
    const { price, priceEl } = eval(fn) // eslint-disable-line no-eval
    const { ctx, dpr } = eval(comp)('.tv-lightweight-charts') // eslint-disable-line no-eval
    const ts = price.chart.timeScale()
    const pr = priceEl.getBoundingClientRect()
    const pane = price.chart.panes()[0].getHTMLElement().getBoundingClientRect()
    const ox = pane.x - pr.x
    const oy = pane.y - pr.y
    const boxes = price.prims.edges._data.boxes
    const bars = price.series.data().filter((x) => x.close != null)
    const out = []
    for (const b of boxes) {
      // the box's pixel column with the most free space between its edges and the candle
      let best = null
      for (const bar of bars) {
        if (bar.time < b.from || bar.time > b.to) continue
        const x = ts.timeToCoordinate(bar.time)
        const yt = price.series.priceToCoordinate(b.top)
        const yb = price.series.priceToCoordinate(b.bottom)
        const yh = price.series.priceToCoordinate(bar.high)
        const yl = price.series.priceToCoordinate(bar.low)
        const gapTop = yh - yt
        const gapBot = yb - yl
        const g = Math.max(gapTop, gapBot)
        if (!best || g > best.g) best = { g, x, y: gapTop > gapBot ? (yt + yh) / 2 : (yl + yb) / 2, xc: x, yc: (price.series.priceToCoordinate(bar.open) + price.series.priceToCoordinate(bar.close)) / 2, body: Math.abs(price.series.priceToCoordinate(bar.open) - price.series.priceToCoordinate(bar.close)) }
      }
      if (!best) continue
      const px = (x, y) => Array.from(ctx.getImageData(Math.round((ox + x) * dpr), Math.round((oy + y) * dpr), 1, 1).data)
      out.push({ frozen: b.frozen, gap: best.g, inBox: px(best.x + 3, best.y), candle: best.body > 3 ? px(best.xc, best.yc) : null })
    }
    return out
  }, { fn: CHARTS_FN, comp: COMPOSITE_FN })
  await frame(page, anchor - 10 * 60, shifted(`${D30}T13:00`))
  const on = await sample()
  const saved = await page.evaluate((fn) => { const { price } = eval(fn); const d = price.prims.boxShade._data; price.prims.boxShade.setData(null); window.__crdqShade = d; return true }, CHARTS_FN) // eslint-disable-line no-eval
  await page.waitForTimeout(400)
  const off = await sample()
  await page.evaluate((fn) => { const { price } = eval(fn); price.prims.boxShade.setData(window.__crdqShade) }, CHARTS_FN) // eslint-disable-line no-eval
  await page.waitForTimeout(400)
  const res = { saved, boxes: on.map((x, i) => ({ frozen: x.frozen, gapPx: Math.round(x.gap), on: x.inBox, off: off[i].inBox, candleOn: x.candle, candleOff: off[i].candle })) }
  // shade adds magenta: red and blue rise more than green; current more than frozen; candle body pixels unchanged
  const lift = (r) => (r.on[0] - r.off[0]) + (r.on[2] - r.off[2]) - 2 * (r.on[1] - r.off[1])
  for (const r of res.boxes) r.magentaLift = lift(r)
  const curL = res.boxes.filter((r) => !r.frozen && r.gapPx > 6).map((r) => r.magentaLift)
  const frzL = res.boxes.filter((r) => r.frozen && r.gapPx > 6).map((r) => r.magentaLift)
  res.curL = curL
  res.frzL = frzL
  expect(curL.length).toBeGreaterThan(0)
  expect(Math.min(...curL)).toBeGreaterThan(0)
  if (frzL.length) {
    expect(Math.min(...frzL)).toBeGreaterThan(0)
    expect(Math.max(...frzL)).toBeLessThan(Math.min(...curL))
  }
  for (const r of res.boxes) if (r.candleOn) expect(r.candleOn).toEqual(r.candleOff)
  await shot(page, 'shading-1030-1157.png')
  // set end inside the session: the window shade is slate (r ≈ g ≈ b lift) above the boxes, the box fill magenta
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D30}T12:40`), anchor - 10 * 60, shifted(`${D30}T13:00`))
  await frame(page, anchor - 10 * 60, shifted(`${D30}T13:00`))
  const win = await page.evaluate(({ fn, comp }) => {
    const { price, priceEl } = eval(fn) // eslint-disable-line no-eval
    const { ctx, dpr } = eval(comp)('.tv-lightweight-charts') // eslint-disable-line no-eval
    const pr = priceEl.getBoundingClientRect()
    const pane = price.chart.panes()[0].getHTMLElement().getBoundingClientRect()
    const x = price.chart.timeScale().timeToCoordinate(price.prims.window._data.from) + 4
    const y = 6
    return Array.from(ctx.getImageData(Math.round((pane.x - pr.x + x) * dpr), Math.round((pane.y - pr.y + y) * dpr), 1, 1).data)
  }, { fn: CHARTS_FN, comp: COMPOSITE_FN })
  await page.evaluate((fn) => { const { price } = eval(fn); window.__crdqWin = price.prims.window._data; price.prims.window.setData(null) }, CHARTS_FN) // eslint-disable-line no-eval
  await page.waitForTimeout(400)
  const winOff = await page.evaluate(({ fn, comp }) => {
    const { price, priceEl } = eval(fn) // eslint-disable-line no-eval
    const { ctx, dpr } = eval(comp)('.tv-lightweight-charts') // eslint-disable-line no-eval
    const pr = priceEl.getBoundingClientRect()
    const pane = price.chart.panes()[0].getHTMLElement().getBoundingClientRect()
    const x = price.chart.timeScale().timeToCoordinate(window.__crdqWin.from) + 4
    return Array.from(ctx.getImageData(Math.round((pane.x - pr.x + x) * dpr), Math.round((pane.y - pr.y + 6) * dpr), 1, 1).data)
  }, { fn: CHARTS_FN, comp: COMPOSITE_FN })
  await page.evaluate((fn) => { const { price } = eval(fn); price.prims.window.setData(window.__crdqWin) }, CHARTS_FN) // eslint-disable-line no-eval
  res.window = { on: win, off: winOff }
  const dr = win[0] - winOff[0]
  const dg = win[1] - winOff[1]
  const dbb = win[2] - winOff[2]
  expect(dr + dg + dbb).toBeGreaterThan(0)
  expect(Math.abs(dr - dg)).toBeLessThanOrEqual(3) // slate: no magenta cast
  await shot(page, 'shading-set-end-1240.png')
  log('item-4-shading.json', res)
})

test('item 2: more than 4 boxes → "n older boxes not shown" (2026-10-29, anchor 06:31)', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D29, { storage: EDGES })
  await anchorAt(page, shifted(`${D29}T06:31`), shifted(`${D29}T06:20`), shifted(`${D29}T13:00`))
  const d = await drawn(page)
  const dropped = d.boxes[d.boxes.length - 1].box.dropped
  const t = await panelText(page)
  log('item-2-dropped.json', { dropped, sections: t.sections.length, droppedText: t.dropped })
  expect(dropped).toBeGreaterThan(0)
  expect(t.sections.length).toBe(5)
  expect(t.dropped).toBe(`${dropped} older box${dropped === 1 ? '' : 'es'} not shown`)
  await shot(page, 'panel-1029-0631-dropped.png')
})

// Percentiles: every badge = boxPercentiles of its box readout against the committed JSON; tiers by p.
async function checkBadges(page, res) {
  const d = await drawn(page)
  const segs = await page.evaluate(() => null)
  const t = await panelText(page)
  const out = []
  for (let k = 0; k < d.boxes.length; k += 1) {
    const b = d.boxes[k].box
    const s = t.sections[k]
    const ro = b.readout
    out.push({ from: pt(b.from), minutes: ro?.minutes, delta: ro?.delta, vol: ro?.vol, dPrice: ro?.dPrice, seg: [ro?.fromSeg, ro?.toSeg], badges: s.badges.map(({ kind, p, tier, text, weight }) => ({ kind, p, tier, text, weight })) })
    for (const bd of s.badges) {
      expect(bd.tier).toBe(pctTier(bd.p))
      if (bd.tier === 'normal') expect(Number(bd.weight)).toBeLessThan(600)
      else expect(Number(bd.weight)).toBeGreaterThanOrEqual(600)
      if (bd.tier === 'strong') expect(Number(bd.weight)).toBeGreaterThanOrEqual(800)
    }
  }
  res.boxes = out
  void segs
  return { d, t }
}

test('item 5: badges = the JSON baseline (Ironbeam on 2026-10-30), tiers by brightness; three train windows (Databento) match the hand check', async ({ page }) => {
  test.setTimeout(600_000)
  const res = {}
  await open(page, D30, { storage: EDGES })
  await anchorAt(page, shifted(`${D30}T06:31`), shifted(`${D30}T06:20`), shifted(`${D30}T13:00`))
  const { d } = await checkBadges(page, res)
  // recompute each from its readout (source: Ironbeam RTH on 10-30)
  const segs = [{ name: 'eth', source: 'ironbeam' }, { name: 'rth', source: 'ironbeam' }]
  const t = await panelText(page)
  for (let k = 0; k < d.boxes.length; k += 1) {
    const pc = boxPercentiles(d.boxes[k].box.readout, ptMinute(d.boxes[k].box.from), segs, BASE_JSON)
    const got = Object.fromEntries(t.sections[k].badges.map((x) => [x.kind, x.p]))
    if (!pc) { expect(t.sections[k].badges.filter((x) => !t.sections[k].current || true).length).toBe(0); continue }
    expect(Math.abs(got.share - pc.share.p)).toBeLessThan(0.01)
    expect(Math.abs(got.cpp - pc.cpp.p)).toBeLessThan(0.01)
  }
  const anyTitle = t.sections.flatMap((s) => s.badges)[0]?.title || ''
  res.title = anyTitle
  expect(anyTitle).toMatch(/Ironbeam baseline \(99 train sessions 2026-01-02 → 2026-06-05/)
  expect(anyTitle).toMatch(/median ~9 percentile points/)
  res.tiers = t.sections.flatMap((s) => s.badges.map((x) => x.tier))
  await shot(page, 'badges-1030-0631.png')
  // Step 0's three train windows as anchor + set end (Databento on the chart for these dates)
  res.examples = []
  for (const [day, a, e, hand] of [
    ['2026-02-10', '07:05', '07:30', { delta: -1718, vol: 92876, share: 58.412, cpp: 69.378 }],
    ['2026-04-15', '09:12', '09:24', { delta: 725, vol: 18923, share: 57.741, cpp: 30.327 }],
    ['2026-05-20', '10:40', '11:35', { delta: -154, vol: 68938, share: 7.688, cpp: 88.987 }],
  ]) {
    await page.goto('about:blank')
    await open(page, day, { storage: EDGES })
    await anchorAt(page, shifted(`${day}T${a}`), shifted(`${day}T${a}`) - 20 * 60, shifted(`${day}T${e}`) + 40 * 60)
    await pickAt(page, 'flow-v3-end-pick', shifted(`${day}T${e}`), shifted(`${day}T${a}`) - 20 * 60, shifted(`${day}T${e}`) + 40 * 60)
    const dd = await drawn(page)
    const tt = await panelText(page)
    const one = dd.boxes.length === 1
    const ro = dd.boxes[dd.boxes.length - 1].box.readout
    const got = Object.fromEntries(tt.sections[tt.sections.length - 1].badges.map((x) => [x.kind, x.p]))
    const row = { day, window: `${a}–${e}`, boxes: dd.boxes.length, readout: { delta: ro.delta, vol: ro.vol, dPrice: ro.dPrice, minutes: ro.minutes }, hand, panel: got }
    if (one) {
      expect(ro.delta).toBe(hand.delta)
      expect(ro.vol).toBe(hand.vol)
      expect(Math.abs(got.share - hand.share)).toBeLessThan(0.01)
      expect(Math.abs(got.cpp - hand.cpp)).toBeLessThan(0.01)
      row.match = true
    } else row.match = 'box froze inside the window (more than one box); compared in the unit test instead'
    res.examples.push(row)
  }
  log('item-5-badges.json', res)
})

// Verdict inputs as the page holds them, for the hand check.
async function verdictInputs(page) {
  const d = await drawn(page)
  const t = await panelText(page)
  const cur = d.boxes.find((x) => !x.frozen)
  const sec = t.sections.find((s) => s.current)
  return {
    box: { from: pt(cur.box.from), to: pt(cur.box.to), top: cur.top, bottom: cur.bottom, high: cur.box.high, low: cur.box.low, lastClose: cur.box.lastClose,
      readout: cur.box.readout && { delta: cur.box.readout.delta, vol: cur.box.readout.vol, dPrice: cur.box.readout.dPrice, minutes: cur.box.readout.minutes, share: cur.box.readout.vol ? cur.box.readout.delta / cur.box.readout.vol : null } },
    badges: sec.badges.map(({ kind, p, tier }) => ({ kind, p, tier })),
    levels: (d.levels || []).map((l) => ({ kind: l.kind, price: l.price, start: pt(l.start), broken: l.broken, end: pt(l.end) })),
    verdict: sec.verdict,
    summary: sec.summary,
  }
}

test('item 7: 2026-10-28 (FOMC) shows NO TRADES', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D28, { storage: EDGES })
  await anchorAt(page, shifted(`${D28}T07:00`), shifted(`${D28}T06:40`), shifted(`${D28}T13:00`))
  const v = await verdictInputs(page)
  log('item-7-fomc.json', v)
  expect(v.verdict.kind).toBe('no-trades')
  expect(v.verdict.pill).toBe('NO TRADES') // CR-DS item 1: the pill is the label's head; the reason sits under it
  expect(v.verdict.title).toMatch(/^v1 rules engine .*not a tested edge\.$/) // CR-DU item 2: the verdict is the v1 engine's
  await shot(page, 'verdict-1028-fomc.png')
})

// Scan: replay 2026-10-29 / 10-30 to a time, anchor 30 min earlier, record the verdict and its inputs (hand check in
// the spec's smoke notes; nothing tuned).
test('item 7: verdict scan on 2026-10-29 / 10-30 (replays; inputs logged for the hand check)', async ({ page }) => {
  test.setTimeout(1_800_000)
  const rows = []
  for (const day of [D29, D30]) {
    for (const until of Array.from({ length: 36 }, (_, k) => { const m = 7 * 60 + k * 10; return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}` })) {
      const [h, m] = until.split(':').map(Number)
      const a = `${String(h - (m < 30 ? 1 : 0)).padStart(2, '0')}:${String((m + 30) % 60).padStart(2, '0')}`
      await page.goto('about:blank')
      await page.unrouteAll({ behavior: 'ignoreErrors' })
      await open(page, day, { storage: EDGES_LEVELS, until: `${day}T${until}` })
      await anchorAt(page, shifted(`${day}T${a}`), shifted(`${day}T${a}`) - 20 * 60, shifted(`${day}T${until}`))
      if (!(await page.getByTestId('flow-v3-box-popup').count())) { rows.push({ day, until, anchor: a, verdict: null }); continue }
      rows.push({ day, until, anchor: a, ...(await verdictInputs(page)) })
      log('item-7-scan.json', rows)
    }
  }
  log('item-7-scan.json', rows)
  const kinds = rows.map((r) => r.verdict?.kind).filter(Boolean)
  expect(kinds.length).toBeGreaterThan(0)
})

// Ryan's smoke addition: his 2026-10-30 entries, report only (thresholds untouched).
test('smoke addition: Ryan\'s 2026-10-30 entries — verdict and evidence at each (report only)', async ({ page }) => {
  test.setTimeout(900_000)
  const rows = []
  for (const [entry, anchor, session] of [['06:11:46', '05:00', 'FULL'], ['07:06:47', '06:58', 'RTH'], ['07:49:17', '07:40', 'RTH'], ['09:22:50', '09:15', 'RTH']]) {
    await page.goto('about:blank')
    await page.unrouteAll({ behavior: 'ignoreErrors' })
    await open(page, D30, { storage: EDGES_LEVELS, session, until: `${D30}T${entry}` })
    await anchorAt(page, shifted(`${D30}T${anchor}`), shifted(`${D30}T${anchor}`) - 20 * 60, shifted(`${D30}T${entry.slice(0, 5)}`) + 5 * 60)
    const has = await page.getByTestId('flow-v3-box-popup').count()
    const row = { entry, anchor, session, panel: has ? await panelText(page) : null, inputs: has ? await verdictInputs(page) : null }
    rows.push(row)
    await shot(page, `ryan-entry-${entry.replace(/:/g, '')}.png`)
    log('smoke-ryan-entries.json', rows)
  }
  expect(rows.length).toBe(4)
})

// CR-DQ amendment: the three hand-checked verdicts re-run (report as they come out; nothing tuned).
test('amendment: the three hand-checked replays (10-30 09:30 / 08:30, 10-29 07:20)', async ({ page }) => {
  test.setTimeout(600_000)
  const rows = []
  for (const [day, until, a] of [[D30, '09:30', '09:00'], [D30, '08:30', '08:00'], [D29, '07:20', '06:50']]) {
    await page.goto('about:blank')
    await page.unrouteAll({ behavior: 'ignoreErrors' })
    await open(page, day, { storage: EDGES_LEVELS, until: `${day}T${until}` })
    await anchorAt(page, shifted(`${day}T${a}`), shifted(`${day}T${a}`) - 20 * 60, shifted(`${day}T${until}`))
    rows.push({ day, until, anchor: a, ...(await verdictInputs(page)) })
    await shot(page, `amend-hand-${day.slice(5)}-${until.replace(':', '')}.png`)
    log('amendment-hand-checks.json', rows)
  }
  expect(rows.length).toBe(3)
})
