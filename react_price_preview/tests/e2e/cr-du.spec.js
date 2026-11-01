// CR-DU — v1 rules engine in the box panel, the v1-trades overlay and its blocked dates (specs/CR-DU-v1-rules-engine.md
// item 6). Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_du_smoke_backend.py        (read-only DB, capped pool, :8068)
//   frontend   npm --prefix react_price_preview run dev -- --port 5197 --strictPort
// then:       CRBV_BASE_URL=http://localhost:5197 CRDU_API_BASE=http://127.0.0.1:8068 npx playwright test cr-du
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { makeShifter } from '../../src/flowV3.js'

const API_BASE = process.env.CRDU_API_BASE || 'http://127.0.0.1:8068'
const HERE = path.dirname(fileURLToPath(import.meta.url))
const LOGS = path.resolve(HERE, '../../../scripts/logs/cr-du/smoke')
const ASSETS = path.resolve(HERE, '../../../specs/assets/cr-du')
const BACKTEST = path.resolve(HERE, '../../../scripts/logs/cr-du/backtest_results.json')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height',
  'ib-react-flow-v3-slots', 'ib-react-price-zoom', 'ib-react-box-panel-pos']
const layers = (o) => ({ 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: false, profile: false, heat: false, marks: false, ...o }) })
const EDGES = layers({ edges: true })
const D30 = '2026-10-30'
const TRAIN = '2026-02-03' // a train session without a set-2 card (its trades are in the backtest's per-trade rows)
const SET2 = '2026-02-04' // a session with a set-2 card
const HOLDOUT = '2026-07-10'

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
    await page.route('**/api/ironbeam/v1-engine/inputs?*', cut)
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

const verdict = async (page) => {
  const v = page.getByTestId('flow-v3-box-verdict')
  await v.waitFor()
  return {
    kind: await v.getAttribute('data-kind'),
    pill: await page.getByTestId('flow-v3-box-verdict-pill').innerText(),
    reason: (await page.getByTestId('flow-v3-box-verdict-reason').count()) ? await page.getByTestId('flow-v3-box-verdict-reason').innerText() : '',
    tag: (await page.getByTestId('flow-v3-box-engine-tag').count()) ? await page.getByTestId('flow-v3-box-engine-tag').innerText() : '',
    engine: (await page.getByTestId('flow-v3-box-engine-line').count()) ? await page.getByTestId('flow-v3-box-engine-line').innerText() : '',
    rows: await page.getByTestId('flow-v3-box-evidence-row').evaluateAll((els) => els.map((e) => [e.dataset.key, e.dataset.tone, e.innerText.replace(/\s+/g, ' ')])),
  }
}

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

test('1. 10-30 #5 box (anchor 07:40, replay to 07:49): the engine verdict at box min 8 vs "box still forming" at 10', async ({ page }) => {
  const a = shifted(`${D30}T07:40`)
  await open(page, D30, { storage: EDGES, until: `${D30}T07:49` })
  await anchorAt(page, a, a - 30 * 60, shifted(`${D30}T08:00`))
  const v8 = await verdict(page)
  await shot(page, '1-min8.png')
  expect(v8.reason).not.toMatch(/box still forming/)
  expect(v8.tag).toBe('engine v1.4 — untested') // beside the pill, in the panel's muted secondary text
  expect(v8.rows.find((r) => r[0] === 'formed')[2]).toMatch(/formed/)
  expect(v8.rows.find((r) => r[0] === 'formed')[2]).toMatch(/min 8/)
  // ⚙ box minimum 10 (stored, after the CR-DU migration): box still forming
  const page2 = await page.context().newPage()
  await page2.addInitScript(() => {
    try {
      const k = 'ib-react-flow-v3-settings'
      const s = JSON.parse(window.localStorage.getItem(k) || '{}')
      window.localStorage.setItem(k, JSON.stringify({ ...s, boxRulesV: 2, boxRules: { ...(s.boxRules || {}), minAgeMin: 10 } }))
    } catch (e) { /* ignore */ }
  })
  await open(page2, D30, { until: `${D30}T07:49`, keepStorage: true })
  await anchorAt(page2, a, a - 30 * 60, shifted(`${D30}T08:00`))
  const v10 = await verdict(page2)
  await shot(page2, '1-min10.png')
  expect(v10.reason).toMatch(/rule 6: box still forming/)
  log('1-verdicts.json', { min8: v8, min10: v10 })
})

async function overlay(page) {
  const t = page.getByTestId('flow-v3-v1-trades-toggle')
  await expect(t).toBeEnabled({ timeout: 60_000 })
  await t.click()
  await page.waitForFunction(() => window.__crDuV1 && window.__crDuV1(), null, { timeout: 120_000 })
  await page.waitForTimeout(1500)
  return page.evaluate(() => window.__crDuV1())
}

test('2. v1 overlay on a train day equals the backtest', async ({ page }) => {
  await open(page, TRAIN)
  const o = await overlay(page)
  const bt = JSON.parse(fs.readFileSync(BACKTEST, 'utf8')).trades.primary.filter((x) => x.date === TRAIN)
  const key = (x) => [x.side, x.entryT, x.entry, x.stop, x.target, x.exitT, x.exit, x.reason, x.pts]
  expect(o.trades.map(key)).toEqual(bt.map(key))
  expect(o.hits.length).toBe(2 * o.trades.length + (o.open ? 1 : 0))
  // hover the first entry: that minute's verdict
  const h = o.hits[0]
  const plot = await page.evaluate((fn) => { const r = eval(fn).priceEl.getBoundingClientRect(); return { x: r.x, y: r.y } }, CHARTS_FN) // eslint-disable-line no-eval
  await page.mouse.move(plot.x + h.x, plot.y + h.y - 30)
  await page.getByTestId('flow-v3-v1-hover').waitFor({ timeout: 10_000 })
  const hover = await page.getByTestId('flow-v3-v1-hover').innerText()
  await shot(page, '2-train-overlay.png')
  log('2-train.json', { overlay: o, backtest: bt.length, hover })
})

test('3. v1 overlay on 10-30 (forward, Ironbeam)', async ({ page }) => {
  await open(page, D30)
  const o = await overlay(page)
  await shot(page, '3-1030-overlay.png')
  log('3-1030.json', { trades: o.trades.map((x) => ({ side: x.side, entryT: x.entryT, entry: x.entry, exitT: x.exitT, exit: x.exit, reason: x.reason, pts: x.pts })), open: o.open, minutes: o.minutes })
  expect(o.minutes).toBeGreaterThan(300)
})

test('4. blocked dates: holdout and set 2 in progress', async ({ page }) => {
  const out = {}
  for (const [d, want] of [[HOLDOUT, 'holdout date: not shown'], [SET2, 'set 2 in progress']]) {
    await open(page, d)
    const t = page.getByTestId('flow-v3-v1-trades-toggle')
    await expect(t).toHaveAttribute('data-blocked', want, { timeout: 60_000 })
    await expect(t).toBeDisabled()
    out[d] = { blocked: await t.getAttribute('data-blocked'), text: await t.innerText() }
    if (d === HOLDOUT) await shot(page, '4-holdout.png')
  }
  log('4-blocked.json', out)
})

test('5. Amendment 5: 2026-04-15 box 09:18–09:29 (the screenshot) reads 11 min · 2.50 pts, not 0.00', async ({ page }) => {
  const D = '2026-04-15'
  const a = shifted(`${D}T09:18`)
  await open(page, D, { storage: EDGES })
  await anchorAt(page, a, a - 30 * 60, shifted(`${D}T09:45`))
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D}T09:29`), a - 30 * 60, shifted(`${D}T09:45`))
  const v = await verdict(page)
  await shot(page, '5-0415-height.png')
  log('5-0415.json', v)
  expect(v.rows.find((r) => r[0] === 'formed')[2]).toMatch(/11 min · 2\.50 pts \(min 8 · 2\) · tested 1\.50 below/)
  expect(v.rows.find((r) => r[0] === 'location')[2]).toMatch(/1\.50 below the bottom/)
  expect(v.rows.find((r) => r[0] === 'absorption')[2]).toMatch(/7040\.00 \(09:23\)/) // PT, not the epoch
})
