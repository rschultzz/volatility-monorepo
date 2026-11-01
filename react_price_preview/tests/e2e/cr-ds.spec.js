// CR-DS — box panel as a colour-coded table: verdict pill, evidence rows, read history, frozen rows
// (specs/CR-DS-panel-table.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    CRDP_PORT=8068 apps/web/.venv/bin/python -u scripts/cr_dp_smoke_backend.py   (read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5195 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5194
// then:       CRBV_BASE_URL=http://localhost:5195 CRDQ_API_BASE=http://127.0.0.1:8068 CRDQ_BASELINE_URL=http://localhost:5194 \
//             npx playwright test cr-ds   (from react_price_preview/)
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { boxPercentiles, kfmt, makeShifter, pctTier } from '../../src/flowV3.js'

const API_BASE = process.env.CRDQ_API_BASE || 'http://127.0.0.1:8068'
const BASELINE = process.env.CRDQ_BASELINE_URL || 'http://localhost:5194'
const HERE = path.dirname(fileURLToPath(import.meta.url))
const LOGS = path.resolve(HERE, '../../../scripts/logs/cr-ds/smoke')
const ASSETS = path.resolve(HERE, '../../../specs/assets/cr-ds')
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
        lines: [...s.querySelectorAll('[data-testid=flow-v3-box-panel-summary] + div > div > div')].map((d) => d.innerText),
        badges: [...s.querySelectorAll('[data-testid^=flow-v3-box-pct-]')].map((b) => ({ kind: b.dataset.testid.slice(16), p: Number(b.dataset.p), tier: b.dataset.tier, text: b.innerText, title: b.title, weight: getComputedStyle(b).fontWeight, color: getComputedStyle(b).color })),
        verdict: s.querySelector('[data-testid=flow-v3-box-verdict]') ? {
          kind: s.querySelector('[data-testid=flow-v3-box-verdict]').dataset.kind,
          pill: s.querySelector('[data-testid=flow-v3-box-verdict-pill]').innerText,
          evidence: s.querySelector('[data-testid=flow-v3-box-verdict-evidence]').innerText,
          title: s.querySelector('[data-testid=flow-v3-box-verdict]').title,
        } : null,
      })),
    }
  })
}

// Same box objects + payload segments → the panel's percentiles, recomputed here from the JSON (hand check).

// CR-DS: a verdict block (current: flow-v3-box-verdict; frozen: flow-v3-box-frozen-verdict) as data.
const verdictOf = (loc) => loc.evaluate((el) => ({
  kind: el.dataset.kind, tone: el.dataset.tone, evidence: el.dataset.evidence,
  pill: el.querySelector('[data-testid=flow-v3-box-verdict-pill]').innerText,
  pillBorder: getComputedStyle(el.querySelector('[data-testid=flow-v3-box-verdict-pill]')).borderLeftColor,
  reason: el.querySelector('[data-testid=flow-v3-box-verdict-reason]')?.innerText ?? '',
  title: el.querySelector('[data-testid=flow-v3-box-verdict-pill]').title,
  rows: [...el.querySelectorAll('[data-testid=flow-v3-box-evidence-row]')].map((r) => ({
    key: r.dataset.key, tone: r.dataset.tone, bar: getComputedStyle(r.firstElementChild).backgroundColor,
    check: r.children[1].innerText, value: r.children[2].innerText, reads: r.children[3].innerText,
  })),
}))
const tonesOf = (v) => Object.fromEntries(v.rows.map((r) => [r.key, r.tone]))
const RGB = { good: 'rgb(111, 143, 114)', bad: 'rgb(168, 100, 100)', idea: 'rgb(176, 138, 74)', neutral: 'rgb(100, 116, 139)' }
async function current(page) {
  const panel = page.getByTestId('flow-v3-box-popup')
  await panel.waitFor()
  const v = await verdictOf(panel.getByTestId('flow-v3-box-verdict'))
  const h = panel.getByTestId('flow-v3-box-history')
  return {
    verdict: v,
    history: (await h.count()) ? await h.innerText() : null,
    historyMs: Number(await panel.getAttribute('data-history-ms')),
    frozenRows: await panel.getByTestId('flow-v3-box-frozen-row').allInnerTexts(),
  }
}
const colourOk = (v) => {
  for (const r of v.rows) expect(r.bar, `${r.key} bar colour`).toBe(RGB[r.tone])
}

test('smoke 1: 2026-10-30 anchor 07:39, set end 08:02 — NO ACTION "short read, but price not at the top"; share / c/pt / absorption green, location red', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D30, { storage: EDGES_LEVELS })
  const a = shifted(`${D30}T07:39`)
  await anchorAt(page, a, a - 20 * 60, shifted(`${D30}T08:30`))
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D30}T08:02`), a - 20 * 60, shifted(`${D30}T08:30`))
  const c = await current(page)
  log('smoke-1-0739-0802.json', c)
  await shot(page, 'smoke-1-0739-0802.png')
  expect(c.verdict.kind).toBe('none')
  expect(c.verdict.pill).toBe('NO ACTION')
  expect(c.verdict.reason).toBe('short read, but price not at the top')
  const t = tonesOf(c.verdict)
  expect(t.share).toBe('good')
  expect(t.effort).toBe('good')
  expect(t.absorption).toBe('good')
  expect(t.location).toBe('bad')
  expect(c.verdict.rows.find((r) => r.key === 'location').value).toMatch(/below the bottom/)
  colourOk(c.verdict)
  expect(c.verdict.pillBorder).toBe(RGB.neutral)
  // read history: reported as it comes out (spec expectation: includes 07:49 (short))
  expect(c.history).toMatch(/^Read live: /)
})

test('smoke 2: same box with set end 07:49 (report), and anchored at 07:38 (the box CR-DS read from a 06:30 chain) to 07:49', async ({ page }) => {
  test.setTimeout(300_000)
  const out = {}
  await open(page, D30, { storage: EDGES_LEVELS })
  const a = shifted(`${D30}T07:39`)
  await anchorAt(page, a, a - 20 * 60, shifted(`${D30}T08:30`))
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D30}T07:49`), a - 20 * 60, shifted(`${D30}T08:30`))
  out.anchor0739 = await current(page)
  await shot(page, 'smoke-2-0739-0749.png')
  await page.goto('about:blank')
  await open(page, D30, { storage: EDGES_LEVELS })
  const a2 = shifted(`${D30}T07:38`) // CR-DU amendment 7: a 06:30 anchor no longer chains to this box
  await anchorAt(page, a2, a2 - 10 * 60, shifted(`${D30}T08:30`))
  await pickAt(page, 'flow-v3-end-pick', shifted(`${D30}T07:49`), a2 - 10 * 60, shifted(`${D30}T08:30`))
  out.anchor0630 = await current(page)
  await shot(page, 'smoke-2-0630-0749.png')
  log('smoke-2-end-0749.json', out)
  colourOk(out.anchor0739.verdict)
  expect(out.anchor0630.verdict.kind).toBe('short')
  expect(out.anchor0630.verdict.pillBorder).toBe(RGB.good)
  // CR-DU amendment 7: the panel's own verdict (not the engine's rule 5), as CR-DS shipped it
  for (const k of ['event', 'formed', 'location', 'absorption']) expect(tonesOf(out.anchor0630.verdict)[k], k).toBe('good')
  expect(out.anchor0630.history).toContain('07:49 (short)')
  // Ryan (chat 2026-10-31): effort meets no advance, so progress (+3.5) is slate "not needed", not red
  expect(out.anchor0630.verdict.rows.find((r) => r.key === 'progress')).toMatchObject({ tone: 'neutral', reads: 'not needed (effort)' })
})

test('smoke 3: CONFLICT (2026-10-30 until 08:30, anchor 08:00) shows green and red rows', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D30, { storage: EDGES_LEVELS, until: `${D30}T08:30` })
  const a = shifted(`${D30}T08:00`)
  await anchorAt(page, a, a - 20 * 60, shifted(`${D30}T08:30`))
  const c = await current(page)
  log('smoke-3-conflict.json', c)
  await shot(page, 'smoke-3-conflict.png')
  expect(c.verdict.kind).toBe('conflict')
  expect(c.verdict.pillBorder).toBe(RGB.bad)
  const tones = Object.values(tonesOf(c.verdict))
  expect(tones).toContain('good')
  expect(tones).toContain('bad')
  colourOk(c.verdict)
})

test('smoke 4: 2026-10-28 (FOMC) — NO TRADES, event row red', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D28, { storage: EDGES })
  await anchorAt(page, shifted(`${D28}T07:00`), shifted(`${D28}T06:40`), shifted(`${D28}T13:00`))
  const c = await current(page)
  log('smoke-4-fomc.json', c)
  await shot(page, 'smoke-4-fomc.png')
  expect(c.verdict.kind).toBe('no-trades')
  expect(c.verdict.pill).toBe('NO TRADES')
  expect(c.verdict.rows.find((r) => r.key === 'event')).toMatchObject({ tone: 'bad', reads: 'no trades' })
  expect(c.verdict.pillBorder).toBe(RGB.neutral)
  colourOk(c.verdict)
})

test('smoke 5 (CR-DU amendment 7): a manual box has no frozen rows; the break is a flag (2026-10-30 anchor 06:30); live replay time', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, D30, { storage: EDGES_LEVELS })
  const a = shifted(`${D30}T06:30`)
  await anchorAt(page, a, a - 10 * 60, shifted(`${D30}T13:00`))
  const c = await current(page)
  expect(c.frozenRows.length).toBe(0)
  await expect(page.getByTestId('flow-v3-box-broke')).toHaveText('broke ↓ 07:10')
  await shot(page, 'smoke-5-frozen-expanded.png')
  log('smoke-5-frozen.json', { current: c })
  expect(c.historyMs).toBeLessThan(250)
})

// No later bars: the replay's read minutes from a full-day load = the live verdict when the payload stops there.
test('smoke 6: read history vs live verdicts with until = t (2026-10-30, anchor 06:30)', async ({ page }) => {
  test.setTimeout(900_000)
  await open(page, D30, { storage: EDGES_LEVELS })
  const a = shifted(`${D30}T06:30`)
  await anchorAt(page, a, a - 10 * 60, shifted(`${D30}T13:00`))
  const panel = page.getByTestId('flow-v3-box-popup')
  const full = { current: await current(page), frozen: await panel.getByTestId('flow-v3-box-frozen-row').allInnerTexts() }
  const rows = []
  for (const until of ['07:48', '07:49', '07:50', '09:22', '09:23', '09:26', '09:27']) {
    await page.goto('about:blank')
    await page.unrouteAll({ behavior: 'ignoreErrors' })
    await open(page, D30, { storage: EDGES_LEVELS, until: `${D30}T${until}:59` })
    await anchorAt(page, a, a - 10 * 60, shifted(`${D30}T${until}`) + 10 * 60)
    const c = await current(page)
    rows.push({ until, kind: c.verdict.kind, pill: c.verdict.pill, reason: c.verdict.reason, history: c.history })
    log('smoke-6-until.json', { full, rows })
  }
  log('smoke-6-until.json', { full, rows })
  expect(rows.length).toBe(7)
})
