// CR-DY — selectable boxes: drawn box, engine trade boxes, one panel (specs/CR-DY-selectable-boxes.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_du_smoke_backend.py        (read-only DB, capped pool, :8068)
//   frontend   npm --prefix react_price_preview run dev -- --port 5197 --strictPort
// then:       CRBV_BASE_URL=http://localhost:5197 CRDU_API_BASE=http://127.0.0.1:8068 npx playwright test tests/e2e/cr-dy.spec.js
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRDU_API_BASE || 'http://127.0.0.1:8068'
const HERE = path.dirname(fileURLToPath(import.meta.url))
const LOGS = path.resolve(HERE, '../../../scripts/logs/cr-dy/smoke')
const ASSETS = path.resolve(HERE, '../../../specs/assets/cr-dy')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height',
  'ib-react-flow-v3-slots', 'ib-react-price-zoom', 'ib-react-box-panel-pos']
const layers = (o) => ({ 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: false, profile: false, heat: false, marks: false, ...o }) })
const EDGES = layers({ edges: true })
const D30 = '2026-10-30'
const D26 = '2026-01-26' // a set-2 session: engine blocked
const TRAIN = '2026-02-06'

const shifted = (ptIso) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}
const hhmm = (shiftedT) => new Date(shiftedT * 1000).toISOString().slice(11, 16)
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


const barPoint = (page, t) => page.evaluate(({ t: tt, fn }) => {
  const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
  const rect = priceEl.getBoundingClientRect()
  const data = price.series.data().filter((x) => x.close != null)
  const bar = data.find((x) => x.time >= tt) || data[data.length - 1]
  return { x: rect.x + price.chart.timeScale().timeToCoordinate(bar.time), y: rect.y + price.series.priceToCoordinate(bar.close) }
}, { t, fn: CHARTS_FN })

const FLOW_FN = `(() => {
  const el = document.querySelector('[data-testid=flow-v3-panel] .tv-lightweight-charts')
  const key = Object.keys(el.parentElement).find((k) => k.startsWith('__reactFiber'))
  let f = el.parentElement[key]
  const reg = {}
  for (let d = 0; f && d < 40; d += 1, f = f.return) {
    for (let h = f.memoizedState; h; h = h.next) {
      const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
      if (c && c.delta && c.rolling && c.cvd) { reg.cvd = reg.cvd || c.cvd; reg.delta = reg.delta || c.delta }
    }
  }
  return { flow: { reg } }
})()`
const lineData = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  return flow.reg.cvd.data().filter((p) => p.value != null).map((p) => p.time)
}, FLOW_FN)
const boxes = (page) => page.evaluate(() => window.__crDyBoxes())
const rectOf = (st, id) => st.rects.find((r) => r.id === id)
// A point inside a drawn rectangle, in page px.
const inRect = (page, r) => page.evaluate(({ r: q, fn }) => {
  const { priceEl } = eval(fn) // eslint-disable-line no-eval
  const b = priceEl.getBoundingClientRect()
  return { x: b.x + (q.x0 + q.x1) / 2, y: b.y + (q.y0 + q.y1) / 2 }
}, { r, fn: CHARTS_FN })
const title = (page) => page.getByTestId('flow-v3-box-panel-title').innerText()

async function drawBox(page, from, to, frameFrom = from - 20 * 60, frameTo = to + 30 * 60) {
  await frame(page, frameFrom, frameTo)
  await page.getByTestId('flow-v3-draw-box').click()
  const a = await barPoint(page, from)
  await page.mouse.click(a.x, a.y)
  await page.waitForTimeout(700) // lightweight-charts swallows a second click inside its 500 ms double-click window
  const b = await barPoint(page, to)
  await page.mouse.click(b.x, b.y)
  await page.waitForTimeout(600)
}

test('1. 10-30 draw 07:34–07:49: rectangle, title, rows for that window, selected; the edge rule', async ({ page }) => {
  await open(page, D30, { storage: EDGES })
  const [a, z] = [shifted(`${D30}T07:34`), shifted(`${D30}T07:49`)]
  await drawBox(page, a, z)
  await page.getByTestId('flow-v3-box-verdict').waitFor()
  expect(await title(page)).toBe('Your box 07:34–07:49')
  expect(await page.getByTestId('flow-v3-box-panel-summary').innerText()).toContain('Box 07:34–07:49')
  const st = await boxes(page)
  expect(st.selected).toBe('drawn')
  expect(st.list).toHaveLength(1)
  expect(st.list[0]).toMatchObject({ from: a, to: z })
  expect(rectOf(st, 'drawn').selected).toBe(true)
  const rows = await page.getByTestId('flow-v3-box-evidence-row').evaluateAll((els) => els.map((e) => e.dataset.key))
  log('1-draw.json', { list: st.list, rows, summary: await page.getByTestId('flow-v3-box-panel-summary').innerText(),
    broke: (await page.getByTestId('flow-v3-box-broke').count()) ? await page.getByTestId('flow-v3-box-broke').innerText() : null })
  expect(rows.length).toBeGreaterThan(3)
  expect([st.list[0].high, st.list[0].low]).toEqual([7659.75, 7636.5]) // ruling 1: the wick range, display only
  expect(st.list[0].top).toBe(7654.25) // the edges stay the engine rule
  await expect(page.getByTestId('flow-v3-box-panel-rules')).toHaveCount(0) // ruling 2: drawn boxes keep panel rules
  // ruling 3: with an end set at 1440 px the chips wrap: nothing scrolls or clips, the v1 chip is inside the header
  const hdr = await page.evaluate(() => {
    const h = document.querySelector('[data-testid=flow-v3-panel] .flow-panel-header')
    const c = document.querySelector('[data-testid=flow-v3-v1-trades-toggle]').getBoundingClientRect()
    const r = h.getBoundingClientRect()
    return { over: h.scrollWidth - h.clientWidth, right: c.right, hRight: r.right, h: r.height }
  })
  log('1-header.json', hdr)
  expect(hdr.over).toBeLessThanOrEqual(0)
  expect(hdr.right).toBeLessThanOrEqual(hdr.hRight)
  await shot(page, '1-draw-0734-0749.png')
})

test('2. past date: nothing between the two clicks, Esc cancels the draw; session open sets the start while armed', async ({ page }) => {
  await open(page, D30, { storage: EDGES })
  const a = shifted(`${D30}T07:34`)
  await frame(page, a - 20 * 60, a + 40 * 60)
  await page.getByTestId('flow-v3-draw-box').click()
  await expect(page.getByTestId('flow-v3-draw-box')).toContainText('click start bar')
  const p = await barPoint(page, a)
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(400)
  await expect(page.getByTestId('flow-v3-draw-box')).toContainText('click end bar')
  expect(await page.locator('[data-anchor]').first().getAttribute('data-anchor')).toBe(String(a))
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  expect((await boxes(page)).list).toHaveLength(0)
  await page.keyboard.press('Escape')
  await page.waitForTimeout(300)
  expect(await page.locator('[data-anchor]').first().getAttribute('data-anchor')).toBe('')
  await expect(page.getByTestId('flow-v3-draw-box')).toContainText('draw box')
  // quick pick as the start
  await page.getByTestId('flow-v3-draw-box').click()
  await page.getByTestId('flow-v3-anchor-open').click()
  await page.waitForTimeout(300)
  await expect(page.getByTestId('flow-v3-draw-box')).toContainText('click end bar')
  const open630 = shifted(`${D30}T06:30`)
  expect(await page.locator('[data-anchor]').first().getAttribute('data-anchor')).toBe(String(open630))
  await frame(page, open630 - 300, open630 + 40 * 60)
  const e = await barPoint(page, open630 + 15 * 60)
  await page.mouse.click(e.x, e.y)
  await page.getByTestId('flow-v3-box-verdict').waitFor()
  expect(await title(page)).toBe('Your box 06:30–06:45')
})

test('3. 10-30 v1 trades on: one box per trade at its decision minute; each click moves the panel and the highlight; Esc and an empty click clear', async ({ page }) => {
  await open(page, D30, { storage: EDGES })
  await page.getByTestId('flow-v3-v1-trades-toggle').click()
  await expect.poll(() => page.evaluate(() => window.__crDuV1?.()?.minutes ?? 0), { timeout: 60_000 }).toBeGreaterThan(300)
  const sim = await page.evaluate(() => window.__crDuV1())
  const trades = [...sim.trades, ...(sim.open ? [sim.open] : [])]
  expect(trades.length).toBeGreaterThan(0)
  await frame(page, shifted(`${D30}T06:30`), shifted(`${D30}T13:00`))
  const st0 = await boxes(page)
  expect(st0.list).toHaveLength(trades.length)
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await shot(page, '3-engine-boxes.png')
  const seen = []
  for (let k = 0; k < trades.length; k += 1) {
    const tr = trades[k]
    const id = `v1-${k}`
    const lb = st0.list.find((b) => b.id === id)
    expect([lb.top, lb.bottom]).toEqual([tr.box.top, tr.box.bottom])
    const span = 25 * 60
    await frame(page, lb.from - span, lb.to + span)
    const r = rectOf(await boxes(page), id)
    expect(r, `${id} on screen`).toBeTruthy()
    const pt = await inRect(page, r)
    await page.mouse.move(pt.x, pt.y)
    await page.waitForTimeout(250)
    await page.mouse.click(pt.x, pt.y)
    await page.getByTestId('flow-v3-box-verdict').waitFor()
    await page.waitForTimeout(300)
    const st = await boxes(page)
    expect(st.selected).toBe(id)
    expect(st.rects.filter((x) => x.selected).map((x) => x.id)).toEqual([id])
    const t = await title(page)
    expect(t).toBe(`Engine box ${hhmm(lb.from)}–${hhmm(lb.to)} · trade ${k + 1} (${tr.side})`)
    const tradeLine = await page.getByTestId('flow-v3-box-trade').innerText()
    expect(tradeLine).toContain(`${tr.side} · entry ${tr.entry.toFixed(2)}`)
    expect(tradeLine).toContain(`stop ${tr.stop.toFixed(2)}`)
    expect(tradeLine).toContain(`TP ${tr.target.toFixed(2)}`)
    const eng = await page.getByTestId('flow-v3-box-engine').innerText()
    expect(eng).toContain(`Engine at ${hhmm(lb.to)}: ${tr.side.toUpperCase()}`)
    const pill = await page.getByTestId('flow-v3-box-verdict-pill').innerText()
    expect(pill).toBe(tr.side.toUpperCase()) // ruling 2: the engine's verdict, as "Engine at …"
    const panelRules = await page.getByTestId('flow-v3-box-panel-rules').innerText()
    expect(panelRules).toMatch(/^Panel rules \(p80 lean\): /)
    expect((st.list.find((b) => b.id === id).high)).toBe(tr.box.high) // ruling 1: wick lines from the engine's own box
    expect((st.list.find((b) => b.id === id).low)).toBe(tr.box.low)
    seen.push({ id, title: t, trade: tradeLine, engine: eng, pill, panelRules })
    if (k === 0) await shot(page, '3-engine-box-selected.png')
  }
  log('3-engine-boxes.json', { seen, n: trades.length })
  // hover: box and its mark (the panel shows trade k-1 selected; hover the first)
  await page.keyboard.press('Escape')
  await page.waitForTimeout(300)
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  expect((await boxes(page)).selected).toBeNull()
  expect((await boxes(page)).list).toHaveLength(trades.length) // the boxes stay
  const lb = st0.list[st0.list.length - 1]
  await frame(page, lb.from - 25 * 60, lb.to + 25 * 60)
  const r = rectOf(await boxes(page), lb.id)
  const pt = await inRect(page, r)
  await page.mouse.move(pt.x, pt.y)
  await expect.poll(async () => (await boxes(page)).hover).toBe(lb.id)
  expect(rectOf(await boxes(page), lb.id).hover).toBe(true)
  // click → selected, empty click → cleared
  await page.mouse.click(pt.x, pt.y)
  await page.getByTestId('flow-v3-box-verdict').waitFor()
  await page.waitForTimeout(700) // past the chart's double-click window
  const empty = await inRect(page, { x0: r.x1 + 120, x1: r.x1 + 130, y0: 40, y1: 50 })
  await page.mouse.click(empty.x, empty.y)
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveCount(0)
})

test('4. drawn + engine boxes: the highlight moves between them, the delta line follows the selection', async ({ page }) => {
  await open(page, D30, { storage: EDGES })
  const [a, z] = [shifted(`${D30}T09:00`), shifted(`${D30}T09:20`)]
  await drawBox(page, a, z)
  await page.getByTestId('flow-v3-v1-trades-toggle').click()
  await expect.poll(() => page.evaluate(() => window.__crDuV1?.()?.minutes ?? 0), { timeout: 60_000 }).toBeGreaterThan(300)
  await frame(page, shifted(`${D30}T06:30`), shifted(`${D30}T13:00`))
  const st0 = await boxes(page)
  expect(st0.selected).toBe('drawn')
  const eb = st0.list.find((b) => b.id !== 'drawn')
  await frame(page, eb.from - 25 * 60, eb.to + 25 * 60)
  const pt = await inRect(page, rectOf(await boxes(page), eb.id))
  await page.mouse.click(pt.x, pt.y)
  await expect.poll(async () => (await boxes(page)).selected).toBe(eb.id)
  await expect(page.getByTestId('flow-v3-box-anchor')).toHaveCount(0)
  await expect.poll(async () => { // the delta line reads the selected box: first point = start + 1 bar, last = decision bar
    const l = await lineData(page)
    return [l[0], l[l.length - 1]]
  }).toEqual([eb.from + 60, eb.to])
  // Esc clears the engine selection; your box stays, Esc again removes it
  await page.keyboard.press('Escape')
  await expect.poll(async () => (await boxes(page)).selected).toBeNull()
  expect((await boxes(page)).list.some((b) => b.id === 'drawn')).toBe(true)
  await page.keyboard.press('Escape')
  await expect.poll(async () => (await boxes(page)).list.some((b) => b.id === 'drawn')).toBe(false)
})

test('5. 2026-01-26 (set-2 session, engine blocked): drawn box works, no engine boxes, the reason shown once', async ({ page }) => {
  await open(page, D26, { storage: EDGES, until: `${D26}T08:50` })
  const [a, z] = [shifted(`${D26}T08:31`), shifted(`${D26}T08:45`)]
  await drawBox(page, a, z, a - 20 * 60, shifted(`${D26}T09:00`))
  await page.getByTestId('flow-v3-box-verdict').waitFor()
  expect(await title(page)).toBe('Your box 08:31–08:45')
  const st = await boxes(page)
  expect(st.list.map((b) => b.id)).toEqual(['drawn'])
  await expect(page.getByTestId('flow-v3-v1-trades-toggle')).toBeDisabled()
  expect(await page.getByTestId('flow-v3-v1-trades-toggle').innerText()).toContain('set 2 in progress')
  const n = await page.evaluate(() => (document.body.innerText.match(/set 2 in progress/g) || []).length)
  expect(n).toBe(1)
  await shot(page, '5-0126-blocked.png')
})

test('6. an armed click over the box panel reaches the bar under it', async ({ page }) => {
  await open(page, D30, { storage: EDGES })
  await drawBox(page, shifted(`${D30}T09:00`), shifted(`${D30}T09:30`), shifted(`${D30}T08:30`), shifted(`${D30}T10:30`))
  const panel = page.getByTestId('flow-v3-box-popup')
  await panel.waitFor()
  const b = await panel.boundingBox()
  await page.getByTestId('flow-v3-anchor-pick').click()
  await expect(panel).toHaveAttribute('data-armed', '1')
  const before = await page.locator('[data-anchor]').first().getAttribute('data-anchor')
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2)
  await page.waitForTimeout(500)
  const after = await page.locator('[data-anchor]').first().getAttribute('data-anchor')
  expect(after).not.toBe(before) // the pick landed on a bar
  expect(await page.getByTestId('flow-v3-anchor-pick').getAttribute('aria-pressed')).toBe('false')
  await expect(page.getByTestId('flow-v3-box-popup')).toHaveAttribute('data-armed', '0')
})

test('7. a review save sends the drawn box start, end, top, bottom', async ({ page }) => {
  const c = {
    id: 'X:fake:1', cr_id: 'X', set_name: 'fake', session_date: TRAIN, entry_ts_pt: `${TRAIN}T08:00:00`, entry_price: 6900, stop_price: 6890,
    direction: 'long', level_price: 6900, meta: {}, reviews: [], reviewed: false, revealed: false, n_reviews: 0, label: null,
  }
  const posts = []
  const json = (route, body) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  await page.route('**/api/setup-review/**', (route) => {
    const u = new URL(route.request().url())
    if (u.pathname.endsWith('/review')) {
      posts.push(route.request().postDataJSON())
      return json(route, { ok: true, review: { id: 1, label: 'take', reasons: [], comment: null, reviewed_at: new Date().toISOString(), revealed_at: null, post_reveal: false } })
    }
    if (u.pathname.endsWith('/sets')) return json(route, { ok: true, sets: [{ cr_id: 'X', set_name: 'fake', n: 1 }] })
    if (u.pathname.endsWith('/candidates')) return json(route, { ok: true, candidates: [c], reasons: [] })
    if (u.pathname.includes('/candidates/')) return json(route, { ok: true, candidate: c })
    return json(route, { ok: true })
  })
  await page.addInitScript(() => { try { window.localStorage.setItem('dash.review.set', JSON.stringify({ cr: 'X', set: 'fake' })) } catch (e) { /* ignore */ } })
  await open(page, TRAIN, { storage: EDGES })
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.locator('[data-testid="review-row"]').first().click()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await settle(page)
  const [a, z] = [shifted(`${TRAIN}T07:30`), shifted(`${TRAIN}T07:45`)]
  await drawBox(page, a, z, a - 20 * 60, shifted(`${TRAIN}T08:00`))
  const st = await boxes(page)
  const lb = st.list.find((b) => b.id === 'drawn')
  await page.getByTestId('review-label-take').click()
  await page.getByTestId('review-save').click()
  await expect.poll(() => posts.length, { timeout: 20_000 }).toBe(1)
  log('7-post.json', posts[0])
  expect(posts[0]).toMatchObject({ label: 'take', anchor_ts: `${TRAIN}T07:30:00`, box_end_ts: `${TRAIN}T07:45:00`, box_top: lb.top, box_bottom: lb.bottom, box_min_age_min: 8, box_min_height_pts: 2 })
})
