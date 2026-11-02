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
const TRAIN = '2026-02-06' // a train session without a set-2 card (its trades are in the backtest's per-trade rows)
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
    engineBox: (await page.getByTestId('flow-v3-box-engine').count()) ? await page.getByTestId('flow-v3-box-engine').innerText() : '',
    engine: (await page.getByTestId('flow-v3-box-engine-line').count()) ? await page.getByTestId('flow-v3-box-engine-line').innerText() : '',
    rows: await page.getByTestId('flow-v3-box-evidence-row').evaluateAll((els) => els.map((e) => [e.dataset.key, e.dataset.tone, e.innerText.replace(/\s+/g, ' ')])),
  }
}

test.describe.configure({ mode: 'serial' })
test.setTimeout(240_000)

const ENGINE_LINE = /^Engine at \d\d:\d\d: (NO ACTION( — rule \d+: .+)?|SHORT|LONG|NO TRADES( — .+)?)/
const summary = async (page) => (await page.locator('[data-testid="flow-v3-box-panel-current"] [data-testid="flow-v3-box-panel-summary"]').innerText()).replace(/^[▸▾]\s*/, '')

test('1. 10-30 manual box (anchor 07:43, replay to 07:51): box min 8 vs "box still forming" at 10; the engine line is separate', async ({ page }) => {
  const a = shifted(`${D30}T07:43`)
  await open(page, D30, { storage: EDGES, until: `${D30}T07:52` })
  await anchorAt(page, a, a - 30 * 60, shifted(`${D30}T08:00`))
  const v8 = await verdict(page)
  await shot(page, '1-min8.png')
  expect(v8.kind).not.toBe('forming')
  expect(v8.tag).toBe('engine v1.4 — untested') // on the engine line, in the panel's muted secondary text
  expect(v8.engineBox).toMatch(ENGINE_LINE)
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
  await open(page2, D30, { until: `${D30}T07:52`, keepStorage: true })
  await anchorAt(page2, a, a - 30 * 60, shifted(`${D30}T08:00`))
  const v10 = await verdict(page2)
  await shot(page2, '1-min10.png')
  expect(v10.kind).toBe('forming')
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
  expect(bt.length).toBeGreaterThan(0) // not vacuous
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

// CR-DU amendment 7 — manual box, engine line, delta line. 01-26 holds set-2 cards (engine blocked): no screenshot, and the
// log keeps the signs only.
const D26 = '2026-01-26'
const lineData = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  return flow.reg.cvd.data().map((p) => ({ time: p.time, value: p.value ?? null }))
}, FLOW_FN)
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
const headShare = async (page) => {
  await page.locator('[data-testid="flow-v3-box-panel-current"] [data-testid="flow-v3-box-panel-summary"]').click()
  const head = await page.getByTestId('flow-v3-box-popup-head').innerText()
  const m = /share ([+-]?[\d.]+) %/.exec(head.replace('−', '-'))
  return m ? Number(m[1]) : null
}

test('6. 2026-01-26 anchor 08:31 (card replay to 08:49): rows and line cover 08:31 →; the anchor shows; the engine is blocked; line sign = panel Δ share', async ({ page }) => {
  const a = shifted(`${D26}T08:31`)
  await open(page, D26, { storage: EDGES, until: `${D26}T08:50` })
  await anchorAt(page, a, a - 20 * 60, shifted(`${D26}T09:00`))
  await page.getByTestId('flow-v3-box-verdict').waitFor()
  expect(await summary(page)).toBe('Box 08:31–08:49 · current')
  expect(await page.getByTestId('flow-v3-box-anchor').innerText()).toBe('anchor 08:31')
  expect(await page.getByTestId('flow-v3-box-broke').count()).toBe(0)
  const v = await verdict(page)
  expect(v.engineBox).toBe('')
  expect(v.engine).toBe('') // CR-DY: the reason is on the v1 chip, once
  const line = await lineData(page)
  const drawn = line.filter((p) => p.value != null)
  expect(hhmm(drawn[0].time)).toBe('08:32') // nothing is summed at the start bar
  expect(hhmm(drawn[drawn.length - 1].time)).toBe('08:49')
  expect(await page.locator('[data-box-ticks]').first().getAttribute('data-box-ticks')).toBe('')
  const share = await headShare(page)
  expect(share).not.toBeNull()
  const same = Math.sign(drawn[drawn.length - 1].value) === Math.sign(share)
  log('6-0126.json', { summary: await summary(page), firstLine: hhmm(drawn[0].time), lastLine: hhmm(drawn[drawn.length - 1].time), sameSign: same })
  expect(same).toBe(true)
})

for (const [name, day, a1, a2, cut] of [
  ['10-30 (Ironbeam)', D30, '07:10', '07:43', '08:30'],
  ['train day with no set-2 card', TRAIN, '07:10', '07:40', '08:30'],
]) {
  test(`7. engine line, ${name}: its box comes from its own chain and two anchors don't move it; the rows do`, async ({ page }) => {
    await open(page, day, { storage: EDGES, until: `${day}T${cut}` })
    const out = {}
    for (const a of [a1, a2]) {
      const t = shifted(`${day}T${a}`)
      await page.keyboard.press('Escape')
      await anchorAt(page, t, t - 20 * 60, shifted(`${day}T${cut}`))
      const v = await verdict(page)
      out[a] = { summary: await summary(page), anchor: await page.getByTestId('flow-v3-box-anchor').innerText(), engineBox: v.engineBox, tag: v.tag, formed: v.rows.find((r) => r[0] === 'formed')?.[2] }
      expect(out[a].summary.startsWith(`Box ${a}–`)).toBe(true)
      expect(out[a].anchor).toBe(`anchor ${a}`)
      expect(v.engineBox).toMatch(ENGINE_LINE)
    }
    await shot(page, `7-engine-line-${day}.png`)
    log(`7-${day}.json`, out)
    expect(out[a1].engineBox).toBe(out[a2].engineBox)
    expect(out[a1].summary).not.toBe(out[a2].summary)
  })
}

test('8. a held break is a flag, the box stays; "restart here" re-anchors, the line resets there with a tick', async ({ page }) => {
  const D = '2026-10-28'
  const a = shifted(`${D}T07:00`)
  await open(page, D, { storage: EDGES })
  await anchorAt(page, a, a - 20 * 60, shifted(`${D}T10:00`))
  await page.getByTestId('flow-v3-box-broke').waitFor()
  const flag = await page.getByTestId('flow-v3-box-broke').innerText()
  expect(flag).toMatch(/^broke [↑↓] \d\d:\d\d$/)
  expect((await summary(page)).startsWith('Box 07:00–')).toBe(true)
  expect(await page.getByTestId('flow-v3-box-panel-frozen').count()).toBe(0)
  await shot(page, '8-broke-flag.png')
  const at = flag.slice(-5)
  await page.getByTestId('flow-v3-box-restart').click()
  await page.waitForTimeout(1500)
  const t = shifted(`${D}T${at}`)
  expect(Number(await page.locator('[data-anchor]').first().getAttribute('data-anchor'))).toBe(t)
  expect((await summary(page)).startsWith(`Box ${at}–`)).toBe(true)
  expect(await page.getByTestId('flow-v3-box-anchor').innerText()).toBe(`anchor ${at}`)
  expect(await page.locator('[data-box-ticks]').first().getAttribute('data-box-ticks')).toBe(String(t))
  const line = await lineData(page)
  // data() omits whitespace rows: the new start bar has no point at all (nothing summed there, as cr-dp item 6)
  expect(line.find((p) => p.time === t)?.value ?? null).toBeNull()
  expect(line.some((p) => p.value != null && p.time < t)).toBe(true) // the old span stays drawn
  expect(line.some((p) => p.value != null && p.time > t)).toBe(true) // the new span starts after it
  await shot(page, '8-restarted.png')
  log('8-restart.json', { flag, at })
  // a freshly picked anchor drops the kept span and the tick
  await page.keyboard.press('Escape')
  await anchorAt(page, shifted(`${D}T09:00`), shifted(`${D}T08:40`), shifted(`${D}T10:00`))
  expect(await page.locator('[data-box-ticks]').first().getAttribute('data-box-ticks')).toBe('')
})

test('9. amendment 7b: a review save sends the anchor and the box min / height in force', async ({ page }) => {
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
  await page.locator('.tv-lightweight-charts').first().waitFor() // the card's replay remounts the chart
  await settle(page)
  const t = shifted(`${TRAIN}T07:30`)
  await anchorAt(page, t, t - 20 * 60, shifted(`${TRAIN}T08:00`))
  await page.getByTestId('review-label-take').click()
  await page.getByTestId('review-save').click()
  await expect.poll(() => posts.length, { timeout: 20_000 }).toBe(1)
  log('9-post.json', posts[0])
  expect(posts[0]).toMatchObject({ label: 'take', anchor_ts: `${TRAIN}T07:30:00`, box_min_age_min: 8, box_min_height_pts: 2 })
})

// One-minute alignment, measured (Ryan, chat 2026-11-01): the same bar through the chart, panel, delta line and Flow v3 bar.
for (const [name, day, hm, until, db] of [
  ['10-30 07:49', D30, '07:49', null, { o: 7657, h: 7657, l: 7654, c: 7656.5 }],
  ['01-26 08:31', D26, '08:31', `${D26}T08:50`, { o: 6984.5, h: 6986.25, l: 6984.25, c: 6986, delta: 877 - 919 }],
]) {
  test(`10. alignment ${name}: every layer labels this bar by its start`, async ({ page }) => {
    const t = shifted(`${day}T${hm}`)
    await open(page, day, { storage: EDGES, until })
    await anchorAt(page, t, t - 15 * 60, t + 15 * 60)
    const r = await page.evaluate(({ t, fn, fn2 }) => {
      const { price, priceEl } = eval(fn) // eslint-disable-line no-eval
      const { flow } = eval(fn2) // eslint-disable-line no-eval
      const c = price.series.data().find((x) => x.time === t)
      const opt = price.chart.options()
      const x = price.chart.timeScale().timeToCoordinate(t)
      const back = price.chart.timeScale().coordinateToTime(x)
      const fm = opt.localization?.timeFormatter
      const near = (s) => s.data().filter((p) => Math.abs(p.time - t) <= 120).map((p) => ({ time: p.time, value: p.value ?? null }))
      return { candle: c && { time: c.time, o: c.open, h: c.high, l: c.low, c: c.close }, crosshairTime: back, crosshairLabel: fm ? fm(back) : null, tick: opt.timeScale?.tickMarkFormatter ? opt.timeScale.tickMarkFormatter(t) : null, x, line: near(flow.reg.cvd), delta: near(flow.reg.delta), rectX: priceEl.getBoundingClientRect().x }
    }, { t, fn: CHARTS_FN, fn2: FLOW_FN })
    const out = {
      candle: { label: hhmm(r.candle.time), ohlc: r.candle }, crosshair: { time: hhmm(r.crosshairTime), label: r.crosshairLabel },
      panel: { summary: await summary(page), anchor: await page.getByTestId('flow-v3-box-anchor').innerText() },
      line: r.line.map((p) => ({ time: hhmm(p.time), hasValue: p.value != null })),
      delta: r.delta.map((p) => ({ time: hhmm(p.time), value: Math.abs(p.value) })),
    }
    log(`10-align-${day}.json`, out)
    expect([r.candle.o, r.candle.h, r.candle.l, r.candle.c]).toEqual([db.o, db.h, db.l, db.c])
    expect(hhmm(r.crosshairTime)).toBe(hm)
    expect(out.panel.anchor).toBe(`anchor ${hm}`)
    expect(out.panel.summary.startsWith(`Box ${hm}–`)).toBe(true)
    expect(r.line.find((p) => p.time === t)?.value ?? null).toBeNull() // the anchor bar's own flow is not summed
    expect(r.line.find((p) => p.time === t + 60)?.value ?? null).not.toBeNull()
    if (db.delta != null) expect(r.delta.find((p) => p.time === t)?.value).toBe(db.delta)
  })
}
