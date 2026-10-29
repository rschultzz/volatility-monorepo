// CR-DP — chart cleanup, box stats popup, delta-share anchored line, adjustable flow slots, y-axis zoom
// (specs/CR-DP-chart-cleanup.md). Read-only.
//
// Servers (started outside Playwright):
//   backend    CRDP_PORT=8067 apps/web/.venv/bin/python -u scripts/cr_dp_smoke_backend.py   (read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5191 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5192
// then:       CRBV_BASE_URL=http://localhost:5191 CRDP_API_BASE=http://127.0.0.1:8067 CRDP_BASELINE_URL=http://localhost:5192 \
//             npx playwright test cr-dp   (from react_price_preview/)
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { boxLabelText, kfmt } from '../../src/flowV3.js'

const API_BASE = process.env.CRDP_API_BASE || 'http://127.0.0.1:8067'
const BASELINE = process.env.CRDP_BASELINE_URL || 'http://localhost:5192'
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-dp/smoke')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height',
  'ib-react-flow-v3-slots', 'ib-react-price-zoom']
const V3 = { 'ib-react-flow-mode-v3': '1' } // origin/main needs it to open v3; the branch ignores (and removes) it
const layers = (o) => ({ 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: false, profile: false, heat: false, marks: false, ...o }) })
const EDGES = { ...V3, ...layers({ edges: true }) }
const D29 = '2026-10-29'
const D29_UNTIL = process.env.CRDP_D29_UNTIL || `${D29}T11:00`

const shifted = (ptIso) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}
const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}

async function open(page, day, { base = '', storage = {}, session = 'RTH', until = null } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdp-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crdp-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
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
  await page.waitForTimeout(2500)
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

// Arm 📍 and click the candle at `shiftedT` on the price chart (as cr-dn.spec.js).
async function anchorAt(page, shiftedT, frameTo = shiftedT + 4 * 3600) {
  await frame(page, shiftedT - 30 * 60, frameTo)
  await page.getByTestId('flow-v3-anchor-pick').click()
  const p = await page.evaluate(({ t, fn }) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    const bar = price.series.data().find((x) => x.time === t)
    return { x: rect.x + price.chart.timeScale().timeToCoordinate(t), y: rect.y + price.series.priceToCoordinate(bar.close) }
  }, { t: shiftedT, fn: CHARTS_FN })
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(1500)
}

const currentBoxLabel = (page) => page.evaluate((fn) => {
  const { price } = eval(fn) // eslint-disable-line no-eval
  const e = price.prims?.edges?._data
  const b = e ? e.boxes.find((x) => !x.frozen) : null
  return b ? { label: b.label, top: b.top, bottom: b.bottom, readout: b.box?.readout || null } : null
}, CHARTS_FN)

const popupBox = (page) => page.getByTestId('flow-v3-box-popup').evaluate((el) => {
  const r = el.getBoundingClientRect()
  return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), dragged: el.dataset.dragged }
})

test('items 1–3: header groups in order, no mode toggle, no pills; an approximate session shows the one note', async ({ page }) => {
  await open(page, D29, { storage: { ...V3, 'ib-react-flow-mode': 'v2', ...layers({ profile: true, levels: true }) } })
  const groups = await page.locator('[data-testid=flow-v3-panel] .flow-panel-header > *').evaluateAll((els) => els.map((e) => e.innerText.replace(/\n/g, ' · ')))
  log('header-10-29.json', groups)
  expect(groups[1]).toBe('📍 set anchor · set end · session open · last swing · clear')
  expect(groups[2]).toBe('Edges off · Edge effort off')
  expect(groups[3]).toBe('Absorption levels on · Profile on · Effort off')
  expect(groups[groups.length - 1]).toBe('⚙ · ?')
  await expect(page.locator('[data-testid^=flow-mode-]')).toHaveCount(0)
  await expect(page.getByTestId('flow-v3-cells-badge')).toHaveCount(0)
  await expect(page.getByTestId('flow-v3-approx-note')).toHaveCount(0)
  // a stored Current / v2 mode opened as v3 and the retired keys are gone
  expect(await page.evaluate(() => ['ib-react-flow-mode', 'ib-react-flow-mode-v3'].map((k) => window.localStorage.getItem(k)))).toEqual([null, null])
  // the ⚙ / ? group sits at the far right of the header
  const gap = await page.evaluate(() => {
    const h = document.querySelector('[data-testid=flow-v3-panel] .flow-panel-header')
    const last = h.lastElementChild.getBoundingClientRect()
    return Math.round(h.getBoundingClientRect().right - last.right)
  })
  expect(gap).toBeLessThan(16)

  const p2 = await page.context().newPage()
  await open(p2, '2026-10-20', { storage: { ...V3, ...layers({ profile: true, levels: true }) } })
  await expect(p2.getByTestId('flow-v3-approx-note')).toHaveText('approx data — absorption & edges off')
  await expect(p2.getByTestId('flow-v3-marks-note')).toHaveCount(0)
  await expect(p2.getByTestId('flow-v3-cells-badge')).toHaveCount(0)
})

test('item 4: 2026-10-29 anchor 06:49, Edges on — popup near the box with origin/main\'s numbers + Δ share; drags and stays; clear / new anchor / × / Edges off', async ({ browser }) => {
  const a = shifted(`${D29}T06:49`)
  const ctxB = await browser.newContext()
  const ctxM = await browser.newContext()
  const b = await ctxB.newPage()
  const m = await ctxM.newPage()
  await open(b, D29, { storage: EDGES, until: D29_UNTIL })
  await open(m, D29, { base: BASELINE, storage: EDGES, until: D29_UNTIL })
  await anchorAt(b, a)
  await anchorAt(m, a)

  // origin/main: the header's edge readout, the in-pane "box from…" line and the current box's label
  const mainHeader = await m.getByTestId('flow-v3-edge-readout').innerText()
  const mainPane = await m.getByTestId('flow-v3-readout').innerText()
  const mainBox = await currentBoxLabel(m)
  const brBox = await currentBoxLabel(b)
  const head = await b.getByTestId('flow-v3-box-popup-head').innerText()
  const top = await b.getByTestId('flow-v3-box-popup-top').innerText()
  const bottom = await b.getByTestId('flow-v3-box-popup-bottom').innerText()
  log('popup-06-49.json', { mainHeader, mainPane, mainLabel: mainBox?.label, branchLabel: brBox?.label, head, top, bottom })

  // one source: the branch box label equals origin/main's; the popup carries the same Δ · pts · m · /pt
  expect(brBox.label).toBe(mainBox.label)
  const [d, pts, mins, cpp] = mainBox.label.split(' · ')
  expect(head).toContain(`${d} · share `)
  expect(head).toContain(` · ${pts} · ${mins} · ${cpp}`)
  // the current box (by the replay time later breaks have held, so it need not start at the anchor) starts where
  // origin/main's in-pane "box from hh:mm" says
  const from = /^box from (\d\d:\d\d)/.exec(mainPane)[1]
  expect(head.startsWith(`box from ${from} · `)).toBe(true)
  // Δ share = Σ delta ÷ Σ volume of the same readout
  const ro = brBox.readout
  const share = ro.delta / ro.vol
  expect(head).toContain(`share ${share >= 0 ? '+' : ''}${(share * 100).toFixed(2)} %`)
  // Top / Bottom: price, main-side contracts and held minutes as origin/main's header readout
  const edge = (txt, name) => {
    const seg = txt.split(' | ').find((s) => s.includes(`${name} `))
    const mm = new RegExp(`${name} ([0-9.]+) · ([0-9.k]+) \\w+(?: · held (\\d+) m)?`).exec(seg)
    return { price: mm[1], contracts: mm[2], held: mm[3] ?? null }
  }
  const mt = edge(mainHeader, 'Top')
  const mb = edge(mainHeader, 'Bottom')
  expect(top).toMatch(new RegExp(`^Top ${mt.price} · buyers ${mt.contracts} / sellers [0-9.k]+ · ${mt.held == null ? 'not touched' : `held ${mt.held} m`}`))
  expect(bottom).toMatch(new RegExp(`^Bottom ${mb.price} · sellers ${mb.contracts} / buyers [0-9.k]+ · ${mb.held == null ? 'not touched' : `held ${mb.held} m`}`))
  // the in-pane "box from…" line is gone on the branch
  await expect(b.getByTestId('flow-v3-readout')).toHaveCount(0)
  await expect(b.getByTestId('flow-v3-edge-readout')).toHaveCount(0)

  // default place: just right of the box's top-right (or clamped inside the plot)
  const place = await b.evaluate((fn) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const e = price.prims.edges._data.boxes.find((x) => !x.frozen)
    const pane = price.chart.panes()[0].getHTMLElement().getBoundingClientRect()
    // the pane element is the table row (price axis included): the plot ends at the time scale's width
    return { x: pane.x + price.chart.timeScale().timeToCoordinate(e.to), y: pane.y + price.series.priceToCoordinate(e.top), pane: { x: pane.x, y: pane.y, r: pane.x + price.chart.timeScale().width(), b: pane.bottom } }
  }, CHARTS_FN)
  const p0 = await popupBox(b)
  log('popup-place.json', { place, p0 })
  expect(p0.dragged).toBe('0')
  expect(p0.x).toBeGreaterThanOrEqual(Math.round(place.pane.x))
  expect(p0.x + p0.w).toBeLessThanOrEqual(Math.round(place.pane.r) + 1)
  expect(p0.y + p0.h).toBeLessThanOrEqual(Math.round(place.pane.b) + 1)
  if (place.x + 8 + p0.w < place.pane.r - 4) expect(Math.abs(p0.x - (place.x + 8))).toBeLessThanOrEqual(2)
  expect(p0.w * p0.h).toBeLessThan(0.08 * (place.pane.r - place.pane.x) * (place.pane.b - place.pane.y)) // compact

  // drag by its body: it moves and stays (also across a re-render: Profile on / off)
  await b.mouse.move(p0.x + 20, p0.y + 8)
  await b.mouse.down()
  await b.mouse.move(p0.x - 200, p0.y + 120, { steps: 8 })
  await b.mouse.up()
  const p1 = await popupBox(b)
  expect(p1.dragged).toBe('1')
  expect(Math.abs(p1.x - (p0.x - 220))).toBeLessThanOrEqual(2)
  expect(Math.abs(p1.y - (p0.y + 112))).toBeLessThanOrEqual(2)
  await b.getByTestId('flow-v3-profile-toggle').click()
  await b.waitForTimeout(1500)
  await b.getByTestId('flow-v3-profile-toggle').click()
  await b.waitForTimeout(1500)
  await frame(b, a - 3600, a + 5 * 3600) // a pan / zoom doesn't move a dragged popup
  const p2 = await popupBox(b)
  expect([p2.x, p2.y]).toEqual([p1.x, p1.y])

  // clear removes it; a new anchor brings it back at its default place
  await b.getByTestId('flow-v3-anchor-clear').click()
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await anchorAt(b, a)
  const p3 = await popupBox(b)
  expect(p3.dragged).toBe('0')
  // × hides it until the next anchor; Esc clears; Edges off removes it
  await b.getByTestId('flow-v3-box-popup-close').click()
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await anchorAt(b, shifted(`${D29}T07:00`))
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(1)
  await b.keyboard.press('Escape')
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await anchorAt(b, a)
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(1)
  await b.getByTestId('flow-v3-edges-toggle').click()
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await b.screenshot({ path: path.join(LOGS, 'popup-edges-off.png') })
  await ctxB.close()
  await ctxM.close()
})
