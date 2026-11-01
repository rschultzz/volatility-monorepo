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
  return b ? { label: b.label ?? null, top: b.top, bottom: b.bottom, readout: b.box?.readout || null } : null
}, CHARTS_FN).then((x) => (x && x.label == null && x.readout ? { ...x, label: boxLabelText(x.readout) } : x)) // CR-DQ item 1: no in-chart label

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
  expect(groups[3]).toBe('Absorption levels on · Profile on · Effort off · My trades off · v1 trades off') // CR-DR item 2: My trades joins the Display group; CR-DU item 4b: v1 trades after it
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

  // CR-DQ item 3 (re-pointed): default place bottom-left inside the plot (was: right of the box's top-right)
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
  expect(Math.abs(p0.x - (place.pane.x + 4))).toBeLessThanOrEqual(2)
  expect(Math.abs(p0.y + p0.h - (place.pane.b - 4))).toBeLessThanOrEqual(2)
  expect(p0.h).toBeLessThanOrEqual(0.5 * (place.pane.b - place.pane.y) + 2) // CR-DQ: at most half the plot

  // drag (CR-DQ item 3: by its header): it moves and stays (also across a re-render: Profile on / off)
  await b.mouse.move(p0.x + 20, p0.y + 8)
  await b.mouse.down()
  await b.mouse.move(p0.x - 200, p0.y + 120, { steps: 8 })
  await b.mouse.up()
  const p1 = await popupBox(b)
  expect(p1.dragged).toBe('1')
  // clamped inside the plot (it starts bottom-left, so left / down stop at the plot's edge)
  expect(p1.x).toBeGreaterThanOrEqual(Math.round(place.pane.x) - 1)
  expect(p1.y + p1.h).toBeLessThanOrEqual(Math.round(place.pane.b) + 1)
  await b.getByTestId('flow-v3-profile-toggle').click()
  await b.waitForTimeout(1500)
  await b.getByTestId('flow-v3-profile-toggle').click()
  await b.waitForTimeout(1500)
  await frame(b, a - 3600, a + 5 * 3600) // a pan / zoom doesn't move a dragged popup
  const p2 = await popupBox(b)
  expect([p2.x, p2.y]).toEqual([p1.x, p1.y])

  // clear removes it; CR-DQ item 3 (re-pointed): a new anchor reuses the remembered place
  await b.getByTestId('flow-v3-anchor-clear').click()
  await expect(b.getByTestId('flow-v3-box-popup')).toHaveCount(0)
  await anchorAt(b, a)
  const p3 = await popupBox(b)
  expect(p3.dragged).toBe('1')
  expect([p3.x, p3.y + p3.h]).toEqual([p1.x, p1.y + p1.h])
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

const ptShift = (utc) => {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hourCycle: 'h23',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
    .formatToParts(new Date(utc * 1000)).map((x) => [x.type, x.value]))
  return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second) / 1000
}
const hm = (shiftedT) => new Date(shiftedT * 1000).toISOString().slice(11, 16)
const lineData = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  return flow.reg.cvd.data().map((p) => ({ time: p.time, value: p.value ?? null }))
}, CHARTS_FN)

async function pickAt(page, testId, shiftedT) {
  await page.getByTestId(testId).click()
  const p = await page.evaluate(({ t, fn }) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    const bar = price.series.data().find((x) => x.time === t)
    return { x: rect.x + price.chart.timeScale().timeToCoordinate(t), y: rect.y + price.series.priceToCoordinate(bar.close) }
  }, { t: shiftedT, fn: CHARTS_FN })
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(1500)
}

const v2Payload = (page, until) => page.evaluate(async ({ api, day, until: u }) => {
  const url = new URL(`${api}/api/ironbeam/flow`)
  Object.entries({ trade_date: day, session: 'RTH', mode: 'v2', m_swing: 8, m_push: 500, m_div: 300, m_fade: 0.1, m_gap: 30, effort: 1, until: u })
    .forEach(([k, v]) => url.searchParams.set(k, String(v)))
  return (await fetch(url, { credentials: 'include' })).json()
}, { api: API_BASE, day: D29, until })

// Σ delta ÷ Σ vol of the v2 minute bars after `from` up to `last` (shifted PT epochs), straight from the payload.
const handShare = (payload, from, last) => {
  let sd = 0
  let sv = 0
  for (const bar of payload.bars) {
    const t = ptShift(Number(bar.time))
    if (t > from && t <= last) {
      sd += Number(bar.delta) || 0
      sv += Number(bar.vol) || 0
    }
  }
  return { sd, sv, share: sd / sv }
}

test('item 6: delta-share line (2026-10-29): box 10:15–10:34 ≈ +1 % = hand Σ delta ÷ Σ volume; per box with Edges on; fixed ±10 % with guides; raw CVD / no anchor = origin/main', async ({ browser }) => {
  const until = process.env.CRDP_D29_UNTIL_FULL || `${D29}T13:00`
  const a = shifted(`${D29}T10:15`)
  const e = shifted(`${D29}T10:34`)
  const ctxB = await browser.newContext()
  const ctxM = await browser.newContext()
  const b = await ctxB.newPage()
  const m = await ctxM.newPage()
  await open(b, D29, { storage: EDGES, until })
  await open(m, D29, { base: BASELINE, storage: EDGES, until })

  // no anchor: session CVD, identical to origin/main
  expect(await lineData(b)).toEqual(await lineData(m))

  // box 10:15 → 10:34 (anchor + set end, Edges on): one segment
  await anchorAt(b, a, a + 3600)
  await pickAt(b, 'flow-v3-end-pick', e)
  const payload = await v2Payload(b, until)
  const h = handShare(payload, a, e)
  const line = await lineData(b)
  const boxes = await b.evaluate((fn) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    return price.prims.edges._data.boxes.map((x) => ({ from: x.from, lastUtc: x.box.to, frozen: x.frozen, readout: x.box.readout }))
  }, CHARTS_FN)
  const popupHead = await b.getByTestId('flow-v3-box-popup-head').innerText()
  const rec = { boxes: boxes.map((x) => ({ from: hm(x.from), last: hm(ptShift(x.lastUtc)), frozen: x.frozen, share: x.readout ? x.readout.delta / x.readout.vol : null })),
    hand: h, lineAtEnd: line.find((p) => p.time === e)?.value ?? null, popupHead }
  log('item6-1015-1034.json', rec)
  expect(boxes.length).toBe(1) // no break inside 10:15–10:34
  expect(Math.abs(rec.lineAtEnd - Math.max(-0.1, Math.min(0.1, h.share)))).toBeLessThan(1e-9)
  expect(Math.abs(boxes[0].readout.delta / boxes[0].readout.vol - h.share)).toBeLessThan(1e-9)
  expect(popupHead).toContain(`share ${h.share >= 0 ? '+' : ''}${(h.share * 100).toFixed(2)} %`)
  expect(Math.abs(h.share - 0.01)).toBeLessThan(0.01) // "near +1 %"
  expect(line.find((p) => p.time === a)?.value ?? null).toBeNull() // nothing summed at the start bar
  expect(line.filter((p) => p.value != null && (p.time <= a || p.time > e))).toEqual([]) // only the window is drawn

  // the fixed scale and guides
  const scale = await b.evaluate((fn) => {
    const { flow } = eval(fn) // eslint-disable-line no-eval
    const r = flow.reg.cvd.priceScale().getVisibleRange()
    return { from: r.from, to: r.to }
  }, CHARTS_FN)
  log('item6-scale.json', scale)
  expect(scale.from).toBeLessThanOrEqual(-0.1 + 1e-9)
  expect(scale.to).toBeGreaterThanOrEqual(0.1 - 1e-9)
  expect(scale.to - scale.from).toBeLessThan(0.25)

  // anchor 06:49 with Edges on (no end): each box restarts; each box's last value = its readout's share = hand sums
  await b.getByTestId('flow-v3-anchor-clear').click()
  await anchorAt(b, shifted(`${D29}T06:49`), shifted(`${D29}T13:00`))
  const segs = await b.evaluate((fn) => {
    const { price } = eval(fn) // eslint-disable-line no-eval
    return price.prims.edges._data.boxes.map((x) => ({ from: x.from, lastUtc: x.box.to, readout: x.box.readout }))
  }, CHARTS_FN)
  const line2 = await lineData(b)
  const perBox = segs.map((x) => {
    const last = ptShift(x.lastUtc)
    const hs = handShare(payload, x.from, last)
    return { from: hm(x.from), last: hm(last), hand: hs.share, line: line2.find((p) => p.time === last)?.value ?? null, start: line2.find((p) => p.time === x.from)?.value ?? null }
  })
  log('item6-boxes-0649.json', perBox)
  for (const x of perBox) {
    expect(Math.abs(x.line - Math.max(-0.1, Math.min(0.1, x.hand)))).toBeLessThan(1e-9)
    expect(x.start).toBeNull()
  }

  // ⚙ raw CVD: the line is origin/main's anchored (per-box) CVD again
  await anchorAt(m, shifted(`${D29}T06:49`), shifted(`${D29}T13:00`))
  await b.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  await b.getByTestId('flow-v3-anchored-line').selectOption('raw')
  await b.getByRole('button', { name: 'Apply' }).click()
  await b.waitForTimeout(1200)
  expect(await lineData(b)).toEqual(await lineData(m))
  await ctxB.close()
  await ctxM.close()
})

test('item 5: no SIGNALS / ANALOGUES pills or panels, no /signals request; REVIEW and LANDSCAPE still open', async ({ page }) => {
  test.setTimeout(240_000)
  const signalCalls = []
  page.on('request', (r) => { if (/\/signals\/|\/api\/analogues/.test(r.url())) signalCalls.push(r.url()) })
  await open(page, D29, { storage: V3 })
  const pills = await page.locator('button[aria-label]').evaluateAll((els) => els.map((e) => e.innerText.trim()).filter((t) => /^[A-Z]+$/.test(t)))
  log('item5-pills.json', pills)
  expect(pills).toEqual(['SMILE', 'GEX', 'LANDSCAPE', 'REVIEW'])
  await expect(page.getByRole('button', { name: 'Show SIGNALS' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Show day analogue comparison panel' })).toHaveCount(0)
  // LANDSCAPE: the panel's request returns and the panel draws
  const land = page.waitForResponse((r) => r.url().includes('/api/gex-landscape'), { timeout: 60_000 })
  await page.getByRole('button', { name: 'Show GEX landscape panel' }).click()
  expect((await land).status()).toBe(200)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: path.join(LOGS, 'item5-landscape.png') })
  // REVIEW: the panel opens (a fresh page: the open landscape panel sits over the pill row's right end)
  const p2 = await page.context().newPage()
  p2.on('request', (r) => { if (/\/signals\/|\/api\/analogues/.test(r.url())) signalCalls.push(r.url()) })
  await open(p2, D29, { storage: V3 })
  await p2.getByRole('button', { name: 'Show setup review panel' }).click()
  await expect(p2.getByTestId('review-panel')).toBeVisible()
  await p2.waitForTimeout(65_000) // longer than the removed SignalPanel's 60 s scan
  expect(signalCalls).toEqual([])
})

const slotMargins = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  const r6 = (v) => Math.round(v * 1e6) / 1e6
  const pm = (id) => { const m = flow.chart.priceScale(id).options().scaleMargins; return { top: r6(m.top), bottom: r6(m.bottom) } }
  return { cvd: pm('cvd'), right: pm('right'), delta: pm('delta'), height: flow.chart.options().height }
}, CHARTS_FN)

test('item 7: slot dividers resize slots (≥ 24 px, pane height unchanged), persist across reload; Reset layout restores slots and the price / Flow split', async ({ page }) => {
  await open(page, D29, { storage: V3 })
  const m0 = await slotMargins(page)
  const paneH0 = await page.getByTestId('flow-v3-panel').evaluate((el) => el.getBoundingClientRect().height)
  const d0 = page.getByTestId('flow-v3-slot-divider-0')
  await expect(d0).toHaveCount(1)
  await expect(page.getByTestId('flow-v3-slot-divider-1')).toHaveCount(1)
  const r = await d0.boundingBox()
  await page.mouse.move(r.x + 200, r.y + r.height / 2)
  await page.mouse.down()
  await page.mouse.move(r.x + 200, r.y + r.height / 2 + 12, { steps: 6 })
  await page.mouse.up()
  await page.waitForTimeout(500)
  const m1 = await slotMargins(page)
  log('item7-margins.json', { m0, m1 })
  const H = m1.height - 28 // pane 0 (time axis 28 px)
  // the CVD slot grew by 12 px, the middle slot (45 px at the default 160 px pane) shrank by 12 px, the delta slot is unchanged
  expect(Math.round(((1 - m1.cvd.bottom) - (1 - m0.cvd.bottom)) * H)).toBe(12)
  expect(Math.round((m1.right.top - m0.right.top) * H)).toBe(12)
  expect(m1.delta).toEqual(m0.delta)
  expect(m1.height).toBe(m0.height)
  expect(await page.getByTestId('flow-v3-panel').evaluate((el) => el.getBoundingClientRect().height)).toBe(paneH0)
  // drag far: the middle slot stops at 24 px
  const r2 = await d0.boundingBox()
  await page.mouse.move(r2.x + 200, r2.y + r2.height / 2)
  await page.mouse.down()
  await page.mouse.move(r2.x + 200, r2.y + 400, { steps: 6 })
  await page.mouse.up()
  const m2 = await slotMargins(page)
  expect(Math.round(((1 - m2.right.bottom) - m2.right.top) * H)).toBe(24)
  // persists across a reload
  await page.reload()
  await page.getByTestId('flow-v3-panel').waitFor()
  await page.waitForTimeout(2500)
  expect(await slotMargins(page)).toEqual(m2)
  // 7c: the existing price / Flow splitter changes the split and is remembered
  const sp = await page.locator('.panel-splitter').boundingBox()
  await page.mouse.move(sp.x + 300, sp.y + sp.height / 2)
  await page.mouse.down()
  await page.mouse.move(sp.x + 300, sp.y + sp.height / 2 - 60, { steps: 6 })
  await page.mouse.up()
  await page.waitForTimeout(500)
  const paneH1 = await page.getByTestId('flow-v3-panel').evaluate((el) => el.getBoundingClientRect().height)
  expect(paneH1).toBe(paneH0 + 60)
  expect(await page.evaluate(() => window.localStorage.getItem('ib-react-flow-panel-height'))).toBe(String(paneH0 + 60))
  // Reset layout: slots and split back to their defaults
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  await page.getByTestId('flow-v3-reset-layout').click()
  await page.waitForTimeout(800)
  expect(await page.getByTestId('flow-v3-panel').evaluate((el) => el.getBoundingClientRect().height)).toBe(paneH0)
  expect(await slotMargins(page)).toEqual(m0)
  expect(await page.evaluate(() => window.localStorage.getItem('ib-react-flow-v3-slots'))).toBeNull()
})

const priceRange = (page) => page.evaluate((fn) => {
  const { price } = eval(fn) // eslint-disable-line no-eval
  const sc = price.chart.priceScale('right')
  const r = sc.getVisibleRange()
  return { auto: sc.options().autoScale, from: r.from, to: r.to, bars: price.series.data().length }
}, CHARTS_FN)
const flowRanges = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  const g = (id) => { const r = flow.chart.priceScale(id).getVisibleRange(); return r ? { from: r.from, to: r.to } : null }
  return { cvd: g('cvd'), right: g('right'), delta: g('delta') }
}, CHARTS_FN)
// drag vertically on an axis at (x, y) by dy px
async function axisDrag(page, x, y, dy) {
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x, y + dy, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(600)
}

test('item 8: price axis drag zooms, survives a reload and live refreshes, double-click resets; Flow v3 axis zooms the slot under the pointer, double-click resets', async ({ page }) => {
  test.setTimeout(200_000)
  const live = process.env.CRDP_LIVE_DATE || '2026-10-30'
  await open(page, live, { storage: V3, session: 'FULL' })
  const pr = await page.locator('.tv-lightweight-charts').first().boundingBox()
  const axis = await page.evaluate((fn) => { const { price } = eval(fn); return price.chart.timeScale().width() }, CHARTS_FN) // eslint-disable-line no-eval
  const r0 = await priceRange(page)
  await axisDrag(page, pr.x + axis + 30, pr.y + 250, 120)
  const r1 = await priceRange(page)
  log('item8-price.json', { r0, r1 })
  expect(r1.auto).toBe(false)
  expect(r1.to - r1.from).not.toBeCloseTo(r0.to - r0.from, 1)
  // live refreshes (bars keep arriving on the live session) don't undo it: wait for a new bar
  let r2 = await priceRange(page)
  for (let i = 0; i < 45 && r2.bars <= r1.bars; i += 1) {
    await page.waitForTimeout(2000)
    r2 = await priceRange(page)
  }
  expect(r2.bars, 'a live bar arrived').toBeGreaterThan(r1.bars)
  log('item8-price-live.json', { r1, r2 })
  expect(r2.auto).toBe(false)
  expect(r2.from).toBeCloseTo(r1.from, 6)
  expect(r2.to).toBeCloseTo(r1.to, 6)
  // a reload restores it for the same date
  await page.reload()
  await page.getByTestId('flow-v3-panel').waitFor()
  await page.waitForTimeout(3000)
  const r3 = await priceRange(page)
  expect(r3.auto).toBe(false)
  expect(r3.from).toBeCloseTo(r1.from, 6)
  expect(r3.to).toBeCloseTo(r1.to, 6)
  // double-click the axis: auto-fit again, and the stored range is dropped
  await page.mouse.dblclick(pr.x + axis + 30, pr.y + 250)
  await page.waitForTimeout(800)
  expect((await priceRange(page)).auto).toBe(true)
  expect(await page.evaluate(() => window.localStorage.getItem('ib-react-price-zoom'))).toBeNull()

  // Flow v3: drag on its axis over the anchored-line slot stretches that slot only
  const fl = await page.getByTestId('flow-v3-panel').locator('.tv-lightweight-charts').boundingBox()
  const fw = await page.evaluate((fn) => { const { flow } = eval(fn); return flow.chart.timeScale().width() }, CHARTS_FN) // eslint-disable-line no-eval
  const m = await slotMargins(page)
  const H = m.height - 28
  const yCvd = fl.y + ((m.cvd.top + (1 - m.cvd.bottom)) / 2) * H
  const f0 = await flowRanges(page)
  await axisDrag(page, fl.x + fw + 25, yCvd, -50)
  const f1 = await flowRanges(page)
  log('item8-flow.json', { f0, f1, zoom: await page.evaluate(() => window.localStorage.getItem('ib-react-flow-v3-zoom')) })
  expect(f1.cvd.to - f1.cvd.from).toBeLessThan(0.8 * (f0.cvd.to - f0.cvd.from)) // stretched
  expect(f1.delta).toEqual(f0.delta)
  expect(f1.right).toEqual(f0.right)
  // compress the delta slot
  const yDelta = fl.y + ((m.delta.top + (1 - m.delta.bottom)) / 2) * H
  await axisDrag(page, fl.x + fw + 25, yDelta, 40)
  const f2 = await flowRanges(page)
  expect(f2.delta.to - f2.delta.from).toBeGreaterThan(1.2 * (f0.delta.to - f0.delta.from))
  // reload keeps both; double-click the anchored-line slot resets only it
  await page.reload()
  await page.getByTestId('flow-v3-panel').waitFor()
  await page.waitForTimeout(3000)
  const f3 = await flowRanges(page)
  expect(f3.cvd.to - f3.cvd.from).toBeLessThan(0.8 * (f0.cvd.to - f0.cvd.from))
  await page.mouse.dblclick(fl.x + fw + 25, yCvd)
  await page.waitForTimeout(800)
  const f4 = await flowRanges(page)
  expect((f4.cvd.to - f4.cvd.from) / (f0.cvd.to - f0.cvd.from)).toBeGreaterThan(0.9)
  expect(f4.delta.to - f4.delta.from).toBeGreaterThan(1.2 * (f0.delta.to - f0.delta.from))
})

test('amendment 1: ⚙ swing size / effort ratio default 8 / 1.5, write the stored v2 setting and feed the legs request', async ({ page }) => {
  await open(page, D29, { storage: EDGES, until: D29_UNTIL })
  const swingReqs = []
  page.on('request', (r) => {
    const u = new URL(r.url())
    if (u.pathname.endsWith('/api/ironbeam/flow') && u.searchParams.get('mode') === 'v2') swingReqs.push(u.searchParams.get('m_swing'))
  })
  await page.getByLabel('Open Flow v3 settings').click()
  await expect(page.getByTestId('flow-v3-swing-size')).toHaveValue('8')
  await expect(page.getByTestId('flow-v3-effort-ratio')).toHaveValue('1.5')
  await page.getByTestId('flow-v3-swing-size').fill('4')
  await page.getByTestId('flow-v3-effort-ratio').fill('2')
  await page.getByRole('button', { name: 'Apply' }).click()
  await page.waitForTimeout(2500)
  const stored = await page.evaluate(() => JSON.parse(window.localStorage.getItem('ib-react-flow-v2-markers')))
  expect(stored.swing).toBe(4)
  expect(stored.effortRatio).toBe(2)
  expect(swingReqs).toContain('4')
  // out-of-range values clamp to v2's ranges; Defaults restores 8 / 1.5
  await page.getByLabel('Open Flow v3 settings').click()
  await expect(page.getByTestId('flow-v3-swing-size')).toHaveValue('4')
  await page.getByTestId('flow-v3-swing-size').fill('99')
  await page.getByTestId('flow-v3-effort-ratio').fill('0.2')
  await page.getByRole('button', { name: 'Apply' }).click()
  const clamped = await page.evaluate(() => JSON.parse(window.localStorage.getItem('ib-react-flow-v2-markers')))
  expect([clamped.swing, clamped.effortRatio]).toEqual([50, 1])
  await page.getByLabel('Open Flow v3 settings').click()
  await page.getByRole('button', { name: 'Defaults' }).click()
  await page.getByRole('button', { name: 'Apply' }).click()
  const def = await page.evaluate(() => JSON.parse(window.localStorage.getItem('ib-react-flow-v2-markers')))
  expect([def.swing, def.effortRatio]).toEqual([8, 1.5])
  log('amendment-1-swing.json', { stored, swingReqs: [...new Set(swingReqs)], clamped, def })
})

test('amendment 2: a box spanning ETH (Ironbeam approx) and RTH (Databento exact) shows "mixed sources" in the popup; an RTH-only box does not', async ({ page }) => {
  const day = '2026-05-01'
  await open(page, day, { storage: EDGES, session: 'FULL' })
  const segs = await page.evaluate(async ({ api, d }) => {
    const r = await fetch(`${api}/api/ironbeam/flow/cells?trade_date=${d}&session=FULL&a_k=3&a_m=3`)
    return (await r.json()).segments.map((s) => [s.name, s.source, s.exact])
  }, { api: API_BASE, d: day })
  // a box that spans the 06:30 PT open (the open often breaks the box, so try a few anchor / end pairs)
  const tried = []
  let box = null
  for (const [a, e] of [['06:20', '06:32'], ['06:25', '06:31'], ['06:15', '06:35'], ['06:28', '06:31'], ['05:50', '06:40']]) {
    await page.keyboard.press('Escape')
    await anchorAt(page, shifted(`${day}T${a}`), shifted(`${day}T08:00`))
    await pickAt(page, 'flow-v3-end-pick', shifted(`${day}T${e}`))
    box = await currentBoxLabel(page)
    tried.push({ a, e, from: box?.readout?.fromSeg, to: box?.readout?.toSeg })
    if (box?.readout?.fromSeg === 'eth' && box?.readout?.toSeg === 'rth') break
  }
  const mixed = page.getByTestId('flow-v3-box-popup-mixed')
  const text = (await mixed.count()) ? await mixed.innerText() : null
  log('amendment-2-mixed.json', { segs, tried, readout: box?.readout, text })
  expect(box?.readout?.fromSeg).toBe('eth')
  expect(box?.readout?.toSeg).toBe('rth')
  expect(text).toBe('mixed sources (Ironbeam approx + Databento)')
  // RTH only (anchor 07:00, end 07:40): no line
  await page.keyboard.press('Escape')
  await anchorAt(page, shifted(`${day}T07:00`), shifted(`${day}T09:00`))
  await pickAt(page, 'flow-v3-end-pick', shifted(`${day}T07:40`))
  await expect(page.getByTestId('flow-v3-box-popup-head')).toBeAttached() // CR-DQ: in the collapsed current section
  await expect(page.getByTestId('flow-v3-box-popup-mixed')).toHaveCount(0)
})
