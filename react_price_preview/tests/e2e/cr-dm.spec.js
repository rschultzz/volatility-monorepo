// CR-DM — Flow v3 volume per minute, rolling net delta in ⚙, live effort label (specs/CR-DM-volume-strip-live-effort.md).
// Read-only. Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_dm_smoke_backend.py   (port 8065, read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5186 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5187
//   hand check apps/web/.venv/bin/python -u scripts/cr_dm_hand_check.py   (writes scripts/logs/cr-dm/smoke/hand_check.json)
// then:       CRBV_BASE_URL=http://localhost:5186 npx playwright test tests/e2e/cr-dm.spec.js   (from react_price_preview/)
// The review replay uses a browser-only fake candidate on 2026-10-29 (list and detail intercepted; B8: nothing written).
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { volumeCap } from '../../src/flowV3.js'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8065'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5187'
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-dm/smoke')
const HAND = path.join(LOGS, 'hand_check.json')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort', 'ib-react-flow-panel-height']
const V3 = { 'ib-react-flow-mode-v3': '1' }
const EFFORT = { ...V3, 'ib-react-flow-v2-effort': '1' }
// ⚙ rolling net delta on (stored as the settings dialog's Apply stores it)
const ROLLING_ON = { ...V3, 'ib-react-flow-v3-settings': JSON.stringify({ rollingMin: 10, k: 3, m: 3, sellLayer: true, buyLayer: true, boxHoldMin: 15, rollingOn: true, liveMinMove: 4, breakHoldMin: 5 }) }
const LIVE_ID = 'cr-dm-live-effort'

const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}
const kfmt = (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`)
// AggressorFlowV2Panel.effortLabel (CR-CD A5), copied: the confirmed labels' text
function effortLabel(leg) {
  const pts = Number(leg.pts)
  const parts = [`${pts >= 0 ? '+' : '-'}${Math.abs(pts).toFixed(1)}`, kfmt(Number(leg.vol)), leg.cpp == null ? '–/pt' : `${kfmt(Number(leg.cpp))}/pt`]
  if (leg.ratio != null) parts.push(`×${Number(leg.ratio).toFixed(1)}`)
  return parts.join(' · ')
}
// PT wall clock of a UTC epoch as a shifted epoch (the charts' times); 2026-10 is PDT.
const shift = (utc) => utc - 7 * 3600

async function open(page, day, { base = '', storage = {}, session = 'RTH' } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values }) => {
    try {
      if (!window.sessionStorage.getItem('crdm-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crdm-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage })
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0&flow_session=${session}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The price chart, its candle series and effort markers; the Flow v3 chart and its series registry.
const CHARTS_FN = `(() => {
  const walk = (el, pick) => {
    const key = Object.keys(el.parentElement).find((k) => k.startsWith('__reactFiber'))
    let f = el.parentElement[key]
    const out = {}
    for (let d = 0; f && d < 60; d += 1, f = f.return) {
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
    if (c.api && typeof c.api.markers === 'function' && c.series) (o.markerApis = o.markerApis || []).push(c.api)
  })
  const flowEl = document.querySelector('[data-testid=flow-v3-panel] .tv-lightweight-charts')
  const flow = flowEl ? walk(flowEl, (c, o) => {
    if (typeof c.timeScale === 'function') o.chart = o.chart || c
    if (c.delta && c.rolling && c.cvd) o.reg = o.reg || c
  }) : {}
  return { priceEl, flowEl, price, flow }
})()`

// Pane layout and the middle / lower series.
const layout = (page) => page.evaluate((fn) => {
  const { flow } = eval(fn) // eslint-disable-line no-eval
  const c = flow.chart
  const r = flow.reg
  const m = (id) => { try { return c.priceScale(id).options().scaleMargins } catch (e) { return null } }
  const h0 = c.options().height - c.timeScale().height()
  const px = (s) => (s ? [+(s.top * h0).toFixed(1), +((1 - s.bottom) * h0).toFixed(1)] : null)
  const onRight = (s) => s && s.options().priceScaleId === 'right'
  const pts = (s) => (s ? s.data().filter((d) => d.value != null).map((d) => [d.time, d.value]) : [])
  return {
    panel: document.querySelector('[data-testid=flow-v3-panel]').getBoundingClientRect().height,
    price: document.querySelector('.react-top-pane').getBoundingClientRect().height,
    pane0: h0,
    margins: { cvd: m('cvd'), right: m('right'), delta: m('delta'), vol: r.vol && !onRight(r.vol) ? m('vol') : null },
    px: { cvd: px(m('cvd')), mid: px(m('right')), vol: r.vol && !onRight(r.vol) ? px(m('vol')) : null, delta: px(m('delta')) },
    mid: onRight(r.vol) ? 'vol' : onRight(r.rolling) ? 'rolling' : null,
    vol: pts(r.vol),
    rolling: pts(r.rolling),
    readout: document.querySelector('[data-testid=flow-v3-readout]')?.textContent,
  }
}, CHARTS_FN)

async function v2Payload(request, day, until = null) {
  const qs = `trade_date=${day}&session=RTH&resample=1m&ema_len=14&mode=v2&effort=1&m_swing=8&m_push=500&m_div=300&m_fade=0.1&m_gap=30`
  const r = await request.get(`${API_BASE}/api/ironbeam/flow?${qs}${until ? `&until=${until}` : ''}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

// Baseline = origin/main after CR-DM (re-pointed in CR-DN amendment 4; the pre-CR-DM comparison is in CR-DM's smoke).
test('default: volume per minute in the middle slot; pane height and slots as origin/main', async ({ browser, request }) => {
  test.setTimeout(300_000)
  const res = {}
  for (const [name, base] of [['branch', ''], ['main', BASELINE]]) {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    await open(page, '2026-10-27', { base, storage: V3 })
    await settle(page)
    res[name] = await layout(page)
    await page.locator('[data-testid=flow-v3-panel]').screenshot({ path: path.join(LOGS, `default-${name}.png`) })
    await ctx.close()
  }
  const { branch, main } = res
  log('default-layout.json', { branch: { ...branch, vol: branch.vol.length, rolling: branch.rolling.length }, main: { ...main, vol: main.vol.length, rolling: main.rolling.length } })
  expect(branch.panel).toBe(220)
  expect(branch.panel).toBe(main.panel)
  expect(branch.price).toBe(main.price)
  expect(branch.margins).toEqual(main.margins)
  expect(branch.px).toEqual(main.px)
  expect(main.mid).toBe('vol')
  expect(branch.mid).toBe('vol')
  expect(branch.rolling).toEqual([]) // off by default (C2 deviation approved)
  expect(branch.readout).toContain('volume per minute')
  // the volume bars are the payload's per-minute `vol` (the same rows the delta bars draw), drawn at most at the
  // session's p98 minute volume (CR-DN item 6; hover keeps the true volume)
  const p = await v2Payload(request, '2026-10-27')
  const rows = p.bars.filter((b) => b.seg === 'rth')
  const cap = volumeCap(rows)
  expect(branch.vol).toEqual(rows.map((b) => [shift(b.time), Math.min(b.vol, cap)]))
})

test('hover a minute: time, volume, delta and delta share', async ({ page, request }) => {
  test.setTimeout(300_000)
  await open(page, '2026-10-27', { storage: V3 })
  await settle(page)
  const p = await v2Payload(request, '2026-10-27')
  const bar = p.bars.find((b) => b.seg === 'rth' && shift(b.time) === Date.UTC(2026, 9, 27, 7, 22) / 1000)
  const pt = await page.evaluate(({ t, fn }) => {
    const { flowEl, flow } = eval(fn) // eslint-disable-line no-eval
    const r = flowEl.getBoundingClientRect()
    return { x: r.x + flow.chart.timeScale().timeToCoordinate(t), y: r.y + 60 }
  }, { t: shift(bar.time), fn: CHARTS_FN })
  await page.mouse.move(pt.x, pt.y)
  const tip = page.getByTestId('flow-v3-tooltip')
  await expect(tip).toBeVisible()
  // at full-day zoom a bar is ~1 px wide, so check the minute the tooltip names against that minute's payload row
  const text = await tip.innerText()
  const [hh, mm] = text.slice(0, 5).split(':').map(Number)
  const row = p.bars.find((b) => b.seg === 'rth' && shift(b.time) === Date.UTC(2026, 9, 27, hh, mm) / 1000)
  const share = row.delta / row.vol
  log('minute-hover.json', { text, row })
  expect(Math.abs(hh * 60 + mm - (7 * 60 + 22))).toBeLessThanOrEqual(1)
  expect(text).toContain(`vol ${kfmt(row.vol)}`)
  expect(text).toContain(`Δ ${row.delta >= 0 ? '+' : '-'}${kfmt(Math.abs(row.delta))}`)
  expect(text).toContain(`share ${share >= 0 ? '+' : ''}${share.toFixed(2)}`)
  expect(text).not.toContain('-min') // no rolling value while rolling is off
})

test('⚙ rolling net delta on: old series back in its slot, volume one slot down, pane grows by one slot', async ({ browser }) => {
  test.setTimeout(300_000)
  // origin/main with rolling on (stored setting): the layout the toggle must reach
  const ctxMain = await browser.newContext()
  const main = await ctxMain.newPage()
  await open(main, '2026-10-27', { base: BASELINE, storage: ROLLING_ON })
  await settle(main)
  const mainL = await layout(main)
  await ctxMain.close()

  const ctx = await browser.newContext()
  const page = await ctx.newPage()
  await open(page, '2026-10-27', { storage: V3 })
  await settle(page)
  const off = await layout(page)
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  const box = page.getByTestId('flow-v3-rolling-toggle')
  await expect(box).not.toBeChecked()
  await expect(page.getByTestId('flow-v3-live-min-move')).toHaveValue('4')
  await box.check()
  await page.getByRole('button', { name: 'Apply' }).click()
  await expect.poll(async () => (await layout(page)).panel, { timeout: 20_000 }).toBe(271)
  await page.waitForTimeout(800)
  const on = await layout(page)
  await page.locator('[data-testid=flow-v3-panel]').screenshot({ path: path.join(LOGS, 'rolling-on.png') })
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  await page.getByTestId('flow-v3-rolling-toggle').uncheck()
  await page.getByRole('button', { name: 'Apply' }).click()
  await expect.poll(async () => (await layout(page)).panel, { timeout: 20_000 }).toBe(220)
  await page.waitForTimeout(800)
  const off2 = await layout(page)
  await ctx.close()
  const slim = (l) => ({ ...l, vol: l.vol.length, rolling: l.rolling.length })
  log('rolling-toggle.json', { main: slim(mainL), off: slim(off), on: slim(on), off2: slim(off2) })

  expect(on.panel - off.panel).toBe(51) // one slot (0.32 × pane 0 of 160 px)
  expect(off.price - on.price).toBe(51)
  expect(on.mid).toBe('rolling')
  expect(on.panel).toBe(mainL.panel)
  expect(on.mid).toBe(mainL.mid)
  expect(on.rolling).toEqual(mainL.rolling) // same series as origin/main with rolling on
  expect(on.vol).toEqual(off.vol)
  expect(on.margins).toEqual(mainL.margins)
  const near = (a, b) => expect(Math.abs(a - b)).toBeLessThanOrEqual(0.6)
  const len = (s) => s[1] - s[0]
  near(len(on.px.cvd), len(mainL.px.cvd))
  near(len(on.px.mid), len(mainL.px.mid))
  near(len(on.px.delta), len(mainL.px.delta))
  near(len(on.px.vol), len(on.px.mid)) // volume gets a slot the same height as rolling
  expect(on.px.vol[0]).toBeGreaterThan(on.px.mid[1])
  expect(on.px.delta[0]).toBeGreaterThan(on.px.vol[1])
  expect(on.readout).toContain('rolling 10 min')
  expect({ ...off2, vol: 0 }).toEqual({ ...off, vol: 0 })
})

// ── Item 3: review replay on 2026-10-29 through the REVIEW panel (a fake candidate per replay time).
function fakeCandidate(until) {
  const id = `CR-DM:smoke:${until}`
  return {
    id, cr_id: 'CR-DM', set_name: 'cr-dm:smoke', cell: 'smoke', direction: 'short', session_date: '2026-10-29',
    entry_ts_pt: `2026-10-29T${until}:00`, confirm_ts_pt: `2026-10-29T${until}:00`, entry_price: null, extreme_price: null,
    level_price: null, stop_price: null, level_gross_bn: null, level_net_bn: null, label: null, meta: {},
    n_reviews: 0, reviewed: false, revealed: false, revealed_at: null, retest_ts_pt: null, test_ts_pt: null,
  }
}

async function openReplay(page, until, flowLog) {
  const c = fakeCandidate(until)
  await page.route('**/api/setup-review/candidates?*', async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    body.sets = [...(body.sets || []), { cr_id: c.cr_id, set_name: c.set_name, loaded_at: '2099-01-01T00:00:00+00:00', n: 1 }]
    body.candidates = [c]
    await route.fulfill({ response: r, json: body })
  })
  await page.route(`**/api/setup-review/candidates/${encodeURIComponent(c.id)}`, (route) => route.fulfill({ json: { ok: true, candidate: { ...c, reviews: [] } } }))
  await page.route('**/api/setup-review/candidates/*/review', (route) => route.abort())
  await page.route('**/api/setup-review/candidates/*/reveal', (route) => route.abort())
  page.on('response', async (r) => {
    const u = new URL(r.url())
    if (u.pathname.endsWith('/api/ironbeam/flow') && u.searchParams.get('mode') === 'v2' && u.searchParams.get('trade_date') === '2026-10-29') {
      try { flowLog.push({ until: u.searchParams.get('until'), body: await r.json() }) } catch (e) { /* aborted */ }
    }
  })
  await open(page, '2026-10-29', { storage: EFFORT })
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.locator(`[data-testid="review-row"][data-candidate-id="${c.id}"]`).click({ timeout: 60_000 })
  await expect(page.getByTestId('review-active')).toContainText(c.id, { timeout: 60_000 })
  // a live date never goes network-idle: wait for the replay's v2 response, then let the chart draw
  await expect.poll(() => flowLog.filter((f) => f.until && f.body?.bars).length, { timeout: 120_000 }).toBeGreaterThan(0)
  await page.waitForTimeout(3000)
}

const effortMarkers = (page) => page.evaluate((fn) => {
  const { price } = eval(fn) // eslint-disable-line no-eval
  const api = (price.markerApis || []).find((a) => a.markers().some((m) => /pts|\/pt/.test(m.text || '')))
  return api ? api.markers().map((m) => ({ id: m.id ?? null, time: m.time, text: m.text, color: m.color, position: m.position })) : []
}, CHARTS_FN)

test('review replay 2026-10-29: the provisional label is the hand computation, uses no bar after until, and hands off', async ({ browser }) => {
  test.setTimeout(900_000)
  const hand = JSON.parse(fs.readFileSync(HAND, 'utf8'))
  expect(hand.ok).toBe(true)
  const out = []
  for (const cs of hand.cases) {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    const flowLog = []
    await openReplay(page, cs.until, flowLog)
    const u = Date.UTC(2026, 9, 29, ...cs.until.split(':').map(Number)) / 1000 + 7 * 3600 // until as a UTC epoch
    // responses after the replay began (the page first loads the live session without until)
    const first = flowLog.findIndex((f) => f.until)
    const seen = flowLog.slice(first).filter((f) => f.body?.bars)
    await page.waitForTimeout(16_000) // one live-poll period (15 s): no un-capped refresh may arrive in replay
    expect(flowLog.slice(first).filter((f) => !f.until)).toEqual([])
    const markers = await effortMarkers(page)
    const live = markers.find((m) => m.id === LIVE_ID) || null
    out.push({ until: cs.until, requests: seen.map((f) => f.until), live, n: markers.length })
    await page.screenshot({ path: path.join(LOGS, `replay-${cs.until.replace(':', '')}.png`) })
    // the App asked with until and every response it got stops before it
    expect(seen.length).toBeGreaterThan(0)
    for (const f of seen) {
      expect(f.until).toBe(`2026-10-29T${cs.until}:00`)
      expect(f.body.bars.every((b) => b.time < u)).toBe(true)
      expect(f.body.live_leg.last_time).toBeLessThan(u)
    }
    if (cs.label == null) {
      expect(live).toBeNull()
    } else {
      expect(live?.text).toBe(cs.label)
      expect(live.time).toBe(shift(cs.hand.end_time)) // at the running extreme
      expect(live.position).toBe(cs.hand.dir === 'up' ? 'aboveBar' : 'belowBar')
      expect(markers.filter((m) => m.time > shift(u - 60))).toEqual([]) // nothing drawn after until
    }
    // the confirmed labels are exactly the payload's legs, unchanged
    const legs = seen[seen.length - 1].body.legs
    expect(markers.filter((m) => m.id !== LIVE_ID).map((m) => m.text).sort()).toEqual(legs.map(effortLabel).sort())
    if (cs.until === hand.handoff.after) {
      const leg = hand.handoff.leg
      expect(markers.some((m) => m.id !== LIVE_ID && m.time === shift(leg.end_time) && m.text === effortLabel(leg))).toBe(true)
      expect(live?.time).not.toBe(shift(leg.end_time))
    }
    if (cs.until === hand.handoff.confirm) {
      // hover the provisional label (zoomed to ±30 bars, as a reader would): the numbers and "provisional — leg not confirmed"
      await page.evaluate(({ t, fn }) => {
        const { price } = eval(fn) // eslint-disable-line no-eval
        const i = price.series.data().findIndex((x) => x.time === t)
        price.chart.timeScale().setVisibleLogicalRange({ from: i - 30, to: i + 30 })
      }, { t: live.time, fn: CHARTS_FN })
      await page.waitForTimeout(800)
      const p = await page.evaluate(({ t, fn }) => {
        const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
        const r = priceEl.getBoundingClientRect()
        const bar = price.series.data().find((x) => x.time === t)
        return { x: r.x + price.chart.timeScale().timeToCoordinate(t), yHigh: r.y + price.series.priceToCoordinate(bar.high), yLow: r.y + price.series.priceToCoordinate(bar.low) }
      }, { t: live.time, fn: CHARTS_FN })
      const tip = page.getByTestId('live-effort-tooltip')
      for (let dy = 4; dy <= 40 && !(await tip.isVisible()); dy += 2) {
        await page.mouse.move(p.x, live.position === 'aboveBar' ? p.yHigh - dy : p.yLow + dy)
        await page.waitForTimeout(120)
      }
      await expect(tip).toBeVisible()
      const text = await tip.innerText()
      await page.screenshot({ path: path.join(LOGS, 'live-hover.png') })
      out[out.length - 1].hover = text
      const h = cs.hand
      expect(text).toContain('provisional — leg not confirmed')
      expect(text).toContain(`${h.pts >= 0 ? '+' : ''}${h.pts.toFixed(2)} pts so far`)
      expect(text).toContain(`volume ${kfmt(h.vol)}`)
      expect(text).toContain(`${kfmt(h.cpp)}/pt`)
      expect(text).toContain(`${h.ratio.toFixed(2)}×`)
      expect(text).toContain(`${h.minutes} min so far`)
    }
    await ctx.close()
  }
  log('replay.json', out)
})

test('Flow v2 with Effort on draws no live label (Flow v3 only)', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, '2026-10-27', { storage: { 'ib-react-flow-mode': 'v2', 'ib-react-flow-v2-effort': '1' } })
  await settle(page)
  const markers = await effortMarkers(page)
  expect(markers.length).toBeGreaterThan(0)
  expect(markers.some((m) => m.id === LIVE_ID)).toBe(false)
})
