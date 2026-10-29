// CR-DI — Edge effort strip and box restart on a break (specs/CR-DI-edge-effort-strip.md). Read-only.
//
// Servers (started outside Playwright), as cr-dg.spec.js:
//   backend    apps/web/.venv/bin/python -u scripts/cr_di_smoke_backend.py   (port 8064, read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5183 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5184
// then:       CRBV_BASE_URL=http://localhost:5183 CRDF_BASELINE_URL=http://localhost:5184 npx playwright test cr-di
//
// Expected values come from the shipped flowV3.js run in Node on the cells payload the page itself loaded.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { dirArrow, drawnBoxes, edgeEffortSeries, edgeReadout, edgeText, edgeVisits, flattenBars, kfmt, makeShifter } from '../../src/flowV3.js'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8064'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5184'
// CRDI_SIDE_A=<baseline url> runs the pixel test main-vs-main: how much the live session differs between two loads
const SIDE_A = process.env.CRDI_SIDE_A || ''
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-di/smoke')
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-di')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers', 'ib-react-flow-v3-edge-effort']
const V3 = { 'ib-react-flow-mode-v3': '1' }
const EDGES = { 'ib-react-flow-v3-layers': JSON.stringify({ levels: false, edges: true, profile: false, heat: false, marks: false }) }
const STRIP = { ...V3, ...EDGES, 'ib-react-flow-v3-edge-effort': '1' }
const TODAY = '2026-10-28'
const MAGENTA = [232, 121, 249]

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
      if (!window.sessionStorage.getItem('crdi-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crdi-init', '1')
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
  await page.goto(`${base}/?trade_date=${day}&interval=1min&days_either_side=0&flow_session=${session}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The price chart (first chart) and its candle series; the Flow v3 pane's chart and its series registry.
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

// Arm 📍 and click the candle at `shiftedT` on the price chart.
async function anchorAt(page, shiftedT) {
  await frame(page, shiftedT - 30 * 60, shiftedT + 30 * 60)
  await page.getByTestId('flow-v3-anchor-pick').click()
  const p = await page.evaluate(({ t, fn }) => {
    const { priceEl, price } = eval(fn) // eslint-disable-line no-eval
    const rect = priceEl.getBoundingClientRect()
    const bar = price.series.data().find((x) => x.time === t)
    return { x: rect.x + price.chart.timeScale().timeToCoordinate(t), y: rect.y + price.series.priceToCoordinate(bar.close) }
  }, { t: shiftedT, fn: CHARTS_FN })
  await page.mouse.click(p.x, p.y)
  await page.waitForTimeout(800)
}

async function cellsOnLoad(page, run) {
  const resp = page.waitForResponse((r) => r.url().includes('/flow/cells') && r.ok(), { timeout: 180_000 })
  await run()
  return (await resp).json()
}

// What the strip draws: per-candle columns, labels, pane count.
async function stripState(page) {
  return page.evaluate((fn) => {
    const { flow } = eval(fn) // eslint-disable-line no-eval
    const d = flow.reg?.stripPrim?._data
    return {
      panes: flow.chart.panes().length,
      cols: d ? [...d.cols.entries()].map(([i, c]) => [d.times[i], ...c]) : null,
      labels: d?.labels ? { top: d.labels.top.map((v) => v.text), bottom: d.labels.bottom.map((v) => v.text) } : null,
    }
  }, CHARTS_FN)
}

// Composited PNG-free fingerprint of a chart element (all its canvases in order) for pixel identity.
async function canvasHash(page, selector) {
  return page.evaluate((sel) => {
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
    return out.toDataURL()
  }, selector)
}

test('chip off: price chart and Flow v3 pane pixel-identical to origin/main (2026-10-27, 2026-10-28)', async ({ browser }) => {
  test.setTimeout(600_000)
  const shot = async (base, day, storage, anchorOpen) => {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    await open(page, day, { base, storage })
    await settle(page)
    if (anchorOpen) {
      await page.getByTestId('flow-v3-anchor-open').click()
      await page.waitForTimeout(2500)
    }
    await frame(page, shifted(`${day}T06:30`), shifted(`${day}T13:00`))
    await page.mouse.move(5, 5)
    await page.waitForTimeout(600)
    const res = {
      price: await canvasHash(page, '.react-top-pane .tv-lightweight-charts'),
      flow: await canvasHash(page, '[data-testid=flow-v3-panel] .flow-stage'),
      readout: anchorOpen ? await page.getByTestId('flow-v3-edge-readout').innerText() : null,
      // CR-DN item 6 changes the volume bars on purpose: the pane's other series are compared by data
      series: await page.evaluate(() => {
        const el = document.querySelector('[data-testid=flow-v3-panel] .tv-lightweight-charts')
        const key = Object.keys(el.parentElement).find((k) => k.startsWith('__reactFiber'))
        let f = el.parentElement[key]
        for (let d = 0; f && d < 40; d += 1, f = f.return) {
          for (let h = f.memoizedState; h; h = h.next) {
            const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
            if (c && c.delta && c.rolling && c.cvd) return JSON.stringify([c.cvd.data(), c.delta.data(), c.rolling.data()])
          }
        }
        return null
      }),
      png: await page.locator('.react-top-pane').screenshot(),
    }
    await ctx.close()
    return res
  }
  const rows = []
  for (const day of ['2026-10-27', TODAY]) {
    for (const [name, storage, anchorOpen] of [['v3, Edges off', V3, false], ['v3, Edges on, anchor session open', { ...V3, ...EDGES }, true]]) {
      let a
      let b
      let tries = 0
      // the live session gains a bar a minute: load both builds at once and retry if a minute boundary split them
      do {
        ;[a, b] = await Promise.all([shot(SIDE_A, day, storage, anchorOpen), shot(BASELINE, day, storage, anchorOpen)])
        tries += 1
      } while (day === TODAY && (a.price !== b.price || a.series !== b.series) && tries < 6)
      rows.push({ day, name, tries, price_equal: a.price === b.price, flow_equal: a.flow === b.flow, flow_series_equal: a.series === b.series, branch_readout: a.readout, main_readout: b.readout })
      if (a.price !== b.price || a.flow !== b.flow) {
        // kept for diagnosis: both composites (data URLs -> png) in the logs
        fs.mkdirSync(LOGS, { recursive: true })
        const slug = `${day}-${name.replace(/[^a-z0-9]+/gi, '-')}`
        for (const [who, r] of [['branch', a], ['main', b]]) {
          for (const k of ['price', 'flow']) fs.writeFileSync(path.join(LOGS, `diff-${slug}-${k}-${who}.png`), Buffer.from(r[k].split(',')[1], 'base64'))
        }
      }
      if (name === 'v3, Edges off') {
        fs.mkdirSync(ASSETS, { recursive: true })
        fs.writeFileSync(path.join(ASSETS, `chip-off-${day}.png`), a.png)
      }
    }
  }
  log('pixel-identity.json', rows)
  for (const r of rows.filter((x) => x.name === 'v3, Edges off')) {
    expect(r.price_equal, `${r.day} ${r.name}: price chart = origin/main`).toBeTruthy()
    expect(r.flow_series_equal, `${r.day} ${r.name}: Flow v3 CVD / delta / rolling = origin/main (volume bars: CR-DN item 6)`).toBeTruthy()
  }
})

const STEP0_0809 = {
  top: ['2.5k / 2.3k', '8.6k / 7.8k', '4.3k / 4.1k', '1.0k / 881'],
  bottom: ['3.3k / 3.2k', '1.7k / 1.6k', '1.9k / 1.2k', '9.8k / 9.5k'],
}

for (const until of [`${TODAY}T07:40`, `${TODAY}T08:09`, null]) {
  test(`2026-10-28 anchor 07:00, chip on, ${until ? `replay until ${until.slice(11)}` : 'to the end'}: header = current box = strip sums; labels`, async ({ page }) => {
    test.setTimeout(300_000)
    const cells = await cellsOnLoad(page, async () => { await open(page, TODAY, { storage: STRIP, until }) })
    await settle(page)
    const anchor = shifted(`${TODAY}T07:00`)
    await anchorAt(page, anchor)
    const r = edgeReadout(cells, flattenBars(cells), anchor, ptShift, 15)
    const want = edgeText(r, pt)
    await expect.poll(async () => page.getByTestId('flow-v3-edge-readout').innerText(), { timeout: 20_000 }).toBe(want)
    await expect(page.getByTestId('flow-v3-edge-effort-toggle')).toHaveAttribute('aria-pressed', 'true')
    await expect(page.getByTestId('flow-v3-edge-effort-toggle')).toBeEnabled()
    const st = await stripState(page)
    const from = ptShift(r.from)
    const drawnCur = st.cols.filter((c) => c[0] >= from)
    const sum = (k) => drawnCur.reduce((a, c) => a + c[k], 0)
    const series = edgeEffortSeries(r)
    const v = edgeVisits(series)
    const labels = { top: v.top.map((x) => `${kfmt(x.main)} / ${kfmt(x.opp)}`), bottom: v.bottom.map((x) => `${kfmt(x.main)} / ${kfmt(x.opp)}`) }
    const res = {
      until, header: want, panes: st.panes,
      current: { from: pt(r.from), top: r.top.price, bottom: r.bottom.price, buyers: r.top.contracts, sellers: r.bottom.contracts },
      strip_current_top_buy: sum(1), strip_current_bottom_sell: sum(3),
      frozen: r.frozen.map((f) => ({ from: pt(f.from), broke: pt(f.breakAt), dir: f.dir, bottom: f.bottom.price, top: f.top.price,
        top_buy: f.top.contracts, top_sell: f.top.opposite, bottom_sell: f.bottom.contracts, bottom_buy: f.bottom.opposite, held: f.heldMin })),
      labels_drawn: st.labels, labels_expected: labels,
    }
    expect(st.panes).toBe(2)
    expect(sum(1), 'strip top buys over the current box = header buyers').toBe(r.top.contracts)
    expect(sum(3), 'strip bottom sells over the current box = header sellers').toBe(r.bottom.contracts)
    expect(st.labels).toEqual(labels)
    if (until === `${TODAY}T08:09`) {
      expect(r.frozen.length).toBe(0)
      expect([r.top.contracts, r.bottom.contracts]).toEqual([16446, 16613]) // Step 0b
      expect(st.labels).toEqual(STEP0_0809)
      // hover the 07:14 minute in the strip
      const h = await page.evaluate(({ t, fn }) => {
        const { flowEl, flow } = eval(fn) // eslint-disable-line no-eval
        const rect = flowEl.getBoundingClientRect()
        return { x: rect.x + flow.chart.timeScale().timeToCoordinate(t), y: rect.y + flow.chart.paneSize(0).height + 1 + 20 }
      }, { t: shifted(`${TODAY}T07:14`), fn: CHARTS_FN })
      await page.mouse.move(h.x, h.y)
      const tip = page.getByTestId('flow-v3-edge-effort-tooltip')
      await expect(tip).toContainText('07:14 · edge effort')
      await expect(tip).toContainText('Top 7769.25: buyers 2.1k / sellers 1.5k / net +519')
      await expect(tip).toContainText('visit 07:13–07:24: so far 2.6k buyers / 2.0k sellers of 8.6k / 7.8k')
      await expect(tip).toContainText('Bottom 7762.00: sellers 0 / buyers 0 / net +0')
      res.hover_0714 = await tip.innerText()
      fs.mkdirSync(ASSETS, { recursive: true })
      await frame(page, shifted(`${TODAY}T06:55`), shifted(`${TODAY}T08:10`))
      await page.mouse.move(h.x, h.y)
      await page.waitForTimeout(400)
      fs.writeFileSync(path.join(ASSETS, 'strip-0809-anchor-0700.png'), await page.screenshot())
    }
    if (until == null) {
      // amendment 9 at N = 15: frozen 07:00–08:16 (7762–7769.25 ↑), frozen 08:16–08:39 (↑), current from 08:39 or later
      expect(r.frozen.length).toBeGreaterThanOrEqual(2)
      expect(res.frozen[0]).toMatchObject({ from: '07:00', broke: '08:16', dir: 'up', bottom: 7762, top: 7769.25 })
      expect(res.frozen[1]).toMatchObject({ from: '08:16', broke: '08:39', dir: 'up' })
      expect(want).toMatch(/^Box from \d{2}:\d{2} [↑↓] · /)
      // the oldest drawn frozen box (only the last 4 are drawn: on a live day later breaks push the 07:00 box out)
      // is drawn faded on the price chart, and its hover gives its stats and how many older boxes were dropped
      const { boxes, dropped } = drawnBoxes(r)
      const fb = boxes[0]
      res.drawn_frozen = boxes.length - 1
      res.dropped = dropped
      const span = (fb.breakAt - fb.from) / 60
      await frame(page, ptShift(fb.from) - 10 * 60, ptShift(fb.breakAt) + 10 * 60)
      const probe = await page.evaluate(({ a, b, price, fn }) => {
        const { priceEl, price: pc } = eval(fn) // eslint-disable-line no-eval
        const rect = priceEl.getBoundingClientRect()
        const x0 = pc.chart.timeScale().timeToCoordinate(a)
        const x1 = pc.chart.timeScale().timeToCoordinate(b)
        const y = pc.series.priceToCoordinate(price)
        const dpr = window.devicePixelRatio || 1
        const out = document.createElement('canvas')
        out.width = Math.round(rect.width * dpr)
        out.height = Math.round(rect.height * dpr)
        const ctx = out.getContext('2d')
        for (const c of priceEl.querySelectorAll('canvas')) {
          const r = c.getBoundingClientRect()
          if (r.width && r.height) ctx.drawImage(c, (r.x - rect.x) * dpr, (r.y - rect.y) * dpr, r.width * dpr, r.height * dpr)
        }
        const px = []
        for (let x = x0 + 4; x < x1 - 4; x += 1) px.push([...ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data].slice(0, 3))
        // hover where the dashed line is not under a candle: the first sampled x with a magenta-ish pixel
        let hx = (x0 + x1) / 2
        for (let k = 0; k < px.length; k += 1) {
          const [rr, g, bb] = px[k]
          if (rr - g > 25 && bb - g > 25) { hx = x0 + 4 + k; break }
        }
        return { x: rect.x + hx, y: rect.y + y, px }
      }, { a: ptShift(fb.from), b: ptShift(fb.breakAt), price: fb.top.price, fn: CHARTS_FN })
      // faded magenta over the dark background: red and blue clearly above green
      const magentaish = probe.px.filter(([rr, g, bb]) => rr - g > 25 && bb - g > 25).length
      res.frozen_line_magenta_px = `${magentaish} / ${probe.px.length} (box ${pt(fb.from)}–${pt(fb.breakAt)} top ${fb.top.price}, ${span} min)`
      if (!(magentaish > 5)) {
        log('probe-debug.json', { y: probe.y, x: probe.x, px: probe.px.slice(0, 60) })
        fs.writeFileSync(path.join(LOGS, 'probe-debug.png'), await page.screenshot())
      }
      expect(magentaish, `faded magenta dashes on the oldest drawn frozen top ${fb.top.price}`).toBeGreaterThan(5)
      await page.mouse.move(probe.x, probe.y)
      const tip = page.getByTestId('flow-v3-box-tooltip')
      // CR-DN item 3b: "how it ended" reads "broke ↑/↓ hh:mm"
      await expect(tip).toContainText(`Box ${pt(fb.from)}–${pt(fb.breakAt)} · broke ${dirArrow(fb.dir)} ${pt(fb.breakAt)}`)
      await expect(tip).toContainText(`Top ${fb.top.price.toFixed(2)}: buyers ${kfmt(fb.top.contracts)} / sellers ${kfmt(fb.top.opposite)}`)
      await expect(tip).toContainText(`Bottom ${fb.bottom.price.toFixed(2)}: sellers ${kfmt(fb.bottom.contracts)} / buyers ${kfmt(fb.bottom.opposite)}`)
      if (dropped) await expect(tip).toContainText(`${dropped} older box${dropped === 1 ? '' : 'es'} not drawn`)
      res.box_hover = await tip.innerText()
      fs.mkdirSync(ASSETS, { recursive: true })
      fs.writeFileSync(path.join(ASSETS, 'boxes-end-anchor-0700.png'), await page.screenshot())
    }
    log(`sums-${until ? until.slice(11).replace(':', '') : 'end'}.json`, res)
  })
}

test('approximate-data session: no strip, chip disabled with the reason', async ({ page }) => {
  test.setTimeout(300_000)
  const day = process.env.CRDI_APPROX_DATE || '2026-10-23'
  const cells = await cellsOnLoad(page, async () => { await open(page, day, { storage: STRIP }) })
  await settle(page)
  expect(cells.segments.some((s) => !s.exact), `${day} has approximate cells`).toBeTruthy()
  await page.getByTestId('flow-v3-anchor-open').click()
  await page.waitForTimeout(2500)
  const chip = page.getByTestId('flow-v3-edge-effort-toggle')
  await expect(chip).toBeDisabled()
  await expect(chip).toHaveAttribute('title', /exact data only.*approximate \(1-s\)/)
  const st = await stripState(page)
  expect(st.panes).toBe(1)
  log('approximate.json', { day, title: await chip.getAttribute('title'), panes: st.panes, readout: await page.getByTestId('flow-v3-edge-readout').innerText() })
})

test('chip gating and the ⚙ box hold setting', async ({ page }) => {
  test.setTimeout(300_000)
  await open(page, '2026-10-27', { storage: V3 })
  await settle(page)
  const chip = page.getByTestId('flow-v3-edge-effort-toggle')
  await expect(chip).toHaveText('Edge effort off') // off by default (C2)
  await expect(chip).toBeDisabled()
  await expect(chip).toHaveAttribute('title', 'Edge effort needs Edges on')
  await page.getByTestId('flow-v3-edges-toggle').click()
  await expect(chip).toHaveAttribute('title', /needs an anchor/)
  await page.getByTestId('flow-v3-anchor-open').click()
  await expect(chip).toBeEnabled({ timeout: 30_000 })
  // amendment 2: the pane grows by the strip's 64 px and pane 0 (CVD / rolling / delta) keeps its height
  const sizes = () => page.evaluate((fn) => {
    const { flow } = eval(fn) // eslint-disable-line no-eval
    const panel = document.querySelector('[data-testid=flow-v3-panel]').getBoundingClientRect().height
    const price = document.querySelector('.react-top-pane').getBoundingClientRect().height
    return { panel, price, pane0: flow.chart.panes()[0].getHeight(), pane1: flow.chart.panes()[1]?.getHeight() ?? null, panes: flow.chart.panes().length }
  }, CHARTS_FN)
  const off1 = await sizes()
  await chip.click()
  await expect(chip).toHaveText('Edge effort on')
  await expect.poll(async () => (await sizes()).panes, { timeout: 20_000 }).toBe(2)
  await page.waitForTimeout(800)
  const on = await sizes()
  expect(await page.evaluate(() => localStorage.getItem('ib-react-flow-v3-edge-effort'))).toBe('1')
  await expect.poll(async () => (await stripState(page)).panes, { timeout: 20_000 }).toBe(2)
  await chip.click()
  await expect.poll(async () => (await stripState(page)).panes).toBe(1)
  await page.waitForTimeout(800)
  const off2 = await sizes()
  log('pane-growth.json', { off1, on, off2 })
  expect(on.panel - off1.panel).toBe(64)
  expect(off1.price - on.price).toBe(64)
  expect(on.pane0).toBe(off1.pane0)
  expect(on.pane1).toBe(63) // + the 1-px separator = 64
  expect(off2).toEqual(off1)
  expect(await page.evaluate(() => localStorage.getItem('ib-react-flow-v3-edge-effort'))).toBe(null)
  await page.getByRole('button', { name: 'Open Flow v3 settings' }).click()
  await expect(page.getByTestId('flow-v3-box-hold')).toHaveValue('15')
})
