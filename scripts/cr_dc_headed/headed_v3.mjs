// CR-DC (from CR-CH's scripts/cr_ch_repro/headed.mjs): headed real Google Chrome, the full Dash page
// (scripts/cr_dc_headed/dash_app.py, read-only), the same seeded hard zoom / pan / axis-drag sequence,
// with Flow v3 + Heatmap + Profile + Absorbed on. node headed_v3.mjs <cell> [durationSec]
// cells: v3all | v2. Records crashes, page errors, CDP heap metrics each second and the JS heap after a
// forced GC every 30 s (heap growth check). Results in scripts/logs/cr_dc/.
import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

const require = createRequire('/Users/ryan/code/volatility-monorepo/.claude/worktrees/cr-dc/react_price_preview/package.json')
const { chromium } = require('@playwright/test')
const DASH = process.env.CRCH_DASH || 'http://127.0.0.1:8072'
const LIVE = process.env.CRCH_LIVE_DATE || '2026-10-28'
const OUT = '/Users/ryan/code/volatility-monorepo/.claude/worktrees/cr-dc/scripts/logs/cr_dc'
const cell = process.argv[2] || 'v3all'
const DURATION = Number(process.argv[3] || 600)
const STORAGE = {
  v3all: { 'ib-react-flow-mode': 'current', 'ib-react-flow-mode-v3': '1', 'ib-react-flow-v3-layers': JSON.stringify({ heat: true, profile: true, marks: true }) },
  v2: { 'ib-react-flow-mode': 'v2' },
}[cell.split('@')[0]]
Object.assign(STORAGE, JSON.parse(process.env.CRCH_EXTRA_STORAGE || '{}'))

let seed = 20261025
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 }
const t0 = Date.now()
const ts = () => `${((Date.now() - t0) / 1000).toFixed(1)}s ${new Date().toISOString()}`
const events = []
const note = (kind, detail = '') => { const e = { t: ts(), kind, detail: String(detail).slice(0, 300) }; events.push(e); console.log(`${e.t} ${kind} ${e.detail}`) }

// own profile dir per run, so Crashpad dumps from a reproduced crash are kept (CRCH_UDD prints it)
const UDD = fs.mkdtempSync(path.join(process.env.CRCH_UDD_ROOT || (fs.mkdirSync('/private/tmp/claude-501/-Users-ryan--claude-spares-spare-Rp2S12/c8ba578b-37f0-4ab3-9d11-2afc95bf47e6/scratchpad/udd', { recursive: true }), '/private/tmp/claude-501/-Users-ryan--claude-spares-spare-Rp2S12/c8ba578b-37f0-4ab3-9d11-2afc95bf47e6/scratchpad/udd'), `${cell}-`))
// CRCH_CHANNEL=chromium → Playwright's bundled (non-official) Chromium; CRCH_ARGS = extra launch flags (space-separated);
// CRCH_LOGFILE → --enable-logging --log-file (CHECK messages land there in non-official builds)
const CHANNEL = process.env.CRCH_CHANNEL || 'chrome'
const EXTRA_ARGS = (process.env.CRCH_ARGS || '').split(' ').filter(Boolean)
const LOG_ARGS = process.env.CRCH_LOGFILE ? ['--enable-logging', `--log-file=${process.env.CRCH_LOGFILE}`, '--v=0'] : []
const ctx = await chromium.launchPersistentContext(UDD, { ...(CHANNEL === 'chromium' ? {} : { channel: CHANNEL }), headless: false, viewport: null,
  args: ['--window-size=1728,1080', `--window-position=${process.env.CRCH_WINPOS || '0,0'}`, '--disable-backgrounding-occluded-windows', ...EXTRA_ARGS, ...LOG_ARGS] })
note('launch', JSON.stringify({ channel: CHANNEL, args: [...EXTRA_ARGS, ...LOG_ARGS] }))
const browser = { close: () => ctx.close() }
ctx.on('close', () => note('context-closed'))
note('udd', UDD)
await ctx.addInitScript((s) => {
  try {
    if (!sessionStorage.getItem('crch-init')) {
      for (const [k, v] of Object.entries(s)) localStorage.setItem(k, v)
      sessionStorage.setItem('crch-init', '1')
    }
  } catch (e) { /* */ }
}, STORAGE)
// read-only: refuse setup-review / trade-log / signal-label writes
await ctx.route('**/api/setup-review/**', (r) => (r.request().method() === 'GET' ? r.continue() : r.abort()))
await ctx.route('**/api/trade-log/trades/**', (r) => r.abort())
await ctx.route('**/api/backtests-v2/signals/**', (r) => (r.request().url().includes('/scan') ? r.fulfill({ json: { ok: true, signals: [] } }) : r.abort()))

const page = ctx.pages()[0] || await ctx.newPage()
let crashed = false
const recent = []
page.on('crash', () => {
  crashed = true; note('PAGE-CRASH', JSON.stringify(recent.slice(-15)))
  // a crashed page can leave pending Playwright calls unresolved: write the result and exit
  try { flushCounts() } catch { /* */ }
  setTimeout(() => { fs.writeFileSync(path.join(OUT, `headed-${cell}.json`), JSON.stringify({ cell, crashed: true, events }, null, 1)); note('end', 'crashed=true'); process.exit(0) }, 4000)
})
page.on('pageerror', (e) => note('pageerror', e.message))
page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource|SignalPanel/.test(m.text())) note('console-error', m.text()) })

await page.goto(`${DASH}/?tab=price-chart`)
await page.waitForFunction(() => !!window.dash_clientside?.set_props && !!document.getElementById('ib-react-preview-frame'), null, { timeout: 120_000 })
const setDate = async (d) => { await page.evaluate((x) => window.dash_clientside.set_props('trade-date', { date: x }), d); note('date', d) }
await setDate(LIVE)

const chartFrame = async () => {
  for (let i = 0; i < 240; i += 1) {
    const f = page.frames().find((x) => x.url().includes('/react-preview?') && x.url().includes('trade_date='))
    if (f) {
      try {
        if (await f.evaluate(() => !!document.querySelector('.tv-lightweight-charts') && !document.querySelector('.status-card'))) return f
      } catch { /* navigating */ }
    }
    await page.waitForTimeout(500)
  }
  throw new Error('chart frame not ready')
}
const effortCount = (f) => f.evaluate(() => {
  const chartEl = document.querySelector('.tv-lightweight-charts')
  if (!chartEl) return -1
  const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
  let n = 0
  let fib = chartEl.parentElement[key]
  for (let d = 0; fib && d < 40; d += 1, fib = fib.return) {
    for (let h = fib.memoizedState; h && typeof h === 'object' && 'next' in h; h = h.next) {
      const v = h.memoizedState
      const c = v && typeof v === 'object' && 'current' in v ? v.current : null
      if (c && c.api && typeof c.api.markers === 'function') for (const m of c.api.markers()) if (String(m.text).includes('/pt')) n += 1
    }
  }
  return n
}).catch(() => -2)

let frame = await chartFrame()
const cdpSamples = []
let countsSnap = null
const flushCounts = () => fs.writeFileSync(path.join(OUT, `heap-${cell}.json`), JSON.stringify({ gc: gcHeap, cdp: cdpSamples, events }))
const gcHeap = []
{
  const cdp = await ctx.newCDPSession(page)
  await cdp.send('HeapProfiler.enable')
  const gcTick = async () => {
    if (crashed) return
    try {
      await cdp.send('HeapProfiler.collectGarbage')
      const h = await cdp.send('Runtime.getHeapUsage')
      gcHeap.push({ t: Math.round((Date.now() - t0) / 1000), used: h.usedSize, total: h.totalSize })
      note('heap-after-gc', `${(h.usedSize / 1048576).toFixed(1)} MB`)
    } catch { /* crashed */ }
    setTimeout(gcTick, 30000)
  }
  setTimeout(gcTick, 20000)
  await cdp.send('Performance.enable')
  const tick = async () => {
    if (crashed) return
    try {
      const { metrics } = await cdp.send('Performance.getMetrics')
      const m = Object.fromEntries(metrics.map((x) => [x.name, x.value]))
      cdpSamples.push({ wall: Date.now(), LayoutCount: m.LayoutCount, RecalcStyleCount: m.RecalcStyleCount, Nodes: m.Nodes, LayoutObjects: m.LayoutObjects, JSEventListeners: m.JSEventListeners, JSHeapUsedSize: m.JSHeapUsedSize, Documents: m.Documents })
    } catch { /* crashed */ }
    setTimeout(tick, 1000)
  }
  tick()
}
await page.waitForTimeout(8000)
note('ready', `v3=${await frame.evaluate(() => !!document.querySelector('[data-testid=flow-v3-panel]')).catch(() => 'n/a')} heat=${await frame.evaluate(() => document.querySelector('[data-testid=flow-v3-heatmap-toggle]')?.getAttribute('aria-pressed')).catch(() => 'n/a')}`)

async function boxes(f) {
  let el = await f.$('.tv-lightweight-charts').catch(() => null)
  for (let i = 0; !el && i < 60; i += 1) { await page.waitForTimeout(500); frame = await chartFrame(); f = frame; el = await f.$('.tv-lightweight-charts').catch(() => null) }
  const b = await el.boundingBox()
  // price axis ≈ right 64 px, time axis ≈ bottom 28 px of the main pane (top 62 % of the chart element)
  return { plot: { x: b.x + 10, y: b.y + 10, w: b.width - 84, h: b.height * 0.6 - 20 }, priceAxis: { x: b.x + b.width - 40, y: b.y + 20, h: b.height * 0.6 - 40 }, timeAxisY: b.y + b.height - 14, b }
}
let bx = await boxes(frame)
let shot30 = false
let lastLogDump = 0
const v2logs = {}
const geoms = {}
const schedule = [[300, '2026-05-01'], [420, LIVE]]
let n = 0
while (!crashed && (Date.now() - t0) / 1000 < DURATION) {
  const t = (Date.now() - t0) / 1000
  if (!shot30 && t >= 30) { shot30 = true; await page.screenshot({ path: path.join(OUT, `headed-${cell}-30s.png`) }).catch(() => {}) }
  if (schedule.length && t >= schedule[0][0]) {
    await page.screenshot({ path: path.join(OUT, `headed-${cell}-${Math.round(t)}s.png`) }).catch(() => {})
    await setDate(schedule.shift()[1])
    frame = await chartFrame()
    await page.waitForTimeout(4000)
    bx = await boxes(frame)
    note('after-date', `v3=${await frame.evaluate(() => !!document.querySelector('[data-testid=flow-v3-panel]')).catch(() => 'n/a')}`)
  }
  const { plot, priceAxis } = bx
  const px = plot.x + rnd() * plot.w
  const py = plot.y + rnd() * plot.h
  const r = rnd()
  try {
    if (r < 0.35) { // wheel zoom, hard
      recent.push(`${((Date.now() - t0) / 1000).toFixed(1)} wheel`)
      await page.mouse.move(px, py)
      for (let i = 0; i < 6; i += 1) await page.mouse.wheel(0, (rnd() < 0.5 ? -1 : 1) * (100 + rnd() * 900))
    } else if (r < 0.6) { // pan
      recent.push(`${((Date.now() - t0) / 1000).toFixed(1)} pan`)
      await page.mouse.move(px, py); await page.mouse.down()
      await page.mouse.move(px + (rnd() - 0.5) * plot.w * 1.6, py + (rnd() - 0.5) * plot.h, { steps: 8 }); await page.mouse.up()
    } else if (r < 0.85) { // drag the price axis far: squash / stretch the price scale to extremes
      recent.push(`${((Date.now() - t0) / 1000).toFixed(1)} priceAxisDrag`)
      const y0 = priceAxis.y + rnd() * priceAxis.h
      await page.mouse.move(priceAxis.x, y0); await page.mouse.down()
      await page.mouse.move(priceAxis.x, y0 + (rnd() - 0.5) * 1600, { steps: 10 }); await page.mouse.up()
    } else if (r < 0.95) { // drag the time axis (bar spacing)
      recent.push(`${((Date.now() - t0) / 1000).toFixed(1)} timeAxisDrag`)
      await page.mouse.move(px, bx.timeAxisY); await page.mouse.down()
      await page.mouse.move(px + (rnd() - 0.5) * 1400, bx.timeAxisY, { steps: 8 }); await page.mouse.up()
    } else { // reset the price scale
      recent.push(`${((Date.now() - t0) / 1000).toFixed(1)} priceAxisReset`)
      await page.mouse.dblclick(priceAxis.x, priceAxis.y + priceAxis.h / 2)
    }
  } catch (e) { note('input-error', e.message) }
  n += 1
  if (Date.now() - lastLogDump > 5000) {
    lastLogDump = Date.now()
    try { flushCounts() } catch { /* */ }
  }
  if (n % 200 === 0) note('progress', `actions=${n}`)
  await page.waitForTimeout(40 + rnd() * 160)
}
if (!crashed) await page.screenshot({ path: path.join(OUT, `headed-${cell}-end.png`) }).catch(() => {})
note('end', `crashed=${crashed} actions=${n}`)
flushCounts()
fs.writeFileSync(path.join(OUT, `headed-${cell}.json`), JSON.stringify({ cell, durationSec: DURATION, crashed, actions: n, gc: gcHeap, events }, null, 1))
await Promise.race([browser.close().catch(() => {}), new Promise((r) => setTimeout(r, 5000))])
process.exit(0)
