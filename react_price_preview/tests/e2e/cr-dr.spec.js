// CR-DR — My trades on the price chart, the trade review panel, the Trade Log tab's tags / filters / breakdown
// (specs/CR-DR-trade-log-v2.md). Ironbeam routes read-only; the trade-log PATCHes write Ryan's review fields on ONE trade
// (#5) and put them back to their stored values at the end.
//
// Servers (started outside Playwright; ONE DB-backed server):
//   backend    apps/web/.venv/bin/python -u scripts/cr_dr_smoke_backend.py              (port 8069)
//   frontend   npm --prefix react_price_preview run dev -- --port 5197 --strictPort
//   trade log  npm --prefix react_backtests_v2 run dev -- --port 5198 --strictPort      (/api/* routed to the backend here)
// then:       CRBV_BASE_URL=http://localhost:5197 npx playwright test cr-dr   (from react_price_preview/)
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API = process.env.CRDR_API_BASE || 'http://127.0.0.1:8069'
const TRADE_LOG_URL = process.env.CRDR_TRADE_LOG_URL || 'http://localhost:5198/backtests-v2-preview/'
const HERE = path.dirname(fileURLToPath(import.meta.url))
const LOGS = path.resolve(HERE, '../../../scripts/logs/cr-dr/smoke')
const ASSETS = path.resolve(HERE, '../../../specs/assets/cr-dr')
const D30 = '2026-10-30'
const REVIEW_KEYS = ['setup_chips', 'pre_trade_reason', 'mistake_chips', 'checklist_followed', 'lesson', 'grade']

const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}
const shot = async (page, name) => {
  fs.mkdirSync(ASSETS, { recursive: true })
  await page.screenshot({ path: path.join(ASSETS, name) })
}
const shifted = (pt) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(pt)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) / 1000
}
const api = async (request, p, opts) => {
  const r = await request.fetch(`${API}${p}`, opts)
  return r.json()
}

async function open(page, day, { myTrades = false } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(() => {
    try {
      if (!window.sessionStorage.getItem('crdr-init')) {
        for (const k of Object.keys(window.localStorage)) if (k.startsWith('ib-react-')) window.localStorage.removeItem(k)
        window.localStorage.setItem('ib-react-flow-mode-v3', '1')
        window.localStorage.setItem('ib-react-gex-min-abs-b', '9999') // no GEX lines (their click popups are not this CR's)
        window.sessionStorage.setItem('crdr-init', '1')
      }
    } catch (e) { /* ignore */ }
  })
  await page.goto(`/?trade_date=${day}&interval=1min&days_either_side=0&flow_session=RTH&api_base=${encodeURIComponent(API)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByTestId('flow-v3-panel').waitFor()
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  if (myTrades) {
    await page.getByTestId('flow-v3-my-trades-toggle').click()
    await page.waitForFunction(() => (window.__crDrTradeHits?.() || []).length > 0, null, { timeout: 60_000 })
  }
  await page.waitForTimeout(800)
}

// the price chart objects through React's fiber tree (test-only, as cr-dq.spec.js)
const CHARTS_FN = `(() => {
  const el = document.querySelector('.tv-lightweight-charts')
  const key = Object.keys(el.parentElement).find((k) => k.startsWith('__reactFiber'))
  let f = el.parentElement[key]
  const o = {}
  for (let d = 0; f && d < 40; d += 1, f = f.return) {
    for (let h = f.memoizedState; h; h = h.next) {
      const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
      if (!c) continue
      if (typeof c.timeScale === 'function') o.chart = o.chart || c
      if (typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick') o.series = o.series || c
    }
  }
  return { el, ...o }
})()`

// frame the chart on [from, to] (shifted epochs)
async function frame(page, from, to) {
  await page.evaluate(({ a, b, fn }) => {
    const { chart, series } = eval(fn) // eslint-disable-line no-eval
    const data = series.data().filter((x) => x.close != null)
    const i0 = Math.max(0, data.findIndex((x) => x.time >= a))
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to, fn: CHARTS_FN })
  await page.waitForTimeout(800)
}

// every drawn marker with the candle under it and the price at its y
async function hitsWithBars(page) {
  return page.evaluate((fn) => {
    const { el, chart, series } = eval(fn) // eslint-disable-line no-eval
    const rect = el.getBoundingClientRect()
    return (window.__crDrTradeHits?.() || []).map((h) => ({
      ...h, pageX: rect.x + h.x, pageY: rect.y + h.y,
      barTime: chart.timeScale().coordinateToTime(h.x),
      priceAtY: series.coordinateToPrice(h.priceY),
      tickPx: Math.abs(series.priceToCoordinate(7650) - series.priceToCoordinate(7650.25)),
    }))
  }, CHARTS_FN)
}

test.describe.configure({ mode: 'serial' })

let trades30 = []
let t5 = null
let saved5 = null
test.beforeAll(async ({ request }) => {
  trades30 = (await api(request, `/api/trade-log/chart?from=${D30}&to=${D30}`)).trades
  t5 = trades30.find((t) => t.entry_pt.endsWith('07:49:17'))
  saved5 = Object.fromEntries(REVIEW_KEYS.map((k) => [k, t5[k]]))
  saved5.anchor_ts_pt = t5.anchor_pt
  log('trades-2026-10-30.json', trades30)
})
test.afterAll(async ({ request }) => {
  if (t5) {
    const r = await api(request, `/api/trade-log/trades/${t5.id}`, { method: 'PATCH', data: saved5 })
    log('restore-5.json', { sent: saved5, ok: r.ok, after: REVIEW_KEYS.map((k) => [k, r.trade?.[k]]) })
  }
})

test('2026-10-30: four trades, P&L = vault log #3–#6', async () => {
  expect(trades30.map((t) => [t.entry_pt.slice(11), t.direction, t.realized_pts])).toEqual([
    ['06:11:46', 'long', 1.75], ['07:06:47', 'long', -7.5], ['07:49:17', 'short', 13.75], ['09:22:50', 'short', -5.25],
  ])
})

test('My trades is off by default; nothing drawn', async ({ page }) => {
  await open(page, D30)
  await expect(page.getByTestId('flow-v3-my-trades-toggle')).toHaveText('My trades off')
  await expect(page.getByTestId('flow-v3-my-trades-toggle')).toHaveAttribute('aria-pressed', 'false')
  expect(await page.evaluate(() => (window.__crDrTradeHits?.() || []).length)).toBe(0)
  await expect(page.getByTestId('trade-review-panel')).toHaveCount(0)
})

test('markers sit on the fill bar and the fill price', async ({ page }) => {
  await open(page, D30, { myTrades: true })
  await frame(page, shifted(`${D30}T06:00:00`), shifted(`${D30}T09:40:00`))
  await shot(page, 'my-trades-2026-10-30.png')
  const hits = await hitsWithBars(page)
  const rows = []
  for (const t of trades30) {
    for (const [kind, pt, price] of [['entry', t.entry_pt, t.entry_price], ['exit', t.exit_pt, t.exit_price]]) {
      const h = hits.find((x) => x.id === t.id && x.kind === kind)
      const minute = Math.floor(shifted(pt) / 60) * 60
      rows.push({ id: t.id, kind, pt, price, barTime: h?.barTime, minute, priceAtY: h?.priceAtY })
      expect(h, `${t.id} ${kind}`).toBeTruthy()
      expect(h.barTime).toBe(minute)
      expect(Math.abs(h.priceAtY - price)).toBeLessThan(0.01 + 0.25 / Math.max(1, h.tickPx)) // within a pixel
    }
  }
  log('markers-2026-10-30.json', rows)
  // a zoomed view on #5 for the eye
  await frame(page, shifted(`${D30}T07:35:00`), shifted(`${D30}T08:15:00`))
  await shot(page, 'my-trades-5-zoom.png')
})

test('#5: click opens the panel — facts, MFE / MAE, stop, at-entry context with anchor 07:40', async ({ page, request }) => {
  await api(request, `/api/trade-log/trades/${t5.id}`, { method: 'PATCH', data: { anchor_ts_pt: `${D30}T07:40:00` } })
  await open(page, D30, { myTrades: true })
  await frame(page, shifted(`${D30}T07:30:00`), shifted(`${D30}T08:20:00`))
  const h = (await hitsWithBars(page)).find((x) => x.id === t5.id && x.kind === 'entry')
  await page.mouse.click(h.pageX, h.pageY)
  const panel = page.getByTestId('trade-review-panel')
  await expect(panel).toBeVisible()
  await expect(panel).toHaveAttribute('data-trade-id', String(t5.id))
  await expect(page.getByTestId('trade-review-entry')).toContainText('entry 07:49:17 @ 7655.75')
  await expect(page.getByTestId('trade-review-exit')).toContainText('exit 08:08:12 @ 7642.00')
  await expect(page.getByTestId('trade-review-result')).toContainText('+13.75 pts · +$68.75')
  await expect(page.getByTestId('trade-review-result')).toContainText('18m 55s in trade')
  await expect(page.getByTestId('trade-review-mfe')).toHaveText('MFE +14.00 · MAE −1.75 pts')
  await expect(page.getByTestId('trade-review-stop')).toContainText('no — first stop after 109 s')
  await expect(page.getByTestId('trade-review-stop')).toContainText('stop 7647.75')
  await expect(page.getByTestId('trade-review-anchor')).toContainText('anchor: 07:40')
  await expect(page.getByTestId('trade-review-verdict')).toHaveAttribute('data-kind', 'forming', { timeout: 90_000 })
  await expect(page.getByTestId('trade-review-evidence')).toContainText('box 8 min')
  await expect(page.getByTestId('trade-review-box')).toContainText('box 07:40–07:48 · 7652.25–7656.50')
  await expect(page.getByTestId('trade-review-box')).toContainText('share +8.14 %')
  await expect(page.getByTestId('trade-review-levels')).toContainText('buyers absorbed 7660.25 (+4.50')
  await expect(page.getByTestId('trade-review-last5')).toContainText('last 5 min (07:44–07:48): Δ +2,564 · vol 26.4k')
  // default place: bottom-right inside the plot (right of the panel left of the price axis)
  const box = await panel.boundingBox()
  const chart = await page.locator('.tv-lightweight-charts').first().boundingBox()
  expect(box.x + box.width).toBeLessThan(chart.x + chart.width - 40)
  expect(box.x).toBeGreaterThan(chart.x + chart.width / 2)
  log('panel-5.json', { text: await panel.innerText(), box, chart })
  await shot(page, 'panel-5.png')
})

test('N / P step through the day; the panel follows; Esc closes', async ({ page }) => {
  await open(page, D30, { myTrades: true })
  await frame(page, shifted(`${D30}T07:30:00`), shifted(`${D30}T08:20:00`))
  const h = (await hitsWithBars(page)).find((x) => x.id === t5.id && x.kind === 'exit')
  await page.mouse.click(h.pageX, h.pageY)
  await expect(page.getByTestId('trade-review-header')).toContainText('Trade 3 / 4')
  await page.keyboard.press('n')
  await expect(page.getByTestId('trade-review-entry')).toContainText('entry 09:22:50')
  await expect(page.getByTestId('trade-review-header')).toContainText('Trade 4 / 4')
  await expect(page.getByTestId('trade-review-next')).toBeDisabled()
  // the chart scrolled so 09:22 is in view
  const inView = await page.evaluate(({ fn, t }) => {
    const { chart } = eval(fn) // eslint-disable-line no-eval
    const x = chart.timeScale().timeToCoordinate(t)
    return x != null && x > 0 && x < chart.timeScale().width()
  }, { fn: CHARTS_FN, t: shifted(`${D30}T09:22:00`) })
  expect(inView).toBe(true)
  await page.getByTestId('trade-review-prev').click()
  await page.getByTestId('trade-review-prev').click()
  await expect(page.getByTestId('trade-review-entry')).toContainText('entry 07:06:47')
  await expect(page.getByTestId('trade-review-stop')).toContainText('no stop while open')
  await shot(page, 'panel-4.png')
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('trade-review-panel')).toHaveCount(0)
})

test('chips, checklist, grade and notes autosave; filters and the checklist split see them', async ({ page, request }) => {
  await open(page, D30, { myTrades: true })
  await frame(page, shifted(`${D30}T07:30:00`), shifted(`${D30}T08:20:00`))
  const h = (await hitsWithBars(page)).find((x) => x.id === t5.id && x.kind === 'entry')
  await page.mouse.click(h.pageX, h.pageY)
  await page.getByTestId('trade-setup-chip-fade top').click()
  await page.getByTestId('trade-mistake-chip-target past level').click()
  await page.getByTestId('trade-checklist-yes').click()
  await page.getByTestId('trade-grade-4').click()
  await page.getByTestId('trade-review-reason').fill('A19 short at the top: buying, no advance')
  await page.getByTestId('trade-review-lesson').fill('smoke: CR-DR')
  await expect(page.getByTestId('trade-review-save')).toHaveAttribute('data-state', 'saved', { timeout: 15_000 })
  await expect(page.getByTestId('trade-review-save')).toContainText('saved ')
  await shot(page, 'panel-5-saved.png')

  const got = (await api(request, `/api/trade-log/trades/${t5.id}`)).trade
  expect([got.setup_chips, got.mistake_chips, got.checklist_followed, got.grade, got.pre_trade_reason, got.lesson])
    .toEqual([['fade top'], ['target past level'], true, 4, 'A19 short at the top: buying, no advance', 'smoke: CR-DR'])
  const byFilter = async (q) => (await api(request, `/api/trade-log/trades?${q}`)).trades.map((t) => t.id)
  expect(await byFilter('setup_chip=fade%20top')).toEqual([t5.id])
  expect(await byFilter('mistake_chip=target%20past%20level&checklist=yes&grade=4')).toEqual([t5.id])
  expect(await byFilter('checklist=no')).toEqual([])
  const agg = await api(request, '/api/trade-log/aggregate')
  const yes = agg.by_checklist.find((r) => r.key === 'yes')
  expect(yes).toEqual({ key: 'yes', n: 1, pts: 13.75 })
  expect(agg.by_setup.find((r) => r.key === 'fade top')).toEqual({ key: 'fade top', n: 1, pts: 13.75 })
  log('saved-5.json', { got, agg })

  // the Trade Log tab (react_backtests_v2): columns, filters and the breakdown row
  await page.route('**/api/**', async (route) => {
    const u = new URL(route.request().url())
    if (u.pathname.startsWith('/api/trade-log/')) {
      try {
        const r = await route.fetch({ url: `${API}${u.pathname}${u.search}` })
        return await route.fulfill({ response: r })
      } catch (e) {
        return undefined // the page closed while this request was in flight
      }
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{"ok":false,"strategies":[]}' })
  })
  await page.goto(TRADE_LOG_URL)
  await page.getByRole('button', { name: 'Trade Log' }).click()
  await expect(page.getByTestId('trade-log-by-checklist')).toContainText('yes 1 · +13.75 pts')
  await expect(page.getByTestId('trade-log-by-setup')).toContainText('fade top 1 · +13.75 pts')
  await expect(page.getByTestId('trade-log-by-mistake')).toContainText('target past level 1 · +13.75 pts')
  await page.getByTestId('trade-log-filter-checklist').selectOption('yes')
  await expect(page.locator('table.results-table tbody tr')).toHaveCount(1)
  await expect(page.locator('table.results-table tbody tr').first()).toContainText('fade top')
  await expect(page.locator('table.results-table tbody tr').first()).toContainText('target past level')
  await shot(page, 'trade-log-filtered.png')
  await page.getByTestId('trade-log-filter-checklist').selectOption('')
  await page.getByTestId('trade-log-filter-setup').selectOption('none')
  const nNone = agg.total_trades - 1
  await expect(page.locator('table.results-table tbody tr')).toHaveCount(nNone)
  await page.getByTestId('trade-log-filter-setup').selectOption('')
  await expect(page.locator('table.results-table tbody tr')).toHaveCount(agg.total_trades)
  await shot(page, 'trade-log-all.png')
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})

test('holdout date (2026-10-26): no MFE / MAE (B4)', async ({ request }) => {
  const t26 = (await api(request, '/api/trade-log/chart?from=2026-10-26&to=2026-10-26')).trades
  expect(t26.length).toBe(2)
  for (const t of t26) {
    const r = (await api(request, `/api/trade-log/trades/${t.id}/review`)).review
    expect([r.holdout, r.mfe_pts, r.mae_pts]).toEqual([true, null, null])
  }
})
