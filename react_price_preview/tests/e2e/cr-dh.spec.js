// CR-DH — GEX levels Corrected by default on the Dashboard GEX chart and the price chart
// (specs/CR-DH-gex-levels-corrected-default.md). Read-only.
//
// Servers (started outside Playwright, one DB-backed server):
//   backend   apps/web/.venv/bin/python -u scripts/cr_dh_smoke_backend.py   (read-only DB, port 8067;
//             Dashboard GEX block at /, price chart API, origin/main figure at /smoke/origin-figure)
//   frontend  npm --prefix react_price_preview run dev -- --port 5186 --strictPort
// then:      npx playwright test cr-dh   (from react_price_preview/)
//
// Every test starts from a fresh browser context (= fresh profile). Screenshots go to specs/assets/cr-dh/.
import { test, expect } from '@playwright/test'

const API_BASE = process.env.CRDH_API_BASE || 'http://127.0.0.1:8067'
const BASE_URL = process.env.CRDH_BASE_URL || 'http://localhost:5186'
const SHOTS = '../specs/assets/cr-dh'
const DAY = '2026-10-27'
const PRE_FORWARD = '2024-03-15'

// ---------- price chart ----------

// Loads the chart and returns the `levels` param of the last /api/ironbeam/bars request (null = Current).
async function loadChart(page, day = DAY) {
  const sent = []
  const onRequest = (r) => {
    if (r.url().includes('/api/ironbeam/bars')) sent.push(new URL(r.url()).searchParams.get('levels'))
  }
  page.on('request', onRequest)
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto(`${BASE_URL}/?trade_date=${day}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor({ timeout: 90_000 })
  await expect.poll(() => sent.length, { timeout: 90_000 }).toBeGreaterThan(0)
  await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
  page.off('request', onRequest)
  return sent[sent.length - 1]
}

async function chartRadio(page) {
  await page.getByRole('button', { name: 'Open price chart settings' }).click()
  const v = await page.locator('input[name="gex-levels-mode"]:checked').getAttribute('value')
  await page.getByRole('button', { name: 'Cancel' }).click()
  return v
}

async function applyChartMode(page, value) {
  await page.getByRole('button', { name: 'Open price chart settings' }).click()
  await page.locator(`input[name="gex-levels-mode"][value="${value}"]`).check()
  await page.getByRole('button', { name: 'Apply' }).click()
  if (await page.getByRole('button', { name: 'Cancel' }).count()) await page.getByRole('button', { name: 'Cancel' }).click()
}

const chartStored = (page) => page.evaluate(() => ({
  v2: window.localStorage.getItem('ib-react-gex-levels-mode-v2'),
  old: window.localStorage.getItem('ib-react-gex-levels-mode'),
}))

test('price chart: fresh profile opens on Corrected', async ({ page }) => {
  expect(await loadChart(page)).toBe('corrected')
  expect(await chartRadio(page)).toBe('corrected')
  expect((await chartStored(page)).v2).toBe('corrected')
  await page.screenshot({ path: `${SHOTS}/price-chart-${DAY}-fresh-corrected.png` })
})

test('price chart: Current survives reload, then back to Corrected', async ({ page }) => {
  await loadChart(page)
  await applyChartMode(page, 'current')
  expect((await chartStored(page)).v2).toBe('current')
  expect(await loadChart(page)).toBe(null)
  expect(await chartRadio(page)).toBe('current')
  await applyChartMode(page, 'corrected')
  expect(await loadChart(page)).toBe('corrected')
})

test("price chart: old key 'current' opens on Corrected the first time", async ({ page }) => {
  await page.addInitScript(() => {
    if (!window.sessionStorage.getItem('crdh-seeded')) {
      window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
      window.sessionStorage.setItem('crdh-seeded', '1')
    }
  })
  expect(await loadChart(page)).toBe('corrected')
  expect(await chartRadio(page)).toBe('corrected')
  const s = await chartStored(page)
  expect(s.old).toBe('current') // untouched, ignored
  expect(s.v2).toBe('corrected')
})

// ---------- Dashboard GEX chart ----------

async function loadDash(page, day) {
  await page.setViewportSize({ width: 1000, height: 900 })
  await page.goto(`${API_BASE}/?date=${day}`)
  await expect.poll(() => dashTitle(page), { timeout: 120_000 }).toContain(day)
}

const dashFigure = (page) => page.evaluate(() => {
  const gd = document.querySelector('#GEX_GRAPH .js-plotly-plot')
  return gd && gd.data ? { data: gd.data.map((t) => ({ name: t.name, x: Array.from(t.x), y: Array.from(t.y) })), layout: gd.layout } : null
})

const dashTitle = async (page) => {
  const f = await dashFigure(page)
  const t = f && f.layout && f.layout.title
  return (t && (t.text || t)) || ''
}

const dashRadio = (page) => page.locator('#gex-levels-mode input:checked').getAttribute('value')

async function setDashMode(page, label, day) {
  await page.locator('#gex-levels-mode label', { hasText: label }).click()
  const head = label === 'Corrected' ? 'GEX by Corrected Level' : 'GEX by Discounted Level —'
  await expect.poll(() => dashTitle(page), { timeout: 120_000 }).toContain(head)
  await expect.poll(() => dashTitle(page)).toContain(day)
}

test('Dashboard: fresh profile opens on Corrected', async ({ page }) => {
  await loadDash(page, DAY)
  expect(await dashRadio(page)).toBe('corrected')
  expect(await dashTitle(page)).toBe(`GEX by Corrected Level — ${DAY} (SPX)`)
  await page.screenshot({ path: `${SHOTS}/dashboard-${DAY}-fresh-corrected.png` })
})

test('Dashboard: Current survives reload and equals origin/main', async ({ page, request }) => {
  await loadDash(page, DAY)
  await setDashMode(page, 'Current', DAY)
  await page.reload()
  await expect.poll(() => dashTitle(page), { timeout: 120_000 }).toBe(`GEX by Discounted Level — ${DAY} (SPX)`)
  expect(await dashRadio(page)).toBe('current')
  const origin = await (await request.get(`${API_BASE}/smoke/origin-figure?date=${DAY}`)).json()
  const fig = await dashFigure(page)
  expect(fig.data.map((t) => t.name)).toEqual(origin.data.map((t) => t.name))
  fig.data.forEach((t, i) => {
    expect(t.x).toEqual(origin.data[i].x)
    expect(t.y).toEqual(origin.data[i].y)
  })
  for (const k of ['xaxis', 'yaxis']) {
    expect(fig.layout[k].title.text).toBe(origin.layout[k].title.text)
    expect(fig.layout[k].range).toEqual(origin.layout[k].range)
  }
  expect(fig.layout.title.text).toBe(origin.layout.title.text)
  await page.screenshot({ path: `${SHOTS}/dashboard-${DAY}-current.png` })
  await setDashMode(page, 'Corrected', DAY)
  await page.reload()
  await expect.poll(() => dashTitle(page), { timeout: 120_000 }).toContain('GEX by Corrected Level')
})

test(`Dashboard Corrected levels equal the price chart's corrected lines (${DAY})`, async ({ page, request }) => {
  await loadDash(page, DAY)
  const fig = await dashFigure(page)
  const [puts, calls] = fig.data
  const byLevel = new Map(puts.y.map((lvl, i) => [lvl, { put: puts.x[i], call: calls.x[i] }]))
  const bars = await (await request.get(
    `${API_BASE}/api/ironbeam/bars?trade_date=${DAY}&interval=1min&days_either_side=0&gex_min_abs_b=40&levels=corrected`,
    { timeout: 180_000 },
  )).json()
  const segs = bars.gex_segments.filter((s) => s.session_date === DAY)
  expect(segs.length).toBeGreaterThan(0)
  for (const s of segs) {
    const row = byLevel.get(s.level)
    expect(row, `level ${s.level}`).toBeTruthy()
    expect(Math.abs(row.put - s.put_gamma)).toBeLessThan(1e-3)
    expect(Math.abs(row.call - s.call_gamma)).toBeLessThan(1e-3)
  }
  console.log(`CR-DH: ${segs.length} price-chart corrected lines on ${DAY}, all equal to the Dashboard (of ${puts.y.length} Dashboard levels)`)
})

test(`Dashboard: pre-forward-era date shows the labelled fallback (${PRE_FORWARD})`, async ({ page, request }) => {
  await loadDash(page, PRE_FORWARD)
  expect(await dashRadio(page)).toBe('corrected')
  expect(await dashTitle(page)).toBe(`GEX by Discounted Level (corrected unavailable: pre-forward era) — ${PRE_FORWARD} (SPX)`)
  const origin = await (await request.get(`${API_BASE}/smoke/origin-figure?date=${PRE_FORWARD}`)).json()
  const fig = await dashFigure(page)
  fig.data.forEach((t, i) => expect(t.y).toEqual(origin.data[i].y))
  await page.screenshot({ path: `${SHOTS}/dashboard-${PRE_FORWARD}-fallback.png` })
})
