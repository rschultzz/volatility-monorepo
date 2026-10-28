// CR-DB — compact review card (specs/CR-DB-review-card-compact.md). Read-only: nothing is saved
// to the DB (B8 — no smoke label lands in Ryan's cr-cz set; the Save test intercepts the POST).
//
// Servers (started outside Playwright):
//   backend           apps/web/.venv/bin/python -u scripts/cr_cc_smoke_backend.py   (port 8061)
//   frontend, branch  npm --prefix react_price_preview run dev -- --port 5176 --strictPort
//   frontend, main    the same from an origin/main checkout on port 5177 (CRDB_MAIN_URL)
// then:              npx playwright test cr-db   (from react_price_preview/)
//
// "card": one cr-cz card at 1440 × 900 and at 900 × 600 — direction label, edge, key facts, details collapsed;
//   Save, the Take / Skip / Unsure buttons and both CR-CZ chips inside the viewport without scrolling.
// "keys + save": T / S / U set the label; Save sends label + chips, intercepted in the browser.
// "chart": the fade / continuation edge lines on the box.
// "pixel": the price chart with the review panel closed is byte-identical to origin/main's.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCC_API_BASE || 'http://127.0.0.1:8061'
const BRANCH_URL = process.env.CRDB_BRANCH_URL || 'http://localhost:5176'
const MAIN_URL = process.env.CRDB_MAIN_URL || 'http://localhost:5177'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-db')
const SET = 'cr-cz:cp-ranges'
const CHIPS = ['take fade (called side)', 'take continuation (other edge)']

async function api(request, p) {
  const r = await request.get(`${API_BASE}/api/setup-review${p}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

async function pickCard(request) {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}`)
  expect(list.candidates.length).toBeGreaterThan(0)
  const c = list.candidates.find((x) => !x.revealed && x.direction === 'short') || list.candidates[0]
  return (await api(request, `/candidates/${encodeURIComponent(c.id)}`)).candidate
}

async function openCard(page, c, viewport) {
  await page.setViewportSize(viewport)
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
      window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
      window.localStorage.setItem('ib-react-gex-levels-mode-v2', 'current') // CR-DH key
    } catch (e) { /* ignore */ }
  })
  await page.goto(`${BRANCH_URL}/?trade_date=2026-10-23&interval=1min&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption(SET)
  await page.locator(`[data-testid="review-row"][data-candidate-id="${c.id}"]`).click()
  await expect(page.getByTestId('review-banner')).toContainText(`Review: ${c.session_date}`, { timeout: 60_000 })
  await expect(page.getByTestId('review-active')).toContainText(String(c.id))
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
  await page.waitForLoadState('networkidle', { timeout: 20_000 }).catch(() => {})
  await page.waitForTimeout(800)
}

// fully inside the viewport and inside the panel, with no scrolling of the page or the panel
async function expectInView(page, locator) {
  const vp = page.viewportSize()
  const box = await locator.boundingBox()
  const panel = await page.getByTestId('review-panel').boundingBox()
  expect(box, 'element rendered').not.toBeNull()
  expect(box.y).toBeGreaterThanOrEqual(panel.y)
  expect(box.y + box.height).toBeLessThanOrEqual(Math.min(vp.height, panel.y + panel.height) + 0.5)
  expect(box.x).toBeGreaterThanOrEqual(panel.x - 0.5)
  expect(box.x + box.width).toBeLessThanOrEqual(panel.x + panel.width + 0.5)
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 900, height: 600 }]) {
  test(`card at ${viewport.width}×${viewport.height}: direction, facts, actions in view`, async ({ page, request }) => {
    fs.mkdirSync(ASSETS, { recursive: true })
    const c = await pickCard(request)
    await openCard(page, c, viewport)
    const long = c.direction === 'long'
    const edge = long ? c.meta.range.bottom : c.meta.range.top

    await expect(page.getByTestId('review-direction')).toContainText(long ? 'LONG the bottom' : 'SHORT the top')
    await expect(page.getByTestId('review-edge')).toContainText(`edge ${Number(edge).toFixed(2).replace(/\.00$/, '')}`)
    await expect(page.getByTestId('review-edge')).toContainText('limit')
    const facts = page.getByTestId('review-key-facts')
    for (const k of ['Box (closes)', 'Range start', 'Approach', 'In-range delta share', 'Progress', 'Contracts / pt']) {
      await expect(facts).toContainText(k)
    }
    // details collapsed: no long rule text, no field names
    await expect(page.getByTestId('review-details')).toHaveCount(0)
    await expect(page.getByTestId('review-active')).not.toContainText('entry_rule')

    for (const loc of [
      page.getByTestId('review-direction'),
      page.getByTestId('review-save'),
      page.getByTestId('review-label-take'),
      page.getByTestId('review-label-skip'),
      page.getByTestId('review-label-unsure'),
      ...CHIPS.map((ch) => page.getByTestId('review-reasons').getByRole('button', { name: ch, exact: true })),
    ]) await expectInView(page, loc)
    const scrolled = await page.evaluate(() => ({
      page: window.scrollY,
      panel: document.querySelector('[data-testid=review-panel]').scrollTop,
    }))
    expect(scrolled).toEqual({ page: 0, panel: 0 })
    await page.screenshot({ path: path.join(ASSETS, `card-${viewport.width}x${viewport.height}.png`) })

    // details open: long text as a full-width paragraph, the actions still in view
    await page.getByTestId('review-details-toggle').click()
    await expect(page.getByTestId('review-details')).toBeVisible()
    const para = page.getByTestId('review-meta-text').filter({ hasText: 'entry rule' })
    await expect(para).toHaveCount(1)
    const pBox = await para.boundingBox()
    const dBox = await page.getByTestId('review-details').boundingBox()
    expect(pBox.width).toBeGreaterThan(dBox.width * 0.9)
    await expectInView(page, page.getByTestId('review-save'))
    await page.screenshot({ path: path.join(ASSETS, `card-details-${viewport.width}x${viewport.height}.png`) })
  })
}

test('keys T / S / U set the label; Save sends label and chips (intercepted, no DB write)', async ({ page, request }) => {
  const c = await pickCard(request)
  const posts = []
  await page.route('**/api/setup-review/candidates/*/review', async (route) => {
    posts.push(route.request().postDataJSON())
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true }) })
  })
  await openCard(page, c, { width: 1440, height: 900 })
  for (const [key, id] of [['t', 'take'], ['s', 'skip'], ['u', 'unsure'], ['t', 'take']]) {
    await page.keyboard.press(key)
    await expect(page.getByTestId(`review-label-${id}`)).toHaveAttribute('aria-pressed', 'true')
  }
  for (const ch of CHIPS) await page.getByTestId('review-reasons').getByRole('button', { name: ch, exact: true }).click()
  await page.getByTestId('review-save').click()
  await expect.poll(() => posts.length).toBe(1)
  expect(posts[0]).toEqual({ label: 'take', reasons: CHIPS, comment: '' })
  const after = await api(request, `/candidates/${encodeURIComponent(c.id)}`)
  expect(after.candidate.reviews || []).toEqual(c.reviews || []) // nothing reached the DB
})

test('chart: fade and continuation edge lines on the box', async ({ page, request }) => {
  const c = await pickCard(request)
  await openCard(page, c, { width: 1440, height: 900 })
  const lines = await page.evaluate(() => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    let f = chartEl.parentElement[key]
    for (let d = 0; f && d < 30; d += 1, f = f.return) {
      for (let h = f.memoizedState; h; h = h.next) {
        const v = h.memoizedState
        const cur = v && typeof v === 'object' && 'current' in v ? v.current : null
        if (cur && typeof cur.priceToCoordinate === 'function' && cur.seriesType() === 'Candlestick') {
          return cur.priceLines().map((l) => ({ title: l.options().title, price: l.options().price, color: l.options().color }))
        }
      }
    }
    return []
  })
  const long = c.direction === 'long'
  const fade = lines.find((l) => l.title === 'fade edge')
  const cont = lines.find((l) => l.title === 'continuation edge')
  expect(fade.price).toBe(Number(long ? c.meta.range.bottom : c.meta.range.top))
  expect(cont.price).toBe(Number(long ? c.meta.range.top : c.meta.range.bottom))
  for (const l of [fade, cont]) expect(['#60a5fa', '#cbd5e1']).toContain(l.color)
  await page.mouse.move(5, 5)
  await page.screenshot({ path: path.join(ASSETS, `chart-edges-${c.session_date}.png`) })
})

test('pixel: price chart with the review panel closed is identical to origin/main', async ({ browser }) => {
  fs.mkdirSync(ASSETS, { recursive: true })
  const shots = {}
  for (const [name, base] of [['main', MAIN_URL], ['branch', BRANCH_URL], ['branch2', BRANCH_URL]]) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    const page = await ctx.newPage()
    await page.addInitScript(() => {
      try {
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.localStorage.setItem('ib-react-gex-levels-mode', 'current')
        window.localStorage.setItem('ib-react-gex-levels-mode-v2', 'current') // CR-DH key
      } catch (e) { /* ignore */ }
    })
    await page.goto(`${base}/?trade_date=2026-10-23&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}`)
    await page.locator('.tv-lightweight-charts').first().waitFor()
    await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
    await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
    await page.waitForTimeout(1500)
    await page.mouse.move(1, 1)
    shots[name] = await page.locator('.tv-lightweight-charts').first().screenshot({ path: path.join(ASSETS, `chart-closed-${name}.png`) })
    await ctx.close()
  }
  expect(shots.branch.equals(shots.branch2)).toBe(true) // the capture itself is deterministic
  expect(shots.branch.equals(shots.main)).toBe(true)
})
