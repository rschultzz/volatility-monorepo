// CR-DK — review panel + chart small fixes (specs/CR-DK-review-and-chart-fixes.md). Read-only.
//
// Servers (started outside Playwright), as cr-dg.spec.js:
//   backend    apps/web/.venv/bin/python -u scripts/cr_dk_smoke_backend.py   (port 8064, read-only DB, capped pool)
//   frontend   npm --prefix react_price_preview run dev -- --port 5181 --strictPort
//   baseline   the same frontend at origin/main (a detached worktree), --port 5180
// then:       CRBV_BASE_URL=http://localhost:5181 npx playwright test tests/e2e/cr-dk.spec.js   (from react_price_preview/)
//
// "crosshair": with the cursor on the 07:22 PT candle of 2026-10-27 (07:20 on 5-minute bars) the flow pane's linked
//   crosshair sits on the flow bar of the same minute, and the flow pane scrolls with the price chart (Flow current
//   and v3; one session and a day either side). origin/main is logged (CRDK_SIDES=main).
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRDF_API_BASE || 'http://127.0.0.1:8064'
const BASELINE = process.env.CRDF_BASELINE_URL || 'http://localhost:5180'
const LOGS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../scripts/logs/cr-dk/smoke')
const FLOW_KEYS = ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort', 'ib-react-flow-mode-v3',
  'ib-react-flow-v3-settings', 'ib-react-flow-v3-layers']
const DAY = '2026-10-27'

const shifted = (ptIso) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}
const hhmm = (shiftedT) => (Number.isFinite(shiftedT) ? new Date(shiftedT * 1000).toISOString().slice(11, 16) : null)
const log = (name, obj) => {
  fs.mkdirSync(LOGS, { recursive: true })
  fs.writeFileSync(path.join(LOGS, name), JSON.stringify(obj, null, 1))
}

async function open(page, day, { base = '', storage = {}, levels = 'current', interval = '1min', either = 0, session = 'FULL' } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ keys, values, mode }) => {
    try {
      if (!window.sessionStorage.getItem('crdk-init')) {
        for (const k of keys) window.localStorage.removeItem(k)
        for (const [k, v] of Object.entries(values)) window.localStorage.setItem(k, v)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.localStorage.setItem('ib-react-gex-levels-mode', mode)
        window.localStorage.setItem('ib-react-gex-levels-mode-v2', mode)
        window.sessionStorage.setItem('crdk-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { keys: FLOW_KEYS, values: storage, mode: levels })
  await page.goto(`${base}/?trade_date=${day}&interval=${interval}&days_either_side=${either}&flow_session=${session}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 120_000 })
  await page.waitForLoadState('networkidle', { timeout: 120_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The lightweight-charts instance (and its candle / first series) behind a chart element, from the React fiber.
const CHART_OF = `((chartEl) => {
  const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
  let f = chartEl.parentElement[key]
  let chart = null
  let series = null
  for (let d = 0; f && d < 30; d += 1, f = f.return) {
    for (let h = f.memoizedState; h; h = h.next) {
      const c = h.memoizedState && typeof h.memoizedState === 'object' && 'current' in h.memoizedState ? h.memoizedState.current : null
      if (c && typeof c.timeScale === 'function') chart = chart || c
      if (c && typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick') series = series || c
    }
  }
  return { chart, series }
})`

async function frame(page, from, to) {
  await page.evaluate(({ a, b, fn }) => {
    const { chart, series } = eval(fn)(document.querySelector('.tv-lightweight-charts')) // eslint-disable-line no-eval
    const data = series.data().filter((x) => x.close != null)
    const i0 = data.findIndex((x) => x.time >= a)
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to, fn: CHART_OF })
  await page.waitForTimeout(800)
}

async function point(page, time) {
  return page.evaluate(({ t, fn }) => {
    const el = document.querySelector('.tv-lightweight-charts')
    const { chart } = eval(fn)(el) // eslint-disable-line no-eval
    const rect = el.getBoundingClientRect()
    return { x: rect.x + chart.timeScale().timeToCoordinate(t), spacing: chart.timeScale().options().barSpacing, rect: { y: rect.y, h: rect.height } }
  }, { t: time, fn: CHART_OF })
}

// The flow pane's linked crosshair (the dashed 1-px line): its page x, the flow chart's time under it, and where the
// flow chart draws the target minute.
async function flowCrosshair(page, target) {
  return page.evaluate(({ t, fn }) => {
    const line = [...document.querySelectorAll('div')].find((d) => d.style.borderLeft.includes('dashed') &&
      d.style.borderLeft.includes('191, 219, 254'))
    if (!line) return null
    let host = line.parentElement
    while (host && !host.querySelector('.tv-lightweight-charts')) host = host.parentElement
    const el = host?.querySelector('.tv-lightweight-charts')
    if (!el) return null
    const { chart } = eval(fn)(el) // eslint-disable-line no-eval
    const rect = el.getBoundingClientRect()
    const lx = line.getBoundingClientRect().x
    const ts = chart.timeScale()
    return { x: lx, time: Number(ts.coordinateToTime(lx - rect.x)), targetX: rect.x + ts.timeToCoordinate(t), spacing: ts.options().barSpacing }
  }, { t: target, fn: CHART_OF })
}

// Cursor across the 07:22 candle (left quarter, centre, right quarter): price-chart tooltip time and the flow pane's
// crosshair time / x offset from the flow bar of 07:22, in bars.
// The flow pane's scroll must match the price chart too: the price-vs-flow x of the bar is the same with a day either
// side as with none (Flow v3 sits a constant ≈ 9 px off at 0 days: different price-scale widths, out of scope).
const V3 = { 'ib-react-flow-mode-v3': '1' }
const CROSS_VARIANTS = [
  { name: 'current-0d', storage: {}, either: 0 },
  { name: 'current-1d', storage: {}, either: 1, same: 'current-0d' },
  { name: 'v3-0d', storage: V3, either: 0 },
  { name: 'v3-1d', storage: V3, either: 1, same: 'v3-0d' },
  { name: 'v3-5min-0d', storage: V3, either: 0, interval: '5min', bar: '07:20' },
  { name: 'v3-5min-1d', storage: V3, either: 1, interval: '5min', bar: '07:20', same: 'v3-5min-0d' },
]

async function crossAt(page, base, v) {
  await open(page, DAY, { base, storage: v.storage, either: v.either, interval: v.interval || '1min' })
  await settle(page)
  const t = shifted(`${DAY}T${v.bar || '07:22'}`)
  await frame(page, t - 20 * 60, t + 20 * 60)
  const p = await point(page, t)
  const y = p.rect.y + p.rect.h * 0.4
  const tip = page.locator('div', { hasText: /^Cursor$/ }).locator('xpath=following-sibling::div[1]')
  const out = []
  for (const dx of [-0.3, 0, 0.3]) {
    await page.mouse.move(p.x + dx * p.spacing, y)
    await page.waitForTimeout(300)
    const f = await flowCrosshair(page, t)
    out.push({ dx, tooltip: await tip.innerText(), flow: f && { time: hhmm(f.time), offsetBars: +((f.x - f.targetX) / f.spacing).toFixed(2), priceVsFlowPx: +(p.x - f.targetX).toFixed(1) } })
  }
  return out
}

test('crosshair: the flow pane follows the 07:22 PT candle', async ({ browser }) => {
  test.setTimeout(900_000)
  const res = {}
  const sides = process.env.CRDK_SIDES ? process.env.CRDK_SIDES.split(',') : ['branch', 'main']
  for (const v of CROSS_VARIANTS) {
    for (const [name, base] of [['branch', ''], ['main', BASELINE]].filter(([n]) => sides.includes(n))) {
      const ctx = await browser.newContext()
      res[`${v.name}:${name}`] = await crossAt(await ctx.newPage(), base, v)
      await ctx.close()
    }
  }
  log(`crosshair-0722-${sides.join('-')}.json`, res)
  if (!sides.includes('branch')) return
  for (const v of CROSS_VARIANTS) {
    for (const s of res[`${v.name}:branch`]) {
      expect(s.tooltip, v.name).toBe(`10/27 ${v.bar || '07:22'}:00 PT`)
      expect(s.flow?.time, v.name).toBe(v.bar || '07:22')
      expect(Math.abs(s.flow.offsetBars), v.name).toBeLessThan(0.05)
      if (v.same) expect(Math.abs(s.flow.priceVsFlowPx - res[`${v.same}:branch`][0].flow.priceVsFlowPx), v.name).toBeLessThan(2)
    }
  }
})

// ── Review panel (items 1, 2). B8: every Save is intercepted in the browser and never reaches the backend (which is
// read-only anyway); the DB's reviews are compared before and after.
const SET = 'cr-cz:cp-ranges'
const STORED = 'take continuation (other edge)'
const RETIRED = 'take fade (called side)'

async function api(request, p) {
  const r = await request.get(`${API_BASE}/api/setup-review${p}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

async function openCard(page, c) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
      window.localStorage.setItem('ib-react-gex-levels-mode-v2', 'current')
    } catch (e) { /* ignore */ }
  })
  await page.goto(`/?trade_date=2026-10-23&interval=1min&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption(SET)
  await page.locator(`[data-testid="review-row"][data-candidate-id="${c.id}"]`).click()
  await expect(page.getByTestId('review-active')).toContainText(String(c.id), { timeout: 60_000 })
  await page.waitForLoadState('networkidle', { timeout: 30_000 }).catch(() => {})
}

test('chip: "take opposite" shown, stored string sent (intercepted, no write); take fade not offered', async ({ page, request }) => {
  const list = await api(request, `/candidates?set=${encodeURIComponent(SET)}`)
  expect(list.reasons).toContain(STORED)
  expect(list.reasons).not.toContain(RETIRED)
  const c = list.candidates.find((x) => !x.reviewed) || list.candidates.find((x) => x.revealed)
  const before = (await api(request, `/candidates/${encodeURIComponent(c.id)}`)).candidate
  const posts = []
  await page.route('**/api/setup-review/candidates/*/review', async (route) => {
    posts.push(route.request().postDataJSON())
    await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: 'smoke: not saved' }) })
  })
  await openCard(page, c)
  const chips = page.getByTestId('review-reasons')
  const chip = chips.getByRole('button', { name: 'take opposite', exact: true })
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('title', "You'd take the opposite trade (approach direction, at the other edge of the box).")
  await expect(chips.getByRole('button', { name: RETIRED })).toHaveCount(0)
  await expect(chips.getByRole('button', { name: STORED })).toHaveCount(0)
  if (!c.revealed) await page.getByTestId('review-label-take').click()
  await chip.click()
  await page.getByTestId('review-save').click()
  await expect.poll(() => posts.length).toBe(1)
  expect(posts[0].reasons).toEqual([STORED])
  const after = (await api(request, `/candidates/${encodeURIComponent(c.id)}`)).candidate
  expect(after.reviews || []).toEqual(before.reviews || [])   // nothing reached the DB
  log('chip-post.json', { candidate: c.id, revealed: c.revealed, post: posts[0], chips: await chips.getByRole('button').allInnerTexts() })
})

// 2026-01-16 09:32 (CR-CZ card 18): pre-reveal comment + one post-reveal note → Notes (2), the pre-reveal comment first.
// The same card checks the retired chip on a saved review: its detail is intercepted to add "take fade (called side)"
// to the saved reasons (browser only; the DB has no review carrying it).
const CARD_0116 = 'CR-CZ:cr-cz:cp-ranges:18'

test('notes: 2026-01-16 09:32 shows Notes (2) with the pre-reveal comment first; Saved keeps label + reasons', async ({ page, request }) => {
  const c = (await api(request, `/candidates/${encodeURIComponent(CARD_0116)}`)).candidate
  const blind = c.reviews.find((r) => !r.post_reveal)
  const post = c.reviews.filter((r) => r.post_reveal)
  expect(c.revealed).toBe(true)
  expect(blind.comment).toBeTruthy()
  expect(post.length).toBe(1)
  await openCard(page, c)
  const notes = page.getByTestId('review-notes')
  await expect(notes).toContainText('Notes (2)')
  const items = page.getByTestId('review-note')
  await expect(items).toHaveCount(2)
  await expect(items.nth(0)).toHaveAttribute('data-before-reveal', 'true')
  await expect(items.nth(0)).toContainText('before reveal')
  await expect(items.nth(0)).toContainText(blind.comment.slice(0, 40))
  await expect(items.nth(1)).toContainText(post[0].comment.slice(0, 40))
  const saved = page.getByTestId('review-saved')
  await expect(saved).toContainText(blind.label.toUpperCase())
  await expect(saved).toContainText('take opposite')      // stored "take continuation (other edge)", displayed
  await expect(saved).not.toContainText(STORED)
  await expect(saved).not.toContainText(blind.comment.slice(0, 40))
  await page.getByTestId('review-active').screenshot({ path: path.join(LOGS, 'notes-0116.png') })
  log('notes-0116.json', { saved: await saved.innerText(), notes: await items.allInnerTexts() })
})

test('retired chip: a saved review carrying "take fade (called side)" still shows it (intercepted detail)', async ({ page, request }) => {
  const c = (await api(request, `/candidates/${encodeURIComponent(CARD_0116)}`)).candidate
  await page.route(`**/api/setup-review/candidates/${encodeURIComponent(CARD_0116)}`, async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    for (const rv of body.candidate.reviews) if (!rv.post_reveal) rv.reasons = [...rv.reasons, RETIRED]
    await route.fulfill({ response: r, json: body })
  })
  await openCard(page, c)
  const saved = page.getByTestId('review-saved')
  await expect(saved).toContainText(RETIRED)
  await expect(saved).toContainText('take opposite')
  await expect(page.getByTestId('review-reasons').getByRole('button', { name: RETIRED })).toHaveCount(0)
  log('retired-chip.json', { saved: await saved.innerText() })
})

// ── Item 3: Levels: Corrected with no corrected lines → the API's reason on the price chart.
for (const v of [
  { day: '2024-03-15', levels: 'corrected', notice: 'No corrected GEX levels for this date (pre-forward era) — switch Levels to Current' },
  { day: '2024-03-15', levels: 'current', notice: null },
  { day: '2026-10-27', levels: 'corrected', notice: null },
]) {
  test(`levels notice: ${v.day} ${v.levels} → ${v.notice ? 'notice' : 'none'}`, async ({ page }) => {
    await open(page, v.day, { levels: v.levels })
    await settle(page)
    const n = page.getByTestId('levels-notice')
    if (v.notice) {
      await expect(n).toHaveText(v.notice)
      await page.screenshot({ path: path.join(LOGS, `levels-notice-${v.day}.png`) })
    } else {
      await expect(n).toHaveCount(0)
    }
  })
}
