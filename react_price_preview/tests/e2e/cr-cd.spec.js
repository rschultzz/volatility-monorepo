// CR-CD step 4 — Flow v2 smoke (specs/CR-CD-order-flow-pane-v2.md). Read-only.
//
// Servers (started outside Playwright):
//   backend   apps/web/.venv/bin/python -u scripts/cr_cd_smoke_backend.py   (port 8062, read-only DB)
//   frontend  npm --prefix react_price_preview run dev -- --port 5177 --strictPort
// then:      CRBV_BASE_URL=http://localhost:5177 npx playwright test cr-cd   (from react_price_preview/)
//
// "toggle": default Current (no mode on the flow requests); v2 sends mode=v2 and shows the v2
//   pane with its source badge; markers start off; the choice survives a reload; back to Current.
// "markers + settings": markers on draws D/P/F; the settings dialog changes the m_* parameters.
// "screenshots": 2026-05-01, 2026-04-01, 2026-03-02 (train), RTH framed, markers on; per-session
//   marker counts for the defaults and for A1 as written go to specs/assets/cr-cd/smoke-counts.json.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCD_API_BASE || 'http://127.0.0.1:8062'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-cd')
const DAYS = ['2026-05-01', '2026-04-01', '2026-03-02']

function shifted(ptIso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(ptIso)
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000
}

async function open(page, day, { fresh = true, mode } = {}) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(({ freshStorage, initialMode }) => {
    try {
      if (freshStorage && !window.sessionStorage.getItem('crcd-init')) {
        window.localStorage.removeItem('ib-react-flow-mode')
        window.localStorage.removeItem('ib-react-flow-v2-markers')
        if (initialMode) window.localStorage.setItem('ib-react-flow-mode', initialMode)
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        window.sessionStorage.setItem('crcd-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, { freshStorage: fresh, initialMode: mode })
  await page.goto(`/?trade_date=${day}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
}

async function settle(page) {
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
  await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// Frame the price chart (and so both linked panes) to [from, to] shifted PT times (test-only).
async function frame(page, from, to) {
  await page.evaluate(({ a, b }) => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    const find = (pred) => {
      let f = chartEl.parentElement[key]
      for (let d = 0; f && d < 30; d += 1, f = f.return) {
        for (let h = f.memoizedState; h; h = h.next) {
          const v = h.memoizedState
          const c = v && typeof v === 'object' && 'current' in v ? v.current : null
          if (c && pred(c)) return c
        }
      }
      return null
    }
    const chart = find((c) => typeof c.timeScale === 'function')
    const series = find((c) => typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick')
    const data = series.data().filter((x) => x.close != null)
    const i0 = data.findIndex((x) => x.time >= a)
    let i1 = data.findIndex((x) => x.time >= b)
    if (i1 === -1) i1 = data.length - 1
    chart.timeScale().setVisibleLogicalRange({ from: i0 - 2, to: i1 + 2 })
  }, { a: from, b: to })
  await page.waitForTimeout(800)
}

test('toggle: default Current, v2 sends mode=v2, persists, back to Current', async ({ page }) => {
  const flowUrls = []
  page.on('request', (r) => {
    if (r.url().includes('/api/ironbeam/flow')) flowUrls.push(r.url())
  })
  await open(page, '2026-05-01')
  await settle(page)
  await expect(page.getByTestId('flow-mode-current')).toHaveAttribute('aria-pressed', 'true')
  expect(flowUrls.length).toBeGreaterThan(0)
  expect(flowUrls.every((u) => !u.includes('mode='))).toBeTruthy()

  flowUrls.length = 0
  await page.getByTestId('flow-mode-v2').click()
  await expect(page.getByTestId('flow-v2-panel')).toBeVisible()
  await settle(page)
  expect(flowUrls.some((u) => u.includes('mode=v2'))).toBeTruthy()
  await expect(page.getByTestId('flow-v2-source')).toHaveText('Databento')
  await expect(page.getByTestId('flow-v2-markers-toggle')).toHaveAttribute('aria-pressed', 'false')

  await page.reload()
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await expect(page.getByTestId('flow-v2-panel')).toBeVisible()          // remembered

  await page.getByTestId('flow-mode-current').click()
  await expect(page.getByTestId('flow-v2-panel')).toHaveCount(0)
  await expect(page.getByTestId('flow-mode-current')).toHaveAttribute('aria-pressed', 'true')
})

test('markers on, settings change the request parameters, guide opens', async ({ page }) => {
  await open(page, '2026-05-01', { mode: 'v2' })
  await settle(page)
  await page.getByTestId('flow-v2-markers-toggle').click()
  await expect(page.getByTestId('flow-v2-markers-toggle')).toHaveAttribute('aria-pressed', 'true')

  const req = page.waitForRequest((r) => r.url().includes('mode=v2') && r.url().includes('m_swing=4'))
  await page.getByRole('button', { name: 'Open Flow v2 marker settings' }).click()
  await page.getByRole('button', { name: 'Loose (A1)' }).click()
  await page.getByRole('button', { name: 'Apply' }).click()
  const u = new URL((await req).url())
  expect(u.searchParams.get('m_gap')).toBe('15')
  expect(u.searchParams.get('m_push')).toBe('0')

  await page.getByRole('button', { name: 'How to read Flow v2' }).click()
  await expect(page.getByRole('dialog', { name: 'How to read Flow v2' })).toContainText('visual aids')
  await expect(page.getByRole('dialog', { name: 'How to read Flow v2' })).toContainText('opposite sign')
})

test('screenshots and marker counts on 3 train days', async ({ page, request }) => {
  const counts = {}
  for (const day of DAYS) {
    const row = {}
    for (const [name, qs] of [['defaults', ''], ['a1_as_written', '&m_swing=4&m_push=0&m_div=0&m_fade=0&m_gap=15']]) {
      const r = await request.get(`${API_BASE}/api/ironbeam/flow?trade_date=${day}&session=FULL&mode=v2${qs}`, { timeout: 120_000 })
      expect(r.ok()).toBeTruthy()
      const p = await r.json()
      const by = {}
      for (const m of p.markers) by[m.kind] = (by[m.kind] || 0) + 1
      row[name] = { total: p.markers.length, ...by }
      row.source = p.source
      row.symbol = p.symbol
      row.absorption_bars = p.bars.filter((b) => b.absorb).length
      row.baseline = p.baseline && { sessions: p.baseline.sessions, abs_median: p.baseline.abs_median, abs_p20: p.baseline.abs_p20 }
    }
    counts[day] = row

    await page.context().clearCookies()
    await open(page, day, { mode: 'v2' })
    await settle(page)
    const toggle = page.getByTestId('flow-v2-markers-toggle')
    if ((await toggle.getAttribute('aria-pressed')) !== 'true') await toggle.click()   // persisted across days
    await expect(toggle).toHaveAttribute('aria-pressed', 'true')
    await page.waitForTimeout(500)
    await frame(page, shifted(`${day}T06:30`), shifted(`${day}T12:59`))
    await page.screenshot({ path: path.join(ASSETS, `v2-${day}-rth.png`) })
    if (day === '2026-05-01') {
      await frame(page, shifted(`${day}T07:00`), shifted(`${day}T08:45`))
      await page.screenshot({ path: path.join(ASSETS, `v2-${day}-0700-0845.png`) })
    }
  }
  fs.writeFileSync(path.join(ASSETS, 'smoke-counts.json'), JSON.stringify(counts, null, 2) + '\n')
  for (const day of DAYS) expect(counts[day].source).toBe('databento')
})

// A3 — review replay fits the price axis to the setup zone ± max(25, 0.5·IM). The cc:fixture
// candidates are all revealed, so the review API responses are intercepted to present candidate 5
// (2026-05-01) as unreviewed (read-only: nothing is posted). Needs scripts/cr_cc_smoke_backend.py
// on 8061 (review routes); CRCD_REVIEW_API_BASE overrides.
const REVIEW_API = process.env.CRCD_REVIEW_API_BASE || 'http://127.0.0.1:8061'

async function priceRange(page) {
  return page.evaluate(() => {
    const chartEl = document.querySelector('.tv-lightweight-charts')
    const key = Object.keys(chartEl.parentElement).find((k) => k.startsWith('__reactFiber'))
    let f = chartEl.parentElement[key]
    for (let d = 0; f && d < 30; d += 1, f = f.return) {
      for (let h = f.memoizedState; h; h = h.next) {
        const v = h.memoizedState
        const c = v && typeof v === 'object' && 'current' in v ? v.current : null
        if (c && typeof c.timeScale === 'function') {
          const s = c.priceScale('right')
          return { ...s.getVisibleRange(), auto: s.options().autoScale }
        }
      }
    }
    return null
  })
}

test('A3: replay fits the price axis to the setup zone; exit restores autoscale', async ({ page }) => {
  const asUnreviewed = (c) => {
    const { outcome, reviews, ...rest } = c
    return { ...rest, reviewed: false, revealed: false, n_reviews: 0, label: null, revealed_at: null, reviews: [] }
  }
  await page.route('**/api/setup-review/candidates?*', async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    body.candidates = body.candidates.map((c) => (c.id === 'CR-CC:cc:fixture:5' ? asUnreviewed(c) : c))
    await route.fulfill({ response: r, json: body })
  })
  await page.route('**/api/setup-review/candidates/CR-CC%3Acc%3Afixture%3A5', async (route) => {
    const r = await route.fetch()
    const body = await r.json()
    body.candidate = asUnreviewed(body.candidate)
    await route.fulfill({ response: r, json: body })
  })

  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto(`/?trade_date=2026-10-23&interval=1min&api_base=${encodeURIComponent(REVIEW_API)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await settle(page)
  const normal = await priceRange(page)
  expect(normal.auto).toBe(true)

  await page.getByRole('button', { name: 'Show setup review panel' }).click()
  await page.getByTestId('review-panel').getByLabel('Set', { exact: true }).selectOption('cc:fixture')
  const detail = page.waitForResponse((r) => r.url().includes('/candidates/CR-CC%3Acc%3Afixture%3A5'))
  await page.locator('[data-testid="review-row"][data-candidate-id="CR-CC:cc:fixture:5"]').click()
  const cand = (await (await detail).json()).candidate
  await expect(page.getByTestId('review-banner')).toContainText('Review: 2026-05-01')
  await settle(page)

  const pad = Math.max(25, 0.5 * cand.implied_move_1d)
  const prices = [cand.level_price, cand.extreme_price, cand.entry_price, cand.stop_price]
  const fit = await priceRange(page)
  expect(fit.auto).toBe(false)
  expect(fit.from).toBeCloseTo(Math.min(...prices) - pad, 1)
  expect(fit.to).toBeCloseTo(Math.max(...prices) + pad, 1)
  expect(fit.to - fit.from).toBeLessThan(normal.to - normal.from)
  await page.screenshot({ path: path.join(ASSETS, 'a3-replay-fit-2026-05-01.png') })

  await page.getByRole('button', { name: 'Exit review' }).click()
  await expect(page.getByTestId('review-banner')).toHaveCount(0)
  await settle(page)
  expect((await priceRange(page)).auto).toBe(true)
})
