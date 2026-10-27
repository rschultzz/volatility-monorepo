// CR-CH — Flow v2 pane must not recompute its alignment on every App render (specs/CR-CH-price-chart-renderer-crash.md).
// Read-only.
//
// Servers (started outside Playwright):
//   backend    apps/web/.venv/bin/python -u scripts/cr_cd_smoke_backend.py   (port 8062, read-only DB)
//   frontend   npm --prefix react_price_preview run dev -- --port 5177 --strictPort
// then:       CRBV_BASE_URL=http://localhost:5177 npx playwright test cr-ch   (from react_price_preview/)
//
// Before the fix App passed `payloads={flowV2Payloads.map(...)}`, a new array every render, so the
// pane's allBars / allMarkers / aligned / snapped memos re-ran on every App render (3–6×/s during
// drag/zoom, ~150 ms each), which crashed Chrome 155's renderer. Here: interaction that re-renders App
// (wheel zoom and drag on the price chart) must leave the pane's payloads prop and every useMemo value
// in the pane referentially unchanged, while the pane itself did re-render (new props object).
import { test, expect } from '@playwright/test'

const API_BASE = process.env.CRCD_API_BASE || 'http://127.0.0.1:8062'

async function open(page, day, storage) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript((init) => {
    try {
      if (!window.sessionStorage.getItem('crch-init')) {
        for (const k of ['ib-react-flow-mode', 'ib-react-flow-v2-markers', 'ib-react-flow-v2-effort']) {
          window.localStorage.removeItem(k)
        }
        window.localStorage.setItem('ib-react-gex-min-abs-b', '40')
        for (const [k, v] of Object.entries(init)) window.localStorage.setItem(k, v)
        window.sessionStorage.setItem('crch-init', '1')
      }
    } catch (e) { /* ignore */ }
  }, storage)
  await page.goto(`/?trade_date=${day}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.waitForFunction(() => !document.querySelector('.status-card'), null, { timeout: 60_000 })
  await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
  await page.waitForTimeout(1500)
}

// The AggressorFlowV2Panel fiber's props, payloads and useMemo values (test-only, through React's fiber tree).
async function snapshotPane(page, slot) {
  return page.evaluate((s) => {
    const el = document.querySelector('[data-testid="flow-v2-panel"]')
    const key = Object.keys(el).find((k) => k.startsWith('__reactFiber'))
    let f = el[key]
    while (f && !(typeof f.type === 'function' && f.type.name === 'AggressorFlowV2Panel')) f = f.return
    if (!f) return null
    const memos = []
    for (let h = f.memoizedState; h; h = h.next) {
      const v = h.memoizedState
      // useMemo / useCallback store [value, deps]
      if (Array.isArray(v) && v.length === 2 && Array.isArray(v[1])) memos.push(v[0])
    }
    window.__crch = window.__crch || {}
    window.__crch[s] = { props: f.memoizedProps, payloads: f.memoizedProps.payloads, memos }
    return { memos: memos.length, payloads: f.memoizedProps.payloads?.length ?? null }
  }, slot)
}

test('Flow v2: interaction re-renders the pane without recomputing its alignment', async ({ page }) => {
  await open(page, '2026-05-01', { 'ib-react-flow-mode': 'v2' })
  await expect(page.getByTestId('flow-v2-panel')).toBeVisible()
  const before = await snapshotPane(page, 'before')
  expect(before).not.toBeNull()
  expect(before.payloads).toBeGreaterThan(0)
  expect(before.memos).toBeGreaterThanOrEqual(5)

  // wheel zoom + drag on the price pane: App's shared logical range / interaction state change
  const chart = page.locator('.tv-lightweight-charts').first()
  const box = await chart.boundingBox()
  const x = box.x + box.width * 0.5
  const y = box.y + box.height * 0.3
  for (let i = 0; i < 6; i += 1) {
    await page.mouse.move(x, y)
    await page.mouse.wheel(0, i % 2 ? 300 : -300)
    await page.waitForTimeout(120)
    await page.mouse.down()
    await page.mouse.move(x + (i % 2 ? 120 : -120), y, { steps: 6 })
    await page.mouse.up()
    await page.waitForTimeout(120)
  }
  await page.waitForTimeout(800)

  const after = await snapshotPane(page, 'after')
  const cmp = await page.evaluate(() => {
    const { before: b, after: a } = window.__crch
    return {
      rerendered: a.props !== b.props,
      samePayloads: a.payloads === b.payloads,
      changedMemos: a.memos.map((v, i) => (v === b.memos[i] ? null : i)).filter((i) => i !== null),
    }
  })
  expect(after.memos).toBe(before.memos)
  expect(cmp.rerendered).toBe(true)          // the interaction did re-render the pane
  expect(cmp.samePayloads).toBe(true)        // CR-CH: same payloads array
  expect(cmp.changedMemos).toEqual([])       // no memo (allBars, allMarkers, aligned, snapped, …) recomputed
})
