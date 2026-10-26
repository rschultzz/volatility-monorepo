// CR-CK Part 3 — Overview "Stickiness" rows (specs/CR-CK-skew-stickiness.md C12–C14). Read-only.
//
// Servers (started outside Playwright):
//   backend           apps/web/.venv/bin/python -u scripts/cr_ck_smoke_backend.py        (port 8064, read-only DB)
//   frontend, branch  npm --prefix react_price_preview run dev -- --port 5178 --strictPort
//   frontend, main    the same from an origin/main checkout on port 5179 (CRCK_MAIN_URL)
// then:              npx playwright test cr-ck   (from react_price_preview/)
//
// "pixel": with the setting off (the default), the Overview panel on this branch is byte-identical to origin/main's,
//   and no request goes to /api/ironbeam/skew-stickiness.
// "on": the setting adds one row per selected time with the Part 2 numbers for 2026-10-26, in blue / slate only.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API_BASE = process.env.CRCK_API_BASE || 'http://127.0.0.1:8064'
const BRANCH_URL = process.env.CRCK_BRANCH_URL || 'http://localhost:5178'
const MAIN_URL = process.env.CRCK_MAIN_URL || 'http://localhost:5179'
const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../specs/assets/cr-ck')
const DAY = '2026-10-26'
const TIMES = ['07:30', '07:50', '08:48', '11:30']

async function open(page, base) {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.addInitScript(() => {
    try {
      if (!window.sessionStorage.getItem('crck-init')) {
        window.localStorage.clear()
        window.sessionStorage.setItem('crck-init', '1')
      }
    } catch (e) { /* ignore */ }
  })
  const q = `trade_date=${DAY}&selected_times=${TIMES.join(',')}&interval=1min&days_either_side=0`
  await page.goto(`${base}/?${q}&api_base=${encodeURIComponent(API_BASE)}`)
  await page.locator('.tv-lightweight-charts').first().waitFor()
  await page.getByRole('button', { name: 'Show SMILE' }).click()
  const win = page.getByTestId('smile-window')
  await win.waitFor({ timeout: 60_000 })
  await expect.poll(() => win.locator('tbody tr td:first-child').allInnerTexts(), { timeout: 60_000 })
    .toEqual(expect.arrayContaining(TIMES))
  await page.waitForLoadState('networkidle', { timeout: 60_000 }).catch(() => {})
  await page.waitForTimeout(1500)
  return win
}

test('pixel: setting off is identical to origin/main and makes no stickiness request', async ({ browser }) => {
  fs.mkdirSync(ASSETS, { recursive: true })
  const shots = {}
  for (const [name, base] of [['main', MAIN_URL], ['branch', BRANCH_URL], ['branch2', BRANCH_URL]]) {
    const ctx = await browser.newContext()
    const page = await ctx.newPage()
    const sticky = []
    page.on('request', (r) => { if (r.url().includes('/api/ironbeam/skew-stickiness')) sticky.push(r.url()) })
    const win = await open(page, base)
    shots[name] = await win.screenshot({ path: path.join(ASSETS, `overview-off-${name}.png`) })
    if (name !== 'main') expect(sticky).toEqual([])
    await ctx.close()
  }
  expect(shots.branch.equals(shots.branch2)).toBe(true)          // the capture itself is deterministic
  expect(shots.branch.equals(shots.main)).toBe(true)
})

test('on: one stickiness row per selected time, Part 2 numbers, Dash palette only', async ({ page }) => {
  const win = await open(page, BRANCH_URL)
  const overlap = async () => {
    const chartBox = await win.locator('svg').first().boundingBox()
    const tableBox = await win.locator('table').boundingBox()
    return { chart: chartBox.height, overlap: chartBox.y + chartBox.height - tableBox.y }
  }
  const off = await overlap()
  await page.getByRole('button', { name: 'Open price chart settings' }).click()
  await page.getByLabel(/Stickiness/).check()
  await page.getByRole('button', { name: 'Apply' }).click()
  // standalone (no parent Dash) the dialog stays open after Apply with a sync note; the setting is already applied
  if (await page.getByRole('button', { name: 'Cancel' }).count()) await page.getByRole('button', { name: 'Cancel' }).click()
  const rows = win.locator('tr[data-stickiness-row="1"]')
  await expect(rows).toHaveCount(TIMES.length, { timeout: 30_000 })
  await expect(rows.nth(1)).toContainText('ratio +0.38', { timeout: 30_000 })
  const texts = await rows.allInnerTexts()
  expect(texts[0]).toContain('needs a previous time')
  expect(texts[1]).toContain('Δput -0.47')
  expect(texts[1]).toContain('SS -1.23')
  expect(texts[1]).toContain('in noise')
  expect(texts[2]).toContain('move too small')
  expect(texts[2]).not.toContain('late-day')
  expect(texts[3]).toContain('late-day: low reliability')     // A1.7: window 08:48-11:30 ends after 11:00
  // every colour inside the stickiness rows is blue / light grey / slate
  const colors = await rows.evaluateAll((els) => {
    const out = new Set()
    for (const el of els) for (const n of [el, ...el.querySelectorAll('*')]) out.add(getComputedStyle(n).color)
    return [...out]
  })
  const allowed = ['rgb(148, 163, 184)', 'rgb(96, 165, 250)', 'rgb(100, 116, 139)', 'rgb(226, 232, 240)']
  for (const c of colors) expect(allowed).toContain(c)
  // the window grows with the extra rows: the smile keeps its size and overlaps the table no more than when off
  const on = await overlap()
  expect(Math.abs(on.chart - off.chart)).toBeLessThanOrEqual(1)
  expect(on.overlap).toBeLessThanOrEqual(off.overlap + 1)
  await win.screenshot({ path: path.join(ASSETS, 'overview-on-branch.png') })
  // the setting survives a reload
  await page.reload()
  await page.getByRole('button', { name: 'Show SMILE' }).click()
  await expect(page.getByTestId('smile-window').locator('tr[data-stickiness-row="1"]')).toHaveCount(TIMES.length,
    { timeout: 60_000 })
})
