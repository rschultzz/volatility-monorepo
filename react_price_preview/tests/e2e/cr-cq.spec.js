// CR-CQ — price chart Min |GEX| filter defaults to 40 and never silently resets to 0
// (specs/CR-CQ-gex-filter-default-40.md). Read-only.
//
// Servers (started outside Playwright):
//   backend   CRCK_PORT=8066 apps/web/.venv/bin/python -u scripts/cr_ck_smoke_backend.py   (read-only DB)
//   frontend  npm --prefix react_price_preview run dev -- --port 5176 --strictPort
// then:      npx playwright test cr-cq   (from react_price_preview/)
//
// The value in force is read two ways: the settings slider and the gex_min_abs_b the chart sends to
// /api/ironbeam/bars (what is actually plotted). Every test starts from a fresh browser context.
import { test, expect } from '@playwright/test'

const API_BASE = process.env.CRCQ_API_BASE || 'http://127.0.0.1:8066'
const BASE_URL = process.env.CRCQ_BASE_URL || 'http://localhost:5176'
const DAY = '2026-10-26'

function chartUrl(extra = '') {
  return `${BASE_URL}/?trade_date=${DAY}&interval=1min&days_either_side=0&api_base=${encodeURIComponent(API_BASE)}${extra}`
}

// Loads the chart and returns the gex_min_abs_b of the last /api/ironbeam/bars request.
async function load(page, extra = '') {
  const sent = []
  const onRequest = (r) => {
    if (r.url().includes('/api/ironbeam/bars')) sent.push(new URL(r.url()).searchParams.get('gex_min_abs_b'))
  }
  page.on('request', onRequest)
  await page.goto(chartUrl(extra))
  await page.locator('.tv-lightweight-charts').first().waitFor({ timeout: 60_000 })
  await expect.poll(() => sent.length, { timeout: 60_000 }).toBeGreaterThan(0)
  await page.waitForLoadState('networkidle', { timeout: 30_000 }).catch(() => {})
  page.off('request', onRequest)
  return sent[sent.length - 1]
}

const slider = (page) => page.locator('label:has-text("Min |GEX| (B)") + input[type="range"]').first()

async function sliderValue(page) {
  await page.getByRole('button', { name: 'Open price chart settings' }).click()
  const v = await slider(page).inputValue()
  await page.getByRole('button', { name: 'Cancel' }).click()
  return v
}

async function applySlider(page, value) {
  await page.getByRole('button', { name: 'Open price chart settings' }).click()
  await slider(page).fill(String(value))
  await page.getByRole('button', { name: 'Apply' }).click()
  // standalone (no parent Dash) the dialog stays open after Apply with a sync note; the value is already applied
  if (await page.getByRole('button', { name: 'Cancel' }).count()) await page.getByRole('button', { name: 'Cancel' }).click()
}

const stored = (page) => page.evaluate(() => ({
  v2: window.localStorage.getItem('ib-react-gex-min-abs-b-v2'),
  legacy: window.localStorage.getItem('ib-react-gex-min-abs-b'),
}))

test('fresh profile shows 40 and plots with 40', async ({ page }) => {
  expect(await load(page)).toBe('40')
  expect(await sliderValue(page)).toBe('40')
  expect((await stored(page)).v2).toBe('40')
})

test('set 60 → refresh → 60', async ({ page }) => {
  await load(page)
  await applySlider(page, 60)
  expect((await stored(page)).v2).toBe('60')
  expect(await load(page)).toBe('60')
  expect(await sliderValue(page)).toBe('60')
})

test('cleared / invalid stored value → refresh → previous value or 40, never 0', async ({ page }) => {
  // A blank or junk v2 entry falls back to the legacy key's last positive value (here 60) ...
  await page.addInitScript(() => {
    if (window.sessionStorage.getItem('crcq-init')) return
    window.sessionStorage.setItem('crcq-init', '1')
    window.localStorage.setItem('ib-react-gex-min-abs-b', '60')
    window.localStorage.setItem('ib-react-gex-min-abs-b-v2', '')
  })
  expect(await load(page)).toBe('60')
  // ... and with nothing usable stored, to 40. Empty URL param (what an empty Dash input would pass) is ignored too.
  await page.evaluate(() => {
    window.localStorage.setItem('ib-react-gex-min-abs-b', 'abc')
    window.localStorage.setItem('ib-react-gex-min-abs-b-v2', '  ')
  })
  expect(await load(page, '&gex_min_abs_b=')).toBe('40')
  expect(await sliderValue(page)).toBe('40')
})

test('a 0 written by the old code is not trusted; an explicit 0 is kept', async ({ page }) => {
  // The pre-CR-CQ code wrote 0 into the legacy key on every load with empty storage.
  await page.addInitScript(() => {
    if (window.sessionStorage.getItem('crcq-init')) return
    window.sessionStorage.setItem('crcq-init', '1')
    window.localStorage.setItem('ib-react-gex-min-abs-b', '0')
  })
  expect(await load(page)).toBe('40')
  await applySlider(page, 0)
  expect((await stored(page)).v2).toBe('0')
  expect(await load(page)).toBe('0')
  expect(await sliderValue(page)).toBe('0')
})
