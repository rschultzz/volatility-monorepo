// CR-BV smoke tests (Playwright). They drive the real price chart against a running
// backend, so both servers are started outside Playwright:
//   backend   apps/web/.venv/bin/python -u scripts/cr_bv_smoke_backend.py   (read-only DB, port 8060)
//   frontend  npm --prefix react_price_preview run dev -- --port 5175 --strictPort
// then:      npx playwright test   (from react_price_preview/)
// CRBV_BASE_URL / CRBV_API_BASE override the two addresses.
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.CRBV_BASE_URL || 'http://localhost:5175',
    browserName: 'chromium',
  },
})
