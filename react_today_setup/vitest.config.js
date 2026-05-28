import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.js'],
  },
  resolve: {
    alias: {
      // Mirror the vite.config.js alias so components can import from 'web-shared'
      'web-shared': path.resolve(__dirname, '../packages/web-shared/src/index.js'),
      // Stub lightweight-charts so MiniPriceChart (via web-shared/index.js) doesn't
      // crash in jsdom — it uses DOM APIs unavailable in the test environment.
      'lightweight-charts': path.resolve(__dirname, 'src/__tests__/__mocks__/lightweight-charts.js'),
    },
  },
})
