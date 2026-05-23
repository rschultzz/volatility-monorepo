import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'serve' ? '/' : '/react-preview/',
  resolve: {
    alias: {
      'web-shared': path.resolve(__dirname, '../packages/web-shared/src/index.js'),
      // Force shared-package deps to resolve from this app's node_modules.
      'lightweight-charts': path.resolve(__dirname, 'node_modules/lightweight-charts'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
}))