import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: '/today-setup/',
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8060',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            const creds = Buffer.from('ryan:ChangeThisPassword123!').toString('base64')
            proxyReq.setHeader('Authorization', `Basic ${creds}`)
          })
        },
      },
    },
  },
  resolve: {
    alias: {
      'web-shared': path.resolve(__dirname, '../packages/web-shared/src/index.js'),
      // Force shared-package deps to resolve from this app's node_modules.
      'lightweight-charts': path.resolve(__dirname, 'node_modules/lightweight-charts'),
    },
  },
});
