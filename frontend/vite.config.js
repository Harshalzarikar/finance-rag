import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Route API calls to the FastAPI server during development so the app always
    // uses one relative base path, identical to production.
    //
    // Use 127.0.0.1 (not localhost): on Windows localhost resolves to IPv6 and
    // can land on a different listener than the API. Override the port with
    // VITE_PROXY_TARGET if the API runs elsewhere.
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
