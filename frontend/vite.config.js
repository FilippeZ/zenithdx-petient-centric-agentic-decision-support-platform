import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/login': 'http://localhost:8000',
      '/register': 'http://localhost:8000',
      '/upload': 'http://localhost:8000',
      '/uploads': 'http://localhost:8000',
      '/patients': 'http://localhost:8000',
      '/reports': 'http://localhost:8000',
      '/doctor': 'http://localhost:8000',
    }
  }
})
