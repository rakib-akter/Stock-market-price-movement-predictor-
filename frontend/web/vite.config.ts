import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies "/api/*" to the FastAPI backend so the SPA and API
// share an origin during development (no CORS headaches). The backend base URL
// can be overridden with VITE_API_TARGET.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
