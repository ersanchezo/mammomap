import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // Proxy /api calls to the Modal backend in local dev
  // so you never have to deal with CORS during development
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target:      process.env.VITE_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
        secure:      false,
      },
    },
  },

  // Expose env variables prefixed with VITE_ to the browser bundle
  // Set VITE_API_URL in .env.local (dev) or the Vercel dashboard (prod)
  envPrefix: "VITE_",

  build: {
    outDir:        "dist",
    sourcemap:     true,
    rollupOptions: {
      output: {
        // Split large vendor chunks for better caching
        manualChunks: {
          react: ["react", "react-dom"],
        },
      },
    },
  },
});
