import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import type { IncomingMessage, ServerResponse } from "node:http";

// The backend is reached through the SSH tunnel on this laptop's port 8000.
// 127.0.0.1 (not "localhost") avoids Node picking IPv6 ::1 when the tunnel only
// listens on IPv4. Override with BACKEND_URL=http://host:port npm run dev
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
// Set FRONTEND_HOST=0.0.0.0 to let the presenter's PHONE open /presenter on the laptop's LAN address.
const host = process.env.FRONTEND_HOST ?? "127.0.0.1";

// When the backend dies mid-stream (server restart, tunnel drop), the dev proxy
// would otherwise keep the browser's side of an SSE stream open forever, so the
// page never notices. Tear the browser connection down when upstream closes.
type ProxyLike = {
  on(event: "proxyRes", cb: (proxyRes: IncomingMessage, req: IncomingMessage, res: ServerResponse) => void): void;
};
function closeClientWhenUpstreamCloses(proxy: ProxyLike) {
  proxy.on("proxyRes", (proxyRes, _req, res) => {
    proxyRes.on("close", () => {
      if (!proxyRes.complete) res.destroy();
    });
  });
}

// /admin is BOTH the admin page (browser navigation) and the admin API prefix. Page loads ask for text/html:
// serve the SPA for those, proxy everything else (fetch calls) to the backend.
const spaBypass = (req: IncomingMessage) => (req.headers.accept?.includes("text/html") ? "/index.html" : undefined);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host,
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true, configure: closeClientWhenUpstreamCloses },
      "/admin": { target: backend, changeOrigin: true, configure: closeClientWhenUpstreamCloses, bypass: spaBypass },
    },
  },
  build: { chunkSizeWarningLimit: 1500 },
  test: { environment: "node" },
});
