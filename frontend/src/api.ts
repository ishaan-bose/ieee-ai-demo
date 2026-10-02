// Types mirror backend/app/api/health.py (SPEC 7). Later phases generate these
// from shared/openapi.json.
export interface Health {
  ok: boolean;
  device: "cuda" | "cpu";
  gpu_name: string | null;
  vram_gb: number | null;
  queue_length: number;
  demo_mode: boolean;
  paused: boolean;
  version: string;
}

export interface Tick {
  count: number;
  server_time: number;
}

export async function getHealth(timeoutMs = 4000): Promise<Health> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch("/api/health", { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as Health;
  } finally {
    clearTimeout(timer);
  }
}
