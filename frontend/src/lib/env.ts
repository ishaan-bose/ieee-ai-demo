// Runtime flags. `?mock=1` (sticky for the browser session) runs the whole demo with NO backend: a mock server lives in lib/mock.ts.
const q = typeof location !== "undefined" ? new URLSearchParams(location.search) : new URLSearchParams();

function sticky(name: string): boolean {
  try {
    if (q.get(name) === "1") sessionStorage.setItem(name, "1");
    if (q.get(name) === "0") sessionStorage.removeItem(name);
    return sessionStorage.getItem(name) === "1";
  } catch {
    return q.get(name) === "1";
  }
}

export const MOCK = sticky("mock");
export const urlParam = (k: string) => q.get(k);

// Simulated server outage in mock mode (key `O`), so the offline behaviour can be checked by eye.
let mockOffline = q.get("offline") === "1";
const listeners = new Set<() => void>();
export const isMockOffline = () => mockOffline;
export function setMockOffline(v: boolean) { mockOffline = v; listeners.forEach((l) => l()); }
export const onMockOfflineChange = (fn: () => void) => { listeners.add(fn); return () => listeners.delete(fn); };

export const BOOTH_URL: string = (import.meta.env.VITE_BOOTH_URL as string | undefined) ?? (typeof location !== "undefined" ? `${location.origin}/build` : "/build");
