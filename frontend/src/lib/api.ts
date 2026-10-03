// Typed client for the backend contract (SPEC 7). With ?mock=1 every call is answered by lib/mock.ts instead.
import { MOCK } from "./env";
import { mockApi } from "./mock";
import { sseEvents } from "./sse";
import { ApiError, type AdminQueue, type AdminStatus, type Health, type PresenterState, type Preview, type RaceEvent, type RaceRequest, type SubmissionIn, type SubmissionOut, type SubmissionStatus } from "./types";
import type { Config } from "./chessConfig";

async function call<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}, timeoutMs = 8000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(path, { method, signal: ctrl.signal, headers: { Accept: "application/json", ...(body !== undefined ? { "Content-Type": "application/json" } : {}), ...headers }, body: body !== undefined ? JSON.stringify(body) : undefined });
    const text = await res.text();
    let json: unknown = null;
    try { json = text ? JSON.parse(text) : null; } catch { /* non-JSON error page (proxy down) */ }
    if (!res.ok) {
      const e = (json ?? {}) as { error?: string; message?: string };
      throw new ApiError(res.status, e.error ?? "http_error", e.message ?? `HTTP ${res.status}`);
    }
    return json as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(0, "offline", e instanceof Error ? e.message : "network error");
  } finally {
    clearTimeout(timer);
  }
}

const real = {
  health: () => call<Health>("GET", "/api/health", undefined, {}, 4000),
  createRace: (b: RaceRequest) => call<{ race_id: string }>("POST", "/api/demo/races", b),
  raceStream: async function* (id: string, signal?: AbortSignal): AsyncGenerator<RaceEvent> {
    for await (const e of sseEvents(`/api/demo/races/${id}/stream`, {}, signal)) yield e as RaceEvent;
  },
  abortRace: (id: string) => call<{ ok: boolean }>("POST", `/api/demo/races/${id}/abort`),
  preview: (c: Config) => call<Preview>("POST", "/api/submissions/preview", c),
  submit: (b: SubmissionIn) => call<SubmissionOut>("POST", "/api/submissions", b),
  submission: (id: string) => call<SubmissionStatus>("GET", `/api/submissions/${id}`),
  presenterGet: () => call<PresenterState>("GET", "/api/presenter", undefined, {}, 2500),
  presenterPut: (s: Omit<PresenterState, "updated_at">) => call<PresenterState>("PUT", "/api/presenter", s, {}, 2500),
  configSchema: () => call<{ defaults: Config }>("GET", "/api/config/schema"),
};

// ---- admin (X-Admin-Token kept in memory only, entered once)
let adminToken = "";
export const setAdminToken = (t: string) => { adminToken = t; };
export const hasAdminToken = () => adminToken.length > 0;
const A = () => ({ "X-Admin-Token": adminToken });
const realAdmin = {
  queue: () => call<AdminQueue>("GET", "/admin/queue", undefined, A()),
  kill: (id: number) => call<{ ok: boolean; status: string }>("POST", `/admin/jobs/${id}/kill`, undefined, A()),
  remove: (id: number) => call<{ ok: boolean; status: string }>("DELETE", `/admin/jobs/${id}`, undefined, A()),
  redo: (id: number) => call<{ ok: boolean; status: string }>("POST", `/admin/jobs/${id}/redo`, undefined, A()),
  pause: () => call<{ ok: boolean }>("POST", "/admin/queue/pause", undefined, A()),
  resume: () => call<{ ok: boolean }>("POST", "/admin/queue/resume", undefined, A()),
  demoMode: (enabled: boolean) => call<{ ok: boolean }>("POST", "/admin/demo-mode", { enabled }, A()),
  reset: () => call<{ ok: boolean; snapshot: string }>("POST", "/admin/reset", { confirm: "RESET" }, A(), 30000),
  exportAll: async (format: "json" | "csv" = "json"): Promise<Blob> => {
    const res = await fetch(`/admin/export?format=${format}`, { headers: A() });
    if (!res.ok) throw new ApiError(res.status, "export_failed", `HTTP ${res.status}`);
    return res.blob();
  },
  stream: async function* (signal?: AbortSignal): AsyncGenerator<AdminStatus> {
    for await (const e of sseEvents("/admin/stream", { headers: A() }, signal)) if (e.event === "status") yield e.data as AdminStatus;
  },
};

export const api = MOCK ? { ...mockApi, admin: mockApi.admin as unknown as typeof realAdmin } : { ...real, admin: realAdmin };
export { ApiError };
