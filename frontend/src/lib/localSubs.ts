// Every submission is saved to localStorage BEFORE it is sent (SPEC 5.1). Pending ones upload automatically when the server returns.
import type { Config } from "./chessConfig";

export interface LocalSub {
  client_id: string;
  nickname: string; model_name: string; contact?: string; consent: boolean; config: Config;
  created_at: number;
  status: "pending" | "sent";
  submission_id?: string; participant_code?: string; param_count?: number; tier?: string;
}

const KEY = "byoai_submissions_v1";

export function loadLocal(): LocalSub[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? "[]") as LocalSub[]; } catch { return []; }
}
function store(list: LocalSub[]) { try { localStorage.setItem(KEY, JSON.stringify(list)); } catch { /* private mode: nothing to do */ } }

export function saveLocal(sub: Omit<LocalSub, "status" | "created_at">): LocalSub {
  const list = loadLocal();
  const rec: LocalSub = { ...sub, created_at: Date.now(), status: "pending" };
  store([...list.filter((s) => s.client_id !== rec.client_id), rec]);
  return rec;
}

export function markSent(client_id: string, res: { submission_id: string; participant_code: string; param_count: number; tier: string }) {
  // pick the fields explicitly: the server's response has its own `status` ("queued") that must not overwrite ours
  store(loadLocal().map((s) => (s.client_id === client_id
    ? { ...s, status: "sent" as const, submission_id: res.submission_id, participant_code: res.participant_code, param_count: res.param_count, tier: res.tier } : s)));
}

export const pendingSubs = () => loadLocal().filter((s) => s.status === "pending");

export function backupJson(): string {
  return JSON.stringify(loadLocal(), null, 2);
}

export function newClientId(): string {
  return (crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`).replace(/-/g, "");
}
