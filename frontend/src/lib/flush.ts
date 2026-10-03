import { api } from "./api";
import { markSent, pendingSubs } from "./localSubs";

let inflight: Promise<number> | null = null;

/** Upload submissions that were made while the server was unreachable (idempotent thanks to client_id). Returns how many went through.
 *  Concurrent callers share ONE in-flight upload, so every caller can safely re-read the local list when its promise resolves. */
export function flushPending(): Promise<number> {
  inflight ??= (async () => {
    let sent = 0;
    try {
      for (const s of pendingSubs()) {
        try {
          const res = await api.submit({ nickname: s.nickname, model_name: s.model_name, contact: s.contact, consent: s.consent, config: s.config, client_id: s.client_id });
          markSent(s.client_id, res);
          sent++;
        } catch (e) {
          const status = (e as { status?: number }).status;
          if (status === 0 || status === undefined) break; // still offline: try again later
          // a 4xx means the server rejected it for good; it stays pending so the participant sees it in the backup
        }
      }
    } finally { inflight = null; }
    return sent;
  })();
  return inflight;
}
