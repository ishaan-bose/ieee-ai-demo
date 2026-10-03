import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import { onMockOfflineChange } from "../lib/env";
import { flushPending } from "../lib/flush";
import type { Health } from "../lib/types";

interface Ctx { online: boolean; health: Health | null; checked: boolean }
const HealthCtx = createContext<Ctx>({ online: false, health: null, checked: false });
export const useHealth = () => useContext(HealthCtx);

/** Polls /api/health every few seconds (SPEC 5.1). Offline is a NORMAL state: races switch to recordings, browser-only acts keep working,
 *  and pending submissions upload by themselves when the server comes back. */
export function HealthProvider({ children, intervalMs = 3000 }: { children: ReactNode; intervalMs?: number }) {
  const [state, setState] = useState<Ctx>({ online: false, health: null, checked: false });
  const was = useRef(false);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const h = await api.health();
        if (stop) return;
        setState({ online: true, health: h, checked: true });
        if (!was.current) flushPending().catch(() => undefined);
        was.current = true;
      } catch {
        if (stop) return;
        setState({ online: false, health: null, checked: true });
        was.current = false;
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    const off = onMockOfflineChange(tick);
    return () => { stop = true; clearInterval(id); off(); };
  }, [intervalMs]);
  return <HealthCtx.Provider value={state}>{children}</HealthCtx.Provider>;
}
