import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { STAGES, stageIndex, unlockedAt, type Stage } from "./stages";
import { urlParam } from "../lib/env";
import { api } from "../lib/api";

type Ev = "run" | "skip" | "reset";
type Handler = () => void;

interface Ctx {
  stages: Stage[]; index: number; stage: Stage; unlocked: Set<string>;
  go(d: number): void; jump(id: string): void;
  resetNonce: number; forceFallback: boolean; presenterMode: boolean;
  menuOpen: boolean; setMenuOpen(v: boolean): void;
  on(ev: Ev, h: Handler): () => void;
  emit(ev: Ev): void;
}
const StageCtx = createContext<Ctx>(null as unknown as Ctx);
export const useStage = () => useContext(StageCtx);

/** Subscribe to Space (run), S (skip to cached result) or R (reset stage) while the component is mounted. */
export function useStageEvent(ev: Ev, handler: Handler) {
  const { on } = useStage();
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => on(ev, () => ref.current()), [on, ev]);
}

/** `<Unlock id="gauge">…</Unlock>`: children exist only once their id is unlocked; they pop in with a short animation. */
export function useUnlocked(id: string): boolean {
  return useStage().unlocked.has(id);
}

const STORE_KEY = "byoai_stage";
export const CHANNEL = "byoai-stage";

export function StageProvider({ children }: { children: ReactNode }) {
  const [index, setIndex] = useState(() => {
    const p = urlParam("stage");
    if (p && stageIndex(p) >= 0) return stageIndex(p);
    try { const v = Number(sessionStorage.getItem(STORE_KEY)); return Number.isInteger(v) && v >= 0 && v < STAGES.length ? v : 0; } catch { return 0; }
  });
  const [resetNonce, setResetNonce] = useState(0);
  const [fallbacks, setFallbacks] = useState<Record<string, boolean>>({});
  const [menuOpen, setMenuOpen] = useState(false);
  const [presenterMode, setPresenterMode] = useState(() => { try { return localStorage.getItem("byoai_presenter") === "1"; } catch { return false; } });
  const handlers = useRef<Record<Ev, Set<Handler>>>({ run: new Set(), skip: new Set(), reset: new Set() });

  const go = useCallback((d: number) => setIndex((i) => Math.min(STAGES.length - 1, Math.max(0, i + d))), []);
  const jump = useCallback((id: string) => { const i = stageIndex(id); if (i >= 0) { setIndex(i); setMenuOpen(false); } }, []);
  const on = useCallback((ev: Ev, h: Handler) => { handlers.current[ev].add(h); return () => { handlers.current[ev].delete(h); }; }, []);
  const emit = useCallback((ev: Ev) => handlers.current[ev].forEach((h) => h()), []);
  const stage = STAGES[index];

  useEffect(() => {
    try { sessionStorage.setItem(STORE_KEY, String(index)); } catch { /* ignore */ }
    // Publish the stage so the presenter's phone (/presenter) and other tabs follow along.
    const msg = { stage_id: stage.id, stage_index: index, title: stage.title };
    try { const ch = new BroadcastChannel(CHANNEL); ch.postMessage(msg); ch.close(); } catch { /* old browser */ }
    api.presenterPut(msg).catch(() => undefined);
  }, [index, stage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "ArrowRight": e.preventDefault(); go(1); break;
        case "ArrowLeft": e.preventDefault(); go(-1); break;
        case " ": e.preventDefault(); emit("run"); break;
        case "s": case "S": emit("skip"); break;
        case "r": case "R": setResetNonce((n) => n + 1); emit("reset"); break;
        case "g": case "G": setMenuOpen((v) => !v); break;
        case "f": case "F": setFallbacks((f) => ({ ...f, [stage.id]: !f[stage.id] })); break;
        case "p": case "P": setPresenterMode((v) => { const n = !v; try { localStorage.setItem("byoai_presenter", n ? "1" : "0"); } catch { /* ignore */ } return n; }); break;
        case "Escape": setMenuOpen(false); break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, emit, stage.id]);

  const value = useMemo<Ctx>(() => ({
    stages: STAGES, index, stage, unlocked: unlockedAt(index), go, jump, resetNonce, forceFallback: !!fallbacks[stage.id], presenterMode,
    menuOpen, setMenuOpen, on, emit,
  }), [index, stage, go, jump, resetNonce, fallbacks, presenterMode, menuOpen, on, emit]);
  return <StageCtx.Provider value={value}>{children}</StageCtx.Provider>;
}
