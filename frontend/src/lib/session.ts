// Session stats for the finale (SPEC 5.3 Act 4): runs, parameters trained, FLOPs this session, duel score.
import { useSyncExternalStore } from "react";

export interface SessionStats { runs: number; paramsTrained: number; flops: number; humanDuel: { score: number; rounds: number } | null; startedAt: number }
let state: SessionStats = { runs: 0, paramsTrained: 0, flops: 0, humanDuel: null, startedAt: Date.now() };
const subs = new Set<() => void>();
const set = (s: SessionStats) => { state = s; subs.forEach((f) => f()); };

export const sessionStats = () => state;
export function addRun(params: number, flops: number) { set({ ...state, runs: state.runs + 1, paramsTrained: state.paramsTrained + params, flops: state.flops + flops }); }
export function setHumanDuel(score: number, rounds: number) { set({ ...state, humanDuel: { score, rounds } }); }
export function useSession(): SessionStats {
  return useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => state);
}

// Test hook (mock mode only): lets the Playwright layout check put large values into the finale's stat boxes.
if (typeof window !== "undefined" && new URLSearchParams(location.search).get("mock") === "1") {
  (window as unknown as { __byoai: unknown }).__byoai = { set: (p: Partial<SessionStats>) => set({ ...state, ...p }) };
}
