// A thin ring shows the time cap while a run is live (SPEC 5.1). Global store; <TimerRing/> reads it.
import { useSyncExternalStore } from "react";

interface Timer { active: boolean; total: number; startedAt: number }
let state: Timer = { active: false, total: 0, startedAt: 0 };
const subs = new Set<() => void>();
const set = (t: Timer) => { state = t; subs.forEach((f) => f()); };

export const startRunTimer = (totalSeconds: number) => set({ active: true, total: totalSeconds, startedAt: performance.now() });
export const stopRunTimer = () => set({ ...state, active: false });
export const useRunTimer = () => useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => state);
