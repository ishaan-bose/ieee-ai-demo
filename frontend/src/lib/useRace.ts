// Race client (SPEC 5.3 Act 2, 9): live SSE from the GPU box, or a RECORDED stream when offline / forced (F) / skipped (S).
// Recordings: frontend/public/cache/races/<key>.json (one lane each), composed into up to 3 lanes; the nearest cached config is used
// for anything not on the grid. If nothing is cached, synthetic demo curves (lib/raceSim) keep the show going, flagged "demo data".
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { loadProbeLabels } from "./doodle";
import { laneCfg, nearestKey, type LaneCfg, type RaceKind } from "./raceGrid";
import { simulateLane, type LaneTick } from "./raceSim";
import { addRun } from "./session";
import { startRunTimer, stopRunTimer } from "./runTimer";
import type { RaceEvent } from "./types";

export interface LaneState { cfg: LaneCfg; ticks: LaneTick[]; latest: LaneTick | null }
export interface RaceState {
  status: "idle" | "running" | "done";
  source: "live" | "recorded" | "demo" | null;
  t: number; maxSeconds: number; lanes: LaneState[]; probeLabels: number[]; reason?: string; note?: string;
}
const IDLE: RaceState = { status: "idle", source: null, t: 0, maxSeconds: 0, lanes: [], probeLabels: [], };

interface CacheIndex { lanes: { key: string; kind: string; config: Record<string, unknown> }[] }
let indexPromise: Promise<CacheIndex | null> | null = null;
const loadIndex = () => (indexPromise ??= fetch("/cache/races/index.json").then((r) => (r.ok && r.headers.get("content-type")?.includes("json") ? (r.json() as Promise<CacheIndex>) : null)).catch(() => null));


async function recordedTicks(kind: RaceKind, cfg: LaneCfg, maxSeconds: number, probeLabels: number[]): Promise<{ ticks: LaneTick[]; demo: boolean }> {
  const idx = await loadIndex();
  const key = idx ? nearestKey(kind, cfg, idx.lanes) : null;
  if (key) {
    try {
      const r = await fetch(`/cache/races/${key}.json`);
      if (r.ok) {
        const rec = (await r.json()) as { ticks: LaneTick[] };
        return { ticks: rec.ticks.filter((t) => t.t <= maxSeconds + 1e-6), demo: false };
      }
    } catch { /* fall through */ }
  }
  return { ticks: simulateLane(cfg, maxSeconds, probeLabels), demo: true };
}

export function useRace(opts: { forceFallback: boolean; online: boolean }) {
  const [state, setState] = useState<RaceState>(IDLE);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const raceIdRef = useRef<string | null>(null);
  const runRef = useRef(0);
  const lastReq = useRef<{ kind: RaceKind; cfgs: LaneCfg[]; maxSeconds: number } | null>(null);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    stopRunTimer();
  }, []);
  useEffect(() => clear, [clear]);

  const finish = useCallback((reason: string, ticks: LaneState[]) => {
    stopRunTimer();
    for (const l of ticks) if (l.latest) addRun(110_000, 6 * 110_000 * l.latest.samples_seen);
    setState((s) => ({ ...s, status: "done", reason }));
  }, []);

  const playRecorded = useCallback(async (kind: RaceKind, cfgs: LaneCfg[], maxSeconds: number, startAt: number, runId: number, note?: string) => {
    const probeLabels = await loadProbeLabels();
    const rec = await Promise.all(cfgs.map((c) => recordedTicks(kind, c, maxSeconds, probeLabels)));
    if (runRef.current !== runId) return;
    const demo = rec.some((r) => r.demo);
    const lanes: LaneState[] = cfgs.map((cfg) => ({ cfg, ticks: [], latest: null }));
    const t0 = performance.now() - startAt * 1000;
    startRunTimer(Math.max(0.5, maxSeconds - startAt));
    const step = () => {
      const t = (performance.now() - t0) / 1000;
      lanes.forEach((l, i) => { l.ticks = rec[i].ticks.filter((k) => k.t <= t); l.latest = l.ticks[l.ticks.length - 1] ?? null; });
      const done = t >= maxSeconds;
      setState({ status: done ? "done" : "running", source: demo ? "demo" : "recorded", t: Math.min(t, maxSeconds), maxSeconds, lanes: lanes.map((l) => ({ ...l })), probeLabels, reason: done ? "time" : undefined, note });
      if (done) { if (timerRef.current) clearInterval(timerRef.current); timerRef.current = null; finish("time", lanes); }
    };
    step();
    timerRef.current = setInterval(step, 100);
  }, [finish]);

  const start = useCallback(async (kind: RaceKind, cfgs: LaneCfg[], maxSeconds: number) => {
    clear();
    const runId = ++runRef.current;
    lastReq.current = { kind, cfgs, maxSeconds };
    const probeLabels = await loadProbeLabels();
    setState({ status: "running", source: null, t: 0, maxSeconds, lanes: cfgs.map((cfg) => ({ cfg, ticks: [], latest: null })), probeLabels });
    if (opts.forceFallback || !opts.online) { await playRecorded(kind, cfgs, maxSeconds, 0, runId); return; }

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let lastT = 0;
    const lanes: LaneState[] = cfgs.map((cfg) => ({ cfg, ticks: [], latest: null }));
    try {
      const { race_id } = await api.createRace({ kind, lanes: cfgs.map((c) => ({ id: c.id, config: c })), max_seconds: maxSeconds });
      raceIdRef.current = race_id;
      let gotFirst = false;
      const watchdog = setTimeout(() => { if (!gotFirst) ctrl.abort(); }, 9000); // no tick in 9 s: the GPU is busy or unreachable
      startRunTimer(maxSeconds);
      for await (const ev of api.raceStream(race_id, ctrl.signal) as AsyncIterable<RaceEvent>) {
        if (runRef.current !== runId) return;
        if (ev.event === "tick") {
          gotFirst = true;
          lastT = ev.data.t;
          for (const lt of ev.data.lanes) {
            const l = lanes.find((x) => x.cfg.id === lt.id);
            if (l) { const { id: _id, ...tick } = lt; l.ticks = [...l.ticks, { t: ev.data.t, ...tick }]; l.latest = l.ticks[l.ticks.length - 1]; }
          }
          setState({ status: "running", source: "live", t: ev.data.t, maxSeconds, lanes: lanes.map((l) => ({ ...l })), probeLabels });
        } else if (ev.event === "done") {
          clearTimeout(watchdog);
          setState((s) => ({ ...s, source: "live" }));
          finish(ev.data.reason, lanes);
          return;
        } else { throw new Error(ev.data.message); }
      }
      clearTimeout(watchdog);
      if (runRef.current === runId && lastT < maxSeconds - 1) throw new Error("stream ended early");
    } catch {
      if (runRef.current !== runId) return;
      // Server unreachable, 503 (no data), stream cut, or silent: continue from the recording at the time we reached.
      await playRecorded(kind, cfgs, maxSeconds, lastT, runId, "switched to the recorded race");
    }
  }, [clear, finish, opts.forceFallback, opts.online, playRecorded]);

  /** S: jump straight to the recorded result. */
  const skip = useCallback(async () => {
    const req = lastReq.current;
    if (!req) return;
    const liveId = raceIdRef.current;
    clear();
    if (liveId) { api.abortRace(liveId).catch(() => undefined); raceIdRef.current = null; }
    const runId = ++runRef.current;
    const probeLabels = await loadProbeLabels();
    const rec = await Promise.all(req.cfgs.map((c) => recordedTicks(req.kind, c, req.maxSeconds, probeLabels)));
    const lanes: LaneState[] = req.cfgs.map((cfg, i) => ({ cfg, ticks: rec[i].ticks, latest: rec[i].ticks[rec[i].ticks.length - 1] ?? null }));
    if (runRef.current !== runId) return;
    setState({ status: "done", source: rec.some((r) => r.demo) ? "demo" : "recorded", t: req.maxSeconds, maxSeconds: req.maxSeconds, lanes, probeLabels, reason: "time" });
    for (const l of lanes) if (l.latest) addRun(110_000, 6 * 110_000 * l.latest.samples_seen);
  }, [clear]);

  const abort = useCallback(() => {
    const id = raceIdRef.current;
    clear();
    runRef.current++;
    if (id) api.abortRace(id).catch(() => undefined);
    raceIdRef.current = null;
    setState((s) => ({ ...s, status: s.status === "running" ? "done" : s.status, reason: "aborted" }));
  }, [clear]);

  const reset = useCallback(() => { clear(); runRef.current++; setState(IDLE); lastReq.current = null; }, [clear]);
  return { state, start, skip, abort, reset, laneCfg };
}
