import { useEffect, useRef, useState } from "react";
import { urlParam } from "../lib/env";

interface Snap { fps: number; frameP95: number; frameMax: number; longTasks: number; longMax: number; longTotal: number; heapMB: number | null }

/** Shift+D toggles a small performance readout (off by default, `?perf=1` starts it open). Nothing runs while it is closed: no observer, no frame loop.
 *  Numbers: frames per second over the last 2 s, 95th-percentile and worst frame time, long tasks (main-thread blocks over 50 ms) since opening,
 *  and the JS heap (Chromium only). Click the box to copy the numbers as one line you can paste into a bug report. */
export function PerfOverlay() {
  const [open, setOpen] = useState(() => urlParam("perf") === "1");
  const [snap, setSnap] = useState<Snap | null>(null);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && (e.key === "D" || e.key === "d")) setOpen((v) => !v);
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);

  const frames = useRef<number[]>([]);
  useEffect(() => {
    if (!open) { setSnap(null); return; }
    let raf = 0, last = performance.now(), longTasks = 0, longMax = 0, longTotal = 0;
    frames.current = [];
    const frame = (t: number) => { frames.current.push(t - last); last = t; if (frames.current.length > 240) frames.current.shift(); raf = requestAnimationFrame(frame); };
    raf = requestAnimationFrame(frame);
    let obs: PerformanceObserver | null = null;
    try { obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) { longTasks++; longTotal += e.duration; longMax = Math.max(longMax, e.duration); } }); obs.observe({ type: "longtask", buffered: false }); } catch { /* not supported (e.g. Safari): the count stays 0 */ }
    const tick = () => {
      const a = [...frames.current].sort((x, y) => x - y), sum = a.reduce((s, x) => s + x, 0);
      const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
      setSnap({ fps: sum ? (a.length / sum) * 1000 : 0, frameP95: a[Math.floor(a.length * 0.95)] ?? 0, frameMax: a[a.length - 1] ?? 0, longTasks, longMax, longTotal, heapMB: mem ? mem.usedJSHeapSize / 1048576 : null });
    };
    const id = setInterval(tick, 500);
    tick();
    return () => { cancelAnimationFrame(raf); clearInterval(id); obs?.disconnect(); };
  }, [open]);

  if (!open) return null;
  const line = snap ? `fps=${snap.fps.toFixed(0)} frame_p95=${snap.frameP95.toFixed(0)}ms frame_max=${snap.frameMax.toFixed(0)}ms longtasks=${snap.longTasks} longtask_max=${snap.longMax.toFixed(0)}ms longtask_total=${snap.longTotal.toFixed(0)}ms heap=${snap.heapMB === null ? "n/a" : snap.heapMB.toFixed(1) + "MB"} dpr=${window.devicePixelRatio} viewport=${window.innerWidth}x${window.innerHeight} ua=${navigator.userAgent.replace(/^Mozilla\/5.0 /, "")}` : "";
  const bad = (cond: boolean) => (cond ? "text-red-300" : "text-emerald-300");
  return (
    <div data-testid="perf-overlay" className="fixed right-2 top-16 z-50 w-56 cursor-copy select-none rounded-lg bg-black/80 p-2 font-mono text-xs leading-5 text-slate-200 ring-1 ring-slate-600"
      onClick={() => { navigator.clipboard?.writeText(line).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1200); }).catch(() => undefined); }} title="click to copy these numbers; Shift+D hides">
      <div className="mb-0.5 flex justify-between text-slate-400"><span>performance</span><span>{copied ? "copied" : "Shift+D"}</span></div>
      {snap ? (<>
        <div className="flex justify-between"><span>frames/s</span><span className={bad(snap.fps < 45)}>{snap.fps.toFixed(0)}</span></div>
        <div className="flex justify-between"><span>frame p95 / max</span><span className={bad(snap.frameP95 > 33)}>{snap.frameP95.toFixed(0)} / {snap.frameMax.toFixed(0)} ms</span></div>
        <div className="flex justify-between"><span>long tasks</span><span className={bad(snap.longTasks > 0)}>{snap.longTasks} (max {snap.longMax.toFixed(0)} ms)</span></div>
        <div className="flex justify-between"><span>JS heap</span><span>{snap.heapMB === null ? "n/a" : `${snap.heapMB.toFixed(1)} MB`}</span></div>
      </>) : <div>measuring…</div>}
    </div>
  );
}
