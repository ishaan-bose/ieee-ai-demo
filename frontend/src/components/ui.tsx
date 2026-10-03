import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, type ReactNode } from "react";
import { useUnlocked } from "../state/StageProvider";
import { useRunTimer } from "../lib/runTimer";
import { paintStrokes, type DoodleRecord } from "../lib/doodle";
import type { Stroke } from "../lib/rasterizer";

/** Children exist only once `id` is unlocked (progressive disclosure); they pop in with a short animation. */
export function Unlock({ id, children, className }: { id: string; children: ReactNode; className?: string }) {
  const on = useUnlocked(id);
  return (
    <AnimatePresence>
      {on && (
        <motion.div key={id} className={className} initial={{ opacity: 0, scale: 0.92, y: 14 }} animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }} transition={{ type: "spring", stiffness: 260, damping: 22 }}>
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Accuracy gauge (SPEC 5.1): a bar whose colour blends red -> yellow -> green relative to CHANCE for the task. */
export function gaugeColor(value: number, chance: number): string {
  const t = Math.min(1, Math.max(0, (value - chance) / (1 - chance)));
  const e = Math.min(1, Math.max(0, (t - 0.12) / 0.78)), s = e * e * (3 - 2 * e); // stays red near chance
  const hue = s < 0.5 ? 4 + s * 2 * 50 : 54 + (s - 0.5) * 2 * 76; // red 4 -> yellow 54 -> green 130
  return `hsl(${hue} 85% 52%)`;
}

export function Gauge({ value, chance, label, small, broke }: { value: number; chance: number; label?: string; small?: boolean; broke?: boolean }) {
  const color = broke ? "#ef4444" : gaugeColor(value, chance);
  return (
    <div className="w-full" data-testid="gauge">
      <div className="flex items-baseline justify-between">
        <span className={small ? "text-lg text-slate-300" : "text-2xl text-slate-200"}>{label ?? "accuracy"}</span>
        <span className="font-mono text-base text-slate-400">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className={`relative mt-1 overflow-hidden rounded-full bg-slate-800 ${small ? "h-3" : "h-5"}`}>
        <motion.div className="h-full rounded-full" animate={{ width: `${Math.max(1, value * 100)}%`, backgroundColor: color }} transition={{ duration: 0.35 }} />
        <div className="absolute top-0 h-full w-0.5 bg-slate-400/70" style={{ left: `${chance * 100}%` }} title="chance" />
      </div>
    </div>
  );
}

/** Thin ring in the corner showing the time cap of a live run. */
export function TimerRing() {
  const t = useRunTimer();
  const ref = useRef<SVGCircleElement>(null);
  useEffect(() => {
    if (!t.active) return;
    let raf = 0;
    const loop = () => {
      const f = Math.min(1, (performance.now() - t.startedAt) / 1000 / t.total);
      ref.current?.setAttribute("stroke-dashoffset", String(2 * Math.PI * 22 * f));
      if (f < 1) raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [t.active, t.startedAt, t.total]);
  if (!t.active) return null;
  return (
    <svg className="pointer-events-none fixed right-5 top-5 z-30" width="56" height="56" viewBox="0 0 56 56" data-testid="timer-ring">
      <circle cx="28" cy="28" r="22" fill="none" stroke="#1e293b" strokeWidth="4" />
      <circle ref={ref} cx="28" cy="28" r="22" fill="none" stroke="#38bdf8" strokeWidth="4" strokeLinecap="round" strokeDasharray={2 * Math.PI * 22} strokeDashoffset={0} transform="rotate(-90 28 28)" />
    </svg>
  );
}

/** A static doodle thumbnail. */
export function Doodle({ record, strokes, size = 120, color }: { record?: DoodleRecord; strokes?: Stroke[]; size?: number; color?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const d = strokes ?? record?.d;
  useEffect(() => {
    const c = ref.current;
    if (!c || !d) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr; c.height = size * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);
    paintStrokes(ctx, d, size, { color: color ?? "#e2e8f0", width: Math.max(2, size / 40) });
  }, [d, size, color]);
  return <canvas ref={ref} style={{ width: size, height: size }} className="rounded-lg bg-slate-900" />;
}

export function Badge({ children, tone = "slate", title }: { children: ReactNode; tone?: "slate" | "red" | "amber" | "sky" | "green"; title?: string }) {
  const tones = { slate: "bg-slate-800 text-slate-300", red: "bg-red-600 text-white", amber: "bg-amber-500/20 text-amber-300", sky: "bg-sky-500/20 text-sky-300", green: "bg-emerald-500/20 text-emerald-300" };
  return <span title={title} className={`inline-block rounded-md px-2 py-0.5 text-sm font-semibold ${tones[tone]}`}>{children}</span>;
}

export const LANE_COLORS = ["#56B4E9", "#E69F00", "#CC79A7"]; // colourblind-safe (Okabe-Ito): sky, orange, pink
export const CLASS_COLORS = ["#56B4E9", "#E69F00"]; // blue vs orange for two-class plots

export function Credits() {
  return (
    <footer className="fixed bottom-1 left-2 z-10 max-w-md rounded bg-slate-900/80 px-2 py-1 text-[10px] leading-tight text-slate-500">
      Quick, Draw! dataset © Google, CC BY 4.0 (<a className="underline" href="https://github.com/googlecreativelab/quickdraw-dataset" target="_blank" rel="noreferrer">link</a>) ·
      Speech Commands v0.02 © Google, CC BY 4.0 (<a className="underline" href="https://arxiv.org/abs/1804.03209" target="_blank" rel="noreferrer">link</a>) ·
      Lichess open database of evaluated positions, CC0 (<a className="underline" href="https://database.lichess.org/#evals" target="_blank" rel="noreferrer">link</a>) · thank you, Lichess!
    </footer>
  );
}
