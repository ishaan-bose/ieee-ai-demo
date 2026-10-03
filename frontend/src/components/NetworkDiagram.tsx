import { useEffect, useRef } from "react";
import type { DiagramData } from "../lib/diagram";
import { CLASSES } from "../lib/doodle";

interface Particle { l: number; i: number; j: number; t0: number; s: number }

/** Hero visual 2/3: a network drawn as glowing layers. `pulse` changes -> signals flow forward (cyan) or backward (red, `mode="backward"`). */
export function NetworkDiagram({ data, pulse, mode = "forward", width = 760, height = 360, labels = CLASSES }: {
  data: DiagramData | null; pulse: number; mode?: "forward" | "backward"; width?: number; height?: number; labels?: string[];
}) {
  const cv = useRef<HTMLCanvasElement>(null);
  const parts = useRef<Particle[]>([]);
  const t0 = useRef(0);
  const dataRef = useRef<DiagramData | null>(data);
  dataRef.current = data;

  useEffect(() => {
    if (!data) return;
    const ps: Particle[] = [];
    data.edges.forEach((layer, l) => layer.forEach((row, i) => row.forEach((s, j) => { if (s > 0.25) for (let k = 0; k < 2; k++) ps.push({ l, i, j, t0: Math.random() * 0.25, s }); })));
    parts.current = ps;
    t0.current = performance.now();
  }, [pulse, data]);

  useEffect(() => {
    const c = cv.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr; c.height = height * dpr;
    let raf = 0;
    const draw = () => {
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const d = dataRef.current;
      if (!d) { raf = requestAnimationFrame(draw); return; }
      const nc = d.cols.length, padX = 70, padY = 24;
      const xs = d.cols.map((_, c2) => padX + (c2 * (width - 2 * padX - 70)) / (nc - 1));
      const ys = d.cols.map((col) => col.shown.map((_, i) => padY + (col.shown.length === 1 ? 0.5 : i / (col.shown.length - 1)) * (height - 2 * padY)));
      const back = mode === "backward";
      const base = back ? "239,68,68" : "56,189,248";
      d.edges.forEach((layer, l) => layer.forEach((row, i) => row.forEach((s, j) => {
        ctx.strokeStyle = `rgba(${base},${0.03 + 0.28 * s})`; ctx.lineWidth = 0.6 + 1.6 * s;
        ctx.beginPath(); ctx.moveTo(xs[l], ys[l][i]); ctx.lineTo(xs[l + 1], ys[l + 1][j]); ctx.stroke();
      })));
      const el = (performance.now() - t0.current) / 1000;
      ctx.save(); ctx.globalCompositeOperation = "lighter";
      for (const p of parts.current) {
        const L = d.edges.length;
        const layerIdx = back ? L - 1 - p.l : p.l;
        const u = (el - layerIdx * 0.32 - p.t0) / 0.55;
        if (u < 0 || u > 1) continue;
        const f = back ? 1 - u : u;
        const x = xs[p.l] + (xs[p.l + 1] - xs[p.l]) * f, y = ys[p.l][p.i] + (ys[p.l + 1][p.j] - ys[p.l][p.i]) * f;
        const g = ctx.createRadialGradient(x, y, 0, x, y, 9);
        g.addColorStop(0, `rgba(${back ? "255,160,120" : "190,240,255"},${0.95 * p.s})`); g.addColorStop(1, `rgba(${base},0)`);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.3); ctx.fill();
      }
      ctx.restore();
      d.cols.forEach((col, c2) => col.shown.forEach((_, i) => {
        const a = Math.min(1, Math.max(0, col.acts[i])), x = xs[c2], y = ys[c2][i];
        const isTop = c2 === nc - 1 && i === d.top;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 6 + 14 * a);
        glow.addColorStop(0, isTop ? "rgba(251,191,36,0.95)" : `rgba(${base},${0.15 + 0.8 * a})`); glow.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, 6 + 14 * a, 0, 6.3); ctx.fill();
        ctx.fillStyle = isTop ? "#fbbf24" : `rgba(226,242,255,${0.35 + 0.65 * a})`; ctx.beginPath(); ctx.arc(x, y, 3 + 4 * a, 0, 6.3); ctx.fill();
        if (c2 === nc - 1) { ctx.fillStyle = isTop ? "#fde68a" : "#94a3b8"; ctx.font = `${isTop ? "bold " : ""}15px system-ui`; ctx.fillText(labels[i] ?? String(i), x + 16, y + 5); }
      }));
      ctx.fillStyle = "#64748b"; ctx.font = "13px system-ui"; ctx.textAlign = "center";
      d.cols.forEach((col, c2) => ctx.fillText(`${col.label} (${col.total})`, xs[c2], height - 5));
      ctx.textAlign = "start";
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [width, height, mode, labels]);
  return <canvas ref={cv} style={{ width, height }} className="rounded-2xl bg-slate-900/60" data-testid="network" />;
}
