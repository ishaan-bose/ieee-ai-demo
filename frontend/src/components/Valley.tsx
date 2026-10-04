import { useCallback, useEffect, useRef, useState } from "react";
import { fitCanvas } from "../lib/canvas";

// Hero visual 3 (cartoon). Gradient descent on a 1-D valley: L(w) = k/2 (w - w*)^2 + ripples. A CARTOON of the real landscape,
// which has millions of dimensions. step*k < ~0.1 crawls, ~0.1-1 glides, 1-2 bounces past the bottom, > 2 explodes.
export const K = 4, W_STAR = 0.2;
export const loss = (w: number) => 0.5 * K * (w - W_STAR) ** 2 + 0.05 * Math.sin(5 * w);
export const grad = (w: number) => K * (w - W_STAR) + 0.25 * Math.cos(5 * w);
export function verdict(step: number): string {
  const x = step * K;
  return x < 0.08 ? "crawling…" : x < 0.9 ? "gliding" : x < 2 ? "overshooting!" : "exploding";
}

export function useValley(step: number) {
  const [w, setW] = useState(-1.4);
  const [trail, setTrail] = useState<number[]>([-1.4]);
  const [running, setRunning] = useState(false);
  const wRef = useRef(-1.4);
  const reset = useCallback(() => { wRef.current = -1.4; setW(-1.4); setTrail([-1.4]); setRunning(false); }, []);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      const nw = wRef.current - step * grad(wRef.current);
      wRef.current = Math.max(-6, Math.min(6, nw));
      setW(wRef.current); setTrail((t) => [...t.slice(-60), wRef.current]);
      if (Math.abs(nw) > 5.9) setRunning(false);
    }, 140);
    return () => clearInterval(id);
  }, [running, step]);
  return { w, trail, running, setRunning, reset };
}

/** 2-D fallback / light version of the valley. (Valley3D is the react-three-fiber one; this is used if WebGL is unavailable.) */
export function Valley2D({ w, trail, size = 520 }: { w: number; trail: number[]; size?: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const H = 300;
    const ctx = fitCanvas(cv.current!, size, H);
    const X = (v: number) => ((v + 2.2) / 4.4) * size, Y = (v: number) => H - 20 - Math.min(6, v) * 38;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const v = -2.2 + (i / 200) * 4.4; if (i === 0) ctx.moveTo(X(v), Y(loss(v) + 0.1)); else ctx.lineTo(X(v), Y(loss(v) + 0.1)); }
    ctx.strokeStyle = "#64748b"; ctx.lineWidth = 4; ctx.stroke();
    ctx.strokeStyle = "rgba(251,191,36,0.5)"; ctx.lineWidth = 2; ctx.beginPath();
    trail.forEach((t, i) => { const x = X(Math.max(-2.2, Math.min(2.2, t))), y = Y(loss(t) + 0.1) - 14; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    const bx = X(Math.max(-2.2, Math.min(2.2, w))), by = Y(loss(w) + 0.1) - 14;
    const g = ctx.createRadialGradient(bx - 4, by - 4, 2, bx, by, 16); g.addColorStop(0, "#fde68a"); g.addColorStop(1, "#d97706");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(bx, by, 14, 0, 6.3); ctx.fill();
  }, [w, trail, size]);
  return <canvas ref={cv} style={{ width: size, height: 300 }} className="rounded-2xl bg-slate-900/60" data-testid="valley2d" />;
}
