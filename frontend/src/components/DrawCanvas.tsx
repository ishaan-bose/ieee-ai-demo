import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import type { Stroke } from "../lib/rasterizer";

export interface DrawHandle { clear(): void; strokes(): Stroke[] }

/** Free-hand drawing canvas. Strokes are kept in canvas pixel coordinates (floats); the shared rasterizer normalizes them before the model sees them. */
export const DrawCanvas = forwardRef<DrawHandle, { size?: number; disabled?: boolean; onStrokeEnd?: (strokes: Stroke[]) => void; onClear?: () => void }>(
  function DrawCanvas({ size = 420, disabled, onStrokeEnd, onClear }, ref) {
    const cv = useRef<HTMLCanvasElement>(null);
    const strokes = useRef<Stroke[]>([]);
    const cur = useRef<Stroke | null>(null);

    const repaint = useCallback(() => {
      const c = cv.current;
      if (!c) return;
      const ctx = c.getContext("2d")!;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      ctx.strokeStyle = "#f1f5f9"; ctx.lineWidth = Math.max(6, size / 55); ctx.lineCap = "round"; ctx.lineJoin = "round";
      for (const [xs, ys] of [...strokes.current, ...(cur.current ? [cur.current] : [])]) {
        ctx.beginPath();
        xs.forEach((x, i) => (i === 0 ? ctx.moveTo(x, ys[i]) : ctx.lineTo(x, ys[i])));
        if (xs.length === 1) ctx.lineTo(xs[0] + 0.1, ys[0]);
        ctx.stroke();
      }
    }, [size]);

    useEffect(() => {
      const c = cv.current!;
      const dpr = window.devicePixelRatio || 1;
      c.width = size * dpr; c.height = size * dpr;
      repaint();
    }, [size, repaint]);

    useImperativeHandle(ref, () => ({
      clear() { strokes.current = []; cur.current = null; repaint(); onClear?.(); },
      strokes: () => strokes.current,
    }), [repaint, onClear]);

    const pos = (e: React.PointerEvent) => { const r = cv.current!.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * size, ((e.clientY - r.top) / r.height) * size] as const; };
    return (
      <canvas ref={cv} data-testid="draw-canvas" style={{ width: size, height: size, touchAction: "none" }} className={`rounded-2xl border-2 border-slate-700 bg-slate-900 ${disabled ? "opacity-60" : "cursor-crosshair"}`}
        onPointerDown={(e) => { if (disabled) return; cv.current!.setPointerCapture(e.pointerId); const [x, y] = pos(e); cur.current = [[x], [y]]; repaint(); }}
        onPointerMove={(e) => { if (!cur.current) return; const [x, y] = pos(e); cur.current[0].push(x); cur.current[1].push(y); repaint(); }}
        onPointerUp={() => { if (!cur.current) return; strokes.current = [...strokes.current, cur.current]; cur.current = null; repaint(); onStrokeEnd?.(strokes.current); }}
        onPointerCancel={() => { cur.current = null; repaint(); }} />
    );
  });
