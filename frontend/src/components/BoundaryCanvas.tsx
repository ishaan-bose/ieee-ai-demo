import { useEffect, useRef } from "react";

export interface Line { w1: number; w2: number; b: number }
const EXT = 1.2; // the view covers [-1.2, 1.2]^2
const BLUE: [number, number, number] = [86, 180, 233], ORANGE: [number, number, number] = [230, 159, 0];

/** Hero visual 1: the decision boundary as a smooth two-colour field with the data points on top (blue vs orange, colourblind-safe). */
export function BoundaryCanvas({ X, y, grid, G = 48, line, size = 520 }: { X: Float32Array; y: Uint8Array; grid?: Float32Array | null; G?: number; line?: Line | null; size?: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  const off = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = cv.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr; c.height = size * dpr;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0f172a"; ctx.fillRect(0, 0, size, size);
    const px = (x: number) => ((x + EXT) / (2 * EXT)) * size, py = (v: number) => ((EXT - v) / (2 * EXT)) * size;
    if (grid) {
      const o = (off.current ??= document.createElement("canvas"));
      o.width = G; o.height = G;
      const octx = o.getContext("2d")!;
      const img = octx.createImageData(G, G);
      for (let i = 0; i < G * G; i++) {
        const p = Math.min(1, Math.max(0, grid[i]));
        const k = Math.abs(p - 0.5) * 2; // 0 at the boundary, 1 deep inside a region
        const col = p > 0.5 ? ORANGE : BLUE;
        img.data[4 * i] = 15 + (col[0] - 15) * (0.25 + 0.45 * k); img.data[4 * i + 1] = 23 + (col[1] - 23) * (0.25 + 0.45 * k);
        img.data[4 * i + 2] = 42 + (col[2] - 42) * (0.25 + 0.45 * k); img.data[4 * i + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(o, 0, 0, size, size);
    }
    if (line) { // w1*x + w2*y + b = 0 across the view
      ctx.strokeStyle = "#f8fafc"; ctx.lineWidth = 3; ctx.setLineDash([]);
      const pts: [number, number][] = [];
      if (Math.abs(line.w2) > 1e-6) { for (const x of [-EXT, EXT]) pts.push([x, -(line.w1 * x + line.b) / line.w2]); }
      else if (Math.abs(line.w1) > 1e-6) { for (const v of [-EXT, EXT]) pts.push([-line.b / line.w1, v]); }
      if (pts.length === 2) { ctx.beginPath(); ctx.moveTo(px(pts[0][0]), py(pts[0][1])); ctx.lineTo(px(pts[1][0]), py(pts[1][1])); ctx.stroke(); }
    }
    for (let i = 0; i < y.length; i++) {
      ctx.beginPath(); ctx.arc(px(X[2 * i]), py(X[2 * i + 1]), 5.5, 0, 6.3);
      ctx.fillStyle = y[i] ? "#E69F00" : "#56B4E9"; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#0f172a"; ctx.stroke();
    }
  }, [X, y, grid, G, line, size]);
  return <canvas ref={cv} data-testid="boundary" style={{ width: size, height: size }} className="rounded-3xl border-2 border-slate-700" />;
}
