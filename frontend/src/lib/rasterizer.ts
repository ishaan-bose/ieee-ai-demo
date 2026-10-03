// Stroke JSON -> 28x28 uint8 image. Spec: shared/RASTERIZER.md (Python twin: backend/app/data/rasterizer.py).
export const SIZE = 28;
const MARGIN = 2;
const LINE_WIDTH = 2.0;
const NORM = 255;

export type Stroke = [number[], number[]];

export function normalizeStrokes(strokes: Stroke[]): Float64Array[] {
  // returns one Float64Array [x0,y0,x1,y1,...] per stroke, in canvas coordinates
  const live = strokes.filter((s) => s[0].length > 0);
  if (!live.length) return [];
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [xs, ys] of live) {
    for (let i = 0; i < xs.length; i++) {
      minx = Math.min(minx, xs[i]); maxx = Math.max(maxx, xs[i]);
      miny = Math.min(miny, ys[i]); maxy = Math.max(maxy, ys[i]);
    }
  }
  let extent = Math.max(maxx - minx, maxy - miny);
  if (extent === 0) extent = 1;
  const k = (NORM / extent) * ((SIZE - 2 * MARGIN) / NORM);
  return live.map(([xs, ys]) => {
    const out = new Float64Array(xs.length * 2);
    for (let i = 0; i < xs.length; i++) {
      out[2 * i] = (xs[i] - minx) * k + MARGIN;
      out[2 * i + 1] = (ys[i] - miny) * k + MARGIN;
    }
    return out;
  });
}

export function rasterize(strokes: Stroke[]): Uint8Array {
  const out = new Uint8Array(SIZE * SIZE);
  const norm = normalizeStrokes(strokes);
  if (!norm.length) return out;
  const segs: number[] = []; // ax, ay, bx, by
  for (const p of norm) {
    const n = p.length / 2;
    if (n === 1) segs.push(p[0], p[1], p[0], p[1]);
    else for (let i = 0; i < n - 1; i++) segs.push(p[2 * i], p[2 * i + 1], p[2 * i + 2], p[2 * i + 3]);
  }
  const S = segs.length / 4;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const px = c + 0.5, py = r + 0.5;
      let best = Infinity;
      for (let s = 0; s < S; s++) {
        const ax = segs[4 * s], ay = segs[4 * s + 1];
        const abx = segs[4 * s + 2] - ax, aby = segs[4 * s + 3] - ay;
        const apx = px - ax, apy = py - ay;
        const denom = Math.max(abx * abx + aby * aby, 1e-12);
        let t = (apx * abx + apy * aby) / denom;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const dx = apx - t * abx, dy = apy - t * aby;
        const d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
      }
      const cov = Math.min(1, Math.max(0, LINE_WIDTH / 2 + 0.5 - Math.sqrt(best)));
      out[r * SIZE + c] = Math.floor(255 * cov + 0.5);
    }
  }
  return out;
}

/** Model input: pixel/255 as float32, flattened to 784. */
export function toModelInput(img: Uint8Array): Float32Array {
  const x = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) x[i] = img[i] / 255;
  return x;
}
