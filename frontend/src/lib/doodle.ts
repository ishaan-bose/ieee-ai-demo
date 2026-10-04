// Quick, Draw! records in the browser (SPEC 4.1): load the real sample files, and stroke helpers for replay/partial drawing.
import type { Stroke } from "./rasterizer";

export interface DoodleRecord { c: number; k: string; d: Stroke[] }
export const CLASSES = ["cat", "bicycle", "house", "pizza", "lightning", "star", "fish", "tree", "umbrella", "sword"];

const cache = new Map<string, Promise<DoodleRecord[]>>();
export function loadDoodles(name: "duel" | "probe" | "samples"): Promise<DoodleRecord[]> {
  let p = cache.get(name);
  if (!p) { p = fetch(`/data/${name}.json`).then((r) => { if (!r.ok) throw new Error(name); return r.json() as Promise<DoodleRecord[]>; }); cache.set(name, p); }
  return p;
}

/** Cumulative path length at the END of each stroke, in the data's own units; last element = total. */
export function strokeLengths(d: Stroke[]): number[] {
  let total = 0;
  return d.map(([xs, ys]) => {
    for (let i = 1; i < xs.length; i++) total += Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    total += 6; // a little time for pen-up / dots
    return total;
  });
}

/** The part of a drawing visible when a fraction `frac` of its total path length has been drawn (constant drawing speed). */
export function partialStrokes(d: Stroke[], frac: number): Stroke[] {
  const ends = strokeLengths(d);
  const total = ends[ends.length - 1];
  const target = Math.min(1, Math.max(0, frac)) * total;
  const out: Stroke[] = [];
  let start = 0;
  for (let s = 0; s < d.length; s++) {
    const [xs, ys] = d[s];
    if (target >= ends[s]) { out.push([xs, ys]); start = ends[s]; continue; }
    if (target <= start) break;
    let remain = target - start;
    const px: number[] = [xs[0]], py: number[] = [ys[0]];
    for (let i = 1; i < xs.length && remain > 0; i++) {
      const seg = Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
      if (seg <= remain) { px.push(xs[i]); py.push(ys[i]); remain -= seg; }
      else { const f = remain / seg; px.push(xs[i - 1] + (xs[i] - xs[i - 1]) * f); py.push(ys[i - 1] + (ys[i] - ys[i - 1]) * f); remain = 0; }
    }
    out.push([px, py]);
    break;
  }
  return out;
}

export function strokeCountVisible(d: Stroke[], frac: number): number {
  const ends = strokeLengths(d);
  const target = frac * ends[ends.length - 1];
  return ends.filter((e) => e <= target + 1e-9).length;
}

/** Draw strokes into a 2D context scaled so the larger side fills `size` (minus padding). Used for replay, thumbnails, previews. */
export function paintStrokes(ctx: CanvasRenderingContext2D, strokes: Stroke[], size: number, opts: { color?: string; width?: number; pad?: number; bounds?: Stroke[] } = {}) {
  const pad = opts.pad ?? size * 0.06;
  const ref = opts.bounds ?? strokes;
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [xs, ys] of ref) for (let i = 0; i < xs.length; i++) { minx = Math.min(minx, xs[i]); maxx = Math.max(maxx, xs[i]); miny = Math.min(miny, ys[i]); maxy = Math.max(maxy, ys[i]); }
  if (!isFinite(minx)) return;
  const ext = Math.max(maxx - minx, maxy - miny, 1);
  const k = (size - 2 * pad) / ext;
  ctx.strokeStyle = opts.color ?? "#e2e8f0";
  ctx.lineWidth = opts.width ?? Math.max(2, size / 60);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  for (const [xs, ys] of strokes) {
    ctx.beginPath();
    xs.forEach((x, i) => { const px = pad + (x - minx) * k, py = pad + (ys[i] - miny) * k; if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    if (xs.length === 1) ctx.lineTo(pad + (xs[0] - minx) * k + 0.1, pad + (ys[0] - miny) * k);
    ctx.stroke();
  }
}

let probePromise: Promise<number[]> | null = null;
/** Labels of the 16 probe-wall doodles (probe.json). */
export const loadProbeLabels = () => (probePromise ??= loadDoodles("probe").then((a) => a.map((x) => x.c)).catch(() => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]));
