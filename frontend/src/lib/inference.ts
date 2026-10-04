// Browser inference for the bundled models (SPEC 9): a small TypeScript forward pass over the manifest written by
// backend/scripts/export_weights.py. Layers: dense, conv2d (stride 1, same padding), relu, maxpool, flatten.
// Python twin (reference): backend/app/export_format.py::numpy_forward. Golden: shared/golden/inference.json.
import { mulberry32, gaussian } from "./rng";

export interface TensorRef { offset: number; shape: number[] }
export type Layer =
  | { type: "dense"; in: number; out: number; w: TensorRef; b: TensorRef }
  | { type: "conv2d"; in_ch: number; out_ch: number; k: number; w: TensorRef; b: TensorRef }
  | { type: "relu" }
  | { type: "maxpool"; k: number }
  | { type: "flatten" };

export interface ModelSpec {
  layers: Layer[];
  input: { kind: string; shape: number[]; note?: string };
  classes?: string[] | null;
  val_acc?: number | null;
  val_acc_partial?: number | null;
}

export interface Model {
  name: string;
  spec: ModelSpec;
  weights: Float32Array;
  isMock: boolean;
}

/** Output of every layer for one example, plus the final output. Shapes: dense -> [n]; conv/pool -> [c,h,w]. */
export interface Trace { outputs: { shape: number[]; data: Float32Array }[]; out: Float32Array }

const tensor = (m: Model, r: TensorRef) => m.weights.subarray(r.offset, r.offset + r.shape.reduce((a, b) => a * b, 1));

export function forwardTrace(m: Model, input: Float32Array, inputShape: number[] = m.spec.input.shape): Trace {
  let data = input;
  let shape = inputShape.slice();
  const outputs: Trace["outputs"] = [];
  for (const L of m.spec.layers) {
    if (L.type === "dense") {
      const w = tensor(m, L.w), b = tensor(m, L.b);
      const out = new Float32Array(L.out);
      for (let o = 0; o < L.out; o++) {
        let s = b[o];
        const row = o * L.in;
        for (let i = 0; i < L.in; i++) s += w[row + i] * data[i];
        out[o] = s;
      }
      data = out; shape = [L.out];
    } else if (L.type === "relu") {
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) out[i] = data[i] > 0 ? data[i] : 0;
      data = out;
    } else if (L.type === "flatten") {
      shape = [data.length];
    } else if (L.type === "maxpool") {
      const [c, h, w] = shape;
      const k = L.k, oh = Math.floor(h / k), ow = Math.floor(w / k);
      const out = new Float32Array(c * oh * ow);
      for (let ch = 0; ch < c; ch++)
        for (let y = 0; y < oh; y++)
          for (let x = 0; x < ow; x++) {
            let best = -Infinity;
            for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) best = Math.max(best, data[ch * h * w + (y * k + dy) * w + x * k + dx]);
            out[ch * oh * ow + y * ow + x] = best;
          }
      data = out; shape = [c, oh, ow];
    } else if (L.type === "conv2d") {
      const [, h, w] = shape;
      const wt = tensor(m, L.w), bs = tensor(m, L.b);
      const k = L.k, p = k >> 1, inC = L.in_ch;
      const out = new Float32Array(L.out_ch * h * w);
      for (let oc = 0; oc < L.out_ch; oc++)
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            let s = bs[oc];
            for (let ic = 0; ic < inC; ic++)
              for (let dy = 0; dy < k; dy++) {
                const yy = y + dy - p;
                if (yy < 0 || yy >= h) continue;
                for (let dx = 0; dx < k; dx++) {
                  const xx = x + dx - p;
                  if (xx < 0 || xx >= w) continue;
                  s += wt[((oc * inC + ic) * k + dy) * k + dx] * data[ic * h * w + yy * w + xx];
                }
              }
            out[oc * h * w + y * w + x] = s;
          }
      data = out; shape = [L.out_ch, h, w];
    }
    outputs.push({ shape: shape.slice(), data });
  }
  return { outputs, out: data };
}

export const forward = (m: Model, x: Float32Array, shape?: number[]) => forwardTrace(m, x, shape).out;

export function softmax(z: ArrayLike<number>): Float32Array {
  let mx = -Infinity;
  for (let i = 0; i < z.length; i++) mx = Math.max(mx, z[i]);
  const e = new Float32Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - mx); s += e[i]; }
  for (let i = 0; i < e.length; i++) e[i] /= s;
  return e;
}

export function topK(p: ArrayLike<number>, k: number): { index: number; p: number }[] {
  return Array.from(p, (v, i) => ({ index: i, p: v })).sort((a, b) => b.p - a.p).slice(0, k);
}

// ---------------------------------------------------------------- loading (browser)

interface Manifest { version: number; models: Record<string, ModelSpec & { file: string }> }
let manifestPromise: Promise<Manifest | null> | null = null;
const modelCache = new Map<string, Promise<Model>>();

async function loadManifest(): Promise<Manifest | null> {
  manifestPromise ??= fetch("/models/manifest.json")
    .then((r) => (r.ok && r.headers.get("content-type")?.includes("json") ? (r.json() as Promise<Manifest>) : null))
    .catch(() => null);
  return manifestPromise;
}

const DOODLE_CLASSES = ["cat", "bicycle", "house", "pizza", "lightning", "star", "fish", "tree", "umbrella", "sword"];
const AUDIO_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];

const MOCK_SPECS: Record<string, { layers: [string, number, number][]; input: ModelSpec["input"]; classes: string[] }> = {
  doodle: { layers: [["dense", 784, 256], ["relu", 0, 0], ["dense", 256, 128], ["relu", 0, 0], ["dense", 128, 10]], input: { kind: "doodle", shape: [784] }, classes: DOODLE_CLASSES },
  raw_audio: { layers: [["dense", 4000, 128], ["relu", 0, 0], ["dense", 128, 64], ["relu", 0, 0], ["dense", 64, 10]], input: { kind: "raw_audio", shape: [4000] }, classes: AUDIO_CLASSES },
};

/** Seeded random-weight stand-in used when the real weights are not bundled yet (flagged `isMock`). */
export function makeMockModel(name: string): Model {
  const spec = MOCK_SPECS[name] ?? MOCK_SPECS.doodle;
  const rng = mulberry32(name.length * 7919);
  const layers: Layer[] = [];
  const buf: number[] = [];
  for (const [t, i, o] of spec.layers) {
    if (t === "relu") { layers.push({ type: "relu" }); continue; }
    const std = Math.sqrt(2 / i);
    const wOff = buf.length;
    for (let k = 0; k < i * o; k++) buf.push(gaussian(rng) * std);
    const bOff = buf.length;
    for (let k = 0; k < o; k++) buf.push(0);
    layers.push({ type: "dense", in: i, out: o, w: { offset: wOff, shape: [o, i] }, b: { offset: bOff, shape: [o] } });
  }
  return { name, isMock: true, weights: Float32Array.from(buf), spec: { layers, input: spec.input, classes: spec.classes } };
}

export function makeMockLogmelModel(): Model {
  const rng = mulberry32(4242);
  const buf: number[] = [];
  const push = (n: number, std: number) => { const off = buf.length; for (let i = 0; i < n; i++) buf.push(gaussian(rng) * std); return off; };
  const layers: Layer[] = [
    { type: "conv2d", in_ch: 1, out_ch: 8, k: 3, w: { offset: push(72, 0.3), shape: [8, 1, 3, 3] }, b: { offset: push(8, 0), shape: [8] } },
    { type: "relu" }, { type: "maxpool", k: 2 },
    { type: "conv2d", in_ch: 8, out_ch: 16, k: 3, w: { offset: push(1152, 0.12), shape: [16, 8, 3, 3] }, b: { offset: push(16, 0), shape: [16] } },
    { type: "relu" }, { type: "maxpool", k: 2 }, { type: "flatten" },
    { type: "dense", in: 3840, out: 64, w: { offset: push(3840 * 64, 0.03), shape: [64, 3840] }, b: { offset: push(64, 0), shape: [64] } },
    { type: "relu" },
    { type: "dense", in: 64, out: 10, w: { offset: push(640, 0.2), shape: [10, 64] }, b: { offset: push(10, 0), shape: [10] } },
  ];
  return { name: "logmel_audio", isMock: true, weights: Float32Array.from(buf), spec: { layers, input: { kind: "logmel", shape: [1, 40, 98] }, classes: AUDIO_CLASSES } };
}

/** The bundled model by name ("doodle" | "raw_audio" | "logmel_audio"), or a flagged mock when it is not available. */
export function getModel(name: string): Promise<Model> {
  let p = modelCache.get(name);
  if (!p) {
    p = (async () => {
      const man = await loadManifest();
      const spec = man?.models?.[name];
      if (spec) {
        try {
          const r = await fetch(`/${spec.file}`);
          if (r.ok) return { name, isMock: false, spec, weights: new Float32Array(await r.arrayBuffer()) } as Model;
        } catch { /* fall through to the mock */ }
      }
      return name === "logmel_audio" ? makeMockLogmelModel() : makeMockModel(name);
    })();
    modelCache.set(name, p);
  }
  return p;
}
