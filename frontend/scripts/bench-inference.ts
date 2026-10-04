// Browser-inference timing (SPEC 11 item 7): times the TypeScript forward pass of every bundled model and the log-mel feature
// extraction. Node/V8 is the same JS engine as Chrome, so this is a good estimate; run it on the LAPTOP for the real number:
//     cd frontend && npm run bench:inference
// Needs frontend/public/models/ (produced by backend/scripts/export_weights.py, copied from the server's artifacts.tgz).
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { N_FRAMES, N_MELS, logMel } from "../src/lib/audio";
import { forward, makeMockLogmelModel, makeMockModel, type Model, type ModelSpec } from "../src/lib/inference";

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, "../public");
const manifestPath = resolve(dir, "models/manifest.json");
const budgetMs = 50; // a forward pass must stay well under a frame budget of interactive use

function time(fn: () => void, warm = 5, n = 40): { mean: number; p95: number } {
  for (let i = 0; i < warm; i++) fn();
  const t: number[] = [];
  for (let i = 0; i < n; i++) { const s = performance.now(); fn(); t.push(performance.now() - s); }
  t.sort((a, b) => a - b);
  return { mean: t.reduce((a, b) => a + b, 0) / n, p95: t[Math.floor(n * 0.95)] };
}

const models: { name: string; model: Model; shape: number[]; real: boolean }[] = [];
if (existsSync(manifestPath)) {
  const man = JSON.parse(readFileSync(manifestPath, "utf8")) as { models: Record<string, ModelSpec & { file: string }> };
  for (const [name, spec] of Object.entries(man.models)) {
    const buf = readFileSync(resolve(dir, spec.file));
    models.push({ name, model: { name, spec, isMock: false, weights: new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)) }, shape: spec.input.shape, real: true });
  }
} else {
  console.log(`(no ${manifestPath}: timing RANDOM-WEIGHT stand-ins of the same shapes; run export_weights.py for the real ones)`);
  const mk = (n: string, m: Model, shape: number[]) => models.push({ name: n, model: m, shape, real: false });
  mk("doodle", makeMockModel("doodle"), [784]); mk("raw_audio", makeMockModel("raw_audio"), [4000]); mk("logmel_audio", makeMockLogmelModel(), [1, N_MELS, N_FRAMES]);
}

let worst = 0;
for (const { name, model, shape, real } of models) {
  const x = Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => Math.random());
  const r = time(() => forward(model, x, shape));
  worst = Math.max(worst, r.p95);
  console.log(`${name.padEnd(14)} ${real ? "real  " : "random"} weights: forward ${r.mean.toFixed(2)} ms mean, ${r.p95.toFixed(2)} ms p95 (${(model.weights.length / 1e6).toFixed(2)}M floats)`);
}
const wave = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(i / 9) * 0.3);
const f = time(() => logMel(wave), 2, 15);
console.log(`log-mel features (1 s of audio): ${f.mean.toFixed(2)} ms mean, ${f.p95.toFixed(2)} ms p95`);
worst = Math.max(worst, f.p95);
console.log(worst < budgetMs ? `PASS: everything under ${budgetMs} ms (p95)` : `SLOW: something exceeds ${budgetMs} ms: use a smaller input / fp16 (SPEC 5.3 Act 3 [verify])`);
