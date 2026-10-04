// Synthetic race curves: used by the mock server (?mock=1) and as the last-resort fallback when no recording is cached.
// Deterministic per config. Qualitatively like the real thing (cross-entropy beats MSE, a tiny/huge learning rate is bad,
// batch 1 is noisy, full batch makes few updates), but it is NOT real training and the UI says so ("recorded"/"demo data").
import type { LaneCfg } from "./raceGrid";
import { mulberry32 } from "./rng";

export interface LaneTick { t: number; step: number; samples_seen: number; updates: number; loss: number | null; acc: number; probe_preds: number[] }

const N_TRAIN = 1_090_830;

function lossGain(c: LaneCfg): { asym: number; gain: number; scale: number } {
  switch (c.loss) {
    case "ce": return { asym: 0.9, gain: 1, scale: 1 };
    case "focal": return { asym: 0.9 - 0.008 * c.gamma, gain: 0.85 - 0.04 * c.gamma, scale: 0.6 };
    case "label_smooth": return { asym: 0.89 - 0.05 * c.eps, gain: 0.9 - 0.2 * c.eps, scale: 1.1 };
    case "huber": return { asym: 0.86, gain: 0.45 + 0.1 * Math.min(c.delta, 2), scale: 0.08 };
    case "l1": return { asym: 0.78, gain: 0.3, scale: 0.15 };
    default: return { asym: 0.84, gain: 0.5, scale: 0.1 };
  }
}

export function updatesPerSecond(b: LaneCfg["batch_size"]): number {
  if (b === "full") return 0.8;
  return Math.max(2, 2300 / (1 + Math.pow(b / 150, 0.9)));
}

export function simulateLane(c: LaneCfg, maxSeconds: number, probeLabels: number[], tickEvery = 0.5): LaneTick[] {
  const seedBase = [...JSON.stringify([c.loss, c.lr, c.batch_size, c.delta, c.gamma, c.eps])].reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) >>> 0, 7);
  const rng = mulberry32(seedBase);
  const { asym, gain, scale } = lossGain(c);
  const lrFactor = Math.exp(-Math.pow(Math.log10(c.lr / 0.05), 2) / (2 * 0.7 * 0.7));
  const diverged = c.lr >= 0.8;
  const bsz = c.batch_size === "full" ? N_TRAIN : c.batch_size;
  const batchGain = 1 + 0.15 * Math.log2(Math.max(1, bsz));
  const ups = updatesPerSecond(c.batch_size);
  const thr = probeLabels.map((_, i) => 0.15 + 0.85 * ((i * 0.6180339887) % 1));
  const noiseSd = (0.035 / (1 + Math.log2(Math.max(1, bsz)) / 2)) * (c.lr > 0.2 ? 3 : 1);
  const out: LaneTick[] = [];
  for (let t = tickEvery; t <= maxSeconds + 1e-9; t += tickEvery) {
    const updates = Math.floor(ups * t);
    const x = updates * 0.05 * gain * lrFactor * batchGain;
    let progress = 1 - Math.exp(-x / 400);
    if (diverged) progress = Math.max(0, 0.02 - (t / maxSeconds) * 0.02);
    const acc = Math.min(0.999, Math.max(0.03, 0.1 + (asym - 0.1) * progress + (rng() - 0.5) * 2 * noiseSd));
    const p = (acc - 0.1) / (asym - 0.1);
    const preds = probeLabels.map((y, i) => (p + (rng() - 0.5) * 0.12 > thr[i] ? y : (y + 1 + (i % 9)) % 10));
    const loss = diverged && t > maxSeconds * 0.3 ? null : (0.15 + 2.2 * Math.exp(-x / 400)) * scale * (1 + (rng() - 0.5) * 0.04);
    out.push({ t: +t.toFixed(3), step: updates, samples_seen: updates * bsz, updates, loss, acc, probe_preds: preds });
  }
  return out;
}
