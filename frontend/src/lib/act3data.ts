// Cached Act 3 data (precomputed by backend/scripts/train_showcase.py): metrics trap + overfitting. Synthetic stand-ins when missing.
import { mulberry32 } from "./rng";

export interface BinMetrics { accuracy: number; recall: number | null; precision: number | null; f1?: number; confusion: number[][] }
export interface TrapData {
  isMock: boolean; positive_fraction?: { test: number };
  always_no: BinMetrics; naive: { test: BinMetrics; epochs: { epoch: number; val_accuracy: number; val_recall: number | null }[] };
  weighted: { test: BinMetrics };
}
export interface OverfitRun { n: number; epochs: { epoch: number; train_loss: number; train_acc: number; val_loss: number; val_acc: number }[] }
export interface OverfitData { isMock: boolean; default_n: number; runs: OverfitRun[] }

async function getJson<T>(path: string): Promise<T | null> {
  try { const r = await fetch(path); return r.ok && r.headers.get("content-type")?.includes("json") ? ((await r.json()) as T) : null; } catch { return null; }
}

export async function loadTrap(): Promise<TrapData> {
  const d = await getJson<TrapData>("/cache/act3/trap.json");
  if (d) return { ...d, isMock: false };
  const n = 2000, pos = 40, tn = n - pos;
  return {
    isMock: true,
    always_no: { accuracy: tn / n, recall: 0, precision: null, confusion: [[tn, 0], [pos, 0]] },
    naive: { test: { accuracy: 0.981, recall: 0.2, precision: 0.62, confusion: [[1971, 5], [32, 8]] }, epochs: Array.from({ length: 8 }, (_, i) => ({ epoch: i + 1, val_accuracy: 0.975 + 0.001 * i, val_recall: Math.min(0.3, 0.02 * i) })) },
    weighted: { test: { accuracy: 0.93, recall: 0.9, precision: 0.22, confusion: [[1824, 136], [4, 36]] } },
  };
}

export async function loadOverfit(): Promise<OverfitData> {
  const d = await getJson<OverfitData>("/cache/act3/overfit.json");
  if (d) return { ...d, isMock: false };
  const rng = mulberry32(5);
  const runs: OverfitRun[] = [100, 300, 1000, 3000, 10000, 30000].map((n) => {
    const E = Math.max(12, Math.min(60, Math.round(60 * Math.sqrt(300 / n))));
    const ceiling = 0.55 + 0.4 * (1 - Math.exp(-Math.log10(n / 60) * 0.9));
    return { n, epochs: Array.from({ length: E }, (_, i) => {
      const t = (i + 1) / E;
      const tr = Math.min(0.999, 0.15 + (1 - 0.15) * (1 - Math.exp(-t * 5)) * (n < 3000 ? 1 : 0.93));
      const va = Math.max(0.1, 0.12 + (ceiling - 0.12) * (1 - Math.exp(-t * 4)) + (rng() - 0.5) * 0.02 - (n < 1000 ? 0.04 * t : 0));
      return { epoch: i + 1, train_loss: 2.3 * Math.exp(-t * 4), train_acc: tr, val_loss: 1.4 + (n < 1000 ? 1.6 * t : 0.1 * (1 - t)), val_acc: va };
    }) };
  });
  return { isMock: true, default_n: 300, runs };
}
