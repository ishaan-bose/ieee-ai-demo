// Mirror of backend/app/training/race_grid.py (the planned grid of race lanes) + UI option lists.
export type BatchSize = number | "full";
export interface LaneCfg {
  id: string;
  loss: "mse" | "ce" | "l1" | "huber" | "focal" | "label_smooth";
  delta: number; gamma: number; eps: number;
  lr: number; batch_size: BatchSize;
  optimizer: "sgd" | "momentum" | "adam";
  hidden: number[]; activation: string;
}
export type RaceKind = "loss" | "lr" | "batch";

export const BASE: Omit<LaneCfg, "id"> = { loss: "ce", delta: 1, gamma: 2, eps: 0.1, lr: 0.03, batch_size: 64, optimizer: "momentum", hidden: [128, 64], activation: "relu" };
export const BATCH_RACE_LR = 0.02;
export const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];
export const BATCHES: BatchSize[] = [1, 8, 32, 128, 512, "full"];

export const LOSS_CHOICES: { loss: LaneCfg["loss"]; label: string; blurb: string; param?: { key: "delta" | "gamma" | "eps"; label: string; min: number; max: number; step: number; def: number } }[] = [
  { loss: "mse", label: "MSE", blurb: "squared error on the output probabilities" },
  { loss: "ce", label: "Cross-entropy", blurb: "the classification standard" },
  { loss: "l1", label: "L1 / MAE", blurb: "absolute error on the probabilities" },
  { loss: "huber", label: "Huber", blurb: "squared for small errors, absolute for big ones", param: { key: "delta", label: "δ", min: 0.1, max: 3, step: 0.1, def: 1 } },
  { loss: "focal", label: "Focal", blurb: "focus on the hard examples", param: { key: "gamma", label: "γ", min: 0, max: 5, step: 0.5, def: 2 } },
  { loss: "label_smooth", label: "Label-smoothed CE", blurb: "never be 100% sure", param: { key: "eps", label: "ε", min: 0, max: 0.5, step: 0.05, def: 0.1 } },
];

export const laneCfg = (id: string, over: Partial<LaneCfg> = {}): LaneCfg => ({ id, ...BASE, ...over });

/** Cache key of a lane config, if it is exactly on the recorded grid. Nearest-match is in nearestKey(). */
export function nearestKey(kind: RaceKind, cfg: LaneCfg, available: { key: string; kind: string; config: Record<string, unknown> }[]): string | null {
  const pool = available.filter((a) => a.kind === kind);
  if (!pool.length) return null;
  const dist = (a: Record<string, unknown>): number => {
    if (kind === "lr") return Math.abs(Math.log10(cfg.lr) - Math.log10(a.lr as number));
    if (kind === "batch") {
      const b = a.batch_size as BatchSize, c = cfg.batch_size;
      if (b === "full" || c === "full") return b === c ? 0 : 50;
      return Math.abs(Math.log2(c) - Math.log2(b));
    }
    if (a.loss !== cfg.loss) return 100;
    const p = cfg.loss === "huber" ? "delta" : cfg.loss === "focal" ? "gamma" : cfg.loss === "label_smooth" ? "eps" : null;
    return p ? Math.abs((cfg[p] as number) - (a[p] as number)) : 0;
  };
  return pool.map((a) => ({ key: a.key, d: dist(a.config) })).sort((x, y) => x.d - y.d)[0].key;
}
