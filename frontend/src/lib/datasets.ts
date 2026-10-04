// Playground datasets (SPEC 4.4: generated in the browser, no files). Points live in [-1, 1]^2; label 0 = blue, 1 = orange.
import { gaussian, mulberry32 } from "./rng";

export interface Dataset {
  X: Float32Array; // n x 2
  y: Uint8Array;
}

export function twoClusters(n = 200, seed = 1): Dataset {
  const rng = mulberry32(seed);
  const X = new Float32Array(2 * n);
  const y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    y[i] = c;
    const cx = c ? 0.5 : -0.5, cy = c ? 0.35 : -0.35;
    X[2 * i] = cx + gaussian(rng) * 0.2;
    X[2 * i + 1] = cy + gaussian(rng) * 0.2;
  }
  return { X, y };
}

export function spirals(n = 200, seed = 2, noise = 0.03): Dataset {
  const rng = mulberry32(seed);
  const X = new Float32Array(2 * n);
  const y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    const k = Math.floor(i / 2) / (n / 2);
    const r = 0.1 + 0.85 * k;
    const th = 2.6 * Math.PI * k + c * Math.PI;
    y[i] = c;
    X[2 * i] = r * Math.cos(th) + gaussian(rng) * noise;
    X[2 * i + 1] = r * Math.sin(th) + gaussian(rng) * noise;
  }
  return { X, y };
}

/** Point-by-point interpolation between two datasets with the same n (the clusters "morph" into spirals). */
export function lerpPoints(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

/** Least-squares line w1*x + w2*y + b ~ +-1 (a small 3x3 linear solve). Class 1 -> +1, class 0 -> -1. */
export function leastSquaresLine(X: Float32Array, y: Uint8Array): { w1: number; w2: number; b: number } {
  const n = y.length;
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const r = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const f = [X[2 * i], X[2 * i + 1], 1];
    const t = y[i] ? 1 : -1;
    for (let a = 0; a < 3; a++) {
      r[a] += f[a] * t;
      for (let c = 0; c < 3; c++) A[a][c] += f[a] * f[c];
    }
  }
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let k = col + 1; k < 3; k++) if (Math.abs(A[k][col]) > Math.abs(A[piv][col])) piv = k;
    [A[col], A[piv]] = [A[piv], A[col]];
    [r[col], r[piv]] = [r[piv], r[col]];
    for (let k = col + 1; k < 3; k++) {
      const m = A[k][col] / A[col][col];
      for (let c = col; c < 3; c++) A[k][c] -= m * A[col][c];
      r[k] -= m * r[col];
    }
  }
  const w = [0, 0, 0];
  for (let k = 2; k >= 0; k--) {
    let s = r[k];
    for (let c = k + 1; c < 3; c++) s -= A[k][c] * w[c];
    w[k] = s / A[k][k];
  }
  return { w1: w[0], w2: w[1], b: w[2] };
}

export function lineAccuracy(X: Float32Array, y: Uint8Array, line: { w1: number; w2: number; b: number }): number {
  let ok = 0;
  for (let i = 0; i < y.length; i++) ok += (line.w1 * X[2 * i] + line.w2 * X[2 * i + 1] + line.b > 0 ? 1 : 0) === y[i] ? 1 : 0;
  return ok / y.length;
}
