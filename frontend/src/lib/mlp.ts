// A small MLP with backprop for the Act 1 playground (runs inside a Web Worker). 2 inputs -> hidden layers -> 1 sigmoid output.
import { getActivation, type ActivationDef } from "./activations";
import { gaussian, mulberry32 } from "./rng";

export interface MlpConfig {
  hidden: number[]; // widths of the hidden layers (depth = hidden.length)
  activation: string; // "linear" = no nonlinearity (linear layers only)
  params: Record<string, number>;
  seed: number;
  lr: number;
}

export class MLP {
  W: Float64Array[] = []; // layer l: out x in, row-major
  b: Float64Array[] = [];
  sizes: number[];
  act: ActivationDef;
  p: Record<string, number>;
  lr: number;
  private m: Float64Array[] = []; // Adam state
  private v: Float64Array[] = [];
  private t = 0;
  private rng: () => number;

  constructor(cfg: MlpConfig) {
    this.sizes = [2, ...cfg.hidden, 1];
    this.act = getActivation(cfg.activation);
    this.p = cfg.params;
    this.lr = cfg.lr;
    this.rng = mulberry32(cfg.seed);
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const fanIn = this.sizes[l], fanOut = this.sizes[l + 1];
      const std = this.act.name === "relu" || this.act.name === "leaky_relu" ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));
      const w = new Float64Array(fanIn * fanOut);
      for (let i = 0; i < w.length; i++) w[i] = gaussian(this.rng) * std;
      this.W.push(w);
      this.b.push(new Float64Array(fanOut));
      this.m.push(new Float64Array(w.length + fanOut));
      this.v.push(new Float64Array(w.length + fanOut));
    }
  }

  get paramCount(): number {
    return this.W.reduce((s, w, l) => s + w.length + this.b[l].length, 0);
  }

  /** Forward one point. Returns the output probability. */
  predict(x: number, y: number): number {
    let a = Float64Array.of(x, y);
    for (let l = 0; l < this.W.length; l++) {
      const out = this.sizes[l + 1], inn = this.sizes[l];
      const z = new Float64Array(out);
      for (let o = 0; o < out; o++) {
        let s = this.b[l][o];
        for (let i = 0; i < inn; i++) s += this.W[l][o * inn + i] * a[i];
        z[o] = s;
      }
      if (l === this.W.length - 1) return 1 / (1 + Math.exp(-z[0]));
      for (let o = 0; o < out; o++) z[o] = this.act.f(z[o], this.p);
      a = z;
    }
    return 0.5;
  }

  /** Full pass over (X, y) in mini-batches with Adam. Returns the mean loss (NaN if the network blew up). */
  trainEpoch(X: Float32Array, y: Uint8Array, batch = 64, order?: Int32Array): number {
    const n = y.length;
    const idx = order ?? Int32Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { // shuffle
      const j = Math.floor(this.rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    let total = 0;
    for (let s = 0; s < n; s += batch) {
      const bi = idx.subarray(s, Math.min(n, s + batch));
      total += this.step(X, y, bi);
    }
    return total / n;
  }

  private step(X: Float32Array, y: Uint8Array, bi: Int32Array): number {
    const L = this.W.length;
    const gW = this.W.map((w) => new Float64Array(w.length));
    const gb = this.b.map((b) => new Float64Array(b.length));
    let loss = 0;
    for (const k of bi) {
      // forward, keeping pre-activations z and activations a
      const as: Float64Array[] = [Float64Array.of(X[2 * k], X[2 * k + 1])];
      const zs: Float64Array[] = [];
      for (let l = 0; l < L; l++) {
        const out = this.sizes[l + 1], inn = this.sizes[l];
        const z = new Float64Array(out);
        for (let o = 0; o < out; o++) {
          let s = this.b[l][o];
          for (let i = 0; i < inn; i++) s += this.W[l][o * inn + i] * as[l][i];
          z[o] = s;
        }
        zs.push(z);
        as.push(l === L - 1 ? z : z.map((v) => this.act.f(v, this.p)));
      }
      const p = 1 / (1 + Math.exp(-zs[L - 1][0]));
      const eps = 1e-9;
      loss += -(y[k] ? Math.log(p + eps) : Math.log(1 - p + eps));
      // backward (sigmoid + BCE gives dL/dz = p - y)
      let delta = Float64Array.of(p - y[k]);
      for (let l = L - 1; l >= 0; l--) {
        const out = this.sizes[l + 1], inn = this.sizes[l];
        for (let o = 0; o < out; o++) {
          gb[l][o] += delta[o];
          for (let i = 0; i < inn; i++) gW[l][o * inn + i] += delta[o] * as[l][i];
        }
        if (l > 0) {
          const nd = new Float64Array(inn);
          for (let i = 0; i < inn; i++) {
            let s = 0;
            for (let o = 0; o < out; o++) s += this.W[l][o * inn + i] * delta[o];
            nd[i] = s * this.act.df(zs[l - 1][i], this.p);
          }
          delta = nd;
        }
      }
    }
    // Adam update
    this.t++;
    const b1 = 0.9, b2 = 0.999, lr = this.lr, scale = 1 / bi.length;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let l = 0; l < L; l++) {
      const nw = this.W[l].length;
      for (let i = 0; i < nw + this.b[l].length; i++) {
        const g = (i < nw ? gW[l][i] : gb[l][i - nw]) * scale;
        this.m[l][i] = b1 * this.m[l][i] + (1 - b1) * g;
        this.v[l][i] = b2 * this.v[l][i] + (1 - b2) * g * g;
        const upd = (lr * (this.m[l][i] / c1)) / (Math.sqrt(this.v[l][i] / c2) + 1e-8);
        if (i < nw) this.W[l][i] -= upd; else this.b[l][i - nw] -= upd;
      }
    }
    return loss;
  }

  accuracy(X: Float32Array, y: Uint8Array): number {
    let ok = 0;
    for (let k = 0; k < y.length; k++) ok += (this.predict(X[2 * k], X[2 * k + 1]) > 0.5 ? 1 : 0) === y[k] ? 1 : 0;
    return ok / y.length;
  }

  loss(X: Float32Array, y: Uint8Array): number {
    let s = 0;
    for (let k = 0; k < y.length; k++) {
      const p = this.predict(X[2 * k], X[2 * k + 1]);
      s += -(y[k] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
    }
    return s / y.length;
  }

  /** Probability on a G x G grid over [-1.2, 1.2]^2 (row 0 = top = y max). */
  grid(G: number, extent = 1.2): Float32Array {
    const out = new Float32Array(G * G);
    for (let r = 0; r < G; r++)
      for (let c = 0; c < G; c++) out[r * G + c] = this.predict(-extent + ((c + 0.5) / G) * 2 * extent, extent - ((r + 0.5) / G) * 2 * extent);
    return out;
  }

  /** For a linear-only network: the single equivalent layer W = W_L ... W_1, b (returns w1, w2, bias of the output logit). */
  collapsed(): { w1: number; w2: number; b: number } {
    // propagate the affine map through the layers: z = A x + c with A (k x 2)
    let A = [[1, 0], [0, 1]];
    let c = [0, 0];
    for (let l = 0; l < this.W.length; l++) {
      const out = this.sizes[l + 1], inn = this.sizes[l];
      const nA = Array.from({ length: out }, () => [0, 0]);
      const nc = new Array(out).fill(0);
      for (let o = 0; o < out; o++) {
        nc[o] = this.b[l][o];
        for (let i = 0; i < inn; i++) {
          const w = this.W[l][o * inn + i];
          nA[o][0] += w * A[i][0];
          nA[o][1] += w * A[i][1];
          nc[o] += w * c[i];
        }
      }
      A = nA; c = nc;
    }
    return { w1: A[0][0], w2: A[0][1], b: c[0] };
  }
}
