import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { MLP } from "../mlp";
import { ACTIVATIONS, defaultParams, getActivation } from "../activations";
import { leastSquaresLine, lineAccuracy, spirals, twoClusters } from "../datasets";
import { forward, makeMockLogmelModel, makeMockModel, softmax, type Layer, type Model } from "../inference";
import { DEFAULTS, countParams, defaultConfig, featureDim, previewLocal, tierFor, widthsOf } from "../chessConfig";

const root = resolve(__dirname, "../../../..");
const golden = (n: string) => JSON.parse(readFileSync(resolve(root, "shared/golden", n), "utf8"));

describe("activations", () => {
  it("derivatives match finite differences", () => {
    for (const a of ACTIVATIONS) {
      if (a.dead) continue;
      const p = defaultParams(a);
      for (const x of [-2.3, -0.7, 0.4, 1.9]) {
        const h = 1e-5;
        const num = (a.f(x + h, p) - a.f(x - h, p)) / (2 * h);
        expect(Math.abs(num - a.df(x, p)), `${a.name} at ${x}`).toBeLessThan(1e-3);
      }
    }
  });
  it("hard step has zero slope everywhere; leaky relu at alpha=1 is linear", () => {
    const hs = getActivation("hard_step");
    expect([-1, 0.5, 3].every((x) => hs.df(x, {}) === 0)).toBe(true);
    const lr = getActivation("leaky_relu");
    expect(lr.f(-2, { alpha: 1 })).toBe(-2);
  });
});

describe("playground MLP", () => {
  it("gradients: training lowers the loss on clusters", () => {
    const d = twoClusters(120, 1);
    const net = new MLP({ hidden: [8], activation: "relu", params: {}, seed: 1, lr: 0.02 });
    const before = net.loss(d.X, d.y);
    for (let i = 0; i < 60; i++) net.trainEpoch(d.X, d.y);
    expect(net.loss(d.X, d.y)).toBeLessThan(before * 0.5);
    expect(net.accuracy(d.X, d.y)).toBeGreaterThan(0.95);
  });
  it("finds a spiral with ReLU, but a deep LINEAR stack cannot (and collapses to one line)", () => {
    const d = spirals(200, 2);
    const relu = new MLP({ hidden: [24, 24, 24], activation: "relu", params: {}, seed: 3, lr: 0.02 });
    for (let i = 0; i < 500; i++) relu.trainEpoch(d.X, d.y);
    expect(relu.accuracy(d.X, d.y)).toBeGreaterThan(0.9);
    const lin = new MLP({ hidden: [16, 16, 16], activation: "linear", params: {}, seed: 3, lr: 0.02 });
    for (let i = 0; i < 200; i++) lin.trainEpoch(d.X, d.y);
    expect(lin.accuracy(d.X, d.y)).toBeLessThan(0.65);
    const c = lin.collapsed();
    for (const [x, y] of [[0.3, -0.2], [-0.8, 0.5]]) { // the collapsed single layer reproduces the deep stack
      const p = 1 / (1 + Math.exp(-(c.w1 * x + c.w2 * y + c.b)));
      expect(Math.abs(p - lin.predict(x, y))).toBeLessThan(1e-6);
    }
  });
  it("depth 1 / width 2 ReLU still fails on spirals", () => {
    const d = spirals(200, 2);
    const tiny = new MLP({ hidden: [2], activation: "relu", params: {}, seed: 4, lr: 0.02 });
    for (let i = 0; i < 400; i++) tiny.trainEpoch(d.X, d.y);
    expect(tiny.accuracy(d.X, d.y)).toBeLessThan(0.8);
  });
  it("least-squares line separates clusters, not spirals", () => {
    const c = twoClusters(200, 1), s = spirals(200, 2);
    expect(lineAccuracy(c.X, c.y, leastSquaresLine(c.X, c.y))).toBeGreaterThan(0.95);
    expect(lineAccuracy(s.X, s.y, leastSquaresLine(s.X, s.y))).toBeLessThan(0.65);
  });
  it("a hard-step network cannot learn (zero gradients)", () => {
    const d = spirals(100, 2);
    const net = new MLP({ hidden: [8, 8], activation: "hard_step", params: {}, seed: 1, lr: 0.02 });
    for (let i = 0; i < 40; i++) net.trainEpoch(d.X, d.y);
    expect(net.accuracy(d.X, d.y)).toBeLessThan(0.7);
  });
});

describe("inference vs PyTorch golden", () => {
  const g = golden("inference.json");
  for (const m of g.models) {
    it(`${m.name} forward matches`, () => {
      const model: Model = { name: m.name, isMock: false, weights: Float32Array.from(m.weights), spec: { layers: m.layers as Layer[], input: { kind: "t", shape: m.input_shape } } };
      for (const c of m.cases) {
        const y = forward(model, Float32Array.from(c.x));
        c.y.forEach((v: number, i: number) => expect(Math.abs(y[i] - v)).toBeLessThan(g.tolerance));
      }
    });
  }
  it("mock models run and give a probability distribution", () => {
    const m = makeMockModel("doodle");
    const p = softmax(forward(m, new Float32Array(784).fill(0.1)));
    expect(p.length).toBe(10);
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-5);
    const lm = makeMockLogmelModel();
    expect(forward(lm, new Float32Array(40 * 98).fill(0.1), [1, 40, 98]).length).toBe(10);
  });
});

describe("chess config matches the backend", () => {
  const g = golden("../config_defaults.json");
  for (const [i, c] of (g.param_cases as { config: any; param_count: number; tier: string; in_dim: number; matmul_params: number }[]).entries()) {
    it(`param case ${i}`, () => {
      const cfg = { ...defaultConfig(), ...c.config, input_extras: { ...DEFAULTS.input_extras, ...(c.config.input_extras ?? {}) } };
      const p = previewLocal(cfg);
      expect(featureDim(cfg.input_extras)).toBe(c.in_dim);
      expect(p.paramCount).toBe(c.param_count);
      expect(p.tier.name).toBe(c.tier);
      expect(p.flopsPerSample).toBe(6 * c.matmul_params);
    });
  }
  it("helpers", () => {
    expect(widthsOf({ layers: 2, width: 9, layer_widths: null })).toEqual([9, 9]);
    expect(widthsOf({ layers: 2, width: 9, layer_widths: [4, 5, 6] })).toEqual([4, 5, 6]);
    expect(countParams(773, [32, 32])).toBe(773 * 32 + 32 + 32 * 32 + 32 + 32 + 1);
    expect(tierFor(10).name).toBe("Light");
    expect(previewLocal({ ...defaultConfig(), layers: 16, width: 4096 }).errors[0]).toMatch(/too many parameters/);
  });
});
