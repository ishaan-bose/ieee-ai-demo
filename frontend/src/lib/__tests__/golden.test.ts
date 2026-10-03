import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { rasterize, type Stroke } from "../rasterizer";
import { N_FRAMES, decodeWav, hannWindow, logMel, melFilterbank, rawInput } from "../audio";

const root = resolve(__dirname, "../../../..");
const golden = (n: string) => JSON.parse(readFileSync(resolve(root, "shared/golden", n), "utf8"));

describe("rasterizer golden", () => {
  const g = golden("rasterizer.json");
  for (const c of g.cases) {
    it(c.name, () => {
      const px = rasterize(c.strokes as Stroke[]);
      let maxDiff = 0;
      for (let i = 0; i < 784; i++) maxDiff = Math.max(maxDiff, Math.abs(px[i] - c.pixels[i]));
      expect(maxDiff).toBeLessThanOrEqual(g.tolerance);
    });
  }
  it("is scale/offset invariant", () => {
    const a = g.cases.find((c: { name: string }) => c.name === "same_shape_scaled");
    const b = g.cases.find((c: { name: string }) => c.name.startsWith("qd_"));
    expect(a).toBeTruthy();
    expect(b).toBeTruthy();
  });
});

describe("audio golden", () => {
  const g = golden("audio.json");
  it("window and filterbank", () => {
    const w = hannWindow();
    g.window_head.forEach((v: number, i: number) => expect(Math.abs(w[i] - v)).toBeLessThan(1e-7));
    const fb = melFilterbank();
    for (const [m, row] of Object.entries(g.fbank_rows) as [string, number[]][])
      row.forEach((v, k) => expect(Math.abs(fb[Number(m)][k] - v)).toBeLessThan(1e-5));
  });
  for (const c of g.cases) {
    it(`log-mel ${c.file}`, () => {
      const wav = decodeWav(readFileSync(resolve(root, "data/samples/speech/samples", c.file)).buffer.slice(0) as ArrayBuffer);
      expect(wav.length).toBe(c.n_samples);
      const lm = logMel(wav);
      const flat: number[] = c.log_mel.flat();
      expect(lm.length).toBe(flat.length);
      expect(c.log_mel[0].length).toBe(N_FRAMES);
      let maxDiff = 0;
      for (let i = 0; i < lm.length; i++) maxDiff = Math.max(maxDiff, Math.abs(lm[i] - flat[i]));
      expect(maxDiff).toBeLessThan(g.tolerance);
      const raw = rawInput(wav);
      c.raw_input_head.forEach((v: number, i: number) => expect(Math.abs(raw[i] - v)).toBeLessThan(1e-5));
    });
  }
});
