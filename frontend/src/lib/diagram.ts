// Turns a forward trace of a dense model into what the network diagram shows (a sample of nodes per layer + edge strengths).
import type { Model, Trace } from "./inference";

export interface DiagramData {
  cols: { label: string; shown: number[]; acts: number[]; total: number }[];
  edges: number[][][]; // edges[l][i][j]: strength (0..1) between shown node i of col l and shown node j of col l+1
  top: number; // index (in the last column) of the strongest output
}

const sample = (n: number, k: number) => (n <= k ? Array.from({ length: n }, (_, i) => i) : Array.from({ length: k }, (_, i) => Math.round((i * (n - 1)) / (k - 1))));

/** `input` is the model input (e.g. 784 pixels); only dense+relu stacks are supported (the doodle model). */
export function diagramFrom(m: Model, input: Float32Array, trace: Trace, probs: Float32Array, maxHidden = 14): DiagramData {
  const dense = m.spec.layers.map((l, i) => ({ l, i })).filter((x) => x.l.type === "dense");
  const cols: DiagramData["cols"] = [];
  const inInk = Array.from(input, (v, i) => ({ v, i })).filter((p) => p.v > 0.2).map((p) => p.i);
  const inShown = (inInk.length >= maxHidden ? sample(inInk.length, maxHidden).map((k) => inInk[k]) : sample(input.length, maxHidden));
  cols.push({ label: "pixels", shown: inShown, acts: inShown.map((i) => input[i]), total: input.length });
  let mx = 1e-6;
  dense.forEach(({ i }, di) => {
    const isLast = di === dense.length - 1;
    const out = isLast ? probs : trace.outputs[i + 1]?.data ?? trace.outputs[i].data; // after the relu that follows
    const n = out.length;
    const shown = sample(n, isLast ? n : maxHidden);
    const vals = shown.map((k) => out[k]);
    if (!isLast) mx = Math.max(mx, ...vals);
    cols.push({ label: isLast ? "answer" : `layer ${di + 1}`, shown, acts: vals, total: n });
  });
  // normalise hidden activations to 0..1
  for (let c = 1; c < cols.length - 1; c++) { const cm = Math.max(1e-6, ...cols[c].acts); cols[c].acts = cols[c].acts.map((v) => v / cm); }
  const edges: number[][][] = [];
  dense.forEach(({ l }, di) => {
    if (l.type !== "dense") return;
    const w = m.weights.subarray(l.w.offset, l.w.offset + l.in * l.out);
    const from = cols[di], to = cols[di + 1];
    let emax = 1e-9;
    const e = from.shown.map((fi, a) => to.shown.map((tj) => { const s = Math.abs(w[tj * l.in + fi] * from.acts[a]); emax = Math.max(emax, s); return s; }));
    edges.push(e.map((row) => row.map((s) => s / emax)));
  });
  return { cols, edges, top: probs.indexOf(Math.max(...probs)) };
}
