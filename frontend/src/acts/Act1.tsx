import { AnimatePresence, motion } from "framer-motion";
import { line as d3line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ActivationCard, type CardResult } from "../components/ActivationCard";
import { BoundaryCanvas, type Line } from "../components/BoundaryCanvas";
import { Badge, Gauge, Unlock } from "../components/ui";
import { ACTIVATIONS, defaultParams, getActivation } from "../lib/activations";
import { leastSquaresLine, lerpPoints, lineAccuracy, spirals, twoClusters } from "../lib/datasets";
import { mulberry32 } from "../lib/rng";
import { startRunTimer, stopRunTimer } from "../lib/runTimer";
import { addRun } from "../lib/session";
import { stageIndex } from "../state/stages";
import { useStage, useStageEvent } from "../state/StageProvider";

const G = 48;
const FRAME_MS = 33; // the morph / line animations re-render the whole act per step: ~30 steps a second look the same as 60 and cost half
const N = 200;
const sub = (n: number) => String.fromCharCode(0x2080 + n);
const clusters = twoClusters(N, 1), spiral = spirals(N, 2);

interface WorkerDone { type: "done"; id: number; acc: number; loss: number; grid: Float32Array; broke?: "nan" | "flat"; stopped?: boolean; paramCount: number }
interface WorkerProgress { type: "progress"; id: number; epoch: number; acc: number; loss: number; grid: Float32Array }

function randomLine(rng: () => number): Line {
  const a = rng() * Math.PI * 2;
  return { w1: Math.cos(a) * 2, w2: Math.sin(a) * 2, b: (rng() - 0.5) * 1.2 };
}

function Blocks({ depth, fused }: { depth: number; fused: boolean }) {
  return (
    <div className="flex items-center gap-2 text-xl" data-testid="blocks">
      <AnimatePresence mode="wait">
        {!fused ? (
          <motion.div key="stack" className="flex items-center gap-2" exit={{ opacity: 0, scale: 0.6 }}>
            {Array.from({ length: depth }, (_, i) => (
              <motion.div key={i} layout initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-lg bg-sky-600/30 px-3 py-2 font-mono text-sky-200 ring-1 ring-sky-400/50">W{sub(i + 1)}</motion.div>
            ))}
          </motion.div>
        ) : (
          <motion.div key="fused" initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: "spring", stiffness: 180, damping: 14 }}
            className="rounded-xl bg-amber-500/20 px-5 py-3 font-mono text-2xl text-amber-200 ring-2 ring-amber-400" data-testid="fused">
            {Array.from({ length: depth }, (_, i) => `W${sub(depth - i)}`).join("·")} = W
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ReluPlot() {
  const xs = Array.from({ length: 41 }, (_, i) => -2 + i * 0.1);
  const sx = scaleLinear().domain([-2, 2]).range([6, 194]), sy = scaleLinear().domain([-0.3, 2]).range([94, 6]);
  const d = d3line<number>().x((x) => sx(x)).y((x) => sy(Math.max(0, x)))(xs) ?? "";
  return (
    <svg width="200" height="100" className="rounded-lg bg-slate-900" data-testid="relu-plot">
      <line x1={sx(0)} x2={sx(0)} y1="0" y2="100" stroke="#334155" /><line x1="0" x2="200" y1={sy(0)} y2={sy(0)} stroke="#334155" />
      <path d={d} fill="none" stroke="#56B4E9" strokeWidth="3" /><text x="10" y="18" fontSize="15" className="fill-slate-200">max(0, x)</text>
    </svg>
  );
}

export function Act1() {
  const { stage, forceFallback, resetNonce } = useStage();
  const sIdx = stageIndex(stage.id);
  const isLate = sIdx >= stageIndex("a1-spirals");
  const [morph, setMorph] = useState(isLate ? 1 : 0);
  const [line, setLine] = useState<Line>(() => randomLine(mulberry32(7)));
  const [grid, setGrid] = useState<Float32Array | null>(null);
  const [acc, setAcc] = useState<number | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [running, setRunning] = useState(false);
  const [broke, setBroke] = useState<string | null>(null);
  const [depth, setDepth] = useState(3);
  const [width, setWidth] = useState(16);
  const [fused, setFused] = useState(false);
  const [slow, setSlow] = useState(false);
  const [found, setFound] = useState(false);
  const [selected, setSelected] = useState("relu");
  const [params, setParams] = useState<Record<string, Record<string, number>>>(() => Object.fromEntries(ACTIVATIONS.map((a) => [a.name, defaultParams(a)])));
  const [results, setResults] = useState<Record<string, CardResult>>({});
  const worker = useRef<Worker | null>(null);
  const runId = useRef(0);
  const rng = useRef(mulberry32(11));
  const morphRaf = useRef(0);
  const lineRaf = useRef(0);

  const X = useMemo(() => lerpPoints(clusters.X, spiral.X, morph), [morph]);
  const y = clusters.y; // labels agree between the two datasets (i % 2)

  // The worker is created on the first training run, not on mount: entering Act 1 stays cheap (a module worker costs a full module-graph load,
  // twice under StrictMode in dev), and Acts that never train never pay for it. It is terminated when Act 1 unmounts.
  const getWorker = useCallback(() => (worker.current ??= new Worker(new URL("../workers/mlp.worker.ts", import.meta.url), { type: "module" })), []);
  useEffect(() => () => { worker.current?.terminate(); worker.current = null; }, []);

  const stopRun = useCallback(() => { worker.current?.postMessage({ type: "stop" }); runId.current++; setRunning(false); stopRunTimer(); }, []);

  // stage-driven setup (also works when jumping straight to a later stage with the G menu)
  useEffect(() => {
    stopRun();
    setGrid(null); setAcc(null); setEpoch(0); setBroke(null); setFused(false); setFound(false);
    if (stage.id === "a1-clusters") setLine(randomLine(rng.current));
    const target = sIdx >= stageIndex("a1-spirals") ? 1 : 0;
    cancelAnimationFrame(morphRaf.current);
    const from = morph, t0 = performance.now();
    let lastStep = 0;
    const loop = () => {
      const now = performance.now(), f = Math.min(1, (now - t0) / 1400);
      if (f >= 1 || now - lastStep >= FRAME_MS) { lastStep = now; setMorph(from + (target - from) * f); }
      if (f < 1) morphRaf.current = requestAnimationFrame(loop);
    };
    if (from !== target) morphRaf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(morphRaf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage.id, resetNonce]);

  const usingLine = sIdx < stageIndex("a1-depth");
  const lineAcc = lineAccuracy(X, y, line);
  const shownAcc = usingLine ? lineAcc : acc ?? 0.5;

  const animateLine = useCallback((to: Line) => {
    const from = line, t0 = performance.now();
    let lastStep = 0;
    cancelAnimationFrame(lineRaf.current);
    const loop = () => {
      const now = performance.now(), f = Math.min(1, (now - t0) / 1100), e = f * f * (3 - 2 * f);
      if (f >= 1 || now - lastStep >= FRAME_MS) { lastStep = now; setLine({ w1: from.w1 + (to.w1 - from.w1) * e, w2: from.w2 + (to.w2 - from.w2) * e, b: from.b + (to.b - from.b) * e }); }
      if (f < 1) lineRaf.current = requestAnimationFrame(loop);
    };
    loop();
  }, [line]);
  useEffect(() => () => cancelAnimationFrame(lineRaf.current), []);

  const train = useCallback((activation: string, hidden: number[], epochs: number) => {
    stopRun();
    const id = ++runId.current;
    setRunning(true); setBroke(null); setEpoch(0);
    startRunTimer(15);
    const w = getWorker();
    const act = getActivation(activation);
    const p = params[activation] ?? {};
    w.onmessage = (e: MessageEvent<WorkerProgress | WorkerDone>) => {
      const m = e.data;
      if (m.id !== runId.current) return;
      if (m.type === "progress") { setGrid(m.grid); setAcc(m.acc); setEpoch(m.epoch); return; }
      setGrid(m.grid); setAcc(m.acc); setRunning(false); stopRunTimer(); setEpoch(m.stopped ? 0 : epochs);
      const b = m.broke ?? (act.dead ? "flat" : undefined);
      setBroke(b ?? null);
      setResults((r) => ({ ...r, [activation]: { acc: m.acc, broke: b ?? null } }));
      addRun(m.paramCount, 6 * m.paramCount * epochs * N);
    };
    w.postMessage({ type: "train", id, X: Float32Array.from(X), y: Uint8Array.from(y), epochs, slow, grid: G, cfg: { hidden, activation, params: p, seed: 5, lr: activation === "linear" ? 0.01 : 0.02 } });
  }, [X, y, params, slow, stopRun, getWorker]);

  const trainFor = useCallback(() => {
    if (stage.id === "a1-depth") train("linear", Array(depth).fill(8), 160);
    else if (stage.id === "a1-relu") train("relu", Array(depth).fill(width), 450);
    else if (stage.id === "a1-gallery") train(selected, Array(3).fill(16), 350);
  }, [stage.id, depth, width, selected, train]);

  useStageEvent("run", () => {
    switch (stage.id) {
      case "a1-clusters": setLine(randomLine(rng.current)); break;
      case "a1-knobs": case "a1-spirals": setFound(true); animateLine(leastSquaresLine(X, y)); break;
      case "a1-fuse": setFused((f) => !f); break;
      case "a1-depth": case "a1-relu": case "a1-gallery": if (running) stopRun(); else trainFor(); break;
    }
  });
  useStageEvent("skip", () => { if (["a1-depth", "a1-relu", "a1-gallery"].includes(stage.id)) trainFor(); });
  useStageEvent("reset", () => { stopRun(); setGrid(null); setAcc(null); setResults({}); setFused(false); });
  useEffect(() => { const k = (e: KeyboardEvent) => { if (e.key === "l" || e.key === "L") setSlow((s) => !s); }; window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k); }, []);
  void forceFallback;

  const onParam = useCallback((name: string, v: number) => { const a = getActivation(name); if (a.params) setParams((p) => ({ ...p, [name]: { [a.params!.key]: v } })); }, []);

  const showNet = !usingLine;
  const inGallery = stage.id === "a1-gallery";
  const showStack = stage.id === "a1-depth" || stage.id === "a1-fuse";
  const size = inGallery ? 270 : 520;
  const eq = `${line.w1.toFixed(2)}·x + ${line.w2.toFixed(2)}·y + ${line.b.toFixed(2)}`.replace(/\+ -/g, "− ");

  return (
    <div className="flex flex-col gap-4" data-testid="act1">
      <div className="flex items-start gap-10">
        <div>
          <Unlock id="plot"><BoundaryCanvas X={X} y={y} grid={showNet ? grid : null} G={G} line={usingLine ? line : null} size={size} /></Unlock>
          <Unlock id="gauge"><div className="mt-4" style={{ width: size }}><Gauge value={shownAcc} chance={0.5} label={usingLine ? "how well does this line split them?" : "accuracy"} broke={!!broke} /></div></Unlock>
          <div className="mt-2 flex h-8 items-center gap-3 text-lg text-slate-400">
            {running && <span>training… epoch {epoch}</span>}{broke && <Badge tone="red">BROKE: {broke === "nan" ? "NaN loss" : "accuracy stuck"}</Badge>}
            {slow && <Badge tone="amber">slow-mo (L)</Badge>}
          </div>
        </div>
        <div className="flex w-[640px] flex-col gap-5">
          {usingLine && (<>
            <Unlock id="guess"><button onClick={() => setLine(randomLine(rng.current))} className="w-fit rounded-xl bg-slate-800 px-5 py-3 text-2xl hover:bg-slate-700">Guess again <span className="text-base text-slate-500">(Space)</span></button></Unlock>
            <Unlock id="sliders">
              <div className="space-y-2 rounded-2xl bg-slate-900/60 p-4 text-xl" data-testid="sliders">
                {([["w1", "tilt", -3, 3], ["w2", "tilt", -3, 3], ["b", "slide", -2, 2]] as const).map(([k, label, lo, hi]) => (
                  <label key={k} className="flex items-center gap-3"><span className="w-20 text-slate-300">{label}</span>
                    <input type="range" min={lo} max={hi} step={0.01} value={line[k]} className="flex-1 accent-sky-400" onChange={(e) => setLine({ ...line, [k]: Number(e.target.value) })} />
                    <span className="w-16 font-mono text-slate-400">{line[k].toFixed(2)}</span></label>
                ))}
              </div>
            </Unlock>
            <Unlock id="equation"><div className="font-mono text-3xl text-sky-200" data-testid="equation">{eq} <span className="text-slate-500">&gt; 0 ?</span></div></Unlock>
            <Unlock id="findbest"><button onClick={() => { setFound(true); animateLine(leastSquaresLine(X, y)); }} className="w-fit rounded-xl bg-sky-500 px-5 py-3 text-2xl font-semibold text-slate-950 hover:bg-sky-400">Find best knobs <span className="text-base opacity-70">(Space)</span></button></Unlock>
            <Unlock id="word-parameters"><div className="text-3xl text-amber-300">These three numbers are the model's <b>parameters</b>.{found && <span className="ml-2 text-xl text-slate-400">best line: {(lineAcc * 100).toFixed(0)}%</span>}</div></Unlock>
          </>)}
          {!usingLine && (
            <Unlock id="depth">
              <div className="space-y-3 rounded-2xl bg-slate-900/60 p-4">
                <label className="flex items-center gap-3 text-xl"><span className="w-24 text-slate-300">depth</span>
                  <input type="range" min={1} max={8} value={depth} className="flex-1 accent-sky-400" onChange={(e) => { setDepth(Number(e.target.value)); setFused(false); }} data-testid="depth" /><span className="w-8 font-mono">{depth}</span></label>
                <Unlock id="width">
                  <label className="flex items-center gap-3 text-xl"><span className="w-24 text-slate-300">width</span>
                    <input type="range" min={2} max={64} value={width} className="flex-1 accent-orange-400" onChange={(e) => setWidth(Number(e.target.value))} data-testid="width" /><span className="w-8 font-mono">{width}</span></label>
                </Unlock>
                <div className="flex items-center gap-4"><button onClick={() => (running ? stopRun() : trainFor())} className="rounded-xl bg-emerald-500 px-5 py-3 text-2xl font-semibold text-slate-950 hover:bg-emerald-400" data-testid="train">{running ? "Stop" : "Train"} <span className="text-base opacity-70">(Space)</span></button>
                  {!showStack && !inGallery && <Unlock id="relu-plot"><ReluPlot /></Unlock>}</div>
                <p className="text-lg text-slate-400">{sIdx < stageIndex("a1-relu") ? "Layers: linear only (no bend)." : stage.id === "a1-gallery" ? `Click a card, then train: 3 layers × 16 units with ${getActivation(selected).label}.` : `Layers: linear + ReLU. ${depth} × ${width}.`}</p>
              </div>
            </Unlock>
          )}
          {showStack && <Unlock id="blocks"><div className="rounded-2xl bg-slate-900/60 p-4"><div className="mb-1 text-slate-400">the stack you are training</div><Blocks depth={depth} fused={fused && stage.id === "a1-fuse"} /></div></Unlock>}
          {stage.id === "a1-fuse" && <Unlock id="fuse"><p className="text-2xl text-slate-300">{fused ? "A line of a line of a line is still… one line." : "Press Space to fuse the blocks."}</p></Unlock>}
        </div>
      </div>
      <Unlock id="gallery">
        <div className="grid grid-cols-5 gap-2" data-testid="gallery">
          {ACTIVATIONS.map((a) => (
            <div key={a.name}><ActivationCard def={a} params={params[a.name]} selected={selected === a.name} onSelect={setSelected} result={results[a.name]} onParam={onParam} /></div>
          ))}
        </div>
      </Unlock>
    </div>
  );
}
