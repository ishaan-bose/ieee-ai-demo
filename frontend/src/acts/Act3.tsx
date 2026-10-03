import { motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Badge, Gauge, Unlock } from "../components/ui";
import { LineChart } from "../components/charts";
import { loadOverfit, loadTrap, type OverfitData, type TrapData } from "../lib/act3data";
import { FMAX, FMIN, HOP, N_FRAMES, N_MELS, N_SAMPLES, NFFT, SR, WIN, decodeWav, framePower, logMel, melFilterbank, padOrTrim, rawInput } from "../lib/audio";
import { createMic, type MicHandle } from "../lib/mic";
import { forward, getModel, softmax, topK, type Model } from "../lib/inference";
import { useStage, useStageEvent } from "../state/StageProvider";

const WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];
interface SampleClip { file: string; label: number; word: string }
const FB = melFilterbank();

// ------------------------------------------------------------------ canvases
function viridis(t: number): string {
  const stops: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1), i = Math.floor(x), f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function Wave({ x, window: win, frame, w = 820, h = 130 }: { x: Float32Array; window?: boolean; frame?: number; w?: number; h?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!, dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = h * dpr;
    const ctx = c.getContext("2d")!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    if (win) { // the sliding 25 ms windows, hopping every 10 ms
      for (let t = 0; t < N_FRAMES; t += 1) {
        const x0 = ((t * HOP) / N_SAMPLES) * w, ww = (WIN / N_SAMPLES) * w;
        ctx.fillStyle = t === frame ? "rgba(251,191,36,0.55)" : t % 2 === 0 ? "rgba(56,189,248,0.08)" : "rgba(56,189,248,0.04)";
        ctx.fillRect(x0, t % 2 === 0 ? 0 : h / 2, ww, h / 2);
      }
    }
    ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 1.2; ctx.beginPath();
    const step = x.length / w;
    for (let px = 0; px < w; px++) {
      let mn = 1, mx = -1;
      for (let i = Math.floor(px * step); i < Math.min(x.length, Math.floor((px + 1) * step)); i++) { mn = Math.min(mn, x[i]); mx = Math.max(mx, x[i]); }
      ctx.moveTo(px, h / 2 - mx * h * 0.45); ctx.lineTo(px, h / 2 - mn * h * 0.45 + 0.5);
    }
    ctx.stroke();
  }, [x, win, frame, w, h]);
  return <canvas ref={ref} style={{ width: w, height: h }} className="rounded-xl bg-slate-900" />;
}

function Bars({ values, w = 820, h = 150, color = "#38bdf8", label }: { values: ArrayLike<number>; w?: number; h?: number; color?: string; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!, dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = h * dpr;
    const ctx = c.getContext("2d")!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    let mx = -Infinity, mn = Infinity;
    for (let i = 0; i < values.length; i++) { mx = Math.max(mx, values[i]); mn = Math.min(mn, values[i]); }
    const lo = Math.min(0, mn), bw = w / values.length;
    ctx.fillStyle = color;
    for (let i = 0; i < values.length; i++) { const v = (values[i] - lo) / (mx - lo || 1); ctx.fillRect(i * bw + 0.5, h - 16 - v * (h - 24), Math.max(1, bw - 1), v * (h - 24)); }
    ctx.fillStyle = "#94a3b8"; ctx.font = "13px system-ui"; ctx.fillText(label ?? "", 6, 14);
  }, [values, w, h, color, label]);
  return <canvas ref={ref} style={{ width: w, height: h }} className="rounded-xl bg-slate-900" />;
}

function Spectrogram({ lm, w = 820, h = 200, playhead }: { lm: Float32Array; w?: number; h?: number; playhead?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!;
    c.width = N_FRAMES; c.height = N_MELS;
    const ctx = c.getContext("2d")!, img = ctx.createImageData(N_FRAMES, N_MELS);
    for (let m = 0; m < N_MELS; m++) for (let t = 0; t < N_FRAMES; t++) {
      const v = (lm[m * N_FRAMES + t] + 1.5) / 5; // normalised log-mel is ~N(0,1)
      const col = viridis(v).match(/\d+/g)!.map(Number), o = 4 * ((N_MELS - 1 - m) * N_FRAMES + t); // low frequencies at the bottom
      img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [lm]);
  return (
    <div className="relative" style={{ width: w, height: h }}>
      <canvas ref={ref} style={{ width: w, height: h, imageRendering: "pixelated" }} className="rounded-xl" data-testid="spectrogram" />
      {playhead !== undefined && <div className="absolute top-0 h-full w-0.5 bg-amber-300" style={{ left: `${(playhead / N_FRAMES) * 100}%` }} />}
    </div>
  );
}

// ------------------------------------------------------------------ the two models listening
function ModelCard({ title, blurb, model, input, shape, truth, tally }: { title: string; blurb: string; model: Model | null; input: Float32Array | null; shape?: number[]; truth: number | null; tally: { ok: number; n: number } }) {
  const probs = useMemo(() => (model && input ? softmax(forward(model, input, shape)) : null), [model, input, shape]);
  const top = probs ? topK(probs, 3) : [];
  const ok = truth !== null && top[0]?.index === truth;
  return (
    <div className="w-[340px] rounded-2xl bg-slate-900/70 p-4" data-testid={`model-${title.split(" ")[0].toLowerCase()}`}>
      <div className="flex items-center justify-between"><div className="text-2xl font-semibold text-slate-100">{title}</div>{model?.isMock && <Badge tone="amber" title="trained weights not bundled yet">demo weights</Badge>}</div>
      <div className="text-base text-slate-400">{blurb}</div>
      <div className="mt-3 text-5xl font-bold" style={{ color: truth === null ? "#e2e8f0" : ok ? "#4ade80" : "#f87171" }}>{top[0] ? WORDS[top[0].index] : "…"}{truth !== null && top[0] ? (ok ? " ✓" : " ✗") : ""}</div>
      <div className="mt-2 space-y-1">{top.map((t) => (<div key={t.index} className="flex items-center gap-2 text-lg text-slate-300"><span className="w-16">{WORDS[t.index]}</span><div className="h-3 flex-1 rounded-full bg-slate-800"><div className="h-full rounded-full bg-sky-400" style={{ width: `${t.p * 100}%` }} /></div><span className="w-10 text-right font-mono text-sm text-slate-400">{(t.p * 100).toFixed(0)}%</span></div>))}</div>
      <div className="mt-3"><Gauge value={tally.n ? tally.ok / tally.n : 0} chance={0.1} small label={`on the clips so far (${tally.ok}/${tally.n})`} /></div>
    </div>
  );
}

// ------------------------------------------------------------------ 3.1 preprocessing
const STEPS = ["waveform", "windows", "Fourier transform", "mel scale", "log"] as const;

function Preprocess() {
  const [clips, setClips] = useState<SampleClip[]>([]);
  const [idx, setIdx] = useState(0);
  const [wave, setWave] = useState<Float32Array>(new Float32Array(N_SAMPLES));
  const [source, setSource] = useState<"sample" | "mic">("sample");
  const [micError, setMicError] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [step, setStep] = useState(0);
  const [frame, setFrame] = useState(40);
  const [models, setModels] = useState<{ raw: Model | null; mel: Model | null }>({ raw: null, mel: null });
  const [tally, setTally] = useState({ raw: { ok: 0, n: 0 }, mel: { ok: 0, n: 0 } });
  const mic = useRef<MicHandle | null>(null);
  const { resetNonce } = useStage();

  useEffect(() => { Promise.all([getModel("raw_audio"), getModel("logmel_audio")]).then(([raw, mel]) => setModels({ raw, mel })); }, []);
  useEffect(() => {
    fetch("/data/speech/samples.json").then((r) => r.json()).then((s: SampleClip[]) => setClips(s)).catch(() => undefined);
    return () => mic.current?.close();
  }, []);
  const loadClip = useCallback(async (i: number, list = clips) => {
    if (!list.length) return;
    const c = list[(i + list.length) % list.length];
    const buf = await (await fetch(`/data/speech/samples/${c.file}`)).arrayBuffer();
    setWave(Float32Array.from(padOrTrim(decodeWav(buf)))); setSource("sample"); setIdx((i + list.length) % list.length);
  }, [clips]);
  useEffect(() => { if (clips.length) loadClip(0, clips); }, [clips, loadClip, resetNonce]);

  const truth = source === "sample" && clips[idx] ? clips[idx].label : null;
  const lm = useMemo(() => logMel(wave), [wave]);
  const raw = useMemo(() => rawInput(wave), [wave]);
  const w64 = useMemo(() => Float64Array.from(wave), [wave]);
  // tally accuracy over sample clips as they are tried (once per clip)
  const tried = useRef(new Set<string>());
  useEffect(() => {
    if (truth === null || !models.raw || !models.mel || tried.current.has(clips[idx].file)) return;
    tried.current.add(clips[idx].file);
    const a = topK(softmax(forward(models.raw, raw)), 1)[0].index === truth, b = topK(softmax(forward(models.mel, lm, [1, N_MELS, N_FRAMES])), 1)[0].index === truth;
    setTally((t) => ({ raw: { ok: t.raw.ok + (a ? 1 : 0), n: t.raw.n + 1 }, mel: { ok: t.mel.ok + (b ? 1 : 0), n: t.mel.n + 1 } }));
  }, [truth, models, raw, lm, idx, clips]);

  const energyFrame = useMemo(() => { let best = 0, be = -1; for (let t = 0; t < N_FRAMES; t++) { let e = 0; for (let i = 0; i < WIN; i += 2) e += w64[t * HOP + i] ** 2; if (e > be) { be = e; best = t; } } return best; }, [w64]);
  useEffect(() => setFrame(energyFrame), [energyFrame]);
  const power = useMemo(() => framePower(w64, frame), [w64, frame]);
  const mel = useMemo(() => FB.map((row) => { let e = 0; for (let k = 0; k < power.length; k++) e += row[k] * power[k]; return e; }), [power]);
  const logmel = useMemo(() => mel.map((e) => Math.log(e + 1e-6)), [mel]);

  const startRec = async () => {
    try { mic.current ??= createMic(); await mic.current.start(); setRecording(true); setMicError(null); }
    catch (e) { setMicError(e instanceof Error ? e.message : "microphone unavailable"); setRecording(false); if (clips.length) loadClip(idx); }
  };
  const stopRec = () => {
    if (!recording) return;
    setRecording(false);
    const x = mic.current?.stop();
    if (x) { setWave(x); setSource("mic"); } else setMicError("nothing was recorded");
  };
  useEffect(() => {
    const dn = (e: KeyboardEvent) => { if ((e.key === "t" || e.key === "T") && !e.repeat) startRec(); if (e.key === ",") loadClip(idx - 1); if (e.key === ".") loadClip(idx + 1); };
    const up = (e: KeyboardEvent) => { if (e.key === "t" || e.key === "T") stopRec(); };
    window.addEventListener("keydown", dn); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", dn); window.removeEventListener("keyup", up); };
  }); // eslint-disable-line react-hooks/exhaustive-deps
  useStageEvent("run", () => setStep((s) => (s + 1) % STEPS.length));

  const play = async () => { const ctx = new AudioContext({ sampleRate: SR }); const b = ctx.createBuffer(1, N_SAMPLES, SR); b.copyToChannel(wave as unknown as Float32Array<ArrayBuffer>, 0); const s = ctx.createBufferSource(); s.buffer = b; s.connect(ctx.destination); s.start(); };

  return (
    <div className="flex items-start gap-8" data-testid="act3-pre">
      <div className="flex w-[840px] flex-col gap-3">
        <Unlock id="mic">
          <div className="flex items-center gap-3 text-xl">
            <button onPointerDown={startRec} onPointerUp={stopRec} onPointerLeave={stopRec} className={`rounded-xl px-5 py-3 text-2xl font-semibold ${recording ? "bg-red-500 text-white" : "bg-sky-500 text-slate-950 hover:bg-sky-400"}`} data-testid="ptt">{recording ? "● recording… release" : "Hold to speak (T)"}</button>
            <button onClick={() => loadClip(idx - 1)} className="rounded-lg bg-slate-800 px-3 py-2 hover:bg-slate-700">◀</button>
            <span className="min-w-[210px] text-slate-300" data-testid="clip-name">{source === "mic" ? "your recording" : clips[idx] ? `clip ${idx + 1}/${clips.length}: “${clips[idx].word}”` : "loading clips…"}</span>
            <button onClick={() => loadClip(idx + 1)} className="rounded-lg bg-slate-800 px-3 py-2 hover:bg-slate-700">▶</button>
            <button onClick={play} className="rounded-lg bg-slate-800 px-3 py-2 hover:bg-slate-700">🔊</button>
          </div>
          {micError && <div className="mt-1 text-lg text-amber-300" data-testid="mic-error">Microphone not available ({micError}). Using the pre-recorded clips instead. (, and . switch clips)</div>}
        </Unlock>
        <Unlock id="spectrogram">
          <div className="space-y-2">
            <div className="flex gap-2">{STEPS.map((s, i) => <button key={s} onClick={() => setStep(i)} className={`rounded-lg px-3 py-1.5 text-lg ${i === step ? "bg-sky-500 text-slate-950" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`}>{i + 1}. {s}</button>)}<span className="self-center text-sm text-slate-500">Space: next step</span></div>
            <motion.div key={step} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2" data-testid={`step-${step}`}>
              {step === 0 && <><Wave x={wave} /><p className="text-lg text-slate-400">Sound is a wiggle: 16,000 numbers per second.</p></>}
              {step === 1 && <><Wave x={wave} window frame={frame} /><p className="text-lg text-slate-400">Cut into overlapping 25 ms windows, one every 10 ms ({N_FRAMES} of them).</p></>}
              {step === 2 && <><Wave x={wave} window frame={frame} h={90} /><Bars values={power.slice(0, 129)} label={`window ${frame}: how much of each frequency (0 – ${SR / 2} Hz, up to 4 kHz shown)`} /><p className="text-lg text-slate-400">The Fourier transform asks, window by window: how much of each pitch is in here?</p></>}
              {step === 3 && <><Bars values={mel} color="#E69F00" label={`window ${frame}: ${N_MELS} mel bands (${FMIN}–${FMAX} Hz, spaced like our ears)`} /><p className="text-lg text-slate-400">Ears hear low pitches in more detail: squeeze the frequencies onto the mel scale.</p></>}
              {step === 4 && <><Bars values={logmel} color="#CC79A7" h={110} label="log: loudness compressed, like our ears" /><Spectrogram lm={lm} h={150} playhead={frame} /><p className="text-lg text-slate-400">Stack every window: a picture of the sound (time →, pitch ↑). This is what the good model sees.</p></>}
            </motion.div>
            {step !== 4 && <Spectrogram lm={lm} h={step === 3 ? 90 : 110} />}
            <input type="range" min={0} max={N_FRAMES - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} className="w-full accent-amber-400" title="pick a window" />
          </div>
        </Unlock>
      </div>
      <Unlock id="models">
        <div className="flex flex-col gap-4">
          <ModelCard title="Raw-wave AI" blurb="gets the plain waveform (4,000 numbers)" model={models.raw} input={raw} truth={truth} tally={tally.raw} />
          <ModelCard title="Spectrogram AI" blurb="gets the log-mel picture (40 × 98)" model={models.mel} input={lm} shape={[1, N_MELS, N_FRAMES]} truth={truth} tally={tally.mel} />
        </div>
      </Unlock>
    </div>
  );
}

// ------------------------------------------------------------------ 3.2 metrics trap
function Matrix({ m, dim }: { m: number[][]; dim?: boolean }) {
  const cells = [["true negatives", m[0][0], "#1d4e89"], ["false alarms", m[0][1], "#7c2d12"], ["MISSED marvins", m[1][0], "#991b1b"], ["found marvins", m[1][1], "#166534"]] as const;
  return (
    <div className={`grid grid-cols-2 gap-2 ${dim ? "opacity-40" : ""}`} data-testid="confusion">
      {cells.map(([label, v, color]) => (<div key={label} className="w-[190px] rounded-xl p-3 text-center" style={{ backgroundColor: color }}><div className="font-mono text-4xl text-white">{v}</div><div className="text-base text-slate-200">{label}</div></div>))}
    </div>
  );
}
const pct = (v: number | null) => (v === null ? "n/a" : `${(v * 100).toFixed(0)}%`);

function Trap() {
  const [data, setData] = useState<TrapData | null>(null);
  const [build, setBuild] = useState(0);
  useEffect(() => { loadTrap().then(setData); }, []);
  useStageEvent("run", () => setBuild((b) => Math.min(3, b + 1)));
  useStageEvent("reset", () => setBuild(0));
  if (!data) return null;
  const n = data.naive.test, a = data.always_no, w = data.weighted.test;
  return (
    <div className="flex items-start gap-10" data-testid="act3-trap">
      <Unlock id="trap">
        <div className="w-[560px] space-y-5">
          <p className="text-2xl text-slate-300">A detector for the word <b className="text-amber-300">“marvin”</b>, which is only 2% of the clips.</p>
          <div className="space-y-3 rounded-2xl bg-slate-900/70 p-5"><Gauge value={a.accuracy} chance={0.5} label="“Always say NO”: accuracy" /><Gauge value={n.accuracy} chance={0.5} label="our trained detector: accuracy" /></div>
          {build === 0 && <p className="text-2xl text-sky-300">Same score! Is the detector any good? <span className="text-slate-500">(Space)</span></p>}
          {build >= 1 && <div className="text-2xl text-slate-200">“Always say NO” never finds a single marvin: recall {pct(a.recall)}. <span className="text-slate-500">Accuracy was a trap.</span></div>}
          {data.isMock && <Badge tone="amber">demo data</Badge>}
        </div>
      </Unlock>
      {build >= 1 && (
        <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
          <div className="text-xl text-slate-400">Our detector on unseen clips</div>
          <Matrix m={n.confusion} />
          {build >= 2 && <div className="space-y-1 text-2xl" data-testid="recall"><div>accuracy <b className="text-slate-100">{pct(n.accuracy)}</b></div><div>recall (marvins found) <b className="text-red-300">{pct(n.recall)}</b></div><div>precision <b className="text-slate-100">{pct(n.precision)}</b></div></div>}
          {build >= 3 && (<div className="rounded-2xl bg-slate-900/70 p-4 text-xl text-slate-200" data-testid="weighted"><b className="text-emerald-300">Fix: tell the AI that a missed marvin costs 49× more.</b><div className="mt-1">accuracy {pct(w.accuracy)} · recall <b className="text-emerald-300">{pct(w.recall)}</b> · precision {pct(w.precision)}</div><div className="text-base text-slate-400">The honest number depends on what you care about.</div></div>)}
        </motion.div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 3.3 overfitting
function Overfit() {
  const [data, setData] = useState<OverfitData | null>(null);
  const [ri, setRi] = useState(1);
  const [shown, setShown] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => { loadOverfit().then((d) => { setData(d); setRi(Math.max(0, d.runs.findIndex((r) => r.n === d.default_n))); setShown(9999); }); }, []);
  const replay = useCallback(() => {
    if (!data) return;
    if (timer.current) clearInterval(timer.current);
    const E = data.runs[ri].epochs.length;
    setShown(0);
    let e = 0;
    timer.current = setInterval(() => { e++; setShown(e); if (e >= E && timer.current) clearInterval(timer.current); }, 6000 / E);
  }, [data, ri]);
  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);
  useStageEvent("run", replay);
  useStageEvent("reset", replay);
  if (!data) return null;
  const run = data.runs[ri], eps = run.epochs.slice(0, shown);
  const last = eps[eps.length - 1];
  const sweep = data.runs.map((r, i) => ({ x: i, n: r.n, tr: r.epochs[r.epochs.length - 1].train_acc, va: r.epochs[r.epochs.length - 1].val_acc }));
  return (
    <div className="flex items-start gap-10" data-testid="act3-overfit">
      <Unlock id="overfit">
        <div className="w-[760px] space-y-4">
          <p className="text-2xl text-slate-300">A big model, <b className="text-sky-300">{run.n.toLocaleString()}</b> training clips.</p>
          <LineChart width={760} height={300} xMax={run.epochs.length} yMin={0} yMax={1} xLabel="epoch" title="accuracy: training (blue) vs. new data (orange)"
            series={[{ color: "#56B4E9", points: eps.map((e) => ({ x: e.epoch, y: e.train_acc })) }, { color: "#E69F00", points: eps.map((e) => ({ x: e.epoch, y: e.val_acc })) }]} />
          <label className="flex items-center gap-4 text-2xl text-slate-300">training-set size
            <input type="range" min={0} max={data.runs.length - 1} value={ri} className="w-72 accent-sky-400" data-testid="size-slider" onChange={(e) => { setRi(Number(e.target.value)); setShown(9999); }} />
            <span className="font-mono text-sky-300">{run.n.toLocaleString()}</span>
            <button onClick={replay} className="rounded-lg bg-slate-800 px-3 py-1 text-lg hover:bg-slate-700">replay (Space)</button></label>
          {last && <p className="text-2xl text-slate-200">train {(last.train_acc * 100).toFixed(0)}% · new data <b className="text-amber-300">{(last.val_acc * 100).toFixed(0)}%</b> · gap <b className="text-red-300">{((last.train_acc - last.val_acc) * 100).toFixed(0)} points</b></p>}
          {data.isMock && <Badge tone="amber">demo data</Badge>}
        </div>
        <div className="w-[480px] space-y-3"><LineChart width={480} height={260} xMax={data.runs.length - 1} yMin={0} yMax={1} title="final accuracy vs. training-set size" xLabel="more data →"
          series={[{ color: "#56B4E9", points: sweep.map((s) => ({ x: s.x, y: s.tr })) }, { color: "#E69F00", points: sweep.map((s) => ({ x: s.x, y: s.va })) }]} />
          <div className="flex justify-between text-sm text-slate-500">{data.runs.map((r) => <span key={r.n}>{r.n >= 1000 ? `${r.n / 1000}k` : r.n}</span>)}</div>
          <p className="text-xl text-slate-400">More data closes the gap between memorising and understanding.</p></div>
      </Unlock>
    </div>
  );
}

export function Act3() {
  const { stage } = useStage();
  if (stage.id === "a3-preprocess") return <Preprocess />;
  if (stage.id === "a3-trap") return <Trap />;
  return <Overfit />;
}
void NFFT;
