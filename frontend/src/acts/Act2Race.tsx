import { useEffect, useMemo, useState } from "react";
import { Badge, Gauge, LANE_COLORS, Unlock } from "../components/ui";
import { LineChart } from "../components/charts";
import { NetworkDiagram } from "../components/NetworkDiagram";
import { ProbeWall } from "../components/ProbeWall";
import { Valley2D, useValley, verdict } from "../components/Valley";
import { Valley3D } from "../components/Valley3D";
import { diagramFrom, type DiagramData } from "../lib/diagram";
import { loadDoodles } from "../lib/doodle";
import { MOCK } from "../lib/env";
import { forwardTrace, getModel, softmax, type Model } from "../lib/inference";
import { rasterize, toModelInput } from "../lib/rasterizer";
import { BASE, BATCH_RACE_LR, LOSS_CHOICES, laneCfg, type BatchSize, type LaneCfg, type RaceKind } from "../lib/raceGrid";
import { useRace, type LaneState } from "../lib/useRace";
import { useHealth } from "../state/health";
import { useStage, useStageEvent } from "../state/StageProvider";

const MAX_SECONDS = 25;
const fmt = (n: number) => n.toLocaleString("en-US");

interface LossPick { loss: LaneCfg["loss"]; params: { delta: number; gamma: number; eps: number } }

function LaneRow({ lane, color, name, kind }: { lane: LaneState; color: string; name: string; kind: RaceKind }) {
  const l = lane.latest;
  const loss = lane.ticks.map((t) => ({ x: t.t, y: t.loss ?? NaN }));
  return (
    <div className="flex items-center gap-4 rounded-2xl bg-slate-900/70 p-2.5" data-testid={`lane-${lane.cfg.id}`}>
      <div className="w-[300px]">
        <div className="flex items-center gap-2 text-2xl font-semibold" style={{ color }}><span className="inline-block h-4 w-4 rounded-full" style={{ backgroundColor: color }} />{name}</div>
        <Gauge value={l?.acc ?? 0.1} chance={0.1} label="accuracy" small />
        {kind === "batch" && (
          <div className="mt-1 flex gap-5 font-mono text-lg text-slate-200" data-testid={`counters-${lane.cfg.id}`}>
            <span>{fmt(l?.updates ?? 0)} <span className="font-sans text-sm text-slate-500">updates</span></span>
            <span>{fmt(l?.samples_seen ?? 0)} <span className="font-sans text-sm text-slate-500">doodles</span></span>
          </div>
        )}
      </div>
      <LineChart series={[{ color, points: loss }]} width={330} height={96} xMax={MAX_SECONDS} title="loss (own scale)" />
    </div>
  );
}

/** Acts 2.2 / 2.3 / 2.4: loss race, learning-rate race, batch-size race. Live from the GPU box, or recorded streams (S / F / offline). */
export function Act2Race({ kind }: { kind: RaceKind }) {
  const { online } = useHealth();
  const { forceFallback } = useStage();
  const { state, start, skip, abort, reset } = useRace({ forceFallback, online });
  const [hideBadge, setHideBadge] = useState(false);

  const [lossPicks, setLossPicks] = useState<LossPick[]>([
    { loss: "ce", params: { delta: 1, gamma: 2, eps: 0.1 } }, { loss: "mse", params: { delta: 1, gamma: 2, eps: 0.1 } }, { loss: "focal", params: { delta: 1, gamma: 2, eps: 0.1 } }]);
  const [lrPick, setLrPick] = useState(0.3);
  const [batchPick, setBatchPick] = useState(16);

  const cfgs: { cfg: LaneCfg; name: string }[] = useMemo(() => {
    if (kind === "loss") return lossPicks.map((p) => {
      const ch = LOSS_CHOICES.find((c) => c.loss === p.loss)!;
      return { cfg: laneCfg(p.loss, { loss: p.loss, ...p.params }), name: ch.label + (ch.param ? ` ${ch.param.label}=${p.params[ch.param.key]}` : "") };
    });
    if (kind === "lr") return [{ cfg: laneCfg("small", { lr: 0.001 }), name: "too small (0.001)" }, { cfg: laneCfg("good", { lr: 0.03 }), name: "good (0.03)" }, { cfg: laneCfg("pick", { lr: lrPick }), name: `your pick (${lrPick})` }];
    return [{ cfg: laneCfg("one", { batch_size: 1, lr: BATCH_RACE_LR }), name: "batch 1" }, { cfg: laneCfg("full", { batch_size: "full", lr: BATCH_RACE_LR }), name: "full batch" },
      { cfg: laneCfg("pick", { batch_size: batchPick as BatchSize, lr: BATCH_RACE_LR }), name: `batch ${batchPick}` }];
  }, [kind, lossPicks, lrPick, batchPick]);

  const go = () => (state.status === "running" ? abort() : start(kind, cfgs.map((c) => c.cfg), MAX_SECONDS));
  useStageEvent("run", go);
  useStageEvent("skip", async () => { if (state.status === "idle") { start(kind, cfgs.map((c) => c.cfg), MAX_SECONDS); } skip(); });
  useStageEvent("reset", reset);
  useEffect(() => () => reset(), [kind]); // eslint-disable-line react-hooks/exhaustive-deps

  const lanes = state.lanes.length ? state.lanes : cfgs.map((c) => ({ cfg: c.cfg, ticks: [], latest: null }));
  const accSeries = lanes.map((l, i) => ({ color: LANE_COLORS[i], points: l.ticks.map((t) => ({ x: t.t, y: t.acc })) }));
  const panelId = kind === "loss" ? "lanes-loss" : kind === "lr" ? "lanes-lr" : "lanes-batch";
  const toggleLoss = (loss: LaneCfg["loss"]) => setLossPicks((p) => p.some((x) => x.loss === loss) ? p.filter((x) => x.loss !== loss) : p.length >= 3 ? p : [...p, { loss, params: { delta: 1, gamma: 2, eps: 0.1 } }]);

  return (
    <div className="flex items-start gap-8" data-testid={`race-${kind}`}>
      <Unlock id="probewall"><div><ProbeWall lanes={lanes} labels={state.probeLabels.length ? state.probeLabels : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]} />
        <p className="mt-1 w-[470px] text-sm text-slate-500">16 doodles the AI has never seen. One dot per lane: red = wrong, green = right.</p>
        <div className="mt-2"><LineChart series={accSeries} width={470} height={150} xMax={MAX_SECONDS} yMin={0} yMax={1} title="accuracy vs wall-clock seconds (same for all lanes)" /></div></div></Unlock>
      <Unlock id={panelId}>
        <div className="flex w-[760px] flex-col gap-2.5">
          {kind === "loss" && (
            <div className="grid grid-cols-3 gap-2" data-testid="loss-picker">
              {LOSS_CHOICES.map((c) => {
                const pick = lossPicks.find((p) => p.loss === c.loss);
                return (
                  <div key={c.loss} onClick={() => toggleLoss(c.loss)} className={`cursor-pointer rounded-xl p-2 ring-2 ${pick ? "bg-sky-500/10 ring-sky-400" : "bg-slate-900/70 ring-transparent hover:ring-slate-600"}`}>
                    <div className="text-xl font-semibold text-slate-100">{c.label}</div><div className="text-sm leading-tight text-slate-400">{c.blurb}</div>
                    {c.param && pick && (
                      <label className="mt-1 flex items-center gap-1 text-sm text-slate-300" onClick={(e) => e.stopPropagation()}>{c.param.label}
                        <input type="range" className="w-24 accent-sky-400" min={c.param.min} max={c.param.max} step={c.param.step} value={pick.params[c.param.key]}
                          onChange={(e) => setLossPicks((ps) => ps.map((p) => (p.loss === c.loss ? { ...p, params: { ...p.params, [c.param!.key]: Number(e.target.value) } } : p)))} />
                        <span className="font-mono">{pick.params[c.param.key]}</span></label>)}
                  </div>
                );
              })}
            </div>
          )}
          {kind === "lr" && (
            <label className="flex items-center gap-3 text-xl text-slate-300">your pick (3rd lane): step size
              <input type="range" min={-3.5} max={0.3} step={0.05} value={Math.log10(lrPick)} className="w-64 accent-pink-400" onChange={(e) => setLrPick(+Math.pow(10, Number(e.target.value)).toPrecision(2))} /><span className="font-mono">{lrPick}</span></label>)}
          {kind === "batch" && (
            <label className="flex items-center gap-3 text-xl text-slate-300">your pick (3rd lane): batch size
              <input type="range" min={1} max={9} step={1} value={Math.log2(batchPick)} className="w-64 accent-pink-400" onChange={(e) => setBatchPick(2 ** Number(e.target.value))} /><span className="font-mono">{batchPick}</span></label>)}
          <div className="flex items-center gap-3">
            <button onClick={go} className="rounded-xl bg-emerald-500 px-5 py-2 text-2xl font-semibold text-slate-950 hover:bg-emerald-400" data-testid="start-race">{state.status === "running" ? "Stop" : "Start the race"} <span className="text-base opacity-70">(Space)</span></button>
            {state.status !== "idle" && <span className="font-mono text-xl text-slate-400">{state.t.toFixed(1)}s / {MAX_SECONDS}s</span>}
            {state.source === "live" && <Badge tone="green" title="training on the GPU box right now">{MOCK ? "mock server" : "live"}</Badge>}
            {(state.source === "recorded" || state.source === "demo") && !hideBadge && <span onClick={() => setHideBadge(true)} className="cursor-pointer" title="click to hide"><Badge tone="amber">{state.source === "demo" ? "demo data" : "recorded"}</Badge></span>}
            {state.note && <span className="text-base text-slate-500">{state.note}</span>}
          </div>
          {lanes.map((l, i) => <LaneRow key={l.cfg.id} lane={l} color={LANE_COLORS[i]} name={cfgs[i]?.name ?? l.cfg.id} kind={kind} />)}
        </div>
      </Unlock>
    </div>
  );
}

/** Act 2.3 first half: the backward pass (hero visual 3) next to the rolling-ball valley cartoon. */
export function Act2Backward() {
  const [model, setModel] = useState<Model | null>(null);
  const [diagram, setDiagram] = useState<DiagramData | null>(null);
  const [pulse, setPulse] = useState(0);
  const [stepSize, setStepSize] = useState(0.2);
  const [webgl, setWebgl] = useState(true);
  const valley = useValley(stepSize);
  useEffect(() => { getModel("doodle").then(setModel); }, []);
  useEffect(() => {
    if (!model) return;
    loadDoodles("probe").then((p) => {
      const x = toModelInput(rasterize(p[0].d)), tr = forwardTrace(model, x);
      setDiagram(diagramFrom(model, x, tr, softmax(tr.out)));
    }).catch(() => undefined);
  }, [model]);
  useStageEvent("run", () => { setPulse((n) => n + 1); if (valley.running) valley.setRunning(false); else { valley.reset(); setTimeout(() => valley.setRunning(true), 50); } });
  useStageEvent("reset", valley.reset);
  return (
    <div className="flex items-start gap-8" data-testid="act2-backward">
      <Unlock id="backward"><div><NetworkDiagram data={diagram} pulse={pulse} mode="backward" width={640} height={330} />
        <p className="mt-2 w-[640px] text-lg text-slate-400">The backward pass: the error flows back and tells every knob which way is downhill.</p></div></Unlock>
      <Unlock id="valley">
        <div className="w-[560px]">
          {webgl ? <Valley3D w={valley.w} trail={valley.trail} onError={() => setWebgl(false)} /> : <Valley2D w={valley.w} trail={valley.trail} size={540} />}
          <div className="mt-2 text-base text-amber-300">A cartoon: the real landscape has millions of directions, not one.</div>
          <label className="mt-2 flex items-center gap-3 text-xl text-slate-300">step size
            <input type="range" min={0.01} max={0.7} step={0.01} value={stepSize} className="w-64 accent-amber-400" onChange={(e) => { setStepSize(Number(e.target.value)); valley.reset(); }} />
            <span className="font-mono">{stepSize.toFixed(2)}</span><span className="text-2xl text-sky-300">{verdict(stepSize)}</span></label>
          <button onClick={() => { valley.reset(); setTimeout(() => valley.setRunning(true), 50); setPulse((n) => n + 1); }} className="mt-2 rounded-xl bg-amber-500 px-5 py-2 text-2xl font-semibold text-slate-950 hover:bg-amber-400">Roll (Space)</button>
        </div>
      </Unlock>
    </div>
  );
}

void BASE;
