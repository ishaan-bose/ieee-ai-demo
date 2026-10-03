import { line as d3line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { Badge, gaugeColor } from "./ui";
import type { ActivationDef } from "../lib/activations";

export interface CardResult { acc: number; broke?: "nan" | "flat" | null }

/** One card of the activation gallery: live plots of f(x) (sky) and f'(x) (orange); a slider reshapes both for the families. */
export function ActivationCard({ def, params, onParam, selected, onSelect, result }: {
  def: ActivationDef; params: Record<string, number>; onParam: (v: number) => void; selected: boolean; onSelect: () => void; result?: CardResult;
}) {
  const W = 250, H = 54, xs = Array.from({ length: 81 }, (_, i) => -4 + i * 0.1);
  const fy = xs.map((x) => def.f(x, params)), dy = xs.map((x) => def.df(x, params));
  const all = [...fy, ...dy].filter(Number.isFinite);
  const lo = Math.min(-1, ...all), hi = Math.max(1, ...all);
  const sx = scaleLinear().domain([-4, 4]).range([6, W - 6]), sy = scaleLinear().domain([Math.max(lo, -4), Math.min(hi, 4)]).range([H - 6, 6]);
  const path = (ys: number[]) => d3line<number>().x((_, i) => sx(xs[i])).y((v) => sy(Math.max(-4, Math.min(4, v))))(ys) ?? "";
  const dead = def.dead || (def.name === "leaky_relu" && params.alpha >= 0.995);
  return (
    <div onClick={onSelect} data-testid={`card-${def.name}`} className={`cursor-pointer rounded-2xl p-2 ring-2 transition ${selected ? "bg-sky-500/10 ring-sky-400" : "bg-slate-900/70 ring-transparent hover:ring-slate-600"}`}>
      <div className="flex items-center justify-between"><span className="text-xl font-semibold text-slate-100">{def.label}</span>
        <span className="flex items-center gap-2">{result && !result.broke && <span className="font-mono text-base" style={{ color: gaugeColor(result.acc, 0.5) }}>{(result.acc * 100).toFixed(0)}%</span>}{result?.broke && <Badge tone="red">BROKE</Badge>}</span></div>
      <div className="text-sm text-slate-400">{def.formula}</div>
      <svg width={W} height={H} className="my-1 rounded-lg bg-slate-950">
        <line x1={sx(0)} x2={sx(0)} y1="0" y2={H} stroke="#1e293b" /><line x1="0" x2={W} y1={sy(0)} y2={sy(0)} stroke="#1e293b" />
        <path d={path(dy)} fill="none" stroke="#E69F00" strokeWidth="2" strokeDasharray="4 3" />
        <path d={path(fy)} fill="none" stroke="#56B4E9" strokeWidth="2.5" />
      </svg>
      {def.params && (
        <label className="flex items-center gap-2 text-sm text-slate-300" onClick={(e) => e.stopPropagation()}>
          {def.params.label}<input type="range" className="w-32 accent-sky-400" min={def.params.min} max={def.params.max} step={def.params.step} value={params[def.params.key]} onChange={(e) => onParam(Number(e.target.value))} />
          <span className="w-9 font-mono">{params[def.params.key]}</span>
        </label>
      )}
      {def.note && selected && <p className="mt-1 text-sm text-amber-300">{def.note}</p>}
      {dead && !result && <div className="text-sm text-red-300">slope is flat: nothing to learn</div>}
    </div>
  );
}
