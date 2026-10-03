import { line as d3line } from "d3-shape";
import { scaleLinear, scaleLog } from "d3-scale";
import { useMemo } from "react";

export interface Series { color: string; points: { x: number; y: number }[]; label?: string; dashed?: boolean }

/** Small line chart (SVG). Used for loss curves (one chart per lane: different losses are not comparable), training curves, etc. */
export function LineChart({ series, width = 320, height = 150, xMax, yMin, yMax, logY, title, xLabel, yLabel }: {
  series: Series[]; width?: number; height?: number; xMax?: number; yMin?: number; yMax?: number; logY?: boolean; title?: string; xLabel?: string; yLabel?: string;
}) {
  const m = { l: 40, r: 8, t: title ? 22 : 8, b: 24 };
  const { x, y, paths } = useMemo(() => {
    const all = series.flatMap((s) => s.points.filter((p) => Number.isFinite(p.y) && (!logY || p.y > 0)));
    const xs = xMax ?? Math.max(1, ...all.map((p) => p.x));
    const lo = yMin ?? (all.length ? Math.min(...all.map((p) => p.y)) : 0);
    const hi = yMax ?? (all.length ? Math.max(...all.map((p) => p.y)) : 1);
    const x = scaleLinear().domain([0, xs]).range([m.l, width - m.r]);
    const y = logY ? scaleLog().domain([Math.max(1e-6, lo), Math.max(hi, lo * 1.01 + 1e-6)]).range([height - m.b, m.t]) : scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).nice().range([height - m.b, m.t]);
    const gen = d3line<{ x: number; y: number }>().x((p) => x(p.x)).y((p) => y(p.y)).defined((p) => Number.isFinite(p.y) && (!logY || p.y > 0));
    return { x, y, paths: series.map((s) => gen(s.points) ?? "") };
  }, [series, width, height, xMax, yMin, yMax, logY, m.l, m.r, m.t, m.b]);
  const yt = y.ticks(3);
  return (
    <svg width={width} height={height} className="rounded-lg bg-slate-900/70">
      {title && <text x={m.l} y={14} className="fill-slate-300" fontSize="13">{title}</text>}
      {yt.map((v) => (<g key={v}><line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="#1e293b" /><text x={m.l - 5} y={y(v) + 4} textAnchor="end" fontSize="11" className="fill-slate-500">{Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : +v.toPrecision(2)}</text></g>))}
      {x.ticks(4).map((v) => (<text key={v} x={x(v)} y={height - 8} textAnchor="middle" fontSize="11" className="fill-slate-500">{v}</text>))}
      {xLabel && <text x={width - m.r} y={height - 1} textAnchor="end" fontSize="10" className="fill-slate-600">{xLabel}</text>}
      {yLabel && <text x={4} y={m.t - 2} fontSize="10" className="fill-slate-600">{yLabel}</text>}
      {paths.map((d, i) => (<path key={i} d={d} fill="none" stroke={series[i].color} strokeWidth={2.2} strokeDasharray={series[i].dashed ? "5 4" : undefined} />))}
    </svg>
  );
}
