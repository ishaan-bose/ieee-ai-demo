import { motion } from "framer-motion";
import QRCode from "qrcode";
import { useEffect, useRef, useState } from "react";
import { Unlock } from "../components/ui";
import { useSession } from "../lib/session";
import { useStage, useStageEvent } from "../state/StageProvider";

// ------------------------------------------------------------------ tech tree: every revealed control + the extras unlock together
interface TNode { id: string; label: string; group: "seen" | "extra"; blurb: string }
const NODES: TNode[] = [
  { id: "params", label: "parameters", group: "seen", blurb: "the numbers that get learned" },
  { id: "depth", label: "depth", group: "seen", blurb: "how many layers" },
  { id: "width", label: "width", group: "seen", blurb: "how many units per layer" },
  { id: "act", label: "activation", group: "seen", blurb: "the bend between layers" },
  { id: "loss", label: "loss", group: "seen", blurb: "how a mistake is measured" },
  { id: "lr", label: "step size", group: "seen", blurb: "how big each learning step is" },
  { id: "batch", label: "batch size", group: "seen", blurb: "how many examples per step" },
  { id: "opt", label: "optimizer", group: "extra", blurb: "plain steps, momentum or Adam" },
  { id: "init", label: "initialisation", group: "extra", blurb: "where the knobs start" },
  { id: "norm", label: "normalisation", group: "extra", blurb: "keep the numbers well-behaved" },
  { id: "res", label: "residual links", group: "extra", blurb: "shortcuts around layers" },
  { id: "sched", label: "LR schedule", group: "extra", blurb: "start fast, finish gently" },
  { id: "ema", label: "EMA", group: "extra", blurb: "average the weights over time" },
  { id: "aug", label: "symmetry trick", group: "extra", blurb: "flip the board: free extra data" },
  { id: "inputs", label: "extra inputs", group: "extra", blurb: "castling, attack maps, material…" },
  { id: "slice", label: "data slice", group: "extra", blurb: "train only on endgames, say" },
];

function TechTree() {
  const [shown, setShown] = useState(0);
  const [done, setDone] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const run = () => {
    if (timer.current) clearInterval(timer.current);
    setDone(false); setShown(0);
    let n = 0;
    timer.current = setInterval(() => { n++; setShown(n); if (n >= NODES.length) { if (timer.current) clearInterval(timer.current); setTimeout(() => setDone(true), 500); } }, 320);
  };
  useEffect(() => { run(); return () => { if (timer.current) clearInterval(timer.current); }; }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useStageEvent("run", () => { if (shown < NODES.length) setShown(NODES.length); else if (!done) setDone(true); else run(); });
  useStageEvent("reset", run);
  const cx = 470, cy = 300;
  const pos = (n: TNode) => { const group = NODES.filter((x) => x.group === n.group), k = group.indexOf(n), a = (k / group.length) * Math.PI * 2 - Math.PI / 2 + (n.group === "extra" ? 0.2 : 0); const [rx, ry] = n.group === "seen" ? [215, 128] : [395, 255]; return [cx + Math.cos(a) * rx, cy + Math.sin(a) * ry] as const; };
  return (
    <div className="flex items-start gap-8" data-testid="techtree">
      <Unlock id="techtree">
        <svg width="940" height="600" className="rounded-3xl bg-slate-900/50">
          {NODES.map((n, i) => { const [x, y] = pos(n); return i < shown ? <motion.line key={n.id + "l"} x1={cx} y1={cy} x2={x} y2={y} stroke={done ? "#fbbf24" : n.group === "seen" ? "#38bdf8" : "#64748b"} strokeWidth={done ? 2.5 : 1.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} /> : null; })}
          <circle cx={cx} cy={cy} r={done ? 70 : 54} fill={done ? "#f59e0b" : "#0ea5e9"} className="transition-all duration-500" />
          <text x={cx} y={cy - 2} textAnchor="middle" fontSize="22" fontWeight="700" className="fill-slate-950">{done ? "FULL" : "your"}</text>
          <text x={cx} y={cy + 22} textAnchor="middle" fontSize="22" fontWeight="700" className="fill-slate-950">{done ? "PANEL" : "network"}</text>
          {NODES.map((n, i) => { const [x, y] = pos(n); const on = i < shown; return (
            <motion.g key={n.id} initial={false} animate={{ opacity: on ? 1 : 0, scale: on ? 1 : 0.4 }} style={{ transformOrigin: `${x}px ${y}px` }} data-testid={`node-${n.id}`}>
              <rect x={x - 82} y={y - 22} width="164" height="44" rx="12" fill={done ? "#78350f" : n.group === "seen" ? "#0c4a6e" : "#1e293b"} stroke={done ? "#fbbf24" : n.group === "seen" ? "#38bdf8" : "#64748b"} strokeWidth="2" />
              <text x={x} y={y - 2} textAnchor="middle" fontSize="18" fontWeight="600" className="fill-slate-100">{n.label}</text>
              <text x={x} y={y + 15} textAnchor="middle" fontSize="10.5" className="fill-slate-400">{n.blurb.length > 29 ? n.blurb.slice(0, 28) + "…" : n.blurb}</text>
            </motion.g>); })}
        </svg>
      </Unlock>
      <div className="w-[420px] space-y-4 text-2xl text-slate-300">
        <p>Blue: knobs you already used today.</p><p>Grey: the extras researchers tune.</p>
        {done ? <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl bg-amber-500/15 p-4 text-3xl text-amber-200">All of them unlock together at our booth: design a <b>chess</b> network, train it on a real GPU, and enter the tournament.</motion.p>
          : <p className="text-slate-500">Space: show all</p>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ close
function human(n: number): string {
  const units: [string, number][] = [["", 1], ["K", 1e3], ["M", 1e6], ["B", 1e9], ["T", 1e12], ["P", 1e15]];
  let u = units[0];
  for (const x of units) if (n >= x[1]) u = x;
  return `${(n / u[1]).toFixed(n / u[1] >= 100 ? 0 : 1)}${u[0]}`;
}

/** The only place the finale points to: the club's own site. (Not the dev server, not VITE_BOOTH_URL: the QR must always work on a phone.) */
export const CLUB_URL = "https://ieeecspesu.vercel.app";

function Close() {
  const s = useSession();
  const qr = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (qr.current) QRCode.toCanvas(qr.current, CLUB_URL, { width: 260, margin: 1, color: { dark: "#0f172a", light: "#f8fafc" } }).catch(() => undefined); }, []);
  const mins = Math.max(1, Math.round((Date.now() - s.startedAt) / 60000));
  return (
    <div className="flex flex-col" data-testid="close">
      <Unlock id="stats">
        <div data-testid="stats">
          {/* each box sizes to its content (nowrap + padding), so 4-digit run counts and "3.6P FLOPs" stay inside their background */}
          <div className="flex flex-wrap gap-6">
            {[["training runs", String(s.runs)], ["parameters trained", human(s.paramsTrained)], ["compute used", `${human(s.flops)} FLOPs`]].map(([label, v]) => (
              <motion.div key={label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} data-testid="stat-box" className="min-w-[300px] rounded-3xl bg-slate-900/70 px-8 py-6">
                <div className="whitespace-nowrap font-mono text-5xl font-bold text-sky-300">{v}</div><div className="mt-2 whitespace-nowrap text-2xl text-slate-400">{label}</div>
              </motion.div>))}
          </div>
          <div className="mt-3 text-xl text-slate-500">this session: {mins} min{s.humanDuel ? ` · duel score ${s.humanDuel.score}/${s.humanDuel.rounds}` : ""}</div>
        </div>
      </Unlock>
      <div className="mt-10 flex flex-col gap-8">
        <Unlock id="booth">
          <div className="flex w-full max-w-[1300px] items-center gap-10 rounded-3xl bg-slate-900/70 p-8" data-testid="club-card">
            <canvas ref={qr} className="shrink-0 rounded-xl" style={{ width: 260, height: 260 }} data-testid="qr" />
            <p className="min-w-0 text-4xl font-semibold leading-snug text-slate-100">Scan this link to check out our club! Or alternatively, go to <span className="mt-1 block whitespace-nowrap font-mono text-sky-300" data-testid="club-url">{CLUB_URL}</span></p>
          </div>
        </Unlock>
        <Unlock id="pitch"><p className="max-w-[1100px] text-5xl font-bold leading-snug text-amber-200">We do AI research that makes training cheaper, faster or more accurate.</p></Unlock>
      </div>
    </div>
  );
}

export function Act4() {
  const { stage } = useStage();
  return stage.id === "a4-tech" ? <TechTree /> : <Close />;
}
