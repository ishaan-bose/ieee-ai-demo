import { motion } from "framer-motion";
import { Suspense, lazy, useEffect, useState, type ReactNode } from "react";
import { Act1 } from "../acts/Act1";
import { Act2Forward } from "../acts/Act2Forward";
import { Act2Backward, Act2Race } from "../acts/Act2Race";
import { Duel } from "../acts/Duel";
import { Credits, TimerRing } from "../components/ui";
import { StageMenu } from "../components/StageMenu";
import { MOCK, isMockOffline } from "../lib/env";
import { useHealth } from "../state/health";
import { StageProvider, useStage } from "../state/StageProvider";

const Act3 = lazy(() => import("../acts/Act3").then((m) => ({ default: m.Act3 })));
const Act4 = lazy(() => import("../acts/Act4").then((m) => ({ default: m.Act4 })));

function Body(): ReactNode {
  const { stage } = useStage();
  switch (stage.id) {
    case "a0-duel": return <Duel mode="duel" />;
    case "a2-forward": return <Act2Forward />;
    case "a2-loss": return <Act2Race kind="loss" />;
    case "a2-backward": return <Act2Backward />;
    case "a2-lr": return <Act2Race kind="lr" />;
    case "a2-batch": return <Act2Race kind="batch" />;
    case "a4-rematch": return <Duel mode="rematch" />;
  }
  if (stage.act === 1) return <Act1 />;
  if (stage.act === 3) return <Suspense fallback={null}><Act3 /></Suspense>;
  return <Suspense fallback={null}><Act4 /></Suspense>;
}

/** The layout is designed for a 1600x900 viewport; scale it to whatever the projector/laptop gives us. */
function useFitScale(): number {
  const calc = () => Math.max(0.55, Math.min(1.6, Math.min(window.innerWidth / 1600, window.innerHeight / 900)));
  const [s, setS] = useState(calc);
  useEffect(() => { const f = () => setS(calc()); window.addEventListener("resize", f); return () => window.removeEventListener("resize", f); }, []);
  return s;
}

function Shell() {
  const scale = useFitScale();
  const { stage, index, stages, presenterMode } = useStage();
  const { online, checked } = useHealth();
  return (
    <main style={{ zoom: scale, minHeight: `${100 / scale}vh` }} className="relative overflow-hidden bg-slate-950 px-10 pb-16 pt-6 text-slate-100">
      <header className="mb-5 flex items-end justify-between">
        <div>
          <div className="text-lg uppercase tracking-widest text-sky-400">Act {stage.act}</div>
          <motion.h1 key={stage.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="text-4xl font-bold" data-testid="stage-title">{stage.title}</motion.h1>
        </div>
        <div className="mr-20 flex items-center gap-1.5" aria-hidden>{stages.map((s, i) => <span key={s.id} className={`h-2 rounded-full transition-all ${i === index ? "w-6 bg-sky-400" : i < index ? "w-2 bg-slate-500" : "w-2 bg-slate-800"}`} />)}</div>
      </header>
      {/* No exit animation: with mode="wait" the OLD stage stayed mounted for 250 ms, re-rendering (and re-running its effects) against the NEW stage's context. */}
      <motion.div key={stage.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}><Body /></motion.div>
      <div className="fixed bottom-1 right-3 z-10 text-sm text-slate-600">{stage.hint}</div>
      {/* offline marker: tiny, and only shown in presenter mode (P) so the audience never sees it */}
      {presenterMode && checked && !online && <div className="fixed left-2 top-2 z-30 flex items-center gap-1 text-xs text-red-400" title="server offline: races play recorded streams" data-testid="offline-marker"><span className="h-2.5 w-2.5 rounded-full bg-red-500" />offline</div>}
      {presenterMode && MOCK && <div className="fixed left-2 top-8 z-30 text-xs text-amber-400">mock{isMockOffline() ? " (server off)" : ""} · O toggles server</div>}
      <TimerRing /><StageMenu /><Credits />
    </main>
  );
}

export default function Demo() {
  return <StageProvider><Shell /></StageProvider>;
}
