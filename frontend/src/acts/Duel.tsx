import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Gauge, Unlock, Badge } from "../components/ui";
import { CLASSES, loadDoodles, paintStrokes, partialStrokes, strokeCountVisible, strokeLengths, type DoodleRecord } from "../lib/doodle";
import { forward, getModel, softmax, type Model } from "../lib/inference";
import { rasterize, toModelInput } from "../lib/rasterizer";
import { mulberry32, shuffled } from "../lib/rng";
import { fitCanvas } from "../lib/canvas";
import { urlParam } from "../lib/env";
import { setHumanDuel, useSession } from "../lib/session";
import { useStageEvent } from "../state/StageProvider";

const ROUNDS = 5;
const SPEED = 420; // data units per second: constant-speed replay (the data has no timing)
const COUNTDOWN = 3;
type Phase = "idle" | "drawing" | "reveal" | "final";

interface Round { rec: DoodleRecord; options: number[] }

function makeRounds(duel: DoodleRecord[], seed: number): Round[] {
  const rng = mulberry32(seed);
  const byClass = shuffled([...Array(10).keys()], rng).slice(0, ROUNDS);
  return byClass.map((c) => {
    const pool = duel.filter((d) => d.c === c);
    const rec = pool[Math.floor(rng() * pool.length)];
    const others = shuffled([...Array(10).keys()].filter((k) => k !== c), rng).slice(0, 3);
    return { rec, options: shuffled([c, ...others], rng) };
  });
}

/** Act 0 (opponent = random guesser) and the Act 4 rematch (opponent = the trained model, which sees only the strokes drawn so far). */
export function Duel({ mode }: { mode: "duel" | "rematch" }) {
  const [rounds, setRounds] = useState<Round[] | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [round, setRound] = useState(0);
  const [frac, setFrac] = useState(0);
  const [optionsAt, setOptionsAt] = useState<number | null>(null); // performance.now() when options appeared
  const [now, setNow] = useState(0);
  const [human, setHuman] = useState<number | null>(null);
  const [ai, setAi] = useState<number | null>(null);
  const [score, setScore] = useState({ human: 0, ai: 0, played: 0 });
  const [model, setModel] = useState<Model | null>(null);
  const session = useSession();
  const cv = useRef<HTMLCanvasElement>(null);
  const answered = useRef(false);
  const fracRef = useRef(0);
  const startTs = useRef(0);

  // The seed must be fixed for the life of the component. (It used to be recomputed with Math.random() on EVERY render, which changed `load`,
  // re-ran the effect, set new rounds, re-rendered, and so on forever: an endless render loop that pinned the CPU on the first screen.)
  const [seed] = useState(() => Number(urlParam("seed") ?? Math.floor(Math.random() * 1e6)));
  useEffect(() => {
    let live = true;
    loadDoodles("duel").then((d) => { if (live) setRounds(makeRounds(d, seed)); }).catch(() => undefined);
    return () => { live = false; };
  }, [seed]);
  useEffect(() => { if (mode === "rematch") getModel("doodle").then(setModel); }, [mode]);

  const reset = useCallback(() => { setPhase("idle"); setRound(0); setFrac(0); setOptionsAt(null); setHuman(null); setAi(null); setScore({ human: 0, ai: 0, played: 0 }); answered.current = false; }, []);
  useStageEvent("reset", reset);

  const cur = rounds?.[round];
  const total = cur ? strokeLengths(cur.rec.d).slice(-1)[0] : 1;
  const nStrokes = cur?.rec.d.length ?? 1;

  const startRound = useCallback(() => {
    answered.current = false; fracRef.current = 0; startTs.current = performance.now();
    setHuman(null); setAi(null); setFrac(0); setOptionsAt(null); setPhase("drawing");
  }, []);

  useStageEvent("run", () => {
    if (!rounds) return;
    if (phase === "idle") startRound();
    else if (phase === "reveal") { if (round + 1 >= ROUNDS) setPhase("final"); else { setRound((r) => r + 1); startRound(); } }
    else if (phase === "final") reset();
  });

  const modelGuess = useCallback((c: Round, f: number): number => {
    if (!model) return c.options[Math.floor(Math.random() * 4)];
    const img = rasterize(partialStrokes(c.rec.d, f));
    const p = softmax(forward(model, toModelInput(img)));
    return c.options.reduce((best, o) => (p[o] > p[best] ? o : best), c.options[0]);
  }, [model]);

  const answer = useCallback((choice: number | null) => {
    if (!cur || answered.current) return;
    answered.current = true;
    const aiChoice = mode === "rematch" ? modelGuess(cur, fracRef.current) : cur.options[Math.floor(Math.random() * 4)];
    setHuman(choice); setAi(aiChoice);
    setScore((s) => ({ human: s.human + (choice === cur.rec.c ? 1 : 0), ai: s.ai + (aiChoice === cur.rec.c ? 1 : 0), played: s.played + 1 }));
    setPhase("reveal");
  }, [cur, mode, modelGuess]);

  // replay animation + countdown
  useEffect(() => {
    if (phase !== "drawing" || !cur) return;
    let raf = 0;
    const loop = () => {
      const f = Math.min(1, ((performance.now() - startTs.current) / 1000) * (SPEED / total));
      fracRef.current = f; setFrac(f); setNow(performance.now());
      if (optionsAt === null && (strokeCountVisible(cur.rec.d, f) >= Math.min(3, nStrokes) || f >= 0.4)) setOptionsAt(performance.now());
      if (optionsAt !== null && (performance.now() - optionsAt) / 1000 >= COUNTDOWN) { answer(null); return; }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [phase, cur, total, nStrokes, optionsAt, answer]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (phase === "drawing" && optionsAt !== null && cur && "1234".includes(e.key) && e.key.length === 1) answer(cur.options[Number(e.key) - 1]);
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [phase, optionsAt, cur, answer]);

  useEffect(() => { if (phase === "final" && mode === "duel") setHumanDuel(score.human, ROUNDS); }, [phase, mode, score.human]);

  // paint the replay
  useEffect(() => {
    const c = cv.current;
    if (!c || !cur) return;
    const size = 460;
    const ctx = fitCanvas(c, size, size);
    const shown = phase === "idle" ? [] : phase === "drawing" ? partialStrokes(cur.rec.d, frac) : cur.rec.d;
    paintStrokes(ctx, shown, size, { bounds: cur.rec.d, width: 7, color: "#f1f5f9" });
  }, [cur, frac, phase]);

  const remaining = optionsAt !== null && phase === "drawing" ? Math.max(0, COUNTDOWN - (now - optionsAt) / 1000) : 0;
  const correct = cur?.rec.c;
  const humanAcc = score.played ? score.human / score.played : 0;

  return (
    <div className="flex items-start gap-10" data-testid={mode === "duel" ? "duel" : "rematch"}>
      <Unlock id="doodle">
        <div className="relative">
          <canvas ref={cv} style={{ width: 460, height: 460 }} className="rounded-3xl border-2 border-slate-700 bg-slate-900" />
          {phase === "idle" && <div className="absolute inset-0 grid place-items-center text-3xl text-slate-400">{rounds ? "Space to start" : "loading…"}</div>}
          {phase === "drawing" && optionsAt !== null && (
            <svg className="absolute right-3 top-3" width="64" height="64" viewBox="0 0 64 64">
              <circle cx="32" cy="32" r="26" fill="none" stroke="#1e293b" strokeWidth="6" />
              <circle cx="32" cy="32" r="26" fill="none" stroke="#fbbf24" strokeWidth="6" strokeLinecap="round" strokeDasharray={2 * Math.PI * 26} strokeDashoffset={2 * Math.PI * 26 * (1 - remaining / COUNTDOWN)} transform="rotate(-90 32 32)" />
              <text x="32" y="40" textAnchor="middle" fontSize="24" className="fill-slate-100">{Math.ceil(remaining)}</text>
            </svg>
          )}
        </div>
      </Unlock>
      <div className="flex w-[520px] flex-col gap-6">
        <Unlock id="options">
          <div className="grid grid-cols-2 gap-4" data-testid="options">
            <AnimatePresence>
              {cur && (phase === "reveal" || (phase === "drawing" && optionsAt !== null)) && cur.options.map((o, i) => {
                const revealed = phase === "reveal";
                const tone = !revealed ? "bg-slate-800 hover:bg-slate-700" : o === correct ? "bg-emerald-600" : o === human ? "bg-red-600/80" : "bg-slate-800/50";
                return (
                  <motion.button key={`${round}-${o}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
                    onClick={() => phase === "drawing" && answer(o)} className={`relative rounded-2xl px-5 py-5 text-left text-3xl font-semibold text-slate-100 ${tone}`}>
                    <span className="mr-3 text-xl text-slate-400">{i + 1}</span>{CLASSES[o]}
                    {revealed && <span className="absolute bottom-1 right-3 text-sm">{o === human ? "you " : ""}{o === ai ? (mode === "rematch" ? "AI" : "AI (guess)") : ""}</span>}
                  </motion.button>
                );
              })}
            </AnimatePresence>
            {phase === "reveal" && <p className="col-span-2 text-xl text-slate-400">{round + 1 < ROUNDS ? "Space for the next round" : "Space for the result"}</p>}
          </div>
        </Unlock>
        <Unlock id="score">
          <div className="flex gap-8 text-3xl" data-testid="score">
            <div><div className="text-lg text-slate-400">You</div><div className="font-mono text-5xl text-sky-300">{score.human}</div></div>
            <div><div className="text-lg text-slate-400">{mode === "rematch" ? "Trained AI" : "The AI"}</div><div className="font-mono text-5xl text-amber-300">{score.ai}</div></div>
            <div className="self-end text-lg text-slate-500">round {Math.min(round + (phase === "idle" ? 0 : 1), ROUNDS)}/{ROUNDS}</div>
          </div>
          {mode === "rematch" && session.humanDuel && <p className="mt-2 text-lg text-slate-400">Your first duel: {session.humanDuel.score}/{session.humanDuel.rounds} against the random AI.</p>}
          {mode === "rematch" && model?.isMock && <div className="mt-2"><Badge tone="amber" title="trained weights are not bundled yet (run export_weights.py): random weights stand in">demo weights</Badge></div>}
        </Unlock>
        <Unlock id="gauge"><Gauge value={humanAcc} chance={0.25} label="your accuracy (4 choices)" /></Unlock>
        {phase === "final" && (
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="rounded-2xl bg-slate-800 p-6 text-3xl" data-testid="duel-final">
            {mode === "duel"
              ? (score.human > score.ai ? "You beat a random guesser. Of course you did." : score.human === score.ai ? "A tie with a random guesser?!" : "Lucky AI! A pure guesser beat you.")
              : (score.ai > score.human ? "The trained AI wins." : score.ai === score.human ? "A tie! It is learning fast." : "You win this time.")}
            <div className="mt-2 text-xl text-slate-400">Space to play again</div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
