import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import { DrawCanvas, type DrawHandle } from "../components/DrawCanvas";
import { NetworkDiagram } from "../components/NetworkDiagram";
import { Badge, Unlock } from "../components/ui";
import { CLASSES } from "../lib/doodle";
import { diagramFrom, type DiagramData } from "../lib/diagram";
import { forwardTrace, getModel, softmax, topK, type Model } from "../lib/inference";
import { rasterize, toModelInput, type Stroke } from "../lib/rasterizer";
import { useStageEvent } from "../state/StageProvider";

function Preview28({ img }: { img: Uint8Array | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!, ctx = c.getContext("2d")!;
    const d = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) { const v = img ? img[i] : 0; d.data[4 * i] = v; d.data[4 * i + 1] = v; d.data[4 * i + 2] = v; d.data[4 * i + 3] = 255; }
    ctx.putImageData(d, 0, 0);
  }, [img]);
  return <canvas ref={ref} width={28} height={28} style={{ width: 112, height: 112, imageRendering: "pixelated" }} className="rounded-lg border border-slate-700" />;
}

/** Act 2 step 1, hero visual 2: the volunteer draws; after every stroke the bundled model re-guesses and light flows through the network. */
export function Act2Forward() {
  const [model, setModel] = useState<Model | null>(null);
  const [img, setImg] = useState<Uint8Array | null>(null);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [diagram, setDiagram] = useState<DiagramData | null>(null);
  const [pulse, setPulse] = useState(0);
  const [strokes, setStrokes] = useState(0);
  const draw = useRef<DrawHandle>(null);
  useEffect(() => { getModel("doodle").then(setModel); }, []);

  const blank = useMemo(() => {
    if (!model) return null;
    const x = new Float32Array(784);
    const tr = forwardTrace(model, x);
    return diagramFrom(model, x, tr, new Float32Array(10).fill(0.1));
  }, [model]);

  const onStroke = (s: Stroke[]) => {
    if (!model) return;
    const im = rasterize(s), x = toModelInput(im);
    const tr = forwardTrace(model, x), p = softmax(tr.out);
    setImg(im); setProbs(p); setStrokes(s.length); setDiagram(diagramFrom(model, x, tr, p)); setPulse((n) => n + 1);
  };
  const clear = () => { draw.current?.clear(); setImg(null); setProbs(null); setDiagram(null); setStrokes(0); };
  useStageEvent("reset", clear);
  const top = probs ? topK(probs, 5) : [];

  return (
    <div className="flex items-start gap-8" data-testid="act2-forward">
      <Unlock id="draw">
        <div>
          <DrawCanvas ref={draw} size={380} onStrokeEnd={onStroke} onClear={() => undefined} />
          <div className="mt-3 flex items-center gap-4 text-lg text-slate-400">
            <button className="rounded-lg bg-slate-800 px-4 py-2 hover:bg-slate-700" onClick={clear}>Clear (R)</button>
            <span>strokes: {strokes}</span>
            <div className="flex items-center gap-2"><Preview28 img={img} /><span className="max-w-[110px] leading-tight">what the AI sees (28×28)</span></div>
          </div>
        </div>
      </Unlock>
      <div className="flex flex-col gap-4">
        <Unlock id="network"><NetworkDiagram data={diagram ?? blank} pulse={pulse} width={720} height={330} /></Unlock>
        <Unlock id="top5">
          <div className="w-[720px] space-y-1.5" data-testid="top5">
            <div className="flex items-center gap-3 text-xl text-slate-300">The AI's top 5 guesses {model?.isMock && <Badge tone="amber" title="trained weights are not bundled yet: random weights stand in">demo weights</Badge>}</div>
            {(top.length ? top : Array.from({ length: 5 }, (_, i) => ({ index: -1 - i, p: 0 }))).map((t, rank) => (
              <motion.div key={t.index} layout className="flex items-center gap-3">
                <span className="w-32 truncate text-2xl text-slate-200">{t.index >= 0 ? CLASSES[t.index] : "…"}</span>
                <div className="h-6 flex-1 overflow-hidden rounded-full bg-slate-800"><motion.div className="h-full rounded-full" style={{ backgroundColor: rank === 0 ? "#fbbf24" : "#38bdf8" }} animate={{ width: `${Math.max(1, t.p * 100)}%` }} transition={{ type: "spring", stiffness: 120, damping: 18 }} /></div>
                <span className="w-14 text-right font-mono text-lg text-slate-400">{(t.p * 100).toFixed(0)}%</span>
              </motion.div>
            ))}
          </div>
        </Unlock>
      </div>
    </div>
  );
}
