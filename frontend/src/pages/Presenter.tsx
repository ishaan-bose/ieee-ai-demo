import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useHealth } from "../state/health";
import { CHANNEL } from "../state/StageProvider";
import { STAGES, stageIndex } from "../state/stages";

/** /presenter (SPEC 5.2): the talk track for the CURRENT stage, meant for the presenter's phone. Follows laptop B through a
 *  BroadcastChannel (same browser) or by polling the server (phone on the laptop's LAN address; `FRONTEND_HOST=0.0.0.0 npm run dev`). */
export default function Presenter() {
  const [idx, setIdx] = useState(0);
  const [follow, setFollow] = useState(true);
  const [source, setSource] = useState<"waiting" | "tab" | "server">("waiting");
  const lastBroadcast = useRef(0);
  const { online, checked } = useHealth();
  const followRef = useRef(follow);
  followRef.current = follow;

  useEffect(() => {
    let ch: BroadcastChannel | null = null;
    try {
      ch = new BroadcastChannel(CHANNEL);
      ch.onmessage = (e) => { lastBroadcast.current = Date.now(); const i = stageIndex(e.data.stage_id); if (i >= 0 && followRef.current) { setIdx(i); setSource("tab"); } };
    } catch { /* no BroadcastChannel: polling only */ }
    const poll = setInterval(async () => {
      if (Date.now() - lastBroadcast.current < 4000 || !followRef.current) return;
      try { const s = await api.presenterGet(); const i = stageIndex(s.stage_id); if (i >= 0) { setIdx(i); setSource("server"); } } catch { /* offline */ }
    }, 1500);
    return () => { ch?.close(); clearInterval(poll); };
  }, []);

  const stage = STAGES[idx], next = STAGES[idx + 1];
  return (
    <main className="min-h-screen bg-slate-950 p-5 text-slate-100" data-testid="presenter">
      {checked && !online && <div className="mb-3 rounded-xl bg-red-600 px-4 py-2 text-xl font-semibold" data-testid="presenter-offline">SERVER OFFLINE: races play the recorded streams</div>}
      <div className="flex items-center justify-between text-sm text-slate-400">
        <span>stage {idx + 1}/{STAGES.length} · {source === "waiting" ? "waiting for the demo page…" : `following via ${source}`}</span>
        <label className="flex items-center gap-2"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow</label>
      </div>
      <div className="mt-2 text-lg uppercase tracking-widest text-sky-400">Act {stage.act}</div>
      <h1 className="text-4xl font-bold leading-tight" data-testid="presenter-title">{stage.title}</h1>
      <ul className="mt-5 space-y-4 text-2xl leading-snug">{stage.notes.map((n, i) => <li key={i} className="rounded-xl bg-slate-900 p-4">{n}</li>)}</ul>
      {stage.hint && <p className="mt-5 text-lg text-slate-400">Keys: {stage.hint}</p>}
      {next && <div className="mt-6 rounded-xl border border-slate-800 p-4 text-slate-400"><div className="text-sm uppercase">next</div><div className="text-xl text-slate-200">{next.title}</div><div className="mt-1">{next.notes[0]}</div></div>}
      <div className="mt-6 flex gap-3">
        <button className="flex-1 rounded-xl bg-slate-800 py-4 text-2xl" onClick={() => { setFollow(false); setIdx((i) => Math.max(0, i - 1)); }}>◀ peek back</button>
        <button className="flex-1 rounded-xl bg-slate-800 py-4 text-2xl" onClick={() => { setFollow(false); setIdx((i) => Math.min(STAGES.length - 1, i + 1)); }}>peek ahead ▶</button>
      </div>
      <p className="mt-6 text-sm leading-relaxed text-slate-500">Laptop keys: → next · ← back · Space run · S recorded result · R reset stage · F force recorded · G stage menu · P offline marker on the big screen · L slow-mo (Act 1)</p>
    </main>
  );
}
