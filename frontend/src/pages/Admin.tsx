import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { LineChart } from "../components/charts";
import { Badge } from "../components/ui";
import { api, ApiError, hasAdminToken, setAdminToken } from "../lib/api";
import type { AdminJob, AdminQueue, AdminStatus } from "../lib/types";

const tone = (s: string) => (s === "running" ? "green" : s === "queued" ? "sky" : s === "done" ? "slate" : s === "killed" || s === "error" ? "red" : "amber") as "green" | "sky" | "slate" | "red" | "amber";
const hms = (s: number) => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;

function Modal({ children }: { children: ReactNode }) {
  return <div className="fixed inset-0 z-50 grid place-items-center bg-black/70"><div className="w-[460px] rounded-2xl bg-slate-900 p-6 ring-1 ring-slate-600">{children}</div></div>;
}

/** Hold-to-confirm button: must be held for `ms` (no accidental clicks). */
function HoldButton({ label, onConfirm, disabled, ms = 2000 }: { label: string; onConfirm: () => void; disabled?: boolean; ms?: number }) {
  const [p, setP] = useState(0);
  const t0 = useRef(0);
  const raf = useRef(0);
  const stop = () => { cancelAnimationFrame(raf.current); setP(0); };
  const start = () => {
    if (disabled) return;
    t0.current = performance.now();
    const loop = () => { const f = (performance.now() - t0.current) / ms; if (f >= 1) { setP(0); onConfirm(); return; } setP(f); raf.current = requestAnimationFrame(loop); };
    raf.current = requestAnimationFrame(loop);
  };
  useEffect(() => () => cancelAnimationFrame(raf.current), []);
  return (
    <button disabled={disabled} onPointerDown={start} onPointerUp={stop} onPointerLeave={stop} data-testid="hold-reset" className="relative overflow-hidden rounded-xl bg-red-700 px-5 py-3 text-xl font-bold text-white disabled:opacity-40">
      <span className="absolute inset-y-0 left-0 bg-red-400/60" style={{ width: `${p * 100}%` }} /><span className="relative">{label}</span>
    </button>
  );
}

export default function Admin() {
  const [authed, setAuthed] = useState(hasAdminToken());
  const [token, setToken] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [q, setQ] = useState<AdminQueue | null>(null);
  const [mon, setMon] = useState<AdminStatus | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{ text: string; run: () => Promise<unknown> } | null>(null);
  const [resetText, setResetText] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try { setQ(await api.admin.queue()); setErr(null); }
    catch (e) { if (e instanceof ApiError && e.status === 401) { setAuthed(false); setAuthError("wrong token"); } else setErr(e instanceof Error ? e.message : String(e)); }
  }, []);

  const login = async () => {
    setAdminToken(token); setAuthError(null);
    try { await api.admin.queue(); setAuthed(true); setToken(""); }
    catch (e) { setAdminToken(""); setAuthError(e instanceof ApiError && e.status === 401 ? "wrong token" : e instanceof ApiError && e.code === "admin_disabled" ? "ADMIN_TOKEN is not set on the server" : e instanceof Error ? e.message : "failed"); }
  };

  useEffect(() => { if (!authed) return; refresh(); const id = setInterval(refresh, 2000); return () => clearInterval(id); }, [authed, refresh]);
  useEffect(() => {
    if (!authed) return;
    const ctrl = new AbortController();
    (async () => { for (;;) { try { for await (const s of api.admin.stream(ctrl.signal)) setMon(s); } catch { if (ctrl.signal.aborted) return; } await new Promise((r) => setTimeout(r, 2000)); } })();
    return () => ctrl.abort();
  }, [authed]);

  const act = (text: string, run: () => Promise<unknown>) => setConfirm({ text, run });
  const doConfirm = async () => {
    const c = confirm; setConfirm(null);
    if (!c) return;
    try { await c.run(); setMsg("done"); } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); }
    refresh();
  };
  const download = async (format: "json" | "csv") => {
    try { const blob = await api.admin.exportAll(format); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `export.${format}`; a.click(); } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); }
  };

  if (!authed) return (
    <main className="grid min-h-screen place-items-center bg-slate-950 text-slate-100" data-testid="admin-login">
      <form className="w-[380px] space-y-4 rounded-2xl bg-slate-900 p-6" onSubmit={(e) => { e.preventDefault(); login(); }}>
        <h1 className="text-2xl font-bold">Admin</h1>
        <input type="password" autoFocus placeholder="admin token" className="w-full rounded-lg bg-slate-800 px-3 py-2 text-lg" value={token} onChange={(e) => setToken(e.target.value)} data-testid="token" />
        <button className="w-full rounded-lg bg-sky-500 py-2 text-lg font-semibold text-slate-950" data-testid="login">Enter</button>
        {authError && <p className="text-red-300" data-testid="auth-error">{authError}</p>}
        <p className="text-sm text-slate-500">The token is kept in memory only (a page reload asks again).</p>
      </form>
    </main>
  );

  const cur = q?.jobs.find((j) => j.id === q.current_job_id);
  return (
    <main className="mx-auto max-w-[1400px] p-5 text-slate-100" data-testid="admin">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="mr-4 text-3xl font-bold">Admin</h1>
        {q && <>
          <button onClick={() => act(q.paused ? "Resume the queue?" : "Pause the queue? The running job returns to the front of the queue.", () => (q.paused ? api.admin.resume() : api.admin.pause()))} className={`rounded-lg px-4 py-2 text-lg font-semibold ${q.paused ? "bg-amber-500 text-slate-950" : "bg-slate-800"}`} data-testid="pause">{q.paused ? "▶ Resume queue" : "⏸ Pause queue"}</button>
          <button onClick={() => act(q.demo_mode ? "Turn Demo Mode OFF (competition jobs may run again)?" : "Turn Demo Mode ON? Competition jobs stop and wait; races get the GPU.", () => api.admin.demoMode(!q.demo_mode))} className={`rounded-lg px-4 py-2 text-lg font-semibold ${q.demo_mode ? "bg-fuchsia-500 text-slate-950" : "bg-slate-800"}`} data-testid="demo-mode">{q.demo_mode ? "Demo Mode: ON" : "Demo Mode: off"}</button>
        </>}
        <button onClick={() => download("json")} className="rounded-lg bg-slate-800 px-4 py-2 text-lg hover:bg-slate-700">Export JSON</button>
        <button onClick={() => download("csv")} className="rounded-lg bg-slate-800 px-4 py-2 text-lg hover:bg-slate-700">Export CSV</button>
        {msg && <span className="text-slate-400" data-testid="admin-msg">{msg}</span>}{err && <Badge tone="red">{err}</Badge>}
      </div>

      <div className="mt-4 grid grid-cols-[1fr_440px] gap-5">
        <section className="overflow-x-auto rounded-2xl bg-slate-900/70 p-3">
          <table className="w-full text-left text-base" data-testid="queue-table">
            <thead className="text-sm uppercase text-slate-500"><tr><th className="p-2">#</th><th>status</th><th>who</th><th>model</th><th>tier</th><th>config</th><th>progress</th><th>stop</th><th></th></tr></thead>
            <tbody>
              {q?.jobs.map((j: AdminJob) => (
                <tr key={j.id} className="border-t border-slate-800" data-testid={`job-${j.id}`}>
                  <td className="p-2 font-mono">{j.id}</td><td><Badge tone={tone(j.status)}>{j.status}</Badge></td>
                  <td>{j.nickname ?? "—"} <span className="text-xs text-slate-500">{j.participant_code}</span></td><td>{j.model_name ?? "—"}</td>
                  <td>{j.tier ?? "—"}<div className="text-xs text-slate-500">{j.param_count ? `${(j.param_count / 1e6).toFixed(2)}M` : ""}</div></td>
                  <td className="max-w-[240px] text-sm text-slate-400">{j.config_summary ?? ""}</td>
                  <td className="w-[120px]">{j.progress != null ? <div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full bg-sky-400" style={{ width: `${j.progress * 100}%` }} /></div> : "—"}<div className="text-xs text-slate-500">{j.active_gpu_seconds ? hms(j.active_gpu_seconds) : ""}{j.preemptions ? ` · ${j.preemptions}× preempted` : ""}{j.samples_per_s ? ` · ${Math.round(j.samples_per_s).toLocaleString()} samples/s` : ""}{j.concurrency ? ` · ${j.concurrency} at once` : ""}</div></td>
                  <td className="text-sm" data-testid={`stop-${j.id}`}>{j.stop_reason ? <Badge tone={j.stop_reason === "budget" ? "green" : j.stop_reason === "time" ? "amber" : j.stop_reason === "killed" ? "slate" : "red"} title={j.stop_reason === "time" ? "hit the active-GPU-time cap before its FLOPs budget: MAX_CONCURRENT_JOBS may be too high" : undefined}>{j.stop_reason}</Badge> : (j.error_message ?? "")}</td>
                  <td className="whitespace-nowrap p-2 text-right">
                    {["queued", "running"].includes(j.status) && <button className="mr-1 rounded bg-red-800 px-2 py-1 text-sm" data-testid={`kill-${j.id}`} onClick={() => act(`Kill job ${j.id} (${j.nickname ?? "race"})? Its progress is thrown away.`, () => api.admin.kill(j.id))}>kill</button>}
                    {j.status !== "running" && <button className="mr-1 rounded bg-slate-700 px-2 py-1 text-sm" data-testid={`remove-${j.id}`} onClick={() => act(`Remove job ${j.id} (${j.nickname ?? "race"}) from the queue?`, () => api.admin.remove(j.id))}>remove</button>}
                    {j.kind === "competition" && <button className="rounded bg-sky-800 px-2 py-1 text-sm" data-testid={`redo-${j.id}`} onClick={() => act(`Redo job ${j.id} (${j.nickname}) from scratch? Training restarts at the back of the queue.`, () => api.admin.redo(j.id))}>redo</button>}
                  </td>
                </tr>))}
              {q && q.jobs.length === 0 && <tr><td colSpan={9} className="p-6 text-center text-slate-500">the queue is empty</td></tr>}
            </tbody>
          </table>
        </section>

        <section className="space-y-3 rounded-2xl bg-slate-900/70 p-4" data-testid="monitor">
          <h2 className="text-xl font-semibold">Live monitor {cur && <span className="text-base font-normal text-slate-400">job {cur.id}: {cur.nickname}</span>}</h2>
          {mon ? <>
            <div className="flex gap-6 text-lg"><div>GPU <b className="font-mono">{mon.gpu.utilization != null ? `${mon.gpu.utilization}%` : "n/a"}</b></div><div>memory <b className="font-mono">{mon.gpu.memory_used_gb ?? "?"}/{mon.gpu.memory_total_gb ?? "?"} GB</b></div><div>{mon.gpu.device}</div></div>
            {mon.progress != null && <div className="h-3 overflow-hidden rounded-full bg-slate-800"><div className="h-full bg-emerald-400" style={{ width: `${mon.progress * 100}%` }} /></div>}
            {mon.curves && mon.curves.val.length > 1 ? <LineChart width={400} height={170} series={[{ color: "#56B4E9", points: mon.curves.train.map((p) => ({ x: p.t, y: p.loss })) }, { color: "#E69F00", points: mon.curves.val.map((p) => ({ x: p.t, y: p.loss })) }]} title="current job: train (blue) / val (orange)" xLabel="s" /> : <p className="text-slate-500">no competition job running</p>}
            <pre className="h-[170px] overflow-auto rounded-lg bg-slate-950 p-2 text-xs leading-tight text-slate-400" data-testid="log-tail">{mon.log_tail.join("\n")}</pre>
          </> : <p className="text-slate-500">connecting…</p>}
        </section>
      </div>

      <section className="mt-5 rounded-2xl border border-red-900 bg-red-950/30 p-4" data-testid="nuclear">
        <h2 className="text-xl font-semibold text-red-300">Nuclear reset</h2>
        <p className="text-slate-400">Snapshots the database to a timestamped file, then clears the queue, results and stored runs. Type RESET, then HOLD the button for 2 seconds.</p>
        <div className="mt-3 flex items-center gap-3"><input className="w-40 rounded-lg bg-slate-800 px-3 py-2 text-lg" placeholder="type RESET" value={resetText} onChange={(e) => setResetText(e.target.value)} data-testid="reset-text" />
          <HoldButton label="hold to reset everything" disabled={resetText !== "RESET"} onConfirm={async () => { try { const r = await api.admin.reset(); setMsg(`reset done. snapshot: ${r.snapshot}`); setResetText(""); refresh(); } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); } }} /></div>
      </section>

      {confirm && <Modal><p className="text-xl" data-testid="confirm-text">{confirm.text}</p><div className="mt-5 flex justify-end gap-3"><button className="rounded-lg bg-slate-700 px-4 py-2" onClick={() => setConfirm(null)} data-testid="confirm-no">Cancel</button><button className="rounded-lg bg-red-600 px-4 py-2 font-semibold" onClick={doConfirm} data-testid="confirm-yes">Yes, do it</button></div></Modal>}
    </main>
  );
}
