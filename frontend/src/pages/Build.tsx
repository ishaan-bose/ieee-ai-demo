import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Badge } from "../components/ui";
import { LineChart } from "../components/charts";
import { ACTIVATION_NAMES, DEFAULTS, MAX_PARAMS, TIERS, defaultConfig, previewLocal, widthsOf, type Config } from "../lib/chessConfig";
import { api, ApiError } from "../lib/api";
import { backupJson, loadLocal, markSent, newClientId, saveLocal, type LocalSub } from "../lib/localSubs";
import { flushPending } from "../lib/flush";
import type { Preview, SubmissionStatus } from "../lib/types";
import { useHealth } from "../state/health";

const BATCHES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
const fmt = (n: number) => n.toLocaleString("en-US");
const human = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="block"><span className="text-lg font-medium text-slate-200">{label}</span>{hint && <span className="ml-2 text-sm text-slate-500">{hint}</span>}<div className="mt-1">{children}</div></label>;
}
const input = "w-full rounded-lg bg-slate-800 px-3 py-2 text-lg text-slate-100 outline-none ring-1 ring-slate-700 focus:ring-sky-400";
const sel = input;

export default function Build() {
  const { online } = useHealth();
  const [cfg, setCfg] = useState<Config>(defaultConfig);
  const [nickname, setNickname] = useState("");
  const [modelName, setModelName] = useState("");
  const [contact, setContact] = useState("");
  const [consent, setConsent] = useState(false);
  const [perLayer, setPerLayer] = useState(false);
  const [server, setServer] = useState<Preview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [subs, setSubs] = useState<LocalSub[]>(() => loadLocal());
  const [statuses, setStatuses] = useState<Record<string, SubmissionStatus>>({});
  const set = useCallback(<K extends keyof Config>(k: K, v: Config[K]) => setCfg((c) => ({ ...c, [k]: v })), []);

  const widths = widthsOf(cfg);
  const local = useMemo(() => previewLocal(cfg), [cfg]);
  const tier = server?.valid && server.tier ? TIERS.find((t) => t.name === server.tier) ?? local.tier : local.tier;
  const errors = server && !server.valid ? server.errors : local.errors;

  // the server is authoritative: confirm validity (debounced); offline, the local preview is used
  useEffect(() => {
    if (!online) { setServer(null); return; }
    const t = setTimeout(() => api.preview(cfg).then(setServer).catch(() => setServer(null)), 350);
    return () => clearTimeout(t);
  }, [cfg, online]);

  // participant codes: poll the status of everything sent, every 5 s
  const subsRef = useRef(subs);
  subsRef.current = subs;
  useEffect(() => {
    if (!online) return;
    const tick = async () => {
      const sent = subsRef.current.filter((s) => s.status === "sent" && s.submission_id);
      const out: Record<string, SubmissionStatus> = {};
      await Promise.all(sent.map(async (s) => { try { out[s.client_id] = await api.submission(s.submission_id!); } catch { /* ignore */ } }));
      setStatuses((p) => ({ ...p, ...out }));
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, [online, subs.length]);
  // pending submissions go out by themselves when the server returns (also done globally in HealthProvider)
  useEffect(() => { if (online) flushPending().then(() => setSubs(loadLocal())); }, [online]);

  const valid = errors.length === 0 && nickname.trim() && modelName.trim() && consent;
  const submit = async () => {
    setError(null); setBusy(true);
    const clientId = newClientId();
    saveLocal({ client_id: clientId, nickname: nickname.trim(), model_name: modelName.trim(), contact: contact.trim() || undefined, consent, config: cfg }); // saved BEFORE sending
    setSubs(loadLocal());
    try {
      const res = await api.submit({ nickname: nickname.trim(), model_name: modelName.trim(), contact: contact.trim() || undefined, consent, config: cfg, client_id: clientId });
      markSent(clientId, res);
    } catch (e) {
      if (e instanceof ApiError && e.status === 0) setError("The server is not reachable. Your entry is saved on this device and will upload by itself when it comes back.");
      else setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); setSubs(loadLocal()); }
  };
  const download = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([backupJson()], { type: "application/json" })); a.download = "my-chess-ai-submissions.json"; a.click(); };

  return (
    <main className="mx-auto max-w-[1180px] p-6 text-slate-100" data-testid="build">
      <h1 className="text-4xl font-bold">Build your own chess AI</h1>
      <p className="mt-1 text-xl text-slate-400">You design the network that judges chess positions. It trains on a real GPU. A tournament after the event decides who is best.</p>
      {!online && <div className="mt-3 rounded-xl bg-amber-500/15 px-4 py-2 text-lg text-amber-200" data-testid="build-offline">Server unreachable: you can still fill this in. Entries are saved on this device and upload automatically later.</div>}
      <div className="mt-5 grid grid-cols-[1fr_360px] gap-6">
        <div className="space-y-6">
          <section className="space-y-3 rounded-2xl bg-slate-900/70 p-5">
            <h2 className="text-2xl font-semibold">You</h2>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Nickname" hint="public"><input className={input} maxLength={24} value={nickname} onChange={(e) => setNickname(e.target.value)} data-testid="nickname" /></Field>
              <Field label="Model name" hint="public"><input className={input} maxLength={32} value={modelName} onChange={(e) => setModelName(e.target.value)} data-testid="modelname" /></Field>
            </div>
            <Field label="Contact (email or handle)" hint="optional, PRIVATE: only the organisers see it"><input className={input} maxLength={120} value={contact} onChange={(e) => setContact(e.target.value)} /></Field>
            <label className="flex items-start gap-3 text-base text-slate-300"><input type="checkbox" className="mt-1 h-5 w-5" checked={consent} onChange={(e) => setConsent(e.target.checked)} data-testid="consent" />
              <span>I agree that my model, my nickname and my training logs (anonymised) may be used and stored. If I gave a contact, it is used only to reach me about this event.</span></label>
          </section>

          <section className="space-y-4 rounded-2xl bg-slate-900/70 p-5">
            <h2 className="text-2xl font-semibold">The network</h2>
            <div className="grid grid-cols-2 gap-5">
              <Field label={`Layers: ${perLayer ? widths.length : cfg.layers}`}>
                <input type="range" className="w-full accent-sky-400" min={1} max={16} value={perLayer ? widths.length : cfg.layers} data-testid="layers"
                  onChange={(e) => { const n = Number(e.target.value); if (perLayer) { const w = widths.slice(0, n); while (w.length < n) w.push(w[w.length - 1] ?? cfg.width); set("layer_widths", w); set("layers", n); } else set("layers", n); }} />
              </Field>
              <Field label={`Width: ${cfg.width}`} hint="units per layer">
                <input type="range" className="w-full accent-orange-400" min={3} max={13} step={0.25} value={Math.log2(cfg.width)} disabled={perLayer} data-testid="width"
                  onChange={(e) => set("width", Math.min(8192, Math.max(8, Math.round(2 ** Number(e.target.value)))))} />
              </Field>
            </div>
            <label className="flex items-center gap-2 text-base text-slate-300"><input type="checkbox" checked={perLayer} onChange={(e) => { setPerLayer(e.target.checked); set("layer_widths", e.target.checked ? widthsOf(cfg) : null); }} /> different width for each layer</label>
            {perLayer && <div className="flex flex-wrap gap-2">{widths.map((w, i) => <input key={i} type="number" min={8} max={8192} className="w-24 rounded-lg bg-slate-800 px-2 py-1 text-lg" value={w} onChange={(e) => { const nw = widths.slice(); nw[i] = Math.min(8192, Math.max(8, Number(e.target.value) || 8)); set("layer_widths", nw); }} />)}</div>}
            <div className="grid grid-cols-2 gap-5">
              <Field label="Activation">
                <div className="flex gap-2"><select className={sel} value={cfg.activation.name} onChange={(e) => set("activation", { ...cfg.activation, name: e.target.value })}>{ACTIVATION_NAMES.map((a) => <option key={a}>{a}</option>)}</select>
                  {["leaky_relu", "elu"].includes(cfg.activation.name) && <input type="number" step="0.05" min={0} max={1} className="w-24 rounded-lg bg-slate-800 px-2" title="alpha" value={cfg.activation.alpha} onChange={(e) => set("activation", { ...cfg.activation, alpha: Number(e.target.value) })} />}
                  {["swish", "softplus"].includes(cfg.activation.name) && <input type="number" step="0.1" min={0.1} max={10} className="w-24 rounded-lg bg-slate-800 px-2" title="beta" value={cfg.activation.beta} onChange={(e) => set("activation", { ...cfg.activation, beta: Number(e.target.value) })} />}
                  {cfg.activation.name === "clipped_relu" && <input type="number" step="0.5" min={0.5} max={20} className="w-24 rounded-lg bg-slate-800 px-2" title="cap" value={cfg.activation.cap} onChange={(e) => set("activation", { ...cfg.activation, cap: Number(e.target.value) })} />}</div>
              </Field>
              <Field label="Loss">
                <div className="flex gap-2"><select className={sel} value={cfg.loss.name} onChange={(e) => set("loss", { ...cfg.loss, name: e.target.value as Config["loss"]["name"] })}><option value="mse">mse</option><option value="huber">huber</option><option value="bce">bce (win probability)</option><option value="l1">l1</option></select>
                  {cfg.loss.name === "huber" && <input type="number" step="0.1" min={0.01} max={10} className="w-24 rounded-lg bg-slate-800 px-2" title="delta" value={cfg.loss.delta} onChange={(e) => set("loss", { ...cfg.loss, delta: Number(e.target.value) })} />}</div>
              </Field>
              <Field label={`Learning rate: ${cfg.lr.toPrecision(2)}`}><input type="range" className="w-full accent-pink-400" min={-5} max={0} step={0.05} value={Math.log10(cfg.lr)} data-testid="lr" onChange={(e) => set("lr", +Math.pow(10, Number(e.target.value)).toPrecision(2))} /></Field>
              <Field label="Batch size"><select className={sel} value={cfg.batch_size} onChange={(e) => set("batch_size", Number(e.target.value))}>{BATCHES.map((b) => <option key={b}>{b}</option>)}</select></Field>
              <Field label="Optimizer"><div className="flex gap-2">{(["sgd", "momentum", "adam"] as const).map((o) => <button key={o} onClick={() => set("optimizer", o)} className={`flex-1 rounded-lg px-3 py-2 text-lg ${cfg.optimizer === o ? "bg-sky-500 text-slate-950" : "bg-slate-800"}`}>{o}</button>)}</div></Field>
            </div>
          </section>

          <details className="rounded-2xl bg-slate-900/70 p-5" data-testid="advanced">
            <summary className="cursor-pointer text-2xl font-semibold text-amber-300">At your own risk: advanced knobs <span className="text-base font-normal text-slate-500">(sensible defaults apply if you leave them alone)</span></summary>
            <div className="mt-4 grid grid-cols-2 gap-4">
              <Field label="Target"><select className={sel} value={cfg.target_type} onChange={(e) => set("target_type", e.target.value as Config["target_type"])}><option value="winprob">win probability</option><option value="cp">clipped centipawns</option></select></Field>
              <Field label={`Squash scale K: ${cfg.eval_squash_scale}`}><input type="range" className="w-full" min={50} max={2000} step={10} value={cfg.eval_squash_scale} onChange={(e) => set("eval_squash_scale", Number(e.target.value))} /></Field>
              <Field label={`Mate clip: ${cfg.mate_clip} cp`}><input type="range" className="w-full" min={300} max={10000} step={100} value={cfg.mate_clip} onChange={(e) => set("mate_clip", Number(e.target.value))} /></Field>
              <Field label="Data slice"><select className={sel} value={cfg.data_slice} onChange={(e) => set("data_slice", e.target.value as Config["data_slice"])}><option value="all">all positions</option><option value="endgame">endgames (≤ 10 pieces)</option><option value="balanced">balanced (no mate, |eval| ≤ 100)</option><option value="decisive">decisive (mate or |eval| ≥ 300)</option></select></Field>
              <Field label="Sampling"><select className={sel} value={cfg.sampling} onChange={(e) => set("sampling", e.target.value as Config["sampling"])}><option value="uniform">uniform</option><option value="weighted">weighted toward big evals</option></select></Field>
              <Field label="Initialisation"><select className={sel} value={cfg.init} onChange={(e) => set("init", e.target.value as Config["init"])}><option value="xavier">xavier</option><option value="he">he</option><option value="small_normal">small normal</option></select></Field>
              <Field label="LR schedule"><select className={sel} value={cfg.lr_schedule} onChange={(e) => set("lr_schedule", e.target.value as Config["lr_schedule"])}><option value="constant">constant</option><option value="cosine">cosine</option><option value="linear">linear decay</option></select></Field>
              <Field label={`Warm-up: ${(cfg.warmup_frac * 100).toFixed(0)}% of training`}><input type="range" className="w-full" min={0} max={0.5} step={0.01} value={cfg.warmup_frac} onChange={(e) => set("warmup_frac", Number(e.target.value))} /></Field>
              <Field label={`Gradient clip: ${cfg.grad_clip || "off"}`}><input type="range" className="w-full" min={0} max={10} step={0.1} value={cfg.grad_clip} onChange={(e) => set("grad_clip", Number(e.target.value))} /></Field>
              <Field label="Normalisation"><select className={sel} value={cfg.normalization} onChange={(e) => set("normalization", e.target.value as Config["normalization"])}><option value="none">none</option><option value="layernorm">layer norm</option></select></Field>
              <Field label="Output head"><select className={sel} value={cfg.output_head} onChange={(e) => set("output_head", e.target.value as Config["output_head"])}><option value="linear">linear</option><option value="tanh">tanh</option><option value="sigmoid">sigmoid</option></select></Field>
              <Field label="Seed"><input type="number" className={input} min={0} value={cfg.seed} onChange={(e) => set("seed", Math.max(0, Math.floor(Number(e.target.value)) || 0))} /></Field>
              <div className="col-span-2 space-y-2 text-lg text-slate-300">
                <div className="font-medium text-slate-200">Extra inputs</div>
                {([["stm_castle", "side to move + castling rights"], ["en_passant", "en passant file"], ["material", "material counts"], ["attacks", "attack maps (which squares each side attacks)"]] as const).map(([k, label]) => (
                  <label key={k} className="flex items-center gap-2"><input type="checkbox" checked={cfg.input_extras[k]} onChange={(e) => set("input_extras", { ...cfg.input_extras, [k]: e.target.checked })} />{label}</label>))}
                <div className="flex flex-wrap gap-x-8 gap-y-2 pt-2">
                  <label className="flex items-center gap-2"><input type="checkbox" checked={cfg.perspective_flip} onChange={(e) => set("perspective_flip", e.target.checked)} />score from the side to move's view</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={cfg.color_flip_augmentation} onChange={(e) => set("color_flip_augmentation", e.target.checked)} />colour-flip augmentation</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={cfg.residual} onChange={(e) => set("residual", e.target.checked)} />residual connections</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={cfg.ema.enabled} onChange={(e) => set("ema", { ...cfg.ema, enabled: e.target.checked })} />EMA of weights</label>
                  {cfg.ema.enabled && <label className="flex items-center gap-2">decay <input type="number" step="0.001" min={0.9} max={0.99999} className="w-28 rounded-lg bg-slate-800 px-2" value={cfg.ema.decay} onChange={(e) => set("ema", { ...cfg.ema, decay: Number(e.target.value) })} /></label>}
                </div>
              </div>
            </div>
            <button className="mt-4 rounded-lg bg-slate-800 px-4 py-2 text-lg hover:bg-slate-700" onClick={() => { setCfg(defaultConfig()); setPerLayer(false); }}>Reset everything to the defaults</button>
          </details>
        </div>

        <aside className="sticky top-4 h-fit space-y-4">
          <div className="rounded-2xl bg-slate-900 p-5 ring-1 ring-slate-700" data-testid="live-panel">
            <div className="text-sm uppercase tracking-wide text-slate-500">your network</div>
            <div className="font-mono text-5xl font-bold text-sky-300" data-testid="param-count">{fmt(server?.valid && server.param_count ? server.param_count : local.paramCount)}</div>
            <div className="text-lg text-slate-400">parameters · {human(local.flopsPerSample)}FLOPs per example</div>
            <div className="mt-3 flex items-center gap-3"><Badge tone={errors.length ? "red" : "sky"}><span data-testid="tier">{tier.name}</span></Badge>
              <span className="text-xl text-slate-200" data-testid="depth-text">your bot searches <b className="text-amber-300">{tier.search_depth_full_moves}</b> move{tier.search_depth_full_moves > 1 ? "s" : ""} ahead</span></div>
            {errors.length > 0 && <ul className="mt-3 list-disc pl-5 text-lg text-red-300" data-testid="errors">{errors.map((e) => <li key={e}>{e}</li>)}</ul>}
            <div className="mt-2 text-sm text-slate-500">{online ? (server ? "checked by the server" : "checking…") : "offline: local estimate"} · limit {fmt(MAX_PARAMS)}</div>
          </div>
          <div className="rounded-2xl bg-slate-900/70 p-4 text-base text-slate-300" data-testid="tier-table">
            <div className="mb-1 text-lg font-semibold text-slate-100">Tiers: how far your bot looks ahead</div>
            <table className="w-full"><tbody>{TIERS.map((t, i) => (<tr key={t.name} className={t.name === tier.name ? "text-sky-300" : ""}><td className="py-0.5">{t.name}</td><td>{i === 0 ? "up to" : "up to"} {human(t.max_params)}</td><td className="text-right">{t.search_depth_full_moves} move{t.search_depth_full_moves > 1 ? "s" : ""}</td></tr>))}</tbody></table>
            <div className="mt-2 text-sm text-slate-500">(one move = your move and the reply)</div>
          </div>
          <button disabled={!valid || busy} onClick={submit} className="w-full rounded-2xl bg-emerald-500 py-4 text-3xl font-bold text-slate-950 hover:bg-emerald-400" data-testid="submit">{busy ? "Sending…" : "Enter the tournament"}</button>
          {!valid && <p className="text-base text-slate-500">Need: nickname, model name, the consent tick, and a valid network.</p>}
          {error && <p className="rounded-xl bg-amber-500/15 p-3 text-lg text-amber-200" data-testid="submit-error">{error}</p>}
        </aside>
      </div>

      {subs.length > 0 && (
        <section className="mt-8 rounded-2xl bg-slate-900/70 p-5" data-testid="my-entries">
          <div className="flex items-center justify-between"><h2 className="text-2xl font-semibold">Your entries</h2><button onClick={download} className="rounded-lg bg-slate-800 px-4 py-2 text-lg hover:bg-slate-700" data-testid="backup">Download backup</button></div>
          <div className="mt-3 space-y-3">{subs.slice().reverse().map((s) => {
            const st = statuses[s.client_id];
            return (
              <div key={s.client_id} className="rounded-xl bg-slate-950/60 p-4" data-testid="entry">
                <div className="flex flex-wrap items-center gap-4">
                  <span className="text-xl font-semibold">{s.model_name}</span><span className="text-slate-400">by {s.nickname}</span>
                  {s.status === "pending" ? <Badge tone="amber">saved on this device: waiting for the server</Badge> : <span className="font-mono text-3xl font-bold text-amber-300" data-testid="code">{s.participant_code}</span>}
                  {st && <Badge tone={st.status === "done" ? "green" : st.status === "error" || st.status === "killed" ? "red" : "sky"}>{st.status}{st.queue_position ? ` · place ${st.queue_position}` : ""}</Badge>}
                  {st?.stop_reason && <span className="text-slate-400">stopped: {st.stop_reason}</span>}
                </div>
                {st?.progress && st.status === "running" && <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-800"><div className="h-full bg-sky-400" style={{ width: `${st.progress.fraction * 100}%` }} /></div>}
                {st?.curves && st.curves.val.length > 1 && <div className="mt-2"><LineChart width={520} height={130} series={[{ color: "#56B4E9", points: st.curves.train.map((p) => ({ x: p.t, y: p.loss })) }, { color: "#E69F00", points: st.curves.val.map((p) => ({ x: p.t, y: p.loss })) }]} title="loss: training (blue), validation (orange)" xLabel="seconds" /></div>}
                {st?.val_mse != null && <div className="mt-1 text-lg text-slate-300">validation error (comparable): <b>{st.val_mse.toFixed(4)}</b></div>}
                <div className="mt-1 text-sm text-slate-500">{human(s.param_count ?? previewLocal(s.config).paramCount)} parameters · tier {s.tier ?? previewLocal(s.config).tier.name}</div>
              </div>);
          })}</div>
        </section>
      )}
      <p className="mt-6 text-sm text-slate-600">Defaults come from the server's own config ({Object.keys(DEFAULTS).length} knobs).</p>
    </main>
  );
}
