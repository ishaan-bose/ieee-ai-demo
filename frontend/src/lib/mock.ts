// A complete mock of the backend for `?mock=1`: every stage can be checked by eye with no server.
import { previewLocal, defaultConfig, type Config } from "./chessConfig";
import { isMockOffline } from "./env";
import { ApiError, type AdminJob, type AdminQueue, type AdminStatus, type Health, type PresenterState, type Preview, type RaceEvent, type RaceRequest, type SubmissionIn, type SubmissionOut, type SubmissionStatus } from "./types";
import { laneCfg } from "./raceGrid";
import { loadProbeLabels } from "./doodle";
import { simulateLane } from "./raceSim";

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((res, rej) => {
  const t = setTimeout(res, ms);
  signal?.addEventListener("abort", () => { clearTimeout(t); rej(new DOMException("aborted", "AbortError")); });
});
const offlineCheck = () => { if (isMockOffline()) throw new ApiError(0, "offline", "mock server is switched off (press O)"); };

interface MockJob extends AdminJob { config?: Config; submission_id?: string; client_id?: string; startAt: number }
const jobs: MockJob[] = [];
let nextId = 1;
let paused = false, demoMode = false;
const races = new Map<string, { aborted: boolean }>();
let presenter: PresenterState = { stage_id: "", stage_index: 0, title: "", updated_at: 0 };

let seeded = false;
function seedJobs() {
  if (seeded) return;
  seeded = true;
  for (const [n, m, tier, params] of [["ada", "Neural Nelly", "Light", 330_000], ["linus", "BigBrain", "Heavy", 21_000_000], ["grace", "Compiler", "Medium", 4_000_000]] as const)
    addJob(n, m, tier, params, nextId <= 1 ? "running" : "queued");
}
function addJob(nick: string, model: string, tier: string, params: number, status: string, config?: Config, client_id?: string): MockJob {
  const id = nextId++;
  const j: MockJob = { id, kind: "competition", status, nickname: nick, model_name: model, participant_code: "AI-" + Math.random().toString(36).slice(2, 7).toUpperCase(), tier, param_count: params,
    config_summary: "3x256 relu mse adam lr=0.001 bs=256", priority: 0, stop_reason: null, progress: status === "running" ? 0.3 : 0, samples_seen: 0, flops_used: 0, active_gpu_seconds: 0, preemptions: 0,
    error_message: null, created_at: Date.now() / 1000, started_at: status === "running" ? Date.now() / 1000 : null, finished_at: null, config, submission_id: `mock${id}`, client_id, startAt: Date.now() };
  jobs.push(j);
  return j;
}
function tickJobs() {
  const running = jobs.find((j) => j.status === "running");
  if (running && !paused && !demoMode) {
    running.progress = Math.min(1, (running.progress ?? 0) + 0.02);
    running.active_gpu_seconds += 1;
    if (running.progress >= 1) { running.status = "done"; running.stop_reason = "budget"; running.finished_at = Date.now() / 1000; }
  }
  if (!jobs.some((j) => j.status === "running") && !paused && !demoMode) {
    const next = jobs.find((j) => j.status === "queued");
    if (next) { next.status = "running"; next.started_at = Date.now() / 1000; }
  }
}
if (typeof window !== "undefined") setInterval(() => { if (jobs.length) tickJobs(); }, 1000);

export const mockApi = {
  async health(): Promise<Health> {
    offlineCheck(); await sleep(60);
    seedJobs();
    return { ok: true, device: "cuda", gpu_name: "Mock L40S (no real GPU)", vram_gb: 48, queue_length: jobs.filter((j) => ["queued", "running"].includes(j.status)).length, demo_mode: demoMode, paused, version: "mock" };
  },
  async createRace(body: RaceRequest): Promise<{ race_id: string }> {
    offlineCheck(); await sleep(80);
    const id = Math.random().toString(16).slice(2, 14);
    races.set(id, { aborted: false });
    mockRaceRequests.set(id, body);
    return { race_id: id };
  },
  async *raceStream(id: string, signal?: AbortSignal): AsyncGenerator<RaceEvent> {
    offlineCheck();
    const body = mockRaceRequests.get(id);
    const state = races.get(id);
    if (!body || !state) throw new ApiError(404, "not_found", "unknown race");
    const probe = await loadProbeLabels();
    const sims = body.lanes.map((l) => ({ id: l.id, ticks: simulateLane(laneCfg(l.id, l.config), body.max_seconds, probe) }));
    const n = sims[0].ticks.length;
    for (let i = 0; i < n; i++) {
      await sleep(500, signal);
      if (isMockOffline()) throw new ApiError(0, "offline", "connection lost");
      if (state.aborted) { yield { event: "done", data: { reason: "aborted", final: [] } }; return; }
      yield { event: "tick", data: { t: sims[0].ticks[i].t, lanes: sims.map((s) => ({ id: s.id, ...s.ticks[i] })) } };
    }
    yield { event: "done", data: { reason: "time", final: sims.map((s) => ({ id: s.id, ...s.ticks[n - 1] })) } };
  },
  async abortRace(id: string) { offlineCheck(); const r = races.get(id); if (r) r.aborted = true; return { ok: true }; },
  async preview(cfg: Config): Promise<Preview> {
    offlineCheck(); await sleep(80);
    const p = previewLocal(cfg);
    return { valid: p.errors.length === 0, errors: p.errors, param_count: p.paramCount, tier: p.tier.name, search_depth_full_moves: p.tier.search_depth_full_moves, est_flops_budget: 1e16, flops_per_sample: p.flopsPerSample };
  },
  async submit(body: SubmissionIn): Promise<SubmissionOut> {
    offlineCheck(); await sleep(150);
    seedJobs();
    const dup = body.client_id ? jobs.find((j) => j.client_id === body.client_id) : undefined;
    const p = previewLocal(body.config);
    if (p.errors.length) throw new ApiError(422, "invalid_config", p.errors.join("; "));
    const j = dup ?? addJob(body.nickname, body.model_name, p.tier.name, p.paramCount, "queued", body.config, body.client_id);
    return { submission_id: j.submission_id!, participant_code: j.participant_code!, param_count: j.param_count!, tier: j.tier!, status: "queued", queue_position: jobs.filter((x) => x.status === "queued" && x.id <= j.id).length };
  },
  async submission(id: string): Promise<SubmissionStatus> {
    offlineCheck(); await sleep(40);
    const j = jobs.find((x) => x.submission_id === id);
    if (!j) throw new ApiError(404, "not_found", "unknown submission");
    const fake = Array.from({ length: Math.round((j.progress ?? 0) * 40) }, (_, i) => ({ t: i * 5, step: i * 500, loss: 0.9 / (1 + i * 0.15) + 0.1, val_mse: 0.25 / (1 + i * 0.08) + 0.04 }));
    return { status: j.status as SubmissionStatus["status"], participant_code: j.participant_code!, nickname: j.nickname!, model_name: j.model_name!, tier: j.tier!, param_count: j.param_count!,
      queue_position: j.status === "queued" ? jobs.filter((x) => x.status === "queued" && x.id <= j.id).length : null,
      progress: j.status === "queued" ? null : { fraction: j.progress ?? 0, samples_seen: 0, flops_used: 0, flops_budget: 1e16, active_gpu_seconds: j.active_gpu_seconds, preemptions: j.preemptions },
      stop_reason: j.stop_reason as SubmissionStatus["stop_reason"], curves: { train: fake, val: fake }, val_mse: j.status === "done" ? 0.05 : null };
  },
  async presenterGet(): Promise<PresenterState> { offlineCheck(); return presenter; },
  async presenterPut(s: Omit<PresenterState, "updated_at">) { offlineCheck(); presenter = { ...s, updated_at: Date.now() / 1000 }; return presenter; },
  async configSchema() { return { defaults: defaultConfig() }; },
  admin: {
    async queue(): Promise<AdminQueue> {
      offlineCheck(); seedJobs();
      const cur = jobs.find((j) => j.status === "running");
      return { paused, demo_mode: demoMode, current_job_id: cur?.id ?? null, budget_flops: 1e16, jobs: jobs.filter((j) => j.status !== "removed").map(({ config: _c, startAt: _s, submission_id: _i, client_id: _x, ...rest }) => rest) };
    },
    async kill(id: number) { const j = jobs.find((x) => x.id === id); if (j) { j.status = "killed"; j.stop_reason = "killed"; } return { ok: true, status: "killed" }; },
    async remove(id: number) { const j = jobs.find((x) => x.id === id); if (j) j.status = "removed"; return { ok: true, status: "removed" }; },
    async redo(id: number) { const j = jobs.find((x) => x.id === id); if (j) { j.status = "queued"; j.progress = 0; j.stop_reason = null; } return { ok: true, status: "queued" }; },
    async pause() { paused = true; return { ok: true }; },
    async resume() { paused = false; return { ok: true }; },
    async demoMode(enabled: boolean) { demoMode = enabled; return { ok: true }; },
    async reset() { jobs.length = 0; return { ok: true, snapshot: "(mock) /state/snapshots/demo-mock.db" }; },
    async exportAll(): Promise<Blob> { return new Blob([JSON.stringify({ mock: true, submissions: jobs.map((j) => ({ nickname: j.nickname, tier: j.tier })) }, null, 2)], { type: "application/json" }); },
    async *stream(signal?: AbortSignal): AsyncGenerator<AdminStatus> {
      for (;;) {
        await sleep(1000, signal);
        const cur = jobs.find((j) => j.status === "running");
        yield { t: Date.now() / 1000, paused, demo_mode: demoMode, current_job_id: cur?.id ?? null, kind: cur ? "competition" : null, progress: cur?.progress ?? null,
          curves: cur ? { train: Array.from({ length: 30 }, (_, i) => ({ t: i * 3, step: i, loss: 0.8 / (1 + i * 0.2) + Math.random() * 0.02 })), val: Array.from({ length: 30 }, (_, i) => ({ t: i * 3, step: i, loss: 0.9 / (1 + i * 0.18) })) } : null,
          gpu: { device: "mock", utilization: cur ? 60 + Math.round(Math.random() * 35) : 3, memory_used_gb: 11.2, memory_total_gb: 48 },
          log_tail: ["[mock] scheduler: job running", "[mock] checkpoint saved"] };
      }
    },
  },
};
const mockRaceRequests = new Map<string, RaceRequest>();
