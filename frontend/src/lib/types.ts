// Types mirror backend/app/schemas.py (SPEC 7).
import type { Config } from "./chessConfig";
import type { LaneCfg, RaceKind } from "./raceGrid";

export interface Health { ok: boolean; device: "cuda" | "cpu"; gpu_name: string | null; vram_gb: number | null; queue_length: number; demo_mode: boolean; paused: boolean; version: string }
export interface RaceLaneTick { id: string; step: number; samples_seen: number; updates: number; loss: number | null; acc: number; probe_preds: number[] }
export type RaceEvent =
  | { event: "tick"; data: { t: number; lanes: RaceLaneTick[] } }
  | { event: "done"; data: { reason: "time" | "complete" | "aborted"; final: RaceLaneTick[] } }
  | { event: "error"; data: { message: string } };
export interface RaceRequest { kind: RaceKind; lanes: { id: string; config: Partial<LaneCfg> }[]; max_seconds: number }
export interface Preview { valid: boolean; errors: string[]; param_count?: number | null; tier?: string | null; search_depth_full_moves?: number | null; est_flops_budget?: number | null; flops_per_sample?: number | null }
export interface SubmissionIn { nickname: string; model_name: string; contact?: string; consent: boolean; config: Config; client_id?: string }
export interface SubmissionOut { submission_id: string; participant_code: string; param_count: number; tier: string; status: "queued"; queue_position: number | null }
export interface Progress { fraction: number; samples_seen: number; flops_used: number; flops_budget: number; active_gpu_seconds: number; preemptions: number }
export interface CurvePoint { t: number; step: number; samples_seen?: number; loss: number; val_mse?: number }
export interface SubmissionStatus {
  status: "queued" | "running" | "done" | "killed" | "removed" | "error";
  participant_code: string; nickname: string; model_name: string; tier: string; param_count: number;
  queue_position?: number | null; progress?: Progress | null; stop_reason?: "budget" | "time" | "diverged" | "killed" | "error" | null;
  curves?: { train: CurvePoint[]; val: CurvePoint[] } | null; val_mse?: number | null;
}
export interface AdminJob {
  id: number; kind: string; status: string; nickname: string | null; model_name: string | null; participant_code: string | null; tier: string | null;
  param_count: number | null; config_summary: string | null; priority: number; stop_reason: string | null; progress: number | null; concurrency?: number | null; samples_per_s?: number | null;
  samples_seen: number; flops_used: number; active_gpu_seconds: number; preemptions: number; error_message: string | null;
  created_at: number; started_at: number | null; finished_at: number | null;
}
export interface AdminQueue { paused: boolean; demo_mode: boolean; current_job_id: number | null; budget_flops: number; jobs: AdminJob[] }
export interface AdminStatus {
  t: number; paused: boolean; demo_mode: boolean; current_job_id: number | null; kind: string | null; progress: number | null;
  curves: { train: CurvePoint[]; val: CurvePoint[] } | null;
  gpu: { device: string; utilization: number | null; memory_used_gb: number | null; memory_total_gb: number | null };
  log_tail: string[];
}
export interface PresenterState { stage_id: string; stage_index: number; title: string; updated_at: number }
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}
