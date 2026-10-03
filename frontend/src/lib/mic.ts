// Live microphone, push-to-talk (SPEC 5.3 Act 3.1). Audio never leaves the browser and is never stored.
import { N_SAMPLES, SR } from "./audio";

export interface MicHandle { start(): Promise<void>; stop(): Float32Array | null; close(): void }

/** Box-filter downsample from `srcRate` to 16 kHz. */
export function resampleTo16k(x: Float32Array, srcRate: number): Float32Array {
  if (srcRate === SR) return x;
  const ratio = srcRate / SR, n = Math.floor(x.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let k = a; k < b; k++) s += x[k];
    out[i] = s / (b - a);
  }
  return out;
}

/** Pick the 1-second window with the most energy (the word), padding short recordings; peak-normalise very quiet input. */
export function pickWindow(x: Float32Array): Float32Array {
  if (x.length <= N_SAMPLES) { const out = new Float32Array(N_SAMPLES); out.set(x, Math.floor((N_SAMPLES - x.length) / 2)); return normalise(out); }
  let best = 0, bestE = -1;
  const hop = 800;
  for (let s = 0; s + N_SAMPLES <= x.length; s += hop) {
    let e = 0;
    for (let i = s; i < s + N_SAMPLES; i += 4) e += x[i] * x[i];
    if (e > bestE) { bestE = e; best = s; }
  }
  return normalise(x.slice(best, best + N_SAMPLES));
}

function normalise(x: Float32Array): Float32Array {
  let peak = 0;
  for (const v of x) peak = Math.max(peak, Math.abs(v));
  if (peak > 1e-4 && peak < 0.3) { const g = 0.5 / peak; for (let i = 0; i < x.length; i++) x[i] *= g; }
  return x;
}

export function createMic(): MicHandle {
  let stream: MediaStream | null = null, ctx: AudioContext | null = null, node: ScriptProcessorNode | null = null, src: MediaStreamAudioSourceNode | null = null;
  let chunks: Float32Array[] = [], recording = false;
  return {
    async start() {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("this browser has no microphone access (needs https or localhost)");
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        ctx = new AudioContext();
        src = ctx.createMediaStreamSource(stream);
        node = ctx.createScriptProcessor(4096, 1, 1);
        node.onaudioprocess = (e) => { if (recording) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
        src.connect(node); node.connect(ctx.destination);
      }
      if (ctx!.state === "suspended") await ctx!.resume();
      chunks = []; recording = true;
    },
    stop() {
      recording = false;
      if (!chunks.length || !ctx) return null;
      const total = chunks.reduce((n, c) => n + c.length, 0), all = new Float32Array(total);
      let o = 0;
      for (const c of chunks) { all.set(c, o); o += c.length; }
      chunks = [];
      return pickWindow(resampleTo16k(all, ctx.sampleRate));
    },
    close() { recording = false; node?.disconnect(); src?.disconnect(); stream?.getTracks().forEach((t) => t.stop()); ctx?.close().catch(() => undefined); stream = null; ctx = null; },
  };
}
