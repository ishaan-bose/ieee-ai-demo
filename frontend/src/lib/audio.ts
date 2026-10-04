// Log-mel features. Spec: shared/AUDIO_FEATURES.md (Python twin: backend/app/data/audio_features.py).
import { fft } from "./fft";

export const SR = 16000;
export const N_SAMPLES = 16000;
export const WIN = 400;
export const HOP = 160;
export const NFFT = 512;
export const N_MELS = 40;
export const FMIN = 20;
export const FMAX = 7600;
export const LOG_FLOOR = 1e-6;
export const NORM_EPS = 1e-5;
export const N_FRAMES = 1 + Math.floor((N_SAMPLES - WIN) / HOP); // 98
export const RAW_POOL = 4;
export const RAW_LEN = N_SAMPLES / RAW_POOL;

const mel = (f: number) => 2595 * Math.log10(1 + f / 700);
const invMel = (m: number) => 700 * (Math.pow(10, m / 2595) - 1);

export function hannWindow(): Float64Array {
  const w = new Float64Array(WIN);
  for (let n = 0; n < WIN; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / WIN);
  return w;
}

export function melFilterbank(): Float64Array[] {
  const edges: number[] = [];
  const lo = mel(FMIN), hi = mel(FMAX);
  for (let i = 0; i < N_MELS + 2; i++) edges.push(invMel(lo + ((hi - lo) * i) / (N_MELS + 1)));
  const bins = NFFT / 2 + 1;
  const fb: Float64Array[] = [];
  for (let m = 0; m < N_MELS; m++) {
    const row = new Float64Array(bins);
    const [l, c, h] = [edges[m], edges[m + 1], edges[m + 2]];
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      if (f >= l && f <= c) row[k] = (f - l) / (c - l);
      else if (f > c && f <= h) row[k] = (h - f) / (h - c);
    }
    fb.push(row);
  }
  return fb;
}

const WINDOW = hannWindow();
const FB = melFilterbank();

export function padOrTrim(x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(N_SAMPLES);
  for (let i = 0; i < Math.min(N_SAMPLES, x.length); i++) out[i] = x[i];
  return out;
}

/** Power spectrum of one frame t (257 bins), exposed for the step-by-step visualisation. */
export function framePower(x: Float64Array, t: number): Float64Array {
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  for (let n = 0; n < WIN; n++) re[n] = x[t * HOP + n] * WINDOW[n];
  fft(re, im);
  const p = new Float64Array(NFFT / 2 + 1);
  for (let k = 0; k < p.length; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

/** Float samples in [-1,1] (use int16/32768 first). Returns Float32Array(40*98), row-major [mel][frame]. */
export function logMel(samples: ArrayLike<number>): Float32Array {
  const x = padOrTrim(samples);
  const logm = new Float64Array(N_MELS * N_FRAMES);
  for (let t = 0; t < N_FRAMES; t++) {
    const p = framePower(x, t);
    for (let m = 0; m < N_MELS; m++) {
      let e = 0;
      const row = FB[m];
      for (let k = 0; k < p.length; k++) e += row[k] * p[k];
      logm[m * N_FRAMES + t] = Math.log(e + LOG_FLOOR);
    }
  }
  let mean = 0;
  for (const v of logm) mean += v;
  mean /= logm.length;
  let varsum = 0;
  for (const v of logm) varsum += (v - mean) * (v - mean);
  const std = Math.sqrt(varsum / logm.length);
  const out = new Float32Array(logm.length);
  for (let i = 0; i < out.length; i++) out[i] = (logm[i] - mean) / (std + NORM_EPS);
  return out;
}

/** Raw-waveform model input: average-pooled by 4 -> 4000 floats. */
export function rawInput(samples: ArrayLike<number>): Float32Array {
  const x = padOrTrim(samples);
  const out = new Float32Array(RAW_LEN);
  for (let i = 0; i < RAW_LEN; i++) {
    let s = 0;
    for (let j = 0; j < RAW_POOL; j++) s += x[i * RAW_POOL + j];
    out[i] = s / RAW_POOL;
  }
  return out;
}

/** Decode a 16-bit PCM mono WAV file into floats in [-1,1). */
export function decodeWav(buf: ArrayBuffer): Float32Array {
  const v = new DataView(buf);
  let pos = 12;
  let dataStart = -1, dataLen = 0;
  while (pos + 8 <= v.byteLength) {
    const id = String.fromCharCode(v.getUint8(pos), v.getUint8(pos + 1), v.getUint8(pos + 2), v.getUint8(pos + 3));
    const len = v.getUint32(pos + 4, true);
    if (id === "data") { dataStart = pos + 8; dataLen = Math.min(len, v.byteLength - dataStart); break; }
    pos += 8 + len + (len & 1);
  }
  if (dataStart < 0) throw new Error("no data chunk in WAV");
  const n = dataLen >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = v.getInt16(dataStart + 2 * i, true) / 32768;
  return out;
}
