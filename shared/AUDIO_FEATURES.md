# Audio feature spec (log-mel)

One definition, three implementations that must agree: `backend/app/data/audio_features.py` (numpy, and a
torch twin for batched training) and `frontend/src/lib/audio.ts` (TypeScript, browser).
Golden fixtures: `shared/golden/audio.json` (built from `data/samples/speech/samples/*.wav`).
Tolerance: max absolute difference **<= 1e-3** on the final normalized log-mel values.

## Input
Mono audio at **16 kHz**. Integer clips are converted with `x = int16 / 32768.0`. The clip is exactly
**16000 samples** (1 s): shorter clips are zero-padded at the end, longer are truncated (the data already is).

## Pipeline
1. **Framing**: frame length **400** samples (25 ms), hop **160** samples (10 ms), no centering and no padding.
   Number of frames `T = 1 + floor((16000 - 400) / 160) = 98`.
2. **Window**: periodic Hann, `w[n] = 0.5 - 0.5 * cos(2*pi*n / 400)`, `n = 0..399`.
3. **FFT**: zero-pad each windowed frame to **512** samples, real FFT, power spectrum `|X[k]|^2`, `k = 0..256` (257 bins).
4. **Mel filterbank**: **40** triangular filters, HTK mel scale `mel(f) = 2595 * log10(1 + f/700)`,
   filter edges equally spaced in mel between **20 Hz and 7600 Hz** (42 edge points). FFT bin `k` has frequency
   `k * 16000 / 512`. Filter `m` (edges `lo, ce, hi` in Hz) has weight at frequency `f`:
   `(f - lo) / (ce - lo)` for `lo <= f <= ce`, `(hi - f) / (hi - ce)` for `ce < f <= hi`, else 0. No area normalization.
   `mel_energy[m, t] = sum_k fbank[m, k] * power[t, k]`.
5. **Log**: `L = ln(mel_energy + 1e-6)` (natural log, additive floor `1e-6`).
6. **Normalization**: per clip, `out = (L - mean(L)) / (std(L) + 1e-5)`, mean and (population) std over all
   40 x 98 values. No dataset constants.

## Output
`float32[40][98]` row-major: **mel bin (height) x frame (width)**, so a spectrogram image has frequency
on the vertical axis and time on the horizontal axis. Model input is the flattened or `1 x 40 x 98` array.

## Raw-waveform model input (the "flailing" baseline)
`x = int16 / 32768.0` (16000 values), average-pooled by 4 to **4000** values (so the browser model stays small).
No spectral processing of any kind. See `backend/scripts/train_showcase.py`.

## Constants
`SR=16000, N_SAMPLES=16000, WIN=400, HOP=160, NFFT=512, N_MELS=40, FMIN=20, FMAX=7600, LOG_FLOOR=1e-6, NORM_EPS=1e-5, N_FRAMES=98`.
