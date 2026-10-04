// Performance measurement helpers shared by e2e/perf-probe.mjs (manual numbers) and e2e/perf-regression.mjs (the pass/fail gate).
// Everything is measured in the page itself or through the Chrome DevTools Protocol, with an optional CPU throttle (4x = a weak laptop).
import { chromium } from "playwright";

/** Installed before any page script runs: collects long tasks, frame times and per-key latency. */
const INIT = () => {
  const w = window;
  w.__perf = { longtasks: [], frames: [], keys: [], events: [] };
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) w.__perf.longtasks.push({ t: e.startTime, d: e.duration }); }).observe({ type: "longtask", buffered: true }); } catch { /* unsupported */ }
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) if (e.name === "keydown") w.__perf.events.push({ t: e.startTime, d: e.duration, p: e.processingEnd - e.processingStart }); }).observe({ type: "event", durationThreshold: 16, buffered: true }); } catch { /* unsupported */ }
  let last = performance.now();
  const frame = (t) => { w.__perf.frames.push(t - last); last = t; requestAnimationFrame(frame); };
  requestAnimationFrame(frame);
  // key -> next paint: capture phase so it runs before the app's handlers; two rAFs ~ the frame after the handlers ran.
  window.addEventListener("keydown", (e) => {
    const t0 = performance.now();
    requestAnimationFrame(() => requestAnimationFrame(() => { w.__perf.keys.push({ key: e.key, ms: performance.now() - t0 }); }));
  }, true);
};

export async function launch({ headed = false, softwareGL = false, executablePath = process.env.CHROMIUM || undefined, viewport = { width: 1920, height: 1080 } } = {}) {
  const args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--enable-precise-memory-info", "--js-flags=--expose-gc"];
  if (softwareGL) args.push("--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist");
  const browser = await chromium.launch({ executablePath, headless: !headed, args });
  const ctx = await browser.newContext({ viewport });
  const page = await ctx.newPage();
  await page.addInitScript(INIT);
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Performance.enable");
  const errors = [];
  page.on("console", (m) => m.type() === "error" && !/Failed to load resource/.test(m.text()) && errors.push(m.text())); // (a missing backend is a normal state: its failed requests are not errors)
  page.on("pageerror", (e) => errors.push("PAGEERROR " + e.message));
  return { browser, ctx, page, cdp, errors };
}

export const throttle = (cdp, rate) => cdp.send("Emulation.setCPUThrottlingRate", { rate });

/** page.evaluate that gives up after `ms` (a frozen page never answers). */
export const ev = (page, fn, arg, ms = 20000) => Promise.race([page.evaluate(fn, arg), new Promise((r) => setTimeout(() => r(null), ms))]);

async function metrics(cdp) {
  const { metrics: m } = await cdp.send("Performance.getMetrics");
  const o = Object.fromEntries(m.map((x) => [x.name, x.value]));
  return { task: o.TaskDuration, script: o.ScriptDuration, layout: o.LayoutDuration, style: o.RecalcStyleDuration, heap: o.JSHeapUsedSize, nodes: o.Nodes, listeners: o.JSEventListeners, ts: o.Timestamp };
}

/** Collect for `ms` of wall time while `during()` runs (default: idle). Returns rates over the window. */
export async function window_(page, cdp, ms, during) {
  const m0 = await metrics(cdp);
  const p0 = (await ev(page, () => ({ lt: __perf.longtasks.length, fr: __perf.frames.length, now: performance.now() }))) ?? { lt: 0, fr: 0, now: 0 };
  if (during) await during(); else await page.waitForTimeout(ms);
  const m1 = await metrics(cdp);
  const p1 = (await ev(page, (a) => {
    const lt = __perf.longtasks.slice(a.lt), fr = __perf.frames.slice(a.fr).sort((x, y) => x - y);
    const total = fr.reduce((s, x) => s + x, 0);
    return { lt, n: fr.length, now: performance.now(), p50: fr[Math.floor(fr.length * 0.5)] ?? 0, p95: fr[Math.floor(fr.length * 0.95)] ?? 0, max: fr[fr.length - 1] ?? 0, total };
  }, p0)) ?? { lt: [], n: 0, now: p0.now + 1, p50: 0, p95: 0, max: 0, frozen: true };
  const wall = m1.ts - m0.ts;
  return {
    wall_s: +wall.toFixed(2),
    cpu_pct: +(((m1.task - m0.task) / wall) * 100).toFixed(1),
    script_pct: +(((m1.script - m0.script) / wall) * 100).toFixed(1),
    layout_pct: +(((m1.layout - m0.layout + m1.style - m0.style) / wall) * 100).toFixed(1),
    frozen: !!p1.frozen, fps: +((p1.n / Math.max(1, p1.now - p0.now)) * 1000).toFixed(1),
    frame_p50_ms: +p1.p50.toFixed(1), frame_p95_ms: +p1.p95.toFixed(1), frame_max_ms: +p1.max.toFixed(1),
    longtasks: p1.lt.length, longtask_ms: +p1.lt.reduce((s, x) => s + x.d, 0).toFixed(0), longtask_max_ms: +Math.max(0, ...p1.lt.map((x) => x.d)).toFixed(0),
    heap_mb: +(m1.heap / 1048576).toFixed(1), heap_delta_mb: +((m1.heap - m0.heap) / 1048576).toFixed(1),
    listeners: m1.listeners, nodes: m1.nodes,
  };
}

export async function gc(cdp) { try { await cdp.send("HeapProfiler.collectGarbage"); } catch { /* ignore */ } }
export async function heapMB(cdp) { await gc(cdp); return (await metrics(cdp)).heap / 1048576; }

/** Press a key, return after the app had two frames to render (the page measures the real latency). */
export async function press(page, key) { await page.keyboard.press(key); await page.waitForTimeout(0); }

export async function keyLatencies(page) {
  return (await ev(page, () => ({ keys: __perf.keys.slice(), events: __perf.events.slice() }))) ?? { keys: [], events: [] };
}
export const pct = (arr, p) => { const a = [...arr].sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.floor(a.length * p))] : 0; };
