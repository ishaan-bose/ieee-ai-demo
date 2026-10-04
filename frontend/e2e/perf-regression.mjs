// Performance regression gate. Fails (exit 1) when the demo is slow or leaks, under a 4x CPU throttle at 1920x1080:
//   - any single -> / <- key press takes more than 200 ms to render (key down -> the frame after the handlers ran)
//   - the first screen burns more than 5% of the main thread while idle (nobody touching it)
//   - the JS heap grows by more than 50 MB over 30 key presses
//   - (scan) any stage burns more than 5% of the main thread while idle. Exception: the 3D valley (a2-backward) renders WebGL continuously by design;
//     it only has to stay under 70% here, because headless Chromium draws WebGL in SOFTWARE (a real GPU costs far less).
//   A failed variant is retried once (a slow shared machine can spike a single run); the freeze this guards against fails every attempt.
// Run it against the dev server and the production build:
//   npm run dev                                   then   BASE=http://127.0.0.1:5173 npm run e2e:perf
//   npm run build && npx vite preview             then   BASE=http://127.0.0.1:4173 npm run e2e:perf
// Variants: the plain URL (no backend running) and ?mock=1. Options: --headed (needs a display, e.g. xvfb-run -a), --software-gl, --throttle N, --presses N.
// This test exists because the unit tests and the 1x headless walkthrough did NOT notice an infinite render loop on the first screen.
import { heapMB, keyLatencies, launch, pct, throttle, window_ } from "./perf-lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); if (i < 0) return d; const v = process.argv[i + 1]; return v === undefined || v.startsWith("--") ? true : v; };
const BASE = process.env.BASE ?? "http://127.0.0.1:5173";
const RATE = Number(arg("throttle", 4)), PRESSES = Number(arg("presses", 30));
const headed = arg("headed", false) === true, softwareGL = arg("software-gl", false) === true;
const LIMITS = { keyMs: 200, idleCpuPct: 5, heapMB: 50 };
const SCAN = arg("no-scan", false) !== true;
const VARIANTS = [{ name: "no backend", url: "/" }, { name: "?mock=1", url: "/?mock=1" }];
const STAGES = 8; // the first 8 stages: duel, line, knobs, spirals, depth, fuse, relu, gallery

let failed = false;
const rows = [];

async function runVariant(v) {
  const { browser, page, cdp, errors } = await launch({ headed, softwareGL });
  try {
    await throttle(cdp, RATE);
    await page.goto(BASE + v.url, { waitUntil: "load" });
    await page.getByTestId("stage-title").waitFor({ timeout: 60000 });
    await page.waitForTimeout(3000);
    const idle = await window_(page, cdp, 5000);
    const heap0 = await heapMB(cdp);
    for (let i = 0; i < PRESSES; i++) {
      const k = i % (2 * (STAGES - 1)) < STAGES - 1 ? "ArrowRight" : "ArrowLeft";
      await Promise.race([page.keyboard.press(k), new Promise((r) => setTimeout(r, 30000))]);
      await page.waitForTimeout(350);
    }
    await page.waitForTimeout(1500);
    const { keys } = await keyLatencies(page);
    const ms = keys.filter((k) => k.key.startsWith("Arrow")).map((k) => k.ms);
    const heapGrowth = (await heapMB(cdp)) - heap0;
    const maxKey = Math.max(0, ...ms);
    const row = { variant: v.name, throttle: RATE, presses_seen: ms.length, key_max_ms: Math.round(maxKey), key_p50_ms: Math.round(pct(ms, 0.5)), idle_cpu_pct: idle.cpu_pct, idle_fps: idle.fps, longtasks_idle: idle.longtasks, heap_growth_mb: +heapGrowth.toFixed(1), errors: errors.length };
    const bad = [];
    if (ms.length < PRESSES) bad.push(`only ${ms.length}/${PRESSES} key presses were rendered (page froze?)`);
    if (maxKey > LIMITS.keyMs) bad.push(`slowest key press ${Math.round(maxKey)} ms > ${LIMITS.keyMs} ms`);
    if (idle.frozen || idle.cpu_pct > LIMITS.idleCpuPct) bad.push(`idle CPU on the first screen ${idle.cpu_pct}% > ${LIMITS.idleCpuPct}%`);
    if (heapGrowth > LIMITS.heapMB) bad.push(`heap grew ${heapGrowth.toFixed(1)} MB > ${LIMITS.heapMB} MB`);
    if (errors.length) bad.push(`console errors: ${errors[0]}`);
    return { row, bad };
  } finally { await browser.close(); }
}

for (const v of VARIANTS) {
  let res;
  for (let attempt = 1; attempt <= 2; attempt++) {
    res = await runVariant(v);
    if (!res.bad.length) break;
    if (attempt === 1) console.log(`retry ${v.name}: ${res.bad.join("; ")}`);
  }
  rows.push(res.row);
  console.log(`${res.bad.length ? "FAIL" : "ok  "} ${v.name}: ${JSON.stringify(res.row)}`);
  for (const b of res.bad) console.log("      " + b);
  if (res.bad.length) failed = true;
}

if (SCAN) {
  // Idle CPU of EVERY stage (a never-ending animation loop on a later stage would otherwise go unnoticed): open each directly, leave it alone.
  const ids = ["a0-duel", "a1-clusters", "a1-knobs", "a1-spirals", "a1-depth", "a1-fuse", "a1-relu", "a1-gallery", "a2-forward", "a2-loss", "a2-backward", "a2-lr", "a2-batch", "a3-preprocess", "a3-trap", "a3-overfit", "a4-tech", "a4-rematch", "a4-close"];
  const { browser, page, cdp } = await launch({ headed, softwareGL });
  const hot = [];
  try {
    await throttle(cdp, RATE);
    for (const id of ids) {
      await page.goto(`${BASE}/?mock=1&stage=${id}&seed=3`, { waitUntil: "load" });
      await page.getByTestId("stage-title").waitFor({ timeout: 60000 });
      await page.waitForTimeout(id === "a4-tech" ? 8000 : 2500); // (the tech tree reveals its nodes for ~5 s)
      let w = await window_(page, cdp, 3000);
      const limit = id === "a2-backward" ? 70 : LIMITS.idleCpuPct;
      if (w.cpu_pct > limit) w = await window_(page, cdp, 3000); // one retry
      console.log(`${w.cpu_pct > limit ? "FAIL" : "ok  "} stage ${id.padEnd(14)} idle cpu ${String(w.cpu_pct).padStart(5)}%  (limit ${limit}%)`);
      if (w.cpu_pct > limit) hot.push(id);
    }
  } finally { await browser.close(); }
  if (hot.length) { failed = true; console.log("      stages that never go idle: " + hot.join(", ")); }
}
console.log(failed ? "PERF REGRESSION" : "perf gate passed");
process.exit(failed ? 1 : 0);
