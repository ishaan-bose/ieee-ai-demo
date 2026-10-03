// Prints performance numbers for the first stages. Usage (dev server or `vite preview` must be running):
//   BASE=http://127.0.0.1:5173 node e2e/perf-probe.mjs [--url "/?mock=1"] [--throttle 4] [--headed] [--software-gl] [--presses 30] [--stages 8]
// Needs a display for --headed (use: xvfb-run -a node e2e/perf-probe.mjs --headed).
import { gc, heapMB, keyLatencies, launch, pct, throttle, window_ } from "./perf-lib.mjs";

const arg = (n, d) => { const i = process.argv.indexOf("--" + n); return i < 0 ? d : (process.argv[i + 1]?.startsWith("--") || process.argv[i + 1] === undefined ? true : process.argv[i + 1]); };
const BASE = process.env.BASE ?? "http://127.0.0.1:5173";
const url = arg("url", "/?mock=1"), rate = Number(arg("throttle", 4)), presses = Number(arg("presses", 30)), stages = Number(arg("stages", 8));
const headed = arg("headed", false) === true, softwareGL = arg("software-gl", false) === true;

const { browser, page, cdp, errors } = await launch({ headed, softwareGL });
await throttle(cdp, rate);
await page.goto(BASE + url);
await page.waitForTimeout(3000); // let the first screen settle
const out = { url, throttle: rate, headed, softwareGL };
console.error("settled");
out.idle_first_screen = await window_(page, cdp, 5000);
console.log("idle_first_screen " + JSON.stringify(out.idle_first_screen));
console.error("idle done");
const heap0 = await heapMB(cdp);
// presses: -> through the first `stages` stages and back, `presses` presses in total, 400 ms apart (a fast presenter)
const seq = []; for (let i = 0; seq.length < presses; i++) seq.push(i % (2 * (stages - 1)) < stages - 1 ? "ArrowRight" : "ArrowLeft");
const nodeSide = []; // wall time of page.keyboard.press itself: includes the time a key waits in the input queue behind a busy main thread
const deadline = Date.now() + Number(arg("deadline", 120)) * 1000;
out.keypress_window = await window_(page, cdp, 0, async () => {
  for (const k of seq) {
    if (Date.now() > deadline) { out.aborted_after_presses = nodeSide.length; break; }
    const t = Date.now();
    await Promise.race([page.keyboard.press(k), new Promise((r) => setTimeout(r, 30000))]);
    nodeSide.push(Date.now() - t);
    await page.waitForTimeout(400);
  }
});
out.press_wall_ms = { n: nodeSide.length, p50: pct(nodeSide, 0.5), max: Math.max(0, ...nodeSide) };
console.log("press_wall_ms " + JSON.stringify(out.press_wall_ms) + " aborted_after=" + out.aborted_after_presses + " window=" + JSON.stringify(out.keypress_window));
await page.waitForTimeout(1500);
const { keys, events } = await keyLatencies(page);
const arrow = keys.filter((k) => k.key.startsWith("Arrow")).map((k) => k.ms);
out.key_to_paint_ms = { n: arrow.length, p50: +pct(arrow, 0.5).toFixed(0), p95: +pct(arrow, 0.95).toFixed(0), max: +Math.max(0, ...arrow).toFixed(0) };
out.event_timing_ms = { n: events.length, max: +Math.max(0, ...events.map((e) => e.d)).toFixed(0) };
out.idle_after = await window_(page, cdp, 5000);
out.heap_growth_mb = +((await heapMB(cdp)) - heap0).toFixed(1);
out.errors = errors.slice(0, 5);
console.log(JSON.stringify(out, null, 1));
await browser.close();
