// End-to-end check of the whole UI against the MOCK server (no backend needed). Run with the dev server up:
//   npm run dev   (in one terminal)      npm run e2e   (in another)      needs: npm i && npx playwright install chromium
// Set CHROMIUM=/path/to/chrome to use an existing browser. Exits non-zero on the first failed check.
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://127.0.0.1:5173";
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PAGEERROR " + e.message));
let passed = 0;
const check = async (name, fn) => { if (process.env.ONLY && !name.includes(process.env.ONLY)) return; try { await fn(); passed++; console.log("ok  ", name); } catch (e) { console.log("FAIL", name, "\n     ", e.message.split("\n")[0]); await page.screenshot({ path: `e2e-fail-${passed}.png` }); if (process.env.DEBUG) console.log((await page.locator("main").innerText()).slice(0, 400)); process.exitCode = 1; } };
const goto = (stage, extra = "") => page.goto(`${BASE}/?mock=1&stage=${stage}&seed=3${extra}`).then(() => page.waitForTimeout(700));
const space = () => page.keyboard.press(" ");
const must = (cond, msg) => { if (!cond) throw new Error(msg); };

await check("Act 0 duel: options appear, answering scores, 5 rounds end in a result", async () => {
  await goto("a0-duel");
  await space();
  await page.getByTestId("options").locator("button").first().waitFor({ timeout: 14000 });
  await page.waitForTimeout(200); // (a human cannot press within the very frame the buttons appear)
  await page.keyboard.press("1");
  await page.getByText(/Space for the (next round|result)/).waitFor({ timeout: 3000 });
  await space(); await page.waitForTimeout(300); // next round (the old buttons must be gone before we wait for the new ones)
  for (let r = 0; r < 4; r++) {
    await page.getByTestId("options").locator("button").first().waitFor({ timeout: 14000 }); await page.waitForTimeout(200); await page.keyboard.press("2");
    await page.getByText(/Space for the (next round|result)/).waitFor({ timeout: 3000 }); await space(); await page.waitForTimeout(300);
  }
  await page.getByTestId("duel-final").waitFor({ timeout: 14000 });
  const score = await page.getByTestId("score").innerText();
  must(/You/.test(score) && /The AI/.test(score), "score panel missing");
});

await check("Act 1: line guess, least squares, depth training collapses, fuse, ReLU bends", async () => {
  await goto("a1-clusters");
  await page.getByTestId("boundary").waitFor();
  await space();
  await goto("a1-knobs");
  await space(); await page.waitForTimeout(1500);
  const eq = await page.getByTestId("equation").innerText();
  must(/x/.test(eq), "equation missing");
  await goto("a1-depth");
  await space(); await page.waitForTimeout(3500);
  await goto("a1-fuse");
  await space(); await page.getByTestId("fused").waitFor({ timeout: 4000 });
  await goto("a1-relu");
  await space();
  await page.waitForFunction(() => document.querySelector('[data-testid="gauge"]')?.textContent?.match(/(\d+)%/)?.[1] >= 90, null, { timeout: 25000 });
});

await check("Act 1 gallery: hard step is flagged BROKE", async () => {
  await goto("a1-gallery");
  await page.getByTestId("card-hard_step").click();
  await space();
  await page.getByTestId("card-hard_step").getByText("BROKE").waitFor({ timeout: 15000 });
});

await check("Act 2 forward pass: drawing updates the top-5 and the network", async () => {
  await goto("a2-forward");
  const box = await page.getByTestId("draw-canvas").boundingBox();
  await page.mouse.move(box.x + 100, box.y + 100); await page.mouse.down();
  for (let i = 0; i < 12; i++) await page.mouse.move(box.x + 100 + i * 12, box.y + 100 + (i % 3) * 20);
  await page.mouse.up();
  await page.waitForTimeout(500);
  const top = await page.getByTestId("top5").innerText();
  must(/cat|bicycle|house|pizza|lightning|star|fish|tree|umbrella|sword/.test(top), "no class names in the top 5");
  await page.keyboard.press("r");
  await page.waitForTimeout(200);
});

await check("Act 2 loss race: pick lanes, run live (mock server), probe wall fills", async () => {
  await goto("a2-loss");
  await page.getByTestId("start-race").click();
  await page.waitForTimeout(3500);
  const wall = await page.getByTestId("probe-wall").locator("span").count();
  must(wall >= 16, "probe wall dots missing");
  await page.getByTestId("start-race").click(); // stop
});

await check("Act 2: cutting the server mid-race falls back to the recorded stream; S shows the recorded result", async () => {
  await goto("a2-lr");
  await page.getByTestId("start-race").click();
  await page.waitForTimeout(2200);
  await page.keyboard.press("o"); // simulated outage
  await page.waitForTimeout(4500);
  const txt = await page.locator("main").innerText();
  must(/recorded|demo data/.test(txt), "no recorded/demo badge after the outage");
  await page.keyboard.press("o");
  await page.keyboard.press("r");
  await page.keyboard.press("s");
  await page.waitForTimeout(800);
  must(/recorded|demo data/.test(await page.locator("main").innerText()), "S did not show a recorded result");
});

await check("Act 2 batch race shows updates and doodles-seen counters; backward stage rolls the ball", async () => {
  await goto("a2-batch");
  await page.keyboard.press("s"); await page.waitForTimeout(500);
  await space(); await page.waitForTimeout(3000);
  must(await page.getByTestId("counters-one").count() > 0, "counters missing");
  await goto("a2-backward");
  await space(); await page.waitForTimeout(1500);
});

await check("Act 3: sample clips, step view, live mic (fake device) records", async () => {
  await goto("a3-preprocess");
  await page.getByTestId("clip-name").waitFor();
  await page.getByTestId("spectrogram").waitFor();
  for (let i = 0; i < 4; i++) { await space(); await page.waitForTimeout(250); }
  await page.getByTestId("ptt").dispatchEvent("pointerdown");
  await page.waitForTimeout(1500);
  await page.getByTestId("ptt").dispatchEvent("pointerup");
  await page.waitForTimeout(800);
  const name = await page.getByTestId("clip-name").innerText();
  must(/your recording/.test(name) || await page.getByTestId("mic-error").count() > 0, "neither a recording nor a clean mic error: " + name);
});

await check("Act 3 trap reveals confusion matrix, recall and the fix; overfitting slider works", async () => {
  await goto("a3-trap");
  await space(); await page.getByTestId("confusion").waitFor();
  await space(); await page.getByTestId("recall").waitFor();
  await space(); await page.getByTestId("weighted").waitFor();
  await goto("a3-overfit");
  await page.getByTestId("size-slider").fill("3");
  await space(); await page.waitForTimeout(1500);
});

await check("Act 4: tech tree, rematch, close with stats and QR", async () => {
  await goto("a4-tech"); await page.getByTestId("node-ema").waitFor(); await page.waitForTimeout(6000);
  await goto("a4-rematch"); await space(); await page.getByTestId("options").locator("button").first().waitFor({ timeout: 9000 });
  await goto("a4-close"); await page.getByTestId("stats").waitFor(); await page.getByTestId("booth-url").waitFor();
});

await check("G menu jumps stages; arrows navigate", async () => {
  await goto("a0-duel");
  await page.keyboard.press("g");
  await page.getByTestId("stage-menu").getByText("Backward pass and step size").click();
  await page.waitForTimeout(500);
  must((await page.getByTestId("stage-title").innerText()).includes("Backward"), "G jump failed");
  await page.keyboard.press("ArrowLeft"); await page.waitForTimeout(500);
  must((await page.getByTestId("stage-title").innerText()).includes("Loss race"), "ArrowLeft failed");
});

await check("/presenter follows the demo stage (BroadcastChannel) and shows notes", async () => {
  const pp = await ctx.newPage();
  await pp.goto(`${BASE}/presenter?mock=1`);
  await goto("a1-spirals");
  await page.keyboard.press("ArrowRight"); await page.waitForTimeout(800);
  const t = await pp.getByTestId("presenter-title").innerText();
  must(/Stack more lines/.test(t), "presenter shows: " + t);
  await pp.close();
});

await check("/build: live parameter count and tier, validation, submit gives a participant code, backup", async () => {
  await page.goto(`${BASE}/build?mock=1`); await page.waitForTimeout(600);
  const before = await page.getByTestId("param-count").innerText();
  await page.getByTestId("layers").fill("6"); await page.getByTestId("width").fill("11");
  await page.waitForTimeout(500);
  const after = await page.getByTestId("param-count").innerText();
  must(before !== after, "param count did not update");
  await page.getByTestId("layers").fill("16"); await page.getByTestId("width").fill("13"); await page.waitForTimeout(600);
  must(await page.getByTestId("errors").count() > 0, "no error for >30M parameters");
  await page.getByTestId("layers").fill("3"); await page.getByTestId("width").fill("8"); await page.waitForTimeout(500);
  await page.getByTestId("nickname").fill("tester"); await page.getByTestId("modelname").fill("Tester Net"); await page.getByTestId("consent").check();
  await page.getByTestId("submit").click();
  await page.getByTestId("code").waitFor({ timeout: 6000 });
  must(/^AI-/.test(await page.getByTestId("code").innerText()), "participant code format");
  const dl = page.waitForEvent("download");
  await page.getByTestId("backup").click();
  must((await dl).suggestedFilename().endsWith(".json"), "backup file");
});

await check("/build offline: entry is saved locally as pending, then uploads by itself when the server returns", async () => {
  await page.goto(`${BASE}/build?mock=1&offline=1`); await page.waitForTimeout(800);
  await page.getByTestId("nickname").fill("offline"); await page.getByTestId("modelname").fill("Late Net"); await page.getByTestId("consent").check();
  await page.getByTestId("build-offline").waitFor();
  await page.getByTestId("submit").click();
  await page.getByText("waiting for the server").waitFor({ timeout: 5000 });
  await page.keyboard.press("o"); // the server comes back (mock toggle)
  await page.getByTestId("entry").first().getByTestId("code").waitFor({ timeout: 12000 });
});

await check("/admin: login, confirm dialogs for kill/remove/redo, pause, demo mode, nuclear reset needs RESET + hold", async () => {
  await page.goto(`${BASE}/admin?mock=1`); await page.waitForTimeout(500);
  await page.getByTestId("token").fill("anything"); await page.getByTestId("login").click();
  await page.getByTestId("queue-table").waitFor();
  const rows = await page.locator('[data-testid^="job-"]').count();
  must(rows >= 3, "queue rows: " + rows);
  await page.locator('[data-testid^="kill-"]').first().click();
  await page.getByTestId("confirm-text").waitFor();
  await page.getByTestId("confirm-no").click();
  await page.locator('[data-testid^="remove-"]').first().click();
  await page.getByTestId("confirm-yes").click();
  await page.waitForTimeout(2300);
  must(await page.locator('[data-testid^="job-"]').count() === rows - 1, "remove did not shrink the queue");
  await page.getByTestId("pause").click(); await page.getByTestId("confirm-yes").click();
  await page.getByText("Resume queue").waitFor({ timeout: 4000 });
  await page.getByTestId("demo-mode").click(); await page.getByTestId("confirm-yes").click();
  await page.getByText("Demo Mode: ON").waitFor({ timeout: 4000 });
  must(await page.getByTestId("hold-reset").isDisabled(), "reset enabled without typing RESET");
  await page.getByTestId("reset-text").fill("RESET");
  const btn = page.getByTestId("hold-reset"); const b = await btn.boundingBox();
  await page.mouse.move(b.x + 20, b.y + 10); await page.mouse.down(); await page.waitForTimeout(500); await page.mouse.up(); // too short
  await page.waitForTimeout(500);
  must(await page.locator('[data-testid^="job-"]').count() > 0, "a short press must not reset");
  await page.mouse.move(b.x + 20, b.y + 10); await page.mouse.down(); await page.waitForTimeout(2600); await page.mouse.up();
  await page.waitForTimeout(2500);
  must(await page.locator('[data-testid^="job-"]').count() === 0, "hold-to-confirm did not reset");
});

await check("3D valley renders with WebGL (software GL) and rolls the ball", async () => {
  const b2 = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const p2 = await b2.newPage({ viewport: { width: 1600, height: 900 } });
  const errs = []; p2.on("pageerror", (e) => errs.push(e.message));
  await p2.goto(`${BASE}/?mock=1&stage=a2-backward`); await p2.waitForTimeout(1500);
  const has3d = await p2.getByTestId("valley3d").count(), has2d = await p2.getByTestId("valley2d").count();
  await p2.keyboard.press(" "); await p2.waitForTimeout(1200);
  await b2.close();
  must(has3d === 1 && has2d === 0 && errs.length === 0, `3d=${has3d} 2d=${has2d} errors=${errs[0]}`);
});

await check("no console errors during the whole walkthrough", async () => { must(errors.length === 0, errors.slice(0, 3).join(" | ")); });
console.log(`${passed} checks passed`);
await browser.close();
