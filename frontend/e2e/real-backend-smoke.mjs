// Smoke test of the REAL frontend against the REAL backend (no ?mock). Start the backend first, e.g. on fake data:
//   cd backend && python3 -m tests.fake_data --out /tmp/fake --n 300 && NUM_WORKERS=2 DATA_DIR=/tmp/fake python3 scripts/rasterize_quickdraw.py
//   DATA_DIR=/tmp/fake STATE_DIR=/tmp/state ADMIN_TOKEN=dev BUDGET_FLOPS=3e10 python3 -m app.main     (and npm run dev here)
// then:  ADMIN_TOKEN=dev node e2e/real-backend-smoke.mjs
import { chromium } from "playwright";
const BASE = process.env.BASE ?? "http://127.0.0.1:5173";
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let failed = false;
const check = async (name, fn) => { try { await fn(); console.log("ok  ", name); } catch (e) { failed = true; console.log("FAIL", name, "\n     ", e.message.split("\n")[0]); } };
const must = (c, m) => { if (!c) throw new Error(m); };

await check("health through the proxy", async () => {
  const h = await (await fetch(`${BASE}/api/health`)).json();
  must(h.ok === true && "queue_length" in h, JSON.stringify(h));
});
await check("a live race streams through the Vite proxy and fills the probe wall", async () => {
  await page.goto(`${BASE}/?stage=a2-loss&seed=3`); await page.waitForTimeout(1500);
  await page.getByTestId("start-race").click();
  await page.getByText("live", { exact: true }).waitFor({ timeout: 15000 });
  await page.waitForTimeout(3000);
  must(await page.getByText("recorded", { exact: true }).count() === 0 && await page.getByText("demo data", { exact: true }).count() === 0, "fell back to a recording");
  must(await page.getByTestId("probe-wall").locator("span").count() >= 16, "probe wall missing");
  await page.getByTestId("start-race").click();
});
await check("submit the form: participant code, status moves on, an admin sees it", async () => {
  await page.goto(`${BASE}/build`); await page.waitForTimeout(1200);
  await page.getByTestId("layers").fill("2"); await page.getByTestId("width").fill("5");
  await page.getByTestId("nickname").fill("smoke"); await page.getByTestId("modelname").fill("Smoke Net"); await page.getByTestId("consent").check();
  await page.getByTestId("submit").click();
  await page.getByTestId("code").waitFor({ timeout: 8000 });
  await page.getByText(/queued|running|done/).first().waitFor({ timeout: 12000 });
  await page.goto(`${BASE}/admin`); await page.waitForTimeout(500);
  await page.getByTestId("token").fill(process.env.ADMIN_TOKEN ?? "dev"); await page.getByTestId("login").click();
  await page.getByTestId("queue-table").waitFor({ timeout: 6000 });
  await page.getByText("Smoke Net").first().waitFor({ timeout: 6000 });
  await page.getByTestId("monitor").getByText(/GPU/).first().waitFor({ timeout: 6000 });
});
await check("wrong admin token is rejected", async () => {
  await page.goto(`${BASE}/admin`); await page.waitForTimeout(400);
  await page.getByTestId("token").fill("nope"); await page.getByTestId("login").click();
  await page.getByTestId("auth-error").waitFor({ timeout: 4000 });
});
await check("no page errors", async () => must(errors.length === 0, errors[0]));
await browser.close();
process.exit(failed ? 1 : 0);
