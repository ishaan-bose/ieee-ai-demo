// Layout check of the finale (Act 4 close) at 1920x1080 and 1600x900, with LARGE values in the stat boxes. Needs `npm run dev`.
// Asserts bounding boxes (everything inside its card, nothing overlapping, no horizontal scroll) and DECODES the QR code from a screenshot.
// Screenshots go to e2e/out/. Usage: npm run e2e:act4
import { chromium } from "playwright";
import jsQR from "jsqr";
import { PNG } from "pngjs";
import fs from "node:fs";

const BASE = process.env.BASE ?? "http://127.0.0.1:5173";
const URL_EXPECTED = "https://ieeecspesu.vercel.app";
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
fs.mkdirSync("e2e/out", { recursive: true });
let failed = false;
const fail = (m) => { failed = true; console.log("FAIL", m); };

for (const [w, h] of [[1920, 1080], [1600, 900]]) {
  for (const [label, vals] of [["small", { runs: 0, paramsTrained: 0, flops: 0 }], ["large", { runs: 1234, paramsTrained: 123456789012, flops: 3.6e15 }], ["huge", { runs: 9999, paramsTrained: 9.9e14, flops: 9.99e17 }]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    const errors = []; page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/?mock=1&stage=a4-close&seed=3`);
    await page.getByTestId("club-card").waitFor();
    await page.evaluate((v) => window.__byoai.set(v), vals);
    await page.waitForTimeout(1200);
    const tag = `${w}x${h} ${label}`;
    const r = await page.evaluate(() => {
      const box = (el) => { const b = el.getBoundingClientRect(); return { x: b.x, y: b.y, r: b.right, b: b.bottom, w: b.width, h: b.height }; };
      const inside = (c, i) => i.x >= c.x - 0.5 && i.r <= c.r + 0.5 && i.y >= c.y - 0.5 && i.b <= c.b + 0.5;
      const out = { problems: [] };
      const stats = [...document.querySelectorAll('[data-testid="stat-box"]')];
      out.nStats = stats.length;
      for (const s of stats) {
        const sb = box(s);
        for (const child of s.querySelectorAll("div")) if (!inside(sb, box(child))) out.problems.push("stat text outside its box: " + child.textContent);
        for (const child of s.querySelectorAll("div")) if (child.scrollWidth > child.clientWidth + 1) out.problems.push("stat text clipped: " + child.textContent);
      }
      const card = document.querySelector('[data-testid="club-card"]'), cb = box(card);
      for (const id of ["qr", "club-url"]) if (!inside(cb, box(document.querySelector(`[data-testid="${id}"]`)))) out.problems.push(id + " outside the card");
      for (const p of card.querySelectorAll("p,span,canvas")) if (!inside(cb, box(p))) out.problems.push("card child outside: " + p.tagName);
      const pitch = [...document.querySelectorAll("p")].find((p) => /We do AI research/.test(p.textContent));
      const pb = box(pitch), sb = box(document.querySelector('[data-testid="stats"]'));
      const rects = { stats: sb, card: cb, pitch: pb };
      const overlap = (a, b) => a.x < b.r && b.x < a.r && a.y < b.b && b.y < a.b;
      for (const [a, b] of [["stats", "card"], ["stats", "pitch"], ["card", "pitch"]]) if (overlap(rects[a], rects[b])) out.problems.push(`${a} overlaps ${b}`);
      const vh = innerHeight, vw = innerWidth;
      for (const [n, b] of Object.entries(rects)) if (b.r > vw + 0.5 || b.x < -0.5 || b.b > vh + 0.5) out.problems.push(`${n} leaves the viewport (${Math.round(b.r)}x${Math.round(b.b)} of ${vw}x${vh})`);
      if (document.documentElement.scrollWidth > vw) out.problems.push("horizontal scroll: " + document.documentElement.scrollWidth + " > " + vw);
      if (document.documentElement.scrollHeight > vh + 1) out.problems.push("vertical overflow: " + document.documentElement.scrollHeight + " > " + vh);
      out.rects = Object.fromEntries(Object.entries(rects).map(([k, b]) => [k, [Math.round(b.x), Math.round(b.y), Math.round(b.r), Math.round(b.b)]]));
      out.text = document.querySelector('[data-testid="club-card"] p').textContent;
      out.stats = stats.map((s) => s.textContent);
      return out;
    });
    if (r.nStats !== 3) fail(`${tag}: expected 3 stat boxes, found ${r.nStats}`);
    for (const p of r.problems) fail(`${tag}: ${p}`);
    if (r.text !== `Scan this link to check out our club! Or alternatively, go to ${URL_EXPECTED}`) fail(`${tag}: card text is "${r.text}"`);
    const shot = `e2e/out/act4-${w}x${h}-${label}.png`;
    await page.screenshot({ path: shot });
    // decode the QR from the actual pixels
    const qrPng = PNG.sync.read(await page.getByTestId("qr").screenshot());
    const code = jsQR(new Uint8ClampedArray(qrPng.data), qrPng.width, qrPng.height);
    if (!code || code.data !== URL_EXPECTED) fail(`${tag}: QR decodes to ${code ? JSON.stringify(code.data) : "nothing"}`);
    if (/Build your own chess AI|localhost|Tournament after/.test(await page.locator("main").innerText())) fail(`${tag}: old text/URL still on screen`);
    if (errors.length) fail(`${tag}: page errors ${errors[0]}`);
    console.log(`${r.problems.length ? "    " : "ok  "} ${tag}: stats=${JSON.stringify(r.stats)} rects=${JSON.stringify(r.rects)} qr=${code?.data}`);
    await page.close();
  }
}
await browser.close();
process.exit(failed ? 1 : 0);
