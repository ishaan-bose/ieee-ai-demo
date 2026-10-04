// Copies the small real sample data (committed in ../data/samples) into public/data so the browser can fetch it.
// Runs automatically before `npm run dev` / `npm run build` (predev / prebuild). Idempotent.
import { cpSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const samples = resolve(here, "../../data/samples");
const out = resolve(here, "../public/data");
mkdirSync(resolve(out, "speech"), { recursive: true });
for (const f of ["duel.json", "probe.json", "samples.json", "classes.json"]) {
  const src = resolve(samples, "quickdraw", f);
  if (existsSync(src)) cpSync(src, resolve(out, f));
}
if (existsSync(resolve(samples, "speech/samples"))) {
  cpSync(resolve(samples, "speech/samples"), resolve(out, "speech/samples"), { recursive: true });
  cpSync(resolve(samples, "speech/samples.json"), resolve(out, "speech/samples.json"));
  cpSync(resolve(samples, "speech/classes.json"), resolve(out, "speech/classes.json"));
}
console.log("synced sample data into public/data");
