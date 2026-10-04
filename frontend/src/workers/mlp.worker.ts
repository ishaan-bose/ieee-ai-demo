// Act 1 training runs here so the UI never blocks (SPEC 5). Protocol:
//   in : {type:"train", id, X, y, cfg: MlpConfig, epochs, slow, grid}   out: {type:"progress"|"done", id, epoch, loss, acc, grid, broke?}
//        {type:"stop"}
import { MLP, type MlpConfig } from "../lib/mlp";

interface TrainMsg { type: "train"; id: number; X: Float32Array; y: Uint8Array; cfg: MlpConfig; epochs: number; slow: boolean; grid: number }
let stopFlag = false;
let current = 0;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

self.onmessage = async (e: MessageEvent<TrainMsg | { type: "stop" }>) => {
  const msg = e.data;
  if (msg.type === "stop") { stopFlag = true; return; }
  stopFlag = false;
  current = msg.id;
  const net = new MLP(msg.cfg);
  const n = msg.y.length;
  let lastAcc = 0, flat = 0, broke: "nan" | "flat" | undefined;
  const every = Math.max(1, Math.round(msg.epochs / 60));
  for (let epoch = 1; epoch <= msg.epochs && !stopFlag; epoch++) {
    const loss = net.trainEpoch(msg.X, msg.y, Math.min(64, n));
    if (!Number.isFinite(loss)) { broke = "nan"; break; }
    if (epoch % every === 0 || epoch === msg.epochs) {
      const acc = net.accuracy(msg.X, msg.y);
      flat = Math.abs(acc - lastAcc) < 0.002 ? flat + 1 : 0;
      lastAcc = acc;
      (self as unknown as Worker).postMessage({ type: "progress", id: current, epoch, loss, acc, grid: net.grid(msg.grid), paramCount: net.paramCount });
      if (msg.slow) await sleep(120); else await sleep(0);
    }
  }
  const acc = net.accuracy(msg.X, msg.y);
  const loss = net.loss(msg.X, msg.y);
  if (!broke && (!Number.isFinite(loss) || (acc < 0.56 && flat > 8))) broke = acc < 0.56 ? "flat" : broke;
  (self as unknown as Worker).postMessage({
    type: "done", id: current, epoch: msg.epochs, loss, acc, grid: net.grid(msg.grid), broke, stopped: stopFlag,
    collapsed: msg.cfg.activation === "linear" ? net.collapsed() : undefined, paramCount: net.paramCount,
  });
};
