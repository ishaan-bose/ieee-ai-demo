import { useEffect, useState } from "react";
import { getHealth, type Health, type Tick } from "./api";

type StreamState = "connecting" | "live" | "reconnecting";

function HealthCard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function check() {
    setLoading(true);
    setError(null);
    try {
      setHealth(await getHealth());
    } catch (e) {
      setHealth(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const rows: [string, string][] = health
    ? [
        ["ok", String(health.ok)],
        ["device", health.device],
        ["gpu_name", health.gpu_name ?? "—"],
        ["vram_gb", health.vram_gb?.toString() ?? "—"],
        ["queue_length", String(health.queue_length)],
        ["demo_mode", String(health.demo_mode)],
        ["paused", String(health.paused)],
        ["version", health.version],
      ]
    : [];

  return (
    <section className="rounded-2xl bg-slate-900 p-8 shadow-lg ring-1 ring-slate-800">
      <h2 className="mb-6 text-3xl font-semibold">Server health</h2>
      <button
        onClick={check}
        disabled={loading}
        className="rounded-xl bg-sky-500 px-6 py-3 text-2xl font-semibold text-slate-950 transition hover:bg-sky-400 disabled:opacity-50"
      >
        {loading ? "Checking…" : "Check /api/health"}
      </button>
      {error && (
        <p className="mt-6 text-2xl text-orange-400">
          Server unreachable ({error}). Is the backend running and the SSH tunnel up?
        </p>
      )}
      {health && (
        <dl className="mt-6 grid grid-cols-[auto_1fr] gap-x-8 gap-y-2 text-2xl">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="font-mono text-slate-400">{k}</dt>
              <dd className="font-mono">{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}

function HelloStream() {
  const [tick, setTick] = useState<Tick | null>(null);
  const [state, setState] = useState<StreamState>("connecting");

  useEffect(() => {
    // EventSource retries on its own after a clean disconnect, but a silently dead
    // tunnel (Wi-Fi drop) or an error response can leave it stuck. A watchdog
    // reopens the stream whenever no tick has arrived for STALE_MS.
    const STALE_MS = 4000;
    let es: EventSource | null = null;
    let lastMessage = Date.now();

    const open = () => {
      es?.close();
      lastMessage = Date.now();
      es = new EventSource("/api/dev/hello-stream");
      es.addEventListener("tick", (e) => {
        lastMessage = Date.now();
        setTick(JSON.parse((e as MessageEvent<string>).data) as Tick);
        setState("live");
      });
      es.onerror = () => setState("reconnecting");
    };

    open();
    const watchdog = setInterval(() => {
      if (Date.now() - lastMessage > STALE_MS) {
        setState("reconnecting");
        open();
      }
    }, 1000);
    return () => {
      clearInterval(watchdog);
      es?.close();
    };
  }, []);

  const color = { connecting: "text-slate-400", live: "text-sky-400", reconnecting: "text-orange-400" }[state];

  return (
    <section className="rounded-2xl bg-slate-900 p-8 shadow-lg ring-1 ring-slate-800">
      <h2 className="mb-6 text-3xl font-semibold">Live stream (SSE)</h2>
      <div data-testid="counter" className="font-mono text-8xl font-bold tabular-nums">
        {tick ? tick.count : "–"}
      </div>
      <p className={`mt-4 text-2xl ${color}`}>
        {state === "live" ? "● live" : state === "connecting" ? "connecting…" : "● reconnecting…"}
      </p>
      {tick && (
        <p className="mt-2 text-xl text-slate-500">
          server time {new Date(tick.server_time * 1000).toLocaleTimeString()}
        </p>
      )}
    </section>
  );
}

export default function App() {
  return (
    <main className="min-h-screen bg-slate-950 p-10 text-slate-100">
      <h1 className="mb-2 text-5xl font-bold">Build Your Own AI</h1>
      <p className="mb-10 text-2xl text-slate-400">Phase 1 connectivity check: laptop → SSH tunnel → server</p>
      <div className="grid max-w-5xl gap-8 md:grid-cols-2">
        <HealthCard />
        <HelloStream />
      </div>
    </main>
  );
}
