// Minimal Server-Sent-Events reader on top of fetch (EventSource cannot send headers; admin routes need X-Admin-Token).
export interface SseEvent { event: string; data: unknown }

export async function* sseEvents(url: string, init: RequestInit = {}, signal?: AbortSignal): AsyncGenerator<SseEvent> {
  const res = await fetch(url, { ...init, signal, headers: { Accept: "text/event-stream", ...(init.headers ?? {}) } });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buf += dec.decode(value, { stream: true });
      let i: number;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!block || block.startsWith(":")) continue; // keep-alive comment
        let event = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        try { yield { event, data: JSON.parse(data) }; } catch { /* ignore a malformed block */ }
      }
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}
