import { Suspense, lazy, useEffect } from "react";
import { MOCK, isMockOffline, setMockOffline } from "./lib/env";
import { HealthProvider } from "./state/health";
import Demo from "./pages/Demo";

const Presenter = lazy(() => import("./pages/Presenter"));
const Build = lazy(() => import("./pages/Build"));
const Admin = lazy(() => import("./pages/Admin"));

/** Routes (SPEC 5.2): / demo, /presenter talk track (phone), /build submission form, /admin admin panel (never linked from anywhere). */
/** In mock mode (?mock=1) the O key switches the pretend server off and on, on every page, to check the offline behaviour by eye. */
function useMockOfflineKey() {
  useEffect(() => {
    if (!MOCK) return;
    const f = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.key === "o" || e.key === "O") setMockOffline(!isMockOffline());
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
}

export default function App() {
  useMockOfflineKey();
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  const page = path === "/presenter" ? <Presenter /> : path === "/build" ? <Build /> : path === "/admin" ? <Admin /> : <Demo />;
  return <HealthProvider><Suspense fallback={<div className="p-10 text-slate-500">loading…</div>}>{page}</Suspense></HealthProvider>;
}
