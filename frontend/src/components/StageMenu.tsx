import { AnimatePresence, motion } from "framer-motion";
import { useStage } from "../state/StageProvider";

/** G: hidden stage-jump menu. Every act and sub-step is independently skippable. */
export function StageMenu() {
  const { menuOpen, setMenuOpen, stages, index, jump } = useStage();
  return (
    <AnimatePresence>
      {menuOpen && (
        <motion.div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setMenuOpen(false)}>
          <div className="max-h-[85vh] w-[560px] overflow-auto rounded-2xl bg-slate-900 p-5 ring-1 ring-slate-700" onClick={(e) => e.stopPropagation()} data-testid="stage-menu">
            <h2 className="mb-3 text-xl font-semibold text-slate-200">Jump to stage <span className="text-sm font-normal text-slate-500">(G closes)</span></h2>
            {stages.map((s, i) => (
              <button key={s.id} onClick={() => jump(s.id)} className={`flex w-full items-center gap-3 rounded-lg px-3 py-1.5 text-left text-lg hover:bg-slate-800 ${i === index ? "bg-sky-500/20 text-sky-200" : "text-slate-300"}`}>
                <span className="w-14 text-sm text-slate-500">Act {s.act}</span><span>{s.title}</span>
              </button>
            ))}
            <p className="mt-3 text-sm text-slate-500">→ next · ← back · Space run · S recorded result · R reset stage · F force recorded · P presenter mode</p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
