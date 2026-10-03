import { useEffect, useState } from "react";
import { Doodle, LANE_COLORS } from "./ui";
import { loadDoodles, type DoodleRecord } from "../lib/doodle";
import type { LaneState } from "../lib/useRace";

/** The race's main canvas: 16 held-out doodles (probe.json). Each tile shows one dot per lane: red = wrong, green = right. */
export function ProbeWall({ lanes, labels }: { lanes: LaneState[]; labels: number[] }) {
  const [probe, setProbe] = useState<DoodleRecord[]>([]);
  useEffect(() => { loadDoodles("probe").then(setProbe).catch(() => undefined); }, []);
  return (
    <div className="grid grid-cols-4 gap-3" data-testid="probe-wall">
      {probe.map((rec, i) => (
        <div key={rec.k} className="rounded-xl bg-slate-900/70 p-1.5 text-center">
          <Doodle record={rec} size={104} />
          <div className="mt-1 flex justify-center gap-1.5">
            {(lanes.length ? lanes : [null]).map((l, li) => {
              const pred = l?.latest?.probe_preds[i];
              const ok = pred !== undefined && pred === labels[i];
              return <span key={li} className="h-3.5 w-3.5 rounded-full border-2 transition-colors duration-300" style={{ borderColor: LANE_COLORS[li], backgroundColor: pred === undefined ? "#334155" : ok ? "#22c55e" : "#ef4444" }} />;
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
