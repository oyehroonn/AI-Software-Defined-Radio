"use client";

import { useEffect, useRef, useState } from "react";
import type { AircraftState } from "../components/AircraftLayer";

type Snapshot = {
  timeMs: number;
  statesById: Map<string, AircraftState>;
};

export function useAircraftData(pollMs = 10000, lerpWindowMs = 2500) {
  const [renderStates, setRenderStates] = useState<AircraftState[]>([]);
  const prevSnap = useRef<Snapshot | null>(null);
  const nextSnap = useRef<Snapshot | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchLoop() {
      try {
        const res = await fetch("/api/opensky", { cache: "no-store" });
        if (!res.ok) throw new Error(`OpenSky error: ${res.status}`);
        const data: { time: number; states: AircraftState[] } = await res.json();

        if (cancelled) return;

        const nowMs = Date.now();
        const serverTimeMs = (data.time ?? Math.floor(nowMs / 1000)) * 1000;

        const snap: Snapshot = {
          timeMs: serverTimeMs,
          statesById: new Map(
            (data.states ?? []).map((s) => [s.icao24, { ...s, lastUpdate: data.time }]),
          ),
        };

        prevSnap.current = nextSnap.current ?? snap;
        nextSnap.current = snap;
      } catch (e) {
        console.error(e);
      } finally {
        if (!cancelled) {
          setTimeout(fetchLoop, pollMs);
        }
      }
    }

    fetchLoop();

    return () => {
      cancelled = true;
    };
  }, [pollMs]);

  useEffect(() => {
    let rafId: number;

    const loop = () => {
      const prev = prevSnap.current;
      const next = nextSnap.current;
      const now = performance.now();

      if (!prev || !next) {
        rafId = requestAnimationFrame(loop);
        return;
      }

      const dt = Math.max(0, now - (next.timeMs - pollMs));
      const t = Math.min(1, dt / lerpWindowMs);
      const eased = t * t * (3 - 2 * t); // smoothstep

      const merged: AircraftState[] = [];

      for (const [id, nextState] of next.statesById.entries()) {
        const prevState = prev.statesById.get(id) ?? nextState;

        const lat =
          prevState.lat + (nextState.lat - prevState.lat) * eased;
        const lon =
          prevState.lon + (nextState.lon - prevState.lon) * eased;
        const altitudeMeters =
          (prevState.altitudeMeters ?? 0) +
          ((nextState.altitudeMeters ?? 0) -
            (prevState.altitudeMeters ?? 0)) *
            eased;
        const headingDeg = nextState.headingDeg;

        merged.push({
          ...nextState,
          lat,
          lon,
          altitudeMeters,
          headingDeg,
        });
      }

      setRenderStates(merged);
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [pollMs, lerpWindowMs]);

  return renderStates;
}

