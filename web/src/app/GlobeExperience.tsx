"use client";

import React, { Suspense, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import { Color } from "three";
import Globe from "./components/Globe";
import AircraftLayer, { AircraftState } from "./components/AircraftLayer";
import InteractionLayer from "./components/InteractionLayer";
import { useAircraftData } from "./hooks/useAircraftData";

const BG_COLOR = new Color("#050712");

export default function GlobeExperience() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const aircraftStates: AircraftState[] = useAircraftData(10000, 2500);
  const aircraftById = useMemo(
    () => new Map(aircraftStates.map((s) => [s.icao24, s])),
    [aircraftStates],
  );
  const selectedAircraft = selectedId ? aircraftById.get(selectedId) : null;

  return (
    <div className="h-full w-full relative">
      <div className="pointer-events-none absolute inset-0 z-10 flex items-end justify-end p-4 md:p-6">
        {selectedAircraft && (
          <div className="max-w-sm rounded-2xl bg-black/60 border border-cyan-400/40 shadow-[0_0_40px_rgba(34,211,238,0.35)] px-4 py-3 backdrop-blur-xl">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-xs uppercase tracking-[0.18em] text-cyan-300/80">
                  Active Track
                </div>
                <div className="mt-1 text-sm font-semibold text-cyan-100 tracking-wide hud-glow-text">
                  {selectedAircraft.callsign ?? selectedAircraft.icao24}
                </div>
              </div>
              <div className="text-right text-[10px] text-cyan-200/70">
                <div>
                  ALT{" "}
                  <span className="font-semibold">
                    {Math.round((selectedAircraft.altitudeMeters ?? 0) * 3.28084).toLocaleString()}
                    {" ft"}
                  </span>
                </div>
                <div>
                  SPD{" "}
                  <span className="font-semibold">
                    {Math.round((selectedAircraft.velocityMs ?? 0) * 1.94384)} kt
                  </span>
                </div>
                <div>
                  HDG{" "}
                  <span className="font-semibold">
                    {Math.round(selectedAircraft.headingDeg ?? 0)}°
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <Canvas
        camera={{ position: [0, 1.8, 3.2], fov: 45, near: 0.1, far: 1000 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={[BG_COLOR]} />
        <fog attach="fog" args={[BG_COLOR, 8, 25]} />

        <ambientLight intensity={0.6} color="#444a66" />
        <directionalLight position={[5, 3, 5]} intensity={1.6} color="#ffffff" />
        <directionalLight position={[-4, -2, -5]} intensity={0.3} color="#223355" />

        <Suspense fallback={null}>
          <Stars radius={50} depth={40} count={1200} factor={3} saturation={0} fade />

          <Globe radius={1} />

          <AircraftLayer
            globeRadius={1}
            states={aircraftStates}
            hoveredId={hoveredId}
            selectedId={selectedId}
          />

          <InteractionLayer
            globeRadius={1}
            onHoverChange={setHoveredId}
            onSelect={setSelectedId}
          />

          <EffectComposer>
            <Bloom intensity={1.4} luminanceThreshold={0.2} luminanceSmoothing={0.25} />
            <Vignette eskil={false} offset={0.2} darkness={0.9} />
          </EffectComposer>
        </Suspense>

        <OrbitControls
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          autoRotate
          autoRotateSpeed={0.25}
          minDistance={1.3}
          maxDistance={5.5}
          zoomSpeed={0.5}
          rotateSpeed={0.6}
        />
      </Canvas>
    </div>
  );
}

