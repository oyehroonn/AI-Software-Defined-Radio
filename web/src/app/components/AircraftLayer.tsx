"use client";

import React, { useMemo } from "react";
import { GroupProps, useThree } from "@react-three/fiber";
import * as THREE from "three";

export type AircraftState = {
  icao24: string;
  callsign?: string;
  lat: number;
  lon: number;
  altitudeMeters?: number;
  headingDeg?: number;
  velocityMs?: number;
};

type Props = GroupProps & {
  globeRadius: number;
  states: AircraftState[];
  hoveredId: string | null;
  selectedId: string | null;
};

function createPlaneGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.04);
  shape.lineTo(0.01, 0);
  shape.lineTo(0.006, -0.008);
  shape.lineTo(0.002, -0.004);
  shape.lineTo(-0.002, -0.004);
  shape.lineTo(-0.006, -0.008);
  shape.lineTo(-0.01, 0);
  shape.closePath();
  return new THREE.ShapeGeometry(shape);
}

const planeGeometry = createPlaneGeometry();

function latLonToVector3(
  latDeg: number,
  lonDeg: number,
  radius: number,
  altitudeOffset = 0,
) {
  const lat = THREE.MathUtils.degToRad(latDeg);
  const lon = THREE.MathUtils.degToRad(lonDeg);
  const r = radius + altitudeOffset;

  const x = r * Math.cos(lat) * Math.cos(lon);
  const y = r * Math.sin(lat);
  const z = r * Math.cos(lat) * Math.sin(lon);
  return new THREE.Vector3(x, y, z);
}

function altitudeOffset(globeRadius: number, altMeters?: number) {
  const alt = altMeters ?? 10000;
  const clamped = THREE.MathUtils.clamp(alt, 0, 40000);
  const ratio = clamped / 40000;
  return globeRadius * 0.08 * ratio;
}

export default function AircraftLayer({
  globeRadius,
  states,
  hoveredId,
  selectedId,
  ...groupProps
}: Props) {
  const { camera } = useThree();

  const cameraDistance = camera.position.length();
  const sizeFactor = THREE.MathUtils.clamp(cameraDistance / 3, 0.45, 1.1);

  const markers = useMemo(
    () =>
      states
        .filter(
          (s) =>
            Number.isFinite(s.lat) &&
            Number.isFinite(s.lon) &&
            s.lat >= -90 &&
            s.lat <= 90 &&
            s.lon >= -180 &&
            s.lon <= 180,
        )
        .map((s) => {
          const altOff = altitudeOffset(globeRadius, s.altitudeMeters);
          const pos = latLonToVector3(s.lat, s.lon, globeRadius, altOff);
          return { state: s, position: pos };
        }),
    [states, globeRadius],
  );

  return (
    <group {...groupProps}>
      {markers.map(({ state, position }) => {
        const isHovered = hoveredId === state.icao24;
        const isSelected = selectedId === state.icao24;

        const alt = state.altitudeMeters ?? 0;
        const speed = state.velocityMs ?? 0;

        const altFt = alt * 3.28084;
        const speedKt = speed * 1.94384;

        const isFastHigh = altFt > 20000 && speedKt > 300;

        const intensity = isSelected ? 1.25 : isHovered ? 1.0 : 0.8;
        const baseColor = new THREE.Color(
          isFastHigh ? 0x00e5ff : 0xff2b4f,
        );

        const lightBoost = isSelected ? 0.25 : isHovered ? 0.12 : 0;
        baseColor.offsetHSL(0, 0, lightBoost);

        return (
          <group
            key={state.icao24}
            position={position}
            rotation={[Math.PI / 2, 0, 0]}
            userData={{ aircraftId: state.icao24 }}
          >
            <mesh>
              <cylinderGeometry
                args={[
                  globeRadius * 0.004 * intensity * sizeFactor,
                  globeRadius * 0.0015 * intensity * sizeFactor,
                  globeRadius * 0.04 * intensity * sizeFactor,
                  16,
                ]}
              />
              <meshBasicMaterial
                color={baseColor}
                transparent
                opacity={0.8}
                blending={THREE.NormalBlending}
                depthWrite
              />
            </mesh>

            <mesh position={[0, 0, globeRadius * 0.03 * intensity]}>
              <sphereGeometry
                args={[
                  globeRadius * 0.006 * intensity * sizeFactor,
                  16,
                  16,
                ]}
              />
              <meshBasicMaterial
                color={baseColor}
                transparent
                opacity={0.95}
                blending={THREE.AdditiveBlending}
                depthWrite={false}
              />
            </mesh>

            <group
              position={[
                0,
                0,
                globeRadius * 0.02 * intensity * sizeFactor,
              ]}
              rotation={[0, 0, -THREE.MathUtils.degToRad(state.headingDeg ?? 0)]}
            >
              <mesh scale={1.12}>
                <primitive object={planeGeometry} />
                <meshBasicMaterial
                  color={isFastHigh ? "#22d3ee" : "#e11d48"}
                  transparent
                  opacity={0.35}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                />
              </mesh>
              <mesh>
                <primitive object={planeGeometry} />
                <meshBasicMaterial
                  color={isFastHigh ? "#6ee7ff" : "#f973c9"}
                  transparent
                  opacity={0.95}
                  blending={THREE.NormalBlending}
                  depthWrite
                />
              </mesh>
            </group>

            {isSelected && (
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry
                  args={[
                    globeRadius * 0.012 * intensity,
                    globeRadius * 0.016 * intensity,
                    32,
                  ]}
                />
                <meshBasicMaterial
                  color="#ffffff"
                  transparent
                  opacity={0.7}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

