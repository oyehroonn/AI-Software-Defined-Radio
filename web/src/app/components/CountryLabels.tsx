"use client";

import { Html, useThree } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

type Label = {
  name: string;
  lat: number;
  lon: number;
};

const COUNTRY_LABELS: Label[] = [
  { name: "USA", lat: 39.8, lon: -98.6 },
  { name: "Canada", lat: 62.0, lon: -96.0 },
  { name: "Brazil", lat: -14.2, lon: -51.9 },
  { name: "UK", lat: 55.0, lon: -3.0 },
  { name: "Germany", lat: 51.0, lon: 10.0 },
  { name: "India", lat: 22.4, lon: 79.0 },
  { name: "China", lat: 35.0, lon: 103.0 },
  { name: "Japan", lat: 36.2, lon: 138.3 },
  { name: "Australia", lat: -25.0, lon: 133.0 },
  { name: "South Africa", lat: -30.6, lon: 22.9 },
];

function latLonToVector3(latDeg: number, lonDeg: number, radius: number) {
  const lat = THREE.MathUtils.degToRad(latDeg);
  const lon = THREE.MathUtils.degToRad(lonDeg);
  const x = radius * Math.cos(lat) * Math.cos(lon);
  const y = radius * Math.sin(lat);
  const z = radius * Math.cos(lat) * Math.sin(lon);
  return new THREE.Vector3(x, y, z);
}

export function CountryLabels({ radius = 1 }: { radius?: number }) {
  const { camera } = useThree();
  const cameraDistance = camera.position.length();

  const showLabels = cameraDistance < 2.3;

  const labelPositions = useMemo(
    () =>
      COUNTRY_LABELS.map((l) => ({
        ...l,
        position: latLonToVector3(l.lat, l.lon, radius * 1.01),
      })),
    [radius],
  );

  if (!showLabels) return null;

  return (
    <group>
      {labelPositions.map(({ name, position }) => (
        <Html
          key={name}
          position={position.toArray()}
          distanceFactor={5}
          style={{
            pointerEvents: "none",
            fontSize: "10px",
            color: "#e5f6ff",
            textShadow: "0 0 4px rgba(15,23,42,0.9)",
            background: "rgba(15,23,42,0.8)",
            padding: "2px 6px",
            borderRadius: "9999px",
            border: "1px solid rgba(250,204,21,0.7)",
          }}
        >
          {name}
        </Html>
      ))}
    </group>
  );
}

