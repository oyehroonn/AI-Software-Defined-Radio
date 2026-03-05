"use client";

import React, { useRef } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { Mesh, TextureLoader } from "three";

type GlobeProps = {
  radius: number;
};

export default function Globe({ radius }: GlobeProps) {
  const earthRef = useRef<Mesh>(null);
  const atmosphereRef = useRef<Mesh>(null);

  const [earthMap, bumpMap, cloudsMap] = useLoader(TextureLoader, [
    "/textures/earth-night.jpg",
    "/textures/earth-bump.jpg",
    "/textures/earth-clouds.png",
  ]);

  useFrame((_state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.02;
    }
    if (atmosphereRef.current) {
      atmosphereRef.current.rotation.y += delta * 0.01;
    }
  });

  return (
    <group>
      <mesh ref={earthRef}>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshPhongMaterial
          map={earthMap}
          bumpMap={bumpMap}
          bumpScale={0.015}
          specular="#2424ff"
          shininess={20}
        />
      </mesh>

      <mesh>
        <sphereGeometry args={[radius * 1.01, 96, 96]} />
        <meshPhongMaterial
          map={cloudsMap}
          transparent
          opacity={0.25}
          depthWrite={false}
        />
      </mesh>

      <mesh ref={atmosphereRef} scale={[1.06, 1.06, 1.06]}>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshBasicMaterial
          color="#29f3ff"
          side={1}
          transparent
          opacity={0.22}
        />
      </mesh>
    </group>
  );
}

