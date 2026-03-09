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

  const [dayMap, bumpMap, lightsMap, specularMap, cloudsMap] = useLoader(
    TextureLoader,
    [
      "/textures/earth_day_4096.jpg",
      "/textures/earth-bump.jpg",
      "/textures/earth_lights_2048.png",
      "/textures/earth_specular_2048.jpg",
      "/textures/earth_atmos_4096.jpg",
    ],
  );

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
          map={dayMap}
          bumpMap={bumpMap}
          bumpScale={0.02}
          specularMap={specularMap}
          specular="#ffffff"
          shininess={30}
          emissive="#facc6b"
          emissiveMap={lightsMap}
          emissiveIntensity={0.85}
        />
      </mesh>

      {/* Subtle glow from specular map along coasts/oceans */}
      <mesh>
        <sphereGeometry args={[radius * 1.001, 96, 96]} />
        <meshBasicMaterial
          map={specularMap}
          color="#1e293b"
          transparent
          opacity={0.2}
          depthWrite={false}
        />
      </mesh>

      <mesh>
        <sphereGeometry args={[radius * 1.01, 96, 96]} />
        <meshPhongMaterial
          map={cloudsMap}
          transparent
          opacity={0.3}
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

