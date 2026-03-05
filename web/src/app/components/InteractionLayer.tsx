"use client";

import { useCallback, useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

type Props = {
  globeRadius: number;
  onHoverChange?: (id: string | null) => void;
  onSelect?: (id: string | null) => void;
};

export default function InteractionLayer({
  onHoverChange,
  onSelect,
}: Props) {
  const { camera, gl, scene } = useThree();
  const raycaster = useRef(new THREE.Raycaster());
  const pointer = useRef(new THREE.Vector2());
  const hoveredIdRef = useRef<string | null>(null);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      const rect = gl.domElement.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      pointer.current.set(x * 2 - 1, -(y * 2 - 1));
    },
    [gl.domElement],
  );

  const handleClick = useCallback(() => {
    if (!hoveredIdRef.current) return;
    onSelect?.(hoveredIdRef.current);
  }, [onSelect]);

  useEffect(() => {
    const el = gl.domElement;
    el.addEventListener("pointermove", handlePointerMove);
    el.addEventListener("click", handleClick);
    return () => {
      el.removeEventListener("pointermove", handlePointerMove);
      el.removeEventListener("click", handleClick);
    };
  }, [gl.domElement, handlePointerMove, handleClick]);

  useFrame(() => {
    raycaster.current.setFromCamera(pointer.current, camera);
    const aircraftParents: THREE.Object3D[] = [];
    scene.traverse((obj) => {
      if (obj.userData.aircraftId) aircraftParents.push(obj);
    });
    const intersects = raycaster.current.intersectObjects(aircraftParents, true);
    const top = intersects[0];
    const id = top
      ? (() => {
          let o: THREE.Object3D | null = top.object;
          while (o) {
            if (o.userData.aircraftId) return o.userData.aircraftId as string;
            o = o.parent;
          }
          return null;
        })()
      : null;

    if (id !== hoveredIdRef.current) {
      hoveredIdRef.current = id;
      onHoverChange?.(id);
      // cursor styling is handled by the parent layout if desired
    }
  });

  return null;
}

