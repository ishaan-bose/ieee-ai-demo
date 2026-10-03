import { Canvas, useFrame } from "@react-three/fiber";
import { Component, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import { loss } from "./Valley";

// Hero visual 3: a ball rolling down a 3-D valley. A CARTOON of the real loss landscape: the height only depends on one knob (x),
// the second horizontal direction is just for looks. Falls back to the 2-D version if WebGL is unavailable (see onError).
const X0 = -2.2, X1 = 2.2, Y0 = -1.4, Y1 = 1.4;
export const height = (x: number, y: number) => 0.5 * Math.min(6, loss(Math.max(-6, Math.min(6, x))) + 0.1) + 0.28 * y * y;

function Surface() {
  const geo = useMemo(() => {
    const nx = 90, ny = 40;
    const g = new THREE.PlaneGeometry(X1 - X0, Y1 - Y0, nx, ny);
    g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position, col = new Float32Array(pos.count * 3), c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i), h = height(x, z);
      pos.setY(i, h);
      c.setHSL(0.62 - Math.min(1, h / 3.2) * 0.58, 0.75, 0.32 + Math.min(0.2, h * 0.05)); // blue at the bottom, orange up high
      col.set([c.r, c.g, c.b], i * 3);
    }
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    return g;
  }, []);
  return <mesh geometry={geo}><meshStandardMaterial vertexColors flatShading side={THREE.DoubleSide} roughness={0.85} /></mesh>;
}

function Ball({ w }: { w: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((_, dt) => {
    const m = ref.current;
    if (!m) return;
    const x = Math.max(-5, Math.min(5, w)), y = height(x, 0) + 0.16;
    m.position.x += (x - m.position.x) * Math.min(1, dt * 10);
    m.position.y += (y - m.position.y) * Math.min(1, dt * 10);
    m.rotation.z -= (x - m.position.x) * 0.2;
  });
  return <mesh ref={ref} position={[-1.4, 1.5, 0]}><sphereGeometry args={[0.16, 24, 24]} /><meshStandardMaterial color="#fbbf24" emissive="#f59e0b" emissiveIntensity={0.6} /></mesh>;
}

function Trail({ trail }: { trail: number[] }) {
  const obj = useMemo(() => {
    const pts = trail.map((x) => { const cx = Math.max(-5, Math.min(5, x)); return new THREE.Vector3(cx, height(cx, 0) + 0.19, 0); });
    const g = new THREE.BufferGeometry().setFromPoints(pts.length > 1 ? pts : [new THREE.Vector3(), new THREE.Vector3()]);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color: "#fde68a" }));
  }, [trail]);
  return <primitive object={obj} />;
}

function Rig() {
  useFrame(({ camera, clock }) => {
    const t = clock.elapsedTime * 0.25;
    camera.position.set(Math.sin(t) * 1.6, 3.0, 5.6);
    camera.lookAt(0, 0.7, 0);
  });
  return null;
}

class Boundary extends Component<{ onError: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  componentDidCatch() { this.props.onError(); this.setState({ failed: true }); }
  render() { return this.state.failed ? null : this.props.children; }
}

function hasWebGL(): boolean {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; }
}

export function Valley3D({ w, trail, onError }: { w: number; trail: number[]; onError: () => void }) {
  if (!hasWebGL()) { queueMicrotask(onError); return null; }
  return (
    <Boundary onError={onError}>
      <div className="h-[300px] w-full overflow-hidden rounded-2xl bg-slate-900/60" data-testid="valley3d">
        <Canvas camera={{ fov: 44, position: [0, 3.0, 5.6] }} onCreated={({ gl }) => gl.setClearColor("#0b1220")}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 5, 2]} intensity={1.6} />
          <Surface /><Ball w={w} /><Trail trail={trail} /><Rig />
        </Canvas>
      </div>
    </Boundary>
  );
}
