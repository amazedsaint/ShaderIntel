// shaderintel-react: simulations as components. Switching tabs unmounts one simulation (its GPU
// memory is freed) and mounts the next — the same <Canvas>, device and view throughout.
import { useState } from 'react';
import { Canvas, Stage, BoxLines, Cloth, Smoke, Boids, NBody, Spheres, useArray, useView, useFrame } from 'shaderintel-react';
import { mountApp, Panel, Slider, Select, Toggle, Note, Info } from './_ui.jsx';

const CAMERAS = {
  cloth: { target: [0, 0.8, 0], distance: 3.4, pitch: 0.2, yaw: 0.45, autoRotate: 0 },
  smoke: { target: [0, 0.7, 0], distance: 2.9, pitch: 0.15, yaw: 0.5, autoRotate: 0 },
  boids: { target: [0, 0, 0], distance: 3.4, pitch: 0.35, yaw: 0.5, autoRotate: 0.05 },
  galaxies: { target: [0, 0, 0], distance: 3.2, pitch: 0.5, yaw: 0.3, autoRotate: 0.03 },
};

function ClothScene({ wind }) {
  const ball = { type: 'sphere', center: [0, 0.75, 0.35], radius: 0.28 };
  const center = useArray('vec4f', new Float32Array([0, 0.75, 0.35, 0]));
  return (<>
    <Stage />
    <Cloth width={128} height={128} size={[1.6, 1.6]} origin={[-0.8, 1.7, 0]} substeps={20} pin="top-sparse" wind={[0, 0, wind]}
      colliders={[ball, { type: 'box', center: [0.45, 0.25, -0.2], half: [0.25, 0.25, 0.25] }]} pattern="checker" tiles={16} />
    <Spheres data={center} count={1} radius={ball.radius - 0.01} palette={[[0.75, 0.75, 0.78]]} />
    <BoxLines min={[0.2, 0, -0.45]} max={[0.7, 0.5, 0.05]} color={[0.6, 0.65, 0.7, 1]} />
  </>);
}

function SmokeScene({ fire, vorticity }) {
  const preset = fire
    ? { temperature: 2.2, density: 2.5, emission: 6, tint: [0.55, 0.5, 0.48], buoyancy: 2.2, cooling: 1.6 }
    : { temperature: 1.0, density: 3.5, emission: 0, tint: [0.92, 0.94, 0.97], buoyancy: 1.4, cooling: 0.5 };
  return (<>
    <Stage />
    <Smoke res={80} size={[1, 1.5, 1]} iterations={20} vorticity={vorticity} buoyancy={preset.buoyancy} cooling={preset.cooling}
      sources={[{ center: [0, 0.12, 0], radius: 0.09, density: preset.density, temperature: preset.temperature, velocity: [0, 0.6, 0] }]}
      emission={preset.emission} tint={preset.tint} />
  </>);
}

// The flock flees a predator that follows the pointer while a button is held.
function BoidsScene({ count, cohesion }) {
  const view = useView();
  const [predator, setPredator] = useState(null);
  useFrame(() => {
    const p = view.pointer;
    if (p.down && (p.alt || p.ctrl || p.shift)) {
      const r = view.ray();
      const d = Math.hypot(...view.camera.eye().map((e, a) => e - view.camera.target[a]));
      setPredator(r.origin.map((o, a) => o + r.dir[a] * d));
    } else if (predator) setPredator(null);
  });
  return (<>
    <BoxLines min={[-1, -1, -1]} max={[1, 1, 1]} color={[0.2, 0.25, 0.32, 1]} />
    <Boids count={count} radius={0.05} cohesion={cohesion} predator={predator} size={0.0085} />
  </>);
}

function App() {
  const [tab, setTab] = useState('cloth');
  const [wind, setWind] = useState(2.2);
  const [fire, setFire] = useState(true);
  const [vorticity, setVorticity] = useState(0.35);
  const [count, setCount] = useState(131072);
  const [cohesion, setCohesion] = useState(0.9);
  return (
    <Canvas camera={CAMERAS[tab]} bloom={tab === 'galaxies' || tab === 'smoke' ? { strength: tab === 'galaxies' ? 0.12 : 0.06 } : false}
      tonemap={tab === 'galaxies' ? 'aces' : 'agx'} background={tab === 'boids' ? [0.03, 0.04, 0.06, 1] : tab === 'galaxies' ? [0, 0, 0, 1] : [0.02, 0.025, 0.03, 1]} hud>
      {tab === 'cloth' && <ClothScene wind={wind} />}
      {tab === 'smoke' && <SmokeScene fire={fire} vorticity={vorticity} />}
      {tab === 'boids' && <BoidsScene count={count} cohesion={cohesion} />}
      {tab === 'galaxies' && <NBody count={65536} softening={0.025} dt={0.0025} />}
      <Panel title="Simulations">
        <Select label="simulation" value={tab} options={Object.keys(CAMERAS)} onChange={setTab} />
        {tab === 'cloth' && <Slider label="wind" value={wind} min={0} max={8} step={0.1} onChange={setWind} />}
        {tab === 'smoke' && <><Toggle label="fire" value={fire} onChange={setFire} />
          <Slider label="vorticity" value={vorticity} min={0} max={1.5} step={0.01} onChange={setVorticity} /></>}
        {tab === 'boids' && <><Select label="boids" value={String(count)} options={['32768', '131072', '524288']} onChange={(v) => setCount(+v)} />
          <Slider label="cohesion" value={cohesion} min={0} max={4} step={0.05} onChange={setCohesion} /></>}
        <Note>{{ cloth: 'XPBD cloth, 16k particles, 20 substeps, gusty wind.', smoke: 'Eulerian smoke on a staggered 80³ grid, volumetric light.',
          boids: 'Shift / alt-drag: a predator follows the pointer.', galaxies: '65,536 bodies, O(N²) gravity, autotuned for this GPU.' }[tab]}</Note>
      </Panel>
      <Info><b>Simulations as components</b> — <code>&lt;Cloth&gt;</code>, <code>&lt;Smoke&gt;</code>, <code>&lt;Boids&gt;</code>, <code>&lt;NBody&gt;</code>.
        Switching unmounts one (its GPU memory is freed) and mounts the next on the same device.</Info>
    </Canvas>
  );
}

mountApp(App);
