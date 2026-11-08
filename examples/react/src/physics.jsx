// shaderintel-react: a physics playground. Bodies are components; React state drives the world.
import { useEffect, useRef, useState } from 'react';
import { Canvas, Stage, Physics, Box, Sphere, Capsule, Joint, useView, useWorld, useFrame } from 'shaderintel-react';
import { mountApp, Panel, Slider, Select, Toggle, Button, Note, Info } from './_ui.jsx';

// A pyramid of boxes: just a list of <Box> elements.
function Pyramid({ rows = 12, size = 0.5, z = 0 }) {
  const boxes = [];
  for (let row = 0; row < rows; row++)
    for (let k = 0; k < rows - row; k++)
      boxes.push(<Box key={`${row}-${k}`} position={[(k - (rows - row - 1) / 2) * size * 1.02, size / 2 + row * size, z]} size={[size, size, size]} />);
  return <>{boxes}</>;
}

// A wrecking ball on a chain of capsules: each link is jointed to the previous one by its ref.
function WreckingBall({ pivot = [0, 9, 4.5], links = 10, swing = -9 }) {
  const refs = useRef(Array.from({ length: links + 1 }, () => ({ current: null }))).current;
  const ball = refs[links];
  const linkLen = 0.4;
  useEffect(() => { const t = setTimeout(() => ball.current?.setVelocity([0, 0, swing]), 50); return () => clearTimeout(t); }, []);
  return (
    <>
      {Array.from({ length: links }, (_, k) => (
        <Capsule key={k} ref={refs[k]} position={[pivot[0], pivot[1] - linkLen * (k + 0.5), pivot[2]]} radius={0.06} halfHeight={0.14} />
      ))}
      <Sphere ref={ball} position={[pivot[0], pivot[1] - linkLen * links - 0.9, pivot[2]]} radius={0.9} density={4} />
      <Joint type="ball" a={refs[0]} anchor={pivot} />
      {Array.from({ length: links }, (_, k) => (
        <Joint key={k} type="ball" a={refs[k]} b={refs[k + 1]} anchor={[pivot[0], pivot[1] - linkLen * (k + 1), pivot[2]]} />
      ))}
    </>
  );
}

// Click (alt / ⌘ / ctrl-click on desktop, or the button) to fire balls along the pointer ray.
function Cannon({ fire, onShot }) {
  const view = useView();
  const world = useWorld();
  const shoot = () => {
    const { origin, dir } = view.ray();
    world.addSphere({ position: origin.map((o, a) => o + dir[a] * 1.5), radius: 0.35, density: 3, velocity: dir.map((d) => d * 28) });
    onShot();
  };
  useEffect(() => {
    const down = (e) => { if (e.altKey || e.metaKey || e.ctrlKey) shoot(); };
    view.canvas.addEventListener('pointerdown', down);
    return () => view.canvas.removeEventListener('pointerdown', down);
  });
  useEffect(() => { if (fire) shoot(); }, [fire]);
  return null;
}

function Stats({ onStats }) {
  const world = useWorld();
  useFrame(({ frame }) => { if (frame % 30 === 0) onStats({ bodies: world.count, contacts: world.stats.contacts, colors: world.stats.colors }); });
  return null;
}

function App() {
  const [scene, setScene] = useState('pyramid + ball');
  const [gravity, setGravity] = useState(9.81);
  const [iterations, setIterations] = useState(8);
  const [paused, setPaused] = useState(false);
  const [epoch, setEpoch] = useState(0);        // changing the key remounts <Physics>: a fresh world
  const [fire, setFire] = useState(0);
  const [stats, setStats] = useState({ bodies: 0, contacts: 0, colors: 0 });
  const [shots, setShots] = useState(0);
  return (
    <Canvas camera={{ target: [0, 3, 2], distance: 22, yaw: 0.75, pitch: 0.28, far: 400 }} hud>
      <Stage cell={0.5} />
      <Physics key={`${scene}-${epoch}`} gravity={[0, -gravity, 0]} iterations={iterations} paused={paused}>
        <Pyramid rows={scene === 'tall pyramid' ? 18 : 12} />
        {scene !== 'tall pyramid' && <WreckingBall />}
        <Cannon fire={fire} onShot={() => setShots((s) => s + 1)} />
        <Stats onStats={setStats} />
      </Physics>
      <Panel title="Physics playground">
        <Select label="scene" value={scene} options={['pyramid + ball', 'tall pyramid']} onChange={setScene} />
        <Slider label="gravity" value={gravity} min={0} max={20} step={0.01} onChange={setGravity} />
        <Slider label="iterations" value={iterations} min={1} max={24} step={1} onChange={setIterations} />
        <Toggle label="paused" value={paused} onChange={setPaused} />
        <Button onClick={() => setFire((f) => f + 1)}>Fire a ball</Button>
        <Button onClick={() => { setEpoch((e) => e + 1); setShots(0); }}>Reset</Button>
        <Note>{stats.bodies} bodies · {stats.contacts.toLocaleString()} contacts · {stats.colors} colours · {shots} shots</Note>
      </Panel>
      <Info><b>React physics</b> — every box, link and joint is a component (<code>&lt;Box&gt;</code>, <code>&lt;Capsule&gt;</code>,
        <code>&lt;Joint&gt;</code>); React state sets gravity and solver iterations live. <b>Alt / ⌘-click</b> fires a ball.</Info>
    </Canvas>
  );
}

mountApp(App);
