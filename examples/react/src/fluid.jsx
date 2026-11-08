// shaderintel-react: MLS-MPM fluid two-way coupled with rigid bodies. <Fluid> inside <Physics>
// couples automatically: crates float, sink and splash; nothing is scripted.
import { useState } from 'react';
import { Canvas, Stage, BoxLines, Physics, Box, Sphere, Capsule, Fluid, FluidBlock } from 'shaderintel-react';
import { mountApp, Panel, Slider, Select, Button, Note, Info } from './_ui.jsx';

const SIZE = [3, 1.2, 1.5];

// Invisible static walls keep the bodies in the tank (group 2 is not drawn: renderOptions.groups = 1).
function TankWalls({ inset }) {
  const wall = (p, h) => <Box type="static" position={p} half={h} group={2} mask={1} />;
  return (<>
    {wall([-0.25 + inset, 1, SIZE[2] / 2], [0.25, 1, SIZE[2]])}
    {wall([SIZE[0] + 0.25 - inset, 1, SIZE[2] / 2], [0.25, 1, SIZE[2]])}
    {wall([SIZE[0] / 2, 1, -0.25 + inset], [SIZE[0], 1, 0.25])}
    {wall([SIZE[0] / 2, 1, SIZE[2] + 0.25 - inset], [SIZE[0], 1, 0.25])}
  </>);
}

function Crates({ density }) {
  return (<>
    {[0.25, 0.5, 0.75].map((d, k) => <Box key={k} position={[0.6 + k * 0.9, 0.85, 0.75]} size={[0.24, 0.24, 0.24]} density={d * density} group={1} />)}
    <Sphere position={[1.5, 1.0, 0.35]} radius={0.1} density={3} group={1} />
    <Capsule position={[1.0, 0.9, 1.15]} radius={0.05} halfHeight={0.15} rotation={[0, 0, Math.PI / 2]} density={0.6 * density} group={1} />
  </>);
}

function DamCrates() {
  return (<>{Array.from({ length: 18 }, (_, k) => (
    <Box key={k} position={[1.6 + (k % 3) * 0.32, 0.12 + Math.floor(k / 3) * 0.24, 0.45 + (k % 2) * 0.5]} size={[0.22, 0.22, 0.22]} density={0.6} group={1} />
  ))}</>);
}

function App() {
  const [scene, setScene] = useState('floating crates');
  const [material, setMaterial] = useState('water');
  const [density, setDensity] = useState(1);
  const [grid, setGrid] = useState(64);
  const [epoch, setEpoch] = useState(0);
  const inset = (SIZE[0] / grid) * 3;   // the fluid keeps 3 cells of boundary
  return (
    <Canvas camera={{ target: [1.5, 0.45, 0.75], distance: 3.6, pitch: 0.42, yaw: 0.55 }} hud>
      <Stage cell={0.25} />
      <BoxLines min={[0, 0, 0]} max={SIZE} />
      <Physics key={`${scene}-${grid}-${epoch}`} maxBodies={2048} bounds={[[-1, -1, -1], [4, 4, 3]]} renderOptions={{ groups: 1 }}>
        <Fluid grid={grid} size={SIZE} maxParticles={1 << 20} substeps={8} couple={{ friction: 0.2 }}>
          {scene === 'dam break'
            ? <FluidBlock box={[[0.05, 0.05, 0.05], [0.9, 0.9, 1.45]]} material={material} />
            : <FluidBlock box={[[0.05, 0.05, 0.05], [2.95, 0.45, 1.45]]} material={material} />}
        </Fluid>
        <TankWalls inset={inset} />
        {scene === 'dam break' ? <DamCrates /> : <Crates density={density} />}
      </Physics>
      <Panel title="Fluid + rigid bodies">
        <Select label="scene" value={scene} options={['floating crates', 'dam break']} onChange={setScene} />
        <Select label="material" value={material} options={['water', 'oil', 'honey', 'sand', 'snow']} onChange={setMaterial} />
        <Slider label="crate density ×" value={density} min={0.5} max={4} step={0.05} onChange={setDensity} />
        <Select label="grid" value={String(grid)} options={['48', '64', '96']} onChange={(v) => setGrid(+v)} />
        <Button onClick={() => setEpoch((e) => e + 1)}>Reset</Button>
        <Note>Densities are relative to the fluid: crates at 0.25, 0.5 and 0.75 float at those fractions of their height (Archimedes); the ball sinks.</Note>
      </Panel>
      <Info><b>Two-way coupling</b> — <code>&lt;Fluid&gt;</code> inside <code>&lt;Physics&gt;</code>: rigid bodies are moving boundaries of the
        MPM grid and receive the momentum the fluid loses. Changing the material or a block restarts the fluid.</Info>
    </Canvas>
  );
}

mountApp(App);
