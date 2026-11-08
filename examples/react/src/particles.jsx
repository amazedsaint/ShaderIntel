// shaderintel-react: your own GPU compute in React. useArray allocates GPU memory owned by the
// component, useKernel compiles a WGSL body once, useFrame runs it every frame, <Points> draws it.
import { useLayoutEffect, useState } from 'react';
import { Canvas, Points, useArray, useKernel, useFrame } from 'shaderintel-react';
import { mountApp, Panel, Slider, Select, Button, Note, Info } from './_ui.jsx';

function CurlFlow({ count, speed, scale, shape, seed }) {
  const pos = useArray('vec4f', count);   // a new count re-creates the array (and everything that uses it)

  const spawn = useKernel({
    imports: ['hash'], buffers: { pos }, uniforms: { seed: 'u32', shape: 'u32' },
    code: /* wgsl */`
      var s = rngSeed(i, u.seed);
      var p = randBall(&s);
      if (u.shape == 1u) { p = vec3f(p.x, p.y * 0.08, p.z); }                         // a disc
      if (u.shape == 2u) { p = normalize(p) * (0.9 + 0.1 * rand(&s)); }              // a shell
      pos[i] = vec4f(p, length(p));`,
  }, [pos]);

  const flow = useKernel({
    imports: ['hash', 'noise'], buffers: { pos }, uniforms: { time: 'f32', dt: 'f32', frame: 'u32', speed: 'f32', scale: 'f32' },
    code: /* wgsl */`
      var p = pos[i];
      let v = curl3(p.xyz * u.scale + vec3f(0.0, u.time * 0.05, 0.0));   // divergence-free: no clumping
      p = vec4f(p.xyz + v * (u.speed * u.dt), p.w);
      if (dot(p.xyz, p.xyz) > 1.8) {                                    // respawn what drifts away
        var s = rngSeed(i, u.frame);
        let q = randBall(&s);
        p = vec4f(q * 0.5, length(q));
      }
      pos[i] = p;`,
  }, [pos]);

  // (re-)spawn when the array, the shape or the seed changes — before the next frame is drawn
  useLayoutEffect(() => { spawn.run(count, { seed, shape: { ball: 0, disc: 1, shell: 2 }[shape] }); }, [spawn, count, shape, seed]);

  useFrame(({ dt, time, frame }) => flow.run(count, { time, dt, frame, speed, scale }));
  return <Points data={pos} size={0.002} pixelSize={1} colorBy="w" intensity={Math.min(0.6, 0.22 * 1e6 / count)} />;
}

function App() {
  const [count, setCount] = useState(1_000_000);
  const [speed, setSpeed] = useState(0.12);
  const [scale, setScale] = useState(0.9);
  const [shape, setShape] = useState('ball');
  const [seed, setSeed] = useState(1);
  return (
    <Canvas camera={{ distance: 4.2, pitch: 0.3, autoRotate: 0.08 }} bloom={{ strength: 0.08 }} background={[0, 0, 0, 1]} hud>
      <CurlFlow count={count} speed={speed} scale={scale} shape={shape} seed={seed} />
      <Panel title="Compute particles (hooks)">
        <Select label="particles" value={String(count)} options={['100000', '1000000', '4000000']} onChange={(v) => setCount(+v)} />
        <Select label="start shape" value={shape} options={['ball', 'disc', 'shell']} onChange={setShape} />
        <Slider label="speed" value={speed} min={0} max={0.5} step={0.01} onChange={setSpeed} />
        <Slider label="noise scale" value={scale} min={0.2} max={3} step={0.05} onChange={setScale} />
        <Button onClick={() => setSeed((s) => s + 1)}>Respawn</Button>
        <Note>{count.toLocaleString()} particles advected through curl noise, one compute dispatch per frame.</Note>
      </Panel>
      <Info><b>Hooks</b> — <code>useArray</code> + <code>useKernel</code> + <code>useFrame</code>: a WGSL body is all you write; bindings, uniforms
        and dispatch sizes are generated. Changing the count re-allocates the GPU array; the old one is freed.</Info>
    </Canvas>
  );
}

mountApp(App);
