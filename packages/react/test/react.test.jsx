// shaderintel-react tests: mounted with React (development build, so StrictMode really double-mounts)
// on a real GPU. Run headless: node tools/headless.mjs tests --url http://localhost:8787/packages/react/test/
import { StrictMode, createRef, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import * as SI from 'shaderintel';
import {
  Canvas, useGPU, useView, useFrame, useArray, useKernel, Stage, Points, Physics, Box, Sphere, Joint, Fluid, FluidBlock,
  Fullscreen, SdfScene, Boids, Cloth, Smoke, NBody, useWorld, DEVICE_LINGER_MS,
} from 'shaderintel-react';
import { test, section, note, assert, finish } from '../../../tests/harness.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const frames = async (n) => { for (let k = 0; k < n; k++) await new Promise((r) => requestAnimationFrame(() => r())); };
async function until(cond, ms = 5000, what = 'condition') {
  const t0 = performance.now();
  while (!(await cond())) { if (performance.now() - t0 > ms) throw new Error(`timed out waiting for ${what}`); await sleep(16); }
}

/** Render `element` into a fresh 320×200 box; returns { root, el, unmount }. */
function mount(element) {
  const el = document.createElement('div');
  Object.assign(el.style, { width: '320px', height: '200px', position: 'relative' });
  document.getElementById('stage').appendChild(el);
  const root = createRoot(el);
  flushSync(() => root.render(element));
  return { root, el, unmount() { flushSync(() => root.unmount()); el.remove(); } };
}

let errors = 0;
window.addEventListener('error', () => errors++);

async function run() {
  section('Canvas');

  await test('Canvas initialises the GPU, renders children, runs frame callbacks and cleans up on unmount', async () => {
    let created = null, frameCount = 0, seen = null;
    function Probe() {
      const gpu = useGPU(), view = useView();
      seen = { gpu, view };
      useFrame(() => { frameCount++; });
      return null;
    }
    const m = mount(<Canvas onCreated={(r) => { created = r; }}><Probe /></Canvas>);
    await until(() => created && frameCount > 3, 8000, 'frames');
    assert(seen.gpu === created.gpu && seen.view === created.view, 'hooks see the root');
    assert(m.el.querySelector('canvas').width > 0, 'canvas sized');
    const view = created.view;
    m.unmount();
    assert(view.destroyed, 'view not destroyed on unmount');
    const n = frameCount;
    await frames(3);
    assert(frameCount === n, 'frame callbacks ran after unmount');
  });

  await test('StrictMode: one view, frame callbacks run once per frame, nothing destroyed that is still in use', async () => {
    let calls = 0, views = new Set(), worlds = new Set();
    function Probe() {
      views.add(useView());
      useFrame(() => { calls++; });
      return null;
    }
    function WorldProbe() { worlds.add(useWorld()); return null; }
    const box = createRef();
    const m = mount(
      <StrictMode>
        <Canvas>
          <Probe />
          <Physics>
            <WorldProbe />
            <Box ref={box} position={[0, 2, 0]} />
            <Sphere position={[1, 3, 0]} radius={0.3} />
          </Physics>
        </Canvas>
      </StrictMode>);
    await until(() => calls > 5 && box.current?.id >= 0, 8000, 'frames');
    await sleep(50);
    const before = calls;
    await frames(10);
    const per = (calls - before) / 10;
    assert(per > 0.5 && per < 1.5, `frame callback ran ${per.toFixed(2)}× per frame`);
    assert(views.size === 1, `${views.size} views`);
    const live = box.current.world;
    assert(!live.destroyed, 'the live world was destroyed by the StrictMode remount');
    assert(live.count === 2 && live.free.length === 0, `world has ${live.count} slots, free ${live.free}`);
    note(`worlds created during StrictMode renders: ${worlds.size} (orphans are dev-only and GC'd)`);
    m.unmount();
    await sleep(20);
    assert(live.destroyed, 'world not destroyed after unmount');
  });

  await test('fallback is shown when WebGPU is unavailable', async () => {
    const desc = Object.getOwnPropertyDescriptor(navigator, 'gpu');
    Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
    let reported = null;
    try {
      const m = mount(<Canvas init={{ powerPreference: 'low-power' }} onError={(e) => { reported = e; }}
        fallback={(e) => <p className="fb">no gpu: {e.code}</p>}><Stage /></Canvas>);
      await until(() => m.el.querySelector('.fb'), 3000, 'fallback');
      assert(m.el.querySelector('.fb').textContent === 'no gpu: unsupported', m.el.querySelector('.fb').textContent);
      assert(reported instanceof SI.SIError && reported.code === 'unsupported', 'onError');
      m.unmount();
    } finally {
      if (desc) Object.defineProperty(navigator, 'gpu', desc); else delete navigator.gpu;
    }
  });

  await test('several canvases share one device; it is released after the last unmounts', async () => {
    const roots = [];
    const a = mount(<Canvas onCreated={(r) => roots.push(r)} />);
    const b = mount(<Canvas onCreated={(r) => roots.push(r)} />);
    await until(() => roots.length === 2, 8000, 'two canvases');
    assert(roots[0].gpu === roots[1].gpu, 'two devices');
    const gpu = roots[0].gpu;
    a.unmount(); b.unmount();
    assert(!gpu.destroyed, 'device destroyed immediately (re-mounts could not reuse it)');
    await sleep(DEVICE_LINGER_MS + 300);
    assert(gpu.destroyed, 'device not released');
  });

  section('hooks');

  await test('useArray + useKernel + useFrame: a kernel runs every frame on an array owned by the component', async () => {
    let arr = null, view = null;
    function Counter() {
      const a = useArray('u32', 256);
      const k = useKernel({ buffers: { a }, code: 'a[i] = a[i] + 1u;' }, [a]);
      useFrame(() => k.run(256));
      arr = a; view = useView();
      return null;
    }
    const m = mount(<Canvas><Counter /></Canvas>);
    await until(() => view && view.frame > 10, 8000, 'frames');
    const r = await arr.read();
    assert(r[0] > 5 && r[0] === r[255], `counter ${r[0]}`);
    m.unmount();
    await sleep(20);
    assert(arr.destroyed, 'array not destroyed after unmount');
  });

  await test('useArray uploads new data of the same length without re-creating the array', async () => {
    let set = null, arr = [];
    function Data() {
      const [data, setData] = useState(() => new Float32Array([1, 2, 3, 4]));
      set = setData;
      arr.push(useArray('f32', data));
      return null;
    }
    const m = mount(<Canvas><Data /></Canvas>);
    await until(() => arr.length, 8000, 'mount');
    flushSync(() => set(new Float32Array([5, 6, 7, 8])));
    await sleep(20);
    const last = arr[arr.length - 1];
    assert(arr.every((a) => a === last), 'array re-created');
    const r = await last.read();
    assert(r[0] === 5 && r[3] === 8, `read ${Array.from(r)}`);
    m.unmount();
  });

  section('layers');

  await test('layers are added in JSX order and removed on unmount; live props update uniforms', async () => {
    let view = null, setSize = null, pts = createRef(), stage = createRef();
    function Scene() {
      const [show, setShow] = useState(true);
      const [size, s] = useState(0.01);
      setSize = (v) => s(v); scene.hide = () => setShow(false);
      const data = useArray('vec4f', 64);
      view = useView();
      return (<>
        <Stage ref={stage} cell={0.5} />
        {show && <Points ref={pts} data={data} size={size} />}
        <Fullscreen code={'fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(uv, 0.0, 0.0); }'} />
      </>);
    }
    const scene = {};
    const m = mount(<Canvas><Scene /></Canvas>);
    await until(() => view && view.frame > 2, 8000, 'frames');
    assert(view.layers.length === 3 && view.layers[0] === stage.current && view.layers[1] === pts.current, 'order');
    flushSync(() => setSize(0.05));
    assert(Math.abs(new DataView(pts.current.program.ubytes).getFloat32(pts.current.program.fieldByName.get('size').offset, true) - 0.05) < 1e-7, 'size uniform');
    flushSync(() => scene.hide());
    assert(view.layers.length === 2 && !view.layers.includes(pts.current), 'layer not removed');
    m.unmount();
  });

  section('physics');

  await test('Physics: a box falls and lands on the floor; the ref API moves bodies', async () => {
    const box = createRef(), ball = createRef(), world = createRef();
    const m = mount(<Canvas><Physics ref={world}><Box ref={box} position={[0, 3, 0]} size={[1, 1, 1]} /><Sphere ref={ball} position={[3, 0.5, 0]} radius={0.5} /></Physics></Canvas>);
    await until(() => world.current && box.current?.id >= 0 && world.current.frame > 150, 15000, '150 steps');
    const p = await world.current.readPoses();
    const y = p[box.current.id * 7 + 1];
    assert(Math.abs(y - 0.5) < 0.01, `box rests at y = ${y.toFixed(4)}`);
    ball.current.setVelocity([0, 0, 3]);
    await until(async () => { const q = await world.current.readPoses(); return q[ball.current.id * 7 + 2] > 0.5; }, 5000, 'ball to move');
    m.unmount();
  });

  await test('Physics: unmounting a body frees its slot; a new body reuses it; static and kinematic types', async () => {
    let toggle = null;
    const world = createRef(), kin = createRef();
    function Scene() {
      const [on, setOn] = useState(true);
      const [x, setX] = useState(0);
      toggle = { off: () => setOn(false), on: () => setOn(true), move: (v) => setX(v) };
      return (<Physics ref={world}>
        <Box type="static" position={[0, -0.5, 0]} size={[10, 1, 10]} />
        {on && <Box position={[0, 2, 0]} />}
        <Box ref={kin} type="kinematic" position={[x, 1, 3]} size={[0.5, 0.5, 0.5]} />
      </Physics>);
    }
    const m = mount(<Canvas><Scene /></Canvas>);
    await until(() => world.current?.frame > 5, 8000, 'steps');
    assert(world.current.count === 3, `count ${world.current.count}`);
    flushSync(() => toggle.off());
    assert(world.current.free.length === 1, 'slot not freed');
    flushSync(() => toggle.on());
    assert(world.current.count === 3 && world.current.free.length === 0, 'slot not reused');
    flushSync(() => toggle.move(2));
    await until(async () => { const q = await world.current.readPoses(); return Math.abs(q[kin.current.id * 7] - 2) < 1e-3; }, 5000, 'kinematic body to follow its props');
    m.unmount();
  });

  await test('Joint: declared before its bodies, resolved once they exist; a pendulum swings', async () => {
    const a = createRef(), world = createRef();
    const m = mount(<Canvas><Physics ref={world} floor={null}>
      <Joint type="ball" a={a} b={null} anchor={[0, 3, 0]} />
      <Sphere ref={a} position={[1, 3, 0]} radius={0.1} />
    </Physics></Canvas>);
    await until(() => world.current?.joints.length === 1 && world.current.frame > 30, 8000, 'joint');
    const p = await world.current.readPoses();
    const i = a.current.id * 7;
    const len = Math.hypot(p[i], p[i + 1] - 3, p[i + 2]);
    assert(Math.abs(len - 1) < 0.01 && p[i + 1] < 3, `length ${len.toFixed(4)}, y ${p[i + 1].toFixed(3)}`);
    m.unmount();
  });

  section('simulations');

  await test('Fluid fills from FluidBlocks and couples with Physics; Cloth, Smoke, Boids and NBody step', async () => {
    const fluid = createRef(), cloth = createRef(), smoke = createRef(), boids = createRef(), nbody = createRef();
    let view = null;
    function V() { view = useView(); return null; }
    const m = mount(<Canvas>
      <V />
      <Physics>
        <Fluid ref={fluid} grid={32} maxParticles={1 << 16}>
          <FluidBlock box={[[0.1, 0.1, 0.1], [0.5, 0.4, 0.5]]} material="water" />
          <FluidBlock sphere={[[0.7, 0.6, 0.5], 0.1]} material="jelly" />
        </Fluid>
        <Box position={[0.3, 0.7, 0.3]} size={[0.1, 0.1, 0.1]} density={0.5} />
      </Physics>
      <Cloth ref={cloth} width={32} height={32} />
      <Smoke ref={smoke} res={32} />
      <Boids ref={boids} count={4096} />
      <NBody ref={nbody} count={1024} autotune={false} />
    </Canvas>);
    await until(() => view && view.frame > 20, 15000, 'frames');
    assert(fluid.current.count > 1000, `fluid particles ${fluid.current.count}`);
    assert(fluid.current.coupling, 'fluid not coupled to the world');
    const pos = await cloth.current.pos.read(4);
    assert(Number.isFinite(pos[0]), 'cloth');
    assert(smoke.current.time > 0, 'smoke time');
    const sims = { fluid: fluid.current, cloth: cloth.current, smoke: smoke.current, boids: boids.current, nbody: nbody.current };   // refs are nulled on unmount
    m.unmount();
    await sleep(30);
    for (const [name, sim] of Object.entries(sims)) assert(sim.destroyed, `${name} not destroyed`);
  });

  await test('SdfScene and Fullscreen accept live uniform values', async () => {
    const sdf = createRef();
    let set = null, view = null;
    function Scene() {
      const [r, setR] = useState(0.5);
      set = setR; view = useView();
      return <SdfScene ref={sdf} uniforms={{ r: 'f32' }} values={{ r }} code={'fn map(p: vec3f) -> vec2f { return vec2f(sdSphere(p, u.r), 1.0); }'} />;
    }
    const m = mount(<Canvas><Scene /></Canvas>);
    await until(() => view && view.frame > 3, 8000, 'frames');
    flushSync(() => set(0.8));
    assert(sdf.current.values.r === 0.8, `r = ${sdf.current.values.r}`);
    m.unmount();
  });

  note(`window errors: ${errors}`);
  await test('no uncaught errors', () => assert(errors === 0, `${errors} uncaught errors`));
}

run().catch((e) => { section('fatal'); note(String(e.stack || e)); window.__results.failed++; window.__results.failures.push({ name: 'fatal', error: String(e) }); })
  .finally(finish);
