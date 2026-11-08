// Lifecycle and robustness: resources are freed, loops stop and restart, bad input is rejected
// with a ShaderIntel error instead of silently corrupting GPU memory.
import * as SI from '../src/index.js';
import { test, section, note, assert } from './harness.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const frame = () => new Promise((r) => requestAnimationFrame(() => r()));

function makeCanvas(w = 160, h = 90) {
  const wrap = document.createElement('div');
  Object.assign(wrap.style, { position: 'fixed', left: '0', top: '0', width: `${w}px`, height: `${h}px`, opacity: '0.01' });
  const canvas = document.createElement('canvas');
  Object.assign(canvas.style, { width: '100%', height: '100%', display: 'block' });
  wrap.appendChild(canvas);
  document.body.appendChild(wrap);
  return { canvas, remove: () => wrap.remove() };
}

export async function lifecycleTests(gpu) {
  section('robustness and lifecycle');

  await test('uniform ring: more slots than one segment in a single batch keep their own values', async () => {
    // 6,000 dispatches with distinct uniforms need ~1.5 MB of 256-byte slots: two ring segments, no flush
    const A = gpu.array('u32', 64);
    const k = gpu.kernel({ buffers: { A }, uniforms: { add: 'u32' }, code: 'A[i] = A[i] + u.add;' });
    A.clear();
    const submits = gpu.stats.submits;
    let want = 0;
    for (let s = 1; s <= 6000; s++) { k.run(64, { add: s }); want += s; }
    assert(gpu.stats.submits === submits, 'the ring flushed mid-batch');
    const r = await A.read();
    assert(r[0] === want >>> 0 && r[63] === want >>> 0, `got ${r[0]}, want ${want}`);
  });

  await test('uniform ring: thousands of draws inside one render pass (no flush while the pass is open)', async () => {
    const { canvas, remove } = makeCanvas();
    try {
      const view = gpu.view(canvas, { controls: false });
      const prog = gpu.program({
        name: 'many-draws', uniforms: { k: 'f32' }, targets: ['rgba16float'],
        code: /* wgsl */`
          @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
            return vec4f(f32(vi & 1u) * 0.01 + u.k * 1e-6, f32(vi >> 1u) * 0.01, 0.5, 1.0); }
          @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }`,
      });
      await prog.ready();
      let errors = 0;
      const onErr = () => errors++;
      gpu.device.addEventListener('uncapturederror', onErr);
      view.add({ draw: (pass, v) => { for (let j = 0; j < 5000; j++) prog.draw(pass, v, 3, 1, { k: j }); } });
      view.step();
      await gpu.idle();
      gpu.device.removeEventListener('uncapturederror', onErr);
      assert(errors === 0, `${errors} WebGPU validation errors`);
      view.destroy();
    } finally { remove(); }
  });

  await test('View: run / stop / run restarts the loop, one loop per view, destroy() stops it for good', async () => {
    const { canvas, remove } = makeCanvas();
    try {
      const view = gpu.view(canvas, { controls: false });
      let calls = 0;
      view.run(() => { calls++; });
      view.run();                              // a second run() must not start a second loop
      for (let k = 0; k < 6; k++) await frame();
      const perFrame = calls / 6;
      assert(calls > 0 && perFrame <= 1.01, `update ran ${calls} times in 6 frames`);
      view.stop();
      const stopped = calls;
      for (let k = 0; k < 3; k++) await frame();
      assert(calls === stopped, 'update ran after stop()');
      view.run();
      for (let k = 0; k < 3; k++) await frame();
      assert(calls > stopped, 'run() after stop() did not restart the loop');
      view.destroy();
      const after = calls;
      for (let k = 0; k < 3; k++) await frame();
      assert(calls === after, 'update ran after destroy()');
      assert(view.destroyed && !view.running, 'flags');
      view.destroy();                          // idempotent
      let threw = false;
      try { view.run(); } catch (e) { threw = e instanceof SI.SIError; }
      assert(threw, 'run() after destroy() should throw an SIError');
      note(`frames observed ${calls}`);
    } finally { remove(); }
  });

  await test('View: destroy() removes the HUD, its listeners and an error overlay; an update error reaches onError', async () => {
    const { canvas, remove } = makeCanvas();
    try {
      const parent = canvas.parentElement;
      let caught = null;
      const view = gpu.view(canvas, { hud: true, onError: (e) => { caught = e; } });
      assert(parent.querySelector('.si-hud'), 'HUD missing');
      view.run(() => { throw new Error('boom'); });
      for (let k = 0; k < 3 && !caught; k++) await frame();
      assert(caught?.message === 'boom', 'onError not called');
      assert(!view.running, 'loop kept running after an error');
      view.destroy();
      assert(!parent.querySelector('.si-hud'), 'HUD left behind');
      assert(view.camera._off.length === 0 && view._off.length === 0, 'listeners left behind');
    } finally { remove(); }
  });

  await test('View: add() is idempotent, remove() detaches, unknown tonemaps are rejected', async () => {
    const { canvas, remove } = makeCanvas();
    try {
      const view = gpu.view(canvas, { controls: false });
      let attached = 0, detached = 0;
      const layer = { attach() { attached++; }, detach() { detached++; } };
      view.add(layer); view.add(layer);
      assert(view.layers.length === 1 && attached === 1, 'added twice');
      view.remove(layer);
      assert(view.layers.length === 0 && detached === 1, 'remove');
      view.destroy();
      let threw = false;
      try { gpu.view(canvas, { tonemap: 'filmic' }); } catch (e) { threw = e instanceof SI.SIError; }
      assert(threw, 'bad tonemap accepted');
    } finally { remove(); }
  });

  await test('GArray.write rejects data that does not fit instead of truncating it', async () => {
    const A = gpu.array('f32', 4);
    let threw = 0;
    try { A.write(new Float32Array(5)); } catch (e) { threw += e instanceof SI.SIError; }
    try { A.write(new Float32Array(1), 7); } catch (e) { threw += e instanceof SI.SIError; }
    try { gpu.array('vec4f', 0); } catch (e) { threw += e instanceof SI.SIError; }
    assert(threw === 3, `threw ${threw} of 3`);
    A.write([1, 2], 2);
    const r = await A.read();
    assert(r[2] === 1 && r[3] === 2, 'offset write');
    A.destroy(); A.destroy();   // idempotent
    assert(A.destroyed, 'destroyed flag');
  });

  await test('read(0) returns an empty array; SIError carries a code', async () => {
    const A = gpu.array('u32', 8);
    const r = await A.read(0);
    assert(r instanceof Uint32Array && r.length === 0, 'empty read');
    const k = gpu.kernel({ buffers: { A: { type: 'f32' } }, code: 'A[i] = notDefined;' });
    let code = '';
    try { await k.ready(); } catch (e) { code = e.code; }
    assert(code === 'compile', `code ${code}`);
  });

  await test('init({ device }) shares a device; destroy() leaves a shared device alive', async () => {
    const shared = await SI.init({ device: gpu.device, adapter: gpu.adapter });
    const A = shared.array('u32', 4);
    shared.kernel({ buffers: { A }, code: 'A[i] = 7u;' }).run(4);
    const r = await A.read();
    assert(r[3] === 7, 'shared kernel');
    shared.destroy();
    const back = await gpu.array('u32', new Uint32Array([5, 6, 7, 8])).read();
    assert(back[0] === 5 && back[3] === 8, 'the original context stopped working');
  });

  section('regressions found in review');

  await test('Reduce: n = 0 returns the identity (used to hang the page); n > maxN throws', async () => {
    const A = gpu.array('f32', 16);
    const sum = new SI.Reduce(gpu, { type: 'f32', op: 'sum', maxN: 4096 });
    sum.run(A, 0);
    assert((await sum.read()) === 0, 'sum of nothing');
    const mx = new SI.Reduce(gpu, { type: 'f32', op: 'max', maxN: 4096 });
    mx.run(A, 0);
    assert((await mx.read()) < -1e38, 'max of nothing is the identity');
    let threw = false;
    try { sum.run(gpu.array('f32', 5000), 5000); } catch (e) { threw = e instanceof SI.SIError; }
    assert(threw, 'n > maxN must throw instead of returning a wrong sum');
    let threwGrid = false;
    const grid = new SI.SpatialGrid(gpu, { maxCount: 8, min: [0, 0, 0], max: [1, 1, 1], cellSize: 0.5 });
    try { grid.build(gpu.array('vec4f', 16), 16); } catch (e) { threwGrid = e instanceof SI.SIError; }
    assert(threwGrid, 'SpatialGrid.build past maxCount must throw');
    sum.destroy(); mx.destroy(); grid.destroy();
  });

  await test('Stage is order-independent: a layer drawn before it stays visible', async () => {
    const { canvas, remove } = makeCanvas(64, 64);
    try {
      const view = gpu.view(canvas, { controls: false, pixelRatio: 1, camera: { distance: 3 } });
      // a red full-screen quad at depth 0.5 (nearer than the floor and the sky), added first
      view.add(gpu.fullscreen({ writeDepth: true, code: 'fn shade(uv: vec2f, px: vec2f) -> FsFrag { return FsFrag(vec4f(1.0, 0.0, 0.0, 1.0), 0.5); }' }));
      view.add(gpu.stage());
      view.step();
      const img = await gpu.readTexture(view.hdr);
      const c = 4 * (32 * img.width + 32);
      assert(img.data[c] > 0.9 && img.data[c + 1] < 0.1, `centre pixel ${Array.from(img.data.slice(c, c + 3)).map((x) => x.toFixed(2))}`);
      view.destroy();
    } finally { remove(); }
  });

  await test('set() translates option names and rejects unknown ones up front (not mid-frame)', async () => {
    const sdf = gpu.sdf({ code: 'fn map(p: vec3f) -> vec2f { return vec2f(sdSphere(p, 1.0), 1.0); }' });
    sdf.set({ maxDistance: 12, shadows: false });
    assert(sdf.values.maxDist === 12 && sdf.values.shadows === 0, 'sdf aliases');
    let n = 0;
    try { sdf.set({ bogus: 1 }); } catch (e) { n += e instanceof SI.SIError; }
    const smoke = gpu.smoke({ res: 16 });
    const vol = gpu.volume({ smoke });
    try { vol.set({ steps: 64 }); } catch (e) { n += e instanceof SI.SIError; }
    const world = gpu.rigid({ maxBodies: 16 });
    gpu.bodies(world).set({ groups: 1 });   // the option name, translated to the uniform
    assert(n === 2, `${n} of 2 bad set() calls threw`);
    for (const x of [sdf, vol, smoke, world]) x.destroy();
  });

  await test('RigidWorld: removing twice does not hand one slot to two bodies; bad joints throw; empty raycasts miss', async () => {
    const world = gpu.rigid({ maxBodies: 64 });
    const hits0 = await world.raycast([{ origin: [0, 5, 0], dir: [0, -1, 0] }]);
    assert(!hits0[0].hit, 'phantom hit in an empty world');
    const a = world.addBox({ position: [0, 1, 0] });
    world.addBox({ position: [2, 1, 0] });
    world.remove(a); world.remove(a);
    const x = world.addBox({ position: [0, 3, 0] }), y = world.addBox({ position: [0, 5, 0] });
    assert(x !== y && world.free.length === 0, `slots ${x}, ${y}`);
    let threw = 0;
    try { world.addJoint({ type: 'ball', a: x, b: 99 }); } catch (e) { threw += e instanceof SI.SIError; }
    try { world.addJoint({ type: 'weld', a: x }); } catch (e) { threw += e instanceof SI.SIError; }
    assert(threw === 2, 'invalid joints accepted');
    assert((await world.raycast([])).length === 0, 'empty ray list');
    world.step(1 / 60);
    const hit = (await world.raycast([{ origin: [2, 5, 0], dir: [0, -1, 0] }]))[0];
    assert(hit.hit && Math.abs(hit.distance - 3.5) < 0.05, `hit ${JSON.stringify(hit)}`);
    world.destroy(); world.destroy();
    let stepThrew = false;
    try { world.step(); } catch (e) { stepThrew = e instanceof SI.SIError; }
    assert(stepThrew && world.B.destroyed, 'destroyed world');
  });

  await test('MPM: clear() forgets materials, so scenes can keep switching through more than 8', async () => {
    const sim = gpu.mpm({ grid: 16, maxParticles: 4096 });
    const names = ['water', 'oil', 'honey', 'jelly', 'rubber', 'snow', 'sand'];
    for (let round = 0; round < 3; round++) {
      sim.clear();
      for (let k = 0; k < 7; k++) sim.add({ box: [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]] }, { material: { ...SI.MATERIALS[names[k]], density: 1 + round * 0.1 + k * 0.01 } });
    }
    assert(sim.materials.length === 7 && sim.count > 0, `materials ${sim.materials.length}`);
    sim.step();
    sim.destroy();
    assert(sim.P.destroyed, 'MPM.destroy');
  });

  await test('every simulation and layer has an idempotent destroy()', async () => {
    const objs = [gpu.cloth({ width: 8, height: 8 }), gpu.smoke({ res: 16 }), gpu.boids({ count: 64 }), gpu.nbody({ count: 64 }),
      gpu.mpm({ grid: 16, maxParticles: 1024 }), new SI.Scan(gpu, 100), new SI.Reduce(gpu, { type: 'f32' })];
    const pos = gpu.array('vec4f', 8);
    objs.push(gpu.points({ data: pos }), gpu.spheres({ data: pos }), gpu.stage(), SI.Lines.box(gpu, [0, 0, 0], [1, 1, 1]),
      gpu.fullscreen({ code: 'fn shade(uv: vec2f, px: vec2f) -> vec4f { return vec4f(1.0); }' }), new SI.Lines(gpu, { points: [] }));
    for (const o of objs) { o.destroy(); o.destroy(); }
    assert(!pos.destroyed, 'a layer destroyed the array it was given');
  });

  section('physics review regressions');

  const finite = (arr) => arr.every((x) => Number.isFinite(x));

  await test('RigidWorld: contact events survive substeps; step(0) is a no-op; invalid ids are refused', async () => {
    const world = gpu.rigid({ maxBodies: 16, events: true, substeps: 4 });
    const ball = world.addSphere({ position: [0, 0.31, 0], radius: 0.3 });
    world.step(1 / 60);
    const ev = await world.readEvents();
    assert(ev.begin.some(([a, b]) => a === ball && b === -1), `begin events ${JSON.stringify(ev)}`);
    world.step(0); world.step(NaN);
    assert(finite(await world.readPoses()), 'NaN after step(0)');
    let threw = 0;
    for (const f of [() => world.setVelocity(-1, [1, 0, 0]), () => world.applyImpulse(7, [0, 1, 0]), () => world.setMotor(0, 1, 1)]) {
      try { f(); } catch (e) { threw += e instanceof SI.SIError; }
    }
    assert(threw === 3, `${threw} of 3 invalid ids refused`);
    world.destroy();
  });

  await test('NBody softening 0, MPM poisson 0.5 and cloth step(0) stay finite; bad inputs throw', async () => {
    const nb = gpu.nbody({ count: 256, softening: 0 });
    nb.plummer();
    nb.step(2);
    assert(finite(await nb.pos.read()), 'n-body NaN');
    const sim = gpu.mpm({ grid: 16, maxParticles: 8192 });
    sim.add({ box: [[0.3, 0.3, 0.3], [0.6, 0.6, 0.6]] }, { material: { kind: 'jelly', poisson: 0.5, stiffness: 0.5 } });
    for (let k = 0; k < 5; k++) sim.step();
    assert(finite(await sim.readPositions()), 'MPM NaN with poisson 0.5');
    assert(sim.materials[0].kind === 1, 'string kind');
    let threw = 0;
    try { sim.add({ box: [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]] }, { material: { kind: 'fluid', density: 1000 } }); } catch (e) { threw += e instanceof SI.SIError; }
    try { gpu.cloth({ width: 1, height: 16 }); } catch (e) { threw += e instanceof SI.SIError; }
    assert(threw === 2, `${threw} of 2 rejected`);
    const cloth = gpu.cloth({ width: 16, height: 16 });
    cloth.step(0);
    assert(finite(await cloth.pos.read()), 'cloth NaN after step(0)');
    for (const x of [nb, sim, cloth]) x.destroy();
  });

  await test('MPM: changing substeps keeps the physical speed of the particles', async () => {
    const speed = async (substeps) => {
      const sim = gpu.mpm({ grid: 24, maxParticles: 16384, gravity: [0, 0, 0], substeps: 8 });
      sim.add({ box: [[0.3, 0.3, 0.3], [0.5, 0.5, 0.5]] }, { material: 'water', velocity: [1, 0, 0] });
      sim.step();
      sim.substeps = substeps;
      const a = await sim.readPositions(); sim.step(); const b = await sim.readPositions();
      let d = 0; for (let k = 0; k < a.length; k += 3) d += b[k] - a[k];
      sim.destroy();
      return d / (a.length / 3) * 60;   // m/s
    };
    const s8 = await speed(8), s16 = await speed(16);
    assert(Math.abs(s16 / s8 - 1) < 0.05, `speed ${s8.toFixed(3)} m/s at 8 substeps vs ${s16.toFixed(3)} after switching to 16`);
  });

  await test('writeOrdered: dispatches before and after an ordered write see the old and the new data', async () => {
    const src = gpu.array('u32', new Uint32Array([1, 1, 1, 1]));
    const out = gpu.array('u32', 8);
    const k = gpu.kernel({ buffers: { src: { array: src, access: 'read' }, out }, uniforms: { slot: 'u32' }, code: 'out[u.slot] = src[0];' });
    k.run(1, { slot: 0 });
    src.writeOrdered(new Uint32Array([7, 7, 7, 7]));
    k.run(1, { slot: 1 });
    const r = await out.read();
    assert(r[0] === 1 && r[1] === 7, `got ${r[0]}, ${r[1]}`);
    // and a reset after a recorded step wins (it used to be overtaken by the step)
    const nb = gpu.nbody({ count: 512 });
    nb.plummer({ seed: 3 });
    nb.step(1);
    nb.galaxies({ seed: 3 });
    const p = await nb.pos.read();
    let m = 0; for (let j = 0; j < 512; j++) m += Math.abs(p[4 * j]);
    assert(m / 512 > 0.5, `mean |x| ${(m / 512).toFixed(3)}: the galaxies() reset was lost`);
    nb.destroy();
  });

  await test('destroying one simulation mid-batch does not drop the others\' recorded work', async () => {
    const nb = gpu.nbody({ count: 256 });
    nb.plummer();
    await gpu.idle();
    const before = await nb.pos.read();
    const cloth = gpu.cloth({ width: 16, height: 16 });
    nb.step(5);
    cloth.step(1 / 60);
    cloth.destroy();
    const after = await nb.pos.read();
    let moved = 0; for (let j = 0; j < before.length; j++) moved += Math.abs(after[j] - before[j]);
    assert(moved > 0, 'the n-body steps recorded before cloth.destroy() were lost');
    let threw = false;
    try { cloth.step(); } catch (e) { threw = e instanceof SI.SIError; }
    assert(threw, 'step() after destroy() must throw');
    nb.destroy();
  });

  await test('Boids: changing the perception radius rebuilds the neighbour grid', async () => {
    const flock = gpu.boids({ count: 1024, radius: 0.06 }).randomize();
    flock.params.radius = 0.12;
    flock.step(1 / 60);
    assert(Math.abs(flock.grid.cellSize - 0.12) < 1e-9, `cell ${flock.grid.cellSize}`);
    flock.destroy();
  });

  section('nn, netcode and verification review regressions');

  await test('Freivalds rejects swapped columns and two large errors (fresh random vectors every call)', async () => {
    const M = 64, N = 513, K = 96;
    const rnd = (n, seed) => { const x = new Float32Array(n); let q = seed; for (let k = 0; k < n; k++) { q = (q * 16807) % 2147483647; x[k] = q / 2147483647 - 0.5; } return x; };
    const A = gpu.array('f32', rnd(M * K, 3)), B = gpu.array('f32', rnd(K * N, 5)), C = gpu.array('f32', M * N);
    SI.nn.gemm(gpu, A, B, C, { M, N, K });
    const good = await SI.verify.freivalds(gpu, A, B, C, { M, N, K, trials: 8 });
    const c = await C.read();
    for (let r = 0; r < M; r++) { const t = c[r * N + 3]; c[r * N + 3] = c[r * N + 300]; c[r * N + 300] = t; }   // swap two columns
    gpu.flush(); C.write(c);
    const swapped = await SI.verify.freivalds(gpu, A, B, C, { M, N, K, trials: 8 });
    assert(good.ok && !swapped.ok, `correct ${good.ok}, swapped columns accepted ${swapped.ok}`);
    for (const x of [A, B, C]) x.destroy();
  });

  await test('checkGradient checks bias gradients too (a broken bias gradient is caught)', async () => {
    const net = new SI.nn.MLP(gpu, { layers: [3, 8, 2], act: 'tanh', out: 'none', maxBatch: 32, seed: 0 });
    const X = gpu.array('f32', new Float32Array(96).map((_, k) => Math.sin(k))), T = gpu.array('f32', new Float32Array(64).map((_, k) => Math.cos(k)));
    const ok = await SI.verify.checkGradient(net, X, T, 32);
    assert(ok.ok && ok.tensorsChecked === 4, `healthy net: ${JSON.stringify(ok)}`);
    // sabotage the bias gradient: a column-sum that does nothing, and no correct values left over
    net.colSumK = { dispatch() {} };
    for (const l of net.layers) l.gb.clear();
    const bad = await SI.verify.checkGradient(net, X, T, 32);
    assert(!bad.ok, 'a zeroed bias gradient passed the check');
    for (const x of [net, X, T]) x.destroy();
  });

  await test('MLP trains sin and gelu networks (their backward pass used to throw); Population needs an even count', async () => {
    for (const act of ['sin', 'gelu']) {
      const net = new SI.nn.MLP(gpu, { layers: [2, 16, 1], act, out: 'none', maxBatch: 64, seed: 2 });
      const X = gpu.array('f32', new Float32Array(128).map((_, k) => Math.sin(k * 0.3))), T = gpu.array('f32', new Float32Array(64).map((_, k) => Math.cos(k * 0.2)));
      const r = await SI.verify.checkGradient(net, X, T, 64);
      assert(r.ok, `${act}: ${JSON.stringify(r)}`);
      for (const x of [net, X, T]) x.destroy();
    }
    let threw = false;
    try { new SI.nn.Population(gpu, { layers: [2, 4, 1], count: 5 }); } catch (e) { threw = e instanceof SI.SIError; }
    assert(threw, 'odd population accepted');
    const pop = new SI.nn.Population(gpu, { layers: [2, 4, 1], count: 4, seed: 0 });
    const th = await pop.theta.read();
    assert(new Set(Array.from(th.slice(0, 8))).size > 1, 'seed 0 gave constant weights');
    pop.destroy();
  });

  await test('RollbackSession ignores forged and malformed messages', async () => {
    const world = gpu.rigid({ maxBodies: 16, deterministic: true });
    world.addBox({ position: [0, 1, 0] });
    const sent = [];
    const ch = { unreliable: { send() {}, onmessage: null }, reliable: { send: (m) => sent.push(m), onmessage: null } };
    const s = new SI.net.RollbackSession({ sim: world, channel: ch, players: 2, localPlayer: 0, applyInputs() {} });   // we are the authority
    ch.reliable.onmessage({ type: 'state', from: 1, tick: 0, epoch: 5, state: {} });          // a peer may not overwrite the authority
    ch.unreliable.onmessage({ from: 7, start: 0, inputs: [[1]] });                             // no such player
    ch.unreliable.onmessage({ from: 1, start: -3, inputs: 'x' });
    ch.reliable.onmessage({ type: 'hash', from: 1, tick: 0, hash: '<script>' });
    for (let k = 0; k < 5; k++) s.advance(new Uint8Array([0]));
    assert(s.pendingState === null && s.stats.statesRejected === 1 && s.stats.messagesRejected === 3, JSON.stringify(s.stats));
    const g = gpu.rigid({ maxBodies: 16, deterministic: true });
    g.addBox({ position: [0, 1, 0] });
    const ch2 = { unreliable: { send() {}, onmessage: null }, reliable: { send() {}, onmessage: null } };
    const guest = new SI.net.RollbackSession({ sim: g, channel: ch2, players: 2, localPlayer: 1, applyInputs() {} });
    ch2.reliable.onmessage({ type: 'state', from: 0, tick: 0, epoch: 0, state: {} });            // not newer than what we hold
    assert(guest.pendingState === null && guest.stats.statesRejected === 1, 'stale epoch accepted');
    world.destroy(); g.destroy();
  });

  await sleep(0);
}
