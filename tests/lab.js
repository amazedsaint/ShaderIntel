// Interactive lab helpers for exploring solver behaviour in a page that exposes
// `gpu` and `world` (examples/rigid.html):  const L = await import('/tests/lab.js'); L.install();
// Long experiments run as jobs whose result lands in window.__r.

export function install() {
  const w = window;
  w.__errs = 0; w.__first = [];
  w.gpu.device.addEventListener('uncapturederror', (e) => { w.__errs++; if (w.__first.length < 4) w.__first.push(e.error.message.slice(0, 600)); });
  w.job = (fn) => {
    w.__r = null;
    const t0 = performance.now();
    fn().then((r) => { w.__r = { r, ms: Math.round(performance.now() - t0) }; }).catch((e) => { w.__r = { error: String(e.stack || e) }; });
    return 'started';
  };
  w.run = (frames) => run(w.world, frames);
  return 'lab installed';
}

export async function run(world, frames, dt = 1 / 60) {
  for (let f = 0; f < frames; f++) { world.step(dt); if (f % 30 === 29) await world.ctx.idle(); }
}

/** Max positional drift (mm) of every body between 2 s and `frames` frames. */
export async function drift(world, build, frames = 600) {
  world.clear(); build(world);
  let first = null, maxd = 0, bad = false;
  for (let f = 0; f < frames; f++) {
    world.step(1 / 60);
    if (f === 119) first = await world.readPoses();
    if (f > 119 && f % 60 === 59) {
      const P = await world.readPoses();
      for (let i = 0; i < P.length; i++) { if (!Number.isFinite(P[i])) bad = true; if (i % 7 < 3) maxd = Math.max(maxd, Math.abs(P[i] - first[i])); }
    }
  }
  return bad ? NaN : maxd * 1000;
}

export const scenes = {
  straddle(w) { w.addBox({ position: [-0.525, 0.5, 0], half: [0.5, 0.5, 0.5] }); w.addBox({ position: [0.525, 0.5, 0], half: [0.5, 0.5, 0.5] }); w.addBox({ position: [0, 1.5, 0], half: [0.5, 0.5, 0.5] }); },
  stack10(w) { for (let k = 0; k < 10; k++) w.addBox({ position: [0, 0.5 + k, 0], half: [0.5, 0.5, 0.5] }); },
  pyramid(w) { const n = 20, s = 0.5; for (let row = 0; row < n; row++) for (let k = 0; k < n - row; k++) w.addBox({ position: [(k - (n - row - 1) / 2) * 2.02 * s, s + row * 2 * s, 0], half: [s, s, s] }); },
  tilted(w) { w.addBox({ position: [0, 1.2, 0], half: [0.5, 0.5, 0.5], rotation: [0.4, 0.3, 0.2] }); },
};

/** A chain of `links` light capsules holding a box `ratio`× heavier than the whole chain's links; returns the box sag (mm). */
export async function chain(world, { links = 12, boxDensity = 8, frames = 300 } = {}) {
  world.clear();
  const floor = world.floor; world.floor = null;
  const ids = [], top = 6, seg = 0.4;
  for (let k = 0; k < links; k++) ids.push(world.addCapsule({ position: [0, top - seg / 2 - k * seg, 0], radius: 0.05, halfHeight: seg / 2 - 0.05 }));
  world.addJoint({ type: 'ball', a: ids[0], b: null, anchor: [0, top, 0] });
  for (let k = 1; k < links; k++) world.addJoint({ type: 'ball', a: ids[k - 1], b: ids[k], anchor: [0, top - k * seg, 0] });
  const boxY = top - links * seg - 0.25;
  const heavy = world.addBox({ position: [0, boxY, 0], half: [0.25, 0.25, 0.25], density: boxDensity });
  world.addJoint({ type: 'ball', a: ids[links - 1], b: heavy, anchor: [0, top - links * seg, 0] });
  const sag = [];
  for (let f = 0; f < frames; f += 60) { await run(world, 60); const P = await world.readPoses(); sag.push(+((boxY - P[heavy * 7 + 1]) * 1000).toFixed(2)); }
  world.floor = floor;
  return sag;
}

/** A heavy box (ratio × the mass of each light box) on a stack of light boxes; returns { sinkMM, driftMM }. */
export async function heavyStack(world, { ratio = 100, light = 3, frames = 360 } = {}) {
  world.clear();
  for (let k = 0; k < light; k++) world.addBox({ position: [0, 0.25 + 0.5 * k, 0], half: [0.25, 0.25, 0.25] });
  const top = world.addBox({ position: [0, 0.5 * light + 0.25, 0], half: [0.25, 0.25, 0.25], density: ratio });
  await run(world, 120);
  const a = await world.readPoses();
  await run(world, frames - 120);
  const b = await world.readPoses();
  let d = 0; for (let i = 0; i < a.length; i++) if (i % 7 < 3) d = Math.max(d, Math.abs(a[i] - b[i]));
  return { sinkMM: +((0.5 * light + 0.25 - b[top * 7 + 1]) * 1000).toFixed(2), driftMM: +(d * 1000).toFixed(3), finite: b.every(Number.isFinite) };
}

/** Exercise the gameplay API; returns measurements for each feature. */
export async function features(world) {
  const out = {};
  const floor = world.floor;
  // capsules: one lying on the floor, one standing, a capsule resting across two boxes
  world.clear();
  const lying = world.addCapsule({ position: [0, 0.4, 0], radius: 0.2, halfHeight: 0.5, rotation: [0, 0, Math.PI / 2] });
  const standing = world.addCapsule({ position: [2, 0.8, 0], radius: 0.2, halfHeight: 0.5, angularVelocity: [0, 0, 0] });
  const b1 = world.addBox({ position: [5, 0.25, -0.6], half: [0.25, 0.25, 0.25] });
  const b2 = world.addBox({ position: [5, 0.25, 0.6], half: [0.25, 0.25, 0.25] });
  const across = world.addCapsule({ position: [5, 0.75, 0], radius: 0.2, halfHeight: 0.7, rotation: [Math.PI / 2, 0, 0] });
  await run(world, 240);
  let P = await world.readPoses();
  out.capsules = { lyingY: +P[lying * 7 + 1].toFixed(4), standingY: +P[standing * 7 + 1].toFixed(4), acrossY: +P[across * 7 + 1].toFixed(4), expect: [0.2, 0.7, 0.7] };
  // kinematic platform moving sideways carries a box (friction)
  world.clear();
  const plat = world.addBox({ position: [0, 1, 0], half: [1.5, 0.1, 1.5], kinematic: true });
  const rider = world.addBox({ position: [0, 1.35, 0], half: [0.25, 0.25, 0.25] });
  await run(world, 30);
  for (let f = 0; f < 120; f++) { world.moveKinematic(plat, [f / 120, 1, 0]); world.step(1 / 60); }
  await world.ctx.idle();
  P = await world.readPoses();
  out.kinematic = { platformX: +P[plat * 7].toFixed(4), riderX: +P[rider * 7].toFixed(4), riderY: +P[rider * 7 + 1].toFixed(4) };
  // filtering: a box in group 2 that does not collide with group 1 falls through another box
  world.clear();
  world.addBox({ position: [0, 0.5, 0], half: [0.5, 0.5, 0.5], group: 1, mask: 0xffffffff });
  const ghost = world.addBox({ position: [0, 2, 0], half: [0.3, 0.3, 0.3], group: 2, mask: ~1 >>> 0 });
  await run(world, 120);
  P = await world.readPoses();
  out.filter = { ghostY: +P[ghost * 7 + 1].toFixed(3), expect: 0.3 };
  // sensor + events: a ball falls through a sensor box onto the floor
  world.clear(); world.events = true;
  const sensor = world.addBox({ position: [0, 1, 0], half: [0.5, 0.2, 0.5], static: true, sensor: true });
  const ball = world.addSphere({ position: [0, 2.5, 0], radius: 0.2 });
  const seen = { begin: [], end: [] };
  for (let f = 0; f < 90; f++) { world.step(1 / 60); const e = await world.readEvents(); for (const p of e.begin) seen.begin.push(`${f}:${p}`); for (const p of e.end) seen.end.push(`${f}:${p}`); }
  P = await world.readPoses();
  out.events = { ...seen, ballY: +P[ball * 7 + 1].toFixed(3), sensor, ball };
  world.events = false;
  // raycasts against a box, a sphere and a capsule (CPU-checked distances)
  world.clear();
  world.addBox({ position: [0, 1, 0], half: [0.5, 0.5, 0.5], static: true });
  world.addSphere({ position: [3, 1, 0], radius: 0.5, static: true });
  world.addCapsule({ position: [6, 1, 0], radius: 0.3, halfHeight: 0.5, static: true });
  world.step(1 / 60);
  const hits = await world.raycast([
    { origin: [0, 1, -5], dir: [0, 0, 1] }, { origin: [3, 1, -5], dir: [0, 0, 1] }, { origin: [6, 5, 0], dir: [0, -1, 0] },
    { origin: [6, 1.2, -5], dir: [0, 0, 1] }, { origin: [-5, 10, 0], dir: [0, 1, 0] }, { origin: [-5, 1, 0], dir: [1, 0, 0], mask: 0 },
  ]);
  out.raycast = hits.map((h) => (h.hit ? `${h.body}@${h.distance.toFixed(4)} n=${h.normal.map((v) => v.toFixed(2))}` : 'miss'));
  out.raycastExpect = ['0@4.5000 n=0,0,-1', '1@4.5000 n=0,0,-1', '2@3.2000 n=0,1,0', '2@4.7000 n=0,0,-1', 'miss', 'miss'];
  // snapshot / restore: restoring and re-simulating reproduces the original path bit for bit
  world.clear(); world.deterministic = true;
  let s = 3; const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  for (let k = 0; k < 300; k++) world.addBox({ position: [(r() - 0.5) * 4, 0.5 + k * 0.05, (r() - 0.5) * 4], half: [0.2, 0.2, 0.2], rotation: [r(), r(), r()] });
  await run(world, 60);
  const snap = world.snapshot();
  await run(world, 60);
  const h1 = await world.hash();
  world.restore(snap);
  await run(world, 60);
  const h2 = await world.hash();
  out.snapshot = { h1, h2, same: h1 === h2 };
  world.deterministic = false;
  world.floor = floor;
  return out;
}

/** Joint stress test: chain sag, plus the largest joint force and NaN check in the joints example scenes. */
export async function jointStress(world, buildScenes) {
  const out = { chain: (await chain(world)).slice(-1)[0] };
  for (const [name, build] of Object.entries(buildScenes)) {
    world.clear(); build();
    let nan = false;
    for (let f = 0; f < 240; f++) { world.step(1 / 60); if (f % 60 === 59) { const P = await world.readPoses(); if (!P.every(Number.isFinite)) nan = true; } }
    const js = await world.readJoints();
    const sorted = [...js.force].sort((a, b) => b - a);
    out[name] = `max ${sorted[0].toFixed(0)} p90 ${sorted[Math.floor(sorted.length * 0.1)].toFixed(0)} broken ${js.broken.filter(Boolean).length}${nan ? ' NaN' : ''}`;
  }
  return out;
}

/**
 * Throughput and quality of 2D → 1D hashes of pixel coordinates (16M per run): the classic sin
 * hash, pcg2d(...).x and cheaper integer variants. Quality: χ² of 256 value buckets (uniformity)
 * and of 16 × 16 buckets of horizontally adjacent pairs (independence of neighbours); 255 dof,
 * so ≲ 330 passes.
 */
export async function hashBench(gpu, SI) {
  await import('../bench/baselines.js');   // registers the sin hash module
  const n = 1 << 24, out = gpu.array('f32', n);
  const variants = {
    'sin (baseline)': 'sinHash21(vec2f(f32(x), f32(y)) * 37.0)',
    'pcg2d(p).x': 'u2f(pcg2d(vec2u(x, y)).x)',
    'pcg(x + pcg(y))': 'u2f(pcg(x + pcg(y)))',
    'pcg(x ^ y·φ)': 'u2f(pcg(x ^ (y * 0x9e3779b9u)))',
    'lowbias32(x ^ lowbias32(y))': 'u2f(lowbias32(x ^ lowbias32(y)))',
    'pcg2d(p).y': 'u2f(pcg2d(vec2u(x, y)).y)',
    'pcg3d(x,y,0).x': 'u2f(pcg3d(vec3u(x, y, 0u)).x)',
    'pcg(x + pcg(y + pcg(0)))': 'u2f(pcg(x + pcg(y + pcg(0u))))',
    'hash22 second: pcg(v.x ^ v.y)': 'u2f(pcg(pcg2d(vec2u(x, y)).x ^ pcg2d(vec2u(x, y)).y))',
  };
  const chi = (v) => {
    const B = 256, c = new Float64Array(B), p = new Float64Array(B);
    for (const x of v) c[Math.min(B - 1, Math.floor(x * B))]++;
    for (let k = 0; k + 1 < v.length; k += 2) p[Math.min(15, Math.floor(v[k] * 16)) * 16 + Math.min(15, Math.floor(v[k + 1] * 16))]++;
    const e = v.length / B, ep = (v.length >> 1) / B;
    let a = 0, b = 0;
    for (let k = 0; k < B; k++) { a += (c[k] - e) ** 2 / e; b += (p[k] - ep) ** 2 / ep; }
    return [Math.round(a), Math.round(b)];
  };
  const res = {};
  for (const [name, expr] of Object.entries(variants)) {
    const k = gpu.kernel({ name: `hash-${name}`, imports: ['hash', 'baseline-sinhash'], buffers: { out },
      code: `let x = i & 4095u; let y = i >> 12u; out[i] = ${expr};` });
    await k.ready();
    let ms = Infinity;
    for (let r = 0; r < 3; r++) ms = Math.min(ms, await SI.timeIt(gpu, () => k.run(n), 4));
    k.run(n);
    const [uniform, pairs] = chi(await out.read(1 << 20));
    res[name] = { 'G hashes/s': +(n / ms / 1e6).toFixed(1), chi2: uniform, chi2Pairs: pairs };
  }
  out.destroy();
  return res;
}

/** Latency of a small GPU→CPU read-back, and the CPU (recording) and GPU cost of one duel step and one snapshot. */
export async function latency(gpu) {
  const med = (v) => v.slice().sort((a, b) => a - b)[v.length >> 1];
  const a = gpu.array('u32', 4);
  const reads = [];
  for (let k = 0; k < 20; k++) { const t0 = performance.now(); await a.read(); reads.push(performance.now() - t0); }
  const { createDuel } = await import('./duel.js');
  const D = createDuel(gpu);
  for (let k = 0; k < 60; k++) D.world.step(1 / 60);
  await gpu.idle();
  let t0 = performance.now();
  for (let k = 0; k < 60; k++) D.world.step(1 / 60);
  const record = (performance.now() - t0) / 60;
  await gpu.idle();
  const total = (performance.now() - t0) / 60;
  let snap = null;
  t0 = performance.now();
  for (let k = 0; k < 60; k++) snap = D.world.snapshot(snap);
  await gpu.idle();
  const snapMs = (performance.now() - t0) / 60;
  t0 = performance.now();
  const st = await D.world.exportState(snap);
  const exportMs = performance.now() - t0;
  return { readbackMs: +med(reads).toFixed(2), readbackMaxMs: +Math.max(...reads).toFixed(2), stepRecordMs: +record.toFixed(3), stepTotalMs: +total.toFixed(3),
    snapshotMs: +snapMs.toFixed(3), exportMs: +exportMs.toFixed(1), stateBytes: st.bodies.byteLength + st.contacts.byteLength + st.slots.byteLength + st.ext.byteLength, bodies: D.world.count };
}

/** Per-dispatch cost: 1,000 tiny dispatches, direct and indirect, independent and chained (each reads what the last wrote). */
export async function dispatchCost(gpu) {
  const a = gpu.array('u32', 1024), args = gpu.array('u32', new Uint32Array([1, 1, 1, 0]), { usage: GPUBufferUsage.INDIRECT });
  const k = gpu.kernel({ name: 'tiny', workgroupSize: 64, guard: false, buffers: { a }, code: 'a[lidx] = a[lidx] + 1u;' });
  await k.ready();
  const time = async (fn) => { fn(); await gpu.idle(); const t0 = performance.now(); fn(); await gpu.idle(); return +((performance.now() - t0) / 1000 * 1000).toFixed(2); };
  return {
    'direct µs/dispatch': await time(() => { for (let n = 0; n < 1000; n++) k.dispatch(1); }),
    'indirect µs/dispatch': await time(() => { for (let n = 0; n < 1000; n++) k.dispatchIndirect(args, 0); }),
    'region-split µs/pass': await time(() => { for (let n = 0; n < 1000; n++) gpu.region('x', () => k.dispatch(1)); }),
  };
}

/** A 2,000-body pile stepped 120 frames (a workload to profile): CPU recording and GPU time per step. */
export async function pileSteps(gpu, SI) {
  const world = gpu.rigid({ maxBodies: 4096 });
  let s = 11;
  const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  for (let k = 0; k < 2000; k++) {
    const p = [(R() - 0.5) * 6, 1 + k * 0.01 + R() * 2, (R() - 0.5) * 6];
    if (R() < 0.7) world.addBox({ position: p, half: [0.1 + R() * 0.15, 0.1 + R() * 0.15, 0.1 + R() * 0.15], rotation: [R() * 3, R() * 3, R() * 3] });
    else world.addSphere({ position: p, radius: 0.1 + R() * 0.1 });
  }
  for (let f = 0; f < 240; f++) { world.step(1 / 60); if (f % 30 === 29) await gpu.idle(); }
  await gpu.idle();
  const t0 = performance.now();
  for (let f = 0; f < 120; f++) { world.step(1 / 60); if (f % 10 === 9) await gpu.idle(); }
  await gpu.idle();
  const st = await world.readStats();
  return { bodies: world.count, msPerStep: +((performance.now() - t0) / 120).toFixed(3), ...st };
}

/**
 * Per-kernel GPU time of a rigid step (timestamp queries): every kernel dispatch becomes its own
 * timed pass. Pass sizes slightly inflate small kernels; the ranking is what matters.
 */
export async function profileRigid(gpu) {
  const world = gpu.rigid({ maxBodies: 4096 });
  let s = 11;
  const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  for (let k = 0; k < 2000; k++) {
    const p = [(R() - 0.5) * 6, 1 + k * 0.01 + R() * 2, (R() - 0.5) * 6];
    if (R() < 0.7) world.addBox({ position: p, half: [0.1 + R() * 0.15, 0.1 + R() * 0.15, 0.1 + R() * 0.15], rotation: [R() * 3, R() * 3, R() * 3] });
    else world.addSphere({ position: p, radius: 0.1 + R() * 0.1 });
  }
  for (let f = 0; f < 240; f++) { world.step(1 / 60); if (f % 30 === 29) await gpu.idle(); }
  await gpu.idle();
  const origRegion = gpu.region.bind(gpu);
  gpu.region = (label, fn) => fn();
  const wrap = (obj, label) => { for (const m of ['run', 'dispatch']) { const f = obj[m]; obj[m] = function (...a) { return origRegion(label, () => f.apply(this, a)); }; } };
  for (const k of ['predictK', 'jointPrepK', 'pairCountK', 'pairWriteK', 'pairArgsK', 'reverseK', 'reverseLargeK', 'narrowCountK', 'narrowWriteK',
    'colorInitK', 'colorK', 'bucketK', 'offsetsK', 'scatterK', 'primalK', 'dualK', 'jointDualK', 'velocityK']) if (world[k]) wrap(world[k], k.replace(/K$/, ''));
  wrap(world.pairScan, 'pairScan'); wrap(world.contactScan, 'contactScan');
  const gb = world.grid.build; world.grid.build = function (...a) { return origRegion('grid.build', () => gb.apply(this, a)); };
  const fill = gpu.fill; gpu.fill = function (...a) { return origRegion('fill', () => fill.apply(this, a)); };
  gpu.profiler.reset();
  for (let f = 0; f < 60; f++) { world.step(1 / 60); gpu.flush(); if (f % 10 === 9) await gpu.idle(); }
  await gpu.idle();
  await new Promise((done) => setTimeout(done, 500));
  gpu.region = origRegion; gpu.fill = fill;
  const rep = gpu.profiler.report().sort((a, b) => b.ms - a.ms);
  const total = rep.reduce((t, r) => t + r.ms, 0);
  return { totalMs: +total.toFixed(3), kernels: Object.fromEntries(rep.map((r) => [r.label, +r.ms.toFixed(3)])) };
}

/** The same 150-body scene solved fused and multi-dispatch: state hashes after each 30-step block. */
export async function fusedVsMulti(gpu) {
  const run = async (fusedSolve) => {
    const w = gpu.rigid({ maxBodies: 512, deterministic: true, fusedSolve });
    let s = 5;
    const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    for (let k = 0; k < 150; k++) {
      const p = [(R() - 0.5) * 3, 0.5 + k * 0.05, (R() - 0.5) * 3];
      if (R() < 0.6) w.addBox({ position: p, half: [0.15, 0.1 + R() * 0.1, 0.15], rotation: [R(), R(), R()] });
      else w.addCapsule({ position: p, radius: 0.1, halfHeight: 0.15, rotation: [R(), R(), R()] });
    }
    const a = w.addSphere({ position: [0, 6, 0], radius: 0.3 }), b = w.addSphere({ position: [0.6, 6, 0], radius: 0.3 });
    w.addJoint({ type: 'distance', a, b, length: 0.6 });
    const hashes = [], poses = [];
    for (let f = 0; f < 150; f++) {
      w.step(1 / 60);
      if (f === 0 || f === 4 || f === 29 || f === 149) poses.push(await w.readPoses());
      if (f % 30 === 29) hashes.push(await w.hash());
    }
    return { hashes, poses };
  };
  const [f, m] = [await run(true), await run(false)];
  const maxDiff = (a, b) => { let d = 0; for (let k = 0; k < a.length; k++) if (k % 7 < 3) d = Math.max(d, Math.abs(a[k] - b[k])); return d; };
  return { identical: f.hashes.every((h, k) => h === m.hashes[k]),
    maxPositionDiff: { 'step 1': maxDiff(f.poses[0], m.poses[0]), 'step 5': maxDiff(f.poses[1], m.poses[1]), 'step 30': maxDiff(f.poses[2], m.poses[2]), 'step 150': maxDiff(f.poses[3], m.poses[3]) } };
}

/** Step time of the fused and the multi-dispatch solve for piles of n bodies (deterministic worlds). */
export async function fusedThreshold(gpu) {
  const out = {};
  for (const n of [32, 96, 192, 384, 768]) {
    const row = {};
    for (const fusedSolve of [true, false]) {
      const w = gpu.rigid({ maxBodies: 1024, deterministic: true, fusedSolve });
      let s = 3;
      const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
      for (let k = 0; k < n; k++) w.addBox({ position: [(R() - 0.5) * 4, 0.3 + k * 0.03, (R() - 0.5) * 4], half: [0.15, 0.15, 0.15], rotation: [R(), R(), R()] });
      for (let f = 0; f < 120; f++) { w.step(1 / 60); if (f % 30 === 29) await gpu.idle(); }
      await gpu.idle();
      const t0 = performance.now();
      for (let f = 0; f < 60; f++) { w.step(1 / 60); if (f % 10 === 9) await gpu.idle(); }
      await gpu.idle();
      row[fusedSolve ? 'fused ms' : 'multi ms'] = +((performance.now() - t0) / 60).toFixed(3);
    }
    out[n] = row;
  }
  return out;
}

/** The 6,004-body pile of examples/rigid.html: ms per step after settling, default and deterministic. */
export async function bigPile(gpu) {
  const res = {};
  for (const deterministic of [false, true]) {
    const world = gpu.rigid({ maxBodies: 65535, deterministic });
    let s = 7;
    const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    for (let k = 0; k < 6000; k++) {
      const p = [(r() - 0.5) * 10, 2 + k * 0.012 + r() * 3, (r() - 0.5) * 10];
      if (r() < 0.75) world.addBox({ position: p, half: [0.15 + r() * 0.2, 0.15 + r() * 0.2, 0.15 + r() * 0.2], rotation: [r() * 3, r() * 3, r() * 3] });
      else world.addSphere({ position: p, radius: 0.15 + r() * 0.15 });
    }
    for (let a = 0; a < 4; a++) {
      const ang = a * Math.PI / 2;
      world.addBox({ position: [Math.cos(ang) * 7, 1.5, Math.sin(ang) * 7], half: [0.3, 1.5, 7], rotation: [0, ang, 0], static: true });
    }
    const T0 = performance.now();
    for (let f = 0; f < 900; f++) {
      world.step(1 / 60);
      if (f % 30 === 29) await gpu.idle();
      if (f % 100 === 99) console.log('bigPile', deterministic ? 'det' : 'default', f + 1, 'steps', ((performance.now() - T0) / 1000).toFixed(1) + 's', JSON.stringify(world.stats));
    }
    await gpu.idle();
    let best = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t0 = performance.now();
      for (let k = 0; k < 60; k++) world.step(1 / 60);
      await gpu.idle();
      best = Math.min(best, (performance.now() - t0) / 60);
    }
    const st = await world.readStats();
    res[deterministic ? 'deterministic' : 'default'] = { msPerStep: +best.toFixed(3), contacts: st.contacts, colors: st.colors, bodies: world.count };
  }
  return res;
}

/** Where the time goes when a large world starts: creation, the first step (shader compiles), steady steps. */
export async function pileProbe(gpu) {
  const out = {};
  const T0 = performance.now();
  const say = (...a) => console.log('probe', ((performance.now() - T0) / 1000).toFixed(1) + 's', ...a);
  for (const maxBodies of [2048, 8192, 65535]) {
    const t = [performance.now()];
    const world = gpu.rigid({ maxBodies });
    say(maxBodies, 'created');
    let s = 7;
    const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    const n = Math.min(2000, maxBodies);
    for (let k = 0; k < n; k++) world.addBox({ position: [(r() - 0.5) * 10, 2 + k * 0.012, (r() - 0.5) * 10], half: [0.2, 0.2, 0.2] });
    t.push(performance.now());
    world.step(1 / 60);
    say(maxBodies, 'first step recorded');
    await gpu.idle();
    say(maxBodies, 'first step done');
    t.push(performance.now());
    for (let f = 0; f < 30; f++) world.step(1 / 60);
    await gpu.idle();
    t.push(performance.now());
    out[maxBodies] = { createMs: Math.round(t[1] - t[0]), firstStepMs: Math.round(t[2] - t[1]), msPerStep: +((t[3] - t[2]) / 30).toFixed(2) };
    say(maxBodies, JSON.stringify(out[maxBodies]));
  }
  return out;
}
