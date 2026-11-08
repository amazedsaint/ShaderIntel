import * as SI from '../src/index.js';
import { test, section, note, assert, rng } from './harness.js';

export async function physicsTests(gpu, tag = 'native') {
  section(`MPM [${tag}]`);
  await test('P2G conserves mass and linear momentum exactly (APIC + stress terms sum to zero)', async () => {
    const sim = gpu.mpm({ grid: 48, maxParticles: 60000, substeps: 4 });
    sim.add({ box: [[0.2, 0.2, 0.2], [0.7, 0.6, 0.7]] }, { material: 'water' });
    // randomise velocity, affine C and stress tau so every term of the transfer is exercised
    const n = sim.count, raw = await sim.P.read(n * 8), R = rng(3);
    for (let k = 0; k < n; k++) {
      const o = 32 * k;
      for (let a = 0; a < 3; a++) raw[o + 4 + a] = (R() - 0.5) * 0.4;           // v (cells/step)
      for (let c = 0; c < 3; c++) for (let a = 0; a < 3; a++) raw[o + 8 + 4 * c + a] = (R() - 0.5) * 0.2; // C
      raw[o + 11] = (R() - 0.5) * 0.3; raw[o + 15] = (R() - 0.5) * 0.3; raw[o + 19] = (R() - 0.5) * 0.3;   // tau diag
      raw[o + 23] = (R() - 0.5) * 0.3; raw[o + 27] = (R() - 0.5) * 0.3; raw[o + 31] = (R() - 0.5) * 0.3;   // tau off-diag
    }
    sim.P.write(raw);
    sim.p2g();
    const g = new Int32Array((await sim.grid.read()).buffer);
    const FP = 2 ** 22, mass = 1 / 8;
    let gm = 0, gp = [0, 0, 0];
    for (let k = 0; k < g.length; k += 4) { gm += g[k + 3]; gp[0] += g[k]; gp[1] += g[k + 1]; gp[2] += g[k + 2]; }
    gm /= FP; gp = gp.map((x) => x / FP);
    let pm = 0; const pp = [0, 0, 0];
    for (let k = 0; k < n; k++) { pm += mass; for (let a = 0; a < 3; a++) pp[a] += mass * raw[32 * k + 4 + a]; }
    const relM = Math.abs(gm - pm) / pm;
    const errP = Math.max(...gp.map((x, a) => Math.abs(x - pp[a])));
    note(`mass ${gm.toFixed(4)} vs ${pm.toFixed(4)} (rel ${relM.toExponential(1)}); momentum error ${errP.toExponential(1)} (|p| ~ ${Math.hypot(...pp).toFixed(3)})`);
    assert(relM < 1e-4, 'mass not conserved');
    assert(errP < 1e-3 * Math.max(1, Math.hypot(...pp)) + n * 27 * 4 / FP, 'momentum not conserved');
  });
  await test('tiled P2G is bit-identical to global-atomic P2G', async () => {
    const sim = gpu.mpm({ grid: 48, maxParticles: 80000, substeps: 4 });
    sim.add({ box: [[0.1, 0.1, 0.1], [0.9, 0.4, 0.9]] }, { material: 'water' });
    for (let f = 0; f < 3; f++) sim.step();
    sim._sort();
    const other = gpu.array('vec4i', sim.numNodes);
    sim.p2g(sim.P, true, other); sim.p2g(sim.P, false, sim.grid);
    const [a, b] = [await other.read(), await sim.grid.read()];
    for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) throw new Error(`word ${k} differs`);
  });
  await test('dam break: stable, bounded, volume preserved (mean J ≈ 1)', async () => {
    const sim = gpu.mpm({ grid: 48, maxParticles: 200000, substeps: 8 });
    sim.add({ box: [[0.05, 0.05, 0.05], [0.4, 0.6, 0.95]] }, { material: 'water' });
    sim.add({ sphere: [[0.7, 0.4, 0.5], 0.12] }, { material: 'jelly' });
    for (let f = 0; f < 150; f++) { sim.step(); if (f % 30 === 29) await gpu.idle(); }
    const n = sim.count, raw = await sim.P.read(n * 8);
    let nan = 0, out = 0, Jsum = 0, Jn = 0;
    const u = new Uint32Array(raw.buffer);
    for (let k = 0; k < n; k++) {
      const o = 32 * k;
      for (let a = 0; a < 3; a++) { const x = raw[o + a]; if (!Number.isFinite(x)) nan++; else if (x < 1 || x > sim.dims[a] - 1) out++; }
      if (u[o + 3] === 0) { Jsum += raw[o + 7]; Jn++; }
    }
    const meanJ = Jsum / Jn;
    note(`${n} particles after 150 frames: NaN ${nan}, outside ${out}, mean water J ${meanJ.toFixed(4)}`);
    assert(nan === 0 && out === 0, 'unstable');
    assert(Math.abs(meanJ - 1) < 0.05, 'volume drift');
  });

  if (tag !== 'native') return; // the remaining solvers have no subgroup-specific paths
  section('cloth');
  await test('hanging cloth: structural strain < 2% after settling', async () => {
    const W = 48, cloth = gpu.cloth({ width: W, height: W, size: [1, 1], origin: [-0.5, 1.5, 0], substeps: 20, pin: 'top', wind: [0, 0, 0] });
    for (let f = 0; f < 240; f++) { cloth.step(1 / 60); if (f % 40 === 39) await gpu.idle(); }
    const p = await cloth.pos.read();
    const rest = 1 / (W - 1);
    let maxStrain = 0, nan = 0;
    for (let j = 0; j < W; j++) for (let i = 0; i < W; i++) {
      const a = 4 * (j * W + i);
      if (!Number.isFinite(p[a])) { nan++; continue; }
      if (i + 1 < W) { const b = a + 4; maxStrain = Math.max(maxStrain, Math.abs(Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]) / rest - 1)); }
      if (j + 1 < W) { const b = a + 4 * W; maxStrain = Math.max(maxStrain, Math.abs(Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]) / rest - 1)); }
    }
    note(`max structural strain ${(maxStrain * 100).toFixed(2)}%`);
    assert(nan === 0 && maxStrain < 0.02, 'cloth stretches too much');
  });

  section('smoke');
  await test('pressure projection removes most of the divergence', async () => {
    const smoke = gpu.smoke({ res: 40, size: [1, 1, 1], iterations: 40 });
    for (let f = 0; f < 30; f++) smoke.step(1 / 60);
    const before = await gpu.readTexture(smoke.div);       // divergence of the pre-projection field
    smoke.divK.run(smoke.dims, { invDx: 1 / smoke.dx }, { vel: smoke.velA, dst: smoke.div });
    const after = await gpu.readTexture(smoke.div);
    const rms = (t) => Math.sqrt(t.data.reduce((s, x) => s + x * x, 0) / t.data.length);
    const ratio = rms(after) / rms(before);
    note(`rms divergence ${rms(before).toFixed(4)} → ${rms(after).toFixed(4)} (×${ratio.toFixed(3)})`);
    assert(Number.isFinite(ratio) && ratio < 0.05, 'projection ineffective');
  });

  section('rigid bodies (AVBD)');
  const run = async (world, frames) => { for (let f = 0; f < frames; f++) { world.step(1 / 60); if (f % 30 === 29) await gpu.idle(); } };
  const maxMove = (a, b) => { let d = 0; for (let k = 0; k < a.length; k++) if (k % 7 < 3) d = Math.max(d, Math.abs(a[k] - b[k])); return d; };
  await test('a tilted box lands flat and stays put', async () => {
    const world = gpu.rigid({ maxBodies: 16 });
    world.addBox({ position: [0, 1.2, 0], half: [0.5, 0.5, 0.5], rotation: [0.4, 0.3, 0.2] });
    await run(world, 120);
    const a = await world.readPoses();
    await run(world, 120);
    const b = await world.readPoses();
    const [x, y, z, w] = b.subarray(3, 7);
    const up = Math.max(Math.abs(2 * (x * y + w * z)), Math.abs(1 - 2 * (x * x + z * z)), Math.abs(2 * (y * z - w * x)));  // |y| of the most vertical body axis
    note(`centre height ${b[1].toFixed(4)} m, most vertical axis ${up.toFixed(5)}, drift over 2 s ${(maxMove(a, b) * 1000).toFixed(3)} mm`);
    assert(Math.abs(b[1] - 0.5) < 0.01 && up > 0.999, 'did not come to rest flat on its face');
    assert(maxMove(a, b) < 1e-3, 'resting box drifts');
  });
  await test('a 10-box tower and a 20-row pyramid stand still (8 iterations)', async () => {
    const world = gpu.rigid({ maxBodies: 256, iterations: 8 });
    for (let k = 0; k < 10; k++) world.addBox({ position: [-15, 0.5 + k, 0], half: [0.5, 0.5, 0.5] });
    for (let row = 0; row < 20; row++)
      for (let k = 0; k < 20 - row; k++) world.addBox({ position: [(k - (19 - row) / 2) * 1.01, 0.5 + row, 0], half: [0.5, 0.5, 0.5] });
    await run(world, 120);
    const a = await world.readPoses();
    await run(world, 360);
    const b = await world.readPoses();
    const tower = maxMove(a.subarray(0, 70), b.subarray(0, 70)), pyramid = maxMove(a.subarray(70), b.subarray(70));
    note(`max movement over 6 s after settling: tower ${(tower * 1000).toFixed(2)} mm, pyramid ${(pyramid * 1000).toFixed(2)} mm`);
    assert(tower < 2e-3 && pyramid < 5e-3, 'stack creeps');
  });
  await test('Coulomb friction: sticks on a 20° slope, slides on 40° at g(sin θ − μ cos θ)', async () => {
    const slope = async (deg) => {
      const th = (deg * Math.PI) / 180, nrm = [-Math.sin(th), Math.cos(th)], dir = [-Math.cos(th), -Math.sin(th)];
      const world = gpu.rigid({ maxBodies: 8, floor: null });
      world.addBox({ position: [0, 2, 0], half: [4, 0.25, 2], rotation: [0, 0, th], static: true });
      world.addBox({ position: [nrm[0] * 0.5, 2 + nrm[1] * 0.5, 0], half: [0.25, 0.25, 0.25], rotation: [0, 0, th] });
      const s = [];
      for (let k = 0; k < 3; k++) { await run(world, 30); const p = await world.readPoses(); s.push(p[7] * dir[0] + p[8] * dir[1]); }
      return { moved: s[2] - s[0], acc: (s[2] - 2 * s[1] + s[0]) / 0.25 };   // samples 0.5 s apart
    };
    const lo = await slope(20), hi = await slope(40);
    const expect = 9.81 * (Math.sin((40 * Math.PI) / 180) - 0.6 * Math.cos((40 * Math.PI) / 180));
    note(`20°: moved ${(lo.moved * 1000).toFixed(2)} mm in 1 s · 40°: a = ${hi.acc.toFixed(3)} m/s² (Coulomb: ${expect.toFixed(3)})`);
    assert(Math.abs(lo.moved) < 2e-3, 'box slides below the friction angle');
    assert(Math.abs(hi.acc - expect) < 0.1 * expect, 'sliding acceleration does not match Coulomb friction');
  });
  await test('ball joint: a pendulum swings with period 2π√(I/mgL) and keeps its length', async () => {
    const world = gpu.rigid({ maxBodies: 16, floor: null });
    const L = 1, th0 = 0.2, r = 0.1;
    const id = world.addSphere({ position: [L * Math.sin(th0), 3 - L * Math.cos(th0), 0], radius: r });
    world.addJoint({ type: 'ball', a: id, b: null, anchor: [0, 3, 0] });
    const xs = [];
    let len = 0;
    for (let f = 0; f < 480; f++) {
      world.step(1 / 60);
      if (f % 2 === 1) { const P = await world.readPoses(); xs.push(P[0]); len = Math.max(len, Math.abs(Math.hypot(P[0], P[1] - 3, P[2]) - L)); }
    }
    const cross = [];
    for (let k = 1; k < xs.length; k++) if (xs[k - 1] > 0 && xs[k] <= 0) cross.push((k - 1 + xs[k - 1] / (xs[k - 1] - xs[k])) * 2 / 60);
    const T = (cross[cross.length - 1] - cross[0]) / (cross.length - 1);
    const Tth = 2 * Math.PI * Math.sqrt((0.4 * r * r + L * L) / (9.81 * L));
    note(`period ${T.toFixed(4)} s (theory ${Tth.toFixed(4)} s), max length error ${(len * 1000).toFixed(3)} mm`);
    assert(Math.abs(T / Tth - 1) < 0.01, 'pendulum period');
    assert(len < 1e-3, 'joint stretches');
  });
  await test('joints: motor speed, hinge limit, weld, rope, slider limit, break force', async () => {
    const world = gpu.rigid({ maxBodies: 32, floor: null });
    const run = async (frames) => { for (let f = 0; f < frames; f++) world.step(1 / 60); await gpu.idle(); };
    const r = {};
    const plank = world.addBox({ position: [0.5, 3, 0], half: [0.5, 0.05, 0.2] });
    world.addJoint({ type: 'hinge', a: plank, b: null, anchor: [0, 3, 0], axis: [0, 0, 1], motor: { speed: 2, maxForce: 1000 } });
    await run(120);
    r.motor = (await world.readBodies()).angularVelocity[2];
    world.clear();
    const door = world.addBox({ position: [0.5, 3, 0], half: [0.5, 0.05, 0.2] });
    world.addJoint({ type: 'hinge', a: door, b: null, anchor: [0, 3, 0], axis: [0, 0, 1], limits: [-0.4, 0.3] });
    await run(180);
    let P = await world.readPoses();
    r.limit = Math.atan2(P[1] - 3, P[0]);
    world.clear();
    const w1 = world.addBox({ position: [0, 3, 0], half: [0.2, 0.2, 0.2] });
    const w2 = world.addBox({ position: [0.6, 3, 0], half: [0.2, 0.2, 0.2] });
    world.addJoint({ type: 'ball', a: w1, b: null, anchor: [0, 3.2, 0] });
    world.addJoint({ type: 'fixed', a: w1, b: w2, anchor: [0.3, 3, 0] });
    await run(240);
    P = await world.readPoses();
    r.weld = Math.hypot(P[7] - P[0], P[8] - P[1], P[9] - P[2]);
    world.clear();
    const rs = world.addSphere({ position: [0.5, 4, 0], radius: 0.1 });
    world.addJoint({ type: 'distance', a: rs, b: null, anchor: [0.5, 4, 0], anchorB: [0, 5, 0], length: 2, rope: true });
    await run(120);
    P = await world.readPoses();
    r.rope = Math.hypot(P[0], P[1] - 5, P[2]);
    world.clear();
    const sl = world.addBox({ position: [0, 3, 0], half: [0.2, 0.2, 0.2] });
    world.addJoint({ type: 'slider', a: sl, b: null, anchor: [0, 3, 0], axis: [0, 1, 0], limits: [-1, 0] });
    world.applyImpulse(sl, [0.5, 0, 0.3]);
    await run(180);
    P = await world.readPoses();
    r.slider = [P[0], P[1], P[2]];
    world.clear();
    const weak = world.addBox({ position: [0, 3, 0], half: [0.5, 0.5, 0.5] });
    world.addJoint({ type: 'ball', a: weak, b: null, anchor: [0, 3.5, 0], breakForce: 5 });
    const strong = world.addBox({ position: [3, 3, 0], half: [0.5, 0.5, 0.5] });
    world.addJoint({ type: 'ball', a: strong, b: null, anchor: [3, 3.5, 0], breakForce: 20 });
    await run(60);
    const js = await world.readJoints();
    note(`motor ω ${r.motor.toFixed(4)} (2) · limit ${r.limit.toFixed(4)} (−0.4) · weld ${r.weld.toFixed(4)} m (0.6) · rope ${r.rope.toFixed(4)} m (≤ 2) · ` +
      `slider (${r.slider.map((v) => v.toFixed(3))}) ((0, 2, 0)) · joints broken ${js.broken} carrying ${[...js.force].map((f) => f.toFixed(2))} N`);
    assert(Math.abs(r.motor - 2) < 0.01, 'motor');
    assert(Math.abs(r.limit + 0.4) < 0.01, 'hinge limit');
    assert(Math.abs(r.weld - 0.6) < 1e-3, 'weld');
    assert(r.rope < 2.001, 'rope');
    assert(Math.abs(r.slider[1] - 2) < 0.01 && Math.abs(r.slider[0]) < 1e-3 && Math.abs(r.slider[2]) < 1e-3, 'slider');
    assert(js.broken[0] && !js.broken[1] && Math.abs(js.force[1] - 9.81) < 0.05, 'break force');
  });
  await test('a light chain holds a weight 347× heavier than each link (sag < 10 mm)', async () => {
    const world = gpu.rigid({ maxBodies: 32, floor: null });
    const ids = [], top = 6, seg = 0.4;
    for (let k = 0; k < 12; k++) ids.push(world.addCapsule({ position: [0, top - seg / 2 - k * seg, 0], radius: 0.05, halfHeight: seg / 2 - 0.05 }));
    world.addJoint({ type: 'ball', a: ids[0], b: null, anchor: [0, top, 0] });
    for (let k = 1; k < 12; k++) world.addJoint({ type: 'ball', a: ids[k - 1], b: ids[k], anchor: [0, top - k * seg, 0] });
    const boxY = top - 12 * seg - 0.25;
    const heavy = world.addBox({ position: [0, boxY, 0], half: [0.25, 0.25, 0.25], density: 8 });
    world.addJoint({ type: 'ball', a: ids[11], b: heavy, anchor: [0, top - 12 * seg, 0] });
    for (let f = 0; f < 300; f++) { world.step(1 / 60); if (f % 30 === 29) await gpu.idle(); }
    const P = await world.readPoses();
    const sag = (boxY - P[heavy * 7 + 1]) * 1000;
    note(`box sag after 5 s: ${sag.toFixed(2)} mm over a 4.8 m chain`);
    assert(P.every(Number.isFinite) && Math.abs(sag) < 10, 'chain stretches under a heavy load');
  });
  await test('capsules rest at the right height; kinematic platform carries a box; collision filter; sensor events', async () => {
    const world = gpu.rigid({ maxBodies: 32, events: true });
    const run = async (frames) => { for (let f = 0; f < frames; f++) world.step(1 / 60); await gpu.idle(); };
    const lying = world.addCapsule({ position: [0, 0.4, 0], radius: 0.2, halfHeight: 0.5, rotation: [0, 0, Math.PI / 2] });
    const standing = world.addCapsule({ position: [2, 0.8, 0], radius: 0.2, halfHeight: 0.5 });
    world.addBox({ position: [5, 0.25, -0.6], half: [0.25, 0.25, 0.25] });
    world.addBox({ position: [5, 0.25, 0.6], half: [0.25, 0.25, 0.25] });
    const across = world.addCapsule({ position: [5, 0.75, 0], radius: 0.2, halfHeight: 0.7, rotation: [Math.PI / 2, 0, 0] });
    await run(240);
    let P = await world.readPoses();
    const caps = [P[lying * 7 + 1], P[standing * 7 + 1], P[across * 7 + 1]];
    world.clear();
    const plat = world.addBox({ position: [0, 1, 0], half: [1.5, 0.1, 1.5], kinematic: true });
    const rider = world.addBox({ position: [0, 1.35, 0], half: [0.25, 0.25, 0.25] });
    await run(30);
    for (let f = 0; f < 120; f++) { world.moveKinematic(plat, [f / 120, 1, 0]); world.step(1 / 60); }
    P = await world.readPoses();
    const carried = [P[plat * 7], P[rider * 7]];
    world.clear();
    world.addBox({ position: [0, 0.5, 0], half: [0.5, 0.5, 0.5], group: 1 });
    const ghost = world.addBox({ position: [0, 2, 0], half: [0.3, 0.3, 0.3], group: 2, mask: ~1 >>> 0 });
    await run(120);
    P = await world.readPoses();
    const ghostY = P[ghost * 7 + 1];
    world.clear();
    const sensor = world.addBox({ position: [0, 1, 0], half: [0.5, 0.2, 0.5], static: true, sensor: true });
    const ball = world.addSphere({ position: [0, 2.5, 0], radius: 0.2 });
    const ev = [];
    for (let f = 0; f < 90; f++) {
      world.step(1 / 60);
      const e = await world.readEvents();
      for (const p of e.begin) ev.push(`+${p}`);
      for (const p of e.end) ev.push(`-${p}`);
    }
    note(`capsule heights ${caps.map((v) => v.toFixed(4))} (0.2, 0.7, 0.7) · platform ${carried[0].toFixed(3)} rider ${carried[1].toFixed(3)} · filtered box at ${ghostY.toFixed(3)} (0.3) · events ${ev.join(' ')}`);
    assert(Math.abs(caps[0] - 0.2) < 0.005 && Math.abs(caps[1] - 0.7) < 0.005 && Math.abs(caps[2] - 0.7) < 0.005, 'capsule resting heights');
    assert(Math.abs(carried[1] - carried[0]) < 0.05, 'rider left behind');
    assert(Math.abs(ghostY - 0.3) < 0.01, 'filtered body collided');
    assert(ev.join(' ') === `+${sensor},${ball} -${sensor},${ball} +${ball},-1`, 'sensor/floor events');
  });
  await test('raycasts hit boxes, spheres and capsules at the exact distance, honouring masks', async () => {
    const world = gpu.rigid({ maxBodies: 16 });
    world.addBox({ position: [0, 1, 0], half: [0.5, 0.5, 0.5], static: true });
    world.addSphere({ position: [3, 1, 0], radius: 0.5, static: true });
    world.addCapsule({ position: [6, 1, 0], radius: 0.3, halfHeight: 0.5, static: true });
    world.step(1 / 60);
    const hits = await world.raycast([
      { origin: [0, 1, -5], dir: [0, 0, 1] }, { origin: [3, 1, -5], dir: [0, 0, 1] }, { origin: [6, 5, 0], dir: [0, -1, 0] },
      { origin: [6, 1.2, -5], dir: [0, 0, 1] }, { origin: [-5, 10, 0], dir: [0, 1, 0] }, { origin: [-5, 1, 0], dir: [1, 0, 0], mask: 0 },
      { origin: [-5, 1, 0], dir: [1, 0, 0] },
    ]);
    const got = hits.map((h) => (h.hit ? `${h.body}@${h.distance.toFixed(4)}` : 'miss')).join(' ');
    note(got);
    assert(got === '0@4.5000 1@4.5000 2@3.2000 2@4.7000 miss miss 0@4.5000', 'ray hits');
  });
  await test('determinism: the same scene twice gives bit-identical states; snapshot + restore replays exactly', async () => {
    const world = gpu.rigid({ maxBodies: 1024, deterministic: true });
    const build = () => {
      let s = 7;
      const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
      for (let k = 0; k < 500; k++) {
        const p = [(r() - 0.5) * 5, 1 + k * 0.03 + r(), (r() - 0.5) * 5];
        const kind = r();
        if (kind < 0.5) world.addBox({ position: p, half: [0.15 + r() * 0.15, 0.15 + r() * 0.15, 0.15 + r() * 0.15], rotation: [r() * 3, r() * 3, r() * 3] });
        else if (kind < 0.75) world.addSphere({ position: p, radius: 0.15 + r() * 0.1 });
        else world.addCapsule({ position: p, radius: 0.12, halfHeight: 0.2, rotation: [r() * 3, r() * 3, r() * 3] });
      }
    };
    const hashes = [];
    for (let rep = 0; rep < 2; rep++) {
      world.clear(); build();
      const h = [];
      for (let f = 0; f < 180; f++) { world.step(1 / 60); if (f % 60 === 59) h.push(await world.hash()); }
      hashes.push(h.join(','));
    }
    const snap = world.snapshot();
    for (let f = 0; f < 60; f++) world.step(1 / 60);
    const a = await world.hash();
    world.restore(snap);
    for (let f = 0; f < 60; f++) world.step(1 / 60);
    const b = await world.hash();
    note(`runs: ${hashes[0] === hashes[1] ? 'identical' : 'DIFFERENT'} (${hashes[0].split(',').pop()}); replay after restore: ${a === b ? 'identical' : 'DIFFERENT'} (${a})`);
    assert(hashes[0] === hashes[1], 'two identical runs diverged');
    assert(a === b, 'restore + replay diverged');
  });
  await test('small worlds solve in one fused dispatch: deterministic, and equal to the multi-dispatch solve up to rounding', async () => {
    const build = (fusedSolve) => {
      const w = gpu.rigid({ maxBodies: 256, deterministic: true, fusedSolve });
      let s = 5;
      const R = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
      for (let k = 0; k < 80; k++) w.addBox({ position: [(R() - 0.5) * 3, 0.5 + k * 0.05, (R() - 0.5) * 3], half: [0.15, 0.1 + R() * 0.1, 0.15], rotation: [R(), R(), R()] });
      const a = w.addSphere({ position: [0, 6, 0], radius: 0.3 }), b = w.addSphere({ position: [0.6, 6, 0], radius: 0.3 });
      w.addJoint({ type: 'distance', a, b, length: 0.6 });
      return w;
    };
    const [f1, f2, m] = [build(true), build(true), build(false)];
    for (const w of [f1, f2, m]) for (let k = 0; k < 40; k++) w.step(1 / 60);   // let the stack form: contacts, colours, warm starts
    await gpu.idle();
    // one more step from the same state: the two paths may differ only by rounding
    m.restore(f1.snapshot());
    f1.step(1 / 60); m.step(1 / 60);
    const [pf, pm] = [await f1.readPoses(), await m.readPoses()];
    let d = 0;
    for (let k = 0; k < pf.length; k++) if (k % 7 < 3) d = Math.max(d, Math.abs(pf[k] - pm[k]));
    for (let k = 0; k < 30; k++) f2.step(1 / 60);
    for (let k = 0; k < 29; k++) f1.step(1 / 60);
    const [h1, h2] = [await f1.hash(), await f2.hash()];
    note(`fused vs multi-dispatch after one step from the same state: max |Δx| ${d.toExponential(1)} m; two fused runs: ${h1 === h2 ? 'identical' : 'DIFFERENT'} (${h1})`);
    assert(d < 1e-5 && h1 === h2, 'fused solve wrong or not deterministic');
  });
  await test('2,000 mixed boxes and spheres pile up without NaN or tunnelling', async () => {
    const world = gpu.rigid({ maxBodies: 4096 });
    const R = rng(11);
    for (let k = 0; k < 2000; k++) {
      const p = [(R() - 0.5) * 6, 1 + k * 0.01 + R() * 2, (R() - 0.5) * 6];
      if (R() < 0.7) world.addBox({ position: p, half: [0.1 + R() * 0.15, 0.1 + R() * 0.15, 0.1 + R() * 0.15], rotation: [R() * 3, R() * 3, R() * 3] });
      else world.addSphere({ position: p, radius: 0.1 + R() * 0.1 });
    }
    await run(world, 300);
    const P = await world.readPoses();
    let nan = 0, below = 0, top = 0;
    for (let k = 0; k < world.count; k++) {
      if (!Number.isFinite(P[7 * k]) || !Number.isFinite(P[7 * k + 1])) nan++;
      else { if (P[7 * k + 1] < -0.02) below++; top = Math.max(top, P[7 * k + 1]); }
    }
    const st = await world.readStats();
    note(`${world.count} bodies after 5 s: NaN ${nan}, below the floor ${below}, highest centre ${top.toFixed(2)} m; ${st.contacts} contacts in ${st.colors} colours`);
    assert(nan === 0 && below === 0, 'unstable or tunnelling');
  });
  await test('rolling and spinning keep their speed: balls and a capsule roll 10 s on the flat, a free box spins 2 s', async () => {
    // Regression: small-angle rotation maps braked fast-turning bodies (−58% in 10 s for r = 0.1 m),
    // and anchors / warm starts that assumed a fixed material point let rolling balls sink 2 cm.
    const w = gpu.rigid({ maxBodies: 64, bounds: [[-60, -2, -60], [60, 10, 60]] });
    const cases = [[0.1, 2], [0.3, 8], [1.4, 3]].map(([r, v], k) =>
      ({ r, v, id: w.addSphere({ position: [-50, r, k * 4 - 6], radius: r, velocity: [v, 0, 0], angularVelocity: [0, 0, -v / r] }) }));
    cases.push({ r: 0.2, v: 3, id: w.addCapsule({ position: [-50, 0.2, 8], radius: 0.2, halfHeight: 0.5, rotation: [Math.PI / 2, 0, 0], velocity: [3, 0, 0], angularVelocity: [0, 0, -15] }) });
    const s = gpu.rigid({ maxBodies: 8, floor: null });
    s.gravity = [0, 0, 0];
    const spin = s.addBox({ position: [0, 5, 0], half: [0.5, 0.2, 0.3], angularVelocity: [0, 20, 0] });
    await run(w, 600); await run(s, 120);
    const [b, bs] = [await w.readBodies(), await s.readBodies()];
    const res = cases.map(({ r, v, id }) => ({ r, v, speed: Math.hypot(...b.velocity.slice(3 * id, 3 * id + 3)), sink: r - b.position[3 * id + 1] }));
    const w20 = bs.angularVelocity[3 * spin + 1];
    note(res.map((c) => `r ${c.r}: ${c.v} → ${c.speed.toFixed(3)} m/s, ${(c.sink * 1000).toFixed(2)} mm deep`).join(' · ') + ` · spin 20 → ${w20.toFixed(4)} rad/s`);
    assert(res.every((c) => Math.abs(c.speed - c.v) < 0.01 * c.v && c.sink < 1e-3) && Math.abs(w20 - 20) < 0.02, 'rolling body lost speed or sank');
  });
  await test('a large dynamic block rests on 117 small boxes (more than a small body\'s 64-pair reverse list)', async () => {
    const w = gpu.rigid({ maxBodies: 1024, deterministic: true });
    for (let x = 0; x < 20; x++) for (let z = 0; z < 20; z++) w.addBox({ position: [(x - 9.5) * 0.42, 0.2, (z - 9.5) * 0.42], half: [0.2, 0.2, 0.2] });
    const big = w.addBox({ position: [0.3, 3, -0.2], half: [2, 1, 2], rotation: [0.05, 0.3, 0.02], density: 0.3 });
    await run(w, 240);
    const y = (await w.readBodies()).position[3 * big + 1];
    note(`block centre ${y.toFixed(4)} m (on top of the boxes: 1.4; on the floor: 1.0); large bodies ${w.nLarge}`);
    assert(Math.abs(y - 1.4) < 5e-3, 'the block sank into the boxes');
  });

  section('N-body');
  await test('Plummer sphere conserves energy (symplectic Euler, 200 steps)', async () => {
    const n = 2048, eps = 0.05;
    const sim = gpu.nbody({ count: n, softening: eps, dt: 0.002 });
    sim.plummer({ scale: 0.3 });
    const energy = async () => {
      const [p, v] = [await sim.pos.read(), await sim.vel.read()];
      let ke = 0, pe = 0;
      for (let a = 0; a < n; a++) {
        ke += 0.5 * p[4 * a + 3] * (v[4 * a] ** 2 + v[4 * a + 1] ** 2 + v[4 * a + 2] ** 2);
        for (let b = a + 1; b < n; b++) {
          const dx = p[4 * a] - p[4 * b], dy = p[4 * a + 1] - p[4 * b + 1], dz = p[4 * a + 2] - p[4 * b + 2];
          pe -= (p[4 * a + 3] * p[4 * b + 3]) / Math.sqrt(dx * dx + dy * dy + dz * dz + eps * eps);
        }
      }
      return ke + pe;
    };
    const e0 = await energy();
    sim.step(200);
    const e1 = await energy();
    const drift = Math.abs((e1 - e0) / e0);
    note(`E ${e0.toFixed(5)} → ${e1.toFixed(5)} (relative drift ${drift.toExponential(2)})`);
    assert(drift < 5e-3, 'energy drift too large');
  });
}
