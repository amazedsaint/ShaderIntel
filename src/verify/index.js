// Certificates: small, independent checkers for the results of the big, optimised kernels.
//
// This follows the design of proof assistants (Lean's small kernel; Bend 2's `--verdict`, which
// re-checks every proof with a separate kernel verified in Lean) and of proof-carrying code:
// the producer (an optimised GPU kernel — tiled, subgrouped, autotuned) is untrusted; each
// result comes with a cheap check written as plainly as possible in a few lines, sharing no
// code with the producer. A checker is either exact (prefix sums, colourings, sortedness),
// probabilistic with a stated error bound (Freivalds), or a physical law with a tolerance
// (complementarity, conservation, Coulomb's cone). The laws of the library are listed in
// LAWS.md and run by proveAll() — the analogue of `bend PROOF.bend` before a release.

import { MSELoss } from '../nn/nn.js';
import { COLOR_ROUNDS } from '../physics/rigid.js';
import { destroyAll } from '../core/dispose.js';

const verdict = (ok, evidence) => ({ ok, ...evidence });

/** Exclusive prefix sum: out[0] = 0 and out[k+1] − out[k] = in[k] (u32, wrapping). Exact. */
export async function checkScan(ctx, input, output, n) {
  const bad = ctx.array('u32', 4);
  const k = ctx.kernel({
    name: 'verify-scan', buffers: { a: { type: 'u32', access: 'read' }, s: { type: 'u32', access: 'read' }, bad: { type: 'u32', atomic: true } },
    code: `
      let ok = select(s[i + 1u] - s[i] == a[i], true, i + 1u >= u.n) && (i != 0u || s[0] == 0u);
      if (!ok) { atomicAdd(&bad[0], 1u); atomicMin(&bad[1], i); }`,
  });
  ctx.fill(bad, 0);
  const init = ctx.kernel({ name: 'verify-scan-init', buffers: { bad: { type: 'u32' } }, code: 'if (i == 1u) { bad[1] = 0xffffffffu; }' });
  init.run(4, null, { bad });
  k.run(n, null, { a: input, s: output, bad });
  const r = await bad.read(2);
  bad.destroy();
  return verdict(r[0] === 0, { violations: r[0], first: r[0] ? r[1] : -1 });
}

/** keys[k] ≤ keys[k+1] for k < n − 1 (u32). Exact. */
export async function checkSorted(ctx, keys, n) {
  const bad = ctx.array('u32', 4);
  ctx.kernel({ name: 'verify-sorted', buffers: { a: { type: 'u32', access: 'read' }, bad: { type: 'u32', atomic: true } },
    code: 'if (i + 1u < u.n && a[i] > a[i + 1u]) { atomicAdd(&bad[0], 1u); }' }).run(n, null, { a: keys, bad });
  const r = await bad.read(1);
  bad.destroy();
  return verdict(r[0] === 0, { violations: r[0] });
}

/**
 * a and b hold the same multiset of u32 values: two order-independent fingerprints (sums of
 * independent hashes) must match. Probabilistic: a false accept needs a collision in both 32-bit sums.
 */
export async function checkPermutation(ctx, a, b, n) {
  const fp = ctx.array('u32', 4);
  const k = ctx.kernel({ name: 'verify-perm', imports: ['hash'], buffers: { x: { type: 'u32', access: 'read' }, fp: { type: 'u32', atomic: true } },
    uniforms: { side: 'u32' },
    code: `let h = x[i]; atomicAdd(&fp[2u * u.side], pcg(h ^ 0x9e3779b9u)); atomicAdd(&fp[2u * u.side + 1u], pcg(h * 0x85ebca6bu + 7u));` });
  k.run(n, { side: 0 }, { x: a, fp });
  k.run(n, { side: 1 }, { x: b, fp });
  const r = await fp.read(4);
  fp.destroy();
  return verdict(r[0] === r[2] && r[1] === r[3], { fingerprints: Array.from(r) });
}

/**
 * Freivalds' certificate for C = op(A) op(B): for random ±1 vectors r, compare A(Br) with Cr.
 * O(n²) per trial against the O(n³) product; a wrong C survives a trial with probability ≤ 1/2,
 * so the vectors are fresh random bits on every call (fixed vectors could be defeated, and with
 * N > 2^trials some columns would always share a sign pattern). An error of at least tol times
 * its row's L2 norm in a single entry is caught by every trial.
 * o: { M, N, K, transA, transB, trials (default 4), tol (relative to the row's L2 norm, default 1e-4),
 *      seed (optional: reproducible vectors, which forfeits the guarantee against adversarial C) }
 */
export async function freivalds(ctx, A, B, C, o) {
  const { M, N, K } = o;
  const trials = o.trials ?? 4, tol = o.tol ?? 1e-4;
  const r = ctx.array('f32', N), Br = ctx.array('f32', K), ABr = ctx.array('f32', M), Cr = ctx.array('f32', M), mag = ctx.array('f32', M);
  const matvec = (name, rows, cols, trans) => ctx.kernel({
    name, buffers: { X: { type: 'f32', access: 'read' }, v: { type: 'f32', access: 'read' }, y: { type: 'f32' } },
    uniforms: { cols: 'u32' },
    code: `var s = 0.0; for (var c = 0u; c < u.cols; c++) { s += X[${trans ? `c * u.n + i` : 'i * u.cols + c'}] * v[c]; } y[i] = s;`,
  });
  // scale of (Cr)_i for random ±1 r: the row's L2 norm (an L1 bound would hide single bad entries)
  const absvec = ctx.kernel({ name: 'verify-rownorm', buffers: { X: { type: 'f32', access: 'read' }, y: { type: 'f32' } }, uniforms: { cols: 'u32' },
    code: 'var s = 0.0; for (var c = 0u; c < u.cols; c++) { let x = X[i * u.cols + c]; s += x * x; } y[i] = sqrt(s);' });
  let maxRel = 0;
  const bits = new Uint8Array(N);
  let seed = o.seed === undefined ? 0 : (o.seed >>> 0) || 1;
  for (let t = 0; t < trials; t++) {
    const rv = new Float32Array(N);
    if (o.seed === undefined) globalThis.crypto.getRandomValues(bits);
    else for (let k = 0; k < N; k++) { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; bits[k] = seed; }
    for (let k = 0; k < N; k++) rv[k] = bits[k] & 1 ? 1 : -1;
    ctx.flush();
    r.write(rv);
    // B r : B is K×N (or N×K transposed)
    matvec('verify-mv-b', K, N, !!o.transB).run(K, { cols: N }, { X: B, v: r, y: Br });
    matvec('verify-mv-a', M, K, !!o.transA).run(M, { cols: K }, { X: A, v: Br, y: ABr });
    matvec('verify-mv-c', M, N, false).run(M, { cols: N }, { X: C, v: r, y: Cr });
    absvec.run(M, { cols: N }, { X: C, y: mag });
    const [x, y, m] = [await ABr.read(), await Cr.read(), await mag.read()];
    // relative to the row's L2 norm only: then an entry off by ≥ tol·‖row‖ is always caught
    for (let k = 0; k < M; k++) maxRel = Math.max(maxRel, Math.abs(x[k] - y[k]) / Math.max(1e-6, m[k]));
  }
  for (const b of [r, Br, ABr, Cr, mag]) b.destroy();
  return verdict(maxRel < tol, { maxRelative: maxRel, trials, falseAcceptBound: 2 ** -trials });
}

/**
 * No two dynamic bodies of one colour share a contact or an active joint (RigidWorld, after a
 * step). Exact. Disabled or broken joints, and joints of disabled bodies, are ignored, as the
 * solver ignores them.
 */
export async function checkColoring(world) {
  const nc = (await world.cnts.read(8))[1];
  const colors = await world.color[COLOR_ROUNDS & 1].read(world.count);
  const raw = await world.C[(world.frame - 1) & 1].read(Math.max(1, nc) * 6);
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  const bodies = await world.readBodies();
  const dyn = (b) => b !== 0xffffffff && b < world.count;
  const massOf = async () => { const B = await world.B.read(world.count * 12); return (k) => B[k * 48 + 36]; };
  const m = await massOf();
  let conflicts = 0;
  for (let k = 0; k < nc; k++) {
    const a = u[k * 24], b = u[k * 24 + 1], sensor = u[k * 24 + 3];
    if (sensor || !dyn(b) || m(a) <= 0 || m(b) <= 0) continue;
    if ((bodies.flags[a] | bodies.flags[b]) & 6) continue;
    if (colors[a] === colors[b]) conflicts++;
  }
  const { broken } = await world.readJoints();
  world.joints.forEach((j, k) => {
    if (!j.enabled || broken[k] || !dyn(j.b) || ((bodies.flags[j.a] | bodies.flags[j.b]) & 4)) return;
    if (m(j.a) > 0 && m(j.b) > 0 && colors[j.a] === colors[j.b]) conflicts++;
  });
  return verdict(conflicts === 0, { conflicts, contacts: nc });
}

/**
 * Contact conditions of the last step (RigidWorld): force sign λ_n ≤ 0 and the isotropic Coulomb
 * cone |λ_t| ≤ μ|λ_n| (up to f32 rounding: 1e-6 absolute, coneTol relative), and bounded
 * penetration. The complementarity |λ_n|·gap of separated contacts is reported as evidence but is
 * not part of `ok` (speculative contacts legitimately carry force across a small gap).
 * o: { maxPenetration (m, default 0.02), coneTol (default 1e-3 relative) }
 */
export async function checkContacts(world, o = {}) {
  const nc = (await world.cnts.read(8))[1];
  if (!nc) return verdict(true, { contacts: 0 });
  const raw = await world.C[(world.frame - 1) & 1].read(nc * 6);
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  const B = await world.B.read(world.count * 12);
  const pose = (k) => ({ x: B.subarray(k * 48, k * 48 + 3), q: B.subarray(k * 48 + 4, k * 48 + 8) });
  const rot = (q, v) => {
    const [x, y, z, w] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
  };
  let signViol = 0, coneViol = 0, maxPen = 0, maxCompl = 0;
  for (let k = 0; k < nc; k++) {
    const o6 = k * 24;
    if (u[o6 + 3]) continue;   // sensor
    const a = u[o6], b = u[o6 + 1];
    const n = raw.subarray(o6 + 4, o6 + 7), mu = raw[o6 + 7];
    const lam = raw.subarray(o6 + 16, o6 + 19);
    if (lam[0] > 1e-6) signViol++;
    if (Math.hypot(lam[1], lam[2]) > mu * Math.abs(lam[0]) * (1 + (o.coneTol ?? 1e-3)) + 1e-6) coneViol++;
    // gap between the two anchors along the normal at the end of the step
    const A = pose(a), pa = rot(A.q, raw.subarray(o6 + 8, o6 + 11)).map((v, i) => v + A.x[i]);
    let pb = Array.from(raw.subarray(o6 + 12, o6 + 15));
    if (b !== 0xffffffff) { const P = pose(b); pb = rot(P.q, pb).map((v, i) => v + P.x[i]); }
    const gap = (pa[0] - pb[0]) * n[0] + (pa[1] - pb[1]) * n[1] + (pa[2] - pb[2]) * n[2];
    maxPen = Math.max(maxPen, -gap);
    if (gap > 0.005) maxCompl = Math.max(maxCompl, Math.abs(lam[0]) * gap);
  }
  const ok = signViol === 0 && coneViol === 0 && maxPen <= (o.maxPenetration ?? 0.02);
  return verdict(ok, { contacts: nc, signViolations: signViol, coneViolations: coneViol, maxPenetration: maxPen, complementarity: maxCompl });
}

/** Grid mass and momentum after P2G equal the particles' (MLS-MPM conservation). */
export async function checkMPM(sim, tol = 1e-4) {
  const n = sim.count, raw = await sim.P.read(n * 8);
  sim.p2g();
  const g = new Int32Array((await sim.grid.read()).buffer);
  const FP = 2 ** 22;
  const mats = sim.materials;
  const vol = 1 / sim.ppcAxis ** 3;
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  let pm = 0, px = 0, py = 0, pz = 0, gm = 0, gx = 0, gy = 0, gz = 0;
  for (let k = 0; k < n; k++) {
    const m = (mats[u[32 * k + 3]]?.density ?? 1) * vol;
    pm += m; px += m * raw[32 * k + 4]; py += m * raw[32 * k + 5]; pz += m * raw[32 * k + 6];
  }
  for (let k = 0; k < g.length; k += 4) { gx += g[k]; gy += g[k + 1]; gz += g[k + 2]; gm += g[k + 3]; }
  gm /= FP; gx /= FP; gy /= FP; gz /= FP;
  const massErr = Math.abs(gm - pm) / pm;
  const momErr = Math.hypot(gx - px, gy - py, gz - pz) / Math.max(1e-9, pm);
  return verdict(massErr < tol && momErr < tol + (n * 27 * 4) / FP / pm, { massError: massErr, momentumError: momErr });
}

/**
 * Backpropagation against central finite differences on random parameters — weights and biases
 * of every layer, each sample drawing the parameter tensor and the index independently (an MLP,
 * MSE loss on X → T). Returns the worst relative error.
 * o: { samples (default 16), eps (default 1e-2), tol (default 0.05), seed }
 */
export async function checkGradient(net, X, T, batch, o = {}) {
  const ctx = net.ctx, outN = net.layers[net.layers.length - 1].out;
  const loss = new MSELoss(ctx, batch * outN);
  try {
    const lossAt = async () => { net.forward(X, batch); loss.run(net.output, T, batch * outN); return loss.readLoss(); };
    net.forward(X, batch);
    net.backward(loss.run(net.output, T, batch * outN), batch);
    const params = net.params;   // [{ w, g }] — W and b of every layer
    const samples = o.samples ?? 16, eps = o.eps ?? 1e-2;
    let worst = 0, s = (o.seed ?? 12345) >>> 0 || 1;
    const next = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
    const seen = new Set();
    for (let q = 0; q < samples; q++) {
      const pi = q < params.length ? q : Math.floor(next() * params.length);   // every tensor at least once
      const P = params[pi];
      const W = await P.w.read(), G = await P.g.read();
      const idx = Math.floor(next() * W.length);
      seen.add(pi);
      const w0 = W[idx];
      W[idx] = w0 + eps; ctx.flush(); P.w.write(W); const lp = await lossAt();
      W[idx] = w0 - eps; ctx.flush(); P.w.write(W); const lm = await lossAt();
      W[idx] = w0; ctx.flush(); P.w.write(W);
      const fd = (lp - lm) / (2 * eps);
      worst = Math.max(worst, Math.abs(fd - G[idx]) / Math.max(1e-4, Math.abs(fd), Math.abs(G[idx])));
    }
    return verdict(worst < (o.tol ?? 0.05), { maxRelativeError: worst, samples, tensorsChecked: seen.size, tensors: params.length });
  } finally {
    loss.destroy();
  }
}

/** The same build run twice gives bit-identical state hashes (needs a deterministic RigidWorld). */
export async function checkDeterminism(makeWorld, steps = 120, every = 30) {
  const runs = [];
  for (let r = 0; r < 2; r++) {
    const w = await makeWorld();
    const h = [];
    for (let s = 0; s < steps; s++) { w.step(1 / 60); if (s % every === every - 1) h.push(await w.hash()); }
    runs.push(h);
  }
  return verdict(runs[0].join() === runs[1].join(), { hashes: runs });
}

/** snapshot → step n → restore → step n reproduces the same state hash. */
export async function checkReplay(world, steps = 60) {
  const snap = world.snapshot();
  try {
    for (let s = 0; s < steps; s++) world.step(1 / 60);
    const a = await world.hash();
    world.restore(snap);
    for (let s = 0; s < steps; s++) world.step(1 / 60);
    const b = await world.hash();
    return verdict(a === b, { first: a, replay: b });
  } finally {
    destroyAll(snap.B, snap.J, snap.ext, snap.C, snap.tbl, snap.cnt);
  }
}

/**
 * The laws of the library (see LAWS.md), each an executable check against a fresh scene.
 * proveAll(gpu) runs them all and returns [{ name, statement, ok, evidence }].
 */
export const LAWS = [
  { name: 'scan', statement: 'Scan.run computes an exclusive prefix sum: s[0] = 0, s[k+1] − s[k] = a[k].', async check(gpu, SI) {
    const n = 1 << 18, a = new Uint32Array(n).map((_, k) => (k * 2654435761) % 1000);
    const A = gpu.array('u32', a), S = gpu.array('u32', n + 1), scan = new SI.Scan(gpu, n + 1);
    try { scan.run(A, S, n); return await checkScan(gpu, A, S, n); } finally { destroyAll(A, S, scan); }
  } },
  { name: 'gemm', statement: 'gemm computes C = A·B (Freivalds certificate, false-accept probability ≤ 2⁻⁸).', async check(gpu, SI) {
    const M = 777, N = 513, K = 1029;
    const A = gpu.array('f32', rnd(M * K, 3)), B = gpu.array('f32', rnd(K * N, 7)), C = gpu.array('f32', M * N);
    try { SI.nn.gemm(gpu, A, B, C, { M, N, K }); return await freivalds(gpu, A, B, C, { M, N, K, trials: 8 }); } finally { destroyAll(A, B, C); }
  } },
  { name: 'gemm-rejects', statement: 'Freivalds rejects a product with a single corrupted entry.', async check(gpu, SI) {
    const M = 300, N = 200, K = 400;
    const A = gpu.array('f32', rnd(M * K, 5)), B = gpu.array('f32', rnd(K * N, 9)), C = gpu.array('f32', M * N);
    try {
      SI.nn.gemm(gpu, A, B, C, { M, N, K });
      const c = await C.read(); c[12345] += 0.5; gpu.flush(); C.write(c);
      const r = await freivalds(gpu, A, B, C, { M, N, K, trials: 8 });
      return verdict(!r.ok, { detected: !r.ok, maxRelative: r.maxRelative });
    } finally { destroyAll(A, B, C); }
  } },
  { name: 'colouring', statement: 'No two dynamic bodies of one colour share a contact or a joint.', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 2048 });
    try {
      let s = 7; const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
      for (let k = 0; k < 1000; k++) w.addBox({ position: [(r() - 0.5) * 5, 0.3 + k * 0.02, (r() - 0.5) * 5], half: [0.2, 0.2, 0.2], rotation: [r(), r(), r()] });
      for (let k = 0; k < 120; k++) w.step(1 / 60);
      return await checkColoring(w);
    } finally { w.destroy(); }
  } },
  { name: 'contacts', statement: 'Contact forces push (λ_n ≤ 0), stay in the Coulomb cone (|λ_t| ≤ μ|λ_n|), and a settled 600-body pile of 20–60 g bodies penetrates less than 2 cm (8 iterations; heavier stacks need substeps).', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 2048 });
    try {
      let s = 9; const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
      for (let k = 0; k < 600; k++) (r() < 0.5 ? w.addBox({ position: [(r() - 0.5) * 4, 0.3 + k * 0.02, (r() - 0.5) * 4], half: [0.2, 0.15, 0.25], rotation: [r(), r(), r()] })
        : w.addSphere({ position: [(r() - 0.5) * 4, 0.3 + k * 0.02, (r() - 0.5) * 4], radius: 0.18 }));
      for (let k = 0; k < 360; k++) w.step(1 / 60);
      return await checkContacts(w, { maxPenetration: 0.02 });
    } finally { w.destroy(); }
  } },
  { name: 'determinism', statement: 'A deterministic RigidWorld is a function of its inputs: two runs give bit-identical states.', async check(gpu) {
    const made = [];
    try {
      return await checkDeterminism(() => {
        const w = gpu.rigid({ maxBodies: 1024, deterministic: true });
        made.push(w);
        for (let k = 0; k < 300; k++) w.addBox({ position: [(k % 7) * 0.45 - 1.5, 0.3 + Math.floor(k / 7) * 0.45, (k % 5) * 0.1], half: [0.2, 0.2, 0.2], rotation: [k * 0.1, k * 0.2, 0] });
        return w;
      });
    } finally { destroyAll(made); }
  } },
  { name: 'replay', statement: 'restore(snapshot) followed by the same steps reproduces the same state.', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 1024, deterministic: true });
    try {
      for (let k = 0; k < 200; k++) w.addSphere({ position: [(k % 5) * 0.5, 0.5 + Math.floor(k / 5) * 0.5, (k % 3) * 0.3], radius: 0.2 });
      for (let k = 0; k < 30; k++) w.step(1 / 60);
      return await checkReplay(w, 60);
    } finally { w.destroy(); }
  } },
  { name: 'mpm-conservation', statement: 'MLS-MPM P2G conserves mass and momentum exactly (fixed-point sums).', async check(gpu) {
    const sim = gpu.mpm({ grid: 32, maxParticles: 50000 });
    try {
      sim.add({ box: [[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]] }, { material: 'water', velocity: [0.3, 0.1, -0.2] });
      for (let k = 0; k < 10; k++) sim.step();
      return await checkMPM(sim);
    } finally { sim.destroy(); }
  } },
  { name: 'backprop', statement: 'MLP.backward equals the finite-difference gradient of the MSE loss, for every weight and bias tensor.', async check(gpu, SI) {
    const net = new SI.nn.MLP(gpu, { layers: [3, 16, 16, 2], act: 'tanh', out: 'none', maxBatch: 64, seed: 4 });
    const X = gpu.array('f32', new Float32Array(64 * 3).map((_, k) => Math.sin(k * 0.37))), T = gpu.array('f32', new Float32Array(64 * 2).map((_, k) => Math.cos(k)));
    try { return await checkGradient(net, X, T, 64); } finally { destroyAll(net, X, T); }
  } },
];

function rnd(n, s) { const x = new Float32Array(n); for (let k = 0; k < n; k++) { s = (s * 16807) % 2147483647; x[k] = s / 2147483647 - 0.5; } return x; }

export async function proveAll(gpu, SI, only) {
  const out = [];
  for (const law of LAWS) {
    if (only && !only.includes(law.name)) continue;
    const t0 = performance.now();
    let r;
    try { r = await law.check(gpu, SI); } catch (e) { r = { ok: false, error: String(e.message || e) }; }
    out.push({ name: law.name, statement: law.statement, ok: !!r.ok, ms: Math.round(performance.now() - t0), evidence: r });
  }
  return out;
}
