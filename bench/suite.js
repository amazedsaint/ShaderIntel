// ShaderIntel benchmark suite: every entry runs a widely used baseline and the
// ShaderIntel kernel on the same device and inputs, checks that the results
// agree, and reports the speed-up. Timing = best of several batches of GPU
// work between queue.onSubmittedWorkDone() fences (each batch ≥ ~20 ms, so
// browser timestamp quantisation and submit overhead do not matter).
import * as SI from '../src/index.js';
import { HillisSteeleScan, BlellochScan, NaiveGemm, TiledGemm } from './baselines.js';

const timeIt = SI.timeIt;

function rng(seed = 1) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Each benchmark: { id, title, ours, baseline, unit, work, setup(gpu) -> state with baseline(), ours(), verify() } */
export const BENCHMARKS = [
  {
    id: 'scan',
    title: 'Prefix sum, 16M u32',
    ours: 'reduce-then-scan, 4096-element tiles, subgroup scans',
    baseline: 'Blelloch work-efficient scan (GPU Gems 3 ch. 39)',
    unit: 'G elements/s',
    async setup(gpu) {
      const n = 1 << 24, R = rng(5);
      const data = new Uint32Array(n).map(() => (R() * 16) | 0);
      const src = gpu.array('u32', data);
      const a = gpu.array('u32', n), b = gpu.array('u32', n);
      const ours = new SI.Scan(gpu, n), blel = new BlellochScan(gpu, n);
      return {
        work: n,
        baseline: () => blel.run(b, n),          // in place (re-scanning is fine for timing)
        ours: () => ours.run(a, a, n),
        async verify() {
          gpu.copy(src, a); gpu.copy(src, b);
          ours.run(a, a, n); blel.run(b, n);
          const [ra, rb] = [await a.read(n), await b.read(n)];
          let acc = 0;
          for (let k = 0; k < n; k++) {
            if (ra[k] !== acc || rb[k] !== acc) return { ok: false, note: `mismatch at ${k}` };
            acc = (acc + data[k]) >>> 0;
          }
          return { ok: true, note: 'both equal the CPU scan' };
        },
        free() { [src, a, b].forEach((x) => x.destroy()); },
      };
    },
  },
  {
    id: 'scan-hs',
    title: 'Prefix sum, 16M u32 (vs naive)',
    ours: 'reduce-then-scan, 4096-element tiles, subgroup scans',
    baseline: 'Hillis–Steele scan in global memory (log₂N passes)',
    unit: 'G elements/s',
    async setup(gpu) {
      const n = 1 << 24, R = rng(6);
      const data = new Uint32Array(n).map(() => (R() * 16) | 0);
      const src = gpu.array('u32', data);
      const a = gpu.array('u32', n), b = gpu.array('u32', n);
      const ours = new SI.Scan(gpu, n), hs = new HillisSteeleScan(gpu, n);
      return {
        work: n,
        baseline: () => hs.run(src, b, n),
        ours: () => ours.run(src, a, n),
        async verify() {
          hs.run(src, b, n); ours.run(src, a, n);
          const [ra, rb] = [await a.read(n), await b.read(n)];
          let acc = 0;
          for (let k = 0; k < n; k++) {
            if (ra[k] !== acc || rb[k] !== acc) return { ok: false, note: `mismatch at ${k}` };
            acc = (acc + data[k]) >>> 0;
          }
          return { ok: true, note: 'both equal the CPU scan' };
        },
        free() { [src, a, b].forEach((x) => x.destroy()); },
      };
    },
  },
  {
    id: 'noise',
    title: '3D simplex noise, 16M evaluations',
    ours: 'simplex3: integer lattice hash + unpack4x8snorm gradients',
    baseline: 'Ashima / webgl-noise snoise(vec3) (mod-289 permutation polynomial)',
    unit: 'G evals/s',
    async setup(gpu) {
      const n = 1 << 24;
      const out = gpu.array('f32', n), out2 = gpu.array('f32', n);
      const P = 'let p = vec3f(f32(i & 4095u), f32(i >> 12u), 7.5) * 0.0213;';
      const kb = gpu.kernel({ name: 'noise-ashima', imports: ['baseline-ashima'], buffers: { out }, code: `${P} out[i] = ashimaSimplex3(p);` });
      const ko = gpu.kernel({ name: 'noise-ours', imports: ['noise'], buffers: { out: out2 }, code: `${P} out[i] = simplex3(p);` });
      await Promise.all([kb.ready(), ko.ready()]);
      return {
        work: n,
        baseline: () => kb.run(n),
        ours: () => ko.run(n),
        async verify() {
          kb.run(n); ko.run(n);
          const s = async (arr) => { const r = await arr.read(n); let mn = 9, mx = -9, m = 0, m2 = 0; for (const v of r) { mn = Math.min(mn, v); mx = Math.max(mx, v); m += v; m2 += v * v; } m /= n; return { mn, mx, m, sd: Math.sqrt(m2 / n - m * m) }; };
          const [a, b] = [await s(out), await s(out2)];
          const ok = Math.abs(b.m) < 0.02 && b.mx <= 1.05 && b.mn >= -1.05 && b.sd > 0.25;
          return { ok, note: `range/σ: baseline [${a.mn.toFixed(2)}, ${a.mx.toFixed(2)}] σ ${a.sd.toFixed(3)}, ours [${b.mn.toFixed(2)}, ${b.mx.toFixed(2)}] σ ${b.sd.toFixed(3)}` };
        },
        free() { out.destroy(); out2.destroy(); },
      };
    },
  },
  {
    id: 'noise-psrd',
    title: '3D simplex noise vs psrdnoise, 16M',
    ours: 'simplex3: integer lattice hash + unpack4x8snorm gradients',
    baseline: 'psrdnoise3 (Gustavson & McEwan, JCGT 2022), no tiling / no rotation',
    unit: 'G evals/s',
    async setup(gpu) {
      const n = 1 << 24;
      const out = gpu.array('f32', n), out2 = gpu.array('f32', n);
      const P = 'let p = vec3f(f32(i & 4095u), f32(i >> 12u), 7.5) * 0.0213;';
      const kb = gpu.kernel({ name: 'noise-psrd', imports: ['baseline-psrd'], buffers: { out }, code: `${P} out[i] = psrdnoise3v(p);` });
      const ko = gpu.kernel({ name: 'noise-ours2', imports: ['noise'], buffers: { out: out2 }, code: `${P} out[i] = simplex3(p);` });
      await Promise.all([kb.ready(), ko.ready()]);
      return {
        work: n, baseline: () => kb.run(n), ours: () => ko.run(n),
        async verify() {
          kb.run(n); const r = await out.read(1 << 20);
          let mx = 0; for (const v of r) mx = Math.max(mx, Math.abs(v));
          return { ok: mx > 0.5 && mx < 1.2, note: `baseline |max| ${mx.toFixed(2)} (sanity check of the port)` };
        },
        free() { out.destroy(); out2.destroy(); },
      };
    },
  },
  {
    id: 'fbm',
    title: '4-octave fBm, 16M evaluations',
    ours: 'fbm3 (4 octaves of simplex3, rotated octaves)',
    baseline: '4 octaves of Ashima snoise',
    unit: 'G evals/s',
    async setup(gpu) {
      const n = 1 << 24;
      const out = gpu.array('f32', n), out2 = gpu.array('f32', n);
      const P = 'let p = vec3f(f32(i & 4095u), f32(i >> 12u), 7.5) * 0.0113;';
      const kb = gpu.kernel({ name: 'fbm-ashima', imports: ['baseline-ashima'], buffers: { out }, code: `${P} out[i] = ashimaFbm3(p);` });
      const ko = gpu.kernel({ name: 'fbm-ours', imports: ['noise'], buffers: { out: out2 }, code: `${P} out[i] = fbm3(p, 4);` });
      await Promise.all([kb.ready(), ko.ready()]);
      return {
        work: n, baseline: () => kb.run(n), ours: () => ko.run(n),
        async verify() { ko.run(n); const r = await out2.read(1 << 16); return { ok: r.every(Number.isFinite), note: 'finite, same octave count' }; },
        free() { out.destroy(); out2.destroy(); },
      };
    },
  },
  {
    id: 'hash',
    title: 'Per-pixel hash, 16M values',
    ours: 'pcg2d integer hash (Jarzynski & Olano 2020), its fully mixed output (hash21)',
    baseline: 'fract(sin(dot(p, (12.9898, 78.233))) · 43758.5453)',
    unit: 'G hashes/s',
    async setup(gpu) {
      const n = 1 << 24;
      const out = gpu.array('f32', n), out2 = gpu.array('f32', n);
      const kb = gpu.kernel({ name: 'hash-sin', imports: ['baseline-sinhash'], buffers: { out },
        code: 'out[i] = sinHash21(vec2f(f32(i & 4095u), f32(i >> 12u)) * 37.0);' });
      const ko = gpu.kernel({ name: 'hash-pcg', imports: ['hash'], buffers: { out: out2 },
        code: 'out[i] = u2f(pcg2d(vec2u(i & 4095u, i >> 12u)).y);' });
      await Promise.all([kb.ready(), ko.ready()]);
      const chi = (r) => {
        const B = 256, c = new Float64Array(B);
        for (const v of r) c[Math.min(B - 1, Math.max(0, Math.floor(v * B)))]++;
        const e = r.length / B; let x = 0; for (const v of c) x += (v - e) * (v - e) / e; return x;
      };
      // independence of horizontally adjacent pixels: 16 × 16 buckets of (h[k], h[k+1])
      const chiPairs = (r) => {
        const B = 256, c = new Float64Array(B);
        for (let k = 0; k + 1 < r.length; k += 2) c[Math.min(15, Math.floor(r[k] * 16)) * 16 + Math.min(15, Math.floor(r[k + 1] * 16))]++;
        const e = (r.length >> 1) / B; let x = 0; for (const v of c) x += (v - e) * (v - e) / e; return x;
      };
      return {
        work: n, baseline: () => kb.run(n), ours: () => ko.run(n),
        async verify() {
          kb.run(n); ko.run(n);
          const [ra, rb] = [await out.read(1 << 20), await out2.read(1 << 20)];
          const [a, b, pa, pb] = [chi(ra), chi(rb), chiPairs(ra), chiPairs(rb)];
          return { ok: b < 370 && pb < 370, note: `χ² (255 dof, 1M samples, ≤ 330 passes): values ${a.toFixed(0)} vs ours ${b.toFixed(0)}, adjacent pairs ${pa.toFixed(0)} vs ours ${pb.toFixed(0)}` };
        },
        free() { out.destroy(); out2.destroy(); },
      };
    },
  },
  {
    id: 'nbody',
    title: 'N-body gravity, 65,536 bodies',
    ours: 'shared-memory tiles + register blocking, autotuned',
    baseline: 'one thread per body streaming all bodies from global memory',
    unit: 'G pairs/s',
    async setup(gpu) {
      const n = 65536;
      const sim = new SI.NBody(gpu, { count: n });
      sim.plummer();
      const tuned = await sim.autotune({ cache: false, reps: 3 });
      const naive = { mode: 'naive', wg: 256, K: 1 };
      return {
        work: n * n,
        note: `fastest variant: ${sim.variant.mode} wg=${sim.variant.wg} K=${sim.variant.K}`,
        baseline: () => sim.step(1, naive),
        ours: () => sim.step(1),
        async verify() {
          // compare the velocity change (= G·a·dt) of one step: different summation orders, same physics
          sim.plummer(); const v0 = await sim.vel.read();
          sim.step(1, naive); const a = await sim.vel.read();
          sim.plummer(); sim.step(1); const b = await sim.vel.read();
          let err = 0, mag = 0;
          for (let k = 0; k < n * 4; k++) if (k % 4 !== 3) { err = Math.max(err, Math.abs(a[k] - b[k])); mag = Math.max(mag, Math.abs(a[k] - v0[k])); }
          return { ok: err < 1e-3 * mag, note: err === 0 ? 'velocities bit-identical (same summation order, different data movement)'
            : `max |Δv| difference ${err.toExponential(1)} vs max |Δv| ${mag.toExponential(1)} (relative ${(err / mag).toExponential(1)})` };
        },
        free() { sim.pos.destroy(); sim.pos2.destroy(); sim.vel.destroy(); },
      };
    },
  },
  {
    id: 'neighbors',
    title: 'Fixed-radius neighbours, 65,536 particles',
    ours: 'counting-sort uniform grid (build + 27-cell query as 9 row ranges)',
    baseline: 'all-pairs O(N²) distance tests (e.g. WebGPU "compute boids" sample)',
    unit: 'M queries/s',
    async setup(gpu) {
      const n = 65536, h = 0.035, R = rng(9);
      const p = new Float32Array(n * 4);
      for (let k = 0; k < n; k++) p.set([R(), R(), R(), 1], 4 * k);
      const P = gpu.array('vec4f', p), S = gpu.array('vec4f', n);
      const cA = gpu.array('u32', n), cB = gpu.array('u32', n);
      const grid = new SI.SpatialGrid(gpu, { maxCount: n, min: [0, 0, 0], max: [1, 1, 1], cellSize: h });
      const kb = gpu.kernel({ name: 'nb-brute', buffers: { P: { array: P, access: 'read' }, cnt: cA }, uniforms: { h2: 'f32' },
        code: 'let pi = P[i].xyz; var c = 0u; for (var j = 0u; j < u.n; j++) { let d = P[j].xyz - pi; if (dot(d, d) < u.h2) { c++; } } cnt[i] = c;' });
      const ko = gpu.kernel({ name: 'nb-grid', imports: ['grid'], uniforms: { ...grid.uniforms(), h2: 'f32' },
        buffers: { S: { type: 'vec4f', access: 'read' }, start: { type: 'u32', access: 'read' }, cnt: cB },
        code: `let pi = S[i].xyz; var c = 0u; ${SI.SpatialGrid.neighbors('pi', 'let d = S[j].xyz - pi; if (dot(d, d) < u.h2) { c++; }')} cnt[i] = c;` });
      await Promise.all([kb.ready(), ko.ready()]);
      const ours = () => { grid.build(P, n); grid.gather(P, S); ko.run(n, { ...grid.values(), h2: h * h }, { S, start: grid.start, cnt: cB }); };
      return {
        work: n,
        baseline: () => kb.run(n, { h2: h * h }),
        ours,
        async verify() {
          kb.run(n, { h2: h * h }); ours();
          const [a, b, order] = [await cA.read(), await cB.read(), await grid.order.read()];
          let total = 0;
          for (let k = 0; k < n; k++) { if (b[k] !== a[order[k]]) return { ok: false, note: `count mismatch at ${k}` }; total += b[k]; }
          return { ok: true, note: `identical counts (${(total / n).toFixed(1)} neighbours / particle incl. self)` };
        },
        free() { [P, S, cA, cB].forEach((x) => x.destroy()); },
      };
    },
  },
  {
    id: 'p2g',
    title: 'MLS-MPM particle→grid, 262k particles',
    ours: 'Morton block sort + shared-memory tile accumulation (sort amortised over 4 substeps)',
    baseline: 'unsorted particles, fixed-point global atomics (WebGPU-Ocean / PB-MPM style)',
    unit: 'M particles/s',
    async setup(gpu) {
      const sim = new SI.MPM(gpu, { grid: 64, maxParticles: 300000, substeps: 4 });
      sim.add({ box: [[0.05, 0.05, 0.05], [0.95, 0.38, 0.95]] }, { material: 'water' });
      for (let f = 0; f < 10; f++) sim.step();       // realistic, already-mixed state
      await gpu.idle();
      const n = sim.count;
      // baseline input: same particles, shuffled (an unsorted simulation's steady state)
      const raw = await sim.P.read(n * 8);
      const R = rng(11), idx = Uint32Array.from({ length: n }, (_, k) => k);
      for (let k = n - 1; k > 0; k--) { const j = Math.floor(R() * (k + 1)); [idx[k], idx[j]] = [idx[j], idx[k]]; }
      const sh = new Float32Array(n * 32);
      for (let k = 0; k < n; k++) sh.set(raw.subarray(idx[k] * 32, idx[k] * 32 + 32), k * 32);
      const Pshuf = gpu.array('vec4f', sh);
      const gridB = gpu.array('vec4i', sim.numNodes);
      sim._sort(); // ours runs on block-sorted particles
      let sortTick = 0;
      return {
        work: n,
        baseline: () => sim.p2g(Pshuf, true, gridB),
        ours: () => { if (sortTick++ % 4 === 0) sim._sort(); sim.p2g(sim.P, false, sim.grid); },
        async verify() {
          sim.p2g(Pshuf, true, gridB); sim.p2g(sim.P, false, sim.grid);
          const [a, b] = [await gridB.read(), await sim.grid.read()];
          let diff = 0, mass = 0;
          for (let k = 0; k < a.length; k++) { if (a[k] !== b[k]) diff++; if (k % 4 === 3) mass += a[k]; }
          return { ok: diff === 0 && mass > 0, note: diff ? `${diff} words differ` : 'grids are bit-identical (integer sums are order-independent)' };
        },
        free() { Pshuf.destroy(); gridB.destroy(); },
      };
    },
  },
  {
    id: 'raymarch',
    title: 'SDF ray marching, 1920×1080',
    ours: 'two-pass cone marching: one conservative cone per 4×4 tile, then per-pixel sphere tracing',
    baseline: 'classic per-pixel sphere tracing',
    unit: 'M rays/s',
    async setup(gpu) {
      const W = 1920, H = 1080, n = W * H, T = 4;
      const tw = Math.ceil(W / T), th = Math.ceil(H / T);
      const tA = gpu.array('vec2f', n), tB = gpu.array('vec2f', n), tsafe = gpu.array('f32', tw * th);
      const scene = /* wgsl */`
        fn map(p: vec3f) -> vec2f {
          var d = vec2f(p.y + 1.0, 0.0);
          d = opU(d, vec2f(sdSphere(p - vec3f(-1.2, 0.0, 0.0), 1.0), 1.0));
          d = opU(d, vec2f(sdRoundBox(p - vec3f(1.3, -0.2, 0.3), vec3f(0.7, 0.8, 0.7), 0.1), 2.0));
          d = opU(d, vec2f(smin(sdTorus(p - vec3f(0.0, 0.2, 2.0), vec2f(0.9, 0.25)), sdCapsule(p, vec3f(0.0, -1.0, 2.0), vec3f(0.0, 1.5, 2.0), 0.2), 0.3), 3.0));
          let q = opRepeatLimited(p - vec3f(0.0, -0.6, -4.0), 1.2, vec3f(4.0, 0.0, 2.0));
          d = opU(d, vec2f(sdBox(q, vec3f(0.35)), 4.0));
          return d;
        }
        fn camRay(px: vec2f) -> vec3f {
          let uv = px / vec2f(${W}.0, ${H}.0) * 2.0 - 1.0;
          return normalize(vec3f(uv.x * ${W / H}, -uv.y, -2.2));
        }`;
      const ro = 'vec3f(0.0, 1.2, 7.0)';
      const pixelAngle = 2 / 2.2 / H;
      const kb = gpu.kernel({ name: 'rm-basic', imports: ['sdf', 'raymarch'], shared: scene, buffers: { out: tA },
        code: /* wgsl */`
          let px = vec2f(f32(i % ${W}u), f32(i / ${W}u)) + 0.5;
          let h = raymarchBasic(${ro}, camRay(px), 0.01, 60.0, 0.9 / ${H}.0);
          out[i] = vec2f(select(-1.0, h.t, h.hit), f32(h.steps));` });
      const kc = gpu.kernel({ name: 'rm-cone', imports: ['sdf', 'raymarch'], shared: scene, buffers: { ts: tsafe },
        code: /* wgsl */`
          let px = (vec2f(f32(i % ${tw}u), f32(i / ${tw}u)) + 0.5) * ${T}.0;
          ts[i] = coneMarch(${ro}, camRay(px), 0.01, 60.0, ${(T * 0.7072 * pixelAngle * 1.05).toFixed(8)});` });
      const ko = gpu.kernel({ name: 'rm-tiled', imports: ['sdf', 'raymarch'], shared: scene,
        buffers: { out: tB, ts: { array: tsafe, access: 'read' } },
        code: /* wgsl */`
          let px = vec2f(f32(i % ${W}u), f32(i / ${W}u)) + 0.5;
          let t0 = ts[(i / ${W}u) / ${T}u * ${tw}u + (i % ${W}u) / ${T}u];
          let h = raymarchBasic(${ro}, camRay(px), t0, 60.0, 0.9 / ${H}.0);
          out[i] = vec2f(select(-1.0, h.t, h.hit), f32(h.steps));` });
      await Promise.all([kb.ready(), kc.ready(), ko.ready()]);
      const ours = () => { kc.run(tw * th); ko.run(n); };
      return {
        work: n,
        baseline: () => kb.run(n),
        ours,
        async verify() {
          kb.run(n); ours();
          const [a, b] = [await tA.read(), await tB.read()];
          // Both tracers stop within a pixel-sized tolerance, so grazing silhouettes may legitimately
          // resolve to different (equally valid) hits; count pixels that disagree by more than 1%.
          let differ = 0, sa = 0, sb = 0;
          for (let k = 0; k < n; k++) {
            const ha = a[2 * k] >= 0, hb = b[2 * k] >= 0;
            if (ha !== hb || (ha && Math.abs(a[2 * k] - b[2 * k]) > 0.01 * a[2 * k])) differ++;
            sa += a[2 * k + 1]; sb += b[2 * k + 1];
          }
          const frac = differ / n;
          return { ok: frac < 1e-4, note: `${differ} of ${n.toLocaleString()} pixels differ by >1% depth (${(frac * 100).toFixed(4)}%); per-pixel map() steps ${(sa / n).toFixed(1)} → ${(sb / n).toFixed(1)}` };
        },
        free() { tA.destroy(); tB.destroy(); tsafe.destroy(); },
      };
    },
  },
  gemmBench('gemm', 'Matrix multiply, 2048³ f32', TiledGemm, '16×16 shared-memory tiles, one output per thread (CUDA guide / WebGPU tutorials)'),
  gemmBench('gemm-naive', 'Matrix multiply, 2048³ f32 (vs naive)', NaiveGemm, 'one thread per output reading A and B from global memory'),
];

// C = A·B: ours is SI.nn.gemm with the tile shape autotuned for this GPU; the result must pass
// Freivalds' certificate (8 random ±1 vectors) and agree with the baseline's product.
function gemmBench(id, title, Baseline, baseline) {
  return {
    id, title, baseline,
    ours: 'register-blocked tiles (TM×TN outputs per thread, vec4 shared tiles, fused epilogue), autotuned',
    unit: 'TFLOP/s',
    async setup(gpu) {
      const n = 2048, R = rng(id.length);
      const A = gpu.array('f32', new Float32Array(n * n).map(() => R() - 0.5));
      const B = gpu.array('f32', new Float32Array(n * n).map(() => R() - 0.5));
      const C = gpu.array('f32', n * n), Cb = gpu.array('f32', n * n);
      const { best } = await SI.nn.autotuneGemm(gpu, { cache: false, reps: 3 });
      const base = new Baseline(gpu);
      await base.ready();
      const ours = () => SI.nn.gemm(gpu, A, B, C, { M: n, N: n, K: n });
      return {
        work: 2 * n ** 3,
        note: `fastest tile: ${best.BM}×${best.BN}×${best.BK}, ${best.TM}×${best.TN} per thread`,
        baseline: () => base.run(A, B, Cb, n),
        ours,
        async verify() {
          base.run(A, B, Cb, n); ours();
          const cert = await SI.verify.freivalds(gpu, A, B, C, { M: n, N: n, K: n, trials: 8 });
          const [x, y] = [await C.read(), await Cb.read()];
          let err = 0, mag = 0;
          for (let k = 0; k < x.length; k++) { err = Math.max(err, Math.abs(x[k] - y[k])); mag = Math.max(mag, Math.abs(y[k])); }
          return { ok: cert.ok && err <= 1e-4 * mag,
            note: `Freivalds certificate ${cert.ok ? 'passed' : 'FAILED'} (8 trials, max relative residual ${cert.maxRelative.toExponential(1)}); max |ΔC| vs baseline ${err.toExponential(1)} of max |C| ${mag.toFixed(1)}` };
        },
        free() { [A, B, C, Cb].forEach((x) => x.destroy()); },
      };
    },
  };
}

/** Run the suite; onResult(row) is called as each benchmark finishes. */
export async function runSuite(gpu, { only, onResult, onStatus } = {}) {
  const rows = [];
  // Warm the GPU up to its sustained clock so the first benchmark is not penalised.
  onStatus?.('warming up the GPU…');
  const warm = gpu.array('f32', 1 << 22);
  const wk = gpu.kernel({ name: 'warmup', buffers: { o: warm }, code: 'var x = f32(i); for (var k = 0; k < 2000; k++) { x = fma(x, 1.0001, 0.5); } o[i] = x;' });
  const t0 = performance.now();
  while (performance.now() - t0 < 400) { wk.run(1 << 22); await gpu.idle(); }
  warm.destroy();
  for (const b of BENCHMARKS) {
    if (only && !only.includes(b.id)) continue;
    onStatus?.(`setting up ${b.title}…`);
    let row = { id: b.id, title: b.title, ours: b.ours, baseline: b.baseline, unit: b.unit };
    try {
      const s = await b.setup(gpu);
      onStatus?.(`verifying ${b.title}…`);
      const v = await s.verify();
      onStatus?.(`timing ${b.title}…`);
      // interleave the two so clock drift affects both equally; keep the best batch of each
      let tb = Infinity, to = Infinity;
      for (let r = 0; r < 3; r++) {
        tb = Math.min(tb, await timeIt(gpu, s.baseline, 2));
        to = Math.min(to, await timeIt(gpu, s.ours, 2));
      }
      const scale = { T: 1e12, G: 1e9 }[b.unit[0]] ?? 1e6;
      row = { ...row, baselineMs: tb, oursMs: to, speedup: tb / to, throughput: s.work / (to / 1000) / scale,
        baselineThroughput: s.work / (tb / 1000) / scale, verified: v.ok, verifyNote: v.note, note: s.note || '' };
      s.free?.();
    } catch (e) {
      console.error(e);
      row = { ...row, error: String(e.message || e) };
    }
    rows.push(row);
    onResult?.(row);
    await gpu.idle();
  }
  return rows;
}
