import * as SI from '../src/index.js';
import { test, section, note, assert, assertClose, assertArrayEq, rng } from './harness.js';

export async function coreTests() {
  section('types / layout (CPU)');
  await test('struct layout follows WGSL rules', () => {
    const L = SI.structLayout('P', { n: 'u32', a: 'vec3f', b: 'f32', m: 'mat3x3f', c: 'vec2f', h: 'f16' });
    const off = Object.fromEntries(L.fields.map((f) => [f.name, f.offset]));
    assert(off.n === 0 && off.a === 16 && off.b === 28 && off.m === 32 && off.c === 80 && off.h === 88, JSON.stringify(off));
    assert(L.size === 96, 'size ' + L.size);
  });
  await test('f16 conversion round-trips', () => {
    for (const v of [0, 1, -2.5, 65504, 6.1e-5, 1e-7, 0.333]) {
      const back = SI.fromHalf(SI.toHalf(v));
      assertClose(back, v, v > 1e-4 ? 1e-3 : 1e-4);
    }
  });
}

export async function gpuTests(gpu, tag) {
  section(`kernels [${tag}]`);
  await test('kernel: axpy with uniforms', async () => {
    const n = 100003;
    const x = new Float32Array(n).map((_, k) => k * 0.5);
    const y = new Float32Array(n).map((_, k) => 1 - k);
    const X = gpu.array('f32', x), Y = gpu.array('f32', y);
    const k = gpu.kernel({ buffers: { X, Y }, uniforms: { a: 'f32' }, code: 'Y[i] = u.a * X[i] + Y[i];' });
    await k.ready();
    k.run(n, { a: 3 });
    const r = await Y.read();
    for (let j = 0; j < n; j += 997) assertClose(r[j], 3 * x[j] + y[j], 1e-6, `at ${j}`);
    assertClose(r[n - 1], 3 * x[n - 1] + y[n - 1], 1e-6);
  });
  await test('uniform ring: per-dispatch values within one submit', async () => {
    const A = gpu.array('u32', 64);
    const k = gpu.kernel({ buffers: { A }, uniforms: { add: 'u32' }, code: 'A[i] = A[i] + u.add;' });
    A.clear();
    for (let s = 1; s <= 20; s++) k.run(64, { add: s });
    k.run(64, { add: 20 }); // dedupe path: same bytes as previous slot
    const r = await A.read();
    assert(r[0] === 230 && r[63] === 230, `got ${r[0]}, ${r[63]}`);
  });
  await test('kernel: >65535 workgroups (2D split dispatch)', async () => {
    const n = 65536 * 64 + 7; // 16,384+ groups of 256
    const A = gpu.array('u32', n);
    const k = gpu.kernel({ buffers: { A }, workgroupSize: 64, code: 'A[i] = i * 3u;' });
    k.run(n);
    const r = await A.read();
    assert(r[0] === 0 && r[n - 1] === (n - 1) * 3 && r[123457] === 123457 * 3, 'bad values');
  });
  await test('kernel: 2D dispatch with p/u.size', async () => {
    const W = 37, H = 23;
    const A = gpu.array('u32', W * H);
    const k = gpu.kernel({ buffers: { A }, workgroupSize: [8, 8], code: 'A[p.y * u.size.x + p.x] = p.x * 1000u + p.y;' });
    k.run([W, H]);
    const r = await A.read();
    assert(r[5 * W + 7] === 7005 && r[(H - 1) * W + W - 1] === (W - 1) * 1000 + H - 1, 'bad 2D');
  });
  await test('fill + fillFloat', async () => {
    const A = gpu.array('f32', 1000);
    gpu.fillFloat(A, 2.5);
    const r = await A.read();
    assert(r[0] === 2.5 && r[999] === 2.5, 'fill');
  });
  await test('compile errors are reported with source lines', async () => {
    const k = gpu.kernel({ buffers: { A: { type: 'f32' } }, code: 'A[i] = undefinedThing;' });
    let threw = false;
    try { await k.ready(); } catch (e) { threw = /undefinedThing/.test(e.message); }
    assert(threw, 'expected a readable compile error');
  });

  section(`scan [${tag}]`);
  const R = rng(42);
  for (const n of [1, 3, 4, 17, 4096, 4097, 65537, 1 << 20, (1 << 24) + 13]) {
    await test(`exclusive scan n=${n}`, async () => {
      const data = new Uint32Array(n).map(() => (R() * 16) | 0);
      const src = gpu.array('u32', data);
      const dst = gpu.array('u32', n);
      const scan = new SI.Scan(gpu, n);
      scan.run(src, dst, n);
      const got = await dst.read(n);
      let acc = 0;
      for (let k = 0; k < n; k++) {
        if (got[k] !== acc) throw new Error(`mismatch at ${k}: got ${got[k]} want ${acc}`);
        acc = (acc + data[k]) >>> 0;
      }
      src.destroy(); dst.destroy();
    });
  }
  await test('single-pass scan (decoupled look-back + fallback) n=5M', async () => {
    const n = 5_000_011;
    const data = new Uint32Array(n).map(() => (R() * 9) | 0);
    const src = gpu.array('u32', data), dst = gpu.array('u32', n);
    for (const lookback of ['workgroup', 'thread', ...(gpu.caps.subgroups ? ['subgroup'] : [])]) {
      new SI.Scan(gpu, n, { mode: 'single-pass', lookback, spins: 32 }).run(src, dst, n);
      const got = await dst.read(n);
      let acc = 0;
      for (let k = 0; k < n; k++) { if (got[k] !== acc) throw new Error(`${lookback}: mismatch at ${k}`); acc += data[k]; }
    }
    src.destroy(); dst.destroy();
  });
  await test('scan in place', async () => {
    const n = 300001;
    const data = new Uint32Array(n).map(() => (R() * 5) | 0);
    const a = gpu.array('u32', data);
    new SI.Scan(gpu, n).run(a);
    const got = await a.read(n);
    let acc = 0;
    for (let k = 0; k < n; k++) { if (got[k] !== acc) throw new Error(`at ${k}`); acc += data[k]; }
  });

  section(`reduce [${tag}]`);
  await test('reduce sum/min/max f32', async () => {
    const n = 1234567;
    const data = new Float32Array(n).map(() => R() * 2 - 1);
    const A = gpu.array('f32', data);
    let s = 0, mn = Infinity, mx = -Infinity;
    for (const v of data) { s += v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
    const sum = new SI.Reduce(gpu, { type: 'f32', op: 'sum', maxN: n });
    const min = new SI.Reduce(gpu, { type: 'f32', op: 'min', maxN: n });
    const max = new SI.Reduce(gpu, { type: 'f32', op: 'max', maxN: n });
    sum.run(A); min.run(A); max.run(A);
    assertClose(await sum.read(), s, 1e-3, 'sum');
    assert((await min.read()) === Math.fround(mn), 'min');
    assert((await max.read()) === Math.fround(mx), 'max');
  });
  await test('reduce with map: max speed of vec4 velocities', async () => {
    const n = 99999;
    const v = new Float32Array(n * 4).map(() => R() * 2 - 1);
    v[4 * 777] = 9; v[4 * 777 + 1] = 12; v[4 * 777 + 2] = 0;
    const V = gpu.array('vec4f', v);
    const r = new SI.Reduce(gpu, { input: 'vec4f', type: 'f32', op: 'max', map: 'length(x.xyz)', maxN: n });
    r.run(V);
    assertClose(await r.read(), 15, 1e-6);
  });

  section(`spatial grid [${tag}]`);
  await test('grid build: permutation + cell order + neighbour counts', async () => {
    const n = 50000, h = 0.05;
    const p = new Float32Array(n * 4);
    for (let k = 0; k < n; k++) { p[4 * k] = R(); p[4 * k + 1] = R() * 0.5; p[4 * k + 2] = R(); p[4 * k + 3] = 1; }
    const P = gpu.array('vec4f', p);
    const grid = new SI.SpatialGrid(gpu, { maxCount: n, min: [0, 0, 0], max: [1, 0.5, 1], cellSize: h });
    grid.build(P, n);
    const sorted = gpu.array('vec4f', n);
    grid.gather(P, sorted);
    // count neighbours within h for each particle using the 9-range iteration
    const counts = gpu.array('u32', n);
    const k = gpu.kernel({
      imports: ['grid'], uniforms: { ...grid.uniforms(), h2: 'f32' },
      buffers: { S: { type: 'vec4f', access: 'read' }, start: { type: 'u32', access: 'read' }, counts: { type: 'u32' } },
      code: `let pi = S[i].xyz; var c = 0u;
        ${SI.SpatialGrid.neighbors('pi', 'let d = S[j].xyz - pi; if (dot(d, d) < u.h2) { c++; }')}
        counts[i] = c;`,
    });
    await k.ready();
    k.run(n, { ...grid.values(), h2: h * h }, { S: sorted, start: grid.start, counts });
    const order = await grid.order.read(n);
    const seen = new Uint8Array(n);
    for (const o of order) { assert(o < n && !seen[o], 'order is not a permutation'); seen[o] = 1; }
    const start = await grid.start.read(grid.numCells + 1);
    assert(start[grid.numCells] === n, 'last start != n');
    const sp = await sorted.read(n);
    const cnt = await counts.read(n);
    // brute-force check on a subset
    for (let t = 0; t < 200; t++) {
      const a = (R() * n) | 0;
      let c = 0;
      for (let b = 0; b < n; b++) {
        const dx = sp[4 * b] - sp[4 * a], dy = sp[4 * b + 1] - sp[4 * a + 1], dz = sp[4 * b + 2] - sp[4 * a + 2];
        if (dx * dx + dy * dy + dz * dz < Math.fround(h * h)) c++;
      }
      if (c !== cnt[a]) throw new Error(`neighbour count at ${a}: gpu ${cnt[a]} cpu ${c}`);
    }
    // cells are non-decreasing in sorted order
    const inv = 1 / h, d = grid.dims;
    let last = -1;
    for (let k2 = 0; k2 < n; k2++) {
      const cx = Math.min(d[0] - 1, Math.max(0, Math.floor(sp[4 * k2] * inv)));
      const cy = Math.min(d[1] - 1, Math.max(0, Math.floor(sp[4 * k2 + 1] * inv)));
      const cz = Math.min(d[2] - 1, Math.max(0, Math.floor(sp[4 * k2 + 2] * inv)));
      const c = cx + d[0] * (cy + d[1] * cz);
      assert(c >= last, `cell order broken at ${k2}`);
      last = c;
    }
    note(`grid ${d.join('x')} = ${grid.numCells} cells`);
  });
}
