import * as SI from '../src/index.js';
import { test, section, note, assert, assertClose } from './harness.js';

/** Evaluate a WGSL expression of `x: vec4f` (random input) for n points; returns Float32Array of vec4 results. */
async function evalExpr(gpu, imports, expr, n, scale = 64, shared = '') {
  const out = gpu.array('vec4f', n);
  const k = gpu.kernel({
    imports: ['hash', ...imports], buffers: { out }, uniforms: { scale: 'f32' }, shared,
    code: `
      let h = pcg4d(vec4u(i, i * 7u + 1u, i * 13u + 2u, 99u));
      let x = (u2f4(h) - 0.5) * u.scale;
      out[i] = ${expr};`,
  });
  await k.ready();
  k.run(n, { scale });
  const r = await out.read();
  out.destroy();
  return r;
}

function stats(r, comp = 0) {
  let mn = Infinity, mx = -Infinity, s = 0, s2 = 0;
  const n = r.length / 4;
  for (let k = 0; k < n; k++) { const v = r[4 * k + comp]; mn = Math.min(mn, v); mx = Math.max(mx, v); s += v; s2 += v * v; }
  const mean = s / n;
  return { min: mn, max: mx, mean, std: Math.sqrt(s2 / n - mean * mean) };
}

export async function shaderTests(gpu) {
  section('shader library: compile every module');
  for (const name of SI.moduleNames()) {
    await test(`module ${name} compiles`, async () => {
      const shared = name === 'raymarch' ? 'fn map(p: vec3f) -> vec2f { return vec2f(length(p) - 1.0, 0.0); }' : '';
      const k = gpu.kernel({ name: `compile-${name}`, imports: [name], shared, buffers: { o: { type: 'f32' } }, code: 'o[i] = 0.0;' });
      await k.ready();
    });
  }

  section('noise: range, mean, calibrated amplitude');
  const N = 1 << 20;
  const cases = [
    ['simplex2', 'vec4f(simplex2(x.xy), 0.0, 0.0, 0.0)'],
    ['simplex3', 'vec4f(simplex3(x.xyz), 0.0, 0.0, 0.0)'],
    ['simplex4', 'vec4f(simplex4(x), 0.0, 0.0, 0.0)'],
    ['perlin3', 'vec4f(perlin3(x.xyz), 0.0, 0.0, 0.0)'],
    ['value3', 'vec4f(value3(x.xyz), 0.0, 0.0, 0.0)'],
    ['worley3 F1', 'vec4f(worley3(x.xyz).x, 0.0, 0.0, 0.0)'],
  ];
  for (const [name, expr] of cases) {
    await test(`${name} statistics`, async () => {
      const r = await evalExpr(gpu, ['noise'], expr, N);
      const s = stats(r);
      note(`${name}: min ${s.min.toFixed(3)} max ${s.max.toFixed(3)} mean ${s.mean.toFixed(4)} std ${s.std.toFixed(3)}`);
      if (!name.startsWith('worley')) {
        assert(s.min >= -1.15 && s.max <= 1.15, 'range outside [-1.15, 1.15]');
        assert(Math.abs(s.mean) < 0.02, 'mean not ~0');
        assert(s.std > 0.12, 'too little variance');
      } else assert(s.min >= 0 && s.max < 1.8, 'worley range');
    });
  }

  section('noise: analytic gradients match finite differences');
  for (const [name, fn] of [['simplex3d', 'simplex3d'], ['perlin3d', 'perlin3d'], ['fbm3d', 'fbm3d']]) {
    await test(`${name} gradient`, async () => {
      const call = (a) => fn === 'fbm3d' ? `fbm3d(${a}, 4)` : `${fn}(${a})`;
      const shared = `fn F(p: vec3f) -> vec4f { return ${call('p')}; }
        fn fd(p: vec3f) -> vec3f {
          let e = 1e-3;
          return vec3f(F(p + vec3f(e, 0, 0)).x - F(p - vec3f(e, 0, 0)).x,
                       F(p + vec3f(0, e, 0)).x - F(p - vec3f(0, e, 0)).x,
                       F(p + vec3f(0, 0, e)).x - F(p - vec3f(0, 0, e)).x) / (2.0 * e);
        }`;
      const r = await evalExpr(gpu, ['noise'], 'vec4f(length(F(x.xyz * 0.25).yzw - fd(x.xyz * 0.25)), length(F(x.xyz * 0.25).yzw), 0.0, 0.0)', 1 << 16, 64, shared);
      const err = stats(r, 0), mag = stats(r, 1);
      note(`${name}: max |g - fd| = ${err.max.toExponential(2)}, mean |g| = ${mag.mean.toFixed(3)}`);
      assert(err.mean < 0.01 * mag.mean && err.max < 0.05 * mag.max, 'gradient mismatch');
    });
  }
  await test('simplex2d gradient', async () => {
    const shared = `fn fd(p: vec2f) -> vec2f { let e = 1e-3;
      return vec2f(simplex2d(p + vec2f(e, 0)).x - simplex2d(p - vec2f(e, 0)).x, simplex2d(p + vec2f(0, e)).x - simplex2d(p - vec2f(0, e)).x) / (2.0 * e); }`;
    const r = await evalExpr(gpu, ['noise'], 'vec4f(length(simplex2d(x.xy * 0.25).yz - fd(x.xy * 0.25)), length(simplex2d(x.xy * 0.25).yz), 0.0, 0.0)', 1 << 16, 64, shared);
    const err = stats(r, 0), mag = stats(r, 1);
    assert(err.mean < 0.01 * mag.mean, `gradient mismatch ${err.mean} vs ${mag.mean}`);
  });
  await test('curl3 is divergence-free', async () => {
    const shared = `fn div(p: vec3f) -> f32 { let e = 2e-3;
      return (curl3(p + vec3f(e, 0, 0)).x - curl3(p - vec3f(e, 0, 0)).x + curl3(p + vec3f(0, e, 0)).y - curl3(p - vec3f(0, e, 0)).y
            + curl3(p + vec3f(0, 0, e)).z - curl3(p - vec3f(0, 0, e)).z) / (2.0 * e); }
      fn jac(p: vec3f) -> f32 { let e = 2e-3; return length(curl3(p + vec3f(e, 0, 0)) - curl3(p - vec3f(e, 0, 0))) / (2.0 * e); }`;
    const r = await evalExpr(gpu, ['noise'], 'vec4f(abs(div(x.xyz * 0.25)), jac(x.xyz * 0.25), length(curl3(x.xyz * 0.25)), 0.0)', 1 << 16, 64, shared);
    const d = stats(r, 0), j = stats(r, 1), m = stats(r, 2);
    note(`|div| mean ${d.mean.toExponential(2)} vs |∂v/∂x| mean ${j.mean.toFixed(3)}; |v| mean ${m.mean.toFixed(3)}`);
    assert(d.mean < 0.01 * j.mean, 'curl noise has divergence');
  });

  section('hash / color / sdf');
  await test('pcg / lowbias32 bucket uniformity (chi-square)', async () => {
    const r = await evalExpr(gpu, ['hash'], 'vec4f(u2f(pcg(i)), u2f(lowbias32(i)), u2f(pcg3d(vec3u(i, 0u, 0u)).x), 0.0)', 1 << 20);
    for (const comp of [0, 1, 2]) {
      const B = 256, cnt = new Float64Array(B);
      for (let k = 0; k < r.length / 4; k++) cnt[Math.min(B - 1, Math.floor(r[4 * k + comp] * B))]++;
      const e = (r.length / 4) / B;
      let chi = 0; for (const c of cnt) chi += (c - e) * (c - e) / e;
      // 255 dof: mean 255, sd ~22.6; accept < 255 + 5 sd
      assert(chi < 370, `chi-square ${chi.toFixed(1)} for component ${comp}`);
    }
  });
  await test('2D hashes: adjacent pixels are independent (chi-square of 16×16 neighbour-pair buckets)', async () => {
    // pcg2d(p).x fails this (χ² ≈ 1073); hash21 / hash22 / ihash2 must not.
    const W = 1024;
    const r = await evalExpr(gpu, ['hash'], `vec4f(hash21(bitcast<vec2f>(vec2u(i % ${W}u, i / ${W}u))), hash22(bitcast<vec2f>(vec2u(i % ${W}u, i / ${W}u))), u2f(ihash2(vec2i(i32(i % ${W}u), i32(i / ${W}u)))))`, 1 << 20);
    const notes = [];
    for (const comp of [0, 1, 2, 3]) {
      const B = 256, cnt = new Float64Array(B);
      let pairs = 0;
      for (let k = 0; k + 1 < r.length / 4; k += 2) { cnt[Math.min(15, Math.floor(r[4 * k + comp] * 16)) * 16 + Math.min(15, Math.floor(r[4 * (k + 1) + comp] * 16))]++; pairs++; }
      const e = pairs / B;
      let chi = 0; for (const c of cnt) chi += (c - e) * (c - e) / e;
      notes.push(chi.toFixed(0));
      assert(chi < 370, `neighbour-pair chi-square ${chi.toFixed(1)} for component ${comp}`);
    }
    note(`χ² of adjacent pairs: hash21 ${notes[0]}, hash22 ${notes[1]} / ${notes[2]}, ihash2 ${notes[3]} (≲ 330 independent)`);
  });
  await test('OKLab round trip', async () => {
    const r = await evalExpr(gpu, ['color'], 'vec4f(oklabToLinear(linearToOklab(abs(x.xyz) / 32.0)) - abs(x.xyz) / 32.0, 0.0)', 1 << 14);
    let mx = 0; for (let k = 0; k < r.length; k++) if (k % 4 !== 3) mx = Math.max(mx, Math.abs(r[k]));
    assert(mx < 2e-4, `round trip error ${mx}`);
  });
  await test('OKLab reference value: sRGB white -> L=1', async () => {
    const r = await evalExpr(gpu, ['color'], 'vec4f(linearToOklab(vec3f(1.0)), 0.0)', 4);
    assertClose(r[0], 1.0, 1e-3); assertClose(r[1], 0, 1e-3); assertClose(r[2], 0, 1e-3);
  });
  await test('sdBox / sdSphere / sdTorus exact distances', async () => {
    const r = await evalExpr(gpu, ['sdf'], 'vec4f(sdBox(vec3f(3.0, 0.5, 0.0), vec3f(1.0)), sdBox(vec3f(2.0, 2.0, 2.0), vec3f(1.0)), sdSphere(vec3f(0.0, 0.0, 0.5), 1.0), sdTorus(vec3f(2.0, 0.0, 0.0), vec2f(1.5, 0.25)))', 4);
    assertClose(r[0], 2, 1e-6); assertClose(r[1], Math.sqrt(3), 1e-6); assertClose(r[2], -0.5, 1e-6); assertClose(r[3], 0.25, 1e-6);
  });
  await test('smin: equals min far apart, blends near', async () => {
    const r = await evalExpr(gpu, ['sdf'], 'vec4f(smin(1.0, 3.0, 0.1), smin(1.0, 1.0, 0.1), sminCubic(1.0, 1.0, 0.1), 0.0)', 4);
    assertClose(r[0], 1, 1e-6); assertClose(r[1], 0.9, 1e-5); assertClose(r[2], 0.9, 1e-5);
  });
  await test('tonemappers are monotone and bounded', async () => {
    const r = await evalExpr(gpu, ['tonemap'], 'vec4f(tonemapACES(vec3f(abs(x.x))).x, tonemapAgX(vec3f(abs(x.x))).x, tonemapNeutral(vec3f(abs(x.x))).x, abs(x.x))', 4096, 32);
    const pts = [];
    for (let k = 0; k < r.length / 4; k++) pts.push([r[4 * k + 3], r[4 * k], r[4 * k + 1], r[4 * k + 2]]);
    pts.sort((a, b) => a[0] - b[0]);
    for (let c = 1; c <= 3; c++) {
      for (let k = 1; k < pts.length; k++) assert(pts[k][c] >= pts[k - 1][c] - 1e-4, `tonemap ${c} not monotone`);
      assert(pts[pts.length - 1][c] <= 1.0001 && pts[0][c] >= -1e-4, `tonemap ${c} out of range`);
    }
  });
}
