// Baseline implementations of the techniques most commonly found in WebGPU /
// WebGL code today, re-implemented faithfully so they run on the same device
// and inputs as ShaderIntel's kernels. Each notes where the technique comes from.
import { defineModule } from '../src/index.js';

// ---------------------------------------------------------------------------
// Ashima Arts / Stefan Gustavson "webgl-noise" 3D simplex noise (MIT licence),
// the most widely copied GPU noise (three.js examples, Shadertoy, countless
// tutorials). Ported 1:1 from GLSL (classic 0.6 / 42.0 kernel).
defineModule('baseline-ashima', {
  code: /* wgsl */`
fn a_mod289_3(x: vec3f) -> vec3f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn a_mod289_4(x: vec4f) -> vec4f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn a_permute(x: vec4f) -> vec4f { return a_mod289_4(((x * 34.0) + 1.0) * x); }
fn a_taylorInvSqrt(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }
fn ashimaSimplex3(v: vec3f) -> f32 {
  let C = vec2f(1.0 / 6.0, 1.0 / 3.0);
  let D = vec4f(0.0, 0.5, 1.0, 2.0);
  var i = floor(v + dot(v, C.yyy));
  let x0 = v - i + dot(i, C.xxx);
  let g = step(x0.yzx, x0.xyz);
  let l = 1.0 - g;
  let i1 = min(g.xyz, l.zxy);
  let i2 = max(g.xyz, l.zxy);
  let x1 = x0 - i1 + C.xxx;
  let x2 = x0 - i2 + C.yyy;
  let x3 = x0 - D.yyy;
  i = a_mod289_3(i);
  let p = a_permute(a_permute(a_permute(
            i.z + vec4f(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4f(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4f(0.0, i1.x, i2.x, 1.0));
  let n_ = 0.142857142857;
  let ns = n_ * D.wyz - D.xzx;
  let j = p - 49.0 * floor(p * ns.z * ns.z);
  let x_ = floor(j * ns.z);
  let y_ = floor(j - 7.0 * x_);
  let x = x_ * ns.x + ns.yyyy;
  let y = y_ * ns.x + ns.yyyy;
  let h = 1.0 - abs(x) - abs(y);
  let b0 = vec4f(x.xy, y.xy);
  let b1 = vec4f(x.zw, y.zw);
  let s0 = floor(b0) * 2.0 + 1.0;
  let s1 = floor(b1) * 2.0 + 1.0;
  let sh = -step(h, vec4f(0.0));
  let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  let a1 = b1.xzyw + s1.xzyw * sh.zzww;
  var p0 = vec3f(a0.xy, h.x);
  var p1 = vec3f(a0.zw, h.y);
  var p2 = vec3f(a1.xy, h.z);
  var p3 = vec3f(a1.zw, h.w);
  let norm = a_taylorInvSqrt(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  var m = max(0.6 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  m = m * m;
  return 42.0 * dot(m * m, vec4f(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
fn ashimaFbm3(p: vec3f) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < 4; k++) { s += a * ashimaSimplex3(q); q = q * 2.02; a *= 0.5; }
  return s;
}`,
});

// psrdnoise3 (Gustavson & McEwan, "Tiling simplex noise and flow noise in two
// and three dimensions", JCGT 2022; MIT licence) — the newest reference GPU
// simplex noise, ported to WGSL in its fastest configuration (no tiling,
// no rotation: period = 0, alpha = 0 folded as constants).
defineModule('baseline-psrd', {
  code: /* wgsl */`
fn p_mod289(x: vec4f) -> vec4f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn p_permute(x: vec4f) -> vec4f { let xm = p_mod289(x); return p_mod289(((xm * 34.0) + 10.0) * xm); }
/** Returns (value, gradient). */
fn psrdnoise3(x: vec3f) -> vec4f {
  let uvw = vec3f(x.y + x.z, x.x + x.z, x.x + x.y);           // M * x
  let i0 = floor(uvw);
  let f0 = fract(uvw);
  let g_ = step(f0.xyx, f0.yzz);
  let l_ = 1.0 - g_;
  let g = vec3f(l_.z, g_.xy);
  let l = vec3f(l_.xy, g_.z);
  let o1 = min(g, l);
  let o2 = max(g, l);
  let i1 = i0 + o1;
  let i2 = i0 + o2;
  let i3 = i0 + vec3f(1.0);
  let mi = mat3x3f(-0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5);
  let v0 = mi * i0; let v1 = mi * i1; let v2 = mi * i2; let v3 = mi * i3;
  let x0 = x - v0; let x1 = x - v1; let x2 = x - v2; let x3 = x - v3;
  let hash = p_permute(p_permute(p_permute(
      vec4f(i0.z, i1.z, i2.z, i3.z))
    + vec4f(i0.y, i1.y, i2.y, i3.y))
    + vec4f(i0.x, i1.x, i2.x, i3.x));
  let theta = hash * 3.883222077;
  let sz = hash * -0.006920415 + 0.996539792;
  let Ct = cos(theta);
  let St = sin(theta);
  let szp = sqrt(1.0 - sz * sz);
  let gx = Ct * szp; let gy = St * szp; let gz = sz;
  let g0 = vec3f(gx.x, gy.x, gz.x); let g1 = vec3f(gx.y, gy.y, gz.y);
  let g2 = vec3f(gx.z, gy.z, gz.z); let g3 = vec3f(gx.w, gy.w, gz.w);
  var w = 0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3));
  w = max(w, vec4f(0.0));
  let w2 = w * w;
  let w3 = w2 * w;
  let gdotx = vec4f(dot(g0, x0), dot(g1, x1), dot(g2, x2), dot(g3, x3));
  let n = dot(w3, gdotx);
  let dw = -6.0 * w2 * gdotx;
  let dn = w3.x * g0 + dw.x * x0 + w3.y * g1 + dw.y * x1 + w3.z * g2 + dw.z * x2 + w3.w * g3 + dw.w * x3;
  return 39.5 * vec4f(n, dn);
}
fn psrdnoise3v(x: vec3f) -> f32 { return psrdnoise3(x).x; }`,
});

// The classic one-liner hash from "On generating random numbers..." (Rey 1998),
// still the default in a large share of shaders.
defineModule('baseline-sinhash', {
  code: /* wgsl */`
fn sinHash21(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453); }`,
});

/** Global-memory Hillis–Steele scan: log2(n) full passes (common in tutorials / naive ports). */
export class HillisSteeleScan {
  constructor(ctx, n) {
    this.ctx = ctx;
    this.tmp = ctx.array('u32', n);
    this.k = ctx.kernel({
      name: 'hs-pass', buffers: { src: { type: 'u32', access: 'read' }, dst: { type: 'u32' } }, uniforms: { off: 'u32' },
      code: 'dst[i] = src[i] + select(0u, src[i - u.off], i >= u.off);',
    });
    this.shift = ctx.kernel({
      name: 'hs-shift', buffers: { src: { type: 'u32', access: 'read' }, dst: { type: 'u32' } },
      code: 'dst[i] = select(0u, src[i - 1u], i > 0u);',
    });
  }
  /** Exclusive scan src -> dst (src is preserved; dst/tmp ping-pong, then a shift pass). */
  run(src, dst, n) {
    const other = (x) => (x === dst ? this.tmp : dst);
    let a = src, b = dst;
    for (let off = 1; off < n; off *= 2) { this.k.run(n, { off }, { src: a, dst: b }); a = b; b = other(b); }
    const target = other(a);
    this.shift.run(n, null, { src: a, dst: target });
    if (target !== dst) this.ctx.copy(target, dst, n * 4);
  }
}

/**
 * Work-efficient Blelloch scan in shared memory, 2 elements per thread, block
 * sums scanned recursively, then a uniform add — "Parallel Prefix Sum (Scan)
 * with CUDA", GPU Gems 3 ch. 39 (Harris, Sengupta, Owens), the reference most
 * WebGPU scans are ported from.
 */
export class BlellochScan {
  constructor(ctx, n) {
    this.ctx = ctx;
    this.B = 512; // elements per workgroup
    this.levels = [];
    let m = n;
    while (m > this.B) { m = Math.ceil(m / this.B); this.levels.push(ctx.array('u32', Math.max(4, m))); }
    this.dummy = ctx.array('u32', 4);
    this.scanK = ctx.kernel({
      name: 'blelloch', workgroupSize: 256, guard: false,
      buffers: { data: { type: 'u32' }, sums: { type: 'u32' } }, uniforms: { writeSums: 'u32' },
      shared: 'var<workgroup> t: array<u32, 512>;',
      code: /* wgsl */`
        let base = wgi * 512u;
        let ai = lidx; let bi = lidx + 256u;
        t[ai] = select(0u, data[base + ai], base + ai < u.n);
        t[bi] = select(0u, data[base + bi], base + bi < u.n);
        var off = 1u;
        for (var d = 256u; d > 0u; d = d >> 1u) {
          workgroupBarrier();
          if (lidx < d) { let a = off * (2u * lidx + 1u) - 1u; let b = off * (2u * lidx + 2u) - 1u; t[b] += t[a]; }
          off = off << 1u;
        }
        workgroupBarrier();
        if (lidx == 0u) { if (u.writeSums != 0u) { sums[wgi] = t[511]; } t[511] = 0u; }
        for (var d = 1u; d < 512u; d = d << 1u) {
          off = off >> 1u;
          workgroupBarrier();
          if (lidx < d) { let a = off * (2u * lidx + 1u) - 1u; let b = off * (2u * lidx + 2u) - 1u; let x = t[a]; t[a] = t[b]; t[b] += x; }
        }
        workgroupBarrier();
        if (base + ai < u.n) { data[base + ai] = t[ai]; }
        if (base + bi < u.n) { data[base + bi] = t[bi]; }`,
    });
    this.addK = ctx.kernel({
      name: 'blelloch-add', buffers: { data: { type: 'u32' }, sums: { type: 'u32', access: 'read' } },
      code: 'data[i] += sums[i / 512u];',
    });
  }
  /** In-place exclusive scan. */
  run(data, n) { this._scan(data, n, 0); }
  _scan(data, n, level) {
    const groups = Math.ceil(n / this.B);
    const sums = groups > 1 ? this.levels[level] : this.dummy;
    this.scanK.dispatch(groups, { n, writeSums: groups > 1 ? 1 : 0 }, { data, sums });
    if (groups > 1) {
      this._scan(sums, groups, level + 1);
      this.addK.run(n, null, { data, sums });
    }
  }
}

/**
 * Matrix multiply, one thread per output reading a row of A and a column of B straight from
 * global memory — the first kernel in every GPU-compute tutorial (and WebGPU samples' "naive" GEMM).
 */
export class NaiveGemm {
  constructor(ctx) {
    this.k = ctx.kernel({
      name: 'gemm-naive', dims: 2, workgroupSize: [16, 16, 1],
      buffers: { A: { type: 'f32', access: 'read' }, B: { type: 'f32', access: 'read' }, C: { type: 'f32' } }, uniforms: { N: 'u32' },
      code: 'var s = 0.0; for (var k = 0u; k < u.N; k++) { s = fma(A[p.y * u.N + k], B[k * u.N + p.x], s); } C[p.y * u.N + p.x] = s;',
    });
  }
  ready() { return this.k.ready(); }
  /** C = A·B for n×n row-major matrices. */
  run(A, B, C, n) { this.k.run([n, n], { N: n }, { A, B, C }); }
}

/**
 * The classic shared-memory tiled multiply: 16×16 tiles of A and B staged in workgroup memory,
 * one output per thread (CUDA C Programming Guide §"Shared Memory"; the tiled GEMM of most WebGPU
 * tutorials and ports). n must be a multiple of 16.
 */
export class TiledGemm {
  constructor(ctx) {
    this.k = ctx.kernel({
      name: 'gemm-tiled16', dims: 2, workgroupSize: [16, 16, 1], guard: false,
      buffers: { A: { type: 'f32', access: 'read' }, B: { type: 'f32', access: 'read' }, C: { type: 'f32' } }, uniforms: { N: 'u32' },
      shared: 'var<workgroup> As: array<f32, 256>; var<workgroup> Bs: array<f32, 256>;',
      code: /* wgsl */`
        let r = gid.y; let c = gid.x; var s = 0.0;
        for (var t = 0u; t < u.N; t += 16u) {
          As[lid.y * 16u + lid.x] = A[r * u.N + t + lid.x];
          Bs[lid.y * 16u + lid.x] = B[(t + lid.y) * u.N + c];
          workgroupBarrier();
          for (var k = 0u; k < 16u; k++) { s = fma(As[lid.y * 16u + k], Bs[k * 16u + lid.x], s); }
          workgroupBarrier();
        }
        C[r * u.N + c] = s;`,
    });
  }
  ready() { return this.k.ready(); }
  /** C = A·B for n×n row-major matrices. */
  run(A, B, C, n) { this.k.dispatch([n / 16, n / 16, 1], { N: n }, { A, B, C }); }
}
