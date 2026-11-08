// MLS-MPM (Hu et al. 2018) with APIC transfers, multi-material, in lattice units.
//
// First principles that shape this solver:
//  1. Lattice units (dx = 1 cell, dt = 1 substep). Every quantity a stable
//     simulation produces is O(1): velocities < 1 cell/step (CFL), wave speeds
//     c' = c·dt/dx < ~0.5. That bounds momenta, so 32-bit fixed point (2^22)
//     is exact enough for atomic P2G — WebGPU has no float atomics.
//  2. Stiffness is chosen in lattice units (fraction of the stable wave speed),
//     so a material can never be configured to explode; more substeps simply
//     buy stiffer (more incompressible) materials.
//  3. P2G is the bottleneck (27 nodes x 4 channels of atomics per particle).
//     Particles are counting-sorted into 4^3-cell blocks (Morton order) every
//     few substeps; each workgroup then accumulates its particles into a
//     shared-memory tile with workgroup atomics and flushes only non-empty
//     nodes to global memory — ~10-30x fewer global atomics.
//  4. Stress is evaluated once per substep at the end of G2P and stored as
//     the Kirchhoff stress τ = P Fᵀ in the spare .w lanes of the particle, so
//     P2G is material-agnostic and branch-free.
import { defineModule } from '../core/wgsl.js';
import { Scan } from '../prims/scan.js';
import { Spheres } from '../render/spheres.js';
import { FluidSurface } from '../render/fluid.js';
import { FluidRigidCoupling } from './coupling.js';
import '../shaders/linalg.js';

export const FLUID = 0, JELLY = 1, SNOW = 2, SAND = 3;

/** Material presets. stiffness ∈ (0,1] is a fraction of the max stable lattice wave speed. */
export const MATERIALS = {
  water:   { kind: FLUID, density: 1, stiffness: 1.0, viscosity: 0.0, color: [0.12, 0.42, 0.95] },
  oil:     { kind: FLUID, density: 0.9, stiffness: 0.9, viscosity: 0.3, color: [0.9, 0.62, 0.12] },
  honey:   { kind: FLUID, density: 1.4, stiffness: 0.9, viscosity: 1.0, color: [0.95, 0.6, 0.05] },
  jelly:   { kind: JELLY, density: 1, stiffness: 0.35, poisson: 0.3, color: [0.95, 0.3, 0.35] },
  rubber:  { kind: JELLY, density: 1.2, stiffness: 0.7, poisson: 0.45, color: [0.25, 0.25, 0.28] },
  snow:    { kind: SNOW, density: 0.4, stiffness: 0.6, poisson: 0.2, hardening: 10, compression: 0.025, stretch: 0.0075, color: [0.95, 0.96, 1.0] },
  sand:    { kind: SAND, density: 1.6, stiffness: 0.6, poisson: 0.3, friction: 33, color: [0.86, 0.68, 0.42] },
};

const C_MAX = 0.4;          // max lattice wave speed (cells/step)
const PARTICLE_FLOATS = 32; // 8 x vec4f

defineModule('mpm', {
  deps: ['linalg'],
  code: /* wgsl */`
struct Particle {
  x: vec4f,   // position (lattice units), w = material index (u32 bits)
  v: vec4f,   // velocity (cells / substep), w = J (fluids) or Jp (snow)
  c0: vec4f,  // APIC C columns; w = tau00, tau11, tau22
  c1: vec4f,
  c2: vec4f,
  f0: vec4f,  // deformation gradient F columns; w = tau01, tau02, tau12
  f1: vec4f,
  f2: vec4f,
}
const FP = 4194304.0;            // 2^22 fixed-point scale for atomics
const INV_FP = 1.0 / 4194304.0;
fn toFixed(x: f32) -> i32 { return i32(round(clamp(x * FP, -2.0e9, 2.0e9))); }
fn bspline(fx: vec3f) -> array<vec3f, 3> {
  return array<vec3f, 3>(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}
fn si_part1by2(x: u32) -> u32 {
  var v = x & 0x3ffu;
  v = (v | (v << 16u)) & 0x030000ffu;
  v = (v | (v << 8u)) & 0x0300f00fu;
  v = (v | (v << 4u)) & 0x030c30c3u;
  v = (v | (v << 2u)) & 0x09249249u;
  return v;
}
fn morton3(c: vec3u) -> u32 { return si_part1by2(c.x) | (si_part1by2(c.y) << 1u) | (si_part1by2(c.z) << 2u); }
`,
});

export class MPM {
  /**
   * @param {Context} ctx
   * @param {object} o
   *   maxParticles, grid: cells along the longest axis (default 64), size: [x,y,z] world size (default [1,1,1]),
   *   origin, gravity (m/s^2), substeps (per 1/60 s), timeScale, sortEvery, ppc (particles per cell axis^3, default 8)
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.maxParticles = o.maxParticles || 262144;
    const size = Array.isArray(o.size) ? o.size : [o.size || 1, o.size || 1, o.size || 1];
    const res = o.grid || 64;
    this.dx = Math.max(...size) / res;
    this.dims = size.map((s) => Math.max(8, Math.round(s / this.dx)));
    this.size = this.dims.map((d) => d * this.dx);
    this.origin = o.origin || [0, 0, 0];
    this.gravity = o.gravity || [0, -9.81, 0];
    this.substeps = o.substeps || 8;
    this.timeScale = o.timeScale ?? 1;
    this.sortEvery = o.sortEvery || 4;
    this.ppcAxis = o.ppcAxis || 2;
    this.bound = 3;
    this.count = 0;
    this.stepCount = 0;
    this.materials = [];
    this.materialIndex = new Map();
    this.colliders = [];
    this.dt = 1 / 60 * this.timeScale / this.substeps;

    const nodes = this.dims[0] * this.dims[1] * this.dims[2];
    this.numNodes = nodes;
    this.P = ctx.array('vec4f', this.maxParticles * 8, { label: 'mpm-particles', usage: GPUBufferUsage.VERTEX });
    this.P2 = ctx.array('vec4f', this.maxParticles * 8, { label: 'mpm-particles-2', usage: GPUBufferUsage.VERTEX });
    this.grid = ctx.array('vec4i', nodes, { label: 'mpm-grid' });
    this.matBuf = ctx.array('vec4f', 8 * 3, { label: 'mpm-materials' });
    this.colBuf = ctx.array('vec4f', 16 * 3, { label: 'mpm-colliders' });
    // block sort (Morton order of 4^3-cell blocks)
    const bmax = Math.max(...this.dims.map((d) => Math.ceil(d / 4)));
    const bpow = 1 << Math.ceil(Math.log2(bmax));
    this.numBlocks = bpow ** 3;
    this.blockCount = ctx.array('u32', this.numBlocks + 1, { label: 'mpm-block-count' });
    this.key = ctx.array('u32', this.maxParticles, { label: 'mpm-key' });
    this.rank = ctx.array('u32', this.maxParticles, { label: 'mpm-rank' });
    this.scan = new Scan(ctx, this.numBlocks + 1);
    this._buildKernels();
  }

  /** World-space time step of one substep. */
  get substepDt() { return (1 / 60) * this.timeScale / this.substeps; }

  // ------------------------------------------------------------------ materials
  _material(nameOrDef) {
    const def = typeof nameOrDef === 'string' ? { ...MATERIALS[nameOrDef], name: nameOrDef } : { ...nameOrDef };
    if (!def || def.kind === undefined) throw new Error(`MPM: unknown material ${nameOrDef}`);
    const key = JSON.stringify(def);
    if (this.materialIndex.has(key)) return this.materialIndex.get(key);
    if (this.materials.length >= 8) throw new Error('MPM: at most 8 distinct materials');
    const idx = this.materials.length;
    this.materials.push(def);
    this.materialIndex.set(key, idx);
    this._uploadMaterials();
    return idx;
  }

  _uploadMaterials() {
    const vol = 1 / this.ppcAxis ** 3;
    const data = new Float32Array(8 * 12);
    const u = new Uint32Array(data.buffer);
    this.materials.forEach((m, k) => {
      const mass = (m.density ?? 1) * vol;          // lattice mass: rho' = density (relative) per cell
      const rho = mass / vol;
      const c = C_MAX * Math.min(1, Math.max(0.01, m.stiffness ?? 1));
      const nu = m.poisson ?? 0.3;
      // P-wave modulus M = lambda + 2 mu = rho c^2; recover E, mu, lambda from Poisson's ratio.
      const Mmod = rho * c * c;
      const E = Mmod * (1 + nu) * (1 - 2 * nu) / (1 - nu);
      const mu = E / (2 * (1 + nu));
      const la = E * nu / ((1 + nu) * (1 - 2 * nu));
      const o = k * 12;
      u[o] = m.kind;
      data[o + 1] = mass;
      data[o + 2] = m.kind === FLUID ? rho * c * c : mu;   // fluids: bulk modulus K
      data[o + 3] = la;
      data[o + 4] = (m.viscosity ?? 0) * 0.05 * rho;       // lattice dynamic viscosity
      data[o + 5] = m.hardening ?? 10;
      data[o + 6] = m.compression ?? 0.025;
      data[o + 7] = m.stretch ?? 0.0075;
      const phi = ((m.friction ?? 33) * Math.PI) / 180;
      const sinp = Math.sin(phi);
      data[o + 8] = Math.sqrt(2 / 3) * (2 * sinp) / (3 - sinp); // Drucker–Prager alpha
    });
    this.matBuf.write(data);
  }

  // ------------------------------------------------------------------ emission
  /**
   * Add particles filling a shape with `ppcAxis^3` particles per cell (jittered).
   * shape: { box: [[x0,y0,z0],[x1,y1,z1]] } | { sphere: [[cx,cy,cz], r] } | { points: Float32Array (xyz world) }
   * opts: { material: 'water' | def, velocity: [vx,vy,vz] (m/s), jitter: 0..1 }
   */
  add(shape, opts = {}) {
    const mat = this._material(opts.material || shape.material || 'water');
    const vel = opts.velocity || shape.velocity || [0, 0, 0];
    const pts = shape.points || sampleShape(shape, this.dx / this.ppcAxis, opts.jitter ?? 0.6, opts.seed ?? this.count + 1);
    const n = Math.min(pts.length / 3, this.maxParticles - this.count);
    if (n <= 0) return 0;
    const data = new Float32Array(n * PARTICLE_FLOATS);
    const u = new Uint32Array(data.buffer);
    const s = this.substepDt / this.dx; // m/s -> cells/step
    const lo = this.bound, hi = this.dims.map((d) => d - this.bound);
    for (let k = 0; k < n; k++) {
      const o = k * PARTICLE_FLOATS;
      for (let a = 0; a < 3; a++) {
        const x = (pts[3 * k + a] - this.origin[a]) / this.dx;
        data[o + a] = Math.min(hi[a] - 1e-3, Math.max(lo + 1e-3, x));
      }
      u[o + 3] = mat;
      data[o + 4] = vel[0] * s; data[o + 5] = vel[1] * s; data[o + 6] = vel[2] * s;
      data[o + 7] = 1; // J
      data[o + 20] = 1; data[o + 25] = 1; data[o + 30] = 1; // F = I
    }
    this.ctx.flush();
    this.P.write(data, this.count * 8);
    this.count += n;
    return n;
  }

  /** Remove every particle. */
  clear() { this.count = 0; }

  /**
   * Two-way coupling with a RigidWorld: bodies become moving boundaries of this material and
   * receive the momentum it exchanges with them (buoyancy, drag, splashes). Step the MPM first,
   * then the rigid world, each frame. o: { friction, band, mask, densityScale } (see coupling.js).
   */
  couple(world, o = {}) {
    this.coupling = new FluidRigidCoupling(this.ctx, this, world, o);
    return this.coupling;
  }

  // ------------------------------------------------------------------ colliders
  /**
   * Kinematic colliders, re-uploaded each step. Each: { type: 'sphere'|'box'|'plane',
   * center (world), radius | half: [hx,hy,hz] | normal, velocity (m/s), friction }.
   */
  setColliders(list) { this.colliders = list; }

  _uploadColliders() {
    const data = new Float32Array(16 * 12);
    const u = new Uint32Array(data.buffer);
    const s = this.substepDt / this.dx;
    this.colliders.slice(0, 16).forEach((c, k) => {
      const o = k * 12;
      u[o] = { sphere: 0, box: 1, plane: 2 }[c.type || 'sphere'];
      const ctr = c.center || [0, 0, 0];
      for (let a = 0; a < 3; a++) data[o + 1 + a] = (ctr[a] - this.origin[a]) / this.dx;
      if (c.type === 'plane') { const nrm = c.normal || [0, 1, 0]; data[o + 4] = nrm[0]; data[o + 5] = nrm[1]; data[o + 6] = nrm[2]; }
      else if (c.type === 'box') { const h = c.half || [0.1, 0.1, 0.1]; data[o + 4] = h[0] / this.dx; data[o + 5] = h[1] / this.dx; data[o + 6] = h[2] / this.dx; }
      else data[o + 4] = (c.radius ?? 0.1) / this.dx;
      data[o + 7] = c.friction ?? 0.3;
      const v = c.velocity || [0, 0, 0];
      data[o + 8] = v[0] * s; data[o + 9] = v[1] * s; data[o + 10] = v[2] * s;
    });
    this.colBuf.write(data);
  }

  // ------------------------------------------------------------------ kernels
  _buildKernels() {
    const ctx = this.ctx;
    const sg = ctx.caps.subgroups;
    const WG = 256, PPT = 2;
    this.p2gPerGroup = WG * PPT;
    const shmem = ctx.caps.workgroupStorage;
    this.tileCap = Math.max(343, Math.floor((shmem - 512) / 16));
    const common = { imports: ['mpm'], defines: { DIMS: `vec3u(${this.dims.join('u, ')}u)` } };

    // --- P2G: shared-memory tiled scatter with workgroup atomics
    this.p2gK = ctx.kernel({
      ...common, name: 'mpm-p2g', workgroupSize: WG, guard: false, subgroups: sg,
      buffers: { P: { type: 'Particle', access: 'read' }, grid: { type: 'i32', atomic: true }, MATS: { type: 'vec4f', access: 'read' } },
      uniforms: { vol4: 'f32', forceGlobal: 'u32' },
      shared: /* wgsl */`
        const TILE_CAP = ${this.tileCap}u;
        const PPT = ${PPT}u;
        var<workgroup> tile: array<atomic<i32>, ${this.tileCap * 4}>;
        var<workgroup> bmin: array<atomic<i32>, 3>;
        var<workgroup> bmax: array<atomic<i32>, 3>;
        var<workgroup> wbox: array<i32, 6>;
        fn nodeIndex(g: vec3i) -> u32 { return u32(g.x) + DIMS.x * (u32(g.y) + DIMS.y * u32(g.z)); }
        fn scatter(p: Particle, useTile: bool, o: vec3i, ext: vec3i) {
          let xp = p.x.xyz;
          let base = vec3i(floor(xp - 0.5));
          let fx = xp - vec3f(base);
          var w = bspline(fx);
          let mass = MATS[bitcast<u32>(p.x.w) * 3u].y;
          let tau = mat3x3f(p.c0.w, p.f0.w, p.f1.w,  p.f0.w, p.c1.w, p.f2.w,  p.f1.w, p.f2.w, p.c2.w);
          let C = mat3x3f(p.c0.xyz, p.c1.xyz, p.c2.xyz);
          let affine = -u.vol4 * tau + mass * C;
          let mv = mass * p.v.xyz;
          for (var a = 0; a < 3; a++) {
            for (var b = 0; b < 3; b++) {
              for (var c = 0; c < 3; c++) {
                let off = vec3i(a, b, c);
                let dpos = vec3f(off) - fx;
                let wt = w[a].x * w[b].y * w[c].z;
                let mom = wt * (mv + affine * dpos);
                let q = vec4i(toFixed(mom.x), toFixed(mom.y), toFixed(mom.z), toFixed(wt * mass));
                let g = base + off;
                if (useTile) {
                  let l = g - o;
                  let ti = u32((l.z * ext.y + l.y) * ext.x + l.x) * 4u;
                  atomicAdd(&tile[ti], q.x); atomicAdd(&tile[ti + 1u], q.y);
                  atomicAdd(&tile[ti + 2u], q.z); atomicAdd(&tile[ti + 3u], q.w);
                } else {
                  let gi = nodeIndex(g) * 4u;
                  atomicAdd(&grid[gi], q.x); atomicAdd(&grid[gi + 1u], q.y);
                  atomicAdd(&grid[gi + 2u], q.z); atomicAdd(&grid[gi + 3u], q.w);
                }
              }
            }
          }
        }`,
      code: /* wgsl */`
        let first = wgi * WG * PPT;
        if (lidx < 3u) { atomicStore(&bmin[lidx], 0x3fffffff); atomicStore(&bmax[lidx], -0x3fffffff); }
        workgroupBarrier();
        var lo = vec3i(0x3fffffff);
        var hi = vec3i(-0x3fffffff);
        for (var k = 0u; k < PPT; k++) {
          let pi = first + k * WG + lidx;
          if (pi < u.n) {
            let b = vec3i(floor(P[pi].x.xyz - 0.5));
            lo = min(lo, b); hi = max(hi, b);
          }
        }
        ${sg ? /* wgsl */`
        lo = subgroupMin(lo); hi = subgroupMax(hi);
        if (sgi == 0u) {` : '{'}
          atomicMin(&bmin[0], lo.x); atomicMin(&bmin[1], lo.y); atomicMin(&bmin[2], lo.z);
          atomicMax(&bmax[0], hi.x); atomicMax(&bmax[1], hi.y); atomicMax(&bmax[2], hi.z);
        }
        workgroupBarrier();
        if (lidx == 0u) {
          wbox[0] = atomicLoad(&bmin[0]); wbox[1] = atomicLoad(&bmin[1]); wbox[2] = atomicLoad(&bmin[2]);
          wbox[3] = atomicLoad(&bmax[0]); wbox[4] = atomicLoad(&bmax[1]); wbox[5] = atomicLoad(&bmax[2]);
        }
        let box = workgroupUniformLoad(&wbox);
        let o = vec3i(box[0], box[1], box[2]);
        let ext = vec3i(box[3], box[4], box[5]) - o + 3;
        if (any(ext <= vec3i(0))) { return; }
        let tileNodes = ext.x * ext.y * ext.z;
        let useTile = tileNodes <= i32(TILE_CAP) && u.forceGlobal == 0u;
        for (var k = 0u; k < PPT; k++) {
          let pi = first + k * WG + lidx;
          if (pi < u.n) { scatter(P[pi], useTile, o, ext); }
        }
        if (useTile) {
          workgroupBarrier();
          for (var t = lidx; t < u32(tileNodes); t += WG) {
            let m = atomicLoad(&tile[t * 4u + 3u]);
            let mx = atomicLoad(&tile[t * 4u]);
            let my = atomicLoad(&tile[t * 4u + 1u]);
            let mz = atomicLoad(&tile[t * 4u + 2u]);
            if ((m | mx | my | mz) != 0) {
              let it = i32(t);
              let l = vec3i(it % ext.x, (it / ext.x) % ext.y, it / (ext.x * ext.y));
              let gi = nodeIndex(o + l) * 4u;
              atomicAdd(&grid[gi], mx); atomicAdd(&grid[gi + 1u], my);
              atomicAdd(&grid[gi + 2u], mz); atomicAdd(&grid[gi + 3u], m);
            }
          }
        }`,
    });

    // --- grid update: momentum -> velocity, gravity, walls, colliders
    this.gridK = ctx.kernel({
      ...common, name: 'mpm-grid',
      buffers: { grid: { type: 'vec4i' }, COL: { type: 'vec4f', access: 'read' } },
      uniforms: { gravity: 'vec3f', bound: 'f32', numColliders: 'u32', floorFriction: 'f32' },
      shared: /* wgsl */`
        fn colliderSDF(kind: u32, d: vec3f, s: vec4f) -> f32 {
          if (kind == 0u) { return length(d) - s.x; }
          if (kind == 1u) {
            let q = abs(d) - s.xyz;
            return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
          }
          return dot(d, s.xyz);
        }`,
      code: /* wgsl */`
        let g = grid[i];
        var out = vec4f(0.0);
        if (g.w > 0) {
          let m = f32(g.w) * INV_FP;
          var v = vec3f(g.xyz) * INV_FP / m + u.gravity;
          let c = vec3f(f32(i % DIMS.x), f32((i / DIMS.x) % DIMS.y), f32(i / (DIMS.x * DIMS.y)));
          let hi = vec3f(DIMS) - u.bound;
          // separating slip walls; the floor applies Coulomb friction
          if (c.y < u.bound && v.y < 0.0) {
            let vn = -v.y;
            v.y = 0.0;
            let vt = length(v.xz);
            if (vt > 1e-9) { let f = max(0.0, 1.0 - u.floorFriction * vn / vt); v = vec3f(v.x * f, 0.0, v.z * f); }
          }
          if (c.x < u.bound && v.x < 0.0) { v.x = 0.0; }
          if (c.z < u.bound && v.z < 0.0) { v.z = 0.0; }
          if (c.x > hi.x && v.x > 0.0) { v.x = 0.0; }
          if (c.y > hi.y && v.y > 0.0) { v.y = 0.0; }
          if (c.z > hi.z && v.z > 0.0) { v.z = 0.0; }
          for (var k = 0u; k < u.numColliders; k++) {
            let h = COL[k * 3u]; let s = COL[k * 3u + 1u]; let cv = COL[k * 3u + 2u].xyz;
            let kind = bitcast<u32>(h.x);
            let d = c - h.yzw;
            let dist = colliderSDF(kind, d, s);
            if (dist < 1.0) {
              let e = 0.5;
              let n = normalize(vec3f(
                colliderSDF(kind, d + vec3f(e, 0.0, 0.0), s) - colliderSDF(kind, d - vec3f(e, 0.0, 0.0), s),
                colliderSDF(kind, d + vec3f(0.0, e, 0.0), s) - colliderSDF(kind, d - vec3f(0.0, e, 0.0), s),
                colliderSDF(kind, d + vec3f(0.0, 0.0, e), s) - colliderSDF(kind, d - vec3f(0.0, 0.0, e), s)) + vec3f(0.0, 1e-7, 0.0));
              var vr = v - cv;
              let vn = dot(vr, n);
              if (vn < 0.0) {
                vr -= vn * n;
                let vt = length(vr);
                if (vt > 1e-9) { vr *= max(0.0, 1.0 - s.w * -vn / vt); }
                v = vr + cv;
              }
            }
          }
          out = vec4f(v, m);
        }
        grid[i] = bitcast<vec4i>(out);`,
    });

    // --- G2P: gather velocity + APIC C, advect, update F, plasticity, stress
    this.g2pK = ctx.kernel({
      ...common, name: 'mpm-g2p',
      buffers: { P: { type: 'Particle' }, GV: { type: 'vec4f', access: 'read' }, MATS: { type: 'vec4f', access: 'read' } },
      uniforms: { bound: 'f32', vmax: 'f32' },
      code: /* wgsl */`
        var p = P[i];
        let xp = p.x.xyz;
        let base = vec3i(floor(xp - 0.5));
        let fx = xp - vec3f(base);
        var w = bspline(fx);
        var v = vec3f(0.0);
        var B = mat3x3f();
        for (var a = 0; a < 3; a++) {
          for (var b = 0; b < 3; b++) {
            for (var c = 0; c < 3; c++) {
              let off = vec3i(a, b, c);
              let dpos = vec3f(off) - fx;
              let wt = w[a].x * w[b].y * w[c].z;
              let g = base + off;
              let gv = GV[u32(g.x) + DIMS.x * (u32(g.y) + DIMS.y * u32(g.z))].xyz;
              v += wt * gv;
              B += outer3(wt * gv, dpos);
            }
          }
        }
        let C = 4.0 * B;
        let sp = length(v);
        if (sp > u.vmax) { v *= u.vmax / sp; }
        let x = clamp(xp + v, vec3f(u.bound - 1.0), vec3f(DIMS) - u.bound + 1.0);
        let mid = bitcast<u32>(p.x.w);
        let m0 = MATS[mid * 3u];
        let m1 = MATS[mid * 3u + 1u];
        let m2 = MATS[mid * 3u + 2u];
        let kind = bitcast<u32>(m0.x);
        var F = mat3x3f(p.f0.xyz, p.f1.xyz, p.f2.xyz);
        var tau = mat3x3f();
        var vw = p.v.w;
        if (kind == 0u) {
          // weakly compressible fluid tracked by J only. Linear EOS tau = K (J - 1) I keeps the
          // wave speed constant (stable at any compression); tension is capped to limit clumping.
          let J = clamp(vw * (1.0 + trace3(C)), 0.6, 1.4);
          vw = J;
          tau = (m0.z * min(J - 1.0, 0.02)) * I3 + (m1.x * J) * (C + transpose(C));
          F = I3;
        } else {
          F = (I3 + C) * F;
          let mu0 = m0.z; let la0 = m0.w;
          if (kind == 1u) {
            // Neo-Hookean: tau = mu (F Fᵀ - I) + lambda ln(J) I  (no polar decomposition needed)
            let J = max(determinant(F), 0.05);
            tau = mu0 * (F * transpose(F) - I3) + (la0 * log(J)) * I3;
          } else if (kind == 2u) {
            // snow (Stomakhin 2013): clamp singular values, harden with plastic compression
            let d = svd3(F);
            let s = clamp(d.s, vec3f(1.0 - m1.z), vec3f(1.0 + m1.w));
            let Jold = d.s.x * d.s.y * d.s.z;
            let Jnew = s.x * s.y * s.z;
            vw = clamp(vw * Jold / Jnew, 0.6, 20.0);
            F = d.U * diag3(s) * transpose(d.V);
            let h = exp(m1.y * (1.0 - vw));
            let mu = mu0 * h; let la = la0 * h;
            let R = d.U * transpose(d.V);
            let J = Jnew;
            tau = 2.0 * mu * (F - R) * transpose(F) + (la * J * (J - 1.0)) * I3;
          } else {
            // sand (Klar 2016): Drucker–Prager return mapping in Hencky strain
            let d = svd3(F);
            var e = log(max(abs(d.s), vec3f(1e-4)));
            let tr = e.x + e.y + e.z;
            let dev = e - tr / 3.0;
            let devn = length(dev);
            if (tr >= 0.0) {
              e = vec3f(0.0);
            } else {
              let dg = devn + (3.0 * la0 + 2.0 * mu0) / (2.0 * mu0) * tr * m2.x;
              if (dg > 0.0 && devn > 1e-12) { e = e - dg * dev / devn; }
            }
            let s = exp(e);
            F = d.U * diag3(s) * transpose(d.V);
            let te = e.x + e.y + e.z;
            tau = d.U * diag3(2.0 * mu0 * e + la0 * te) * transpose(d.U);
          }
        }
        P[i] = Particle(vec4f(x, p.x.w), vec4f(v, vw),
          vec4f(C[0], tau[0].x), vec4f(C[1], tau[1].y), vec4f(C[2], tau[2].z),
          vec4f(F[0], tau[1].x), vec4f(F[1], tau[2].x), vec4f(F[2], tau[2].y));`,
    });

    // --- block sort: key = Morton(block), counting sort fused with the particle scatter
    this.keyK = ctx.kernel({
      ...common, name: 'mpm-key',
      buffers: { P: { type: 'Particle', access: 'read' }, key: { type: 'u32' }, rank: { type: 'u32' }, cnt: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let b = vec3u(max(P[i].x.xyz, vec3f(0.0))) / 4u;
        let k = morton3(b);
        key[i] = k;
        rank[i] = atomicAdd(&cnt[k], 1u);`,
    });
    this.reorderK = ctx.kernel({
      ...common, name: 'mpm-reorder',
      buffers: { src: { type: 'Particle', access: 'read' }, dst: { type: 'Particle' }, key: { type: 'u32', access: 'read' },
        rank: { type: 'u32', access: 'read' }, start: { type: 'u32', access: 'read' } },
      code: 'dst[start[key[i]] + rank[i]] = src[i];',
    });
  }

  // ------------------------------------------------------------------ stepping
  /** Advance by one frame (1/60 s x timeScale), split into `substeps`. */
  step() {
    if (this.count === 0) return;
    const ctx = this.ctx;
    this._uploadColliders();
    const dt = this.substepDt;
    const g = this.gravity.map((x) => (x * dt * dt) / this.dx);
    const vol = 1 / this.ppcAxis ** 3;
    for (let s = 0; s < this.substeps; s++) {
      if (this.stepCount % this.sortEvery === 0) ctx.region('mpm sort', () => this._sort());
      ctx.region('mpm p2g', () => this.p2g());
      ctx.region('mpm grid', () => {
        this.gridK.run(this.numNodes, { gravity: g, bound: this.bound, numColliders: Math.min(16, this.colliders.length), floorFriction: 0.4 },
          { grid: this.grid, COL: this.colBuf });
        this.coupling?.substep(s);
      });
      ctx.region('mpm g2p', () => {
        this.g2pK.run(this.count, { bound: this.bound, vmax: 0.9 }, { P: this.P, GV: this.grid, MATS: this.matBuf });
      });
      this.stepCount++;
    }
    this.coupling?.finish();
  }

  /** Clear the grid and scatter particles (P2G). forceGlobal: skip the shared-memory tile (baseline). */
  p2g(P = this.P, forceGlobal = false, grid = this.grid) {
    const vol = 1 / this.ppcAxis ** 3;
    this.ctx.clear(grid);
    this.p2gK.dispatch(Math.ceil(this.count / this.p2gPerGroup), { n: this.count, vol4: 4 * vol, forceGlobal: forceGlobal ? 1 : 0 },
      { P, grid, MATS: this.matBuf });
  }

  _sort() {
    const ctx = this.ctx;
    ctx.clear(this.blockCount);
    this.keyK.run(this.count, null, { P: this.P, key: this.key, rank: this.rank, cnt: this.blockCount });
    this.scan.run(this.blockCount, this.blockCount, this.numBlocks + 1);
    this.reorderK.run(this.count, null, { src: this.P, dst: this.P2, key: this.key, rank: this.rank, start: this.blockCount });
    [this.P, this.P2] = [this.P2, this.P];
  }

  /** Read particle positions (world units) — for tests / export. */
  async readPositions() {
    const raw = await this.P.read(this.count * 8);
    const out = new Float32Array(this.count * 3);
    for (let k = 0; k < this.count; k++)
      for (let a = 0; a < 3; a++) out[3 * k + a] = raw[32 * k + a] * this.dx + this.origin[a];
    return out;
  }

  /** Material color palette (linear RGB) indexed like the particle material ids. */
  palette() { return this.materials.map((m) => m.color || [0.7, 0.7, 0.7]); }

  /** Bitmask of material slots that are fluids (for the surface renderer). */
  fluidMask() { return this.materials.reduce((m, d, k) => (d.kind === FLUID ? m | (1 << k) : m), 0); }

  /**
   * Screen-space water surface for the fluid materials (pair with spheres({ solidsOnly: true })
   * to draw jelly / snow / sand as particles).
   */
  fluid(opts = {}) {
    const sim = this;
    const fs = new FluidSurface(this.ctx, {
      data: () => sim.P, stride: 8, position: 0, count: () => sim.count,
      radius: (this.dx / this.ppcAxis) * 0.9, scale: this.dx, offset: this.origin,
      mask: '((u.matMask >> bitcast<u32>(px.w)) & 1u) == 1u', ...opts,
    });
    return {
      surface: fs,
      set(v) { fs.set(v); return this; },
      afterScene(view) { fs.set({ matMask: sim.fluidMask() }); fs.afterScene(view); },
    };
  }

  /** Sphere-impostor layer that follows the ping-ponged particle buffer. */
  spheres(opts = {}) {
    const sim = this;
    const sp = new Spheres(this.ctx, {
      mask: '((u.matMask >> bitcast<u32>(px.w)) & 1u) == 1u',
      data: this.P, stride: 8, position: 0, velocity: 1, material: 'bitcast<u32>(px.w)',
      count: () => sim.count, radius: (this.dx / this.ppcAxis) * 0.6 * (opts.size ?? 1), scale: this.dx,
      offset: this.origin, palette: opts.palette || this.palette(), velScale: this.dx / this.substepDt, ...opts,
    });
    return {
      spheres: sp,
      set(v) { sp.set(v); return this; },
      draw(pass, view) {
        if (!opts.palette) sim.palette().forEach((c, k) => sp.program.set({ ['pal' + k]: [...c, 1] }));
        sp.program.set({ matMask: opts.solidsOnly ? ~sim.fluidMask() >>> 0 : 0xffffffff });
        sp.program.bind({ P: sim.P });
        sp.draw(pass, view);
      },
    };
  }
}

// ------------------------------------------------------------------ shape sampling
function sampleShape(shape, spacing, jitter, seed) {
  let rnd = seed >>> 0 || 1;
  const rand = () => { rnd ^= rnd << 13; rnd ^= rnd >>> 17; rnd ^= rnd << 5; return ((rnd >>> 0) / 4294967296); };
  let lo, hi, inside;
  if (shape.box) {
    [lo, hi] = shape.box;
    inside = () => true;
  } else if (shape.sphere) {
    const [c, r] = shape.sphere;
    lo = c.map((x) => x - r); hi = c.map((x) => x + r);
    inside = (p) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 <= r * r;
  } else if (shape.cylinder) {
    const [c, r, h] = shape.cylinder;
    lo = [c[0] - r, c[1] - h, c[2] - r]; hi = [c[0] + r, c[1] + h, c[2] + r];
    inside = (p) => (p[0] - c[0]) ** 2 + (p[2] - c[2]) ** 2 <= r * r;
  } else if (shape.sdf) {
    lo = shape.bounds[0]; hi = shape.bounds[1];
    inside = (p) => shape.sdf(p) <= 0;
  } else throw new Error('MPM.add: shape needs box | sphere | cylinder | sdf | points');
  const n = [0, 1, 2].map((a) => Math.max(1, Math.floor((hi[a] - lo[a]) / spacing)));
  const out = [];
  const p = [0, 0, 0];
  for (let z = 0; z < n[2]; z++)
    for (let y = 0; y < n[1]; y++)
      for (let x = 0; x < n[0]; x++) {
        p[0] = lo[0] + (x + 0.5 + (rand() - 0.5) * jitter) * spacing;
        p[1] = lo[1] + (y + 0.5 + (rand() - 0.5) * jitter) * spacing;
        p[2] = lo[2] + (z + 0.5 + (rand() - 0.5) * jitter) * spacing;
        if (inside(p)) out.push(p[0], p[1], p[2]);
      }
  return new Float32Array(out);
}
