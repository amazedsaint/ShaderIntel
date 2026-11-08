// Direct-summation N-body gravity, O(N^2), tuned to run near the ALU roofline.
//
// One interaction is ~20 flops (3 sub, 3 fma for r^2, rsqrt, 2 mul, 3 fma) on
// 16 bytes of data, so the kernel is compute-bound as long as bodies are
// re-used from on-chip memory. Variants (picked per device by autotune()):
//   naive      every thread streams all bodies from global memory (baseline)
//   tiled      GPU Gems 3 ch. 31: workgroup tiles in shared memory
//   tiled-K    + register blocking: each thread integrates K bodies, so every
//              shared-memory load feeds K interactions (fewer loads/flop)
//   sgK        subgroup broadcast: bodies live in registers and are broadcast
//              lane by lane (no shared memory, no barriers) — needs a fixed
//              subgroup size, e.g. 32 on Apple GPUs.
// Integration: symplectic Euler (kick-drift), double-buffered positions.
import { destroyAll } from '../core/dispose.js';
import { SIError } from '../core/errors.js';

const VARIANTS = (caps) => {
  const v = [{ mode: 'naive', wg: 256, K: 1 }];
  // K = 8 spills registers on every GPU we tried; subgroup broadcast lost to shared memory on Apple M2.
  for (const wg of [64, 128, 256]) for (const K of [1, 2, 4]) v.push({ mode: 'tiled', wg, K });
  if (caps.subgroups && caps.subgroupMin === 32 && caps.subgroupMax === 32)
    for (const K of [1, 2]) v.push({ mode: 'sg', wg: 256, K });
  return v;
};

function interact(k) {
  return `{ let d = b.xyz - p${k}.xyz; let r2 = dot(d, d) + u.eps2; let s = inverseSqrt(r2); a${k} += d * (b.w * s * s * s); }`;
}

export class NBody {
  /**
   * @param {Context} ctx
   * @param {object} o { count, G = 1, softening = 0.02, dt = 0.002, variant }
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.count = o.count || 16384;
    this.G = o.G ?? 1;
    this.softening = o.softening ?? 0.02;
    this.dt = o.dt ?? 0.002;
    this.pos = ctx.array('vec4f', this.count, { label: 'nbody-pos', usage: GPUBufferUsage.VERTEX });
    this.pos2 = ctx.array('vec4f', this.count, { label: 'nbody-pos2', usage: GPUBufferUsage.VERTEX });
    this.vel = ctx.array('vec4f', this.count, { label: 'nbody-vel' });
    this.kernels = new Map();
    this.variant = o.variant || { mode: 'tiled', wg: 256, K: 2 };
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    destroyAll(this.pos, this.pos2, this.vel);
  }

  static variants(caps) { return VARIANTS(caps); }

  kernel(v = this.variant) {
    const key = `${v.mode}-${v.wg}-${v.K}`;
    let k = this.kernels.get(key);
    if (k) return k;
    const K = v.K, WG = v.wg;
    const ks = [...Array(K).keys()];
    const loadOwn = ks.map((k) => `let i${k} = base + ${k}u * WG + lidx; let p${k} = pos[min(i${k}, u.n - 1u)]; var a${k} = vec3f(0.0);`).join('\n');
    const store = ks.map((k) => `if (i${k} < u.n) {
        let v = vel[i${k}].xyz + a${k} * (u.G * u.dt);
        vel[i${k}] = vec4f(v, 0.0);
        posOut[i${k}] = vec4f(p${k}.xyz + v * u.dt, p${k}.w);
      }`).join('\n');
    const all = (indent) => ks.map((k) => indent + interact(k)).join('\n');
    let shared = '', body;
    if (v.mode === 'naive') {
      body = `
        ${loadOwn}
        for (var j = 0u; j < u.n; j++) { let b = pos[j];
${all('          ')}
        }`;
    } else if (v.mode === 'tiled') {
      shared = `var<workgroup> tile: array<vec4f, ${WG}>;`;
      body = `
        ${loadOwn}
        let tiles = (u.n + WG - 1u) / WG;
        for (var t = 0u; t < tiles; t++) {
          let j = t * WG + lidx;
          tile[lidx] = select(vec4f(0.0), pos[min(j, u.n - 1u)], j < u.n);
          workgroupBarrier();
          for (var q = 0u; q < WG; q += 4u) {
            ${[0, 1, 2, 3].map((r) => `{ let b = tile[q + ${r}u];\n${all('              ')}\n            }`).join('\n            ')}
          }
          workgroupBarrier();
        }`;
    } else {
      // subgroup broadcast: each lane holds one body; constant-index broadcast over 32 lanes
      body = `
        ${loadOwn}
        let chunks = (u.n + 31u) / 32u;
        for (var t = 0u; t < chunks; t++) {
          let j = t * 32u + sgi;
          let mine = select(vec4f(0.0), pos[min(j, u.n - 1u)], j < u.n);
          ${[...Array(32).keys()].map((l) => `{ let b = subgroupBroadcast(mine, ${l}u);\n${all('            ')}\n          }`).join('\n          ')}
        }`;
    }
    k = this.ctx.kernel({
      name: `nbody-${key}`, workgroupSize: WG, guard: false, subgroups: v.mode === 'sg',
      buffers: { pos: { type: 'vec4f', access: 'read' }, posOut: { type: 'vec4f' }, vel: { type: 'vec4f' } },
      uniforms: { eps2: 'f32', G: 'f32', dt: 'f32' },
      shared,
      code: `let base = wgi * (WG * ${K}u);\n${body}\n${store}`,
    });
    k.bodiesPerGroup = WG * K;
    this.kernels.set(key, k);
    return k;
  }

  /** Advance `steps` steps. */
  step(steps = 1, variant = this.variant) {
    if (this.destroyed) throw new SIError('NBody.step: it was destroyed');
    const k = this.kernel(variant);
    const groups = Math.ceil(this.count / k.bodiesPerGroup);
    this.ctx.region('nbody', () => {
      for (let s = 0; s < steps; s++) {
        // eps2 > 0: a body's interaction with itself would be 0 · ∞ otherwise (softening 0)
        k.dispatch(groups, { n: this.count, eps2: Math.max(this.softening ** 2, 1e-20), G: this.G, dt: this.dt },
          { pos: this.pos, posOut: this.pos2, vel: this.vel });
        [this.pos, this.pos2] = [this.pos2, this.pos];
      }
    });
  }

  /**
   * Time every variant on this device and keep the fastest. Returns
   * [{ variant, ms, gips }] sorted fastest first (gips = 1e9 interactions / s).
   */
  async autotune({ reps = 4, cache = true } = {}) {
    const caps = this.ctx.caps;
    // the variants on offer depend on subgroup support (init({ subgroups: false }) removes some)
    const key = `si-nbody-v2-${caps.vendor}-${caps.architecture}-sg${+caps.subgroups}-${Math.round(Math.log2(this.count))}`;
    const offered = VARIANTS(caps);
    const same = (a, b) => a && b && a.mode === b.mode && a.wg === b.wg && a.K === b.K;
    if (cache && typeof localStorage !== 'undefined') {
      try {
        const c = JSON.parse(localStorage.getItem(key));
        if (c && offered.some((v) => same(v, c.variant))) { this.variant = offered.find((v) => same(v, c.variant)); return c.results; }
      } catch { /* a corrupt cache entry is re-measured */ }
    }
    // timing steps the simulation: keep its state and put it back afterwards
    const savedPos = this.ctx.array('vec4f', this.count, { label: 'nbody-autotune-pos' });
    const savedVel = this.ctx.array('vec4f', this.count, { label: 'nbody-autotune-vel' });
    this.ctx.copy(this.pos, savedPos); this.ctx.copy(this.vel, savedVel);
    const results = [];
    const n2 = this.count * this.count;
    try {
      for (const v of offered) {
        const k = this.kernel(v);
        try { await k.ready(); } catch { continue; }
        // choose a batch so each measurement is ~20 ms of GPU work
        const ms = await timeIt(this.ctx, () => this.step(1, v), reps, n2);
        results.push({ variant: v, ms, gips: n2 / (ms * 1e6) });
      }
    } finally {
      this.ctx.copy(savedPos, this.pos); this.ctx.copy(savedVel, this.vel);
      savedPos.destroy(); savedVel.destroy();
    }
    if (!results.length) throw new SIError('NBody.autotune: no kernel variant compiled on this device');
    results.sort((a, b) => a.ms - b.ms);
    this.variant = results[0].variant;
    if (cache && typeof localStorage !== 'undefined') localStorage.setItem(key, JSON.stringify({ variant: this.variant, results }));
    return results;
  }

  // ---------------------------------------------------------------- initial conditions
  /** Two rotating disk galaxies on a collision course. */
  galaxies(o = {}) {
    const n = this.count, rnd = mulberry(o.seed ?? 7);
    const pos = new Float32Array(n * 4), vel = new Float32Array(n * 4);
    const gal = [
      { c: [-0.9, 0, 0.15], v: [0.18, 0, -0.06], axis: norm([0.15, 1, 0.2]), share: 0.55 },
      { c: [0.9, 0, -0.15], v: [-0.22, 0, 0.07], axis: norm([-0.3, 1, -0.1]), share: 0.45 },
    ];
    const mStar = 1 / n;
    let k = 0;
    for (const g of gal) {
      const count = g === gal[1] ? n - k : Math.floor(n * g.share);
      const M = count * mStar;
      const coreMass = M * 0.25;
      const [t1, t2] = perp(g.axis);
      for (let s = 0; s < count; s++, k++) {
        let x, y, z, vx, vy, vz, m = mStar;
        if (s === 0) { // massive core
          m = coreMass; x = 0; y = 0; z = 0; vx = vy = vz = 0;
        } else {
          const r = 0.03 + 0.5 * -Math.log(1 - rnd() * 0.95) * 0.35;      // exponential disk
          const a = rnd() * Math.PI * 2;
          const h = (rnd() - 0.5) * 0.02 * (1 + r * 2);
          const enclosed = coreMass + (M - coreMass) * (1 - Math.exp(-r / 0.175) * (1 + r / 0.175));
          const vc = Math.sqrt((this.G * enclosed) / Math.sqrt(r * r + this.softening ** 2));
          const ca = Math.cos(a), sa = Math.sin(a);
          x = (t1[0] * ca + t2[0] * sa) * r + g.axis[0] * h;
          y = (t1[1] * ca + t2[1] * sa) * r + g.axis[1] * h;
          z = (t1[2] * ca + t2[2] * sa) * r + g.axis[2] * h;
          // tangent = axis x radial
          const rad = [t1[0] * ca + t2[0] * sa, t1[1] * ca + t2[1] * sa, t1[2] * ca + t2[2] * sa];
          const tan = cross(g.axis, rad);
          const disp = 0.05 * vc;
          vx = tan[0] * vc + (rnd() - 0.5) * disp; vy = tan[1] * vc + (rnd() - 0.5) * disp; vz = tan[2] * vc + (rnd() - 0.5) * disp;
        }
        pos.set([g.c[0] + x, g.c[1] + y, g.c[2] + z, m], 4 * k);
        vel.set([g.v[0] + vx, g.v[1] + vy, g.v[2] + vz, 0], 4 * k);
      }
    }
    this.ctx.flush();   // recorded steps swap pos/pos2 when they run: submit them before re-initialising
    this.pos.write(pos); this.vel.write(vel);
    return this;
  }

  /** Plummer sphere in virial equilibrium (Aarseth, Hénon & Wielen 1974). */
  plummer(o = {}) {
    const n = this.count, rnd = mulberry(o.seed ?? 3), a = o.scale ?? 0.3;
    const pos = new Float32Array(n * 4), vel = new Float32Array(n * 4);
    for (let k = 0; k < n; k++) {
      const r = a / Math.sqrt(Math.pow(rnd() * 0.99 + 1e-6, -2 / 3) - 1);
      const d = randDir(rnd);
      let q, g;
      do { q = rnd(); g = rnd() * 0.1; } while (g > q * q * Math.pow(1 - q * q, 3.5));
      const ve = Math.SQRT2 * Math.pow(r * r + a * a, -0.25) * Math.sqrt(this.G);
      const v = q * ve, dv = randDir(rnd);
      pos.set([d[0] * r, d[1] * r, d[2] * r, 1 / n], 4 * k);
      vel.set([dv[0] * v, dv[1] * v, dv[2] * v, 0], 4 * k);
    }
    this.ctx.flush();   // recorded steps swap pos/pos2 when they run: submit them before re-initialising
    this.pos.write(pos); this.vel.write(vel);
    return this;
  }
}

/** Mean GPU wall time (ms) of fn() measured over batches of ~20 ms. */
export async function timeIt(ctx, fn, reps = 4, workHint = 0) {
  const q = ctx.device.queue;
  fn(); await ctx.idle();                       // warm-up / compile
  let t0 = performance.now();
  fn(); await ctx.idle();
  let one = Math.max(0.05, performance.now() - t0);
  const batch = Math.max(1, Math.min(200, Math.round(20 / one)));
  let best = Infinity;
  for (let r = 0; r < reps; r++) {
    t0 = performance.now();
    for (let b = 0; b < batch; b++) fn();
    ctx.flush();
    await q.onSubmittedWorkDone();
    best = Math.min(best, (performance.now() - t0) / batch);
  }
  return best;
}

function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function perp(n) {
  const t = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const a = norm(cross(n, t));
  return [a, cross(n, a)];
}
function randDir(rnd) {
  const z = rnd() * 2 - 1, a = rnd() * Math.PI * 2, r = Math.sqrt(1 - z * z);
  return [r * Math.cos(a), r * Math.sin(a), z];
}
