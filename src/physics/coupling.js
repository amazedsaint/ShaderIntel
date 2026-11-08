// Two-way coupling between MLS-MPM materials (water, sand, snow, jelly) and a RigidWorld.
//
// Rigid bodies are moving boundaries of the MPM grid. After every MPM grid update, each
// grid node inside a body or within `band` cells of its surface has its velocity relative to the body
// projected (no penetration, Coulomb friction): v_rel ← v_rel − min(0, v_rel·n) n, then
// tangential friction. The momentum that projection adds to the node, m Δv, is taken from
// the body (Newton's third law): J = −Σ m Δv, L = −Σ r × m Δv. Over a frame these sums
// become the body's external impulse, applied at the start of the next rigid step.
// Buoyancy, drag and splashes are not modelled separately — they are what this exchange of
// momentum produces: a board lighter than water floats at Archimedes' waterline to about one grid
// cell (a cube of middling density tips onto an edge, as a real one does).
//
// Per workgroup (a 4×4×4 tile of nodes) the bodies whose bounding spheres touch the tile are
// culled into a candidate list in index order (a workgroup prefix scan, so the list does not
// depend on thread timing), node impulses are summed in fixed point in shared memory, and each
// tile adds its sums to the per-body accumulators with integer atomics: the result is the same
// whatever order the GPU runs things in.
//
// Units: MPM works in lattice units (cells, substeps, density × cells); the rigid world in
// metres, seconds and density × m³ (density 1 = water in both). A lattice momentum p' is
// p' dx⁴/Δt_s in world units.
import { defineModule } from '../core/wgsl.js';
import './rigid-wgsl.js';
import { destroyAll } from '../core/dispose.js';

const MAX_CAND = 128;
const FPJ = 16384;   // fixed-point scale of lattice momentum
const FPL = 1024;    // fixed-point scale of lattice angular momentum

defineModule('coupling', {
  deps: ['rigid'],
  code: /* wgsl */`
/** Signed distance from p to body b centred at c (world units) and the outward normal. */
fn bodySDF(b: Body, c: vec3f, p: vec3f) -> vec4f {
  let s = shapeOf(b);
  let R = qmat(b.q);
  let lp = transpose(R) * (p - c);
  if (s == 0u) {
    let d = length(lp);
    return vec4f(d - b.size.x, R * (lp / max(d, 1e-6)));
  }
  if (s == 2u) {
    let h = b.size.y;
    let q = vec3f(lp.x, lp.y - clamp(lp.y, -h, h), lp.z);
    let d = length(q);
    return vec4f(d - b.size.x, R * select(vec3f(0.0, sign(lp.y), 0.0), q / max(d, 1e-6), d > 1e-6));
  }
  let h = b.size.xyz;
  let q = abs(lp) - h;
  if (any(q > vec3f(0.0))) {
    let dv = lp - clamp(lp, -h, h);
    let d = length(dv);
    return vec4f(d, R * (dv / max(d, 1e-9)));
  }
  var ax = 0u;
  if (q.y > q[ax]) { ax = 1u; }
  if (q.z > q[ax]) { ax = 2u; }
  var n = vec3f(0.0);
  n[ax] = select(-1.0, 1.0, lp[ax] >= 0.0);
  return vec4f(q[ax], R * n);
}
`,
});

export class FluidRigidCoupling {
  /**
   * @param sim   MPM solver
   * @param world RigidWorld
   * o: friction (fluid–body, default 0.2), band (cells, default 0.5), mask (collision group bits of
   *    bodies that interact, default all), densityScale (rigid mass per MPM density unit per m³, default 1)
   */
  constructor(ctx, sim, world, o = {}) {
    this.ctx = ctx;
    this.sim = sim;
    this.world = world;
    this.friction = o.friction ?? 0.2;
    this.band = o.band ?? 0.5;
    this.mask = o.mask ?? 0xffffffff;
    this.densityScale = o.densityScale ?? 1;
    this.enabled = true;
    this.acc = ctx.array('vec4i', world.maxBodies * 2, { label: 'coupling-impulses' });
    const dims = sim.dims;
    this.coupleK = ctx.kernel({
      name: 'mpm-rigid-couple', imports: ['coupling'], dims: 3, workgroupSize: [4, 4, 4], guard: false,
      defines: { DIMS: `vec3u(${dims.join('u, ')}u)` },
      buffers: { grid: { type: 'vec4i' }, B: { type: 'Body', access: 'read' }, gacc: { type: 'i32', atomic: true } },
      uniforms: { nBodies: 'u32', mask: 'u32', origin: 'vec3f', dx: 'f32', invDx: 'f32', lead: 'f32', velToLattice: 'f32',
        friction: 'f32', band: 'f32' },
      shared: /* wgsl */`
        const MAXC = ${MAX_CAND}u;
        var<workgroup> cand: array<u32, ${MAX_CAND}>;
        var<workgroup> wscan: array<u32, 64>;
        var<workgroup> wcount: u32;
        var<workgroup> accJ: array<atomic<i32>, ${MAX_CAND * 3}>;
        var<workgroup> accL: array<atomic<i32>, ${MAX_CAND * 3}>;`,
      code: /* wgsl */`
        let tileLo = vec3f(wid * 4u) - vec3f(u.band + 1.0);
        let tileHi = vec3f(wid * 4u) + vec3f(3.0 + u.band + 1.0);
        if (lidx == 0u) { wcount = 0u; }
        workgroupBarrier();
        // cull bodies against this tile, in index order (workgroup prefix scan per round of 64)
        let rounds = (u.nBodies + 63u) / 64u;
        for (var r = 0u; r < rounds; r++) {
          let bi = r * 64u + lidx;
          var hit = 0u;
          if (bi < u.nBodies) {
            let b = B[bi];
            if ((b.info.z & (F_DISABLED | F_SENSOR)) == 0u && (b.info.x & u.mask) != 0u) {
              let c = (b.x.xyz + b.v.xyz * u.lead - u.origin) * u.invDx;
              let q = clamp(c, tileLo, tileHi);
              if (length(c - q) <= b.x.w * u.invDx) { hit = 1u; }
            }
          }
          wscan[lidx] = hit;
          workgroupBarrier();
          for (var off = 1u; off < 64u; off <<= 1u) {
            let add = select(0u, wscan[max(lidx, off) - off], lidx >= off);
            workgroupBarrier();
            wscan[lidx] += add;
            workgroupBarrier();
          }
          let incl = wscan[lidx];
          let total = wscan[63];
          let base = workgroupUniformLoad(&wcount);
          if (hit == 1u && base + incl - 1u < MAXC) { cand[base + incl - 1u] = bi; }
          workgroupBarrier();
          if (lidx == 0u) { wcount = base + total; }
          workgroupBarrier();
        }
        let nc = min(workgroupUniformLoad(&wcount), MAXC);
        for (var k = lidx; k < nc * 3u; k += 64u) { atomicStore(&accJ[k], 0); atomicStore(&accL[k], 0); }
        workgroupBarrier();
        let g = wid * 4u + lid;
        if (nc > 0u && all(g < DIMS)) {
          let gi = g.x + DIMS.x * (g.y + DIMS.y * g.z);
          let cell = bitcast<vec4f>(grid[gi]);
          let m = cell.w;
          if (m > 0.0) {
            var v = cell.xyz;
            let pw = u.origin + vec3f(g) * u.dx;
            for (var c = 0u; c < nc; c++) {
              let b = B[cand[c]];
              let bc = b.x.xyz + b.v.xyz * u.lead;
              let sd = bodySDF(b, bc, pw);
              if (sd.x * u.invDx < u.band) {
                let r = pw - bc;
                let vb = (b.v.xyz + cross(b.w.xyz, r)) * u.velToLattice;   // body velocity at the node (lattice)
                var vr = v - vb;
                let vn = dot(vr, sd.yzw);
                if (vn < 0.0) {
                  vr -= vn * sd.yzw;
                  let vt = length(vr);
                  if (vt > 1e-9) { vr *= max(0.0, 1.0 - u.friction * (-vn) / vt); }
                  let vnew = vr + vb;
                  if (b.mass.x > 0.0) {
                    let J = -m * (vnew - v);          // reaction on the body (lattice momentum)
                    let L = cross(r * u.invDx, J);
                    atomicAdd(&accJ[c * 3u], i32(round(J.x * ${FPJ}.0)));
                    atomicAdd(&accJ[c * 3u + 1u], i32(round(J.y * ${FPJ}.0)));
                    atomicAdd(&accJ[c * 3u + 2u], i32(round(J.z * ${FPJ}.0)));
                    atomicAdd(&accL[c * 3u], i32(round(L.x * ${FPL}.0)));
                    atomicAdd(&accL[c * 3u + 1u], i32(round(L.y * ${FPL}.0)));
                    atomicAdd(&accL[c * 3u + 2u], i32(round(L.z * ${FPL}.0)));
                  }
                  v = vnew;
                }
              }
            }
            grid[gi] = bitcast<vec4i>(vec4f(v, m));
          }
        }
        workgroupBarrier();
        for (var k = lidx; k < nc * 3u; k += 64u) {
          let c = k / 3u; let a = k % 3u; let bi = cand[c];
          let j = atomicLoad(&accJ[k]); let l = atomicLoad(&accL[k]);
          if (j != 0) { atomicAdd(&gacc[bi * 8u + a], j); }
          if (l != 0) { atomicAdd(&gacc[bi * 8u + 4u + a], l); }
        }`,
    });
    this.applyK = ctx.kernel({
      name: 'mpm-rigid-apply', imports: ['rigid'],
      buffers: { acc: { type: 'vec4i' }, ext: { type: 'vec4f' } },
      uniforms: { momToWorld: 'f32', dx: 'f32' },
      code: /* wgsl */`
        let a = acc[2u * i]; let b = acc[2u * i + 1u];
        if (all(a.xyz == vec3i(0)) && all(b.xyz == vec3i(0))) { return; }
        let J = vec3f(a.xyz) / ${FPJ}.0 * u.momToWorld;
        let L = vec3f(b.xyz) / ${FPL}.0 * u.momToWorld * u.dx;
        ext[2u * i] += vec4f(J, 0.0);
        ext[2u * i + 1u] += vec4f(L, 0.0);
        acc[2u * i] = vec4i(0); acc[2u * i + 1u] = vec4i(0);`,
    });
    this.tiles = dims.map((d) => Math.ceil(d / 4));
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    destroyAll(this.acc);
  }

  /** Called by MPM after the grid update of substep s. */
  substep(s) {
    if (this.destroyed || this.world.destroyed || this.sim.destroyed) return;
    const w = this.world, sim = this.sim;
    if (!this.enabled || !w.count) return;
    const dts = sim.substepDt;
    this.coupleK.dispatch(this.tiles, {
      nBodies: w.count, mask: this.mask >>> 0, origin: sim.origin, dx: sim.dx, invDx: 1 / sim.dx,
      lead: (s + 0.5) * dts, velToLattice: dts / sim.dx, friction: this.friction, band: this.band,
    }, { grid: sim.grid, B: w.B, gacc: this.acc });
  }

  /** Called by MPM after its frame: hand the accumulated impulses to the rigid world. */
  finish() {
    if (this.destroyed || this.world.destroyed || this.sim.destroyed) return;
    const w = this.world, sim = this.sim;
    if (!this.enabled || !w.count) return;
    const dx = sim.dx;
    this.applyK.run(w.count, { momToWorld: this.densityScale * dx ** 4 / sim.substepDt, dx }, { acc: this.acc, ext: w.ext });
  }
}
