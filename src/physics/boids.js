// Boids (Reynolds 1987) at scale: separation, alignment and cohesion over a
// fixed perception radius, with neighbours found through the counting-sort
// uniform grid. Every frame the agents are re-ordered into cell order, so the
// neighbour loop reads contiguous memory (the arrays stay permuted — boids have
// no identity, so no index map is needed).
import { SpatialGrid } from '../prims/grid.js';

export class Boids {
  /**
   * o: count, bounds (half-size of the cube, default 1), radius (perception), maxSpeed, minSpeed,
   *    separation, alignment, cohesion, maxNeighbors (cap on interactions per boid)
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.count = o.count || 65536;
    this.half = o.bounds ?? 1;
    this.params = {
      radius: o.radius ?? 0.06, maxSpeed: o.maxSpeed ?? 0.6, minSpeed: o.minSpeed ?? 0.25,
      separation: o.separation ?? 1.6, alignment: o.alignment ?? 1.0, cohesion: o.cohesion ?? 0.9,
      maxNeighbors: o.maxNeighbors ?? 48, wall: o.wall ?? 2.0,
      predator: [0, 0, 0], predatorRadius: 0, predatorForce: 6,
    };
    const n = this.count;
    this.pos = ctx.array('vec4f', n, { label: 'boids-pos' });
    this.vel = ctx.array('vec4f', n, { label: 'boids-vel' });
    this.sPos = ctx.array('vec4f', n, { label: 'boids-spos' });
    this.sVel = ctx.array('vec4f', n, { label: 'boids-svel' });
    const h = this.half;
    this.grid = new SpatialGrid(ctx, { maxCount: n, min: [-h, -h, -h], max: [h, h, h], cellSize: this.params.radius });
    this.updateK = ctx.kernel({
      name: 'boids-update', imports: ['grid'],
      buffers: { S: { type: 'vec4f', access: 'read' }, SV: { type: 'vec4f', access: 'read' }, start: { type: 'u32', access: 'read' },
        pos: { type: 'vec4f' }, vel: { type: 'vec4f' } },
      uniforms: { ...this.grid.uniforms(), dt: 'f32', r2: 'f32', maxSpeed: 'f32', minSpeed: 'f32', wSep: 'f32', wAli: 'f32', wCoh: 'f32',
        half: 'f32', wall: 'f32', maxNb: 'u32', predator: 'vec3f', predR: 'f32', predF: 'f32' },
      code: /* wgsl */`
        let p = S[i].xyz;
        let v = SV[i].xyz;
        var sep = vec3f(0.0); var ali = vec3f(0.0); var coh = vec3f(0.0);
        var cnt = 0u;
        ${SpatialGrid.neighbors('p', /* wgsl */`
            if (j != i && cnt < u.maxNb) {
              let d = S[j].xyz - p;
              let d2 = dot(d, d);
              if (d2 < u.r2) {
                sep -= d / max(d2, 1e-5);
                ali += SV[j].xyz;
                coh += d;
                cnt++;
              }
            }`)}
        var acc = vec3f(0.0);
        if (cnt > 0u) {
          let inv = 1.0 / f32(cnt);
          acc += u.wSep * sep * inv * u.r2 * 4.0;
          acc += u.wAli * (ali * inv - v);
          acc += u.wCoh * coh * inv * (4.0 / sqrt(u.r2));
        }
        // soft walls: steer back when close to the cube faces
        let edge = u.half * 0.85;
        acc += u.wall * (max(vec3f(0.0), -edge - p) - max(vec3f(0.0), p - edge)) * 8.0;
        // predator (pointer) repulsion
        let dp = p - u.predator;
        let dpl = length(dp);
        if (dpl < u.predR) { acc += u.predF * dp / max(dpl, 1e-3) * (1.0 - dpl / u.predR); }
        var nv = v + acc * u.dt;
        let sp = length(nv);
        nv = nv * clamp(sp, u.minSpeed, u.maxSpeed) / max(sp, 1e-6);
        pos[i] = vec4f(clamp(p + nv * u.dt, vec3f(-u.half), vec3f(u.half)), 1.0);
        vel[i] = vec4f(nv, 0.0);`,
    });
  }

  /** Scatter boids uniformly in the cube with random velocities. */
  randomize(seed = 1) {
    let s = seed >>> 0 || 1;
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
    const n = this.count, p = new Float32Array(n * 4), v = new Float32Array(n * 4), h = this.half * 0.8;
    for (let k = 0; k < n; k++) {
      p.set([(rnd() * 2 - 1) * h, (rnd() * 2 - 1) * h, (rnd() * 2 - 1) * h, 1], 4 * k);
      const z = rnd() * 2 - 1, a = rnd() * 6.283, r = Math.sqrt(1 - z * z), sp = this.params.minSpeed;
      v.set([r * Math.cos(a) * sp, r * Math.sin(a) * sp, z * sp, 0], 4 * k);
    }
    this.pos.write(p); this.vel.write(v);
    return this;
  }

  step(dt = 1 / 60) {
    const P = this.params, g = this.grid;
    this.ctx.region('boids', () => {
      g.build(this.pos, this.count);
      g.gather(this.pos, this.sPos);
      g.gather(this.vel, this.sVel);
      this.updateK.run(this.count, {
        ...g.values(), dt: Math.min(dt, 1 / 30), r2: P.radius * P.radius, maxSpeed: P.maxSpeed, minSpeed: P.minSpeed,
        wSep: P.separation, wAli: P.alignment, wCoh: P.cohesion, half: this.half, wall: P.wall, maxNb: P.maxNeighbors,
        predator: P.predator, predR: P.predatorRadius, predF: P.predatorForce,
      }, { S: this.sPos, SV: this.sVel, start: g.start, pos: this.pos, vel: this.vel });
    });
  }
}
