// XPBD cloth (Macklin, Müller & Chentanez 2016) with "small steps" (Macklin et
// al. 2019): many substeps with a single constraint iteration each, which
// converges faster than many iterations of one big step and makes the
// compliance (α) behave like a real material stiffness independent of the
// iteration count.
//
// Parallel Gauss–Seidel through graph colouring: on a W×H particle grid every
// constraint family (stretch-x, stretch-y, two shear diagonals, bend-x, bend-y)
// splits into 2 colours whose constraints share no particle, so 12 dispatches
// update every constraint once with no atomics and no races.
//
// XPBD update for a distance constraint C = |x_a - x_b| - L (λ reset each substep):
//   Δλ = -C / (w_a + w_b + α/Δt²),  x_a += w_a Δλ n,  x_b -= w_b Δλ n
//
// Long-range attachments (Kim, Chentanez & Müller 2012): each particle is also
// tethered to its nearest pin by a unilateral constraint |x - x_pin| <= L_rest.
// Gauss–Seidel spreads a load only a few edges per substep, so a hanging sheet
// would otherwise sag/stretch under its own weight; one extra dispatch fixes it.

const FAMILIES = [
  // [dx, dy, kind] kind: 0 stretch, 1 shear, 2 bend
  [1, 0, 0], [0, 1, 0], [1, 1, 1], [-1, 1, 1], [2, 0, 2], [0, 2, 2],
];

export class Cloth {
  /**
   * o: width/height (particles), size: [sx, sz] world size, origin [x,y,z] (top-left corner),
   *    pin: 'top' | 'corners' | 'none' | (i, j) => bool, substeps, stretch/shear/bend compliance,
   *    gravity, damping, wind, colliders
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    const W = (this.W = o.width || 128), H = (this.H = o.height || 128);
    this.count = W * H;
    this.size = o.size || [1.6, 1.6];
    this.origin = o.origin || [-0.8, 1.6, 0];
    this.substeps = o.substeps || 16;
    this.compliance = { stretch: o.stretch ?? 0, shear: o.shear ?? 1e-6, bend: o.bend ?? 2e-4 };
    this.gravity = o.gravity || [0, -9.81, 0];
    this.damping = o.damping ?? 0.4;
    this.wind = o.wind || [0, 0, 0];
    this.windNoise = o.windNoise ?? 0.6;
    this.colliders = o.colliders || [];
    this.friction = o.friction ?? 0.4;
    this.thickness = o.thickness ?? 0.01;
    this.time = 0;
    const n = this.count;
    this.pos = ctx.array('vec4f', n, { label: 'cloth-pos', usage: GPUBufferUsage.VERTEX });
    this.prev = ctx.array('vec4f', n, { label: 'cloth-prev' });
    this.vel = ctx.array('vec4f', n, { label: 'cloth-vel' });
    this.nrm = ctx.array('vec4f', n, { label: 'cloth-normal' });
    this.colBuf = ctx.array('vec4f', 16 * 2, { label: 'cloth-colliders' });
    this.tether = ctx.array('vec2u', n, { label: 'cloth-tether' });
    this._kernels();
    this.reset(o.pin || 'top');
    // triangle index buffer for rendering
    const idx = new Uint32Array((W - 1) * (H - 1) * 6);
    let k = 0;
    for (let j = 0; j < H - 1; j++)
      for (let i = 0; i < W - 1; i++) {
        const a = j * W + i, b = a + 1, c = a + W, d = c + 1;
        idx.set([a, c, b, b, c, d], k); k += 6;
      }
    this.indices = ctx.device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    ctx.device.queue.writeBuffer(this.indices, 0, idx);
    this.indexCount = idx.length;
  }

  /** Flat sheet in the xz... actually hanging in the xy plane, with pinned particles. */
  reset(pin = this.pinMode) {
    this.pinMode = pin;
    const { W, H } = this, n = this.count;
    const p = new Float32Array(n * 4);
    const pinFn = typeof pin === 'function' ? pin
      : pin === 'top' ? (i, j) => j === 0
      : pin === 'corners' ? (i, j) => j === 0 && (i === 0 || i === W - 1)
      : pin === 'top-sparse' ? (i, j) => j === 0 && i % 8 === 0
      : () => false;
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++) {
        const k = j * W + i;
        const x = this.origin[0] + (i / (W - 1)) * this.size[0];
        // horizontal sheet when nothing is pinned (drop test), vertical curtain otherwise
        const vertical = pin !== 'none';
        const y = vertical ? this.origin[1] - (j / (H - 1)) * this.size[1] : this.origin[1];
        const z = vertical ? this.origin[2] + 0.002 * Math.sin(i * 0.7) : this.origin[2] + (j / (H - 1)) * this.size[1] - this.size[1] / 2;
        p.set([x, y, z, pinFn(i, j) ? 0 : 1], 4 * k);
      }
    this.pos.write(p); this.prev.write(p);
    this.ctx.clear(this.vel);
    // tethers: nearest pinned particle and its rest distance (static pins)
    const pins = [];
    for (let k = 0; k < n; k++) if (p[4 * k + 3] === 0) pins.push(k);
    const teth = new Uint32Array(n * 2), f32 = new Float32Array(teth.buffer);
    for (let k = 0; k < n; k++) {
      let best = -1, bd = Infinity;
      for (const q of pins) {
        const d = Math.hypot(p[4 * k] - p[4 * q], p[4 * k + 1] - p[4 * q + 1], p[4 * k + 2] - p[4 * q + 2]);
        if (d < bd) { bd = d; best = q; }
      }
      teth[2 * k] = best < 0 || best === k ? 0xffffffff : best;
      f32[2 * k + 1] = bd;
    }
    this.tether.write(teth);
    this.restX = this.size[0] / (W - 1);
    this.restY = this.size[1] / (H - 1);
  }

  _kernels() {
    const ctx = this.ctx, W = this.W, H = this.H;
    const defines = { W: `${W}u`, H: `${H}u` };
    this.predictK = ctx.kernel({
      name: 'cloth-predict', imports: ['noise'], defines,
      buffers: { pos: { type: 'vec4f' }, prev: { type: 'vec4f' }, vel: { type: 'vec4f' }, nrm: { type: 'vec4f', access: 'read' } },
      uniforms: { dt: 'f32', gravity: 'vec3f', wind: 'vec3f', windNoise: 'f32', time: 'f32', damping: 'f32', drag: 'f32' },
      code: /* wgsl */`
        let p = pos[i];
        prev[i] = p;
        if (p.w == 0.0) { return; }
        var v = vel[i].xyz;
        // aerodynamic drag on the normal component of the relative wind (cheap lift/drag model)
        let gust = u.wind * (1.0 + u.windNoise * simplex3(p.xyz * 1.5 + vec3f(u.time * 0.7, 0.0, u.time * 0.4)));
        let n = nrm[i].xyz;
        let rel = gust - v;
        let a = u.gravity + n * dot(rel, n) * u.drag;
        v = (v + a * u.dt) * (1.0 - u.damping * u.dt);
        pos[i] = vec4f(p.xyz + v * u.dt, p.w);`,
    });
    // One kernel solves one colour of one constraint family; (dx, dy) is the
    // particle offset, the colour selects every other constraint along the
    // direction so no two threads touch the same particle.
    this.solveK = ctx.kernel({
      name: 'cloth-solve', defines,
      buffers: { pos: { type: 'vec4f' } },
      uniforms: { dx: 'i32', dy: 'i32', color: 'u32', rest: 'f32', alpha: 'f32', ncx: 'u32', ncy: 'u32' },
      code: /* wgsl */`
        // enumerate constraints of this colour: start particles (si, sj)
        let cx = i % u.ncx;
        let cy = i / u.ncx;
        if (cy >= u.ncy) { return; }
        // colour along the dominant direction: blocks of |d| particles alternate
        var si: i32; var sj: i32;
        let adx = abs(u.dx); let ady = abs(u.dy);
        if (ady == 0) {          // horizontal: x = block * 2|d| + color*|d| + r
          let blk = cx / u32(adx); let r = cx % u32(adx);
          si = i32(blk * u32(2 * adx) + u.color * u32(adx) + r); sj = i32(cy);
        } else {                 // vertical / diagonal: rows alternate
          let blk = cy / u32(ady); let r = cy % u32(ady);
          sj = i32(blk * u32(2 * ady) + u.color * u32(ady) + r); si = i32(cx) + select(0, -u.dx, u.dx < 0);
        }
        let ti = si + u.dx; let tj = sj + u.dy;
        if (si < 0 || sj < 0 || ti < 0 || tj < 0 || si >= i32(W) || ti >= i32(W) || sj >= i32(H) || tj >= i32(H)) { return; }
        let a = u32(sj) * W + u32(si);
        let b = u32(tj) * W + u32(ti);
        let pa = pos[a]; let pb = pos[b];
        let wsum = pa.w + pb.w;
        if (wsum == 0.0) { return; }
        let d = pa.xyz - pb.xyz;
        let len = length(d);
        if (len < 1e-9) { return; }
        let n = d / len;
        let C = len - u.rest;
        let dl = -C / (wsum + u.alpha);
        pos[a] = vec4f(pa.xyz + pa.w * dl * n, pa.w);
        pos[b] = vec4f(pb.xyz - pb.w * dl * n, pb.w);`,
    });
    this.collideK = ctx.kernel({
      name: 'cloth-collide', defines,
      buffers: { pos: { type: 'vec4f' }, prev: { type: 'vec4f', access: 'read' }, vel: { type: 'vec4f' }, COL: { type: 'vec4f', access: 'read' } },
      uniforms: { numColliders: 'u32', thickness: 'f32', friction: 'f32', dt: 'f32', floorY: 'f32' },
      code: /* wgsl */`
        var p = pos[i];
        if (p.w == 0.0) { vel[i] = vec4f(0.0); return; }
        var x = p.xyz;
        let x0 = prev[i].xyz;
        for (var k = 0u; k < u.numColliders; k++) {
          let c = COL[k * 2u]; let s = COL[k * 2u + 1u];
          if (s.w < 0.5) {        // sphere: c.xyz centre, c.w radius
            let d = x - c.xyz;
            let l = length(d);
            let r = c.w + u.thickness;
            if (l < r) {
              let n = d / max(l, 1e-6);
              x = c.xyz + n * r;
              // friction: remove part of the tangential displacement this substep
              let dx = x - x0;
              let tang = dx - dot(dx, n) * n;
              x -= tang * min(u.friction, 1.0);
            }
          } else {                 // axis-aligned box: c.xyz centre, s.xyz half-extent
            let q = x - c.xyz;
            let h = s.xyz + u.thickness;
            if (all(abs(q) < h)) {
              let pen = h - abs(q);
              if (pen.x < pen.y && pen.x < pen.z) { x.x = c.x + sign(q.x) * h.x; }
              else if (pen.y < pen.z) { x.y = c.y + sign(q.y) * h.y; }
              else { x.z = c.z + sign(q.z) * h.z; }
              let dx = x - x0;
              x -= vec3f(dx.x, 0.0, dx.z) * min(u.friction, 1.0) * select(0.0, 1.0, pen.y < pen.x && pen.y < pen.z);
            }
          }
        }
        if (x.y < u.floorY + u.thickness) {
          x.y = u.floorY + u.thickness;
          let dx = x - x0;
          x.x -= dx.x * u.friction; x.z -= dx.z * u.friction;
        }
        pos[i] = vec4f(x, p.w);
        vel[i] = vec4f((x - x0) / u.dt, 0.0);`,
    });
    this.tetherK = ctx.kernel({
      name: 'cloth-tether', buffers: { pos: { type: 'vec4f' }, T: { type: 'vec2u', access: 'read' } }, uniforms: { slack: 'f32' },
      code: /* wgsl */`
        let t = T[i];
        if (t.x == 0xffffffffu) { return; }
        let p = pos[i];
        if (p.w == 0.0) { return; }
        let a = pos[t.x].xyz;
        let L = bitcast<f32>(t.y) * u.slack;
        let d = p.xyz - a;
        let l = length(d);
        if (l > L) { pos[i] = vec4f(a + d * (L / l), p.w); }`,
    });
    this.normalK = ctx.kernel({
      name: 'cloth-normals', defines, buffers: { pos: { type: 'vec4f', access: 'read' }, nrm: { type: 'vec4f' } },
      code: /* wgsl */`
        let ii = i32(i % W); let jj = i32(i / W);
        let l = pos[u32(jj) * W + u32(max(ii - 1, 0))].xyz;
        let r = pos[u32(jj) * W + u32(min(ii + 1, i32(W) - 1))].xyz;
        let d = pos[u32(max(jj - 1, 0)) * W + u32(ii)].xyz;
        let t = pos[u32(min(jj + 1, i32(H) - 1)) * W + u32(ii)].xyz;
        nrm[i] = vec4f(normalize(cross(t - d, r - l) + vec3f(0.0, 1e-9, 0.0)), 0.0);`,
    });
  }

  _uploadColliders() {
    const data = new Float32Array(16 * 8);
    this.colliders.slice(0, 16).forEach((c, k) => {
      if (c.type === 'box') data.set([...c.center, 0, ...(c.half || [0.1, 0.1, 0.1]), 1], k * 8);
      else data.set([...c.center, c.radius ?? 0.2, 0, 0, 0, 0], k * 8);
    });
    this.colBuf.write(data);
  }

  /** Advance one frame of `frameDt` seconds. */
  step(frameDt = 1 / 60) {
    const ctx = this.ctx, W = this.W, H = this.H;
    this._uploadColliders();
    const dt = Math.min(frameDt, 1 / 30) / this.substeps;
    const dt2 = dt * dt;
    ctx.region('cloth', () => {
      for (let s = 0; s < this.substeps; s++) {
        this.time += dt;
        this.predictK.run(this.count, { dt, gravity: this.gravity, wind: this.wind, windNoise: this.windNoise, time: this.time,
          damping: this.damping, drag: 3.0 }, { pos: this.pos, prev: this.prev, vel: this.vel, nrm: this.nrm });
        for (const [dx, dy, kind] of FAMILIES) {
          const rest = Math.hypot(dx * this.restX, dy * this.restY);
          const alpha = [this.compliance.stretch, this.compliance.shear, this.compliance.bend][kind] / dt2;
          const adx = Math.abs(dx), ady = Math.abs(dy);
          for (let color = 0; color < 2; color++) {
            // number of constraint start particles per colour (upper bounds; the kernel discards out-of-range)
            let ncx, ncy;
            if (ady === 0) { ncx = Math.ceil(W / (2 * adx)) * adx; ncy = H; }
            else { ncx = W; ncy = Math.ceil(H / (2 * ady)) * ady; }
            this.solveK.run(ncx * ncy, { dx, dy, color, rest, alpha, ncx, ncy }, { pos: this.pos });
          }
        }
        if (this.tethers !== false) this.tetherK.run(this.count, { slack: 1.005 }, { pos: this.pos, T: this.tether });
        this.collideK.run(this.count, { numColliders: Math.min(16, this.colliders.length), thickness: this.thickness,
          friction: this.friction, dt, floorY: this.floorY ?? -10 }, { pos: this.pos, prev: this.prev, vel: this.vel, COL: this.colBuf });
      }
      this.normalK.run(this.count, null, { pos: this.pos, nrm: this.nrm });
    });
  }
}
