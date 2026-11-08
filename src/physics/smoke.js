// Eulerian smoke / fire on a staggered MAC grid (Harlow & Welch 1965; Stam
// 1999; Fedkiw, Stam & Jensen 2001; Bridson, "Fluid Simulation for Computer
// Graphics").
//
// Why staggered: on a collocated grid, central-difference divergence and
// gradient compose into a *wide* Laplacian, while the cheap 7-point Poisson
// solve uses the compact one — the mismatch leaves ~40% of the divergence
// behind (measured). On a MAC grid velocity components live on cell faces,
// div∘grad is exactly the compact 7-point Laplacian, and the projection is exact
// up to the solver tolerance.
//
// Storage (3D rgba16float textures, hardware trilinear filtering):
//   vel texel (i,j,k) = (u at the -x face, v at the -y face, w at the -z face) of cell (i,j,k)
//   scal texel (i,j,k) = (density, temperature) at the cell centre
// Per step: advect faces and scalars (semi-Lagrangian) -> forces (vorticity
// confinement, buoyancy, sources) -> divergence -> red–black SOR -> projection.

export class Smoke {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.size = o.size || [1, 1.5, 1];
    const res = o.res || 64;
    const dx = Math.max(...this.size) / res;
    this.dx = dx;
    this.dims = this.size.map((s) => Math.max(8, Math.round(s / dx)));
    this.size = this.dims.map((d) => d * dx);
    this.origin = o.origin || [-this.size[0] / 2, 0, -this.size[2] / 2];
    this.iterations = o.iterations ?? 24;
    this.omega = o.omega ?? 1.8;
    this.vorticity = o.vorticity ?? 0.35;
    this.buoyancy = o.buoyancy ?? 1.6;
    this.weight = o.weight ?? 0.08;
    this.cooling = o.cooling ?? 0.9;
    this.dissipation = o.dissipation ?? 0.15;
    this.sources = o.sources || [{ center: [0, 0.12, 0], radius: 0.09, density: 3, temperature: 2, velocity: [0, 0.6, 0] }];
    this.time = 0;
    const TU = GPUTextureUsage;
    const usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_DST | TU.COPY_SRC;
    const tex = (format, label) => ctx.texture({ size: this.dims, format, usage, label, dimension: '3d' });
    this.velA = tex('rgba16float', 'smoke-vel-a');
    this.velB = tex('rgba16float', 'smoke-vel-b');
    this.scalA = tex('rgba16float', 'smoke-scal-a');
    this.scalB = tex('rgba16float', 'smoke-scal-b');
    this.curl = tex('rgba16float', 'smoke-curl');
    this.div = tex('r32float', 'smoke-div');
    this.pressure = tex('r32float', 'smoke-p');
    this.srcBuf = ctx.array('vec4f', 8 * 3, { label: 'smoke-sources' });
    this._kernels();
  }

  _kernels() {
    const ctx = this.ctx;
    const wg = [4, 4, 4];
    const W3 = { storage: 'rgba16float', dim: '3d' };
    const S3 = { sampleType: 'float', dim: '3d' };
    const U3 = { sampleType: 'unfilterable-float', dim: '3d' };
    // Sampling helpers in *grid coordinates* g (cell (i,j,k) spans [i, i+1]^3).
    // A face-centred component is sampled by shifting into its own lattice.
    const mac = /* wgsl */`
      fn dimsF() -> vec3f { return vec3f(u.size); }
      fn sU(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.5, 0.0, 0.0)) / dimsF(), 0.0).x; }
      fn sV(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.0, 0.5, 0.0)) / dimsF(), 0.0).y; }
      fn sW(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.0, 0.0, 0.5)) / dimsF(), 0.0).z; }
      fn sVel(vel: texture_3d<f32>, g: vec3f) -> vec3f { return vec3f(sU(vel, g), sV(vel, g), sW(vel, g)); }`;
    const common = { workgroupSize: wg, samplers: { lin: 'linear' } };
    this.advectVelK = ctx.kernel({
      ...common, name: 'smoke-advect-vel',
      textures: { vel: S3, dst: W3 }, uniforms: { dtg: 'f32', decay: 'f32' }, shared: mac,
      code: /* wgsl */`
        let c = vec3f(p);
        // face positions in grid coordinates
        let gu = c + vec3f(0.0, 0.5, 0.5);
        let gv = c + vec3f(0.5, 0.0, 0.5);
        let gw = c + vec3f(0.5, 0.5, 0.0);
        let nu = sU(vel, gu - u.dtg * sVel(vel, gu));
        let nv = sV(vel, gv - u.dtg * sVel(vel, gv));
        let nw = sW(vel, gw - u.dtg * sVel(vel, gw));
        textureStore(dst, p, vec4f(vec3f(nu, nv, nw) * u.decay, 0.0));`,
    });
    this.advectScalK = ctx.kernel({
      ...common, name: 'smoke-advect-scalar',
      textures: { vel: S3, src: S3, dst: W3 }, uniforms: { dtg: 'f32', decay: 'vec4f' }, shared: mac,
      code: /* wgsl */`
        let g = vec3f(p) + 0.5;
        let back = g - u.dtg * sVel(vel, g);
        textureStore(dst, p, textureSampleLevel(src, lin, back / dimsF(), 0.0) * u.decay);`,
    });
    this.curlK = ctx.kernel({
      ...common, name: 'smoke-curl', textures: { vel: S3, dst: W3 }, uniforms: { invDx: 'f32' }, shared: mac,
      code: /* wgsl */`
        let g = vec3f(p) + 0.5;
        let ex = vec3f(1.0, 0.0, 0.0); let ey = vec3f(0.0, 1.0, 0.0); let ez = vec3f(0.0, 0.0, 1.0);
        let dwdy = sW(vel, g + ey) - sW(vel, g - ey); let dvdz = sV(vel, g + ez) - sV(vel, g - ez);
        let dudz = sU(vel, g + ez) - sU(vel, g - ez); let dwdx = sW(vel, g + ex) - sW(vel, g - ex);
        let dvdx = sV(vel, g + ex) - sV(vel, g - ex); let dudy = sU(vel, g + ey) - sU(vel, g - ey);
        let w = 0.5 * u.invDx * vec3f(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
        textureStore(dst, p, vec4f(w, length(w)));`,
    });
    this._forceKernel();
    this.divK = ctx.kernel({
      ...common, name: 'smoke-div', textures: { vel: U3, dst: { storage: 'r32float', dim: '3d' } }, uniforms: { invDx: 'f32' },
      code: /* wgsl */`
        let c = vec3i(p);
        let m = vec3i(u.size) - 1;
        let f = textureLoad(vel, c, 0).xyz;           // -x, -y, -z faces
        let ux = select(textureLoad(vel, c + vec3i(1, 0, 0), 0).x, 0.0, c.x == m.x);
        let vy = select(textureLoad(vel, c + vec3i(0, 1, 0), 0).y, 0.0, c.y == m.y);
        let wz = select(textureLoad(vel, c + vec3i(0, 0, 1), 0).z, 0.0, c.z == m.z);
        let fx = select(f.x, 0.0, c.x == 0); let fy = select(f.y, 0.0, c.y == 0); let fz = select(f.z, 0.0, c.z == 0);
        textureStore(dst, c, vec4f(u.invDx * ((ux - fx) + (vy - fy) + (wz - fz)), 0.0, 0.0, 0.0));`,
    });
    // Red–black SOR on the compact Laplacian with Neumann walls: a wall face removes
    // that neighbour from the stencil (the diagonal becomes the number of open faces).
    this.sorK = ctx.kernel({
      ...common, name: 'smoke-sor',
      textures: { P: { storage: 'r32float', dim: '3d', access: 'read_write' }, div: U3 },
      uniforms: { parity: 'u32', omega: 'f32', h2: 'f32' },
      code: /* wgsl */`
        if (((p.x + p.y + p.z) & 1u) != u.parity) { return; }
        let c = vec3i(p);
        let m = vec3i(u.size) - 1;
        var s = 0.0; var nb = 0.0;
        if (c.x > 0) { s += textureLoad(P, c - vec3i(1, 0, 0)).x; nb += 1.0; }
        if (c.x < m.x) { s += textureLoad(P, c + vec3i(1, 0, 0)).x; nb += 1.0; }
        if (c.y > 0) { s += textureLoad(P, c - vec3i(0, 1, 0)).x; nb += 1.0; }
        if (c.y < m.y) { s += textureLoad(P, c + vec3i(0, 1, 0)).x; nb += 1.0; }
        if (c.z > 0) { s += textureLoad(P, c - vec3i(0, 0, 1)).x; nb += 1.0; }
        if (c.z < m.z) { s += textureLoad(P, c + vec3i(0, 0, 1)).x; nb += 1.0; }
        let pc = textureLoad(P, c).x;
        let gs = (s - u.h2 * textureLoad(div, c, 0).x) / nb;
        textureStore(P, c, vec4f(mix(pc, gs, u.omega), 0.0, 0.0, 0.0));`,
    });
    this.projectK = ctx.kernel({
      ...common, name: 'smoke-project',
      textures: { vel: U3, P: U3, dst: W3 }, uniforms: { invDx: 'f32' },
      code: /* wgsl */`
        let c = vec3i(p);
        let pc = textureLoad(P, c, 0).x;
        var v = textureLoad(vel, c, 0).xyz;
        // face between cell c-1 and c: subtract the pressure difference across it
        if (c.x > 0) { v.x -= u.invDx * (pc - textureLoad(P, c - vec3i(1, 0, 0), 0).x); } else { v.x = 0.0; }
        if (c.y > 0) { v.y -= u.invDx * (pc - textureLoad(P, c - vec3i(0, 1, 0), 0).x); } else { v.y = 0.0; }
        if (c.z > 0) { v.z -= u.invDx * (pc - textureLoad(P, c - vec3i(0, 0, 1), 0).x); } else { v.z = 0.0; }
        textureStore(dst, c, vec4f(v, 0.0));`,
    });
  }

  /** Vorticity confinement + buoyancy + sources, evaluated on the faces each component lives on. */
  _forceKernel() {
    const ctx = this.ctx;
    const S3 = { sampleType: 'float', dim: '3d' };
    const U3 = { sampleType: 'unfilterable-float', dim: '3d' };
    const W3 = { storage: 'rgba16float', dim: '3d' };
    this.forceK = ctx.kernel({
      workgroupSize: [4, 4, 4], samplers: { lin: 'linear' }, name: 'smoke-forces', imports: ['noise'],
      textures: { vel: U3, curl: S3, scal: S3, velOut: W3, scalOut: W3 },
      buffers: { SRC: { type: 'vec4f', access: 'read' } },
      uniforms: { dt: 'f32', dx: 'f32', eps: 'f32', buoy: 'f32', weight: 'f32', origin: 'vec3f', nsrc: 'u32', cooling: 'f32', time: 'f32' },
      shared: /* wgsl */`
        fn dimsF() -> vec3f { return vec3f(u.size); }
        fn wmag(g: vec3f, D: vec3f) -> f32 { return textureSampleLevel(curl, lin, g / D, 0.0).w; }
        // N x ω with N = ∇|ω| / |∇|ω|| (Fedkiw et al. 2001), sampled at grid position g
        fn confAt(g: vec3f, D: vec3f) -> vec3f {
          let gr = vec3f(wmag(g + vec3f(1.0, 0.0, 0.0), D) - wmag(g - vec3f(1.0, 0.0, 0.0), D),
                         wmag(g + vec3f(0.0, 1.0, 0.0), D) - wmag(g - vec3f(0.0, 1.0, 0.0), D),
                         wmag(g + vec3f(0.0, 0.0, 1.0), D) - wmag(g - vec3f(0.0, 0.0, 1.0), D));
          let l = length(gr);
          if (l < 1e-6) { return vec3f(0.0); }
          return cross(gr / l, textureSampleLevel(curl, lin, g / D, 0.0).xyz);
        }`,
      code: /* wgsl */`
        let c = vec3i(p);
        let cf = vec3f(p);
        let D = dimsF();
        var v = textureLoad(vel, c, 0).xyz;
        let gu = cf + vec3f(0.0, 0.5, 0.5); let gv = cf + vec3f(0.5, 0.0, 0.5); let gw = cf + vec3f(0.5, 0.5, 0.0);
        v += u.dt * u.eps * u.dx * vec3f(confAt(gu, D).x, confAt(gv, D).y, confAt(gw, D).z);
        let sv = textureSampleLevel(scal, lin, gv / D, 0.0);
        v.y += u.dt * (u.buoy * sv.y - u.weight * sv.x);
        var s = textureLoad(scal, c, 0);
        let pw = u.origin + (cf + 0.5) * u.dx;
        for (var k = 0u; k < u.nsrc; k++) {
          let a = SRC[k * 3u]; let b = SRC[k * 3u + 1u]; let svel = SRC[k * 3u + 2u];
          let d = length(pw - a.xyz) / a.w;
          if (d < 1.0) {
            let f = (1.0 - d * d) * (0.75 + 0.5 * simplex3(pw * 9.0 + vec3f(0.0, -u.time * 2.0, u.time)));
            s.x += b.x * f * u.dt;
            s.y = max(s.y, b.y * f);
            v = mix(v, svel.xyz, clamp(f * 40.0 * u.dt, 0.0, 1.0));
          }
        }
        s.y *= exp(-u.cooling * u.dt);
        if (c.x == 0) { v.x = 0.0; }
        if (c.y == 0) { v.y = 0.0; }
        if (c.z == 0) { v.z = 0.0; }
        textureStore(velOut, c, vec4f(v, 0.0));
        textureStore(scalOut, c, s);`,
    });
  }

  _uploadSources() {
    const d = new Float32Array(8 * 12);
    this.sources.slice(0, 8).forEach((s, k) => {
      d.set([...s.center, s.radius ?? 0.1, s.density ?? 2, s.temperature ?? 1, 0, 0, ...(s.velocity || [0, 0, 0]), 0], k * 12);
    });
    this.srcBuf.write(d);
  }

  /** Clear all fields. */
  reset() {
    for (const t of [this.velA, this.velB, this.scalA, this.scalB, this.curl, this.div, this.pressure]) {
      const bytes = t.format === 'r32float' ? 4 : 8;
      const zero = new Uint8Array(t.width * t.height * bytes);
      for (let z = 0; z < t.depth; z++)
        this.ctx.device.queue.writeTexture({ texture: t.texture, origin: [0, 0, z] }, zero, { bytesPerRow: t.width * bytes, rowsPerImage: t.height }, [t.width, t.height, 1]);
    }
  }

  step(dt = 1 / 60) {
    dt = Math.min(dt, 1 / 30);
    this.time += dt;
    this._uploadSources();
    const D = this.dims, dx = this.dx;
    const dtg = dt / dx; // velocity (world/s) -> grid cells per step
    const dec = Math.exp(-this.dissipation * dt);
    this.ctx.region('smoke', () => {
      this.advectVelK.run(D, { dtg, decay: 0.999 }, { vel: this.velA, dst: this.velB });
      this.advectScalK.run(D, { dtg, decay: [dec, 1, 1, 1] }, { vel: this.velA, src: this.scalA, dst: this.scalB });
      this.curlK.run(D, { invDx: 1 / dx }, { vel: this.velB, dst: this.curl });
      this.forceK.run(D, { dt, dx, eps: this.vorticity * 10, buoy: this.buoyancy, weight: this.weight, origin: this.origin,
        nsrc: Math.min(8, this.sources.length), cooling: this.cooling, time: this.time },
      { vel: this.velB, curl: this.curl, scal: this.scalB, velOut: this.velA, scalOut: this.scalA, SRC: this.srcBuf });
      this.divK.run(D, { invDx: 1 / dx }, { vel: this.velA, dst: this.div });
      for (let k = 0; k < this.iterations; k++) {
        this.sorK.run(D, { parity: 0, omega: this.omega, h2: dx * dx }, { P: this.pressure, div: this.div });
        this.sorK.run(D, { parity: 1, omega: this.omega, h2: dx * dx }, { P: this.pressure, div: this.div });
      }
      this.projectK.run(D, { invDx: 1 / dx }, { vel: this.velA, P: this.pressure, dst: this.velB });
      [this.velA, this.velB] = [this.velB, this.velA];
    });
  }
}
