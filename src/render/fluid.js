// Screen-space fluid rendering (van der Laan, Green & Sainz 2009) with a
// narrow-range depth filter (after Truong & Yuksel, I3D 2018).
//
//   1. depth pass     particles -> nearest linear eye depth (r32float, own z-buffer)
//   2. thickness pass particles -> additive chord-length splats (r16float), occluded by the scene
//   3. filter         separable narrow-range filter: samples far *behind* the centre are
//                     dropped (another surface), samples far *in front* are clamped, so
//                     silhouettes stay crisp while the surface is smoothed
//   4. composite      normals from the smoothed depth; Fresnel (Schlick, F0 = 0.02),
//                     sky reflection, refraction of the opaque scene with Beer–Lambert
//                     absorption over the thickness, sun specular; writes depth.
import { RenderProgram } from './program.js';
import { destroyAll } from '../core/dispose.js';
import { SIError } from '../core/errors.js';

export class FluidSurface {
  /**
   * o: data (vec4 GArray or () => GArray), stride, position, count, radius (world),
   *    scale, offset, mask (WGSL expr of px -> bool, which particles are fluid),
   *    absorb [r,g,b] (per world unit), filterRadius (particle radii), iterations
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.o = o;
    const stride = o.stride || 1, pos = o.position || 0;
    const mask = o.mask || 'true';
    this.values = { radius: o.radius ?? 0.01, scale: o.scale ?? 1, offset: o.offset || [0, 0, 0], absorb: o.absorb || [2.6, 0.9, 0.45],
      tint: o.tint || [0.85, 0.95, 1.0], filterRadius: o.filterRadius ?? 9, maxPixels: o.maxPixels ?? 40, sun: norm(o.sun || [0.45, 0.8, 0.3]),
      thicknessScale: o.thicknessScale ?? 1, refraction: o.refraction ?? 0.04, matMask: o.matMask ?? 0xffffffff };
    this.iterations = o.iterations ?? 3;
    const particleVS = (outs) => /* wgsl */`
      struct VO { @builtin(position) pos: vec4f, @location(0) center: vec3f, @location(1) quad: vec3f }
      @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
        let px = P[ii * ${stride}u + ${pos}u];
        let keep = ${mask};
        let world = px.xyz * u.scale + u.offset;
        let c = (cam.view * vec4f(world, 1.0)).xyz;
        let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
        let q = c + vec3f(corner * u.radius * ${outs}, 0.0);
        var clip = cam.proj * vec4f(q, 1.0);
        if (!keep) { clip = vec4f(0.0, 0.0, -2.0, 1.0); }
        return VO(clip, c, q);
      }`;
    const common = { buffers: { P: { type: 'vec4f' } }, topology: 'triangle-strip' };
    const U = { radius: 'f32', scale: 'f32', offset: 'vec3f', matMask: 'u32' };
    this.depthProg = new RenderProgram(ctx, {
      ...common, name: 'ssf-depth', uniforms: U, targets: ['r32float'], depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        ${particleVS('1.5')}
        struct FO { @location(0) z: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let d = normalize(in.quad);
          let b = dot(d, in.center);
          let h = b * b - dot(in.center, in.center) + u.radius * u.radius;
          if (h < 0.0) { discard; }
          let hit = d * (b - sqrt(h));
          let clip = cam.proj * vec4f(hit, 1.0);
          return FO(vec4f(-hit.z, 0.0, 0.0, 1.0), clip.z / clip.w);
        }`,
    });
    this.thickProg = new RenderProgram(ctx, {
      ...common, name: 'ssf-thickness', uniforms: U,
      targets: [{ format: 'r16float', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
      depth: { write: false, compare: 'less' },
      code: /* wgsl */`
        ${particleVS('1.0')}
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let q = (in.quad.xy - in.center.xy) / u.radius;
          let r2 = dot(q, q);
          if (r2 > 1.0) { discard; }
          return vec4f(2.0 * u.radius * sqrt(1.0 - r2), 0.0, 0.0, 0.0);
        }`,
    });
    // separable narrow-range filter on linear depth
    this.filterK = ctx.kernel({
      name: 'ssf-filter', workgroupSize: [8, 8],
      textures: { src: { sampleType: 'unfilterable-float' }, dst: { storage: 'r32float' } },
      uniforms: { dir: 'vec2i', radius: 'f32', pxScale: 'f32', filterRadius: 'f32', maxPixels: 'f32' },
      code: /* wgsl */`
        let c = vec2i(p.xy);
        let zc = textureLoad(src, c, 0).x;
        if (zc > 1e5) { textureStore(dst, c, vec4f(zc, 0.0, 0.0, 0.0)); return; }
        let R = i32(min(u.maxPixels, ceil(u.filterRadius * u.radius * u.pxScale / zc)));
        let sigma = max(f32(R) * 0.5, 0.5);
        let lambda = 6.0 * u.radius;   // narrow range above (behind) the centre
        let mu = u.radius;             // clamp distance in front
        var sum = zc;
        var wsum = 1.0;
        let sz = vec2i(u.size.xy) - 1;
        for (var side = -1; side <= 1; side += 2) {
          var hi = zc + lambda;
          for (var k = 1; k <= R; k++) {
            let q = clamp(c + u.dir * (k * side), vec2i(0), sz);
            var z = textureLoad(src, q, 0).x;
            if (z > 1e5 || z > hi) { continue; }     // background or a different surface behind
            z = max(z, zc - mu);                      // pull foreground samples into the narrow range
            hi = max(hi, z + lambda);                 // range extension along continuous slopes
            let w = exp(-f32(k * k) / (2.0 * sigma * sigma));
            sum += z * w; wsum += w;
          }
        }
        textureStore(dst, c, vec4f(sum / wsum, 0.0, 0.0, 0.0));`,
    });
    this.compositeProg = new RenderProgram(ctx, {
      name: 'ssf-composite', imports: ['pbr', 'color'],
      textures: { zTex: { sampleType: 'unfilterable-float' }, thick: { sampleType: 'float' }, scene: { sampleType: 'float' } },
      samplers: { lin: 'linear' },
      uniforms: { absorb: 'vec3f', tint: 'vec3f', sun: 'vec3f', thicknessScale: 'f32', refraction: 'f32' },
      depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
          return VO(vec4f(xy, 0.0, 1.0));
        }
        fn viewPos(px: vec2i) -> vec3f {
          let z = textureLoad(zTex, px, 0).x;
          let ndc = vec2f((f32(px.x) + 0.5) / cam.viewport.x * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) / cam.viewport.y * 2.0);
          let v = cam.invProj * vec4f(ndc, 1.0, 1.0);
          let dir = v.xyz / v.w;
          return dir / -dir.z * z;
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let px = vec2i(in.pos.xy);
          let z = textureLoad(zTex, px, 0).x;
          if (z > 1e5) { discard; }
          let sz = vec2i(cam.viewport) - 1;
          let pc = viewPos(px);
          let dxp = viewPos(min(px + vec2i(1, 0), sz)) - pc;
          let dxn = pc - viewPos(max(px - vec2i(1, 0), vec2i(0)));
          let dyp = viewPos(min(px + vec2i(0, 1), sz)) - pc;
          let dyn = pc - viewPos(max(px - vec2i(0, 1), vec2i(0)));
          // one-sided differences at the screen border (the clamped neighbour is the pixel itself)
          let ddx = select(dxp, dxn, (abs(dxn.z) < abs(dxp.z) && px.x > 0) || px.x >= sz.x);
          let ddy = select(dyp, dyn, (abs(dyn.z) < abs(dyp.z) && px.y > 0) || px.y >= sz.y);
          let cr = cross(ddy, ddx);
          let cl = length(cr);
          var nv = select(vec3f(0.0, 0.0, 1.0), cr / max(cl, 1e-30), cl > 1e-20);
          if (nv.z < 0.0) { nv = -nv; }
          let n = normalize((cam.invView * vec4f(nv, 0.0)).xyz);
          let wp = (cam.invView * vec4f(pc, 1.0)).xyz;
          let v = normalize(cam.eye - wp);
          let uv = in.pos.xy / cam.viewport;
          let t = textureSampleLevel(thick, lin, uv, 0.0).x * u.thicknessScale;
          // refraction: offset the background lookup along the view-space normal, scaled by thickness
          let ruv = clamp(uv + nv.xy * vec2f(1.0, -1.0) * u.refraction * min(t * 10.0, 1.5), vec2f(0.0), vec2f(1.0));
          let behind = textureSampleLevel(scene, lin, ruv, 0.0).rgb;
          let trans = exp(-u.absorb * t);
          let refr = behind * trans * u.tint + (1.0 - trans) * vec3f(0.02, 0.07, 0.12) * 0.6;   // in-scatter keeps deep water from going black
          let NoV = clamp(dot(n, v), 0.0, 1.0);
          let F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
          let refl = skyColor(reflect(-v, n), u.sun);
          let h = normalize(v + u.sun);
          let spec = vec3f(2.0, 1.85, 1.6) * pow(max(dot(n, h), 0.0), 250.0) * 1.5;
          let col = mix(refr, refl, F) + spec;
          let clip = cam.proj * vec4f(pc, 1.0);
          return FO(vec4f(col, 1.0), clip.z / clip.w);
        }`,
    });
    this.w = 0; this.h = 0;
  }

  /** Update the look (radius, absorb, tint, refraction, filterRadius, ...) or the filter iterations; unknown names throw. */
  set(v) {
    const o = { ...v };
    if ('iterations' in o) { this.iterations = Math.max(0, o.iterations | 0); delete o.iterations; }
    if (o.sun) o.sun = norm(o.sun);
    for (const k of Object.keys(o))
      if (!(k in this.values)) throw new SIError(`FluidSurface.set: unknown value "${k}" (use ${Object.keys(this.values).join(', ')}, iterations)`, 'invalid-argument');
    Object.assign(this.values, o);
    return this;
  }

  _alloc(w, h) {
    for (const t of [this.zA, this.zB, this.zDepth, this.thick, this.sceneCopy]) t?.destroy();
    const TU = GPUTextureUsage, ctx = this.ctx;
    this.zA = ctx.texture({ size: [w, h], format: 'r32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.STORAGE_BINDING, label: 'ssf-z-a' });
    this.zB = ctx.texture({ size: [w, h], format: 'r32float', usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING, label: 'ssf-z-b' });
    this.zDepth = ctx.texture({ size: [w, h], format: 'depth32float', usage: TU.RENDER_ATTACHMENT, label: 'ssf-zbuffer' });
    this.thick = ctx.texture({ size: [w, h], format: 'r16float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'ssf-thickness' });
    this.sceneCopy = ctx.texture({ size: [w, h], format: 'rgba16float', usage: TU.COPY_DST | TU.TEXTURE_BINDING, label: 'ssf-scene' });
    this.w = w; this.h = h;
  }

  get data() { const d = this.o.data; return typeof d === 'function' ? d() : d; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.data.length; }

  afterScene(view) {
    if (this.destroyed || this.data?.destroyed) return;
    const n = this.count;
    if (!n) return;
    if (view.width !== this.w || view.height !== this.h) this._alloc(view.width, view.height);
    const ctx = this.ctx, V = this.values, P = this.data;
    const uni = { radius: V.radius, scale: V.scale, offset: V.offset, matMask: V.matMask };
    // 1. depth
    let pass = ctx.renderPass({
      colorAttachments: [{ view: this.zA.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 1e6, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: this.zDepth.view, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    this.depthProg.draw(pass, view, 4, n, uni, { P });
    pass.end();
    // 2. thickness (occluded by the opaque scene)
    pass = ctx.renderPass({
      colorAttachments: [{ view: this.thick.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: view.depth.view, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    this.thickProg.draw(pass, view, 4, n, uni, { P });
    pass.end();
    // 3. narrow-range filter: (horizontal, vertical) x iterations, ping-pong zA <-> zB
    const pxScale = view.height / (2 * Math.tan(view.camera.fovY / 2));
    for (let it = 0; it < this.iterations; it++) {
      this.filterK.run([this.w, this.h], { dir: [1, 0], radius: V.radius, pxScale, filterRadius: V.filterRadius, maxPixels: V.maxPixels }, { src: this.zA, dst: this.zB });
      this.filterK.run([this.w, this.h], { dir: [0, 1], radius: V.radius, pxScale, filterRadius: V.filterRadius, maxPixels: V.maxPixels }, { src: this.zB, dst: this.zA });
    }
    // 4. composite over a copy of the opaque scene
    ctx.endPass();
    ctx.encoder.copyTextureToTexture({ texture: view.hdr.texture }, { texture: this.sceneCopy.texture }, [this.w, this.h, 1]);
    pass = ctx.renderPass({
      colorAttachments: [{ view: view.hdr.view, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: view.depth.view, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    this.compositeProg.draw(pass, view, 3, 1, { absorb: V.absorb, tint: V.tint, sun: V.sun, thicknessScale: V.thicknessScale, refraction: V.refraction },
      { zTex: this.zA, thick: this.thick, scene: this.sceneCopy });
    pass.end();
  }
  destroy() { this.destroyed = true; destroyAll(this.zA, this.zB, this.zDepth, this.thick, this.sceneCopy); this.zA = null; this.w = this.h = 0; }
}

function norm(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
