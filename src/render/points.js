// Additive glow sprites for very large particle counts (galaxies, sparks,
// curl-noise flows). Each particle is a camera-facing quad with a Gaussian
// falloff, blended additively into the HDR target (no depth write), so millions
// of overlapping points integrate into smooth light — pair with View bloom.
import { RenderProgram } from './program.js';

export class Points {
  /**
   * o: data (vec4f GArray), stride, position, velocity (-1 = none), count, size (world units),
   *    pixelSize (min size in px), color [r,g,b] (linear HDR), colorBy 'constant'|'speed'|'w', speedScale, intensity
   */
  constructor(ctx, o) {
    this.o = o;
    const stride = o.stride || 1, pos = o.position || 0, vel = o.velocity ?? -1;
    this.program = new RenderProgram(ctx, {
      name: 'points', imports: ['color'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { size: 'f32', minPixels: 'f32', intensity: 'f32', speedScale: 'f32', color: 'vec3f', colorBy: 'u32', scale: 'f32', offset: 'vec3f' },
      targets: [{ format: 'rgba16float', blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
      depth: { write: false, compare: 'less' },
      topology: 'triangle-strip',
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) @interpolate(flat) col: vec3f }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let p = P[ii * ${stride}u + ${pos}u];
          let world = p.xyz * u.scale + u.offset;
          let clip = cam.viewProj * vec4f(world, 1.0);
          let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
          // world-space size, but never below minPixels so far points stay visible
          let pxPerUnit = cam.viewport.y / (clip.w * cam.pixelAngle * cam.viewport.y);
          let sizePx = max(u.size * pxPerUnit, u.minPixels);
          let k = sizePx / max(u.size * pxPerUnit, 1e-6);   // >1 when clamped: dim to conserve energy
          let ofs = corner * sizePx / cam.viewport * 2.0 * clip.w;
          var col = u.color;
          ${vel >= 0 ? `if (u.colorBy == 1u) { let v = P[ii * ${stride}u + ${vel}u].xyz; let c = turbo(0.15 + 0.8 * clamp(length(v) * u.speedScale, 0.0, 1.0)); col = c * c * 1.5; }` : ''}
          if (u.colorBy == 2u) { col = turbo(clamp(p.w, 0.0, 1.0)); col = col * col * 1.5; }
          col *= u.intensity / (k * k);
          return VO(vec4f(clip.xy + ofs, clip.zw), corner, col);
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let r2 = dot(in.q, in.q);
          if (r2 > 1.0) { discard; }
          let g = exp(-r2 * 4.0) - 0.0183;
          return vec4f(in.col * g, 0.0);
        }`,
    });
    this.values = { size: o.size ?? 0.01, minPixels: o.pixelSize ?? 1.5, intensity: o.intensity ?? 1, speedScale: o.speedScale ?? 1,
      color: o.color || [1, 0.8, 0.5], colorBy: { constant: 0, speed: 1, w: 2 }[o.colorBy || 'constant'], scale: o.scale ?? 1, offset: o.offset || [0, 0, 0] };
    this.program.set(this.values).bind({ P: o.data });
  }
  set(v) { this.program.set(v); return this; }
  bind(r) { this.program.bind(r); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.data.length; }
  draw(pass, view) { const n = this.count; if (n > 0) this.program.draw(pass, view, 4, n); }
}
