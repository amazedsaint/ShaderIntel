// Oriented instanced glyphs (arrows / fish / darts) for agents: each instance
// is a tiny tetrahedral dart aligned with its velocity, lit with the PBR module.
// 12 vertices per instance (4 triangles), generated in the vertex shader.
import { RenderProgram } from './program.js';

export class Glyphs {
  /** o: pos (vec4f GArray), vel (vec4f GArray), count, size, color, colorBy 'speed'|'constant'|'heading', speedScale */
  constructor(ctx, o) {
    this.o = o;
    this.program = new RenderProgram(ctx, {
      name: 'glyphs', imports: ['pbr', 'color', 'math'],
      buffers: { P: { type: 'vec4f' }, V: { type: 'vec4f' } },
      uniforms: { size: 'f32', color: 'vec3f', colorBy: 'u32', speedScale: 'f32', sun: 'vec3f' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) n: vec3f, @location(1) col: vec3f, @location(2) w: vec3f }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          // dart: tip at +z, base triangle at z=0 (unit sized)
          var corners = array<vec3f, 4>(vec3f(0.0, 0.0, 1.6), vec3f(0.0, 0.55, 0.0), vec3f(-0.5, -0.35, 0.0), vec3f(0.5, -0.35, 0.0));
          var tris = array<u32, 12>(0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u);
          let p = P[ii].xyz;
          let v = V[ii].xyz;
          let f = safeNormalize(v + vec3f(1e-6, 0.0, 0.0));
          let side = safeNormalize(cross(vec3f(0.0, 1.0, 0.0), f) + vec3f(0.0, 0.0, 1e-6));
          let up = cross(f, side);
          let tri = vi / 3u;
          let a = corners[tris[tri * 3u]]; let b = corners[tris[tri * 3u + 1u]]; let c = corners[tris[tri * 3u + 2u]];
          let lc = corners[tris[vi]];
          var ln = normalize(cross(b - a, c - a));
          if (dot(ln, (a + b + c) / 3.0 - vec3f(0.0, -0.04, 0.4)) < 0.0) { ln = -ln; }   // face normals point outwards
          let world = p + (side * lc.x + up * lc.y + f * lc.z) * u.size;
          let n = side * ln.x + up * ln.y + f * ln.z;
          var col = u.color;
          if (u.colorBy == 1u) { let t = turbo(clamp(length(v) * u.speedScale, 0.0, 1.0)); col = t * t; }
          if (u.colorBy == 2u) { col = srgbToLinear(palette(atan2(f.z, f.x) / TAU + 0.5, vec3f(0.5), vec3f(0.45), vec3f(1.0), vec3f(0.0, 0.33, 0.67))); }
          return VO(cam.viewProj * vec4f(world, 1.0), n, col, world);
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let n = normalize(in.n);
          let v = normalize(cam.eye - in.w);
          return vec4f(shadePBR(n, v, in.col, 0.0, 0.5, u.sun, vec3f(2.4), 1.0, 0.9), 1.0);
        }`,
    });
    this.program.set({ size: o.size ?? 0.01, color: o.color || [0.85, 0.85, 0.9], colorBy: { constant: 0, speed: 1, heading: 2 }[o.colorBy || 'heading'],
      speedScale: o.speedScale ?? 1.5, sun: [0.42, 0.8, 0.42] });
  }
  set(v) { this.program.set(v); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.pos.length; }
  draw(pass, view) {
    if (this.o.pos.destroyed || this.o.vel.destroyed) return;
    this.program.draw(pass, view, 12, this.count, null, { P: this.o.pos, V: this.o.vel });
  }
  /** Layers own no GPU memory beyond their pipeline (garbage-collected); kept for a uniform API. */
  destroy() { this.destroyed = true; }
}
