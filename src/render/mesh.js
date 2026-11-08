// Surface mesh whose vertices live in GPU storage buffers (positions + normals
// written by a simulation), drawn with an index buffer. Two-sided PBR shading
// with separate front/back colours and an optional grid pattern from (W, H).
import { RenderProgram } from './program.js';

export class SurfaceMesh {
  /**
   * o: pos, nrm (vec4f GArrays), indices (GPUBuffer, uint32), indexCount,
   *    grid: [W, H] for UV-based patterns, front, back (linear RGB), roughness, pattern: 'checker'|'stripes'|'none'
   */
  constructor(ctx, o) {
    this.o = o;
    this.program = new RenderProgram(ctx, {
      name: 'surface-mesh', imports: ['pbr'],
      buffers: { P: { type: 'vec4f' }, N: { type: 'vec4f' } },
      uniforms: { front: 'vec3f', back: 'vec3f', roughness: 'f32', grid: 'vec2u', pattern: 'u32', sun: 'vec3f', tiles: 'f32' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let p = P[vi].xyz;
          let uv = vec2f(f32(vi % u.grid.x), f32(vi / u.grid.x)) / vec2f(max(vec2u(1u), u.grid - 1u));
          return VO(cam.viewProj * vec4f(p, 1.0), p, N[vi].xyz, uv);
        }
        @fragment fn fs(in: VO, @builtin(front_facing) front: bool) -> @location(0) vec4f {
          var n = normalize(in.n);
          let v = normalize(cam.eye - in.w);
          if (dot(n, v) < 0.0) { n = -n; }
          var albedo = select(u.back, u.front, front);
          let g = in.uv * u.tiles;
          if (u.pattern == 1u) {
            let c = (floor(g.x) + floor(g.y)) % 2.0;
            albedo *= 0.75 + 0.25 * c;
          } else if (u.pattern == 2u) {
            albedo *= 0.8 + 0.2 * step(0.5, fract(g.x));
          }
          // wrapped diffuse approximates light scattering through thin fabric
          let c = shadePBR(n, v, albedo, 0.0, u.roughness, u.sun, vec3f(2.8), 1.0, 1.0);
          let trans = albedo * 0.25 * max(dot(-n, u.sun), 0.0);
          return vec4f(c + trans, 1.0);
        }`,
    });
    this.program.set({ front: o.front || [0.75, 0.12, 0.1], back: o.back || [0.85, 0.75, 0.6], roughness: o.roughness ?? 0.75,
      grid: o.grid || [2, 2], pattern: { none: 0, checker: 1, stripes: 2 }[o.pattern || 'checker'], sun: [0.4, 0.75, 0.52], tiles: o.tiles ?? 12 });
  }
  set(v) { this.program.set(v); return this; }
  draw(pass, view) {
    this.program.drawIndexed(pass, view, this.o.indices, this.o.indexCount, 1, null, { P: this.o.pos, N: this.o.nrm });
  }
}
