// Ray-cast sphere impostors for particle buffers. Each particle is a camera-
// facing quad; the fragment shader intersects the exact sphere, so silhouettes,
// depth and normals are pixel-perfect at 4 vertices per particle.
//
// The source is any vec4f storage buffer where particle k's position lives at
// vec4 index k*stride + position (xyz, optionally scaled/offset into world space).
import { RenderProgram } from './program.js';

export class Spheres {
  /**
   * @param {object} o
   *   data: GArray (vec4f layout), stride (vec4s per particle, default 1), position (vec4 index, default 0)
   *   velocity: vec4 index of velocity or -1;  material: WGSL expr of `px: vec4f` giving a u32 palette index
   *   count: number | () => number;  radius; scale; offset; palette: [[r,g,b], ...] (linear)
   *   colorBy: 'material' | 'speed' | 'height';  speedScale
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.o = o;
    this.palette = (o.palette || [[0.1, 0.45, 0.9], [0.9, 0.35, 0.25], [0.95, 0.95, 0.97], [0.85, 0.7, 0.4], [0.3, 0.8, 0.4], [0.7, 0.4, 0.9], [0.9, 0.8, 0.2], [0.5, 0.5, 0.55]]).slice(0, 8);
    while (this.palette.length < 8) this.palette.push([0.7, 0.7, 0.7]);
    const stride = o.stride || 1;
    const pos = o.position || 0;
    const vel = o.velocity ?? -1;
    const matExpr = o.material || '0u';
    const colorBy = { material: 0, speed: 1, height: 2 }[o.colorBy || 'material'];
    const mask = o.mask || 'true';
    this.program = new RenderProgram(ctx, {
      name: 'spheres',
      imports: ['pbr', 'color'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { radius: 'f32', scale: 'f32', offset: 'vec3f', colorBy: 'u32', speedScale: 'f32', velScale: 'f32',
        pal0: 'vec4f', pal1: 'vec4f', pal2: 'vec4f', pal3: 'vec4f', pal4: 'vec4f', pal5: 'vec4f', pal6: 'vec4f', pal7: 'vec4f',
        sun: 'vec3f', ambient: 'f32', matMask: 'u32' },
      code: /* wgsl */`
        struct VO {
          @builtin(position) pos: vec4f,
          @location(0) center: vec3f,       // view space
          @location(1) quad: vec3f,         // view-space point on the impostor plane
          @location(2) @interpolate(flat) color: vec3f,
        }
        fn paletteColor(k: u32) -> vec3f {
          switch k & 7u {
            case 0u: { return u.pal0.rgb; } case 1u: { return u.pal1.rgb; } case 2u: { return u.pal2.rgb; }
            case 3u: { return u.pal3.rgb; } case 4u: { return u.pal4.rgb; } case 5u: { return u.pal5.rgb; }
            case 6u: { return u.pal6.rgb; } default: { return u.pal7.rgb; }
          }
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let px = P[ii * ${stride}u + ${pos}u];
          let world = px.xyz * u.scale + u.offset;
          let c = (cam.view * vec4f(world, 1.0)).xyz;
          let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
          // Enlarge the quad slightly so the sphere's perspective silhouette fits.
          let q = c + vec3f(corner * u.radius * 1.5, 0.0);
          let keep = ${mask};
          var col = paletteColor(${matExpr});
          ${vel >= 0 ? `
          let v = P[ii * ${stride}u + ${vel}u].xyz * u.velScale;
          if (u.colorBy == 1u) { col = turbo(clamp(length(v) * u.speedScale, 0.0, 1.0)); col = col * col; }` : ''}
          if (u.colorBy == 2u) { col = viridis(clamp(world.y, 0.0, 1.0)); col = col * col; }
          var clip = cam.proj * vec4f(q, 1.0);
          if (!keep) { clip = vec4f(0.0, 0.0, -2.0, 1.0); }   // culled: degenerate, outside the clip volume
          return VO(clip, c, q, col);
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let d = normalize(in.quad);
          let b = dot(d, in.center);
          let h = b * b - dot(in.center, in.center) + u.radius * u.radius;
          if (h < 0.0) { discard; }
          let t = b - sqrt(h);
          let hit = d * t;
          let nv = (hit - in.center) / u.radius;
          let n = normalize((cam.invView * vec4f(nv, 0.0)).xyz);
          let v = normalize((cam.invView * vec4f(-d, 0.0)).xyz);
          let clip = cam.proj * vec4f(hit, 1.0);
          let ao = 0.6 + 0.4 * clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
          let c = shadePBR(n, v, in.color, 0.0, 0.45, u.sun, vec3f(2.6), 1.0, ao * u.ambient);
          return FO(vec4f(c, 1.0), clip.z / clip.w);
        }`,
      topology: 'triangle-strip',
    });
    this.values = { radius: o.radius ?? 0.01, scale: o.scale ?? 1, offset: o.offset || [0, 0, 0], colorBy,
      speedScale: o.speedScale ?? 0.25, velScale: o.velScale ?? 1, sun: normalize3(o.sun || [0.4, 0.8, 0.3]), ambient: o.ambient ?? 1,
      matMask: o.matMask ?? 0xffffffff };
    this.palette.forEach((c, k) => { this.values['pal' + k] = [...c, 1]; });
    this.program.set(this.values).bind({ P: o.data });
  }
  set(v) { Object.assign(this.values, v); this.program.set(v); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.data.length; }
  draw(pass, view) {
    const n = this.count;
    if (n > 0) this.program.draw(pass, view, 4, n);
  }
}

function normalize3(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
