// Instanced renderer for rigid bodies stored in a RigidWorld body buffer
// (12 vec4 per body: x at 0, q at 1, size at 8, mass at 9, info at 11). Every body
// is drawn as its oriented bounding box (36 vertices); boxes shade the faces
// directly, spheres and capsules ray-cast the exact surface inside that box and
// write the true depth. Disabled bodies are not drawn.
import { RenderProgram } from './program.js';

export class Bodies {
  constructor(ctx, world, o = {}) {
    this.world = world;
    this.program = new RenderProgram(ctx, {
      name: 'rigid-bodies', imports: ['pbr', 'color', 'math', 'hash'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { sun: 'vec3f', staticColor: 'vec3f', saturation: 'f32', kinematicColor: 'vec3f', groupMask: 'u32' },
      code: /* wgsl */`
        struct VO {
          @builtin(position) pos: vec4f,
          @location(0) w: vec3f,
          @location(1) n: vec3f,
          @location(2) @interpolate(flat) col: vec3f,
          @location(3) @interpolate(flat) c: vec4f,     // centre, radius
          @location(4) @interpolate(flat) q: vec4f,     // orientation
          @location(5) @interpolate(flat) sh: vec2f,    // (shape, half length)
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let x = P[ii * 12u];
          let q = P[ii * 12u + 1u];
          let sz = P[ii * 12u + 8u];
          let m = P[ii * 12u + 9u];
          let info = bitcast<vec4u>(P[ii * 12u + 11u]);
          let flags = info.z;
          var o: VO;
          // disabled bodies, and bodies outside the drawn collision groups, are not drawn
          if ((flags & 4u) != 0u || (info.x & u.groupMask) == 0u) { o.pos = vec4f(0.0, 0.0, -2.0, 1.0); return o; }
          let shape = u32(sz.w + 0.5);
          var ext = sz.xyz;
          if (shape == 2u) { ext = vec3f(sz.x, sz.y + sz.x, sz.x); }
          let f = vi / 6u;
          let a = f / 2u;
          let s = select(-1.0, 1.0, (f & 1u) == 0u);
          var uv = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
          let t = uv[vi % 6u];
          var lp = vec3f(0.0);
          lp[a] = s; lp[(a + 1u) % 3u] = t.x; lp[(a + 2u) % 3u] = t.y;
          var ln = vec3f(0.0); ln[a] = s;
          let world = x.xyz + qrot(q, lp * ext);
          var col = u.staticColor;
          if (m.x > 0.0) {
            let h = pcg(ii * 2654435761u + 17u);
            col = srgbToLinear(palette(u2f(h), vec3f(0.55), vec3f(0.35 * u.saturation), vec3f(1.0), vec3f(0.0, 0.33, 0.67)));
          } else if ((flags & 1u) != 0u) { col = u.kinematicColor; }
          o.pos = cam.viewProj * vec4f(world, 1.0);
          o.w = world; o.n = qrot(q, ln); o.col = col;
          o.c = vec4f(x.xyz, sz.x); o.q = q; o.sh = vec2f(f32(shape), sz.y);
          return o;
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          var n = normalize(in.n);
          var p = in.w;
          let shape = u32(in.sh.x + 0.5);
          if (shape != 1u) {
            let rd = normalize(in.w - cam.eye);
            // ray in the body frame
            let iq = vec4f(-in.q.xyz, in.q.w);
            let ro = qrot(iq, cam.eye - in.c.xyz);
            let rl = qrot(iq, rd);
            let r = in.c.w;
            let h = select(0.0, in.sh.y, shape == 2u);
            // capsule (a sphere when h = 0): segment (0,-h,0)-(0,h,0), radius r (Quilez)
            let pa = vec3f(0.0, -h, 0.0);
            let ba = vec3f(0.0, 2.0 * h, 0.0);
            let oa = ro - pa;
            let baba = dot(ba, ba); let bard = dot(ba, rl); let baoa = dot(ba, oa);
            let rdoa = dot(rl, oa); let oaoa = dot(oa, oa);
            var t = -1.0;
            if (baba > 1e-12) {
              let a = baba - bard * bard;
              let b = baba * rdoa - baoa * bard;
              let c = baba * oaoa - baoa * baoa - r * r * baba;
              let disc = b * b - a * c;
              if (disc >= 0.0 && a > 1e-12) {
                let tc = (-b - sqrt(disc)) / a;
                let y = baoa + tc * bard;
                if (y > 0.0 && y < baba) { t = tc; }
              }
            }
            if (t < 0.0) {
              // end spheres (the sphere itself when h = 0): nearest hit in front of the eye
              var best = 1e30;
              for (var e = 0; e < 2; e++) {
                let oc = ro - select(pa + ba, pa, e == 0);
                let b = dot(rl, oc);
                let disc = b * b - dot(oc, oc) + r * r;
                if (disc >= 0.0) { let te = -b - sqrt(disc); if (te > 0.0 && te < best) { best = te; } }
              }
              if (best > 1e29) { discard; }
              t = best;
            }
            let lp = ro + rl * t;
            let hh = select(0.0, clamp(dot(lp - pa, ba) / baba, 0.0, 1.0), baba > 1e-12);
            let ln = (lp - pa - hh * ba) / r;
            p = cam.eye + rd * t;
            n = qrot(in.q, ln);
          }
          let v = normalize(cam.eye - p);
          let ao = 0.75 + 0.25 * clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
          let c = shadePBR(n, v, in.col, 0.0, 0.55, u.sun, vec3f(2.8), 1.0, ao);
          let clip = cam.viewProj * vec4f(p, 1.0);
          return FO(vec4f(c, 1.0), clip.z / clip.w);
        }`,
    });
    const sun = o.sun || [0.45, 0.8, 0.35], l = Math.hypot(...sun);
    this.program.set({ sun: sun.map((x) => x / l), staticColor: o.staticColor || [0.25, 0.26, 0.28], saturation: o.saturation ?? 1,
      kinematicColor: o.kinematicColor || [0.55, 0.42, 0.22], groupMask: (o.groups ?? 0xffffffff) >>> 0 });
  }
  /** Update sun, staticColor, kinematicColor, saturation or groups (the visible collision groups). */
  set(v) {
    const o = { ...v };
    if ('groups' in o) { o.groupMask = (o.groups ?? 0xffffffff) >>> 0; delete o.groups; }
    if (o.sun) { const l = Math.hypot(...o.sun) || 1; o.sun = o.sun.map((x) => x / l); }
    this.program.set(o);
    return this;
  }
  draw(pass, view) {
    if (this.world.destroyed) return;
    const n = this.world.count;
    if (n) this.program.draw(pass, view, 36, n, null, { P: this.world.B });
  }
  /** Layers own no GPU memory beyond their pipeline (garbage-collected); kept for a uniform API. */
  destroy() { this.destroyed = true; }
}
