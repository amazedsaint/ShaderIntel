// Fullscreen shader layer — a Shadertoy-style canvas backed by the shader library.
//
//   view.add(gpu.fullscreen({
//     imports: ['noise', 'color'],
//     uniforms: { speed: 'f32' },
//     code: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
//       return vec4f(viridis(0.5 + 0.5 * simplex3(vec3f(uv * 6.0, cam.time * u.speed))), 1.0);
//     }`,
//   }));
//
// uv is in [0,1] with (0,0) at the bottom-left; px is the pixel coordinate.
// `cameraRay(uv)` returns the world-space ray of the view's orbit camera.
import { RenderProgram } from './program.js';

export class Fullscreen {
  constructor(ctx, desc) {
    this.values = {};
    this.program = new RenderProgram(ctx, {
      name: desc.name || 'fullscreen',
      imports: desc.imports, uniforms: desc.uniforms, textures: desc.textures, samplers: desc.samplers,
      buffers: desc.buffers, defines: desc.defines, values: desc.values,
      depth: { write: !!desc.writeDepth, compare: 'always' },
      targets: desc.targets,
      code: /* wgsl */`
        struct FsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
        struct Ray { o: vec3f, d: vec3f }
        fn cameraRay(uv: vec2f) -> Ray {
          let ndc = uv * 2.0 - 1.0;
          let a = cam.invViewProj * vec4f(ndc, 0.0, 1.0);
          let b = cam.invViewProj * vec4f(ndc, 1.0, 1.0);
          let o = a.xyz / a.w;
          return Ray(o, normalize(b.xyz / b.w - o));
        }
        /** View-space depth of a world point, as the depth buffer stores it. */
        fn worldDepth(p: vec3f) -> f32 { let c = cam.viewProj * vec4f(p, 1.0); return c.z / c.w; }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> FsOut {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
          return FsOut(vec4f(xy * 2.0 - 1.0, 0.0, 1.0), xy);
        }
        ${desc.code}
        ${desc.writeDepth
          ? `struct FsFrag { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
             @fragment fn fs(in: FsOut) -> FsFrag { let r = shade(in.uv, in.pos.xy); return FsFrag(r.color, r.depth); }`
          : `@fragment fn fs(in: FsOut) -> @location(0) vec4f { return shade(in.uv, in.pos.xy); }`}
      `,
    });
  }
  ready() { return this.program.ready(); }
  set(values) { this.program.set(values); return this; }
  bind(resources) { this.program.bind(resources); return this; }
  draw(pass, view) { this.program.draw(pass, view, 3, 1); }
}
