// Volumetric renderer for a 3D density/temperature texture (the Smoke solver's
// scalar field). Single scattering with precomputed light transmittance:
//   1. compute: for every voxel march toward the sun -> transmittance texture
//   2. fragment: march the view ray through the box (stopping at the scene's
//      depth buffer), accumulating in-scattered sun light (Henyey–Greenstein
//      phase), sky ambient and blackbody-ish emission from temperature.
// Composited over the HDR scene with premultiplied alpha after the opaque pass.
import { RenderProgram } from './program.js';

export class Volume {
  /** o: smoke (Smoke instance), density, absorption, emission, sun, sunColor, ambient, steps */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.smoke = o.smoke;
    const s = this.smoke;
    this.light = ctx.texture({ size: s.dims, format: 'rgba16float', dimension: '3d',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING, label: 'volume-light' });
    this.values = { density: o.density ?? 30, emission: o.emission ?? 4, sun: norm(o.sun || [0.5, 0.75, 0.3]),
      sunColor: o.sunColor || [3.2, 3.0, 2.7], ambient: o.ambient || [0.25, 0.3, 0.38], g: o.g ?? 0.45, tint: o.tint || [0.92, 0.94, 0.97] };
    this.lightK = ctx.kernel({
      name: 'volume-light', workgroupSize: [4, 4, 4],
      textures: { scal: { sampleType: 'float', dim: '3d' }, dst: { storage: 'rgba16float', dim: '3d' } }, samplers: { lin: 'linear' },
      uniforms: { sunT: 'vec3f', density: 'f32', stepLen: 'f32' },
      code: /* wgsl */`
        var uvw = (vec3f(p) + 0.5) / vec3f(u.size);
        var tau = 0.0;
        for (var k = 0; k < 48; k++) {
          uvw += u.sunT;
          if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { break; }
          tau += textureSampleLevel(scal, lin, uvw, 0.0).x;
        }
        textureStore(dst, p, vec4f(exp(-tau * u.density * u.stepLen), 0.0, 0.0, 1.0));`,
    });
    this.program = new RenderProgram(ctx, {
      name: 'volume', imports: ['hash', 'color'],
      textures: { scal: { sampleType: 'float', dim: '3d' }, lightT: { sampleType: 'float', dim: '3d' }, depthTex: { sampleType: 'depth' } },
      samplers: { lin: 'linear' },
      uniforms: { boxMin: 'vec3f', boxSize: 'vec3f', stepLen: 'f32', density: 'f32', emission: 'f32', sun: 'vec3f', sunColor: 'vec3f',
        ambient: 'vec3f', g: 'f32', tint: 'vec3f' },
      depth: false,
      targets: [{ format: 'rgba16float', blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }],
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
          return VO(vec4f(xy, 0.0, 1.0));
        }
        fn fireColor(t: f32) -> vec3f {
          // smooth ramp: dark red -> orange -> yellow-white, scaled ~ t^3 (radiance grows fast with temperature)
          let x = clamp(t, 0.0, 1.5);
          return vec3f(1.0, 0.35 + 0.5 * x, 0.08 + 0.3 * x * x) * x * x * x;
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let px = in.pos.xy;
          let ndc = vec2f(px.x / cam.viewport.x * 2.0 - 1.0, 1.0 - px.y / cam.viewport.y * 2.0);
          let a = cam.invViewProj * vec4f(ndc, 0.0, 1.0);
          let b = cam.invViewProj * vec4f(ndc, 1.0, 1.0);
          let ro = cam.eye;
          let rd = normalize(b.xyz / b.w - a.xyz / a.w);
          let t0s = (u.boxMin - ro) / rd;
          let t1s = (u.boxMin + u.boxSize - ro) / rd;
          let tn = max(max(min(t0s.x, t1s.x), min(t0s.y, t1s.y)), min(t0s.z, t1s.z));
          let tf = min(min(max(t0s.x, t1s.x), max(t0s.y, t1s.y)), max(t0s.z, t1s.z));
          let d = textureLoad(depthTex, vec2i(px), 0);
          let sp = cam.invViewProj * vec4f(ndc, d, 1.0);
          let tScene = length(sp.xyz / sp.w - ro);
          let tEnd = min(tf, tScene);
          let tStart = max(tn, 0.0);
          if (tStart >= tEnd) { discard; }
          let cosT = dot(rd, u.sun);
          let g = u.g;
          let phase = (1.0 - g * g) / (12.566 * pow(1.0 + g * g - 2.0 * g * cosT, 1.5));
          var T = 1.0;
          var L = vec3f(0.0);
          var t = tStart + u.stepLen * ign(px);
          for (var k = 0; k < 384; k++) {
            if (t >= tEnd || T < 0.01) { break; }
            let uvw = (ro + rd * t - u.boxMin) / u.boxSize;
            let s = textureSampleLevel(scal, lin, uvw, 0.0);
            let sigma = s.x * u.density;
            if (sigma > 1e-4) {
              let lt = textureSampleLevel(lightT, lin, uvw, 0.0).x;
              let a = 1.0 - exp(-sigma * u.stepLen);
              let inscatter = u.tint * (u.sunColor * lt * phase * 4.0 + u.ambient);
              let emit = fireColor(s.y) * u.emission;
              L += T * (a * inscatter + emit * u.stepLen * s.x * 2.0);
              T *= 1.0 - a;
            }
            t += u.stepLen;
          }
          return vec4f(L, 1.0 - T);
        }`,
    });
  }

  set(v) { Object.assign(this.values, v); return this; }

  prepare() {
    const s = this.smoke, D = s.dims;
    const step = 1.0; // one voxel per light step (in voxel units)
    const sunT = this.values.sun.map((c, a) => (c * step) / D[a]);
    this.lightK.run(D, { sunT, density: this.values.density, stepLen: step * s.dx }, { scal: s.scalA, dst: this.light });
  }

  afterScene(view) {
    const s = this.smoke;
    const pass = this.ctx.renderPass({ colorAttachments: [{ view: view.hdr.view, loadOp: 'load', storeOp: 'store' }] });
    this.program.draw(pass, view, 3, 1, {
      boxMin: s.origin, boxSize: s.size, stepLen: s.dx * 0.6, ...this.values,
    }, { scal: s.scalA, lightT: this.light, depthTex: view.depth });
    pass.end();
  }
}

function norm(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
