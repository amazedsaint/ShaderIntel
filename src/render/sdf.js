// SdfScene: render a signed-distance-field scene by writing only `map()`.
//
//   view.add(gpu.sdf({
//     code: `fn map(p: vec3f) -> vec2f {
//       let d = smin(sdSphere(p - vec3f(0, 0.5, 0), 0.5), sdBox(p, vec3f(0.6, 0.2, 0.6)), 0.15);
//       return opU(vec2f(d, 1.0), vec2f(p.y, 0.0));          // (distance, material id)
//     }`,
//   }));
//
// Optional: `fn material(id: f32, p: vec3f, n: vec3f) -> Material` and extra
// uniforms (readable as u.name in both map and material). u.time is the view clock.
//
// Two passes per frame:
//   1. compute: one conservative cone per T×T pixel tile (coneMarch) -> safe start distance
//   2. fragment: per-pixel sphere tracing from that distance, then normal, soft
//      shadow, ambient occlusion and PBR shading; writes depth so rasterised
//      layers (particles, meshes) intersect the SDF scene correctly.
import { RenderProgram } from './program.js';

const MATERIAL_WGSL = /* wgsl */`
struct Material { albedo: vec3f, roughness: f32, metallic: f32, emissive: vec3f }
fn mat(albedo: vec3f, roughness: f32, metallic: f32) -> Material { return Material(albedo, roughness, metallic, vec3f(0.0)); }
fn defaultMaterial(id: f32, p: vec3f) -> Material {
  if (id < 0.5) {
    // ground: soft checker
    let c = floor(p.x * 2.0) + floor(p.z * 2.0);
    let k = 0.5 + 0.5 * (c - 2.0 * floor(c * 0.5));
    return mat(mix(vec3f(0.16), vec3f(0.22), k), 0.85, 0.0);
  }
  let pal = array<vec3f, 6>(vec3f(0.9, 0.25, 0.2), vec3f(0.95, 0.75, 0.25), vec3f(0.2, 0.55, 0.95),
                           vec3f(0.3, 0.8, 0.45), vec3f(0.85, 0.85, 0.9), vec3f(0.7, 0.35, 0.9));
  var pal2 = pal;
  let a = pal2[u32(id - 1.0) % 6u];
  return mat(a, 0.35, select(0.0, 1.0, u32(id) % 5u == 0u));
}`;

export class SdfScene {
  /**
   * o: code (WGSL with fn map, optional fn material), imports, uniforms, values,
   *    tile (default 4), shadows (true), ao (true), maxDistance (60), sun [x,y,z], fog
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.o = o;
    this.tile = o.tile || 4;
    const hasMaterial = /fn\s+material\s*\(/.test(o.code);
    const imports = ['sdf', 'raymarch', 'pbr', 'color', ...(o.imports || [])];
    const userUniforms = o.uniforms || {};
    const camU = { invViewProj: 'mat4x4f', eye: 'vec3f', coneK: 'f32', viewport: 'vec2f', tilesX: 'u32', maxDist: 'f32', time: 'f32' };
    this.coneK = ctx.kernel({
      name: 'sdf-cone', imports, workgroupSize: [8, 8],
      uniforms: { ...camU, ...userUniforms },
      buffers: { tsafe: { type: 'f32' } },
      shared: `${MATERIAL_WGSL}\n${o.code}`,
      code: /* wgsl */`
        let px = (vec2f(p.xy) + 0.5) * f32(${this.tile});
        let uv = vec2f(px.x / u.viewport.x * 2.0 - 1.0, 1.0 - px.y / u.viewport.y * 2.0);
        let a = u.invViewProj * vec4f(uv, 0.0, 1.0);
        let b = u.invViewProj * vec4f(uv, 1.0, 1.0);
        let ro = u.eye;
        let rd = normalize(b.xyz / b.w - a.xyz / a.w);
        tsafe[p.y * u.tilesX + p.x] = coneMarch(ro, rd, 0.0, u.maxDist, u.coneK);`,
    });
    this.program = new RenderProgram(ctx, {
      name: 'sdf-shade', imports,
      uniforms: { tilesX: 'u32', maxDist: 'f32', shadows: 'u32', ao: 'u32', sun: 'vec3f', sunRadiance: 'vec3f', fog: 'f32', tile: 'u32', time: 'f32', ...userUniforms },
      buffers: { tsafe: { type: 'f32' } },
      depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        ${MATERIAL_WGSL}
        ${o.code}
        ${hasMaterial ? '' : 'fn material(id: f32, p: vec3f, n: vec3f) -> Material { return defaultMaterial(id, p); }'}
        struct VO { @builtin(position) pos: vec4f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
          return VO(vec4f(xy, 0.0, 1.0));
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let px = in.pos.xy;
          let ndc = vec2f(px.x / cam.viewport.x * 2.0 - 1.0, 1.0 - px.y / cam.viewport.y * 2.0);
          let a = cam.invViewProj * vec4f(ndc, 0.0, 1.0);
          let b = cam.invViewProj * vec4f(ndc, 1.0, 1.0);
          let ro = cam.eye;
          let rd = normalize(b.xyz / b.w - a.xyz / a.w);
          let tp = vec2u(px) / u.tile;
          let t0 = tsafe[tp.y * u.tilesX + tp.x];
          let h = raymarchBasic(ro, rd, t0, u.maxDist, cam.pixelAngle * 0.75);
          if (!h.hit) { discard; }
          let pos = ro + rd * h.t;
          let n = calcNormal(pos, max(5e-4, h.t * cam.pixelAngle));
          let m = material(h.mat, pos, n);
          var sh = 1.0;
          if (u.shadows != 0u) { sh = softShadow(pos + n * max(0.004, h.t * cam.pixelAngle * 2.0), u.sun, 0.01, 30.0, 10.0); }
          var occ = 1.0;
          if (u.ao != 0u) { occ = calcAO(pos, n); }
          var c = shadePBR(n, -rd, m.albedo, m.metallic, m.roughness, u.sun, u.sunRadiance, sh, occ) + m.emissive;
          let fogF = 1.0 - exp(-u.fog * h.t * h.t * 0.002);
          c = mix(c, skyColor(rd, u.sun) * 0.25, clamp(fogF, 0.0, 1.0));
          let clip = cam.viewProj * vec4f(pos, 1.0);
          return FO(vec4f(c, 1.0), clip.z / clip.w);
        }`,
    });
    const sun = norm(o.sun || [0.55, 0.7, 0.35]);
    this.values = { shadows: o.shadows === false ? 0 : 1, ao: o.ao === false ? 0 : 1, sun, sunRadiance: o.sunRadiance || [3.2, 3.0, 2.7],
      fog: o.fog ?? 0.15, maxDist: o.maxDistance ?? 60, tile: this.tile, ...(o.values || {}) };
    this.tsafe = null;
    this.tw = 0; this.th = 0;
  }

  set(values) { Object.assign(this.values, values); return this; }

  prepare(view) {
    const T = this.tile;
    const tw = Math.ceil(view.width / T), th = Math.ceil(view.height / T);
    if (tw !== this.tw || th !== this.th) {
      this.tsafe?.destroy();
      this.tsafe = this.ctx.array('f32', tw * th, { label: 'sdf-tsafe' });
      this.tw = tw; this.th = th;
    }
    const cone = (T * 0.7072 * 1.05) * (2 * Math.tan(view.camera.fovY / 2)) / view.height;
    const user = {};
    for (const k of Object.keys(this.o.uniforms || {})) if (k in this.values) user[k] = this.values[k];
    this.coneK.run([tw, th], { invViewProj: view.invViewProj, eye: view.camera.eye(), coneK: cone, viewport: [view.width, view.height],
      tilesX: tw, maxDist: this.values.maxDist, time: view.time, ...user }, { tsafe: this.tsafe });
  }

  draw(pass, view) {
    this.program.draw(pass, view, 3, 1, { ...this.values, tilesX: this.tw, time: view.time }, { tsafe: this.tsafe });
  }
}

function norm(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
