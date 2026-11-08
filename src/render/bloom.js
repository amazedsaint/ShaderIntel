// Energy-conserving HDR bloom (progressive downsample / tent upsample, after
// Jimenez, "Next Generation Post Processing in Call of Duty: AW", 2014).
// No threshold: every pixel contributes a little, so bright HDR values glow
// naturally and the image stays physically plausible.
import { destroyAll } from '../core/dispose.js';
export class Bloom {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.strength = o.strength ?? 0.05;
    this.levels = Math.max(1, Math.round(o.levels ?? 6));
    this.radius = o.radius ?? 1;
    this.down = [];
    this.up = [];
    this.w = 0; this.h = 0;
    const sampler = { filter: 'linear' };
    this.downK = ctx.kernel({
      name: 'bloom-down', workgroupSize: [8, 8],
      textures: { src: { sampleType: 'float' }, dst: { storage: 'rgba16float' } }, samplers: { lin: sampler },
      uniforms: { srcTexel: 'vec2f', firstLevel: 'u32' },
      code: /* wgsl */`
        let uv = (vec2f(p.xy) + 0.5) / vec2f(u.size.xy);
        let t = u.srcTexel;
        let a = textureSampleLevel(src, lin, uv + t * vec2f(-2.0, 2.0), 0.0).rgb;
        let b = textureSampleLevel(src, lin, uv + t * vec2f(0.0, 2.0), 0.0).rgb;
        let c = textureSampleLevel(src, lin, uv + t * vec2f(2.0, 2.0), 0.0).rgb;
        let d = textureSampleLevel(src, lin, uv + t * vec2f(-2.0, 0.0), 0.0).rgb;
        let e = textureSampleLevel(src, lin, uv, 0.0).rgb;
        let f = textureSampleLevel(src, lin, uv + t * vec2f(2.0, 0.0), 0.0).rgb;
        let g = textureSampleLevel(src, lin, uv + t * vec2f(-2.0, -2.0), 0.0).rgb;
        let h = textureSampleLevel(src, lin, uv + t * vec2f(0.0, -2.0), 0.0).rgb;
        let ii = textureSampleLevel(src, lin, uv + t * vec2f(2.0, -2.0), 0.0).rgb;
        let j = textureSampleLevel(src, lin, uv + t * vec2f(-1.0, 1.0), 0.0).rgb;
        let k = textureSampleLevel(src, lin, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
        let l = textureSampleLevel(src, lin, uv + t * vec2f(-1.0, -1.0), 0.0).rgb;
        let m = textureSampleLevel(src, lin, uv + t * vec2f(1.0, -1.0), 0.0).rgb;
        var o = e * 0.125 + (a + c + g + ii) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
        if (u.firstLevel == 1u) {
          // a NaN or infinity in the scene would spread over the whole chain: drop it (integer test,
          // which no compiler may fold away), then tame fireflies before they spread
          let bits = bitcast<vec3u>(o) & vec3u(0x7fffffffu);
          o = min(select(o, vec3f(0.0), bits >= vec3u(0x7f800000u)), vec3f(64.0));
        }
        textureStore(dst, vec2i(p.xy), vec4f(max(o, vec3f(0.0)), 1.0));`,
    });
    this.upK = ctx.kernel({
      name: 'bloom-up', workgroupSize: [8, 8],
      textures: { small: { sampleType: 'float' }, cur: { sampleType: 'float' }, dst: { storage: 'rgba16float' } }, samplers: { lin: sampler },
      uniforms: { smallTexel: 'vec2f', radius: 'f32' },
      code: /* wgsl */`
        let uv = (vec2f(p.xy) + 0.5) / vec2f(u.size.xy);
        let t = u.smallTexel * u.radius;
        var s = textureSampleLevel(small, lin, uv, 0.0).rgb * 4.0;
        s += (textureSampleLevel(small, lin, uv + vec2f(t.x, 0.0), 0.0).rgb + textureSampleLevel(small, lin, uv - vec2f(t.x, 0.0), 0.0).rgb
            + textureSampleLevel(small, lin, uv + vec2f(0.0, t.y), 0.0).rgb + textureSampleLevel(small, lin, uv - vec2f(0.0, t.y), 0.0).rgb) * 2.0;
        s += textureSampleLevel(small, lin, uv + t, 0.0).rgb + textureSampleLevel(small, lin, uv - t, 0.0).rgb
           + textureSampleLevel(small, lin, uv + vec2f(t.x, -t.y), 0.0).rgb + textureSampleLevel(small, lin, uv + vec2f(-t.x, t.y), 0.0).rgb;
        let c = textureSampleLevel(cur, lin, uv, 0.0).rgb + s / 16.0;
        textureStore(dst, vec2i(p.xy), vec4f(c, 1.0));`,
    });
  }

  _alloc(w, h) {
    for (const t of [...this.down, ...this.up]) t.destroy();
    this.down = []; this.up = [];
    let cw = w, ch = h;
    const TU = GPUTextureUsage;
    for (let k = 0; k < this.levels; k++) {
      cw = Math.max(1, cw >> 1); ch = Math.max(1, ch >> 1);
      const usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING;
      this.down.push(this.ctx.texture({ size: [cw, ch], format: 'rgba16float', usage, label: `bloom-down${k}` }));
      this.up.push(this.ctx.texture({ size: [cw, ch], format: 'rgba16float', usage, label: `bloom-up${k}` }));
      if (cw <= 4 || ch <= 4) break;
    }
    this.w = w; this.h = h;
  }

  /** Build the bloom chain from the HDR texture; returns the texture to composite (half res). */
  run(hdr) {
    if (hdr.width !== this.w || hdr.height !== this.h) this._alloc(hdr.width, hdr.height);
    const n = this.down.length;
    let src = hdr;
    for (let k = 0; k < n; k++) {
      const dst = this.down[k];
      this.downK.run([dst.width, dst.height], { srcTexel: [1 / src.width, 1 / src.height], firstLevel: k === 0 ? 1 : 0 }, { src, dst });
      src = dst;
    }
    // smallest level: up = down
    let small = this.down[n - 1];
    for (let k = n - 2; k >= 0; k--) {
      const dst = this.up[k];
      this.upK.run([dst.width, dst.height], { smallTexel: [1 / small.width, 1 / small.height], radius: this.radius },
        { small, cur: this.down[k], dst });
      small = dst;
    }
    return n > 1 ? this.up[0] : this.down[0];
  }
  destroy() { this.destroyed = true; destroyAll(this.down, this.up); this.down = []; this.up = []; }
}
