// Scene helpers: a sky + ground "stage" and simple line geometry (box outlines).
import { Fullscreen } from './fullscreen.js';
import { RenderProgram } from './program.js';

/** Analytic sky gradient plus an infinite anti-aliased grid floor at y = height. Writes depth. */
export class Stage {
  constructor(ctx, o = {}) {
    this.layer = new Fullscreen(ctx, {
      name: 'stage', writeDepth: true, imports: ['color', 'pbr'],
      uniforms: { height: 'f32', cell: 'f32', groundColor: 'vec3f', lineColor: 'vec3f', skyTop: 'vec3f', skyBottom: 'vec3f', fog: 'f32', sun: 'vec3f' },
      values: {
        height: o.height ?? 0, cell: o.cell ?? 0.1,
        groundColor: o.groundColor || [0.035, 0.038, 0.045], lineColor: o.lineColor || [0.09, 0.1, 0.12],
        skyTop: o.skyTop || [0.02, 0.025, 0.035], skyBottom: o.skyBottom || [0.07, 0.08, 0.1], fog: o.fog ?? 0.15,
        sun: o.sun || [0.4, 0.8, 0.3],
      },
      code: /* wgsl */`
        fn shade(uv: vec2f, px: vec2f) -> FsFrag {
          let r = cameraRay(uv);
          let sky = mix(u.skyBottom, u.skyTop, clamp(r.d.y * 1.5 + 0.2, 0.0, 1.0));
          // plane intersection evaluated for every pixel so derivatives stay in uniform control flow
          let t = (u.height - r.o.y) / select(r.d.y, -1e-6, abs(r.d.y) < 1e-6);
          let p = r.o + r.d * abs(t);
          let g = p.xz / u.cell;
          let w = fwidth(g);
          let gl = abs(fract(g - 0.5) - 0.5) / max(w, vec2f(1e-4));
          let line = 1.0 - min(min(gl.x, gl.y), 1.0);
          let g10 = p.xz / (u.cell * 10.0);
          let w10 = fwidth(g10);
          let gl10 = abs(fract(g10 - 0.5) - 0.5) / max(w10, vec2f(1e-4));
          let line10 = 1.0 - min(min(gl10.x, gl10.y), 1.0);
          if (t <= 0.0 || r.d.y >= 0.0) { return FsFrag(vec4f(sky, 1.0), 1.0); }
          var c = u.groundColor + u.lineColor * (0.5 * line + line10);
          c *= 0.75 + 0.25 * max(dot(vec3f(0.0, 1.0, 0.0), normalize(u.sun)), 0.0);
          let fogF = 1.0 - exp(-u.fog * t * t * 0.05);
          c = mix(c, sky, clamp(fogF, 0.0, 1.0));
          return FsFrag(vec4f(c, 1.0), worldDepth(p));
        }`,
    });
  }
  set(v) { this.layer.set(v); return this; }
  draw(pass, view) { this.layer.draw(pass, view); }
}

/** Colored line segments (line-list) from a Float32Array of xyz pairs. */
export class Lines {
  constructor(ctx, o) {
    this.ctx = ctx;
    const pts = o.points;
    this.count = pts.length / 3;
    this.data = ctx.array('vec4f', this.count);
    const v4 = new Float32Array(this.count * 4);
    for (let k = 0; k < this.count; k++) { v4[4 * k] = pts[3 * k]; v4[4 * k + 1] = pts[3 * k + 1]; v4[4 * k + 2] = pts[3 * k + 2]; v4[4 * k + 3] = 1; }
    this.data.write(v4);
    this.program = new RenderProgram(ctx, {
      name: 'lines', buffers: { L: { type: 'vec4f' } }, uniforms: { color: 'vec4f' },
      values: { color: o.color || [0.35, 0.42, 0.5, 1] },
      topology: 'line-list',
      depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f { return cam.viewProj * vec4f(L[vi].xyz, 1.0); }
        @fragment fn fs() -> @location(0) vec4f { return u.color; }`,
    });
    this.program.bind({ L: this.data });
  }
  draw(pass, view) { this.program.draw(pass, view, this.count); }

  /** Wireframe box from min/max corners. */
  static box(ctx, min, max, color) {
    const [a, b] = [min, max];
    const c = [[a[0], a[1], a[2]], [b[0], a[1], a[2]], [b[0], b[1], a[2]], [a[0], b[1], a[2]],
      [a[0], a[1], b[2]], [b[0], a[1], b[2]], [b[0], b[1], b[2]], [a[0], b[1], b[2]]];
    const e = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    return new Lines(ctx, { points: new Float32Array(e.flatMap((k) => c[k])), color });
  }
}
