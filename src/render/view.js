// View: canvas + HDR scene target + orbit camera + frame loop + tonemapping.
//
//   const view = gpu.view('#canvas', { camera: { target: [0, 0.3, 0], distance: 2 } });
//   view.add(layer);                    // anything with draw(pass, view) / prepare(view)
//   view.run(({ dt, time }) => sim.step(dt));
//
// Layers may implement:
//   prepare(view)          record compute / off-screen passes before the scene pass
//   draw(pass, view)       draw into the HDR scene pass (rgba16float + depth32float)
//   overlay(pass, view)    draw after tonemapping, straight into the canvas (UI, debug)
import { RenderProgram, CAMERA_SIZE } from './program.js';
import { Bloom } from './bloom.js';
import { m4, v3 } from './math.js';

let viewId = 0;
export const HDR_FORMAT = 'rgba16float';
export const DEPTH_FORMAT = 'depth32float';

export class View {
  constructor(ctx, canvas, opts = {}) {
    this.ctx = ctx;
    this.id = 'v' + ++viewId;
    this.canvas = typeof canvas === 'string' ? document.querySelector(canvas) : canvas;
    if (!this.canvas) throw new Error('ShaderIntel: canvas not found');
    this.context = this.canvas.getContext('webgpu');
    this.context.configure({ device: ctx.device, format: ctx.format, alphaMode: 'opaque' });
    this.layers = [];
    this.background = opts.background || [0.02, 0.025, 0.03, 1];
    this.exposure = opts.exposure ?? 1;
    this.tonemap = opts.tonemap || 'agx';
    this.pixelRatio = opts.pixelRatio || Math.min(2, window.devicePixelRatio || 1);
    this.time = 0;
    this.frame = 0;
    this.dt = 1 / 60;
    this.cameraBuffer = ctx.device.createBuffer({ size: CAMERA_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'camera' });
    this.camData = new ArrayBuffer(CAMERA_SIZE);
    this.camera = new OrbitCamera(this.canvas, opts.camera || {});
    this.controls = opts.controls !== false;
    if (this.controls) this.camera.attach();
    this.pointer = { x: 0, y: 0, ndc: [0, 0], down: false, buttons: 0, inside: false };
    this._pointerEvents();
    this.width = 0; this.height = 0;
    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this.canvas);
    this.bloom = opts.bloom ? new Bloom(ctx, opts.bloom === true ? {} : opts.bloom) : null;
    this._black = ctx.texture({ size: [1, 1], format: 'rgba16float', label: 'black' });
    this.post = new RenderProgram(ctx, {
      name: 'tonemap', imports: ['tonemap', 'hash'],
      textures: { hdr: { sampleType: 'unfilterable-float' }, bloomTex: { sampleType: 'float' } },
      samplers: { lin: 'linear' },
      uniforms: { exposure: 'f32', mode: 'u32', bloomStrength: 'f32' },
      targets: [ctx.format], depth: false,
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
          return VO(vec4f(xy, 0.0, 1.0));
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let uv = in.pos.xy / cam.viewport;
          let hdrc = textureLoad(hdr, vec2i(in.pos.xy), 0).rgb;
          let bl = textureSampleLevel(bloomTex, lin, uv, 0.0).rgb;
          let c = max(mix(hdrc, bl, u.bloomStrength) * u.exposure, vec3f(0.0));
          var m: vec3f;
          switch u.mode {
            case 0u: { m = tonemapAgX(c); }
            case 1u: { m = tonemapACES(c); }
            case 2u: { m = tonemapNeutral(c); }
            case 3u: { m = tonemapReinhard(c, 4.0); }
            default: { m = clamp(c, vec3f(0.0), vec3f(1.0)); }
          }
          return vec4f(encodeDisplay(m, in.pos.xy), 1.0);
        }`,
    });
    if (opts.hud) this.hud = new Hud(this, opts.hud === true ? {} : opts.hud);
  }

  add(layer) { this.layers.push(layer); layer.attach?.(this); return layer; }
  remove(layer) { this.layers = this.layers.filter((l) => l !== layer); }

  _resize() {
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * this.pixelRatio));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * this.pixelRatio));
    if (w === this.width && h === this.height) return;
    this.width = w; this.height = h;
    this.canvas.width = w; this.canvas.height = h;
    this.hdr?.destroy(); this.depth?.destroy();
    const TU = GPUTextureUsage;
    this.hdr = this.ctx.texture({ size: [w, h], format: HDR_FORMAT, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC, label: 'hdr' });
    this.depth = this.ctx.texture({ size: [w, h], format: DEPTH_FORMAT, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'depth' });
    for (const l of this.layers) l.resize?.(this);
  }

  _pointerEvents() {
    const c = this.canvas, p = this.pointer;
    const upd = (e) => {
      const r = c.getBoundingClientRect();
      p.x = e.clientX - r.left; p.y = e.clientY - r.top;
      p.ndc = [(p.x / r.width) * 2 - 1, 1 - (p.y / r.height) * 2];
      p.buttons = e.buttons; p.shift = e.shiftKey; p.alt = e.altKey; p.ctrl = e.ctrlKey || e.metaKey;
    };
    c.addEventListener('pointermove', (e) => { upd(e); p.inside = true; });
    c.addEventListener('pointerdown', (e) => { upd(e); p.down = true; });
    window.addEventListener('pointerup', (e) => { p.down = false; p.buttons = e.buttons; });
    c.addEventListener('pointerleave', () => { p.inside = false; });
  }

  /** World-space ray through the pointer (or an NDC point). */
  ray(ndc = this.pointer.ndc) {
    const inv = this.invViewProj;
    const a = m4.transformPoint(inv, [ndc[0], ndc[1], 0]);
    const b = m4.transformPoint(inv, [ndc[0], ndc[1], 1]);
    return { origin: a, dir: v3.norm(v3.sub(b, a)) };
  }

  /** Upload the camera block (called every frame by render()). */
  updateCamera() {
    const cam = this.camera;
    const aspect = this.width / this.height;
    const proj = m4.perspective(cam.fovY, aspect, cam.near, cam.far);
    const eye = cam.eye();
    const view = m4.lookAt(eye, cam.target, cam.up);
    const viewProj = m4.mul(proj, view);
    this.view = view; this.proj = proj; this.viewProj = viewProj;
    this.invViewProj = m4.invert(viewProj);
    const f = new Float32Array(this.camData);
    f.set(view, 0); f.set(proj, 16); f.set(viewProj, 32); f.set(this.invViewProj, 48);
    f.set(m4.invert(view), 64); f.set(m4.invert(proj), 80);
    f.set(eye, 96); f[99] = this.time;
    f[100] = this.width; f[101] = this.height; f[102] = cam.near; f[103] = cam.far;
    f[104] = cam.fovY; f[105] = (2 * Math.tan(cam.fovY / 2)) / this.height;
    new Uint32Array(this.camData)[106] = this.frame; f[107] = this.dt;
    this.ctx.device.queue.writeBuffer(this.cameraBuffer, 0, this.camData);
  }

  /** Record the frame: layer prepare, scene pass, tonemap, overlays; then submit. */
  render() {
    const ctx = this.ctx;
    this.updateCamera();
    for (const l of this.layers) l.prepare?.(this);
    const bg = this.background;
    const pass = ctx.renderPass({
      colorAttachments: [{ view: this.hdr.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: bg[0], g: bg[1], b: bg[2], a: bg[3] } }],
      depthStencilAttachment: { view: this.depth.view, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    for (const l of this.layers) l.draw?.(pass, this);
    pass.end();
    for (const l of this.layers) l.afterScene?.(this);
    const bloomTex = this.bloom ? this.bloom.run(this.hdr) : this._black;
    const target = this.context.getCurrentTexture().createView();
    const out = ctx.renderPass({ colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    const mode = { agx: 0, aces: 1, neutral: 2, reinhard: 3, none: 4 }[this.tonemap] ?? 0;
    this.post.draw(out, this, 3, 1, { exposure: this.exposure, mode, bloomStrength: this.bloom ? this.bloom.strength : 0 },
      { hdr: this.hdr, bloomTex });
    for (const l of this.layers) l.overlay?.(out, this);
    out.end();
    ctx.flush();
  }

  /** Start the frame loop. update({ dt, time, frame, view }) records simulation work. */
  run(update) {
    this._update = update;
    let last = performance.now();
    const tick = (now) => {
      if (this._stopped) return;
      this.dt = Math.min(1 / 20, Math.max(1e-4, (now - last) / 1000));
      last = now;
      this.camera.update(this.dt);
      try {
        update?.({ dt: this.dt, time: this.time, frame: this.frame, view: this });
        this.render();
      } catch (e) {
        console.error(e);
        this._stopped = true;
        showError(this.canvas, e);
        return;
      }
      this.hud?.update(now);
      this.time += this.dt;
      this.frame++;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return this;
  }

  stop() { this._stopped = true; }

  /** Advance one frame without the animation loop (tests, video capture, headless runs). */
  step(dt = 1 / 60) {
    this.dt = dt;
    this.camera.update(dt);
    this._update?.({ dt, time: this.time, frame: this.frame, view: this });
    this.render();
    this.time += dt;
    this.frame++;
    return this;
  }
}

/** Orbit camera with damping: drag = rotate, right/shift-drag = pan, wheel/pinch = zoom. */
export class OrbitCamera {
  constructor(el, o = {}) {
    this.el = el;
    this.target = o.target ? [...o.target] : [0, 0, 0];
    this.distance = o.distance ?? 3;
    this.yaw = o.yaw ?? 0.6;
    this.pitch = o.pitch ?? 0.35;
    this.fovY = ((o.fov ?? 45) * Math.PI) / 180;
    this.near = o.near ?? 0.01;
    this.far = o.far ?? 200;
    this.up = [0, 1, 0];
    this.minDistance = o.minDistance ?? 0.05;
    this.maxDistance = o.maxDistance ?? 100;
    this.autoRotate = o.autoRotate ?? 0;
    this.vel = { yaw: 0, pitch: 0, zoom: 0, px: 0, py: 0 };
    this.damping = 8;
  }
  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [this.target[0] + this.distance * cp * Math.sin(this.yaw), this.target[1] + this.distance * sp,
      this.target[2] + this.distance * cp * Math.cos(this.yaw)];
  }
  attach() {
    const el = this.el;
    const ptrs = new Map();
    let pinch = 0;
    el.style.touchAction = 'none';
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return; // modifier-drag is reserved for app interaction
      el.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
    });
    el.addEventListener('pointermove', (e) => {
      const p = ptrs.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) this.vel.zoom += (pinch - d) * 0.01;
        pinch = d;
        return;
      }
      const s = 1 / el.clientHeight;
      if (p.pan) { this.vel.px += -dx * s * this.distance; this.vel.py += dy * s * this.distance; }
      else { this.vel.yaw -= dx * s * 4; this.vel.pitch += dy * s * 4; }
    });
    const end = (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.vel.zoom += e.deltaY * 0.0015; }, { passive: false });
  }
  update(dt) {
    const k = Math.min(1, this.damping * dt), v = this.vel;
    this.yaw += v.yaw * k + this.autoRotate * dt;
    this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + v.pitch * k));
    this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance * Math.exp(v.zoom * k)));
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const right = [cy, 0, -sy];
    const up = [-Math.sin(this.pitch) * sy, Math.cos(this.pitch), -Math.sin(this.pitch) * cy];
    this.target = v3.add(this.target, v3.add(v3.scale(right, v.px * k), v3.scale(up, v.py * k)));
    for (const key in v) v[key] *= 1 - k;
  }
}

/** Small on-canvas stats overlay: FPS + per-region GPU times from the profiler. */
export class Hud {
  constructor(view, o = {}) {
    this.view = view;
    this.el = document.createElement('div');
    this.el.className = 'si-hud';
    Object.assign(this.el.style, {
      position: 'absolute', left: '10px', top: '10px', padding: '8px 10px', font: '11px/1.45 ui-monospace,Menlo,monospace',
      color: '#d8dee9', background: 'rgba(10,12,16,0.72)', borderRadius: '6px', pointerEvents: 'none', whiteSpace: 'pre', zIndex: 10,
    });
    (o.parent || view.canvas.parentElement).appendChild(this.el);
    this.extra = o.extra || null;
    this.frames = 0; this.t0 = performance.now(); this.fps = 0;
  }
  update(now) {
    this.frames++;
    if (now - this.t0 < 500) return;
    this.fps = (this.frames * 1000) / (now - this.t0);
    this.frames = 0; this.t0 = now;
    const v = this.view;
    const lines = [`${this.fps.toFixed(0)} fps  ${v.width}×${v.height}`];
    let total = 0;
    for (const r of v.ctx.profiler.report()) { total += r.ms; lines.push(`${r.label.padEnd(14)} ${r.ms.toFixed(2).padStart(6)} ms`); }
    if (total) lines.push(`${'gpu total'.padEnd(14)} ${total.toFixed(2).padStart(6)} ms`);
    if (this.extra) lines.push(...[].concat(this.extra()));
    this.el.textContent = lines.join('\n');
  }
}

export function showError(canvas, e) {
  const div = document.createElement('div');
  Object.assign(div.style, { position: 'absolute', inset: '0', padding: '24px', color: '#ffb4a9', background: 'rgba(20,0,0,0.85)',
    font: '12px/1.5 ui-monospace,Menlo,monospace', whiteSpace: 'pre-wrap', overflow: 'auto', zIndex: 20 });
  div.textContent = String(e && (e.message || e));
  (canvas?.parentElement || document.body).appendChild(div);
}
