// View: canvas + HDR scene target + orbit camera + frame loop + tonemapping.
//
//   const view = gpu.view('#canvas', { camera: { target: [0, 0.3, 0], distance: 2 } });
//   view.add(layer);                    // anything with draw(pass, view) / prepare(view)
//   view.run(({ dt, time }) => sim.step(dt));
//
// Layers may implement:
//   prepare(view)          record compute / off-screen passes before the scene pass
//   draw(pass, view)       draw into the HDR scene pass (rgba16float + depth32float)
//   afterScene(view)       screen-space effects that read the scene (water, volumes)
//   overlay(pass, view)    draw after tonemapping, straight into the canvas (UI, debug)
//   resize(view)           the drawing buffer changed size
//   attach(view) / detach(view)   added to / removed from a view
//
// view.destroy() stops the loop, removes every listener and observer, unconfigures the
// canvas and frees the view's own GPU textures; destroy({ layers: true }) also destroys
// the layers that are still attached.
import { RenderProgram, CAMERA_SIZE } from './program.js';
import { Bloom } from './bloom.js';
import { m4, v3 } from './math.js';
import { SIError } from '../core/errors.js';

let viewId = 0;
export const HDR_FORMAT = 'rgba16float';
export const DEPTH_FORMAT = 'depth32float';
const TONEMAPS = { agx: 0, aces: 1, neutral: 2, reinhard: 3, none: 4 };

export class View {
  constructor(ctx, canvas, opts = {}) {
    this.ctx = ctx;
    this.id = 'v' + ++viewId;
    this.canvas = typeof canvas === 'string' ? document.querySelector(canvas) : canvas;
    if (!this.canvas) throw new SIError(`View: canvas ${typeof canvas === 'string' ? `"${canvas}" ` : ''}not found`);
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) throw new SIError('View: the canvas has no WebGPU context (is it already used by WebGL or a 2D context?)');
    this.context.configure({ device: ctx.device, format: ctx.format, alphaMode: 'opaque' });
    this.layers = [];
    this.background = opts.background || [0.02, 0.025, 0.03, 1];
    this.exposure = opts.exposure ?? 1;
    this.tonemap = opts.tonemap || 'agx';
    if (!(this.tonemap in TONEMAPS)) throw new SIError(`View: unknown tonemap "${this.tonemap}" (use ${Object.keys(TONEMAPS).join(' | ')})`);
    this._dprAuto = !opts.pixelRatio;
    this.pixelRatio = opts.pixelRatio || Math.min(opts.maxPixelRatio ?? 2, window.devicePixelRatio || 1);
    this.maxPixelRatio = opts.maxPixelRatio ?? 2;
    this.onError = opts.onError || null;
    /** Called before every frame with { dt, time, frame, view } (set by run(fn), or directly). */
    this.onFrame = null;
    this.time = 0;
    this.frame = 0;
    this.dt = 1 / 60;
    this.destroyed = false;
    this._off = [];        // listener removers, run by destroy()
    this._raf = 0;
    this._stopped = true;
    this.cameraBuffer = ctx.device.createBuffer({ size: CAMERA_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'camera' });
    this.camData = new ArrayBuffer(CAMERA_SIZE);
    this.camera = new OrbitCamera(this.canvas, opts.camera || {});
    this.controls = opts.controls !== false;
    if (this.controls) this.camera.attach();
    this.pointer = { x: 0, y: 0, ndc: [0, 0], down: false, buttons: 0, inside: false, shift: false, alt: false, ctrl: false };
    this._pointerEvents();
    this.width = 0; this.height = 0;
    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this.canvas);
    this._watchPixelRatio();
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

  /** Add a layer (anything with draw / prepare / afterScene / overlay). Returns the layer. */
  add(layer) {
    if (!layer) throw new SIError('View.add: no layer given');
    if (!this.layers.includes(layer)) { this.layers.push(layer); layer.attach?.(this); }
    return layer;
  }

  /** Remove a layer (it is not destroyed). Returns the layer. */
  remove(layer) {
    if (this.layers.includes(layer)) { this.layers = this.layers.filter((l) => l !== layer); layer.detach?.(this); }
    return layer;
  }

  /** Turn bloom on (true or { strength, ... }) or off (false) after construction. */
  setBloom(o) {
    if (!o) { this.bloom?.destroy?.(); this.bloom = null; return this; }
    const opts = o === true ? {} : o;
    if (this.bloom && opts.strength !== undefined && Object.keys(opts).length === 1) { this.bloom.strength = opts.strength; return this; }
    this.bloom?.destroy?.();
    this.bloom = new Bloom(this.ctx, opts);
    return this;
  }

  /** Show (true or { extra, parent }) or hide (false) the stats overlay. */
  setHud(o) {
    this.hud?.destroy();
    this.hud = o ? new Hud(this, o === true ? {} : o) : null;
    return this;
  }

  /** Render at a different device-pixel ratio (e.g. 1 for speed). */
  setPixelRatio(r) {
    this._dprAuto = false;
    this.pixelRatio = Math.max(0.25, +r || 1);
    this._resize();
    return this;
  }

  _resize() {
    if (this.destroyed) return;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * this.pixelRatio));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * this.pixelRatio));
    if (w === this.width && h === this.height) return;
    const max = this.ctx.limits.maxTextureDimension2D || 8192;
    const s = Math.min(1, max / w, max / h);   // a huge canvas must not exceed the texture limit
    const W = Math.max(1, Math.floor(w * s)), H = Math.max(1, Math.floor(h * s));
    this.width = W; this.height = H;
    this.canvas.width = W; this.canvas.height = H;
    this.hdr?.destroy(); this.depth?.destroy();
    const TU = GPUTextureUsage;
    this.hdr = this.ctx.texture({ size: [W, H], format: HDR_FORMAT, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC, label: 'hdr' });
    this.depth = this.ctx.texture({ size: [W, H], format: DEPTH_FORMAT, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'depth' });
    for (const l of this.layers) l.resize?.(this);
  }

  // Follow devicePixelRatio changes (window moved to another screen, browser zoom).
  _watchPixelRatio() {
    if (typeof matchMedia !== 'function') return;
    let mq = null;
    const onChange = () => {
      if (this._dprAuto) { this.pixelRatio = Math.min(this.maxPixelRatio, window.devicePixelRatio || 1); this._resize(); }
      watch();
    };
    const watch = () => {
      mq?.removeEventListener?.('change', onChange);
      mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      mq.addEventListener?.('change', onChange);
    };
    watch();
    this._off.push(() => mq?.removeEventListener?.('change', onChange));
  }

  _listen(target, type, fn, opts) {
    target.addEventListener(type, fn, opts);
    this._off.push(() => target.removeEventListener(type, fn, opts));
  }

  _pointerEvents() {
    const c = this.canvas, p = this.pointer;
    const upd = (e) => {
      const r = c.getBoundingClientRect();
      p.x = e.clientX - r.left; p.y = e.clientY - r.top;
      p.ndc = [(p.x / (r.width || 1)) * 2 - 1, 1 - (p.y / (r.height || 1)) * 2];
      p.buttons = e.buttons; p.shift = e.shiftKey; p.alt = e.altKey; p.ctrl = e.ctrlKey || e.metaKey;
    };
    this._listen(c, 'pointermove', (e) => { upd(e); p.inside = true; });
    this._listen(c, 'pointerdown', (e) => { upd(e); p.down = true; });
    this._listen(window, 'pointerup', (e) => { p.down = false; p.buttons = e.buttons; });
    this._listen(window, 'pointercancel', () => { p.down = false; p.buttons = 0; });
    this._listen(c, 'pointerleave', () => { p.inside = false; });
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
    if (this.destroyed) return this;
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
    const mode = TONEMAPS[this.tonemap] ?? 0;
    this.post.draw(out, this, 3, 1, { exposure: this.exposure, mode, bloomStrength: this.bloom ? this.bloom.strength : 0 },
      { hdr: this.hdr, bloomTex });
    for (const l of this.layers) l.overlay?.(out, this);
    out.end();
    ctx.flush();
    return this;
  }

  /**
   * Start (or resume) the frame loop. update({ dt, time, frame, view }) records simulation work
   * before every frame; calling run() again replaces the callback (run() with no argument keeps it).
   * An exception stops the loop and is passed to opts.onError, or shown over the canvas.
   */
  run(update) {
    if (this.destroyed) throw new SIError('View.run: the view was destroyed');
    if (update !== undefined) this.onFrame = update;
    this._stopped = false;
    if (this._raf) return this;   // already running: one loop per view
    let last = performance.now();
    const tick = (now) => {
      this._raf = 0;
      if (this._stopped || this.destroyed) return;
      this.dt = Math.min(1 / 20, Math.max(1e-4, (now - last) / 1000));
      last = now;
      this.camera.update(this.dt);
      try {
        this.onFrame?.({ dt: this.dt, time: this.time, frame: this.frame, view: this });
        this.render();
      } catch (e) {
        this._stopped = true;
        this._fail(e);
        return;
      }
      this.hud?.update(now);
      this.time += this.dt;
      this.frame++;
      // (an update that called stop() and run() has already scheduled the next frame)
      if (!this._stopped && !this.destroyed && !this._raf) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
    return this;
  }

  /** Pause the frame loop (run() resumes it). */
  stop() {
    this._stopped = true;
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    return this;
  }

  /** True while the frame loop is scheduled. */
  get running() { return !this._stopped && !this.destroyed; }

  /** Advance one frame without the animation loop (tests, video capture, headless runs). */
  step(dt = 1 / 60) {
    if (this.destroyed) return this;
    this.dt = dt;
    this.camera.update(dt);
    this.onFrame?.({ dt, time: this.time, frame: this.frame, view: this });
    this.render();
    this.time += dt;
    this.frame++;
    return this;
  }

  _fail(e) {
    if (this.onError) { try { this.onError(e); } catch (e2) { console.error(e2); } return; }
    console.error(e);
    this._errorEl = showError(this.canvas, e);
  }

  /**
   * Stop the loop, remove listeners, observers and the HUD, unconfigure the canvas and free the
   * view's GPU textures. With { layers: true } the attached layers are destroyed too.
   */
  destroy({ layers = false } = {}) {
    if (this.destroyed) return;
    this.stop();
    this.destroyed = true;
    this._ro?.disconnect();
    for (const off of this._off.splice(0)) off();
    this.camera.detach();
    this.hud?.destroy();
    this._errorEl?.remove();
    for (const l of this.layers) { l.detach?.(this); if (layers) l.destroy?.(); }
    this.layers = [];
    this.bloom?.destroy?.();
    for (const t of [this.hdr, this.depth, this._black]) t?.destroy();
    this.cameraBuffer.destroy();
    try { this.context.unconfigure(); } catch { /* the device may already be gone */ }
  }
}

/**
 * Orbit camera with damping: drag = rotate, right/shift-drag = pan, wheel/pinch = zoom.
 * Options: target, distance, yaw, pitch (rad), fov (deg), near, far, minDistance, maxDistance,
 * minPitch, maxPitch, autoRotate (rad/s), damping, rotateSpeed, zoomSpeed, panSpeed.
 * Set camera.enabled = false to ignore input without detaching.
 */
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
    this.minPitch = o.minPitch ?? -1.55;
    this.maxPitch = o.maxPitch ?? 1.55;
    this.autoRotate = o.autoRotate ?? 0;
    this.rotateSpeed = o.rotateSpeed ?? 1;
    this.zoomSpeed = o.zoomSpeed ?? 1;
    this.panSpeed = o.panSpeed ?? 1;
    this.enabled = o.enabled ?? true;
    this.vel = { yaw: 0, pitch: 0, zoom: 0, px: 0, py: 0 };
    this.damping = o.damping ?? 8;
    this._off = [];
  }
  /** Field of view in degrees. */
  get fov() { return (this.fovY * 180) / Math.PI; }
  set fov(deg) { this.fovY = (deg * Math.PI) / 180; }
  /** Update any of the constructor options at once (e.g. to frame a new scene). */
  set(o = {}) {
    for (const k of ['distance', 'yaw', 'pitch', 'near', 'far', 'minDistance', 'maxDistance', 'minPitch', 'maxPitch',
      'autoRotate', 'damping', 'rotateSpeed', 'zoomSpeed', 'panSpeed', 'enabled']) if (o[k] !== undefined) this[k] = o[k];
    if (o.target) this.target = [...o.target];
    if (o.fov !== undefined) this.fov = o.fov;
    this.vel = { yaw: 0, pitch: 0, zoom: 0, px: 0, py: 0 };
    return this;
  }
  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [this.target[0] + this.distance * cp * Math.sin(this.yaw), this.target[1] + this.distance * sp,
      this.target[2] + this.distance * cp * Math.cos(this.yaw)];
  }
  attach() {
    if (this._off.length) return this;
    const el = this.el;
    const ptrs = new Map();
    let pinch = 0;
    const on = (target, type, fn, opts) => { target.addEventListener(type, fn, opts); this._off.push(() => target.removeEventListener(type, fn, opts)); };
    const prevTouch = el.style.touchAction;
    el.style.touchAction = 'none';
    this._off.push(() => { el.style.touchAction = prevTouch; });
    on(el, 'contextmenu', (e) => e.preventDefault());
    on(el, 'pointerdown', (e) => {
      if (!this.enabled || e.altKey || e.ctrlKey || e.metaKey) return; // modifier-drag is reserved for app interaction
      el.setPointerCapture?.(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
    });
    on(el, 'pointermove', (e) => {
      const p = ptrs.get(e.pointerId);
      if (!p || !this.enabled) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) this.vel.zoom += (pinch - d) * 0.01 * this.zoomSpeed;
        pinch = d;
        return;
      }
      const s = 1 / (el.clientHeight || 1);
      if (p.pan) { this.vel.px += -dx * s * this.distance * this.panSpeed; this.vel.py += dy * s * this.distance * this.panSpeed; }
      else { this.vel.yaw -= dx * s * 4 * this.rotateSpeed; this.vel.pitch += dy * s * 4 * this.rotateSpeed; }
    });
    const end = (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; };
    on(el, 'pointerup', end);
    on(el, 'pointercancel', end);
    on(el, 'wheel', (e) => { if (!this.enabled) return; e.preventDefault(); this.vel.zoom += e.deltaY * 0.0015 * this.zoomSpeed; }, { passive: false });
    return this;
  }
  /** Remove the input listeners added by attach(). */
  detach() {
    for (const off of this._off.splice(0)) off();
    return this;
  }
  update(dt) {
    const k = Math.min(1, this.damping * dt), v = this.vel;
    this.yaw += v.yaw * k + this.autoRotate * dt;
    this.pitch = Math.max(this.minPitch, Math.min(this.maxPitch, this.pitch + v.pitch * k));
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
    (o.parent || view.canvas.parentElement || document.body).appendChild(this.el);
    this.extra = o.extra || null;
    this.frames = 0; this.t0 = performance.now(); this.fps = 0;
  }
  destroy() { this.el.remove(); }
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
  return div;
}
