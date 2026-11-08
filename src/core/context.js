// Context: device acquisition, capability detection, command batching and the
// uniform ring. Everything recorded through a Context (kernels, clears, copies,
// render passes) goes into one command encoder; compute dispatches share a
// single compute pass until something forces it to end. flush() submits it all.
import { GArray } from './buffer.js';
import { Kernel } from './kernel.js';
import { Profiler } from './profiler.js';
import { arrayCtor, roundUp } from './types.js';

import { SIError } from './errors.js';
export { SIError };

const WANT_FEATURES = [
  'subgroups', 'shader-f16', 'timestamp-query', 'float32-filterable', 'float32-blendable',
  'rg11b10ufloat-renderable', 'indirect-first-instance', 'bgra8unorm-storage',
];
const WANT_LIMITS = [
  'maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ', 'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage',
  'maxComputeWorkgroupsPerDimension', 'maxTextureDimension2D', 'maxTextureDimension3D',
  'maxColorAttachmentBytesPerSample', 'maxSampledTexturesPerShaderStage', 'maxBindGroups',
];

/**
 * Initialise WebGPU.
 * @param {object} [opts]
 *   powerPreference: 'high-performance' | 'low-power'
 *   subgroups / f16 / timestamps: false to force the portable path (testing)
 * @returns {Promise<Context>}
 */
export async function init(opts = {}) {
  if (!isSupported())
    throw new SIError('WebGPU is not available. Use Chrome/Edge 113+, Safari 26+ or Firefox 141+ (Windows), served from https:// or http://localhost.', 'unsupported');
  // Share a device created elsewhere (three.js, another library): pass { device } (and its adapter if known).
  if (opts.device) return new Context(opts.adapter || null, opts.device, opts);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: opts.powerPreference || 'high-performance' });
  if (!adapter) throw new SIError('No WebGPU adapter found (the GPU may be blocklisted or WebGPU disabled in this browser).', 'no-adapter');
  const disabled = new Set();
  if (opts.subgroups === false) disabled.add('subgroups');
  if (opts.f16 === false) disabled.add('shader-f16');
  if (opts.timestamps === false) disabled.add('timestamp-query');
  const requiredFeatures = WANT_FEATURES.filter((f) => adapter.features.has(f) && !disabled.has(f));
  const requiredLimits = {};
  for (const k of WANT_LIMITS) if (k in adapter.limits) requiredLimits[k] = adapter.limits[k];
  let device;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits, label: 'ShaderIntel' });
  } catch (e) {
    throw new SIError(`Could not create a WebGPU device: ${e?.message || e}`, 'no-device');
  }
  return new Context(adapter, device, opts);
}

/** True when this environment exposes WebGPU (navigator.gpu); a GPU adapter may still be unavailable. */
export function isSupported() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export class Context {
  constructor(adapter, device, opts = {}) {
    this.adapter = adapter;
    this.device = device;
    this.limits = device.limits;
    this.destroyed = false;
    this.ownsDevice = !opts.device;   // a shared device is left alive by destroy()
    const info = adapter?.info || device.adapterInfo || {};
    const has = (f) => device.features.has(f);
    this.caps = {
      subgroups: has('subgroups'),
      f16: has('shader-f16'),
      timestamps: has('timestamp-query'),
      float32Filterable: has('float32-filterable'),
      float32Blendable: has('float32-blendable'),
      subgroupMin: info.subgroupMinSize || 0,
      subgroupMax: info.subgroupMaxSize || 0,
      subgroupId: !!navigator.gpu.wgslLanguageFeatures?.has('subgroup_id') && has('subgroups') && opts.subgroupId !== false,
      vendor: info.vendor || 'unknown',
      architecture: info.architecture || '',
      workgroupStorage: device.limits.maxComputeWorkgroupStorageSize,
    };
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.stats = { dispatches: 0, submits: 0 };
    this._enc = null;
    this._pass = null;
    this._passTS = null;
    this._inRegion = false;
    this._pipelines = new Map();
    this._samplers = new Map();
    this._texDead = new WeakSet();
    this._ring = new UniformRing(this);
    this.profiler = new Profiler(this);
    this.lost = device.lost.then((i) => {
      this.destroyed = true;
      if (i.reason !== 'destroyed') console.error(`ShaderIntel: device lost (${i.reason}): ${i.message}`);
      return i;
    });
    this._onUncaptured = (e) => console.error('ShaderIntel WebGPU error:', e.error?.message || e);
    device.addEventListener?.('uncapturederror', this._onUncaptured);
  }

  // ------------------------------------------------------------- resources
  /** Typed GPU array: ctx.array('vec4f', n) or ctx.array('f32', float32Array). */
  array(type, lengthOrData, opts) { return new GArray(this, type, lengthOrData, opts); }

  /** Compute kernel from a WGSL body (see core/kernel.js). */
  kernel(desc) { return new Kernel(this, desc); }

  /** Cached sampler. */
  sampler(filter = 'linear', wrap = 'clamp-to-edge') {
    const key = filter + wrap;
    let s = this._samplers.get(key);
    if (!s) {
      s = this.device.createSampler({ magFilter: filter, minFilter: filter, mipmapFilter: filter,
        addressModeU: wrap, addressModeV: wrap, addressModeW: wrap });
      this._samplers.set(key, s);
    }
    return s;
  }

  /** Texture with a cached default view: { texture, view, id, width, height, depth, format }. */
  texture(desc) {
    const size = desc.size;
    const width = size[0], height = size[1] || 1, depth = size[2] || 1;
    const TU = GPUTextureUsage;
    const texture = this.device.createTexture({
      size: [width, height, depth],
      format: desc.format || 'rgba16float',
      dimension: desc.dimension || (size.length === 3 && depth > 1 ? '3d' : '2d'),
      usage: desc.usage ?? (TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_DST | TU.COPY_SRC),
      sampleCount: desc.sampleCount || 1,
      label: desc.label,
    });
    return { texture, view: texture.createView(), id: 't' + (++texId), width, height, depth, format: texture.format,
      destroy: () => { if (this._texDead.has(texture)) return; this._texDead.add(texture); try { this.flush(); } catch { /* see GArray.destroy */ } texture.destroy(); } };
  }

  // ------------------------------------------------------------- recording
  get encoder() {
    if (!this._enc) this._enc = this.device.createCommandEncoder();
    return this._enc;
  }

  /** The open compute pass (created on demand). */
  computePass() {
    if (!this._pass) {
      const desc = {};
      if (this._passTS) { desc.timestampWrites = this._passTS; this._passTS = null; }
      this._pass = this.encoder.beginComputePass(desc);
    }
    return this._pass;
  }

  endPass() {
    if (this._pass) { this._pass.end(); this._pass = null; }
  }

  /** Begin a render pass in the current batch (caller ends it). */
  renderPass(desc) {
    this.endPass();
    return this.encoder.beginRenderPass(desc);
  }

  /**
   * Time a block of compute work on the GPU. The block becomes one compute
   * pass with timestamp writes; see profiler.report().
   */
  region(label, fn) {
    if (!this.profiler.enabled || this._inRegion) return fn();
    this.endPass();
    this._passTS = this.profiler.alloc(label);
    this._inRegion = true;
    try {
      return fn();
    } finally {
      this.computePass();
      this.endPass();
      this._passTS = null;
      this._inRegion = false;
    }
  }

  /** Submit everything recorded so far. */
  flush() {
    if (this.destroyed) { this._enc = null; this._pass = null; return; }
    this.endPass();
    const q = this.device.queue;
    this._ring.upload(q);
    if (!this._enc) return;
    const collect = this.profiler.resolve(this._enc);
    q.submit([this._enc.finish()]);
    this._enc = null;
    if (this._staging) { for (const b of this._staging) b.destroy(); this._staging = null; }   // freed after the GPU is done with them
    this.stats.submits++;
    if (collect) collect();
  }

  /** Flush and wait for the GPU to finish. */
  async idle() {
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
  }

  // ------------------------------------------------------------- data movement
  /** Zero an array (or any GPUBuffer) inside the current compute pass. */
  clear(arr) { return this.fill(arr, 0); }

  /** Fill every 32-bit word with `value` (u32 bits; use fillFloat for floats). */
  fill(arr, value = 0) {
    if (!this._fillK) {
      this._fillK = this.kernel({ name: 'si_fill', buffers: { dst: { type: 'vec4u' } }, uniforms: { v: 'u32' },
        code: 'dst[i] = vec4u(u.v);' });
    }
    const bytes = arr.byteLength ?? arr.size;
    if (bytes & 15) throw new SIError('fill(): buffer size must be a multiple of 16 bytes');
    this._fillK.run(bytes >>> 4, { v: value >>> 0 }, { dst: arr });
    return this;
  }

  fillFloat(arr, x) { f32tmp[0] = x; return this.fill(arr, u32tmp[0]); }

  /**
   * Write `data` into `dst` (a GArray or GPUBuffer) at `byteOffset`, ordered with the recorded work:
   * unlike queue.writeBuffer (which lands before everything not yet submitted), dispatches recorded
   * before this call see the old contents and those recorded after it see the new ones. For small,
   * frequent uploads (colliders, sources, parameters) between dispatches of one batch.
   */
  writeOrdered(dst, data, byteOffset = 0) {
    const src = ArrayBuffer.isView(data) ? data : new Float32Array(data);
    const bytes = src.byteLength & ~3;
    if (!bytes) return this;
    const target = dst.buffer instanceof GPUBuffer ? dst.buffer : dst;
    if (byteOffset + bytes > target.size) throw new SIError(`writeOrdered: ${bytes} bytes at ${byteOffset} do not fit in ${target.size}`, 'invalid-argument');
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Uint8Array(staging.getMappedRange()).set(new Uint8Array(src.buffer, src.byteOffset, bytes));
    staging.unmap();
    this.endPass();
    this.encoder.copyBufferToBuffer(staging, 0, target, byteOffset, bytes);
    (this._staging ||= []).push(staging);   // destroyed once the batch is submitted
    return this;
  }

  /** Copy `bytes` (default: all) between arrays/buffers. */
  copy(src, dst, bytes, srcOffset = 0, dstOffset = 0) {
    const s = src.buffer instanceof GPUBuffer ? src.buffer : src;
    const d = dst.buffer instanceof GPUBuffer ? dst.buffer : dst;
    bytes = bytes ?? Math.min(src.byteLength ?? s.size, dst.byteLength ?? d.size);
    this.endPass();
    this.encoder.copyBufferToBuffer(s, srcOffset, d, dstOffset, bytes);
    return this;
  }

  /** Read an array back to the CPU (flushes). Resolves to a typed array. */
  async read(arr, count = arr.length) {
    const stride = arr.stride ?? 4;
    const total = arr.byteLength ?? arr.size;
    const bytes = Math.min(total, roundUp(Math.max(0, count) * stride, 4));
    const Ctor0 = arr.type ? arrayCtor(arr.type) : Uint32Array;
    if (bytes === 0) return new Ctor0(0);
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.endPass();
    this.encoder.copyBufferToBuffer(arr.buffer instanceof GPUBuffer ? arr.buffer : arr, 0, staging, 0, bytes);
    this.flush();
    await staging.mapAsync(GPUMapMode.READ);
    const Ctor = arr.type ? arrayCtor(arr.type) : Uint32Array;
    const out = new Ctor(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /**
   * Read a 2D/3D texture back to the CPU (needs COPY_SRC usage). Returns
   * { width, height, depth, channels, data: Float32Array } (half floats are expanded).
   */
  async readTexture(tex) {
    const t = tex.texture || tex;
    const info = { r32float: [4, 1], rgba16float: [8, 4], rgba32float: [16, 4], rg32float: [8, 2], rgba8unorm: [4, 4], r32uint: [4, 1] }[t.format];
    if (!info) throw new SIError(`readTexture: unsupported format ${t.format}`);
    const [bpp, ch] = info;
    const W = t.width, H = t.height, D = t.depthOrArrayLayers;
    const bpr = Math.ceil((W * bpp) / 256) * 256;
    const buf = this.device.createBuffer({ size: bpr * H * D, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.endPass();
    this.encoder.copyTextureToBuffer({ texture: t }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: H }, [W, H, D]);
    this.flush();
    await buf.mapAsync(GPUMapMode.READ);
    const dv = new DataView(buf.getMappedRange());
    const out = new Float32Array(W * H * D * ch);
    for (let z = 0; z < D; z++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const o = (z * H + y) * bpr + x * bpp, q = ((z * H + y) * W + x) * ch;
          for (let c = 0; c < ch; c++) {
            if (bpp / ch === 2) out[q + c] = halfToFloat(dv.getUint16(o + 2 * c, true));
            else if (t.format === 'rgba8unorm') out[q + c] = dv.getUint8(o + c) / 255;
            else if (t.format === 'r32uint') out[q + c] = dv.getUint32(o + 4 * c, true);
            else out[q + c] = dv.getFloat32(o + 4 * c, true);
          }
        }
    buf.unmap();
    buf.destroy();
    return { width: W, height: H, depth: D, channels: ch, data: out };
  }

  // ------------------------------------------------------------- shaders
  /** Create a shader module and a promise that rejects with readable errors. */
  shaderModule(code, label = 'shader') {
    // Parse errors would also surface as an uncaptured validation error; catch that copy here,
    // so a bad shader is reported once, formatted with its source lines (below).
    this.device.pushErrorScope('validation');
    const module = this.device.createShaderModule({ code, label });
    this.device.popErrorScope().catch(() => {});
    const ready = module.getCompilationInfo().then((info) => {
      const errors = info.messages.filter((m) => m.type === 'error');
      const warnings = info.messages.filter((m) => m.type === 'warning');
      if (warnings.length) console.warn(formatMessages(label, code, warnings));
      if (errors.length) {
        const msg = formatMessages(label, code, errors);
        console.error(msg);
        throw new SIError(msg, 'compile');
      }
    });
    ready.catch(() => {});
    return { module, ready };
  }

  _compile(code, slots, name) {
    let c = this._pipelines.get(code);
    if (c) return c;
    const d = this.device;
    const { module, ready } = this.shaderModule(code, name);
    const bgl = d.createBindGroupLayout({
      label: name,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        ...slots.map((s, k) => ({ binding: k + 1, visibility: GPUShaderStage.COMPUTE, ...s.layout })),
      ],
    });
    d.pushErrorScope('validation');
    const pipeline = d.createComputePipeline({
      label: name,
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: 'main' },
    });
    const scope = d.popErrorScope().then(async (err) => {
      if (!err) return;
      const m = `ShaderIntel: pipeline "${name}": ${err.message}`;
      if (await ready.then(() => true, () => false)) console.error(m);   // else the WGSL errors were reported already
      throw new SIError(m);
    });
    c = { pipeline, bgl, module, ready: Promise.all([ready, scope]).then(() => true) };
    c.ready.catch(() => {});
    this._pipelines.set(code, c);
    return c;
  }

  /**
   * Submit pending work and release the device (and with it every buffer, texture and pipeline
   * made on it). A device passed in with init({ device }) is not destroyed, only detached from.
   */
  destroy() {
    if (this.destroyed) return;
    try { this.flush(); } catch { /* the device may be lost already */ }
    this.destroyed = true;
    this.device.removeEventListener?.('uncapturederror', this._onUncaptured);
    if (this.ownsDevice) this.device.destroy();
    else { this._ring.destroy(); this.profiler.destroy(); this._gemmTmp?.destroy(); }
  }
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

let texId = 0;
const f32tmp = new Float32Array(1);
const u32tmp = new Uint32Array(f32tmp.buffer);

/**
 * Per-batch ring of 256-byte uniform slots addressed with dynamic offsets. When a batch needs
 * more than one segment (1 MB), another segment is opened instead of flushing: a flush is
 * impossible while a render pass is being recorded. Bind groups name their segment (gen).
 */
class UniformRing {
  constructor(ctx, size = 1 << 20) {
    this.ctx = ctx;
    this.size = size;
    this.align = Math.max(256, ctx.limits.minUniformBufferOffsetAlignment);
    this.segments = [];
    this.seg = -1;
    this.batch = 0;
    this._open();
  }

  _open() {
    this.seg++;
    if (!this.segments[this.seg]) {
      const staging = new ArrayBuffer(this.size);
      this.segments.push({ staging, u32: new Uint32Array(staging), used: 0,
        buffer: this.ctx.device.createBuffer({ size: this.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `si-uniform-ring-${this.seg}` }) });
    }
    const s = this.segments[this.seg];
    s.used = 0;
    this.buffer = s.buffer; this.u32 = s.u32; this.gen = this.seg; this.used = 0;
  }

  /** Copy the kernel's uniform bytes into a slot; identical consecutive data reuses its slot. */
  push(kernel) {
    const src = kernel.u32;
    if (kernel._ringBatch === this.batch && kernel._ringSeg === this.seg) {
      const base = kernel._ringOff >>> 2;
      let same = true;
      for (let k = 0; k < src.length; k++) if (this.u32[base + k] !== src[k]) { same = false; break; }
      if (same) return kernel._ringOff;
    }
    if (this.used + kernel.U.size > this.size) { this.segments[this.seg].used = this.used; this._open(); }
    const off = this.used;
    this.u32.set(src, off >>> 2);
    this.used = roundUp(off + kernel.U.size, this.align);
    kernel._ringBatch = this.batch;
    kernel._ringSeg = this.seg;
    kernel._ringOff = off;
    return off;
  }

  upload(queue) {
    this.segments[this.seg].used = this.used;
    for (let k = 0; k <= this.seg; k++) {
      const s = this.segments[k];
      if (s.used) queue.writeBuffer(s.buffer, 0, s.staging, 0, s.used);
      s.used = 0;
    }
    this.seg = -1;
    this._open();
    this.batch++;
  }

  destroy() { for (const s of this.segments) s.buffer.destroy(); }
}

function formatMessages(label, code, messages) {
  const lines = code.split('\n');
  const out = [`ShaderIntel: WGSL problems in "${label}":`];
  for (const m of messages) {
    out.push(`  ${m.type} at line ${m.lineNum}:${m.linePos}: ${m.message}`);
    for (let l = Math.max(1, m.lineNum - 2); l <= Math.min(lines.length, m.lineNum + 2); l++)
      out.push(`  ${l === m.lineNum ? '>' : ' '} ${String(l).padStart(4)} | ${lines[l - 1]}`);
  }
  return out.join('\n');
}
