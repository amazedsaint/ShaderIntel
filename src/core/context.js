// Context: device acquisition, capability detection, command batching and the
// uniform ring. Everything recorded through a Context (kernels, clears, copies,
// render passes) goes into one command encoder; compute dispatches share a
// single compute pass until something forces it to end. flush() submits it all.
import { GArray } from './buffer.js';
import { Kernel } from './kernel.js';
import { Profiler } from './profiler.js';
import { arrayCtor, roundUp } from './types.js';

export class SIError extends Error {
  constructor(msg) { super(msg); this.name = 'ShaderIntelError'; }
}

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
  if (typeof navigator === 'undefined' || !navigator.gpu)
    throw new SIError('WebGPU is not available. Use Chrome/Edge 113+, Safari 26+ or Firefox 141+ (Windows), served from https:// or http://localhost.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: opts.powerPreference || 'high-performance' });
  if (!adapter) throw new SIError('No WebGPU adapter found (GPU blocklisted or disabled).');
  const disabled = new Set();
  if (opts.subgroups === false) disabled.add('subgroups');
  if (opts.f16 === false) disabled.add('shader-f16');
  if (opts.timestamps === false) disabled.add('timestamp-query');
  const requiredFeatures = WANT_FEATURES.filter((f) => adapter.features.has(f) && !disabled.has(f));
  const requiredLimits = {};
  for (const k of WANT_LIMITS) if (k in adapter.limits) requiredLimits[k] = adapter.limits[k];
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits, label: 'ShaderIntel' });
  return new Context(adapter, device, opts);
}

export class Context {
  constructor(adapter, device, opts = {}) {
    this.adapter = adapter;
    this.device = device;
    this.limits = device.limits;
    const info = adapter.info || {};
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
    this._ring = new UniformRing(this);
    this.profiler = new Profiler(this);
    this.lost = device.lost.then((i) => { console.error(`ShaderIntel: device lost (${i.reason}): ${i.message}`); return i; });
    device.addEventListener?.('uncapturederror', (e) => console.error('ShaderIntel WebGPU error:', e.error?.message || e));
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
      destroy() { texture.destroy(); } };
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
    this.endPass();
    const q = this.device.queue;
    this._ring.upload(q);
    if (!this._enc) return;
    const collect = this.profiler.resolve(this._enc);
    q.submit([this._enc.finish()]);
    this._enc = null;
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
    const bytes = Math.min(total, roundUp(count * stride, 4));
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
    const module = this.device.createShaderModule({ code, label });
    const ready = module.getCompilationInfo().then((info) => {
      const errors = info.messages.filter((m) => m.type === 'error');
      const warnings = info.messages.filter((m) => m.type === 'warning');
      if (warnings.length) console.warn(formatMessages(label, code, warnings));
      if (errors.length) {
        const msg = formatMessages(label, code, errors);
        console.error(msg);
        throw new SIError(msg);
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
    const scope = d.popErrorScope().then((err) => {
      if (err) { const m = `ShaderIntel: pipeline "${name}": ${err.message}`; console.error(m); throw new SIError(m); }
    });
    c = { pipeline, bgl, module, ready: Promise.all([ready, scope]).then(() => true) };
    c.ready.catch(() => {});
    this._pipelines.set(code, c);
    return c;
  }

  destroy() {
    this.flush();
    this.device.destroy();
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

/** Per-batch ring of 256-byte uniform slots addressed with dynamic offsets. */
class UniformRing {
  constructor(ctx, size = 1 << 20) {
    this.ctx = ctx;
    this.size = size;
    this.staging = new ArrayBuffer(size);
    this.u32 = new Uint32Array(this.staging);
    this.buffer = ctx.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'si-uniform-ring' });
    this.used = 0;
    this.batch = 0;
    this.gen = 0;
    this.align = Math.max(256, ctx.limits.minUniformBufferOffsetAlignment);
  }

  /** Copy the kernel's uniform bytes into a slot; identical consecutive data reuses its slot. */
  push(kernel) {
    const src = kernel.u32;
    if (kernel._ringBatch === this.batch) {
      const base = kernel._ringOff >>> 2;
      let same = true;
      for (let k = 0; k < src.length; k++) if (this.u32[base + k] !== src[k]) { same = false; break; }
      if (same) return kernel._ringOff;
    }
    if (this.used + kernel.U.size > this.size) this.ctx.flush();
    const off = this.used;
    this.u32.set(src, off >>> 2);
    this.used = roundUp(off + kernel.U.size, this.align);
    kernel._ringBatch = this.batch;
    kernel._ringOff = off;
    return off;
  }

  upload(queue) {
    if (this.used) queue.writeBuffer(this.buffer, 0, this.staging, 0, this.used);
    this.used = 0;
    this.batch++;
  }
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
