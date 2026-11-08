// GArray: a typed GPU array. One storage buffer of `length` elements of a WGSL
// type (f32, vec4f, u32, mat3x3f, ...) with upload / readback helpers.
import { layoutOf, arrayCtor } from './types.js';
import { SIError } from './errors.js';

const err = (msg) => new SIError(msg, 'invalid-argument');

let nextId = 1;

export class GArray {
  /**
   * @param {import('./context.js').Context} ctx
   * @param {string} type     WGSL element type ('f32', 'vec4f', 'u32', ...)
   * @param {number|ArrayLike} lengthOrData
   * @param {object} [opts]   { usage, label, atomic }
   */
  constructor(ctx, type, lengthOrData, opts = {}) {
    this.ctx = ctx;
    this.id = nextId++;
    this.layout = layoutOf(type);
    this.type = this.layout.type;
    this.stride = this.layout.stride;
    this.label = opts.label || `${this.type}[]`;
    const data = typeof lengthOrData === 'number' ? null : lengthOrData;
    this.length = data ? Math.floor(data.length / this.layout.components) : lengthOrData;
    if (!(this.length > 0) || !Number.isFinite(this.length)) throw err(`array length must be a positive number (got ${this.length})`);
    this.length = Math.floor(this.length);
    this.byteLength = Math.max(16, Math.ceil((this.length * this.stride) / 16) * 16);
    const GU = GPUBufferUsage;
    this.usage = GU.STORAGE | GU.COPY_SRC | GU.COPY_DST | (opts.usage || 0);
    this.buffer = ctx.device.createBuffer({ size: this.byteLength, usage: this.usage, label: this.label });
    if (data) this.write(data);
  }

  /** Scalars per element as laid out in memory (vec3 occupies 4 slots). */
  get slots() { return this.stride / (this.layout.scalar === 'f16' ? 2 : 4); }

  /**
   * Upload data. Accepts a typed array / number[] of tightly packed components
   * (vec3 data is padded to the 16-byte stride automatically).
   */
  write(data, elementOffset = 0) {
    const L = this.layout;
    const Ctor = arrayCtor(this.type);
    const comps = L.components, slots = this.slots;
    let src;
    if (slots === comps) {
      src = data instanceof Ctor ? data : Ctor.from(data);
    } else {
      // pad vec3 / mat3 columns to their aligned stride
      const n = Math.floor(data.length / comps);
      src = new Ctor(n * slots);
      if (L.cols) {
        const colSlots = L.colStride / (Ctor.BYTES_PER_ELEMENT);
        for (let e = 0; e < n; e++)
          for (let c = 0; c < L.cols; c++)
            for (let r = 0; r < L.rows; r++)
              src[e * slots + c * colSlots + r] = data[e * comps + c * L.rows + r];
      } else {
        for (let e = 0; e < n; e++)
          for (let k = 0; k < comps; k++) src[e * slots + k] = data[e * comps + k];
      }
    }
    if (!(elementOffset >= 0) || elementOffset > this.length)
      throw err(`${this.label}.write: element offset ${elementOffset} is outside the array (length ${this.length})`);
    const room = this.byteLength - elementOffset * this.stride;
    if (src.byteLength > room)
      throw err(`${this.label}.write: ${src.byteLength} bytes at element ${elementOffset} do not fit (the array has room for ${room} bytes)`);
    const aligned = src.byteLength & ~3;
    if (aligned > 0) this.ctx.device.queue.writeBuffer(this.buffer, elementOffset * this.stride, src.buffer, src.byteOffset, aligned);
    return this;
  }

  /**
   * Like write(), but ordered with recorded work (see Context.writeOrdered): dispatches recorded
   * before see the old data, those after see the new. Use it between dispatches of one batch.
   */
  writeOrdered(data, elementOffset = 0) {
    if (!(elementOffset >= 0) || elementOffset > this.length) throw err(`${this.label}.writeOrdered: element offset ${elementOffset} is outside the array`);
    const Ctor = arrayCtor(this.type);
    const src = this.slots === this.layout.components ? (data instanceof Ctor ? data : Ctor.from(data)) : null;
    if (!src) return this.write(data, elementOffset);   // padded types: rare, keep the queue path
    if (src.byteLength > this.byteLength - elementOffset * this.stride) throw err(`${this.label}.writeOrdered: the data does not fit`);
    this.ctx.writeOrdered(this, src, elementOffset * this.stride);
    return this;
  }

  /** Fill with zero bytes (recorded into the current batch, ordered with dispatches). */
  clear() { this.ctx.clear(this); return this; }

  /**
   * Read back to the CPU. Resolves to a typed array with `slots` scalars per
   * element (vec3 arrays keep their padding; use readPacked() to drop it).
   */
  read(count = this.length) { return this.ctx.read(this, count); }

  async readPacked(count = this.length) {
    const raw = await this.read(count);
    const L = this.layout, comps = L.components, slots = this.slots;
    if (comps === slots) return raw;
    const Ctor = raw.constructor, out = new Ctor(count * comps);
    if (L.cols) {
      const colSlots = L.colStride / Ctor.BYTES_PER_ELEMENT;
      for (let e = 0; e < count; e++)
        for (let c = 0; c < L.cols; c++)
          for (let r = 0; r < L.rows; r++) out[e * comps + c * L.rows + r] = raw[e * slots + c * colSlots + r];
    } else {
      for (let e = 0; e < count; e++) for (let k = 0; k < comps; k++) out[e * comps + k] = raw[e * slots + k];
    }
    return out;
  }

  /** Free the GPU memory now (idempotent); the array must not be used afterwards. */
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    // work recorded but not yet submitted may use this buffer; a destroyed buffer would make the
    // whole submit fail (every other dispatch in it lost), so submit that work first
    try { this.ctx.flush(); } catch { /* inside an open render pass: the buffer is destroyed after the frame */ }
    this.buffer.destroy();
  }
}
