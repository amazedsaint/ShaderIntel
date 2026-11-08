// WGSL host-shareable type layout (WGSL spec §14.4 "Memory Layout").
// Every size/alignment here is what the GPU expects, so JS can pack data
// byte-exactly without guessing.

const SCALARS = {
  f32: { size: 4, align: 4, array: Float32Array },
  i32: { size: 4, align: 4, array: Int32Array },
  u32: { size: 4, align: 4, array: Uint32Array },
  f16: { size: 2, align: 2, array: Uint16Array }, // raw half bits on the JS side
};

const SUFFIX = { f: 'f32', i: 'i32', u: 'u32', h: 'f16' };

/** Expand shorthands: vec3f -> vec3<f32>, mat4x4f -> mat4x4<f32>. */
export function canonical(type) {
  type = type.replace(/\s+/g, '');
  let m = /^vec([234])([fiuh])$/.exec(type);
  if (m) return `vec${m[1]}<${SUFFIX[m[2]]}>`;
  m = /^mat([234])x([234])([fh])$/.exec(type);
  if (m) return `mat${m[1]}x${m[2]}<${SUFFIX[m[3]]}>`;
  return type;
}

/**
 * Layout of a WGSL type: { size, align, scalar, components, stride }.
 * `stride` is the array element stride (size rounded up to alignment).
 */
export function layoutOf(type) {
  const t = canonical(type);
  if (SCALARS[t]) {
    const s = SCALARS[t];
    return { type: t, size: s.size, align: s.align, scalar: t, components: 1, stride: s.size };
  }
  let m = /^atomic<(u32|i32)>$/.exec(t);
  if (m) return { type: t, size: 4, align: 4, scalar: m[1], components: 1, stride: 4, atomic: true };
  m = /^vec([234])<(f32|i32|u32|f16)>$/.exec(t);
  if (m) {
    const n = +m[1], s = SCALARS[m[2]];
    const align = (n === 2 ? 2 : 4) * s.size;
    const size = n * s.size;
    return { type: t, size, align, scalar: m[2], components: n, stride: roundUp(size, align) };
  }
  m = /^mat([234])x([234])<(f32|f16)>$/.exec(t);
  if (m) {
    const cols = +m[1], rows = +m[2];
    const col = layoutOf(`vec${rows}<${m[3]}>`);
    const colStride = roundUp(col.size, col.align);
    return {
      type: t, size: cols * colStride, align: col.align, scalar: m[3],
      components: cols * rows, cols, rows, colStride, stride: cols * colStride,
    };
  }
  throw new Error(`ShaderIntel: unsupported WGSL type "${type}"`);
}

export const roundUp = (x, a) => Math.ceil(x / a) * a;

/**
 * Lay out a flat struct of { name: type } members (no nested arrays) using
 * WGSL rules. Returns { fields: [{name,type,offset,layout}], size, align, wgsl }.
 */
export function structLayout(name, members) {
  const fields = [];
  let offset = 0, align = 4;
  for (const [field, type] of Object.entries(members)) {
    const L = layoutOf(type);
    offset = roundUp(offset, L.align);
    fields.push({ name: field, type: L.type, offset, layout: L });
    offset += L.size;
    align = Math.max(align, L.align);
  }
  // Uniform buffers are bound in 16-byte granules; round the struct to 16.
  const size = Math.max(16, roundUp(offset, Math.max(align, 16)));
  const body = fields.map((f) => `  ${f.name}: ${f.type},`).join('\n');
  return { name, fields, size, align, wgsl: `struct ${name} {\n${body}\n}` };
}

/** Write a JS value (number | array | typed array) into a DataView at a field's offset. */
export function writeField(view, field, value) {
  const L = field.layout;
  const base = field.offset;
  const put = (off, v) => {
    switch (L.scalar) {
      case 'f32': view.setFloat32(off, v, true); break;
      case 'i32': view.setInt32(off, v, true); break;
      case 'u32': view.setUint32(off, v >>> 0, true); break;
      case 'f16': view.setUint16(off, toHalf(v), true); break;
    }
  };
  const ss = SCALARS[L.scalar].size;
  if (L.components === 1) {
    put(base, typeof value === 'boolean' ? +value : value);
  } else if (L.cols) {
    // Matrices are given column-major (like gl-matrix / WGSL).
    for (let c = 0; c < L.cols; c++)
      for (let r = 0; r < L.rows; r++)
        put(base + c * L.colStride + r * ss, value[c * L.rows + r] ?? 0);
  } else {
    for (let k = 0; k < L.components; k++) put(base + k * ss, value[k] ?? 0);
  }
}

// IEEE 754 binary16 conversion (round-to-nearest-even), used for f16 uniforms/arrays.
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
export function toHalf(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp >= 31) return sign | (((x >>> 23) & 0xff) === 0xff && mant ? 0x7e00 : 0x7c00);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = sign | (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return h;
}

export function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Typed-array constructor for an element type (vectors map to their scalar). */
export function arrayCtor(type) {
  return SCALARS[layoutOf(type).scalar].array;
}
