// ShaderIntel — WebGPU physics + shader library. Single-file build of src/ (50 modules).
(function (global) {
'use strict';
// ---- src/core/types.js
const __m3 = (() => {
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
function canonical(type) {
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
function layoutOf(type) {
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

const roundUp = (x, a) => Math.ceil(x / a) * a;

/**
 * Lay out a flat struct of { name: type } members (no nested arrays) using
 * WGSL rules. Returns { fields: [{name,type,offset,layout}], size, align, wgsl }.
 */
function structLayout(name, members) {
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
function writeField(view, field, value) {
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
function toHalf(v) {
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

function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Typed-array constructor for an element type (vectors map to their scalar). */
function arrayCtor(type) {
  return SCALARS[layoutOf(type).scalar].array;
}
return { canonical, layoutOf, roundUp, structLayout, writeField, toHalf, fromHalf, arrayCtor };
})();

// ---- src/core/buffer.js
const __m2 = (() => {
const layoutOf = __m3.layoutOf;
const arrayCtor = __m3.arrayCtor;
// GArray: a typed GPU array. One storage buffer of `length` elements of a WGSL
// type (f32, vec4f, u32, mat3x3f, ...) with upload / readback helpers.


let nextId = 1;

class GArray {
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
    if (!(this.length > 0)) throw new Error(`ShaderIntel: array length must be > 0 (got ${this.length})`);
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
    const bytes = Math.min(src.byteLength, this.byteLength - elementOffset * this.stride);
    const aligned = bytes & ~3;
    if (aligned > 0) this.ctx.device.queue.writeBuffer(this.buffer, elementOffset * this.stride, src.buffer, src.byteOffset, aligned);
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

  destroy() { this.buffer.destroy(); }
}
return { GArray };
})();

// ---- src/core/wgsl.js
const __m5 = (() => {
// WGSL module system. WGSL has no #include, so ShaderIntel ships its shader
// library as named modules that a tiny linker stitches together:
//   - dependencies are resolved transitively and each module is emitted once;
//   - `enable` / `requires` / `diagnostic` directives are hoisted to the top
//     (WGSL requires directives before any declaration);
//   - modules may be functions of the device capabilities, so one name can
//     resolve to a subgroup-accelerated or a portable body.
// WGSL allows module-scope declarations in any order, so no topological sort
// of the bodies is needed — only de-duplication.

const registry = new Map();

/**
 * Register a WGSL module.
 * @param {string} name   e.g. 'noise'
 * @param {object} def    { deps?: string[], code: string | (caps) => string, doc?: string }
 */
function defineModule(name, def) {
  if (typeof def === 'string') def = { code: def };
  registry.set(name, { name, deps: def.deps || [], code: def.code, doc: def.doc || '' });
}

const hasModule = (name) => registry.has(name);
const moduleNames = () => [...registry.keys()];
const moduleInfo = (name) => registry.get(name);

const IMPORT_RE = /^[ \t]*(?:\/\/)?[ \t]*#import[ \t]+([^\n]+)$/gm;
const DIRECTIVE_RE = /^[ \t]*(enable|requires)[ \t]+([^;]+);[ \t]*$/gm;
const DIAG_RE = /^[ \t]*diagnostic[ \t]*\([^)]*\)[ \t]*;[ \t]*$/gm;

/** Pull `#import a, b` lines out of user code. */
function parseImports(code) {
  const names = [];
  const stripped = code.replace(IMPORT_RE, (_, list) => {
    for (const n of list.split(/[\s,]+/)) if (n) names.push(n);
    return '';
  });
  return { code: stripped, imports: names };
}

function extractDirectives(code, enables, requires, diags) {
  code = code.replace(DIRECTIVE_RE, (_, kind, list) => {
    const set = kind === 'enable' ? enables : requires;
    for (const n of list.split(',')) if (n.trim()) set.add(n.trim());
    return '';
  });
  return code.replace(DIAG_RE, (d) => { diags.add(d.trim()); return ''; });
}

/**
 * Link user code with library modules.
 * @returns {string} complete WGSL source
 */
function link(userCode, imports = [], caps = {}) {
  const parsed = parseImports(userCode || '');
  const order = [];
  const seen = new Set();
  const visit = (name, from) => {
    if (seen.has(name)) return;
    const mod = registry.get(name);
    if (!mod) throw new Error(`ShaderIntel: unknown WGSL module "${name}"${from ? ` (imported by ${from})` : ''}. Known: ${moduleNames().join(', ')}`);
    seen.add(name);
    for (const d of mod.deps) visit(d, name);
    order.push(mod);
  };
  for (const n of [...imports, ...parsed.imports]) visit(n);

  const enables = new Set(), requires = new Set(), diags = new Set();
  const bodies = order.map((m) => {
    const src = typeof m.code === 'function' ? m.code(caps) : m.code;
    return `// ---- module: ${m.name}\n` + extractDirectives(src, enables, requires, diags).trim();
  });
  const user = extractDirectives(parsed.code, enables, requires, diags);
  const head = [
    ...[...enables].map((e) => `enable ${e};`),
    ...[...requires].map((r) => `requires ${r};`),
    ...diags,
  ].join('\n');
  return `${head}\n${bodies.join('\n\n')}\n\n${user}`.trim() + '\n';
}

/** Format a JS number as a WGSL literal of the requested type. */
function literal(value, type = 'f32') {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (type === 'u32') return `${Math.max(0, Math.round(value)) >>> 0}u`;
  if (type === 'i32') return `${Math.round(value) | 0}i`;
  if (!Number.isFinite(value)) throw new Error(`ShaderIntel: non-finite WGSL literal ${value}`);
  const s = String(value);
  return /[.eE]/.test(s) ? s : s + '.0';
}

/** Turn { NAME: value } into `const NAME = value;` lines (type inferred: integers -> i32 unless suffixed). */
function constants(defs = {}) {
  return Object.entries(defs).map(([k, v]) => {
    if (typeof v === 'string') return `const ${k} = ${v};`;
    if (typeof v === 'boolean') return `const ${k} = ${v};`;
    if (Array.isArray(v)) return `const ${k} = vec${v.length}f(${v.map((x) => literal(x)).join(', ')});`;
    return `const ${k} = ${Number.isInteger(v) ? `${v}` : literal(v)};`;
  }).join('\n');
}
return { defineModule, hasModule, moduleNames, moduleInfo, parseImports, link, literal, constants };
})();

// ---- src/core/kernel.js
const __m4 = (() => {
const structLayout = __m3.structLayout;
const writeField = __m3.writeField;
const layoutOf = __m3.layoutOf;
const roundUp = __m3.roundUp;
const link = __m5.link;
const constants = __m5.constants;
// Kernel: write only the body of a compute shader; ShaderIntel generates the
// bindings, the uniform struct (with exact WGSL layout), the entry point, the
// bounds guard and the dispatch geometry.
//
//   const k = ctx.kernel({
//     buffers:  { pos, vel },                 // GArray -> var<storage, read_write>
//     uniforms: { dt: 'f32', g: 'vec3f' },    // -> struct Params, read as u.dt
//     imports:  ['noise'],                    // shader-library modules
//     code: `vel[i] += vec4f(u.g * u.dt, 0); pos[i] += vel[i] * u.dt;`,
//   });
//   k.run(N, { dt: 1 / 60 });
//
// Inside `code` these names are defined:
//   i    linear invocation index (1D kernels)       p     gid (nD kernels)
//   u    uniforms (u.n = count for 1D, u.size = extent for nD)
//   gid, lid, lidx, wid, nwg, wgi (linear workgroup index)
//   sgi, sgs (subgroup lane / size), sgid, nsg (subgroup index / count) when used.
// These names are reserved: declaring a local called e.g. `nsg` in `code` clashes.



const USES_SUBGROUPS = /\bsgi\b|\bsgs\b|\bsgid\b|\bnsg\b|\bsubgroup[A-Z]\w*\s*\(|\bquad[A-Z]\w*\s*\(/;
const USES_SGID = /\bsgid\b|\bnsg\b/;

let kernelId = 0;

class Kernel {
  constructor(ctx, desc) {
    this.ctx = ctx;
    this.id = ++kernelId;
    this.name = desc.name || `kernel${this.id}`;
    let wg = desc.workgroupSize ?? 256;
    if (typeof wg === 'number') wg = [wg];
    this.wg = [wg[0], wg[1] || 1, wg[2] || 1];
    this.wgCount = this.wg[0] * this.wg[1] * this.wg[2];
    this.dims = desc.dims || (this.wg[2] > 1 ? 3 : this.wg[1] > 1 ? 2 : 1);
    this.guard = desc.guard !== false;

    // ---- bindings: binding 0 is the uniform ring slot, the rest follow in declaration order
    this.slots = [];
    for (const [name, spec] of Object.entries(desc.buffers || {})) this.slots.push(bufferSlot(name, spec));
    for (const [name, spec] of Object.entries(desc.textures || {})) this.slots.push(textureSlot(name, spec));
    for (const [name, spec] of Object.entries(desc.samplers || {})) this.slots.push(samplerSlot(name, spec));
    this.slotByName = new Map(this.slots.map((s, k) => [s.name, k]));

    // ---- uniforms (system fields first)
    const sys = this.dims === 1 ? { n: 'u32' } : { size: 'vec3u' };
    const members = { ...sys, ...(desc.uniforms || {}) };
    this.U = structLayout('Params', members);
    this.ubytes = new ArrayBuffer(this.U.size);
    this.uview = new DataView(this.ubytes);
    this.u32 = new Uint32Array(this.ubytes);
    this.fieldByName = new Map(this.U.fields.map((f) => [f.name, f]));
    if (desc.values) this.set(desc.values);

    // ---- WGSL
    const caps = ctx.caps;
    const body = desc.code || '';
    const shared = desc.shared || '';
    const usesSG = USES_SUBGROUPS.test(body) || USES_SUBGROUPS.test(shared) || desc.subgroups;
    if (usesSG && !caps.subgroups)
      throw new Error(`ShaderIntel: kernel "${this.name}" uses subgroup operations but this device lacks the "subgroups" feature`);
    const usesSgid = usesSG && USES_SGID.test(body);
    const nativeSgid = usesSgid && caps.subgroupId;

    const decls = [this.U.wgsl, '@group(0) @binding(0) var<uniform> u: Params;'];
    this.slots.forEach((s, k) => decls.push(`@group(0) @binding(${k + 1}) ${s.decl}`));

    const params = [
      '@builtin(global_invocation_id) gid: vec3u',
      '@builtin(local_invocation_id) lid: vec3u',
      '@builtin(local_invocation_index) lidx: u32',
      '@builtin(workgroup_id) wid: vec3u',
      '@builtin(num_workgroups) nwg: vec3u',
    ];
    if (usesSG) params.push('@builtin(subgroup_invocation_id) sgi: u32', '@builtin(subgroup_size) sgs: u32');
    if (nativeSgid) params.push('@builtin(subgroup_id) sgid: u32', '@builtin(num_subgroups) nsg: u32');

    const pre = ['let wgi = wid.x + wid.y * nwg.x;'];
    if (usesSgid && !nativeSgid) pre.push('let sgid = lidx / sgs;', `let nsg = (${this.wgCount}u + sgs - 1u) / sgs;`);
    if (this.dims === 1) {
      pre.push(`let i = wgi * ${this.wgCount}u + lidx;`);
      if (this.guard) pre.push('if (i >= u.n) { return; }');
    } else {
      pre.push('let p = gid;');
      if (this.guard) pre.push('if (any(gid >= u.size)) { return; }');
    }

    const src = [
      usesSG ? 'enable subgroups;' : '',
      nativeSgid ? 'requires subgroup_id;' : '',
      `const WG = ${this.wgCount}u;`,
      `const WG_SIZE = vec3u(${this.wg.join('u, ')}u);`,
      constants(desc.defines),
      ...decls,
      shared,
      `@compute @workgroup_size(${this.wg.join(', ')})`,
      `fn main(\n  ${params.join(',\n  ')}\n) {`,
      '  ' + pre.join('\n  '),
      body,
      '}',
    ].join('\n');
    this.code = link(src, desc.imports || [], caps);

    // Binding contracts: the element stride of every storage binding, from the linked WGSL
    // (structs included), so bind() can reject a buffer that is not a whole number of elements.
    const structs = parseStructs(this.code);
    for (const s of this.slots) if (s.kind === 'storage' && s.type) s.stride = strideOf(s.type, structs);

    const compiled = ctx._compile(this.code, this.slots, this.name);
    this.pipeline = compiled.pipeline;
    this.bgl = compiled.bgl;
    this.compiled = compiled;
    this.bound = {};
    for (const s of this.slots) if (s.default) this.bound[s.name] = s.default;
    this.bgCache = new Map();
  }

  /** Resolves when compilation finished; rejects with formatted WGSL errors. */
  ready() { return this.compiled.ready; }

  /** Set persistent uniform values (also accepted by run()). */
  set(values) {
    if (!values) return this;
    for (const [k, v] of Object.entries(values)) {
      const f = this.fieldByName.get(k);
      if (!f) throw new Error(`ShaderIntel: kernel "${this.name}" has no uniform "${k}"`);
      writeField(this.uview, f, v);
    }
    return this;
  }

  /** Persistently bind resources by name: k.bind({ pos: arrB }). */
  bind(resources) {
    if (!resources) return this;
    for (const [k, v] of Object.entries(resources)) {
      if (!this.slotByName.has(k)) throw new Error(`ShaderIntel: kernel "${this.name}" has no binding "${k}"`);
      if (this.bound[k] !== v) checkContract(this, this.slots[this.slotByName.get(k)], v);
      this.bound[k] = v;
    }
    return this;
  }

  /**
   * Dispatch over `count` invocations: a number (1D) or [x, y, z] (nD).
   * `uniforms` and `resources` are merged into the persistent state.
   */
  run(count, uniforms, resources) {
    this.set(uniforms);
    this.bind(resources);
    let groups;
    if (this.dims === 1) {
      const n = typeof count === 'number' ? count : count[0];
      this.u32[0] = n >>> 0;
      groups = split1D(Math.ceil(n / this.wgCount), this.ctx.limits.maxComputeWorkgroupsPerDimension);
      if (n === 0) return this;
    } else {
      const c = typeof count === 'number' ? [count, 1, 1] : [count[0], count[1] || 1, count[2] || 1];
      this.u32[0] = c[0]; this.u32[1] = c[1]; this.u32[2] = c[2];
      groups = [Math.ceil(c[0] / this.wg[0]), Math.ceil(c[1] / this.wg[1]), Math.ceil(c[2] / this.wg[2])];
    }
    return this._record(groups);
  }

  /** Dispatch an explicit number of workgroups ([x,y,z] or x). The size uniform is left as set. */
  dispatch(groups, uniforms, resources) {
    this.set(uniforms);
    this.bind(resources);
    if (typeof groups === 'number') groups = split1D(groups, this.ctx.limits.maxComputeWorkgroupsPerDimension);
    return this._record([groups[0], groups[1] || 1, groups[2] || 1]);
  }

  /** Dispatch with workgroup counts read from a GPU buffer (3 x u32 at byte `offset`). */
  dispatchIndirect(argsArray, offset = 0, uniforms, resources) {
    this.set(uniforms);
    this.bind(resources);
    const off = this.ctx._ring.push(this);   // may flush the batch: take the pass afterwards
    const pass = this.ctx.computePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroup(), [off]);
    pass.dispatchWorkgroupsIndirect(argsArray.buffer || argsArray, offset);
    return this;
  }

  _record(groups) {
    if (groups[0] === 0 || groups[1] === 0 || groups[2] === 0) return this;
    const ctx = this.ctx;
    const off = ctx._ring.push(this);
    const pass = ctx.computePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroup(), [off]);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    ctx.stats.dispatches++;
    return this;
  }

  _bindGroup() {
    const ring = this.ctx._ring;
    let key = ring.gen + '';
    for (const s of this.slots) {
      const r = this.bound[s.name];
      if (!r) throw new Error(`ShaderIntel: kernel "${this.name}": binding "${s.name}" was never provided`);
      key += ':' + resourceId(r);
    }
    let bg = this.bgCache.get(key);
    if (!bg) {
      if (this.bgCache.size > 64) this.bgCache.clear();
      const entries = [{ binding: 0, resource: { buffer: ring.buffer, offset: 0, size: this.U.size } }];
      this.slots.forEach((s, k) => entries.push({ binding: k + 1, resource: resourceOf(s, this.bound[s.name], this.ctx) }));
      bg = this.ctx.device.createBindGroup({ layout: this.bgl, entries, label: this.name });
      this.bgCache.set(key, bg);
    }
    return bg;
  }
}

function split1D(groups, max) {
  if (groups <= max) return [groups, 1, 1];
  const y = Math.ceil(groups / max);
  return [Math.ceil(groups / y), y, 1];
}

// ---------------------------------------------------------------- binding contracts

// struct Name { member: type, ... } → { name: [[member, type], ...] } (comments stripped;
// structs with attributes such as @align/@size are skipped: no contract is better than a wrong one)
function parseStructs(code) {
  const out = new Map();
  const src = code.replace(/\/\/[^\n]*/g, '');
  const re = /struct\s+(\w+)\s*\{([^}]*)\}/g;
  let m;
  while ((m = re.exec(src))) {
    if (m[2].includes('@')) continue;
    const members = [];
    for (const part of m[2].split(/[,;\n]/)) {
      const mm = /^\s*(\w+)\s*:\s*([\w<>, ]+?)\s*$/.exec(part);
      if (mm) members.push([mm[1], mm[2].replace(/\s+/g, '')]);
    }
    out.set(m[1], members);
  }
  return out;
}

// Storage layout (WGSL rules): { size, align } of a type, or null if it cannot be resolved.
function storageLayout(type, structs, depth = 0) {
  if (depth > 8) return null;
  const arr = /^array<(.+),(\d+)u?>$/.exec(type);
  if (arr) {
    const e = storageLayout(arr[1], structs, depth + 1);
    return e && { size: roundUp(e.size, e.align) * +arr[2], align: e.align };
  }
  if (structs.has(type)) {
    let off = 0, align = 1;
    for (const [, t] of structs.get(type)) {
      const L = storageLayout(t, structs, depth + 1);
      if (!L) return null;
      off = roundUp(off, L.align) + L.size;
      align = Math.max(align, L.align);
    }
    return { size: roundUp(off, align), align };
  }
  try { const L = layoutOf(type); return { size: L.size, align: L.align }; } catch { return null; }
}

function strideOf(type, structs) {
  const L = storageLayout(type.replace(/\s+/g, ''), structs);
  return L ? roundUp(L.size, L.align) : 0;
}

function checkContract(kernel, slot, r) {
  if (slot.kind !== 'storage' || !slot.stride || !r) return;
  const bytes = r.byteLength ?? r.size ?? r.buffer?.size;
  if (typeof bytes !== 'number') return;
  if (bytes % slot.stride !== 0)
    throw new Error(`ShaderIntel: kernel "${kernel.name}", binding "${slot.name}": a buffer of ${bytes} bytes is not a whole ` +
      `number of ${slot.type} (stride ${slot.stride} bytes) — the CPU-side record size and the WGSL type disagree`);
}

// ---------------------------------------------------------------- binding slots

function bufferSlot(name, spec) {
  let arr = null, type, access = 'read_write', atomic = false;
  if (spec && spec.buffer instanceof GPUBuffer && spec.layout) { arr = spec; type = spec.type; }
  else if (typeof spec === 'string') type = spec;
  else {
    arr = spec.array || null;
    type = spec.type || (arr && arr.type);
    access = spec.access || access;
    atomic = !!spec.atomic;
    if (spec.uniform) {
      return { name, kind: 'uniform', type, default: arr,
        decl: `var<uniform> ${name}: ${type};`, layout: { buffer: { type: 'uniform' } } };
    }
  }
  if (access === 'read' && atomic) throw new Error(`ShaderIntel: atomic binding "${name}" must be read_write`);
  const elem = atomic ? `atomic<${type}>` : type;
  const ro = access === 'read';
  return {
    name, kind: 'storage', type, default: arr,
    decl: `var<storage, ${ro ? 'read' : 'read_write'}> ${name}: array<${elem}>;`,
    layout: { buffer: { type: ro ? 'read-only-storage' : 'storage' } },
  };
}

function textureSlot(name, spec) {
  const dim = spec.dim || '2d';
  if (spec.storage) {
    const access = spec.access || 'write';
    const wa = { write: 'write-only', read: 'read-only', read_write: 'read-write' }[access];
    return {
      name, kind: 'texture', default: spec.texture || null,
      decl: `var ${name}: texture_storage_${dim}<${spec.storage}, ${access}>;`,
      layout: { storageTexture: { access: wa, format: spec.storage, viewDimension: dim } },
    };
  }
  const sampleType = spec.sampleType || 'float';
  const scalar = { float: 'f32', 'unfilterable-float': 'f32', uint: 'u32', sint: 'i32' }[sampleType];
  if (sampleType === 'depth')
    return { name, kind: 'texture', default: spec.texture || null, decl: `var ${name}: texture_depth_${dim};`,
      layout: { texture: { sampleType: 'depth', viewDimension: dim } } };
  return {
    name, kind: 'texture', default: spec.texture || null,
    decl: `var ${name}: texture_${dim}<${scalar}>;`,
    layout: { texture: { sampleType, viewDimension: dim } },
  };
}

function samplerSlot(name, spec) {
  const filter = (typeof spec === 'string' ? spec : spec.filter) || 'linear';
  const wrap = (typeof spec === 'object' && spec.wrap) || 'clamp-to-edge';
  return {
    name, kind: 'sampler', default: { __sampler: true, filter, wrap },
    decl: `var ${name}: sampler;`,
    layout: { sampler: { type: filter === 'linear' ? 'filtering' : 'non-filtering' } },
  };
}

const idMap = new WeakMap();
let nextResId = 1;
function resourceId(r) {
  if (r.id) return r.id;
  if (r.__sampler) return `s${r.filter}${r.wrap}`;
  let id = idMap.get(r);
  if (!id) { id = 'r' + nextResId++; idMap.set(r, id); }
  return id;
}

function resourceOf(slot, r, ctx) {
  if (slot.kind === 'storage' || slot.kind === 'uniform') {
    const buffer = r.buffer instanceof GPUBuffer ? r.buffer : r;
    return { buffer };
  }
  if (slot.kind === 'sampler') return r.__sampler ? ctx.sampler(r.filter, r.wrap) : r;
  if (r instanceof GPUTexture) return r.createView();
  if (r.view) return r.view;
  return r; // GPUTextureView
}
return { Kernel };
})();

// ---- src/core/profiler.js
const __m6 = (() => {
// GPU profiler built on timestamp queries. A profiled region is exactly one
// compute pass (timestamps can only be written at pass boundaries), so library
// code clears buffers with compute kernels instead of clearBuffer inside
// regions. Results are smoothed with an exponential moving average because
// browsers quantize timestamps (Chrome: 100 µs) unless developer flags are on.

class Profiler {
  constructor(ctx, capacity = 256) {
    this.ctx = ctx;
    this.enabled = ctx.caps.timestamps;
    this.capacity = capacity;
    this.results = new Map(); // label -> { ms (EMA), last, samples }
    this.smoothing = 0.1;
    if (!this.enabled) return;
    const d = ctx.device;
    this.querySet = d.createQuerySet({ type: 'timestamp', count: capacity });
    this.resolveBuf = d.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.pool = [];
    this.labels = [];
  }

  /** Reserve a begin/end pair for `label`; null when profiling is off or full. */
  alloc(label) {
    if (!this.enabled || this.labels.length * 2 + 2 > this.capacity) return null;
    const k = this.labels.length;
    this.labels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 };
  }

  /** Called by the context right before finishing a command encoder. */
  resolve(encoder) {
    if (!this.enabled || this.labels.length === 0) return null;
    const n = this.labels.length * 2;
    encoder.resolveQuerySet(this.querySet, 0, n, this.resolveBuf, 0);
    let rb = this.pool.pop();
    if (!rb) rb = this.ctx.device.createBuffer({ size: this.capacity * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(this.resolveBuf, 0, rb, 0, n * 8);
    const labels = this.labels;
    this.labels = [];
    return () => this._collect(rb, labels);
  }

  async _collect(rb, labels) {
    try {
      await rb.mapAsync(GPUMapMode.READ);
    } catch { return; }
    const t = new BigInt64Array(rb.getMappedRange(0, labels.length * 16));
    const sums = new Map();
    for (let k = 0; k < labels.length; k++) {
      const dt = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
      if (dt >= 0 && dt < 1e4) sums.set(labels[k], (sums.get(labels[k]) || 0) + dt);
    }
    rb.unmap();
    this.pool.push(rb);
    for (const [label, ms] of sums) {
      const r = this.results.get(label);
      if (!r) this.results.set(label, { ms, last: ms, samples: 1 });
      else {
        const a = r.samples < 10 ? 1 / (r.samples + 1) : this.smoothing;
        r.ms += (ms - r.ms) * a; r.last = ms; r.samples++;
      }
    }
  }

  /** Sorted [{ label, ms }] of smoothed GPU times per region. */
  report() {
    return [...this.results].map(([label, r]) => ({ label, ms: r.ms, last: r.last, samples: r.samples }));
  }

  reset() { this.results.clear(); }
}
return { Profiler };
})();

// ---- src/core/context.js
const __m1 = (() => {
const GArray = __m2.GArray;
const Kernel = __m4.Kernel;
const Profiler = __m6.Profiler;
const arrayCtor = __m3.arrayCtor;
const roundUp = __m3.roundUp;
// Context: device acquisition, capability detection, command batching and the
// uniform ring. Everything recorded through a Context (kernels, clears, copies,
// render passes) goes into one command encoder; compute dispatches share a
// single compute pass until something forces it to end. flush() submits it all.





class SIError extends Error {
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
async function init(opts = {}) {
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

class Context {
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
return { SIError, init, Context };
})();

// ---- src/prims/scan.js
const __m7 = (() => {
const defineModule = __m5.defineModule;
// Exclusive prefix sum (u32) over 4096-element tiles (256 threads x 16 values).
//
// Default: reduce-then-scan (reduce tiles -> scan partials recursively ->
// downsweep), 3N traffic, ~55% of memcpy speed on an M2 Max (29 G elements/s).
// Opt-in: { mode: 'single-pass' } — chained scan with decoupled look-back and
// fallback (2N traffic). It is correct (tests cover it) but measured 1.4–2.5x
// SLOWER than reduce-then-scan on Apple M2 / Chrome 152 for every look-back
// variant we tried (serial, subgroup-parallel, single-thread) and every spin
// limit; larger spin limits were slower still — predecessors starve while
// successors poll (no forward-progress guarantee). Kept for GPUs where it wins.
//
// Tiles are loaded with lane-contiguous (coalesced) vec4 reads into workgroup
// memory and then handed to threads as contiguous runs (a transpose through
// shared memory), which measured ~15% faster than per-thread blocked loads.
// The workgroup scan uses subgroupInclusiveAdd and then lets *every* subgroup
// scan the per-subgroup totals redundantly, picking its own offset with
// subgroupShuffle — one barrier, no second round trip through shared memory.
// Without subgroups a Hillis–Steele scan in workgroup memory is used instead.
// Workgroup size keeps (#subgroups <= subgroup size) for every size the adapter
// can produce.


/** WGSL module 'wgscan': workgroup-wide exclusive scan / reduction of one u32 per thread. */
defineModule('wgscan', {
  code: (caps) => caps.subgroups ? /* wgsl */`
enable subgroups;
var<workgroup> si_wsum: array<u32, 64>;
// Returns (exclusive prefix of x across the workgroup, workgroup total).
fn wgScanU32(x: u32, lidx: u32, sgi: u32, sgs: u32) -> vec2u {
  let incl = subgroupInclusiveAdd(x);
  let sg = lidx / sgs;
  if (sgi == sgs - 1u) { si_wsum[sg] = incl; }
  workgroupBarrier();
  let nsg = WG / sgs;
  let v = select(0u, si_wsum[min(sgi, 63u)], sgi < nsg);
  let s = subgroupInclusiveAdd(v);
  let total = subgroupShuffle(s, nsg - 1u);
  let before = select(0u, subgroupShuffle(s, max(sg, 1u) - 1u), sg > 0u);
  workgroupBarrier();
  return vec2u(before + incl - x, total);
}
fn wgSumU32(x: u32, lidx: u32, sgi: u32, sgs: u32) -> u32 {
  let s = subgroupAdd(x);
  let sg = lidx / sgs;
  if (sgi == 0u) { si_wsum[sg] = s; }
  workgroupBarrier();
  let nsg = WG / sgs;
  let t = subgroupAdd(select(0u, si_wsum[min(sgi, 63u)], sgi < nsg));
  workgroupBarrier();
  return t;
}` : /* wgsl */`
var<workgroup> si_wscan: array<u32, WG>;
fn wgScanU32(x: u32, lidx: u32, sgi: u32, sgs: u32) -> vec2u {
  si_wscan[lidx] = x;
  workgroupBarrier();
  for (var off = 1u; off < WG; off = off << 1u) {
    let add = select(0u, si_wscan[max(lidx, off) - off], lidx >= off);
    workgroupBarrier();
    si_wscan[lidx] = si_wscan[lidx] + add;
    workgroupBarrier();
  }
  let incl = si_wscan[lidx];
  let total = si_wscan[WG - 1u];
  workgroupBarrier();
  return vec2u(incl - x, total);
}
fn wgSumU32(x: u32, lidx: u32, sgi: u32, sgs: u32) -> u32 {
  return wgScanU32(x, lidx, sgi, sgs).y;
}`,
});

class Scan {
  /**
   * @param {Context} ctx
   * @param {number} maxN  largest array length this instance will scan
   * @param {object} [o]   { mode: 'reduce-then-scan' (default) | 'single-pass', lookback: 'thread' | 'workgroup' | 'subgroup', spins }
   *
   * single-pass: chained scan with decoupled look-back and fallback (Smith,
   *   Levien & Owens, SPAA 2025). 2N memory traffic. A workgroup publishes its
   *   tile aggregate, then looks back over predecessors' published values; if
   *   a predecessor has not published after a bounded number of polls it
   *   reduces that tile itself, so it never relies on forward-progress
   *   guarantees WebGPU does not give. 32-bit values travel as two 16-bit
   *   halves in separate atomics, each with a ready bit (no 64-bit atomics).
   * reduce-then-scan: 3N traffic, no inter-workgroup communication.
   */
  constructor(ctx, maxN, o = {}) {
    this.ctx = ctx;
    const sg = ctx.caps.subgroups;
    const minSG = ctx.caps.subgroupMin || 32;
    this.wg = sg ? Math.max(32, Math.min(256, minSG * minSG)) : 256;
    this.items = ctx.caps.workgroupStorage >= 20480 ? 16 : 8;
    this.tile = this.wg * this.items;
    this.maxN = maxN;
    this.mode = o.mode || 'reduce-then-scan';
    this.lookback = o.lookback || 'workgroup';
    this.spins = o.spins ?? 256;
    const V = this.tile / 4;                 // vec4 per tile
    const R = this.items / 4;                // vec4 per thread
    const sgArgs = sg ? 'sgi, sgs' : '0u, 0u';
    const common = { workgroupSize: this.wg, guard: false, imports: ['wgscan'], subgroups: sg };

    // Coalesced tile load: lane-contiguous vec4 reads into shared memory, then each
    // thread takes R consecutive vec4 (its contiguous run of `items` values).
    const tileIO = (srcName) => /* wgsl */`
      var<workgroup> sh: array<vec4u, ${V}>;
      fn load4(e: u32) -> vec4u {
        let full = u.n / 4u; let r = u.n % 4u;
        if (e < full) { return ${srcName}[e]; }
        if (e == full && r > 0u) { let x = ${srcName}[e]; return vec4u(x.x, select(0u, x.y, r > 1u), select(0u, x.z, r > 2u), 0u); }
        return vec4u(0u);
      }`;
    const localScan = /* wgsl */`
      for (var q = 0u; q < ${R}u; q++) { sh[q * WG + lidx] = load4(b4 + q * WG + lidx); }
      workgroupBarrier();
      var o: array<vec4u, ${R}>;
      var run = 0u;
      for (var q = 0u; q < ${R}u; q++) {
        let v = sh[lidx * ${R}u + q];
        o[q] = vec4u(run, run + v.x, run + v.x + v.y, run + v.x + v.y + v.z);
        run += v.x + v.y + v.z + v.w;
      }
      let r = wgScanU32(run, lidx, ${sgArgs});`;
    const store = (dstName) => /* wgsl */`
      workgroupBarrier();
      for (var q = 0u; q < ${R}u; q++) { sh[lidx * ${R}u + q] = o[q] + off; }
      workgroupBarrier();
      let n4 = (u.n + 3u) / 4u;
      for (var q = 0u; q < ${R}u; q++) { let e = b4 + q * WG + lidx; if (e < n4) { ${dstName}[e] = sh[q * WG + lidx]; } }`;

    // ---- single pass with decoupled look-back + fallback
    const statusFns = /* wgsl */`
      const READY = 0x80000000u;
      const SPINS = ${this.spins}u;
      var<workgroup> msg: vec2u;
      var<workgroup> ticket: u32;
      fn readStatus(t: u32) -> vec2u {
        let b = 1u + t * 4u;
        let il = atomicLoad(&st[b + 2u]); let ih = atomicLoad(&st[b + 3u]);
        if ((il & ih & READY) != 0u) { return vec2u(2u, (il & 0xffffu) | ((ih & 0xffffu) << 16u)); }
        let al = atomicLoad(&st[b]); let ah = atomicLoad(&st[b + 1u]);
        if ((al & ah & READY) != 0u) { return vec2u(1u, (al & 0xffffu) | ((ah & 0xffffu) << 16u)); }
        return vec2u(0u);
      }
      fn firstLane(m: vec4u) -> u32 {
        if (m.x != 0u) { return countTrailingZeros(m.x); }
        if (m.y != 0u) { return 32u + countTrailingZeros(m.y); }
        if (m.z != 0u) { return 64u + countTrailingZeros(m.z); }
        if (m.w != 0u) { return 96u + countTrailingZeros(m.w); }
        return 128u;
      }
      fn publish(t: u32, slot: u32, v: u32) {
        let b = 1u + t * 4u + slot;
        atomicStore(&st[b], READY | (v & 0xffffu));
        atomicStore(&st[b + 1u], READY | (v >> 16u));
      }`;
    const singlePass = () => ctx.kernel({
      ...common, name: 'scan-single-pass',
      buffers: { src: { type: 'vec4u', access: 'read' }, dst: { type: 'vec4u' }, st: { type: 'u32', atomic: true } },
      shared: tileIO('src') + statusFns,
      code: /* wgsl */`
      if (lidx == 0u) { ticket = atomicAdd(&st[0], 1u); }
      let t = workgroupUniformLoad(&ticket);
      let b4 = t * ${V}u;
      ${localScan}
      let agg = r.y;
      if (lidx == 0u) { publish(t, select(0u, 2u, t == 0u), agg); }
      var prefix = 0u;
      if (t > 0u) {
        ${this.lookback === 'thread' ? `
        // one thread walks back; no barriers inside the loop. A stalled predecessor is
        // reduced by this thread directly (rare: costs one tile of reads).
        if (lidx == 0u) {
          var lb = i32(t) - 1;
          var acc = 0u;
          var spins = 0u;
          loop {
            let m = readStatus(u32(lb));
            if (m.x == 2u) { acc += m.y; break; }
            if (m.x == 1u) { acc += m.y; lb -= 1; spins = 0u; if (lb < 0) { break; } continue; }
            spins += 1u;
            if (spins >= SPINS) {
              var s4 = vec4u(0u);
              for (var q = 0u; q < ${V}u; q++) { s4 += load4(u32(lb) * ${V}u + q); }
              let m2 = readStatus(u32(lb));
              if (m2.x == 2u) { acc += m2.y; break; }
              acc += select(s4.x + s4.y + s4.z + s4.w, m2.y, m2.x == 1u);
              lb -= 1; spins = 0u;
              if (lb < 0) { break; }
            }
          }
          publish(t, 2u, acc + agg);
          msg = vec2u(0u, acc);
        }
        prefix = workgroupUniformLoad(&msg).y;` : `
        var lb = i32(t) - 1;
        var spins = 0u;
        loop {
          ${sg && this.lookback === 'subgroup' ? `
          // parallel look-back: lane l inspects tile lb - l (tiles < 0 read as 'inclusive 0')
          let ti = lb - i32(sgi);
          var stv = vec2u(2u, 0u);
          if (ti >= 0) { stv = readStatus(u32(ti)); }
          let fNR = firstLane(subgroupBallot(stv.x == 0u));
          let fIn = firstLane(subgroupBallot(stv.x == 2u));
          let stop = min(min(fNR, fIn), sgs);
          let take = sgi < stop || (sgi == fIn && fIn < fNR);
          let sum = subgroupAdd(select(0u, stv.y, take));
          if (lidx == 0u) { msg = vec2u(select(0u, 1u, fIn < fNR) | (stop << 1u), sum); }
          let m = workgroupUniformLoad(&msg);
          prefix += m.y;
          if ((m.x & 1u) != 0u) { break; }
          let adv = i32(m.x >> 1u);
          lb -= adv;
          if (adv > 0) { spins = 0u; continue; }` : `
          if (lidx == 0u) { msg = readStatus(u32(lb)); }
          let m = workgroupUniformLoad(&msg);
          if (m.x == 2u) { prefix += m.y; break; }
          if (m.x == 1u) {
            prefix += m.y; lb -= 1; spins = 0u;
            if (lb < 0) { break; }
            continue;
          }`}
          spins += 1u;
          if (spins >= SPINS) {
            var s4 = vec4u(0u);
            for (var q = 0u; q < ${R}u; q++) { s4 += load4(u32(lb) * ${V}u + q * WG + lidx); }
            let fs = wgSumU32(s4.x + s4.y + s4.z + s4.w, lidx, ${sgArgs});
            if (lidx == 0u) { msg = readStatus(u32(lb)); }
            let m2 = workgroupUniformLoad(&msg);
            if (m2.x == 2u) { prefix += m2.y; break; }
            prefix += select(fs, m2.y, m2.x == 1u);
            lb -= 1; spins = 0u;
            if (lb < 0) { break; }
          }
        }
        if (lidx == 0u) { publish(t, 2u, prefix + agg); }`}
      }
      let off = prefix + r.x;
      ${store('dst')}`,
    });

    // ---- reduce-then-scan (3 passes, recursive on the partials)
    this.levels = [];
    let m = maxN;
    while (m > this.tile) { m = Math.ceil(m / this.tile); this.levels.push(ctx.array('u32', Math.max(4, m), { label: 'scan-partials' })); }
    this.dummy = ctx.array('u32', 4, { label: 'scan-dummy' });
    this.reduceK = ctx.kernel({
      ...common, name: 'scan-reduce',
      buffers: { src: { type: 'vec4u', access: 'read' }, partials: { type: 'u32' } },
      shared: tileIO('src'),
      code: /* wgsl */`
      let b4 = wgi * ${V}u;
      var s4 = vec4u(0u);
      for (var q = 0u; q < ${R}u; q++) { s4 += load4(b4 + q * WG + lidx); }
      let total = wgSumU32(s4.x + s4.y + s4.z + s4.w, lidx, ${sgArgs});
      if (lidx == 0u) { partials[wgi] = total; }`,
    });
    const down = (inPlace) => ctx.kernel({
      ...common, name: inPlace ? 'scan-down-inplace' : 'scan-down',
      buffers: inPlace
        ? { dst: { type: 'vec4u' }, partials: { type: 'u32', access: 'read' } }
        : { src: { type: 'vec4u', access: 'read' }, dst: { type: 'vec4u' }, partials: { type: 'u32', access: 'read' } },
      uniforms: { useOffsets: 'u32' },
      shared: tileIO(inPlace ? 'dst' : 'src'),
      code: /* wgsl */`
      let b4 = wgi * ${V}u;
      ${localScan}
      var off = r.x;
      if (u.useOffsets != 0u) { off += partials[wgi]; }
      ${store('dst')}`,
    });
    this.downK = down(false);
    this.downInPlaceK = down(true);
    if (this.mode === 'single-pass') {
      this.spK = singlePass();
      this.status = ctx.array('u32', 1 + 4 * Math.max(1, Math.ceil(maxN / this.tile)), { label: 'scan-status' });
    }
  }

  /**
   * Exclusive scan of the first n u32 of `src` into `dst` (may alias).
   * Elements past n inside the last vec4 are overwritten in dst.
   */
  run(src, dst = src, n = src.length) {
    if (n > this.maxN) throw new Error(`Scan: n=${n} exceeds maxN=${this.maxN}`);
    const tiles = Math.ceil(n / this.tile);
    if (tiles <= 1) return this._down(1, n, 0, src, dst, this.dummy), dst;
    // In place, a fallback could read a predecessor's tile after its outputs overwrote it:
    // WGSL atomics are relaxed, so those stores may become visible before its 'ready' flag.
    // Out-of-place fallbacks only read the untouched source, so only that path is single-pass.
    if (this.mode === 'single-pass' && src !== dst) {
      this.ctx.clear(this.status);
      this.spK.dispatch(tiles, { n }, { src, dst, st: this.status });
      return dst;
    }
    this._scan(src, dst, n, 0);
    return dst;
  }

  _down(groups, n, useOffsets, src, dst, partials) {
    if (src === dst) this.downInPlaceK.dispatch(groups, { n, useOffsets }, { dst, partials });
    else this.downK.dispatch(groups, { n, useOffsets }, { src, dst, partials });
  }

  _scan(src, dst, n, level) {
    const tiles = Math.ceil(n / this.tile);
    if (tiles <= 1) return this._down(1, n, 0, src, dst, this.dummy);
    const partials = this.levels[level];
    this.reduceK.dispatch(tiles, { n }, { src, partials });
    this._scan(partials, partials, tiles, level + 1);
    this._down(tiles, n, 1, src, dst, partials);
  }
}
return { Scan };
})();

// ---- src/prims/reduce.js
const __m8 = (() => {
// Parallel reduction (sum / min / max) with an optional per-element map,
// e.g. max speed of a particle set: new Reduce(ctx, { input: 'vec4f', type: 'f32',
// op: 'max', map: 'length(x.xyz)' }). The result stays on the GPU (result[0]) so
// other kernels can consume it without a CPU round trip; read() fetches it.

const IDENT = {
  sum: { f32: '0.0', u32: '0u', i32: '0i' },
  min: { f32: '3.4e38', u32: '0xffffffffu', i32: '2147483647i' },
  max: { f32: '-3.4e38', u32: '0u', i32: '-2147483647i' },
};
const SGOP = { sum: 'subgroupAdd', min: 'subgroupMin', max: 'subgroupMax' };

const ITEMS = 8;

class Reduce {
  /**
   * @param {Context} ctx
   * @param {object} o  { type: 'f32'|'u32'|'i32'|'vec2f'|'vec4f'..., input?: element type of src,
   *                      op: 'sum'|'min'|'max', map?: WGSL expression of `x`, maxN }
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    const type = o.type || 'f32';
    const input = o.input || type;
    const op = o.op || 'sum';
    const scalar = /u$|u32/.test(type) ? 'u32' : /i$|i32/.test(type) ? 'i32' : 'f32';
    const vecN = /vec([234])/.exec(type)?.[1];
    const ident = vecN ? `${type}(${IDENT[op][scalar]})` : IDENT[op][scalar];
    const combine = op === 'sum' ? 'a + b' : `${op}(a, b)`;
    const sg = ctx.caps.subgroups;
    this.wg = 256;
    this.tile = this.wg * ITEMS;
    this.type = type;
    this.maxN = o.maxN || 1 << 20;
    this.result = ctx.array(type, 1, { label: 'reduce-result' });
    this.levels = [];
    let n = this.maxN;
    while (n > 1) { n = Math.ceil(n / this.tile); this.levels.push(ctx.array(type, Math.max(1, n), { label: 'reduce-partial' })); if (n === 1) break; }

    const wgReduce = sg ? /* wgsl */`
      let s = ${SGOP[op]}(acc);
      if (sgi == 0u) { wsum[lidx / sgs] = s; }
      workgroupBarrier();
      let nsub = WG / sgs;
      let t = ${SGOP[op]}(select(IDENT, wsum[min(sgi, 63u)], sgi < nsub));
      if (lidx == 0u) { dst[wgi] = t; }` : /* wgsl */`
      wsum[lidx] = acc;
      workgroupBarrier();
      for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lidx < s) { let a = wsum[lidx]; let b = wsum[lidx + s]; wsum[lidx] = ${combine}; }
        workgroupBarrier();
      }
      if (lidx == 0u) { dst[wgi] = wsum[0]; }`;
    const make = (srcType, mapExpr, name) => ctx.kernel({
      name, workgroupSize: this.wg, guard: false, subgroups: sg,
      buffers: { src: { type: srcType, access: 'read' }, dst: { type } },
      defines: { IDENT: ident, TILE: `${this.tile}u` },
      shared: /* wgsl */`
        var<workgroup> wsum: array<${type}, ${sg ? 64 : this.wg}>;
        fn combine(a: ${type}, b: ${type}) -> ${type} { return ${combine}; }
        fn mapx(x: ${srcType}) -> ${type} { return ${mapExpr}; }`,
      code: /* wgsl */`
      let base = wgi * TILE + lidx;
      var acc = IDENT;
      for (var k = 0u; k < ${ITEMS}u; k++) {
        let e = base + k * WG;
        if (e < u.n) { acc = combine(acc, mapx(src[e])); }
      }
      ${wgReduce}`,
    });
    this.firstK = make(input, o.map || 'x', 'reduce-first');
    this.restK = input === type && !o.map ? this.firstK : make(type, 'x', 'reduce-rest');
  }

  /** Reduce the first n elements of src. Returns the 1-element result array (on GPU). */
  run(src, n = src.length) {
    let groups = Math.ceil(n / this.tile);
    let k = this.firstK, level = 0, cur = src, count = n;
    for (;;) {
      const dst = groups === 1 ? this.result : this.levels[level];
      k.dispatch(groups, { n: count }, { src: cur, dst });
      if (groups === 1) break;
      cur = dst; count = groups; level++;
      groups = Math.ceil(count / this.tile);
      k = this.restK;
    }
    return this.result;
  }

  /** Read the result to the CPU: a number for scalars, an array for vectors. */
  async read() {
    const v = await this.result.readPacked(1);
    return v.length === 1 ? v[0] : Array.from(v);
  }
}
return { Reduce };
})();

// ---- src/prims/grid.js
const __m9 = (() => {
const defineModule = __m5.defineModule;
const Scan = __m7.Scan;
// Uniform-grid neighbour search by counting sort — O(N), three dispatches + a scan:
//   count:   cell[i] = cellOf(p_i); rank[i] = atomicAdd(&count[cell], 1)
//   scan:    start = exclusiveScan(count)      (numCells + 1 entries)
//   scatter: order[start[cell[i]] + rank[i]] = i
// Cells are linearised x-fastest, so the three cells of a row (x-1..x+1) are
// contiguous in sorted order: a 27-cell neighbourhood is visited as 9 ranges.



defineModule('grid', {
  doc: 'Uniform grid helpers (cell coordinate / linear index).',
  code: /* wgsl */`
fn gridCoord(p: vec3f, origin: vec3f, inv: f32, dims: vec3u) -> vec3i {
  return clamp(vec3i(floor((p - origin) * inv)), vec3i(0), vec3i(dims) - 1);
}
fn gridIndex(c: vec3i, dims: vec3u) -> u32 {
  return u32(c.x) + dims.x * (u32(c.y) + dims.y * u32(c.z));
}`,
});

class SpatialGrid {
  /**
   * @param {Context} ctx
   * @param {object} o { maxCount, min: [x,y,z], max: [x,y,z], cellSize }
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.maxCount = o.maxCount;
    this.cellSize = o.cellSize;
    this.origin = o.min.slice(0, 3);
    this.dims = [0, 1, 2].map((a) => Math.max(1, Math.ceil((o.max[a] - o.min[a]) / o.cellSize)));
    this.numCells = this.dims[0] * this.dims[1] * this.dims[2];
    if (this.numCells > 1 << 26) throw new Error(`SpatialGrid: ${this.numCells} cells is too many; increase cellSize`);
    this.cell = ctx.array('u32', o.maxCount, { label: 'grid-cell' });
    this.rank = ctx.array('u32', o.maxCount, { label: 'grid-rank' });
    this.order = ctx.array('u32', o.maxCount, { label: 'grid-order' });
    this.start = ctx.array('u32', this.numCells + 1, { label: 'grid-start' });
    this.scan = new Scan(ctx, this.numCells + 1);
    const G = this.uniforms();
    this.countK = ctx.kernel({
      name: 'grid-count', imports: ['grid'], uniforms: G,
      buffers: { pos: { type: 'vec4f', access: 'read' }, cell: { type: 'u32' }, rank: { type: 'u32' },
        count: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let c = gridIndex(gridCoord(pos[i].xyz, u.gridOrigin, u.gridInv, u.gridDims), u.gridDims);
        cell[i] = c;
        rank[i] = atomicAdd(&count[c], 1u);`,
    });
    this.scatterK = ctx.kernel({
      name: 'grid-scatter',
      buffers: { cell: { type: 'u32', access: 'read' }, rank: { type: 'u32', access: 'read' },
        start: { type: 'u32', access: 'read' }, order: { type: 'u32' } },
      code: 'order[start[cell[i]] + rank[i]] = i;',
    });
    this.gathers = new Map();
  }

  /** Uniform declarations a kernel needs to evaluate grid cells. */
  uniforms() { return { gridOrigin: 'vec3f', gridInv: 'f32', gridDims: 'vec3u', gridCells: 'u32' }; }

  /** Matching uniform values. */
  values() {
    return { gridOrigin: this.origin, gridInv: 1 / this.cellSize, gridDims: this.dims, gridCells: this.numCells };
  }

  /** Sort `count` particles (pos: vec4f array) into cells. Fills order/start. */
  build(pos, count) {
    this.count = count;
    this.ctx.clear(this.start);
    this.countK.run(count, this.values(), { pos, cell: this.cell, rank: this.rank, count: this.start });
    this.scan.run(this.start, this.start, this.numCells + 1);
    this.scatterK.run(count, null, { cell: this.cell, rank: this.rank, start: this.start, order: this.order });
    return this;
  }

  /** dst[k] = src[order[k]] — permute an attribute array into cell order. */
  gather(src, dst, count = this.count) {
    let k = this.gathers.get(src.type);
    if (!k) {
      k = this.ctx.kernel({
        name: `grid-gather-${src.type}`,
        buffers: { src: { type: src.type, access: 'read' }, dst: { type: src.type }, order: { type: 'u32', access: 'read' } },
        code: 'dst[i] = src[order[i]];',
      });
      this.gathers.set(src.type, k);
    }
    k.run(count, null, { src, dst, order: this.order });
    return dst;
  }

  /**
   * WGSL snippet visiting every particle index `j` (in sorted order) in the
   * 27 cells around position expression `P`. Requires the kernel to bind
   * `start` (this.start) and declare this.uniforms(); import 'grid'.
   */
  static neighbors(P, body, startName = 'start') {
    return /* wgsl */`{
      let si_c0 = gridCoord(${P}, u.gridOrigin, u.gridInv, u.gridDims);
      let si_d = vec3i(u.gridDims);
      let si_x0 = u32(max(si_c0.x - 1, 0));
      let si_x1 = u32(min(si_c0.x + 1, si_d.x - 1));
      for (var si_z = max(si_c0.z - 1, 0); si_z <= min(si_c0.z + 1, si_d.z - 1); si_z++) {
        for (var si_y = max(si_c0.y - 1, 0); si_y <= min(si_c0.y + 1, si_d.y - 1); si_y++) {
          let si_row = u.gridDims.x * (u32(si_y) + u.gridDims.y * u32(si_z));
          let si_end = ${startName}[si_row + si_x1 + 1u];
          for (var j = ${startName}[si_row + si_x0]; j < si_end; j++) {
            ${body}
          }
        }
      }
    }`;
  }
}
return { SpatialGrid };
})();

// ---- src/render/program.js
const __m11 = (() => {
const structLayout = __m3.structLayout;
const writeField = __m3.writeField;
const link = __m5.link;
const constants = __m5.constants;
// RenderProgram: the render-pipeline twin of Kernel. Write @vertex/@fragment
// functions; bindings, the camera block and the uniform struct are generated.
//   binding 0  cam: Camera   (from the View)
//   binding 1  u:   Params   (uniform ring, dynamic offset)
//   binding 2+ buffers (read-only storage), textures, samplers in declaration order



const CAMERA_WGSL = /* wgsl */`
struct Camera {
  view: mat4x4f,
  proj: mat4x4f,
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  invView: mat4x4f,
  invProj: mat4x4f,
  eye: vec3f,
  time: f32,
  viewport: vec2f,
  near: f32,
  far: f32,
  fovY: f32,
  pixelAngle: f32,
  frame: u32,
  dt: f32,
}`;
const CAMERA_SIZE = 6 * 64 + 16 + 16 + 16;

let progId = 0;

class RenderProgram {
  constructor(ctx, desc) {
    this.ctx = ctx;
    this.id = ++progId;
    this.name = desc.name || `program${this.id}`;
    const slots = [];
    for (const [name, spec] of Object.entries(desc.buffers || {})) {
      const type = typeof spec === 'string' ? spec : spec.type || spec.array?.type || spec.type;
      const t = typeof spec === 'object' && spec.layout ? spec.type : type;
      slots.push({ name, kind: 'storage', default: spec && spec.buffer instanceof GPUBuffer ? spec : spec?.array || null,
        decl: `var<storage, read> ${name}: array<${t}>;`, layout: { buffer: { type: 'read-only-storage' } } });
    }
    for (const [name, spec] of Object.entries(desc.textures || {})) {
      const dim = spec.dim || '2d';
      const sampleType = spec.sampleType || 'float';
      const decl = sampleType === 'depth' ? `var ${name}: texture_depth_${dim};`
        : `var ${name}: texture_${dim}<${{ float: 'f32', 'unfilterable-float': 'f32', uint: 'u32', sint: 'i32' }[sampleType]}>;`;
      slots.push({ name, kind: 'texture', default: spec.texture || null, decl, layout: { texture: { sampleType, viewDimension: dim } } });
    }
    for (const [name, spec] of Object.entries(desc.samplers || {})) {
      const filter = (typeof spec === 'string' ? spec : spec.filter) || 'linear';
      const wrap = (typeof spec === 'object' && spec.wrap) || 'clamp-to-edge';
      slots.push({ name, kind: 'sampler', default: { __sampler: true, filter, wrap }, decl: `var ${name}: sampler;`,
        layout: { sampler: { type: filter === 'linear' ? 'filtering' : 'non-filtering' } } });
    }
    this.slots = slots;
    this.U = structLayout('Params', { _pad: 'u32', ...(desc.uniforms || {}) });
    this.ubytes = new ArrayBuffer(this.U.size);
    this.uview = new DataView(this.ubytes);
    this.u32 = new Uint32Array(this.ubytes);
    this.fieldByName = new Map(this.U.fields.map((f) => [f.name, f]));
    if (desc.values) this.set(desc.values);

    const decls = [CAMERA_WGSL, this.U.wgsl,
      '@group(0) @binding(0) var<uniform> cam: Camera;',
      '@group(0) @binding(1) var<uniform> u: Params;'];
    slots.forEach((s, k) => decls.push(`@group(0) @binding(${k + 2}) ${s.decl}`));
    this.code = link([constants(desc.defines), ...decls, desc.code].join('\n'), desc.imports || [], ctx.caps);

    const d = ctx.device;
    const vis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.bgl = d.createBindGroupLayout({
      label: this.name,
      entries: [
        { binding: 0, visibility: vis, buffer: { type: 'uniform' } },
        { binding: 1, visibility: vis, buffer: { type: 'uniform', hasDynamicOffset: true } },
        ...slots.map((s, k) => ({ binding: k + 2, visibility: s.kind === 'storage' ? vis : GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, ...s.layout })),
      ],
    });
    const { module, ready } = ctx.shaderModule(this.code, this.name);
    const targets = (desc.targets || [{ format: 'rgba16float' }]).map((t) => (typeof t === 'string' ? { format: t } : t));
    const depth = desc.depth === false ? undefined : {
      format: desc.depth?.format || 'depth32float',
      depthWriteEnabled: desc.depth?.write ?? true,
      depthCompare: desc.depth?.compare || 'less',
    };
    d.pushErrorScope('validation');
    this.pipeline = d.createRenderPipeline({
      label: this.name,
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      vertex: { module, entryPoint: desc.vertex || 'vs' },
      fragment: desc.fragment === false ? undefined : { module, entryPoint: desc.fragment || 'fs', targets },
      primitive: { topology: desc.topology || 'triangle-list', cullMode: desc.cullMode || 'none' },
      depthStencil: depth,
    });
    const scope = d.popErrorScope().then((err) => {
      if (err) { const m = `ShaderIntel: render pipeline "${this.name}": ${err.message}`; console.error(m); throw new Error(m); }
    });
    this._ready = Promise.all([ready, scope]);
    this._ready.catch(() => {});
    this.bound = {};
    for (const s of slots) if (s.default) this.bound[s.name] = s.default;
    this.bgCache = new Map();
  }

  ready() { return this._ready; }

  set(values) {
    if (!values) return this;
    for (const [k, v] of Object.entries(values)) {
      const f = this.fieldByName.get(k);
      if (!f) throw new Error(`ShaderIntel: program "${this.name}" has no uniform "${k}"`);
      writeField(this.uview, f, v);
    }
    return this;
  }

  bind(resources) {
    if (resources) for (const [k, v] of Object.entries(resources)) this.bound[k] = v;
    return this;
  }

  /** Record a draw into an open render pass. */
  draw(pass, view, vertexCount, instanceCount = 1, uniforms, resources) {
    this.set(uniforms);
    this.bind(resources);
    const ring = this.ctx._ring;
    const off = ring.push(this);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroup(view), [off]);
    pass.draw(vertexCount, instanceCount);
    return this;
  }

  /** Indexed draw (uint32 indices). */
  drawIndexed(pass, view, indexBuffer, indexCount, instanceCount = 1, uniforms, resources) {
    this.set(uniforms);
    this.bind(resources);
    const off = this.ctx._ring.push(this);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroup(view), [off]);
    pass.setIndexBuffer(indexBuffer, 'uint32');
    pass.drawIndexed(indexCount, instanceCount);
    return this;
  }

  _bindGroup(view) {
    const ring = this.ctx._ring;
    let key = `${ring.gen}:${view.id}`;
    for (const s of this.slots) {
      const r = this.bound[s.name];
      if (!r) throw new Error(`ShaderIntel: program "${this.name}": binding "${s.name}" not provided`);
      key += ':' + (r.id || (r.__sampler ? r.filter + r.wrap : idOf(r)));
    }
    let bg = this.bgCache.get(key);
    if (!bg) {
      if (this.bgCache.size > 64) this.bgCache.clear();
      const entries = [
        { binding: 0, resource: { buffer: view.cameraBuffer } },
        { binding: 1, resource: { buffer: ring.buffer, offset: 0, size: this.U.size } },
      ];
      this.slots.forEach((s, k) => {
        const r = this.bound[s.name];
        let res;
        if (s.kind === 'storage') res = { buffer: r.buffer instanceof GPUBuffer ? r.buffer : r };
        else if (s.kind === 'sampler') res = r.__sampler ? this.ctx.sampler(r.filter, r.wrap) : r;
        else res = r.view || (r instanceof GPUTexture ? r.createView() : r);
        entries.push({ binding: k + 2, resource: res });
      });
      bg = this.ctx.device.createBindGroup({ layout: this.bgl, entries, label: this.name });
      this.bgCache.set(key, bg);
    }
    return bg;
  }
}

const ids = new WeakMap();
let nid = 1;
function idOf(o) { let i = ids.get(o); if (!i) { i = 'o' + nid++; ids.set(o, i); } return i; }
return { CAMERA_WGSL, CAMERA_SIZE, RenderProgram };
})();

// ---- src/render/bloom.js
const __m12 = (() => {
// Energy-conserving HDR bloom (progressive downsample / tent upsample, after
// Jimenez, "Next Generation Post Processing in Call of Duty: AW", 2014).
// No threshold: every pixel contributes a little, so bright HDR values glow
// naturally and the image stays physically plausible.
class Bloom {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.strength = o.strength ?? 0.05;
    this.levels = o.levels ?? 6;
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
        if (u.firstLevel == 1u) { o = min(o, vec3f(64.0)); } // tame fireflies before they spread
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
}
return { Bloom };
})();

// ---- src/render/math.js
const __m13 = (() => {
// Minimal column-major mat4 / vec3 helpers (WebGPU clip space: z in [0, 1]).

const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};

const m4 = {
  identity: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),

  /** Right-handed perspective, looking down -z, depth mapped to [0, 1]. */
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f;
    m[10] = far / (near - far); m[11] = -1;
    m[14] = (near * far) / (near - far);
    return m;
  },

  lookAt(eye, target, up = [0, 1, 0]) {
    const z = v3.norm(v3.sub(eye, target));
    const x = v3.norm(v3.cross(up, z));
    const y = v3.cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -v3.dot(x, eye), -v3.dot(y, eye), -v3.dot(z, eye), 1,
    ]);
  },

  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    return o;
  },

  invert(m) {
    const o = new Float32Array(16);
    const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return m4.identity();
    det = 1 / det;
    o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return o;
  },

  transformPoint(m, p) {
    const x = p[0], y = p[1], z = p[2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w, (m[1] * x + m[5] * y + m[9] * z + m[13]) / w, (m[2] * x + m[6] * y + m[10] * z + m[14]) / w];
  },
};
return { v3, m4 };
})();

// ---- src/render/view.js
const __m10 = (() => {
const RenderProgram = __m11.RenderProgram;
const CAMERA_SIZE = __m11.CAMERA_SIZE;
const Bloom = __m12.Bloom;
const m4 = __m13.m4;
const v3 = __m13.v3;
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




let viewId = 0;
const HDR_FORMAT = 'rgba16float';
const DEPTH_FORMAT = 'depth32float';

class View {
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
class OrbitCamera {
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
class Hud {
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

function showError(canvas, e) {
  const div = document.createElement('div');
  Object.assign(div.style, { position: 'absolute', inset: '0', padding: '24px', color: '#ffb4a9', background: 'rgba(20,0,0,0.85)',
    font: '12px/1.5 ui-monospace,Menlo,monospace', whiteSpace: 'pre-wrap', overflow: 'auto', zIndex: 20 });
  div.textContent = String(e && (e.message || e));
  (canvas?.parentElement || document.body).appendChild(div);
}
return { HDR_FORMAT, DEPTH_FORMAT, View, OrbitCamera, Hud, showError };
})();

// ---- src/render/fullscreen.js
const __m14 = (() => {
const RenderProgram = __m11.RenderProgram;
// Fullscreen shader layer — a Shadertoy-style canvas backed by the shader library.
//
//   view.add(gpu.fullscreen({
//     imports: ['noise', 'color'],
//     uniforms: { speed: 'f32' },
//     code: `fn shade(uv: vec2f, px: vec2f) -> vec4f {
//       return vec4f(viridis(0.5 + 0.5 * simplex3(vec3f(uv * 6.0, cam.time * u.speed))), 1.0);
//     }`,
//   }));
//
// uv is in [0,1] with (0,0) at the bottom-left; px is the pixel coordinate.
// `cameraRay(uv)` returns the world-space ray of the view's orbit camera.


class Fullscreen {
  constructor(ctx, desc) {
    this.values = {};
    this.program = new RenderProgram(ctx, {
      name: desc.name || 'fullscreen',
      imports: desc.imports, uniforms: desc.uniforms, textures: desc.textures, samplers: desc.samplers,
      buffers: desc.buffers, defines: desc.defines, values: desc.values,
      depth: { write: !!desc.writeDepth, compare: 'always' },
      targets: desc.targets,
      code: /* wgsl */`
        struct FsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
        struct Ray { o: vec3f, d: vec3f }
        fn cameraRay(uv: vec2f) -> Ray {
          let ndc = uv * 2.0 - 1.0;
          let a = cam.invViewProj * vec4f(ndc, 0.0, 1.0);
          let b = cam.invViewProj * vec4f(ndc, 1.0, 1.0);
          let o = a.xyz / a.w;
          return Ray(o, normalize(b.xyz / b.w - o));
        }
        /** View-space depth of a world point, as the depth buffer stores it. */
        fn worldDepth(p: vec3f) -> f32 { let c = cam.viewProj * vec4f(p, 1.0); return c.z / c.w; }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> FsOut {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
          return FsOut(vec4f(xy * 2.0 - 1.0, 0.0, 1.0), xy);
        }
        ${desc.code}
        ${desc.writeDepth
          ? `struct FsFrag { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
             @fragment fn fs(in: FsOut) -> FsFrag { let r = shade(in.uv, in.pos.xy); return FsFrag(r.color, r.depth); }`
          : `@fragment fn fs(in: FsOut) -> @location(0) vec4f { return shade(in.uv, in.pos.xy); }`}
      `,
    });
  }
  ready() { return this.program.ready(); }
  set(values) { this.program.set(values); return this; }
  bind(resources) { this.program.bind(resources); return this; }
  draw(pass, view) { this.program.draw(pass, view, 3, 1); }
}
return { Fullscreen };
})();

// ---- src/render/spheres.js
const __m15 = (() => {
const RenderProgram = __m11.RenderProgram;
// Ray-cast sphere impostors for particle buffers. Each particle is a camera-
// facing quad; the fragment shader intersects the exact sphere, so silhouettes,
// depth and normals are pixel-perfect at 4 vertices per particle.
//
// The source is any vec4f storage buffer where particle k's position lives at
// vec4 index k*stride + position (xyz, optionally scaled/offset into world space).


class Spheres {
  /**
   * @param {object} o
   *   data: GArray (vec4f layout), stride (vec4s per particle, default 1), position (vec4 index, default 0)
   *   velocity: vec4 index of velocity or -1;  material: WGSL expr of `px: vec4f` giving a u32 palette index
   *   count: number | () => number;  radius; scale; offset; palette: [[r,g,b], ...] (linear)
   *   colorBy: 'material' | 'speed' | 'height';  speedScale
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.o = o;
    this.palette = (o.palette || [[0.1, 0.45, 0.9], [0.9, 0.35, 0.25], [0.95, 0.95, 0.97], [0.85, 0.7, 0.4], [0.3, 0.8, 0.4], [0.7, 0.4, 0.9], [0.9, 0.8, 0.2], [0.5, 0.5, 0.55]]).slice(0, 8);
    while (this.palette.length < 8) this.palette.push([0.7, 0.7, 0.7]);
    const stride = o.stride || 1;
    const pos = o.position || 0;
    const vel = o.velocity ?? -1;
    const matExpr = o.material || '0u';
    const colorBy = { material: 0, speed: 1, height: 2 }[o.colorBy || 'material'];
    const mask = o.mask || 'true';
    this.program = new RenderProgram(ctx, {
      name: 'spheres',
      imports: ['pbr', 'color'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { radius: 'f32', scale: 'f32', offset: 'vec3f', colorBy: 'u32', speedScale: 'f32', velScale: 'f32',
        pal0: 'vec4f', pal1: 'vec4f', pal2: 'vec4f', pal3: 'vec4f', pal4: 'vec4f', pal5: 'vec4f', pal6: 'vec4f', pal7: 'vec4f',
        sun: 'vec3f', ambient: 'f32', matMask: 'u32' },
      code: /* wgsl */`
        struct VO {
          @builtin(position) pos: vec4f,
          @location(0) center: vec3f,       // view space
          @location(1) quad: vec3f,         // view-space point on the impostor plane
          @location(2) @interpolate(flat) color: vec3f,
        }
        fn paletteColor(k: u32) -> vec3f {
          switch k & 7u {
            case 0u: { return u.pal0.rgb; } case 1u: { return u.pal1.rgb; } case 2u: { return u.pal2.rgb; }
            case 3u: { return u.pal3.rgb; } case 4u: { return u.pal4.rgb; } case 5u: { return u.pal5.rgb; }
            case 6u: { return u.pal6.rgb; } default: { return u.pal7.rgb; }
          }
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let px = P[ii * ${stride}u + ${pos}u];
          let world = px.xyz * u.scale + u.offset;
          let c = (cam.view * vec4f(world, 1.0)).xyz;
          let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
          // Enlarge the quad slightly so the sphere's perspective silhouette fits.
          let q = c + vec3f(corner * u.radius * 1.5, 0.0);
          let keep = ${mask};
          var col = paletteColor(${matExpr});
          ${vel >= 0 ? `
          let v = P[ii * ${stride}u + ${vel}u].xyz * u.velScale;
          if (u.colorBy == 1u) { col = turbo(clamp(length(v) * u.speedScale, 0.0, 1.0)); col = col * col; }` : ''}
          if (u.colorBy == 2u) { col = viridis(clamp(world.y, 0.0, 1.0)); col = col * col; }
          var clip = cam.proj * vec4f(q, 1.0);
          if (!keep) { clip = vec4f(0.0, 0.0, -2.0, 1.0); }   // culled: degenerate, outside the clip volume
          return VO(clip, c, q, col);
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let d = normalize(in.quad);
          let b = dot(d, in.center);
          let h = b * b - dot(in.center, in.center) + u.radius * u.radius;
          if (h < 0.0) { discard; }
          let t = b - sqrt(h);
          let hit = d * t;
          let nv = (hit - in.center) / u.radius;
          let n = normalize((cam.invView * vec4f(nv, 0.0)).xyz);
          let v = normalize((cam.invView * vec4f(-d, 0.0)).xyz);
          let clip = cam.proj * vec4f(hit, 1.0);
          let ao = 0.6 + 0.4 * clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
          let c = shadePBR(n, v, in.color, 0.0, 0.45, u.sun, vec3f(2.6), 1.0, ao * u.ambient);
          return FO(vec4f(c, 1.0), clip.z / clip.w);
        }`,
      topology: 'triangle-strip',
    });
    this.values = { radius: o.radius ?? 0.01, scale: o.scale ?? 1, offset: o.offset || [0, 0, 0], colorBy,
      speedScale: o.speedScale ?? 0.25, velScale: o.velScale ?? 1, sun: normalize3(o.sun || [0.4, 0.8, 0.3]), ambient: o.ambient ?? 1,
      matMask: o.matMask ?? 0xffffffff };
    this.palette.forEach((c, k) => { this.values['pal' + k] = [...c, 1]; });
    this.program.set(this.values).bind({ P: o.data });
  }
  set(v) { Object.assign(this.values, v); this.program.set(v); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.data.length; }
  draw(pass, view) {
    const n = this.count;
    if (n > 0) this.program.draw(pass, view, 4, n);
  }
}

function normalize3(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
return { Spheres };
})();

// ---- src/render/scene.js
const __m16 = (() => {
const Fullscreen = __m14.Fullscreen;
const RenderProgram = __m11.RenderProgram;
// Scene helpers: a sky + ground "stage" and simple line geometry (box outlines).



/** Analytic sky gradient plus an infinite anti-aliased grid floor at y = height. Writes depth. */
class Stage {
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
class Lines {
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
return { Stage, Lines };
})();

// ---- src/ui.js
const __m17 = (() => {
// Tiny zero-dependency control panel for examples.
//   const ui = panel('Fluid', [
//     { label: 'substeps', min: 1, max: 16, step: 1, value: 8, onInput: (v) => (sim.substeps = v) },
//     { label: 'reset', onClick: reset },
//     { label: 'material', options: ['water', 'jelly'], value: 'water', onChange: (v) => ... },
//     { label: 'show grid', checked: true, onChange: (v) => ... },
//   ]);

const CSS = `
.si-panel { position: absolute; right: 12px; top: 12px; width: 236px; padding: 10px 12px 12px; z-index: 10;
  background: rgba(14,17,22,0.82); color: #d8dee9; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
  font: 12px/1.4 system-ui, -apple-system, sans-serif; backdrop-filter: blur(6px); }
.si-panel h3 { margin: 0 0 8px; font-size: 13px; font-weight: 600; color: #fff; }
.si-panel .row { display: grid; grid-template-columns: 82px 1fr 38px; align-items: center; gap: 6px; margin: 5px 0; }
.si-panel .row.wide { grid-template-columns: 82px 1fr; }
.si-panel input[type=range] { width: 100%; accent-color: #6cb6ff; }
.si-panel .val { text-align: right; color: #9fb1c7; font-variant-numeric: tabular-nums; }
.si-panel button { width: 100%; margin-top: 6px; padding: 6px; background: #22303f; color: #e6edf3; border: 1px solid #344558;
  border-radius: 5px; cursor: pointer; font: inherit; }
.si-panel button:hover { background: #2b3d51; }
.si-panel select { width: 100%; background: #1b232d; color: #e6edf3; border: 1px solid #344558; border-radius: 4px; padding: 3px; font: inherit; }
.si-panel .note { color: #8b9bb0; margin-top: 8px; font-size: 11px; }
`;

let styled = false;

function panel(title, controls = [], parent = document.body) {
  if (!styled) { const s = document.createElement('style'); s.textContent = CSS; document.head.appendChild(s); styled = true; }
  const el = document.createElement('div');
  el.className = 'si-panel';
  if (title) { const h = document.createElement('h3'); h.textContent = title; el.appendChild(h); }
  const api = { el, values: {} };
  for (const c of controls) {
    if (c.onClick) {
      const b = document.createElement('button');
      b.textContent = c.label;
      b.addEventListener('click', () => c.onClick());
      el.appendChild(b);
      continue;
    }
    if (c.html !== undefined) {
      const d = document.createElement('div'); d.className = 'note'; d.innerHTML = c.html; el.appendChild(d);
      continue;
    }
    const row = document.createElement('div');
    row.className = 'row';
    const lab = document.createElement('label');
    lab.textContent = c.label;
    row.appendChild(lab);
    if (c.options) {
      row.classList.add('wide');
      const sel = document.createElement('select');
      for (const o of c.options) { const opt = document.createElement('option'); opt.value = opt.textContent = o; sel.appendChild(opt); }
      sel.value = c.value ?? c.options[0];
      api.values[c.label] = sel.value;
      sel.addEventListener('change', () => { api.values[c.label] = sel.value; c.onChange?.(sel.value); });
      row.appendChild(sel);
    } else if (c.checked !== undefined || typeof c.value === 'boolean') {
      row.classList.add('wide');
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = c.checked ?? c.value;
      api.values[c.label] = cb.checked;
      cb.addEventListener('change', () => { api.values[c.label] = cb.checked; c.onChange?.(cb.checked); });
      row.appendChild(cb);
    } else {
      const r = document.createElement('input');
      r.type = 'range'; r.min = c.min ?? 0; r.max = c.max ?? 1; r.step = c.step ?? 0.01; r.value = c.value ?? 0;
      const val = document.createElement('span');
      val.className = 'val';
      const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(+(c.step ?? 0.01)) + 1e-9)));
      const fmt = (v) => (+v).toFixed(dec);
      val.textContent = fmt(r.value);
      api.values[c.label] = +r.value;
      r.addEventListener('input', () => { val.textContent = fmt(r.value); api.values[c.label] = +r.value; c.onInput?.(+r.value); });
      row.appendChild(r); row.appendChild(val);
    }
    el.appendChild(row);
  }
  parent.appendChild(el);
  return api;
}
return { panel };
})();

// ---- src/render/fluid.js
const __m19 = (() => {
const RenderProgram = __m11.RenderProgram;
// Screen-space fluid rendering (van der Laan, Green & Sainz 2009) with a
// narrow-range depth filter (after Truong & Yuksel, I3D 2018).
//
//   1. depth pass     particles -> nearest linear eye depth (r32float, own z-buffer)
//   2. thickness pass particles -> additive chord-length splats (r16float), occluded by the scene
//   3. filter         separable narrow-range filter: samples far *behind* the centre are
//                     dropped (another surface), samples far *in front* are clamped, so
//                     silhouettes stay crisp while the surface is smoothed
//   4. composite      normals from the smoothed depth; Fresnel (Schlick, F0 = 0.02),
//                     sky reflection, refraction of the opaque scene with Beer–Lambert
//                     absorption over the thickness, sun specular; writes depth.


class FluidSurface {
  /**
   * o: data (vec4 GArray or () => GArray), stride, position, count, radius (world),
   *    scale, offset, mask (WGSL expr of px -> bool, which particles are fluid),
   *    absorb [r,g,b] (per world unit), filterRadius (particle radii), iterations
   */
  constructor(ctx, o) {
    this.ctx = ctx;
    this.o = o;
    const stride = o.stride || 1, pos = o.position || 0;
    const mask = o.mask || 'true';
    this.values = { radius: o.radius ?? 0.01, scale: o.scale ?? 1, offset: o.offset || [0, 0, 0], absorb: o.absorb || [2.6, 0.9, 0.45],
      tint: o.tint || [0.85, 0.95, 1.0], filterRadius: o.filterRadius ?? 9, maxPixels: o.maxPixels ?? 40, sun: norm(o.sun || [0.45, 0.8, 0.3]),
      thicknessScale: o.thicknessScale ?? 1, refraction: o.refraction ?? 0.04, matMask: o.matMask ?? 0xffffffff };
    this.iterations = o.iterations ?? 3;
    const particleVS = (outs) => /* wgsl */`
      struct VO { @builtin(position) pos: vec4f, @location(0) center: vec3f, @location(1) quad: vec3f }
      @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
        let px = P[ii * ${stride}u + ${pos}u];
        let keep = ${mask};
        let world = px.xyz * u.scale + u.offset;
        let c = (cam.view * vec4f(world, 1.0)).xyz;
        let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
        let q = c + vec3f(corner * u.radius * ${outs}, 0.0);
        var clip = cam.proj * vec4f(q, 1.0);
        if (!keep) { clip = vec4f(0.0, 0.0, -2.0, 1.0); }
        return VO(clip, c, q);
      }`;
    const common = { buffers: { P: { type: 'vec4f' } }, topology: 'triangle-strip' };
    const U = { radius: 'f32', scale: 'f32', offset: 'vec3f', matMask: 'u32' };
    this.depthProg = new RenderProgram(ctx, {
      ...common, name: 'ssf-depth', uniforms: U, targets: ['r32float'], depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        ${particleVS('1.5')}
        struct FO { @location(0) z: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let d = normalize(in.quad);
          let b = dot(d, in.center);
          let h = b * b - dot(in.center, in.center) + u.radius * u.radius;
          if (h < 0.0) { discard; }
          let hit = d * (b - sqrt(h));
          let clip = cam.proj * vec4f(hit, 1.0);
          return FO(vec4f(-hit.z, 0.0, 0.0, 1.0), clip.z / clip.w);
        }`,
    });
    this.thickProg = new RenderProgram(ctx, {
      ...common, name: 'ssf-thickness', uniforms: U,
      targets: [{ format: 'r16float', blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
      depth: { write: false, compare: 'less' },
      code: /* wgsl */`
        ${particleVS('1.0')}
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let q = (in.quad.xy - in.center.xy) / u.radius;
          let r2 = dot(q, q);
          if (r2 > 1.0) { discard; }
          return vec4f(2.0 * u.radius * sqrt(1.0 - r2), 0.0, 0.0, 0.0);
        }`,
    });
    // separable narrow-range filter on linear depth
    this.filterK = ctx.kernel({
      name: 'ssf-filter', workgroupSize: [8, 8],
      textures: { src: { sampleType: 'unfilterable-float' }, dst: { storage: 'r32float' } },
      uniforms: { dir: 'vec2i', radius: 'f32', pxScale: 'f32', filterRadius: 'f32', maxPixels: 'f32' },
      code: /* wgsl */`
        let c = vec2i(p.xy);
        let zc = textureLoad(src, c, 0).x;
        if (zc > 1e5) { textureStore(dst, c, vec4f(zc, 0.0, 0.0, 0.0)); return; }
        let R = i32(min(u.maxPixels, ceil(u.filterRadius * u.radius * u.pxScale / zc)));
        let sigma = max(f32(R) * 0.5, 0.5);
        let lambda = 6.0 * u.radius;   // narrow range above (behind) the centre
        let mu = u.radius;             // clamp distance in front
        var sum = zc;
        var wsum = 1.0;
        let sz = vec2i(u.size.xy) - 1;
        for (var side = -1; side <= 1; side += 2) {
          var hi = zc + lambda;
          for (var k = 1; k <= R; k++) {
            let q = clamp(c + u.dir * (k * side), vec2i(0), sz);
            var z = textureLoad(src, q, 0).x;
            if (z > 1e5 || z > hi) { continue; }     // background or a different surface behind
            z = max(z, zc - mu);                      // pull foreground samples into the narrow range
            hi = max(hi, z + lambda);                 // range extension along continuous slopes
            let w = exp(-f32(k * k) / (2.0 * sigma * sigma));
            sum += z * w; wsum += w;
          }
        }
        textureStore(dst, c, vec4f(sum / wsum, 0.0, 0.0, 0.0));`,
    });
    this.compositeProg = new RenderProgram(ctx, {
      name: 'ssf-composite', imports: ['pbr', 'color'],
      textures: { zTex: { sampleType: 'unfilterable-float' }, thick: { sampleType: 'float' }, scene: { sampleType: 'float' } },
      samplers: { lin: 'linear' },
      uniforms: { absorb: 'vec3f', tint: 'vec3f', sun: 'vec3f', thicknessScale: 'f32', refraction: 'f32' },
      depth: { write: true, compare: 'less' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
          return VO(vec4f(xy, 0.0, 1.0));
        }
        fn viewPos(px: vec2i) -> vec3f {
          let z = textureLoad(zTex, px, 0).x;
          let ndc = vec2f((f32(px.x) + 0.5) / cam.viewport.x * 2.0 - 1.0, 1.0 - (f32(px.y) + 0.5) / cam.viewport.y * 2.0);
          let v = cam.invProj * vec4f(ndc, 1.0, 1.0);
          let dir = v.xyz / v.w;
          return dir / -dir.z * z;
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          let px = vec2i(in.pos.xy);
          let z = textureLoad(zTex, px, 0).x;
          if (z > 1e5) { discard; }
          let sz = vec2i(cam.viewport) - 1;
          let pc = viewPos(px);
          let dxp = viewPos(min(px + vec2i(1, 0), sz)) - pc;
          let dxn = pc - viewPos(max(px - vec2i(1, 0), vec2i(0)));
          let dyp = viewPos(min(px + vec2i(0, 1), sz)) - pc;
          let dyn = pc - viewPos(max(px - vec2i(0, 1), vec2i(0)));
          let ddx = select(dxp, dxn, abs(dxn.z) < abs(dxp.z));
          let ddy = select(dyp, dyn, abs(dyn.z) < abs(dyp.z));
          var nv = normalize(cross(ddy, ddx));
          if (nv.z < 0.0) { nv = -nv; }
          let n = normalize((cam.invView * vec4f(nv, 0.0)).xyz);
          let wp = (cam.invView * vec4f(pc, 1.0)).xyz;
          let v = normalize(cam.eye - wp);
          let uv = in.pos.xy / cam.viewport;
          let t = textureSampleLevel(thick, lin, uv, 0.0).x * u.thicknessScale;
          // refraction: offset the background lookup along the view-space normal, scaled by thickness
          let ruv = clamp(uv + nv.xy * vec2f(1.0, -1.0) * u.refraction * min(t * 10.0, 1.5), vec2f(0.0), vec2f(1.0));
          let behind = textureSampleLevel(scene, lin, ruv, 0.0).rgb;
          let trans = exp(-u.absorb * t);
          let refr = behind * trans * u.tint + (1.0 - trans) * vec3f(0.02, 0.07, 0.12) * 0.6;   // in-scatter keeps deep water from going black
          let NoV = clamp(dot(n, v), 0.0, 1.0);
          let F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
          let refl = skyColor(reflect(-v, n), u.sun);
          let h = normalize(v + u.sun);
          let spec = vec3f(2.0, 1.85, 1.6) * pow(max(dot(n, h), 0.0), 250.0) * 1.5;
          let col = mix(refr, refl, F) + spec;
          let clip = cam.proj * vec4f(pc, 1.0);
          return FO(vec4f(col, 1.0), clip.z / clip.w);
        }`,
    });
    this.w = 0; this.h = 0;
  }

  set(v) { Object.assign(this.values, v); return this; }

  _alloc(w, h) {
    for (const t of [this.zA, this.zB, this.zDepth, this.thick, this.sceneCopy]) t?.destroy();
    const TU = GPUTextureUsage, ctx = this.ctx;
    this.zA = ctx.texture({ size: [w, h], format: 'r32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.STORAGE_BINDING, label: 'ssf-z-a' });
    this.zB = ctx.texture({ size: [w, h], format: 'r32float', usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING, label: 'ssf-z-b' });
    this.zDepth = ctx.texture({ size: [w, h], format: 'depth32float', usage: TU.RENDER_ATTACHMENT, label: 'ssf-zbuffer' });
    this.thick = ctx.texture({ size: [w, h], format: 'r16float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'ssf-thickness' });
    this.sceneCopy = ctx.texture({ size: [w, h], format: 'rgba16float', usage: TU.COPY_DST | TU.TEXTURE_BINDING, label: 'ssf-scene' });
    this.w = w; this.h = h;
  }

  get data() { const d = this.o.data; return typeof d === 'function' ? d() : d; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.data.length; }

  afterScene(view) {
    const n = this.count;
    if (!n) return;
    if (view.width !== this.w || view.height !== this.h) this._alloc(view.width, view.height);
    const ctx = this.ctx, V = this.values, P = this.data;
    const uni = { radius: V.radius, scale: V.scale, offset: V.offset, matMask: V.matMask };
    // 1. depth
    let pass = ctx.renderPass({
      colorAttachments: [{ view: this.zA.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 1e6, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: this.zDepth.view, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    this.depthProg.draw(pass, view, 4, n, uni, { P });
    pass.end();
    // 2. thickness (occluded by the opaque scene)
    pass = ctx.renderPass({
      colorAttachments: [{ view: this.thick.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: { view: view.depth.view, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    this.thickProg.draw(pass, view, 4, n, uni, { P });
    pass.end();
    // 3. narrow-range filter: (horizontal, vertical) x iterations, ping-pong zA <-> zB
    const pxScale = view.height / (2 * Math.tan(view.camera.fovY / 2));
    for (let it = 0; it < this.iterations; it++) {
      this.filterK.run([this.w, this.h], { dir: [1, 0], radius: V.radius, pxScale, filterRadius: V.filterRadius, maxPixels: V.maxPixels }, { src: this.zA, dst: this.zB });
      this.filterK.run([this.w, this.h], { dir: [0, 1], radius: V.radius, pxScale, filterRadius: V.filterRadius, maxPixels: V.maxPixels }, { src: this.zB, dst: this.zA });
    }
    // 4. composite over a copy of the opaque scene
    ctx.endPass();
    ctx.encoder.copyTextureToTexture({ texture: view.hdr.texture }, { texture: this.sceneCopy.texture }, [this.w, this.h, 1]);
    pass = ctx.renderPass({
      colorAttachments: [{ view: view.hdr.view, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: view.depth.view, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    this.compositeProg.draw(pass, view, 3, 1, { absorb: V.absorb, tint: V.tint, sun: V.sun, thicknessScale: V.thicknessScale, refraction: V.refraction },
      { zTex: this.zA, thick: this.thick, scene: this.sceneCopy });
    pass.end();
  }
}

function norm(v) { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }
return { FluidSurface };
})();

// ---- src/shaders/math.js
const __m22 = (() => {
const defineModule = __m5.defineModule;


defineModule('math', {
  doc: 'Constants, rotations, quaternions, orthonormal bases, small helpers.',
  code: /* wgsl */`
const PI = 3.14159265358979;
const TAU = 6.28318530717959;
const INV_PI = 0.318309886183791;
const SQRT2 = 1.41421356237310;

fn sq(x: f32) -> f32 { return x * x; }
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
fn min3(v: vec3f) -> f32 { return min(v.x, min(v.y, v.z)); }
fn dot2(v: vec2f) -> f32 { return dot(v, v); }
fn dot3(v: vec3f) -> f32 { return dot(v, v); }
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 { return c + (x - a) * (d - c) / (b - a); }
/** Quintic smoothstep (C2 continuous). */
fn smootherstep(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
fn safeNormalize(v: vec3f) -> vec3f { let l = dot(v, v); return select(vec3f(0.0), v * inverseSqrt(l), l > 1e-20); }

/** 2D rotation matrix (counter-clockwise). */
fn rot2(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }

/** Rotation about a unit axis (Rodrigues). */
fn rotAxis(axis: vec3f, a: f32) -> mat3x3f {
  let c = cos(a); let s = sin(a); let t = 1.0 - c;
  let x = axis.x; let y = axis.y; let z = axis.z;
  return mat3x3f(
    t * x * x + c,     t * x * y + s * z, t * x * z - s * y,
    t * x * y - s * z, t * y * y + c,     t * y * z + s * x,
    t * x * z + s * y, t * y * z - s * x, t * z * z + c);
}

/** Orthonormal basis around n (Duff et al. 2017, branchless). Columns: t, b, n. */
fn basis(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}

// ---- quaternions (x, y, z, w) with w the scalar part
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn qconj(q: vec4f) -> vec4f { return vec4f(-q.xyz, q.w); }
/** Rotate v by unit quaternion q (15 mul/add form). */
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn qaxis(axis: vec3f, a: f32) -> vec4f { return vec4f(axis * sin(0.5 * a), cos(0.5 * a)); }
fn qmat(q: vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y));
}
`,
});
return {  };
})();

// ---- src/shaders/linalg.js
const __m23 = (() => {
const defineModule = __m5.defineModule;


// Small dense linear algebra for physics kernels: outer products, 3x3 symmetric
// eigen-decomposition (cyclic Jacobi) and SVD built on it, polar decomposition.
defineModule('linalg', {
  doc: '3x3 helpers: outer, identity, symmetric eigen-decomposition (Jacobi), SVD, polar rotation.',
  code: /* wgsl */`
const I3 = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
/** a ⊗ b = a bᵀ */
fn outer3(a: vec3f, b: vec3f) -> mat3x3f { return mat3x3f(a * b.x, a * b.y, a * b.z); }
fn trace3(m: mat3x3f) -> f32 { return m[0].x + m[1].y + m[2].z; }
fn diag3(d: vec3f) -> mat3x3f { return mat3x3f(d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z); }

fn si_jacobiRotate(A: ptr<function, mat3x3f>, V: ptr<function, mat3x3f>, p: i32, q: i32) {
  let apq = (*A)[q][p];
  if (abs(apq) < 1e-30) { return; }
  let theta = ((*A)[q][q] - (*A)[p][p]) / (2.0 * apq);
  let t = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
  let c = inverseSqrt(t * t + 1.0);
  let s = t * c;
  var J = I3;
  J[p][p] = c; J[q][q] = c; J[q][p] = s; J[p][q] = -s;
  *A = transpose(J) * (*A) * J;
  *V = (*V) * J;
}

struct Eig3 { V: mat3x3f, e: vec3f }
/** Symmetric A = V diag(e) Vᵀ (4 cyclic Jacobi sweeps: ~1e-6 relative accuracy in f32). */
fn eigSym3(A0: mat3x3f) -> Eig3 {
  var A = A0;
  var V = I3;
  for (var sweep = 0; sweep < 4; sweep++) {
    si_jacobiRotate(&A, &V, 0, 1);
    si_jacobiRotate(&A, &V, 0, 2);
    si_jacobiRotate(&A, &V, 1, 2);
  }
  return Eig3(V, vec3f(A[0].x, A[1].y, A[2].z));
}

struct SVD3 { U: mat3x3f, s: vec3f, V: mat3x3f }
/**
 * F = U diag(s) Vᵀ with U, V rotations (det = +1); s may have a negative last
 * entry if det(F) < 0. Built from the eigenvectors of FᵀF; U columns are
 * F v_i / s_i, re-orthogonalised for robustness.
 */
fn svd3(F: mat3x3f) -> SVD3 {
  let E = eigSym3(transpose(F) * F);
  // sort eigenpairs by decreasing eigenvalue
  var e = E.e; var V = E.V;
  if (e.x < e.y) { e = e.yxz; V = mat3x3f(V[1], V[0], -V[2]); }
  if (e.x < e.z) { e = e.zyx; V = mat3x3f(V[2], -V[1], V[0]); }
  if (e.y < e.z) { e = e.xzy; V = mat3x3f(-V[0], V[2], V[1]); }
  if (determinant(V) < 0.0) { V[2] = -V[2]; }
  var s = sqrt(max(e, vec3f(0.0)));
  var u0 = F * V[0];
  let l0 = length(u0);
  u0 = select(vec3f(1.0, 0.0, 0.0), u0 / l0, l0 > 1e-12);
  var u1 = F * V[1];
  u1 = u1 - dot(u1, u0) * u0;
  let l1 = length(u1);
  if (l1 > 1e-12) { u1 = u1 / l1; } else {
    u1 = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(u0.x) < 0.9);
    u1 = normalize(u1 - dot(u1, u0) * u0);
  }
  let u2 = cross(u0, u1);
  // sign of the smallest singular value carries det(F)
  if (dot(F * V[2], u2) < 0.0) { s.z = -s.z; }
  return SVD3(mat3x3f(u0, u1, u2), s, V);
}

/** Rotation part of F (polar decomposition) via the SVD: R = U Vᵀ. */
fn polarR(F: mat3x3f) -> mat3x3f {
  let d = svd3(F);
  return d.U * transpose(d.V);
}
`,
});
return {  };
})();

// ---- src/shaders/hash.js
const __m24 = (() => {
const defineModule = __m5.defineModule;


// Integer hashing. Float "fract(sin(x) * 43758.5453)" hashes are both slower
// (sin is a multi-instruction SFU op) and break down at large inputs and on
// GPUs with low-precision sin. Integer hashes are exact, portable and fast.
// pcg / pcg3d / pcg4d: Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT 2020.
// lowbias32: Chris Wellons' hash-prospector (bias ~0.17, two multiplies).
defineModule('hash', {
  doc: 'PCG-family and lowbias32 integer hashes, float conversions, PCG random numbers, low-discrepancy sequences.',
  code: /* wgsl */`
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn pcg2d(p: vec2u) -> vec2u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  return v ^ (v >> vec2u(16u));
}
fn pcg3d(p: vec3u) -> vec3u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn pcg4d(p: vec4u) -> vec4u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  v = v ^ (v >> vec4u(16u));
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  return v;
}
fn lowbias32(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u; x *= 0x7feb352du;
  x ^= x >> 15u; x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}

/** u32 -> uniform float in [0, 1) using the top 24 bits. */
fn u2f(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }
fn u2f2(x: vec2u) -> vec2f { return vec2f(x >> vec2u(8u)) * (1.0 / 16777216.0); }
fn u2f3(x: vec3u) -> vec3f { return vec3f(x >> vec3u(8u)) * (1.0 / 16777216.0); }
fn u2f4(x: vec4u) -> vec4f { return vec4f(x >> vec4u(8u)) * (1.0 / 16777216.0); }

// Float-coordinate hashes in [0, 1). Inputs are hashed by their bit pattern.
// pcg2d's x output skips the final mixing step its y output gets: over horizontally adjacent
// pixels it is correlated (χ² 1073 on 256 pair buckets, where ≲ 330 is independent), so the 2D
// hashes use y, and hash22 derives its second value from both (χ² 226 and 258).
fn hash11(p: f32) -> f32 { return u2f(pcg(bitcast<u32>(p))); }
fn hash21(p: vec2f) -> f32 { return u2f(pcg2d(bitcast<vec2u>(p)).y); }
fn hash22(p: vec2f) -> vec2f { let v = pcg2d(bitcast<vec2u>(p)); return u2f2(vec2u(v.y, pcg(v.x ^ v.y))); }
fn hash31(p: vec3f) -> f32 { return u2f(pcg3d(bitcast<vec3u>(p)).x); }
fn hash33(p: vec3f) -> vec3f { return u2f3(pcg3d(bitcast<vec3u>(p))); }
fn hash44(p: vec4f) -> vec4f { return u2f4(pcg4d(bitcast<vec4u>(p))); }
// Integer-lattice hashes (for cells / ids).
fn ihash2(p: vec2i) -> u32 { return pcg2d(bitcast<vec2u>(p)).y; }
fn ihash3(p: vec3i) -> u32 { return pcg3d(bitcast<vec3u>(p)).x; }

// ---- PCG random number generator (state in a function-scope u32)
fn rngSeed(a: u32, b: u32) -> u32 { return pcg(a ^ pcg(b + 0x9e3779b9u)); }
fn randU32(s: ptr<function, u32>) -> u32 {
  let state = *s;
  *s = state * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand(s: ptr<function, u32>) -> f32 { return u2f(randU32(s)); }
fn rand2(s: ptr<function, u32>) -> vec2f { return vec2f(rand(s), rand(s)); }
fn rand3(s: ptr<function, u32>) -> vec3f { return vec3f(rand(s), rand(s), rand(s)); }
/** Uniform direction on the unit sphere. */
fn randDir(s: ptr<function, u32>) -> vec3f {
  let u = rand2(s);
  let z = 1.0 - 2.0 * u.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let a = 6.28318530718 * u.y;
  return vec3f(r * cos(a), r * sin(a), z);
}
/** Uniform point inside the unit ball. */
fn randBall(s: ptr<function, u32>) -> vec3f { return randDir(s) * pow(rand(s), 1.0 / 3.0); }
/** Standard normal pair (Box–Muller). */
fn randNormal2(s: ptr<function, u32>) -> vec2f {
  let u = rand2(s);
  let r = sqrt(-2.0 * log(max(u.x, 1e-12)));
  let a = 6.28318530718 * u.y;
  return r * vec2f(cos(a), sin(a));
}
/** Cosine-weighted hemisphere sample around +z. */
fn randCosHemi(s: ptr<function, u32>) -> vec3f {
  let u = rand2(s);
  let r = sqrt(u.x);
  let a = 6.28318530718 * u.y;
  return vec3f(r * cos(a), r * sin(a), sqrt(max(0.0, 1.0 - u.x)));
}

// ---- low-discrepancy sequences
/** Roberts R2 sequence: best-known 2D additive recurrence. */
fn r2seq(n: u32) -> vec2f { return fract(vec2f(0.5) + f32(n) * vec2f(0.754877666246693, 0.569840290998053)); }
/** Interleaved gradient noise (Jimenez 2014): cheap per-pixel dither. */
fn ign(px: vec2f) -> f32 { return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715)))); }
`,
});
return {  };
})();

// ---- src/physics/rigid-wgsl.js
const __m21 = (() => {
const defineModule = __m5.defineModule;
// WGSL for the rigid-body solver (physics/rigid.js): data layouts, linearised
// contacts, joint constraint rows, the register-resident 6×6 Newton solve,
// narrowphase manifolds (sphere, box, capsule, floor) and ray tests.





const BODY_VEC4 = 12;
const CONTACT_VEC4 = 6;
const JOINT_VEC4 = 13;
const SHAPES = { sphere: 0, box: 1, capsule: 2 };
const FLAGS = { kinematic: 1, sensor: 2, disabled: 4, fixedRotation: 8 };
const JOINT_TYPES = { ball: 0, hinge: 1, fixed: 2, distance: 3, slider: 4 };
const JOINT_FLAGS = { broken: 1, collide: 2, limit: 4, motor: 8, rope: 16, worldAnchors: 32, disabled: 64 };

defineModule('rigid', {
  deps: ['math', 'hash', 'linalg'],
  code: /* wgsl */`
struct Body {
  x: vec4f,      // position; w = bounding radius
  q: vec4f,      // orientation quaternion (x, y, z, w)
  v: vec4f,      // linear velocity; w = friction coefficient
  w: vec4f,      // angular velocity (world); w = previous velocity along gravity
  x0: vec4f,     // pose at the start of the step
  q0: vec4f,
  y: vec4f,      // inertial target
  qy: vec4f,
  size: vec4f,   // box: half extents; sphere: (r, r, r); capsule: (r, half length, r); w = shape
  mass: vec4f,   // (m, Ixx, Iyy, Izz) in the body frame; m = 0 -> static or kinematic
  props: vec4f,  // (gravity scale, linear damping, angular damping, -)
  info: vec4u,   // (collision group bits, collision mask bits, flags, user data)
}
const F_KINEMATIC = 1u;
const F_SENSOR = 2u;
const F_DISABLED = 4u;
const F_FIXED_ROT = 8u;
struct Contact {
  ab: vec4u,     // (body a, body b or STATIC, sticking, sensor)
  n: vec4f,      // world normal pointing from b to a; w = friction
  ra: vec4f,     // contact anchor on a, a's frame
  rb: vec4f,     // contact anchor on b, b's frame (world point if b is STATIC)
  lam: vec4f,    // (λn, λt1, λt2, -)
  k: vec4f,      // (kn, kt1, kt2, m_eff/Δt²)
}
struct Joint {
  ab: vec4u,     // (body a, body b or STATIC = world, type, flags)
  ra: vec4f,     // anchor on a, a's frame; w = break force (0: unbreakable)
  rb: vec4f,     // anchor on b, b's frame (world point if b is STATIC); w = length (distance joints)
  qrel: vec4f,   // rest orientation of b relative to a: conj(q_a) ⊗ q_b
  axa: vec4f,    // axis in a's frame (hinge, slider); w = lower limit
  axb: vec4f,    // axis in b's frame (world if b is STATIC); w = upper limit
  prm: vec4f,    // (stiffness: 0 = rigid, damping, motor speed, max motor force/torque)
  lam0: vec4f,   // multipliers, rows 0-3
  lam1: vec4f,   //              rows 4-7
  k0: vec4f,     // penalty stiffness, rows 0-3
  k1: vec4f,     //                    rows 4-7
  c00: vec4f,    // constraint values at the start of the step, rows 0-3
  c01: vec4f,    //                                             rows 4-7
}
const STATIC = 0xffffffffu;
const J_BALL = 0u;
const J_HINGE = 1u;
const J_FIXED = 2u;
const J_DISTANCE = 3u;
const J_SLIDER = 4u;
const JF_BROKEN = 1u;
const JF_COLLIDE = 2u;
const JF_LIMIT = 4u;
const JF_MOTOR = 8u;
const JF_ROPE = 16u;
const JF_WORLD_ANCHORS = 32u;
const JF_DISABLED = 64u;

fn shapeOf(b: Body) -> u32 { return u32(b.size.w + 0.5); }
fn unitv(k: u32) -> vec3f { return vec3f(select(0.0, 1.0, k == 0u), select(0.0, 1.0, k == 1u), select(0.0, 1.0, k == 2u)); }
fn qmulv(q: vec4f, w: vec3f) -> vec4f { return qmul(vec4f(w, 0.0), q); }
// Rotations use the exact exponential and logarithm maps. The small-angle forms
// (q + ½Δt ω⊗q, ω = 2Δq.xyz/Δt) lose a fraction ≈ φ²/8 of the spin every step (φ = |ω|Δt),
// which brakes anything that turns fast: a ball rolling at 2 m/s with r = 0.1 lost half its
// speed in 10 s on flat ground.
/** q turned by the world-frame rotation vector w·dt. */
fn qinteg(q: vec4f, w: vec3f, dt: f32) -> vec4f {
  let a = length(w) * dt;
  if (a < 1e-6) { return normalize(q + 0.5 * dt * qmulv(q, w)); }
  let h = 0.5 * a;
  return normalize(qmul(vec4f(w * (dt * sin(h) / a), cos(h)), q));
}
/** Rotation vector taking q0 to q (world frame). */
fn rotvec(q: vec4f, q0: vec4f) -> vec3f {
  var d = qmul(q, qconj(q0));
  if (d.w < 0.0) { d = -d; }
  let s = length(d.xyz);
  if (s < 1e-7) { return 2.0 * d.xyz; }
  return d.xyz * (2.0 * atan2(s, d.w) / s);
}
/** Inverse world inertia R diag(1/I) Rᵀ (zero for fixed rotation or static bodies). */
fn invInertiaWorld(b: Body) -> mat3x3f {
  if (b.mass.x <= 0.0 || (b.info.z & F_FIXED_ROT) != 0u) { return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)); }
  let R = qmat(b.q);
  return R * diag3(1.0 / max(b.mass.yzw, vec3f(1e-30))) * transpose(R);
}

// The part of a body a constraint needs (4 of its 12 vec4s).
struct Pose { x: vec3f, x0: vec3f, q: vec4f, q0: vec4f }

// Contacts are linearised at the start of the step (AVBD: first-order Taylor expansion of
// collision constraints): C(x) = C0 + J0·Δ with constant Jacobians, so every body's local
// problem is exactly quadratic and its Newton step is exact.
// gap: separation of the two anchors at the start of the step (normal part = C0,
//      tangential part = friction drift of a sticking contact);
// rel: displacement of the anchor on a relative to the anchor on b since the step began.
struct Lin { gap: vec3f, rel: vec3f, ra0: vec3f, rb0: vec3f }
fn contactLin(c: Contact, A: Pose, Bp: Pose, hasB: bool) -> Lin {
  var l: Lin;
  l.ra0 = qrot(A.q0, c.ra.xyz);
  let da = A.x - A.x0 + cross(rotvec(A.q, A.q0), l.ra0);
  var db = vec3f(0.0);
  var pb = c.rb.xyz;
  l.rb0 = vec3f(0.0);
  if (hasB) {
    l.rb0 = qrot(Bp.q0, c.rb.xyz);
    db = Bp.x - Bp.x0 + cross(rotvec(Bp.q, Bp.q0), l.rb0);
    pb = Bp.x0 + l.rb0;
  }
  l.gap = A.x0 + l.ra0 - pb;
  l.rel = da - db;
  return l;
}
// Normal constraint. Overlap present at the start of the step is kept with weight α
// (α = 1: only prevent further penetration; α = 0: remove it all); a speculative gap
// (c0 > 0) always counts in full so bodies can close it.
fn contactCn(L: Lin, nrm: vec3f, alpha: f32, slop: f32) -> f32 {
  let c0 = dot(nrm, L.gap) + slop;
  return select(c0 * (1.0 - alpha), c0, c0 > 0.0) + dot(nrm, L.rel);
}

// ------------------------------------------------------------------ joints
// One joint row: its value C at the current poses and the Jacobians for body a (la, aa)
// and body b (lb, ab). kind: 0 equality, 1 inequality C ≥ 0, 2 motor (force bounded by
// ±prm.w), 3 inactive. Jacobians of the angular rows use the small-angle approximation
// around the satisfied state, as AVBD does for rigid-body joints.
// geo: curvature of C along the linear directions (distance joints: 1/len, from H = (I − nnᵀ)/len),
// used for AVBD's lumped geometric-stiffness term |f|·geo on the positional block.
struct JRow { C: f32, la: vec3f, aa: vec3f, lb: vec3f, ab: vec3f, kind: u32, geo: f32 }
const JK_EQ = 0u;
const JK_GE = 1u;
const JK_MOTOR = 2u;
const JK_OFF = 3u;
fn jointRows(t: u32) -> u32 {
  switch t {
    case 0u: { return 3u; }     // ball: point
    case 1u: { return 7u; }     // hinge: point, 2 axis-alignment rows, motor, limit
    case 2u: { return 6u; }     // fixed: point, 3 orientation rows
    case 3u: { return 1u; }     // distance
    default: { return 7u; }     // slider: 2 perpendicular rows, 3 orientation rows, motor, limit
  }
}
fn wrapAngle(a: f32) -> f32 { return a - 6.283185307 * round(a / 6.283185307); }
/** Hinge angle of b relative to a about the hinge axis (exact, any magnitude); a relative to b is its negative. */
fn hingeAngle(j: Joint, qa: vec4f, qb: vec4f) -> f32 {
  let qd = qmul(qmul(qconj(qa), qb), qconj(j.qrel));
  return wrapAngle(2.0 * atan2(dot(qd.xyz, j.axa.xyz), qd.w));
}
/** Small rotation vector taking b's rest orientation (relative to a) to its current one. */
fn orientErr(j: Joint, qa: vec4f, qb: vec4f) -> vec3f {
  var e = qmul(qb, qconj(qmul(qa, j.qrel)));
  if (e.w < 0.0) { e = -e; }
  return 2.0 * e.xyz;
}
fn jointRow(j: Joint, A: Pose, B: Pose, hasB: bool, row: u32, dt: f32) -> JRow {
  var r: JRow;
  r.kind = JK_EQ;
  let t = j.ab.z;
  let raw = qrot(A.q, j.ra.xyz);
  let pa = A.x + raw;
  var rbw = vec3f(0.0);
  var pb = j.rb.xyz;
  var qb = vec4f(0.0, 0.0, 0.0, 1.0);
  var qb0 = qb;
  if (hasB) { rbw = qrot(B.q, j.rb.xyz); pb = B.x + rbw; qb = B.q; qb0 = B.q0; }
  let d = pa - pb;
  if (t == J_DISTANCE) {
    let len = length(d);
    var n = select(vec3f(0.0, 1.0, 0.0), d / len, len > 1e-9);
    r.C = len - j.rb.w;
    r.geo = 1.0 / max(len, 1e-3);
    if ((j.ab.w & JF_ROPE) != 0u) { r.C = -r.C; n = -n; r.kind = JK_GE; }
    r.la = n; r.aa = cross(raw, n); r.lb = -n; r.ab = -cross(rbw, n);
    return r;
  }
  if (t != J_SLIDER && row < 3u) {           // point: anchors coincide
    let e = unitv(row);
    r.C = d[row]; r.la = e; r.aa = cross(raw, e); r.lb = -e; r.ab = -cross(rbw, e);
    return r;
  }
  let aw = qrot(A.q, j.axa.xyz);
  if (t == J_FIXED || (t == J_SLIDER && row >= 2u && row < 5u)) {   // orientation locked
    let k = select(row - 3u, row - 2u, t == J_SLIDER);
    let e = unitv(k);
    r.C = orientErr(j, A.q, qb)[k]; r.aa = -e; r.ab = e;
    return r;
  }
  let T = basis(aw);
  if (t == J_HINGE && row < 5u) {            // hinge axes stay aligned
    var bw = j.axb.xyz;
    if (hasB) { bw = qrot(qb, j.axb.xyz); }
    let tk = select(T[1], T[0], row == 3u);
    r.C = dot(cross(aw, bw), tk); r.aa = -tk; r.ab = tk;
    return r;
  }
  if (t == J_SLIDER && row < 2u) {           // no motion across the slide axis
    let tk = select(T[1], T[0], row == 0u);
    r.C = dot(tk, d); r.la = tk; r.aa = cross(pb - A.x, tk); r.lb = -tk; r.ab = -cross(rbw, tk);
    return r;
  }
  // motor (row 5) and limit (row 6) along the axis: the angle (hinge) or slide (slider) of a
  // relative to b, so for a body attached to the world they are simply its own angle / travel
  var s: f32;
  var la = vec3f(0.0); var aa = aw; var lb = vec3f(0.0); var ab = -aw;
  if (t == J_HINGE) { s = -hingeAngle(j, A.q, qb); }
  else { s = dot(pa - pb, aw); la = aw; aa = cross(aw, A.x - pb); lb = -aw; ab = cross(aw, rbw); }
  if (row == 5u) {
    if ((j.ab.w & JF_MOTOR) == 0u) { r.kind = JK_OFF; return r; }
    let s0 = j.c01.y;                         // value at the start of the step (stored by the prep pass)
    var ds = s - s0;
    if (t == J_HINGE) { ds = wrapAngle(ds); }
    r.C = ds - j.prm.z * dt; r.la = la; r.aa = aa; r.lb = lb; r.ab = ab; r.kind = JK_MOTOR;
    return r;
  }
  if ((j.ab.w & JF_LIMIT) == 0u) { r.kind = JK_OFF; return r; }
  let lo = s - j.axa.w;
  let hi = j.axb.w - s;
  r.kind = JK_GE;
  if (lo < hi) { r.C = lo; r.la = la; r.aa = aa; r.lb = lb; r.ab = ab; }
  else { r.C = hi; r.la = -la; r.aa = -aa; r.lb = -lb; r.ab = -ab; }
  if (r.C > 0.25) { r.kind = JK_OFF; }      // far from both limits: inactive
  return r;
}
fn rowGet(a: vec4f, b: vec4f, row: u32) -> f32 { return select(b[(row - 4u) & 3u], a[row & 3u], row < 4u); }

// ------------------------------------------------------------------ 6x6 solve
// 3x3 Cholesky factor (lower triangle of a symmetric positive-definite matrix) and solve.
struct Chol3 { d: vec3f, l10: f32, l20: f32, l21: f32 }
fn chol3(M: mat3x3f) -> Chol3 {
  var c: Chol3;
  c.d.x = sqrt(max(M[0][0], 1e-30));
  c.l10 = M[0][1] / c.d.x;
  c.l20 = M[0][2] / c.d.x;
  c.d.y = sqrt(max(M[1][1] - c.l10 * c.l10, 1e-30));
  c.l21 = (M[1][2] - c.l20 * c.l10) / c.d.y;
  c.d.z = sqrt(max(M[2][2] - c.l20 * c.l20 - c.l21 * c.l21, 1e-30));
  return c;
}
fn cholSolve(c: Chol3, v: vec3f) -> vec3f {
  let z0 = v.x / c.d.x;
  let z1 = (v.y - c.l10 * z0) / c.d.y;
  let z2 = (v.z - c.l20 * z0 - c.l21 * z1) / c.d.z;
  let x2 = z2 / c.d.z;
  let x1 = (z1 - c.l21 * x2) / c.d.y;
  return vec3f((z0 - c.l10 * x1 - c.l20 * x2) / c.d.x, x1, x2);
}
// Solve the SPD block system [[A, B], [Bᵀ, D]] (x; y) = (a; b) through the Schur complement
// S = D − Bᵀ A⁻¹ B. Same arithmetic as a 6x6 LDLᵀ, but everything stays in registers
// (a dynamically indexed array<f32, 36> spills to memory on most GPUs).
fn solveBlock6(A: mat3x3f, B: mat3x3f, D: mat3x3f, a: vec3f, b: vec3f) -> array<vec3f, 2> {
  let LA = chol3(A);
  let AiB = mat3x3f(cholSolve(LA, B[0]), cholSolve(LA, B[1]), cholSolve(LA, B[2]));
  let Aia = cholSolve(LA, a);
  let y = cholSolve(chol3(D - transpose(B) * AiB), b - transpose(B) * Aia);
  return array<vec3f, 2>(Aia - AiB * y, y);
}
`,
});

// Narrowphase geometry. Spliced into the narrowphase kernels. Manifolds carry a normal
// per point (capsule contacts can have several), always pointing from body b to body a.
const NARROW_WGSL = /* wgsl */`
struct Manifold { count: u32, nrm: array<vec3f, 8>, pa: array<vec3f, 8>, pb: array<vec3f, 8> }

fn projBox(L: vec3f, R: mat3x3f, h: vec3f) -> f32 {
  return h.x * abs(dot(L, R[0])) + h.y * abs(dot(L, R[1])) + h.z * abs(dot(L, R[2]));
}
fn segClosest(p1: vec3f, d1: vec3f, e1: f32, p2: vec3f, d2: vec3f, e2: f32) -> array<vec3f, 2> {
  let r = p1 - p2;
  let b = dot(d1, d2); let c = dot(d1, r); let f = dot(d2, r);
  let den = 1.0 - b * b;
  var s = select(0.0, (b * f - c) / den, den > 1e-6);
  s = clamp(s, -e1, e1);
  let t = clamp(b * s + f, -e2, e2);
  s = clamp(b * t - c, -e1, e1);
  return array<vec3f, 2>(p1 + d1 * s, p2 + d2 * t);
}
// Append a point unless one already sits within dmin of it.
fn addPoint(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, pa: vec3f, pb: vec3f, n: vec3f, s: f32, dmin: f32) {
  let c = (*M).count;
  if (c >= 8u) { return; }
  for (var k = 0u; k < c; k++) { if (length((*M).pa[k] - pa) < dmin) { return; } }
  (*M).pa[c] = pa; (*M).pb[c] = pb; (*M).nrm[c] = n; (*sep)[c] = s;
  (*M).count = c + 1u;
}
// Keep at most 4 points: deepest, farthest from it, and the two maximising the spanned area.
fn reduce4(m: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, axis: vec3f) {
  let n = (*m).count;
  if (n <= 4u) { return; }
  var i0 = 0u;
  for (var k = 1u; k < n; k++) { if ((*sep)[k] < (*sep)[i0]) { i0 = k; } }
  var i1 = i0; var best = -1.0;
  for (var k = 0u; k < n; k++) { let d = length((*m).pb[k] - (*m).pb[i0]); if (d > best) { best = d; i1 = k; } }
  var i2 = i0; var i3 = i0; var amax = 0.0; var amin = 0.0;
  for (var k = 0u; k < n; k++) {
    let a = dot(cross((*m).pb[i1] - (*m).pb[i0], (*m).pb[k] - (*m).pb[i0]), axis);
    if (a > amax) { amax = a; i2 = k; }
    if (a < amin) { amin = a; i3 = k; }
  }
  var idx = array<u32, 4>(i0, i1, i2, i3);
  var out: Manifold; out.count = 0u;
  var outSep: array<f32, 8>;
  for (var q = 0u; q < 4u; q++) {
    var dup = false;
    for (var r = 0u; r < q; r++) { if (idx[r] == idx[q]) { dup = true; } }
    if (dup) { continue; }
    out.pa[out.count] = (*m).pa[idx[q]]; out.pb[out.count] = (*m).pb[idx[q]]; out.nrm[out.count] = (*m).nrm[idx[q]];
    outSep[out.count] = (*sep)[idx[q]];
    out.count += 1u;
  }
  *m = out; *sep = outSep;
}
fn flipManifold(M: Manifold) -> Manifold {
  var o = M;
  for (var k = 0u; k < M.count; k++) { o.nrm[k] = -M.nrm[k]; o.pa[k] = M.pb[k]; o.pb[k] = M.pa[k]; }
  return o;
}

fn boxBox(A: Body, Bb: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  let Ra = qmat(A.q0); let Rb = qmat(Bb.q0);
  let ca = A.x0.xyz; let cb = Bb.x0.xyz; let ha = A.size.xyz; let hb = Bb.size.xyz;
  let d = cb - ca;
  var best = 1e30; var axis = vec3f(0.0, 1.0, 0.0); var kind = 0u; var ai = 0u; var bj = 0u;
  for (var k = 0u; k < 3u; k++) {
    let L = Ra[k];
    let pen = ha[k] + projBox(L, Rb, hb) - abs(dot(d, L));
    if (pen < -margin) { return M; }
    if (pen < best) { best = pen; axis = L; kind = 0u; ai = k; }
  }
  for (var k = 0u; k < 3u; k++) {
    let L = Rb[k];
    let pen = projBox(L, Ra, ha) + hb[k] - abs(dot(d, L));
    if (pen < -margin) { return M; }
    if (pen < best * 0.98 - 1e-4) { best = pen; axis = L; kind = 1u; ai = k; }
  }
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      var L = cross(Ra[i], Rb[j]);
      let len = length(L);
      if (len < 1e-4) { continue; }
      L = L / len;
      let pen = projBox(L, Ra, ha) + projBox(L, Rb, hb) - abs(dot(d, L));
      if (pen < -margin) { return M; }
      // edge axes must be clearly better than face axes (face manifolds are far more stable)
      if (pen < best * 0.9 - 2e-3) { best = pen; axis = L; kind = 2u; ai = i; bj = j; }
    }
  }
  var n = axis;
  if (dot(n, d) > 0.0) { n = -n; }   // n points from b towards a
  var sep: array<f32, 8>;
  if (kind == 2u) {
    // edge–edge: supporting edge of a (toward b, i.e. -n) and of b (toward a, +n)
    var pa = ca; var pb = cb;
    for (var k = 0u; k < 3u; k++) {
      if (k != ai) { pa += Ra[k] * ha[k] * select(1.0, -1.0, dot(Ra[k], n) > 0.0); }
      if (k != bj) { pb += Rb[k] * hb[k] * select(-1.0, 1.0, dot(Rb[k], n) > 0.0); }
    }
    let cp = segClosest(pa, Ra[ai], ha[ai], pb, Rb[bj], hb[bj]);
    M.pa[0] = cp[0]; M.pb[0] = cp[1]; M.nrm[0] = n; M.count = 1u;
    return M;
  }
  // face contact: the axis owner is the reference box
  let refA = kind == 0u;
  var Rr = Ra; var cr = ca; var hr = ha; var Ri = Rb; var cI = cb; var hi = hb;
  if (!refA) { Rr = Rb; cr = cb; hr = hb; Ri = Ra; cI = ca; hi = ha; }
  let toInc = select(n, -n, refA);                    // reference face normal, pointing at the incident box
  let sk = select(-1.0, 1.0, dot(Rr[ai], toInc) > 0.0);
  let rn = Rr[ai] * sk;
  let rc = cr + rn * hr[ai];
  let k1 = (ai + 1u) % 3u; let k2 = (ai + 2u) % 3u;
  // incident face: most anti-parallel to the reference normal
  var ij = 0u; var bd = -1.0;
  for (var k = 0u; k < 3u; k++) { let v = abs(dot(Ri[k], rn)); if (v > bd) { bd = v; ij = k; } }
  let si = select(1.0, -1.0, dot(Ri[ij], rn) > 0.0);
  let ic = cI + Ri[ij] * hi[ij] * si;
  let j1 = (ij + 1u) % 3u; let j2 = (ij + 2u) % 3u;
  let e1 = Ri[j1] * hi[j1]; let e2 = Ri[j2] * hi[j2];
  var poly: array<vec3f, 8>;
  poly[0] = ic + e1 + e2; poly[1] = ic - e1 + e2; poly[2] = ic - e1 - e2; poly[3] = ic + e1 - e2;
  var np = 4u;
  // Sutherland–Hodgman against the 4 side planes of the reference face
  for (var pl = 0u; pl < 4u; pl++) {
    let ax = select(Rr[k2], Rr[k1], pl < 2u);
    let ext = select(hr[k2], hr[k1], pl < 2u);
    let sgn = select(-1.0, 1.0, (pl & 1u) == 0u);
    var outp: array<vec3f, 8>;
    var no = 0u;
    for (var v = 0u; v < np; v++) {
      let a = poly[v];
      let b = poly[(v + 1u) % np];
      let da = sgn * dot(a - cr, ax) - ext;
      let db = sgn * dot(b - cr, ax) - ext;
      if (da <= 0.0 && no < 8u) { outp[no] = a; no += 1u; }
      if ((da <= 0.0) != (db <= 0.0) && no < 8u) { outp[no] = a + (b - a) * (da / (da - db)); no += 1u; }
    }
    poly = outp; np = no;
    if (np == 0u) { return M; }
  }
  for (var v = 0u; v < np; v++) {
    let p = poly[v];
    let s = dot(p - rc, rn);
    if (s > margin) { continue; }
    let onRef = p - rn * s;
    if (refA) { M.pa[M.count] = onRef; M.pb[M.count] = p; } else { M.pa[M.count] = p; M.pb[M.count] = onRef; }
    M.nrm[M.count] = n;
    sep[M.count] = s;
    M.count += 1u;
  }
  reduce4(&M, &sep, rn);
  return M;
}

fn boxFloor(A: Body, floorY: f32, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  let R = qmat(A.q0);
  var sep: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3f(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
    let p = A.x0.xyz + R * (s * A.size.xyz);
    let sp = p.y - floorY;
    if (sp < margin) { M.pa[M.count] = p; M.pb[M.count] = vec3f(p.x, floorY, p.z); M.nrm[M.count] = vec3f(0.0, 1.0, 0.0); sep[M.count] = sp; M.count += 1u; }
  }
  reduce4(&M, &sep, vec3f(0.0, 1.0, 0.0));
  return M;
}

fn capsuleAxis(b: Body) -> vec3f { return qrot(b.q0, vec3f(0.0, 1.0, 0.0)); }

fn capsuleFloor(A: Body, floorY: f32, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let ax = capsuleAxis(A) * A.size.y;
  let r = A.size.x;
  for (var e = 0u; e < 2u; e++) {
    let c = A.x0.xyz + select(-ax, ax, e == 0u);
    let s = c.y - r - floorY;
    if (s < margin) { addPoint(&M, &sep, c - vec3f(0.0, r, 0.0), vec3f(c.x, floorY, c.z), vec3f(0.0, 1.0, 0.0), s, 0.05 * r); }
  }
  return M;
}

// sphere (centre c, radius r) of body a against box Bx (body b)
fn sphereBoxInto(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, c: vec3f, r: f32, Bx: Body, margin: f32) {
  let R = qmat(Bx.q0);
  let lc = transpose(R) * (c - Bx.x0.xyz);
  let h = Bx.size.xyz;
  let qc = clamp(lc, -h, h);
  var n: vec3f; var pb: vec3f; var dist: f32;
  if (any(lc != qc)) {
    let dl = lc - qc;
    dist = length(dl);
    n = R * (dl / dist);
    pb = Bx.x0.xyz + R * qc;
  } else {
    // centre inside the box: push out through the nearest face
    let gap = h - abs(lc);
    var ax = 0u;
    if (gap.y < gap[ax]) { ax = 1u; }
    if (gap.z < gap[ax]) { ax = 2u; }
    var ln = vec3f(0.0); ln[ax] = select(-1.0, 1.0, lc[ax] >= 0.0);
    n = R * ln;
    var fp = lc; fp[ax] = h[ax] * ln[ax];
    pb = Bx.x0.xyz + R * fp;
    dist = -gap[ax];
  }
  let s = dist - r;
  if (s > margin) { return; }
  addPoint(M, sep, c - n * r, pb, n, s, 0.05 * r);
}
fn sphereSphereInto(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, ca: vec3f, ra: f32, cb: vec3f, rb: f32, margin: f32) {
  let d = ca - cb;
  let dist = length(d);
  let s = dist - ra - rb;
  if (s > margin) { return; }
  let n = select(vec3f(0.0, 1.0, 0.0), d / dist, dist > 1e-6);
  addPoint(M, sep, ca - n * ra, cb + n * rb, n, s, 0.05 * min(ra, rb));
}

fn capsuleSphere(Cp: Body, S: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let u = capsuleAxis(Cp);
  let t = clamp(dot(S.x0.xyz - Cp.x0.xyz, u), -Cp.size.y, Cp.size.y);
  sphereSphereInto(&M, &sep, Cp.x0.xyz + u * t, Cp.size.x, S.x0.xyz, S.size.x, margin);
  return M;
}

fn capsuleCapsule(A: Body, Bb: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let ua = capsuleAxis(A); let ub = capsuleAxis(Bb);
  let ha = A.size.y; let hb = Bb.size.y; let ra = A.size.x; let rb = Bb.size.x;
  let ca = A.x0.xyz; let cb = Bb.x0.xyz;
  let cp = segClosest(ca, ua, ha, cb, ub, hb);
  sphereSphereInto(&M, &sep, cp[0], ra, cp[1], rb, margin);
  if (abs(dot(ua, ub)) > 0.95) {   // nearly parallel: the segment ends too, for a stable line contact
    for (var e = 0u; e < 2u; e++) {
      let pa = ca + ua * select(-ha, ha, e == 0u);
      sphereSphereInto(&M, &sep, pa, ra, cb + ub * clamp(dot(pa - cb, ub), -hb, hb), rb, margin);
      let pb = cb + ub * select(-hb, hb, e == 0u);
      sphereSphereInto(&M, &sep, ca + ua * clamp(dot(pb - ca, ua), -ha, ha), ra, pb, rb, margin);
    }
  }
  if (M.count > 0u) { reduce4(&M, &sep, M.nrm[0]); }
  return M;
}

// capsule (body a) against box (body b): both segment ends plus the segment point closest to
// the box (alternating projections between two convex sets converge to the closest pair)
fn capsuleBox(Cp: Body, Bx: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let u = capsuleAxis(Cp); let h = Cp.size.y; let r = Cp.size.x; let c = Cp.x0.xyz;
  sphereBoxInto(&M, &sep, c + u * h, r, Bx, margin);
  sphereBoxInto(&M, &sep, c - u * h, r, Bx, margin);
  let R = qmat(Bx.q0);
  var p = c;
  for (var it = 0; it < 6; it++) {
    let q = Bx.x0.xyz + R * clamp(transpose(R) * (p - Bx.x0.xyz), -Bx.size.xyz, Bx.size.xyz);
    p = c + u * clamp(dot(q - c, u), -h, h);
  }
  sphereBoxInto(&M, &sep, p, r, Bx, margin);
  if (M.count > 0u) { reduce4(&M, &sep, M.nrm[0]); }
  return M;
}

fn floorManifold(A: Body, floorY: f32, margin: f32) -> Manifold {
  let s = shapeOf(A);
  if (s == 1u) { return boxFloor(A, floorY, margin); }
  if (s == 2u) { return capsuleFloor(A, floorY, margin); }
  var M: Manifold; M.count = 0u;
  let p = A.x0.xyz;
  if (p.y - A.size.x - floorY < margin) {
    M.count = 1u; M.nrm[0] = vec3f(0.0, 1.0, 0.0);
    M.pa[0] = p - vec3f(0.0, A.size.x, 0.0); M.pb[0] = vec3f(p.x, floorY, p.z);
  }
  return M;
}

/** Contact manifold between bodies a and b (normals from b to a). */
fn pairManifold(A: Body, Bb: Body, margin: f32) -> Manifold {
  let sa = shapeOf(A); let sb = shapeOf(Bb);
  if (sa == 1u && sb == 1u) { return boxBox(A, Bb, margin); }
  if (sa == 0u && sb == 0u) {
    var M: Manifold; M.count = 0u; var sep: array<f32, 8>;
    sphereSphereInto(&M, &sep, A.x0.xyz, A.size.x, Bb.x0.xyz, Bb.size.x, margin);
    return M;
  }
  if (sa == 0u && sb == 1u) { var M: Manifold; M.count = 0u; var sep: array<f32, 8>; sphereBoxInto(&M, &sep, A.x0.xyz, A.size.x, Bb, margin); return M; }
  if (sa == 1u && sb == 0u) { var M: Manifold; M.count = 0u; var sep: array<f32, 8>; sphereBoxInto(&M, &sep, Bb.x0.xyz, Bb.size.x, A, margin); return flipManifold(M); }
  if (sa == 2u && sb == 2u) { return capsuleCapsule(A, Bb, margin); }
  if (sa == 2u && sb == 0u) { return capsuleSphere(A, Bb, margin); }
  if (sa == 0u && sb == 2u) { return flipManifold(capsuleSphere(Bb, A, margin)); }
  if (sa == 2u && sb == 1u) { return capsuleBox(A, Bb, margin); }
  return flipManifold(capsuleBox(Bb, A, margin));   // box a, capsule b
}
`;

// Ray tests against a body at its current pose: (t, normal); t < 0 is a miss.
const RAY_WGSL = /* wgsl */`
fn rayBody(o: vec3f, d: vec3f, b: Body) -> vec4f {
  let s = shapeOf(b);
  let R = qmat(b.q);
  let lo = transpose(R) * (o - b.x.xyz);
  let ld = transpose(R) * d;
  if (s == 0u) {
    let r = b.size.x;
    let bb = dot(lo, ld);
    let h = bb * bb - dot(lo, lo) + r * r;
    if (h < 0.0) { return vec4f(-1.0); }
    let t = -bb - sqrt(h);
    if (t < 0.0) { return vec4f(-1.0); }
    return vec4f(t, R * ((lo + ld * t) / r));
  }
  if (s == 1u) {
    let h = b.size.xyz;
    let inv = 1.0 / select(ld, vec3f(1e-20), abs(ld) < vec3f(1e-20));
    let t1 = (-h - lo) * inv;
    let t2 = (h - lo) * inv;
    let tn = min(t1, t2);
    let tf = max(t1, t2);
    let tnear = max(max(tn.x, tn.y), tn.z);
    let tfar = min(min(tf.x, tf.y), tf.z);
    if (tnear > tfar || tnear < 0.0) { return vec4f(-1.0); }
    var ln = vec3f(0.0);
    if (tnear == tn.x) { ln.x = -sign(ld.x); } else if (tnear == tn.y) { ln.y = -sign(ld.y); } else { ln.z = -sign(ld.z); }
    return vec4f(tnear, R * ln);
  }
  // capsule (Quilez): segment from (0,-h,0) to (0,h,0), radius r
  let r = b.size.x;
  let pa = vec3f(0.0, -b.size.y, 0.0);
  let ba = vec3f(0.0, 2.0 * b.size.y, 0.0);
  let oa = lo - pa;
  let baba = dot(ba, ba); let bard = dot(ba, ld); let baoa = dot(ba, oa);
  let rdoa = dot(ld, oa); let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  var bq = baba * rdoa - baoa * bard;
  var c = baba * oaoa - baoa * baoa - r * r * baba;
  var h = bq * bq - a * c;
  if (h < 0.0) { return vec4f(-1.0); }
  var t = (-bq - sqrt(h)) / a;
  let y = baoa + t * bard;
  if (!(y > 0.0 && y < baba && a > 1e-12)) {
    let oc = select(lo - (pa + ba), oa, y <= 0.0);
    bq = dot(ld, oc);
    c = dot(oc, oc) - r * r;
    h = bq * bq - c;
    if (h <= 0.0) { return vec4f(-1.0); }
    t = -bq - sqrt(h);
  }
  if (t < 0.0) { return vec4f(-1.0); }
  let p = lo + ld * t;
  let hh = clamp(dot(p - pa, ba) / baba, 0.0, 1.0);
  return vec4f(t, R * ((p - pa - hh * ba) / r));
}
`;
return { BODY_VEC4, CONTACT_VEC4, JOINT_VEC4, SHAPES, FLAGS, JOINT_TYPES, JOINT_FLAGS, NARROW_WGSL, RAY_WGSL };
})();

// ---- src/physics/coupling.js
const __m20 = (() => {
const defineModule = __m5.defineModule;
// Two-way coupling between MLS-MPM materials (water, sand, snow, jelly) and a RigidWorld.
//
// Rigid bodies are moving boundaries of the MPM grid. After every MPM grid update, each
// grid node inside a body or within `band` cells of its surface has its velocity relative to the body
// projected (no penetration, Coulomb friction): v_rel ← v_rel − min(0, v_rel·n) n, then
// tangential friction. The momentum that projection adds to the node, m Δv, is taken from
// the body (Newton's third law): J = −Σ m Δv, L = −Σ r × m Δv. Over a frame these sums
// become the body's external impulse, applied at the start of the next rigid step.
// Buoyancy, drag and splashes are not modelled separately — they are what this exchange of
// momentum produces: a board lighter than water floats at Archimedes' waterline to about one grid
// cell (a cube of middling density tips onto an edge, as a real one does).
//
// Per workgroup (a 4×4×4 tile of nodes) the bodies whose bounding spheres touch the tile are
// culled into a candidate list in index order (a workgroup prefix scan, so the list does not
// depend on thread timing), node impulses are summed in fixed point in shared memory, and each
// tile adds its sums to the per-body accumulators with integer atomics: the result is the same
// whatever order the GPU runs things in.
//
// Units: MPM works in lattice units (cells, substeps, density × cells); the rigid world in
// metres, seconds and density × m³ (density 1 = water in both). A lattice momentum p' is
// p' dx⁴/Δt_s in world units.



const MAX_CAND = 128;
const FPJ = 16384;   // fixed-point scale of lattice momentum
const FPL = 1024;    // fixed-point scale of lattice angular momentum

defineModule('coupling', {
  deps: ['rigid'],
  code: /* wgsl */`
/** Signed distance from p to body b centred at c (world units) and the outward normal. */
fn bodySDF(b: Body, c: vec3f, p: vec3f) -> vec4f {
  let s = shapeOf(b);
  let R = qmat(b.q);
  let lp = transpose(R) * (p - c);
  if (s == 0u) {
    let d = length(lp);
    return vec4f(d - b.size.x, R * (lp / max(d, 1e-6)));
  }
  if (s == 2u) {
    let h = b.size.y;
    let q = vec3f(lp.x, lp.y - clamp(lp.y, -h, h), lp.z);
    let d = length(q);
    return vec4f(d - b.size.x, R * select(vec3f(0.0, sign(lp.y), 0.0), q / max(d, 1e-6), d > 1e-6));
  }
  let h = b.size.xyz;
  let q = abs(lp) - h;
  if (any(q > vec3f(0.0))) {
    let dv = lp - clamp(lp, -h, h);
    let d = length(dv);
    return vec4f(d, R * (dv / max(d, 1e-9)));
  }
  var ax = 0u;
  if (q.y > q[ax]) { ax = 1u; }
  if (q.z > q[ax]) { ax = 2u; }
  var n = vec3f(0.0);
  n[ax] = select(-1.0, 1.0, lp[ax] >= 0.0);
  return vec4f(q[ax], R * n);
}
`,
});

class FluidRigidCoupling {
  /**
   * @param sim   MPM solver
   * @param world RigidWorld
   * o: friction (fluid–body, default 0.2), band (cells, default 1.5), mask (collision group bits of
   *    bodies that interact, default all), densityScale (rigid mass per MPM density unit per m³, default 1)
   */
  constructor(ctx, sim, world, o = {}) {
    this.ctx = ctx;
    this.sim = sim;
    this.world = world;
    this.friction = o.friction ?? 0.2;
    this.band = o.band ?? 0.5;
    this.mask = o.mask ?? 0xffffffff;
    this.densityScale = o.densityScale ?? 1;
    this.enabled = true;
    this.acc = ctx.array('vec4i', world.maxBodies * 2, { label: 'coupling-impulses' });
    const dims = sim.dims;
    this.coupleK = ctx.kernel({
      name: 'mpm-rigid-couple', imports: ['coupling'], dims: 3, workgroupSize: [4, 4, 4], guard: false,
      defines: { DIMS: `vec3u(${dims.join('u, ')}u)` },
      buffers: { grid: { type: 'vec4i' }, B: { type: 'Body', access: 'read' }, gacc: { type: 'i32', atomic: true } },
      uniforms: { nBodies: 'u32', mask: 'u32', origin: 'vec3f', dx: 'f32', invDx: 'f32', lead: 'f32', velToLattice: 'f32',
        friction: 'f32', band: 'f32' },
      shared: /* wgsl */`
        const MAXC = ${MAX_CAND}u;
        var<workgroup> cand: array<u32, ${MAX_CAND}>;
        var<workgroup> wscan: array<u32, 64>;
        var<workgroup> wcount: u32;
        var<workgroup> accJ: array<atomic<i32>, ${MAX_CAND * 3}>;
        var<workgroup> accL: array<atomic<i32>, ${MAX_CAND * 3}>;`,
      code: /* wgsl */`
        let tileLo = vec3f(wid * 4u) - vec3f(u.band + 1.0);
        let tileHi = vec3f(wid * 4u) + vec3f(3.0 + u.band + 1.0);
        if (lidx == 0u) { wcount = 0u; }
        workgroupBarrier();
        // cull bodies against this tile, in index order (workgroup prefix scan per round of 64)
        let rounds = (u.nBodies + 63u) / 64u;
        for (var r = 0u; r < rounds; r++) {
          let bi = r * 64u + lidx;
          var hit = 0u;
          if (bi < u.nBodies) {
            let b = B[bi];
            if ((b.info.z & (F_DISABLED | F_SENSOR)) == 0u && (b.info.x & u.mask) != 0u) {
              let c = (b.x.xyz + b.v.xyz * u.lead - u.origin) * u.invDx;
              let q = clamp(c, tileLo, tileHi);
              if (length(c - q) <= b.x.w * u.invDx) { hit = 1u; }
            }
          }
          wscan[lidx] = hit;
          workgroupBarrier();
          for (var off = 1u; off < 64u; off <<= 1u) {
            let add = select(0u, wscan[max(lidx, off) - off], lidx >= off);
            workgroupBarrier();
            wscan[lidx] += add;
            workgroupBarrier();
          }
          let incl = wscan[lidx];
          let total = wscan[63];
          let base = workgroupUniformLoad(&wcount);
          if (hit == 1u && base + incl - 1u < MAXC) { cand[base + incl - 1u] = bi; }
          workgroupBarrier();
          if (lidx == 0u) { wcount = base + total; }
          workgroupBarrier();
        }
        let nc = min(workgroupUniformLoad(&wcount), MAXC);
        for (var k = lidx; k < nc * 3u; k += 64u) { atomicStore(&accJ[k], 0); atomicStore(&accL[k], 0); }
        workgroupBarrier();
        let g = wid * 4u + lid;
        if (nc > 0u && all(g < DIMS)) {
          let gi = g.x + DIMS.x * (g.y + DIMS.y * g.z);
          let cell = bitcast<vec4f>(grid[gi]);
          let m = cell.w;
          if (m > 0.0) {
            var v = cell.xyz;
            let pw = u.origin + vec3f(g) * u.dx;
            for (var c = 0u; c < nc; c++) {
              let b = B[cand[c]];
              let bc = b.x.xyz + b.v.xyz * u.lead;
              let sd = bodySDF(b, bc, pw);
              if (sd.x * u.invDx < u.band) {
                let r = pw - bc;
                let vb = (b.v.xyz + cross(b.w.xyz, r)) * u.velToLattice;   // body velocity at the node (lattice)
                var vr = v - vb;
                let vn = dot(vr, sd.yzw);
                if (vn < 0.0) {
                  vr -= vn * sd.yzw;
                  let vt = length(vr);
                  if (vt > 1e-9) { vr *= max(0.0, 1.0 - u.friction * (-vn) / vt); }
                  let vnew = vr + vb;
                  if (b.mass.x > 0.0) {
                    let J = -m * (vnew - v);          // reaction on the body (lattice momentum)
                    let L = cross(r * u.invDx, J);
                    atomicAdd(&accJ[c * 3u], i32(round(J.x * ${FPJ}.0)));
                    atomicAdd(&accJ[c * 3u + 1u], i32(round(J.y * ${FPJ}.0)));
                    atomicAdd(&accJ[c * 3u + 2u], i32(round(J.z * ${FPJ}.0)));
                    atomicAdd(&accL[c * 3u], i32(round(L.x * ${FPL}.0)));
                    atomicAdd(&accL[c * 3u + 1u], i32(round(L.y * ${FPL}.0)));
                    atomicAdd(&accL[c * 3u + 2u], i32(round(L.z * ${FPL}.0)));
                  }
                  v = vnew;
                }
              }
            }
            grid[gi] = bitcast<vec4i>(vec4f(v, m));
          }
        }
        workgroupBarrier();
        for (var k = lidx; k < nc * 3u; k += 64u) {
          let c = k / 3u; let a = k % 3u; let bi = cand[c];
          let j = atomicLoad(&accJ[k]); let l = atomicLoad(&accL[k]);
          if (j != 0) { atomicAdd(&gacc[bi * 8u + a], j); }
          if (l != 0) { atomicAdd(&gacc[bi * 8u + 4u + a], l); }
        }`,
    });
    this.applyK = ctx.kernel({
      name: 'mpm-rigid-apply', imports: ['rigid'],
      buffers: { acc: { type: 'vec4i' }, ext: { type: 'vec4f' } },
      uniforms: { momToWorld: 'f32', dx: 'f32' },
      code: /* wgsl */`
        let a = acc[2u * i]; let b = acc[2u * i + 1u];
        if (all(a.xyz == vec3i(0)) && all(b.xyz == vec3i(0))) { return; }
        let J = vec3f(a.xyz) / ${FPJ}.0 * u.momToWorld;
        let L = vec3f(b.xyz) / ${FPL}.0 * u.momToWorld * u.dx;
        ext[2u * i] += vec4f(J, 0.0);
        ext[2u * i + 1u] += vec4f(L, 0.0);
        acc[2u * i] = vec4i(0); acc[2u * i + 1u] = vec4i(0);`,
    });
    this.tiles = dims.map((d) => Math.ceil(d / 4));
  }

  /** Called by MPM after the grid update of substep s. */
  substep(s) {
    const w = this.world, sim = this.sim;
    if (!this.enabled || !w.count) return;
    const dts = sim.substepDt;
    this.coupleK.dispatch(this.tiles, {
      nBodies: w.count, mask: this.mask >>> 0, origin: sim.origin, dx: sim.dx, invDx: 1 / sim.dx,
      lead: (s + 0.5) * dts, velToLattice: dts / sim.dx, friction: this.friction, band: this.band,
    }, { grid: sim.grid, B: w.B, gacc: this.acc });
  }

  /** Called by MPM after its frame: hand the accumulated impulses to the rigid world. */
  finish() {
    const w = this.world, sim = this.sim;
    if (!this.enabled || !w.count) return;
    const dx = sim.dx;
    this.applyK.run(w.count, { momToWorld: this.densityScale * dx ** 4 / sim.substepDt, dx }, { acc: this.acc, ext: w.ext });
  }
}
return { FluidRigidCoupling };
})();

// ---- src/physics/mpm.js
const __m18 = (() => {
const defineModule = __m5.defineModule;
const Scan = __m7.Scan;
const Spheres = __m15.Spheres;
const FluidSurface = __m19.FluidSurface;
const FluidRigidCoupling = __m20.FluidRigidCoupling;
// MLS-MPM (Hu et al. 2018) with APIC transfers, multi-material, in lattice units.
//
// First principles that shape this solver:
//  1. Lattice units (dx = 1 cell, dt = 1 substep). Every quantity a stable
//     simulation produces is O(1): velocities < 1 cell/step (CFL), wave speeds
//     c' = c·dt/dx < ~0.5. That bounds momenta, so 32-bit fixed point (2^22)
//     is exact enough for atomic P2G — WebGPU has no float atomics.
//  2. Stiffness is chosen in lattice units (fraction of the stable wave speed),
//     so a material can never be configured to explode; more substeps simply
//     buy stiffer (more incompressible) materials.
//  3. P2G is the bottleneck (27 nodes x 4 channels of atomics per particle).
//     Particles are counting-sorted into 4^3-cell blocks (Morton order) every
//     few substeps; each workgroup then accumulates its particles into a
//     shared-memory tile with workgroup atomics and flushes only non-empty
//     nodes to global memory — ~10-30x fewer global atomics.
//  4. Stress is evaluated once per substep at the end of G2P and stored as
//     the Kirchhoff stress τ = P Fᵀ in the spare .w lanes of the particle, so
//     P2G is material-agnostic and branch-free.







const FLUID = 0, JELLY = 1, SNOW = 2, SAND = 3;

/** Material presets. stiffness ∈ (0,1] is a fraction of the max stable lattice wave speed. */
const MATERIALS = {
  water:   { kind: FLUID, density: 1, stiffness: 1.0, viscosity: 0.0, color: [0.12, 0.42, 0.95] },
  oil:     { kind: FLUID, density: 0.9, stiffness: 0.9, viscosity: 0.3, color: [0.9, 0.62, 0.12] },
  honey:   { kind: FLUID, density: 1.4, stiffness: 0.9, viscosity: 1.0, color: [0.95, 0.6, 0.05] },
  jelly:   { kind: JELLY, density: 1, stiffness: 0.35, poisson: 0.3, color: [0.95, 0.3, 0.35] },
  rubber:  { kind: JELLY, density: 1.2, stiffness: 0.7, poisson: 0.45, color: [0.25, 0.25, 0.28] },
  snow:    { kind: SNOW, density: 0.4, stiffness: 0.6, poisson: 0.2, hardening: 10, compression: 0.025, stretch: 0.0075, color: [0.95, 0.96, 1.0] },
  sand:    { kind: SAND, density: 1.6, stiffness: 0.6, poisson: 0.3, friction: 33, color: [0.86, 0.68, 0.42] },
};

const C_MAX = 0.4;          // max lattice wave speed (cells/step)
const PARTICLE_FLOATS = 32; // 8 x vec4f

defineModule('mpm', {
  deps: ['linalg'],
  code: /* wgsl */`
struct Particle {
  x: vec4f,   // position (lattice units), w = material index (u32 bits)
  v: vec4f,   // velocity (cells / substep), w = J (fluids) or Jp (snow)
  c0: vec4f,  // APIC C columns; w = tau00, tau11, tau22
  c1: vec4f,
  c2: vec4f,
  f0: vec4f,  // deformation gradient F columns; w = tau01, tau02, tau12
  f1: vec4f,
  f2: vec4f,
}
const FP = 4194304.0;            // 2^22 fixed-point scale for atomics
const INV_FP = 1.0 / 4194304.0;
fn toFixed(x: f32) -> i32 { return i32(round(clamp(x * FP, -2.0e9, 2.0e9))); }
fn bspline(fx: vec3f) -> array<vec3f, 3> {
  return array<vec3f, 3>(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}
fn si_part1by2(x: u32) -> u32 {
  var v = x & 0x3ffu;
  v = (v | (v << 16u)) & 0x030000ffu;
  v = (v | (v << 8u)) & 0x0300f00fu;
  v = (v | (v << 4u)) & 0x030c30c3u;
  v = (v | (v << 2u)) & 0x09249249u;
  return v;
}
fn morton3(c: vec3u) -> u32 { return si_part1by2(c.x) | (si_part1by2(c.y) << 1u) | (si_part1by2(c.z) << 2u); }
`,
});

class MPM {
  /**
   * @param {Context} ctx
   * @param {object} o
   *   maxParticles, grid: cells along the longest axis (default 64), size: [x,y,z] world size (default [1,1,1]),
   *   origin, gravity (m/s^2), substeps (per 1/60 s), timeScale, sortEvery, ppc (particles per cell axis^3, default 8)
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.maxParticles = o.maxParticles || 262144;
    const size = Array.isArray(o.size) ? o.size : [o.size || 1, o.size || 1, o.size || 1];
    const res = o.grid || 64;
    this.dx = Math.max(...size) / res;
    this.dims = size.map((s) => Math.max(8, Math.round(s / this.dx)));
    this.size = this.dims.map((d) => d * this.dx);
    this.origin = o.origin || [0, 0, 0];
    this.gravity = o.gravity || [0, -9.81, 0];
    this.substeps = o.substeps || 8;
    this.timeScale = o.timeScale ?? 1;
    this.sortEvery = o.sortEvery || 4;
    this.ppcAxis = o.ppcAxis || 2;
    this.bound = 3;
    this.count = 0;
    this.stepCount = 0;
    this.materials = [];
    this.materialIndex = new Map();
    this.colliders = [];
    this.dt = 1 / 60 * this.timeScale / this.substeps;

    const nodes = this.dims[0] * this.dims[1] * this.dims[2];
    this.numNodes = nodes;
    this.P = ctx.array('vec4f', this.maxParticles * 8, { label: 'mpm-particles', usage: GPUBufferUsage.VERTEX });
    this.P2 = ctx.array('vec4f', this.maxParticles * 8, { label: 'mpm-particles-2', usage: GPUBufferUsage.VERTEX });
    this.grid = ctx.array('vec4i', nodes, { label: 'mpm-grid' });
    this.matBuf = ctx.array('vec4f', 8 * 3, { label: 'mpm-materials' });
    this.colBuf = ctx.array('vec4f', 16 * 3, { label: 'mpm-colliders' });
    // block sort (Morton order of 4^3-cell blocks)
    const bmax = Math.max(...this.dims.map((d) => Math.ceil(d / 4)));
    const bpow = 1 << Math.ceil(Math.log2(bmax));
    this.numBlocks = bpow ** 3;
    this.blockCount = ctx.array('u32', this.numBlocks + 1, { label: 'mpm-block-count' });
    this.key = ctx.array('u32', this.maxParticles, { label: 'mpm-key' });
    this.rank = ctx.array('u32', this.maxParticles, { label: 'mpm-rank' });
    this.scan = new Scan(ctx, this.numBlocks + 1);
    this._buildKernels();
  }

  /** World-space time step of one substep. */
  get substepDt() { return (1 / 60) * this.timeScale / this.substeps; }

  // ------------------------------------------------------------------ materials
  _material(nameOrDef) {
    const def = typeof nameOrDef === 'string' ? { ...MATERIALS[nameOrDef], name: nameOrDef } : { ...nameOrDef };
    if (!def || def.kind === undefined) throw new Error(`MPM: unknown material ${nameOrDef}`);
    const key = JSON.stringify(def);
    if (this.materialIndex.has(key)) return this.materialIndex.get(key);
    if (this.materials.length >= 8) throw new Error('MPM: at most 8 distinct materials');
    const idx = this.materials.length;
    this.materials.push(def);
    this.materialIndex.set(key, idx);
    this._uploadMaterials();
    return idx;
  }

  _uploadMaterials() {
    const vol = 1 / this.ppcAxis ** 3;
    const data = new Float32Array(8 * 12);
    const u = new Uint32Array(data.buffer);
    this.materials.forEach((m, k) => {
      const mass = (m.density ?? 1) * vol;          // lattice mass: rho' = density (relative) per cell
      const rho = mass / vol;
      const c = C_MAX * Math.min(1, Math.max(0.01, m.stiffness ?? 1));
      const nu = m.poisson ?? 0.3;
      // P-wave modulus M = lambda + 2 mu = rho c^2; recover E, mu, lambda from Poisson's ratio.
      const Mmod = rho * c * c;
      const E = Mmod * (1 + nu) * (1 - 2 * nu) / (1 - nu);
      const mu = E / (2 * (1 + nu));
      const la = E * nu / ((1 + nu) * (1 - 2 * nu));
      const o = k * 12;
      u[o] = m.kind;
      data[o + 1] = mass;
      data[o + 2] = m.kind === FLUID ? rho * c * c : mu;   // fluids: bulk modulus K
      data[o + 3] = la;
      data[o + 4] = (m.viscosity ?? 0) * 0.05 * rho;       // lattice dynamic viscosity
      data[o + 5] = m.hardening ?? 10;
      data[o + 6] = m.compression ?? 0.025;
      data[o + 7] = m.stretch ?? 0.0075;
      const phi = ((m.friction ?? 33) * Math.PI) / 180;
      const sinp = Math.sin(phi);
      data[o + 8] = Math.sqrt(2 / 3) * (2 * sinp) / (3 - sinp); // Drucker–Prager alpha
    });
    this.matBuf.write(data);
  }

  // ------------------------------------------------------------------ emission
  /**
   * Add particles filling a shape with `ppcAxis^3` particles per cell (jittered).
   * shape: { box: [[x0,y0,z0],[x1,y1,z1]] } | { sphere: [[cx,cy,cz], r] } | { points: Float32Array (xyz world) }
   * opts: { material: 'water' | def, velocity: [vx,vy,vz] (m/s), jitter: 0..1 }
   */
  add(shape, opts = {}) {
    const mat = this._material(opts.material || shape.material || 'water');
    const vel = opts.velocity || shape.velocity || [0, 0, 0];
    const pts = shape.points || sampleShape(shape, this.dx / this.ppcAxis, opts.jitter ?? 0.6, opts.seed ?? this.count + 1);
    const n = Math.min(pts.length / 3, this.maxParticles - this.count);
    if (n <= 0) return 0;
    const data = new Float32Array(n * PARTICLE_FLOATS);
    const u = new Uint32Array(data.buffer);
    const s = this.substepDt / this.dx; // m/s -> cells/step
    const lo = this.bound, hi = this.dims.map((d) => d - this.bound);
    for (let k = 0; k < n; k++) {
      const o = k * PARTICLE_FLOATS;
      for (let a = 0; a < 3; a++) {
        const x = (pts[3 * k + a] - this.origin[a]) / this.dx;
        data[o + a] = Math.min(hi[a] - 1e-3, Math.max(lo + 1e-3, x));
      }
      u[o + 3] = mat;
      data[o + 4] = vel[0] * s; data[o + 5] = vel[1] * s; data[o + 6] = vel[2] * s;
      data[o + 7] = 1; // J
      data[o + 20] = 1; data[o + 25] = 1; data[o + 30] = 1; // F = I
    }
    this.ctx.flush();
    this.P.write(data, this.count * 8);
    this.count += n;
    return n;
  }

  /** Remove every particle. */
  clear() { this.count = 0; }

  /**
   * Two-way coupling with a RigidWorld: bodies become moving boundaries of this material and
   * receive the momentum it exchanges with them (buoyancy, drag, splashes). Step the MPM first,
   * then the rigid world, each frame. o: { friction, band, mask, densityScale } (see coupling.js).
   */
  couple(world, o = {}) {
    this.coupling = new FluidRigidCoupling(this.ctx, this, world, o);
    return this.coupling;
  }

  // ------------------------------------------------------------------ colliders
  /**
   * Kinematic colliders, re-uploaded each step. Each: { type: 'sphere'|'box'|'plane',
   * center (world), radius | half: [hx,hy,hz] | normal, velocity (m/s), friction }.
   */
  setColliders(list) { this.colliders = list; }

  _uploadColliders() {
    const data = new Float32Array(16 * 12);
    const u = new Uint32Array(data.buffer);
    const s = this.substepDt / this.dx;
    this.colliders.slice(0, 16).forEach((c, k) => {
      const o = k * 12;
      u[o] = { sphere: 0, box: 1, plane: 2 }[c.type || 'sphere'];
      const ctr = c.center || [0, 0, 0];
      for (let a = 0; a < 3; a++) data[o + 1 + a] = (ctr[a] - this.origin[a]) / this.dx;
      if (c.type === 'plane') { const nrm = c.normal || [0, 1, 0]; data[o + 4] = nrm[0]; data[o + 5] = nrm[1]; data[o + 6] = nrm[2]; }
      else if (c.type === 'box') { const h = c.half || [0.1, 0.1, 0.1]; data[o + 4] = h[0] / this.dx; data[o + 5] = h[1] / this.dx; data[o + 6] = h[2] / this.dx; }
      else data[o + 4] = (c.radius ?? 0.1) / this.dx;
      data[o + 7] = c.friction ?? 0.3;
      const v = c.velocity || [0, 0, 0];
      data[o + 8] = v[0] * s; data[o + 9] = v[1] * s; data[o + 10] = v[2] * s;
    });
    this.colBuf.write(data);
  }

  // ------------------------------------------------------------------ kernels
  _buildKernels() {
    const ctx = this.ctx;
    const sg = ctx.caps.subgroups;
    const WG = 256, PPT = 2;
    this.p2gPerGroup = WG * PPT;
    const shmem = ctx.caps.workgroupStorage;
    this.tileCap = Math.max(343, Math.floor((shmem - 512) / 16));
    const common = { imports: ['mpm'], defines: { DIMS: `vec3u(${this.dims.join('u, ')}u)` } };

    // --- P2G: shared-memory tiled scatter with workgroup atomics
    this.p2gK = ctx.kernel({
      ...common, name: 'mpm-p2g', workgroupSize: WG, guard: false, subgroups: sg,
      buffers: { P: { type: 'Particle', access: 'read' }, grid: { type: 'i32', atomic: true }, MATS: { type: 'vec4f', access: 'read' } },
      uniforms: { vol4: 'f32', forceGlobal: 'u32' },
      shared: /* wgsl */`
        const TILE_CAP = ${this.tileCap}u;
        const PPT = ${PPT}u;
        var<workgroup> tile: array<atomic<i32>, ${this.tileCap * 4}>;
        var<workgroup> bmin: array<atomic<i32>, 3>;
        var<workgroup> bmax: array<atomic<i32>, 3>;
        var<workgroup> wbox: array<i32, 6>;
        fn nodeIndex(g: vec3i) -> u32 { return u32(g.x) + DIMS.x * (u32(g.y) + DIMS.y * u32(g.z)); }
        fn scatter(p: Particle, useTile: bool, o: vec3i, ext: vec3i) {
          let xp = p.x.xyz;
          let base = vec3i(floor(xp - 0.5));
          let fx = xp - vec3f(base);
          var w = bspline(fx);
          let mass = MATS[bitcast<u32>(p.x.w) * 3u].y;
          let tau = mat3x3f(p.c0.w, p.f0.w, p.f1.w,  p.f0.w, p.c1.w, p.f2.w,  p.f1.w, p.f2.w, p.c2.w);
          let C = mat3x3f(p.c0.xyz, p.c1.xyz, p.c2.xyz);
          let affine = -u.vol4 * tau + mass * C;
          let mv = mass * p.v.xyz;
          for (var a = 0; a < 3; a++) {
            for (var b = 0; b < 3; b++) {
              for (var c = 0; c < 3; c++) {
                let off = vec3i(a, b, c);
                let dpos = vec3f(off) - fx;
                let wt = w[a].x * w[b].y * w[c].z;
                let mom = wt * (mv + affine * dpos);
                let q = vec4i(toFixed(mom.x), toFixed(mom.y), toFixed(mom.z), toFixed(wt * mass));
                let g = base + off;
                if (useTile) {
                  let l = g - o;
                  let ti = u32((l.z * ext.y + l.y) * ext.x + l.x) * 4u;
                  atomicAdd(&tile[ti], q.x); atomicAdd(&tile[ti + 1u], q.y);
                  atomicAdd(&tile[ti + 2u], q.z); atomicAdd(&tile[ti + 3u], q.w);
                } else {
                  let gi = nodeIndex(g) * 4u;
                  atomicAdd(&grid[gi], q.x); atomicAdd(&grid[gi + 1u], q.y);
                  atomicAdd(&grid[gi + 2u], q.z); atomicAdd(&grid[gi + 3u], q.w);
                }
              }
            }
          }
        }`,
      code: /* wgsl */`
        let first = wgi * WG * PPT;
        if (lidx < 3u) { atomicStore(&bmin[lidx], 0x3fffffff); atomicStore(&bmax[lidx], -0x3fffffff); }
        workgroupBarrier();
        var lo = vec3i(0x3fffffff);
        var hi = vec3i(-0x3fffffff);
        for (var k = 0u; k < PPT; k++) {
          let pi = first + k * WG + lidx;
          if (pi < u.n) {
            let b = vec3i(floor(P[pi].x.xyz - 0.5));
            lo = min(lo, b); hi = max(hi, b);
          }
        }
        ${sg ? /* wgsl */`
        lo = subgroupMin(lo); hi = subgroupMax(hi);
        if (sgi == 0u) {` : '{'}
          atomicMin(&bmin[0], lo.x); atomicMin(&bmin[1], lo.y); atomicMin(&bmin[2], lo.z);
          atomicMax(&bmax[0], hi.x); atomicMax(&bmax[1], hi.y); atomicMax(&bmax[2], hi.z);
        }
        workgroupBarrier();
        if (lidx == 0u) {
          wbox[0] = atomicLoad(&bmin[0]); wbox[1] = atomicLoad(&bmin[1]); wbox[2] = atomicLoad(&bmin[2]);
          wbox[3] = atomicLoad(&bmax[0]); wbox[4] = atomicLoad(&bmax[1]); wbox[5] = atomicLoad(&bmax[2]);
        }
        let box = workgroupUniformLoad(&wbox);
        let o = vec3i(box[0], box[1], box[2]);
        let ext = vec3i(box[3], box[4], box[5]) - o + 3;
        if (any(ext <= vec3i(0))) { return; }
        let tileNodes = ext.x * ext.y * ext.z;
        let useTile = tileNodes <= i32(TILE_CAP) && u.forceGlobal == 0u;
        for (var k = 0u; k < PPT; k++) {
          let pi = first + k * WG + lidx;
          if (pi < u.n) { scatter(P[pi], useTile, o, ext); }
        }
        if (useTile) {
          workgroupBarrier();
          for (var t = lidx; t < u32(tileNodes); t += WG) {
            let m = atomicLoad(&tile[t * 4u + 3u]);
            let mx = atomicLoad(&tile[t * 4u]);
            let my = atomicLoad(&tile[t * 4u + 1u]);
            let mz = atomicLoad(&tile[t * 4u + 2u]);
            if ((m | mx | my | mz) != 0) {
              let it = i32(t);
              let l = vec3i(it % ext.x, (it / ext.x) % ext.y, it / (ext.x * ext.y));
              let gi = nodeIndex(o + l) * 4u;
              atomicAdd(&grid[gi], mx); atomicAdd(&grid[gi + 1u], my);
              atomicAdd(&grid[gi + 2u], mz); atomicAdd(&grid[gi + 3u], m);
            }
          }
        }`,
    });

    // --- grid update: momentum -> velocity, gravity, walls, colliders
    this.gridK = ctx.kernel({
      ...common, name: 'mpm-grid',
      buffers: { grid: { type: 'vec4i' }, COL: { type: 'vec4f', access: 'read' } },
      uniforms: { gravity: 'vec3f', bound: 'f32', numColliders: 'u32', floorFriction: 'f32' },
      shared: /* wgsl */`
        fn colliderSDF(kind: u32, d: vec3f, s: vec4f) -> f32 {
          if (kind == 0u) { return length(d) - s.x; }
          if (kind == 1u) {
            let q = abs(d) - s.xyz;
            return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
          }
          return dot(d, s.xyz);
        }`,
      code: /* wgsl */`
        let g = grid[i];
        var out = vec4f(0.0);
        if (g.w > 0) {
          let m = f32(g.w) * INV_FP;
          var v = vec3f(g.xyz) * INV_FP / m + u.gravity;
          let c = vec3f(f32(i % DIMS.x), f32((i / DIMS.x) % DIMS.y), f32(i / (DIMS.x * DIMS.y)));
          let hi = vec3f(DIMS) - u.bound;
          // separating slip walls; the floor applies Coulomb friction
          if (c.y < u.bound && v.y < 0.0) {
            let vn = -v.y;
            v.y = 0.0;
            let vt = length(v.xz);
            if (vt > 1e-9) { let f = max(0.0, 1.0 - u.floorFriction * vn / vt); v = vec3f(v.x * f, 0.0, v.z * f); }
          }
          if (c.x < u.bound && v.x < 0.0) { v.x = 0.0; }
          if (c.z < u.bound && v.z < 0.0) { v.z = 0.0; }
          if (c.x > hi.x && v.x > 0.0) { v.x = 0.0; }
          if (c.y > hi.y && v.y > 0.0) { v.y = 0.0; }
          if (c.z > hi.z && v.z > 0.0) { v.z = 0.0; }
          for (var k = 0u; k < u.numColliders; k++) {
            let h = COL[k * 3u]; let s = COL[k * 3u + 1u]; let cv = COL[k * 3u + 2u].xyz;
            let kind = bitcast<u32>(h.x);
            let d = c - h.yzw;
            let dist = colliderSDF(kind, d, s);
            if (dist < 1.0) {
              let e = 0.5;
              let n = normalize(vec3f(
                colliderSDF(kind, d + vec3f(e, 0.0, 0.0), s) - colliderSDF(kind, d - vec3f(e, 0.0, 0.0), s),
                colliderSDF(kind, d + vec3f(0.0, e, 0.0), s) - colliderSDF(kind, d - vec3f(0.0, e, 0.0), s),
                colliderSDF(kind, d + vec3f(0.0, 0.0, e), s) - colliderSDF(kind, d - vec3f(0.0, 0.0, e), s)) + vec3f(0.0, 1e-7, 0.0));
              var vr = v - cv;
              let vn = dot(vr, n);
              if (vn < 0.0) {
                vr -= vn * n;
                let vt = length(vr);
                if (vt > 1e-9) { vr *= max(0.0, 1.0 - s.w * -vn / vt); }
                v = vr + cv;
              }
            }
          }
          out = vec4f(v, m);
        }
        grid[i] = bitcast<vec4i>(out);`,
    });

    // --- G2P: gather velocity + APIC C, advect, update F, plasticity, stress
    this.g2pK = ctx.kernel({
      ...common, name: 'mpm-g2p',
      buffers: { P: { type: 'Particle' }, GV: { type: 'vec4f', access: 'read' }, MATS: { type: 'vec4f', access: 'read' } },
      uniforms: { bound: 'f32', vmax: 'f32' },
      code: /* wgsl */`
        var p = P[i];
        let xp = p.x.xyz;
        let base = vec3i(floor(xp - 0.5));
        let fx = xp - vec3f(base);
        var w = bspline(fx);
        var v = vec3f(0.0);
        var B = mat3x3f();
        for (var a = 0; a < 3; a++) {
          for (var b = 0; b < 3; b++) {
            for (var c = 0; c < 3; c++) {
              let off = vec3i(a, b, c);
              let dpos = vec3f(off) - fx;
              let wt = w[a].x * w[b].y * w[c].z;
              let g = base + off;
              let gv = GV[u32(g.x) + DIMS.x * (u32(g.y) + DIMS.y * u32(g.z))].xyz;
              v += wt * gv;
              B += outer3(wt * gv, dpos);
            }
          }
        }
        let C = 4.0 * B;
        let sp = length(v);
        if (sp > u.vmax) { v *= u.vmax / sp; }
        let x = clamp(xp + v, vec3f(u.bound - 1.0), vec3f(DIMS) - u.bound + 1.0);
        let mid = bitcast<u32>(p.x.w);
        let m0 = MATS[mid * 3u];
        let m1 = MATS[mid * 3u + 1u];
        let m2 = MATS[mid * 3u + 2u];
        let kind = bitcast<u32>(m0.x);
        var F = mat3x3f(p.f0.xyz, p.f1.xyz, p.f2.xyz);
        var tau = mat3x3f();
        var vw = p.v.w;
        if (kind == 0u) {
          // weakly compressible fluid tracked by J only. Linear EOS tau = K (J - 1) I keeps the
          // wave speed constant (stable at any compression); tension is capped to limit clumping.
          let J = clamp(vw * (1.0 + trace3(C)), 0.6, 1.4);
          vw = J;
          tau = (m0.z * min(J - 1.0, 0.02)) * I3 + (m1.x * J) * (C + transpose(C));
          F = I3;
        } else {
          F = (I3 + C) * F;
          let mu0 = m0.z; let la0 = m0.w;
          if (kind == 1u) {
            // Neo-Hookean: tau = mu (F Fᵀ - I) + lambda ln(J) I  (no polar decomposition needed)
            let J = max(determinant(F), 0.05);
            tau = mu0 * (F * transpose(F) - I3) + (la0 * log(J)) * I3;
          } else if (kind == 2u) {
            // snow (Stomakhin 2013): clamp singular values, harden with plastic compression
            let d = svd3(F);
            let s = clamp(d.s, vec3f(1.0 - m1.z), vec3f(1.0 + m1.w));
            let Jold = d.s.x * d.s.y * d.s.z;
            let Jnew = s.x * s.y * s.z;
            vw = clamp(vw * Jold / Jnew, 0.6, 20.0);
            F = d.U * diag3(s) * transpose(d.V);
            let h = exp(m1.y * (1.0 - vw));
            let mu = mu0 * h; let la = la0 * h;
            let R = d.U * transpose(d.V);
            let J = Jnew;
            tau = 2.0 * mu * (F - R) * transpose(F) + (la * J * (J - 1.0)) * I3;
          } else {
            // sand (Klar 2016): Drucker–Prager return mapping in Hencky strain
            let d = svd3(F);
            var e = log(max(abs(d.s), vec3f(1e-4)));
            let tr = e.x + e.y + e.z;
            let dev = e - tr / 3.0;
            let devn = length(dev);
            if (tr >= 0.0) {
              e = vec3f(0.0);
            } else {
              let dg = devn + (3.0 * la0 + 2.0 * mu0) / (2.0 * mu0) * tr * m2.x;
              if (dg > 0.0 && devn > 1e-12) { e = e - dg * dev / devn; }
            }
            let s = exp(e);
            F = d.U * diag3(s) * transpose(d.V);
            let te = e.x + e.y + e.z;
            tau = d.U * diag3(2.0 * mu0 * e + la0 * te) * transpose(d.U);
          }
        }
        P[i] = Particle(vec4f(x, p.x.w), vec4f(v, vw),
          vec4f(C[0], tau[0].x), vec4f(C[1], tau[1].y), vec4f(C[2], tau[2].z),
          vec4f(F[0], tau[1].x), vec4f(F[1], tau[2].x), vec4f(F[2], tau[2].y));`,
    });

    // --- block sort: key = Morton(block), counting sort fused with the particle scatter
    this.keyK = ctx.kernel({
      ...common, name: 'mpm-key',
      buffers: { P: { type: 'Particle', access: 'read' }, key: { type: 'u32' }, rank: { type: 'u32' }, cnt: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let b = vec3u(max(P[i].x.xyz, vec3f(0.0))) / 4u;
        let k = morton3(b);
        key[i] = k;
        rank[i] = atomicAdd(&cnt[k], 1u);`,
    });
    this.reorderK = ctx.kernel({
      ...common, name: 'mpm-reorder',
      buffers: { src: { type: 'Particle', access: 'read' }, dst: { type: 'Particle' }, key: { type: 'u32', access: 'read' },
        rank: { type: 'u32', access: 'read' }, start: { type: 'u32', access: 'read' } },
      code: 'dst[start[key[i]] + rank[i]] = src[i];',
    });
  }

  // ------------------------------------------------------------------ stepping
  /** Advance by one frame (1/60 s x timeScale), split into `substeps`. */
  step() {
    if (this.count === 0) return;
    const ctx = this.ctx;
    this._uploadColliders();
    const dt = this.substepDt;
    const g = this.gravity.map((x) => (x * dt * dt) / this.dx);
    const vol = 1 / this.ppcAxis ** 3;
    for (let s = 0; s < this.substeps; s++) {
      if (this.stepCount % this.sortEvery === 0) ctx.region('mpm sort', () => this._sort());
      ctx.region('mpm p2g', () => this.p2g());
      ctx.region('mpm grid', () => {
        this.gridK.run(this.numNodes, { gravity: g, bound: this.bound, numColliders: Math.min(16, this.colliders.length), floorFriction: 0.4 },
          { grid: this.grid, COL: this.colBuf });
        this.coupling?.substep(s);
      });
      ctx.region('mpm g2p', () => {
        this.g2pK.run(this.count, { bound: this.bound, vmax: 0.9 }, { P: this.P, GV: this.grid, MATS: this.matBuf });
      });
      this.stepCount++;
    }
    this.coupling?.finish();
  }

  /** Clear the grid and scatter particles (P2G). forceGlobal: skip the shared-memory tile (baseline). */
  p2g(P = this.P, forceGlobal = false, grid = this.grid) {
    const vol = 1 / this.ppcAxis ** 3;
    this.ctx.clear(grid);
    this.p2gK.dispatch(Math.ceil(this.count / this.p2gPerGroup), { n: this.count, vol4: 4 * vol, forceGlobal: forceGlobal ? 1 : 0 },
      { P, grid, MATS: this.matBuf });
  }

  _sort() {
    const ctx = this.ctx;
    ctx.clear(this.blockCount);
    this.keyK.run(this.count, null, { P: this.P, key: this.key, rank: this.rank, cnt: this.blockCount });
    this.scan.run(this.blockCount, this.blockCount, this.numBlocks + 1);
    this.reorderK.run(this.count, null, { src: this.P, dst: this.P2, key: this.key, rank: this.rank, start: this.blockCount });
    [this.P, this.P2] = [this.P2, this.P];
  }

  /** Read particle positions (world units) — for tests / export. */
  async readPositions() {
    const raw = await this.P.read(this.count * 8);
    const out = new Float32Array(this.count * 3);
    for (let k = 0; k < this.count; k++)
      for (let a = 0; a < 3; a++) out[3 * k + a] = raw[32 * k + a] * this.dx + this.origin[a];
    return out;
  }

  /** Material color palette (linear RGB) indexed like the particle material ids. */
  palette() { return this.materials.map((m) => m.color || [0.7, 0.7, 0.7]); }

  /** Bitmask of material slots that are fluids (for the surface renderer). */
  fluidMask() { return this.materials.reduce((m, d, k) => (d.kind === FLUID ? m | (1 << k) : m), 0); }

  /**
   * Screen-space water surface for the fluid materials (pair with spheres({ solidsOnly: true })
   * to draw jelly / snow / sand as particles).
   */
  fluid(opts = {}) {
    const sim = this;
    const fs = new FluidSurface(this.ctx, {
      data: () => sim.P, stride: 8, position: 0, count: () => sim.count,
      radius: (this.dx / this.ppcAxis) * 0.9, scale: this.dx, offset: this.origin,
      mask: '((u.matMask >> bitcast<u32>(px.w)) & 1u) == 1u', ...opts,
    });
    return {
      surface: fs,
      set(v) { fs.set(v); return this; },
      afterScene(view) { fs.set({ matMask: sim.fluidMask() }); fs.afterScene(view); },
    };
  }

  /** Sphere-impostor layer that follows the ping-ponged particle buffer. */
  spheres(opts = {}) {
    const sim = this;
    const sp = new Spheres(this.ctx, {
      mask: '((u.matMask >> bitcast<u32>(px.w)) & 1u) == 1u',
      data: this.P, stride: 8, position: 0, velocity: 1, material: 'bitcast<u32>(px.w)',
      count: () => sim.count, radius: (this.dx / this.ppcAxis) * 0.6 * (opts.size ?? 1), scale: this.dx,
      offset: this.origin, palette: opts.palette || this.palette(), velScale: this.dx / this.substepDt, ...opts,
    });
    return {
      spheres: sp,
      set(v) { sp.set(v); return this; },
      draw(pass, view) {
        if (!opts.palette) sim.palette().forEach((c, k) => sp.program.set({ ['pal' + k]: [...c, 1] }));
        sp.program.set({ matMask: opts.solidsOnly ? ~sim.fluidMask() >>> 0 : 0xffffffff });
        sp.program.bind({ P: sim.P });
        sp.draw(pass, view);
      },
    };
  }
}

// ------------------------------------------------------------------ shape sampling
function sampleShape(shape, spacing, jitter, seed) {
  let rnd = seed >>> 0 || 1;
  const rand = () => { rnd ^= rnd << 13; rnd ^= rnd >>> 17; rnd ^= rnd << 5; return ((rnd >>> 0) / 4294967296); };
  let lo, hi, inside;
  if (shape.box) {
    [lo, hi] = shape.box;
    inside = () => true;
  } else if (shape.sphere) {
    const [c, r] = shape.sphere;
    lo = c.map((x) => x - r); hi = c.map((x) => x + r);
    inside = (p) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2 <= r * r;
  } else if (shape.cylinder) {
    const [c, r, h] = shape.cylinder;
    lo = [c[0] - r, c[1] - h, c[2] - r]; hi = [c[0] + r, c[1] + h, c[2] + r];
    inside = (p) => (p[0] - c[0]) ** 2 + (p[2] - c[2]) ** 2 <= r * r;
  } else if (shape.sdf) {
    lo = shape.bounds[0]; hi = shape.bounds[1];
    inside = (p) => shape.sdf(p) <= 0;
  } else throw new Error('MPM.add: shape needs box | sphere | cylinder | sdf | points');
  const n = [0, 1, 2].map((a) => Math.max(1, Math.floor((hi[a] - lo[a]) / spacing)));
  const out = [];
  const p = [0, 0, 0];
  for (let z = 0; z < n[2]; z++)
    for (let y = 0; y < n[1]; y++)
      for (let x = 0; x < n[0]; x++) {
        p[0] = lo[0] + (x + 0.5 + (rand() - 0.5) * jitter) * spacing;
        p[1] = lo[1] + (y + 0.5 + (rand() - 0.5) * jitter) * spacing;
        p[2] = lo[2] + (z + 0.5 + (rand() - 0.5) * jitter) * spacing;
        if (inside(p)) out.push(p[0], p[1], p[2]);
      }
  return new Float32Array(out);
}
return { FLUID, MATERIALS, MPM };
})();

// ---- src/physics/nbody.js
const __m25 = (() => {
// Direct-summation N-body gravity, O(N^2), tuned to run near the ALU roofline.
//
// One interaction is ~20 flops (3 sub, 3 fma for r^2, rsqrt, 2 mul, 3 fma) on
// 16 bytes of data, so the kernel is compute-bound as long as bodies are
// re-used from on-chip memory. Variants (picked per device by autotune()):
//   naive      every thread streams all bodies from global memory (baseline)
//   tiled      GPU Gems 3 ch. 31: workgroup tiles in shared memory
//   tiled-K    + register blocking: each thread integrates K bodies, so every
//              shared-memory load feeds K interactions (fewer loads/flop)
//   sgK        subgroup broadcast: bodies live in registers and are broadcast
//              lane by lane (no shared memory, no barriers) — needs a fixed
//              subgroup size, e.g. 32 on Apple GPUs.
// Integration: symplectic Euler (kick-drift), double-buffered positions.

const VARIANTS = (caps) => {
  const v = [{ mode: 'naive', wg: 256, K: 1 }];
  // K = 8 spills registers on every GPU we tried; subgroup broadcast lost to shared memory on Apple M2.
  for (const wg of [64, 128, 256]) for (const K of [1, 2, 4]) v.push({ mode: 'tiled', wg, K });
  if (caps.subgroups && caps.subgroupMin === 32 && caps.subgroupMax === 32)
    for (const K of [1, 2]) v.push({ mode: 'sg', wg: 256, K });
  return v;
};

function interact(k) {
  return `{ let d = b.xyz - p${k}.xyz; let r2 = dot(d, d) + u.eps2; let s = inverseSqrt(r2); a${k} += d * (b.w * s * s * s); }`;
}

class NBody {
  /**
   * @param {Context} ctx
   * @param {object} o { count, G = 1, softening = 0.02, dt = 0.002, variant }
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.count = o.count || 16384;
    this.G = o.G ?? 1;
    this.softening = o.softening ?? 0.02;
    this.dt = o.dt ?? 0.002;
    this.pos = ctx.array('vec4f', this.count, { label: 'nbody-pos', usage: GPUBufferUsage.VERTEX });
    this.pos2 = ctx.array('vec4f', this.count, { label: 'nbody-pos2', usage: GPUBufferUsage.VERTEX });
    this.vel = ctx.array('vec4f', this.count, { label: 'nbody-vel' });
    this.kernels = new Map();
    this.variant = o.variant || { mode: 'tiled', wg: 256, K: 2 };
  }

  static variants(caps) { return VARIANTS(caps); }

  kernel(v = this.variant) {
    const key = `${v.mode}-${v.wg}-${v.K}`;
    let k = this.kernels.get(key);
    if (k) return k;
    const K = v.K, WG = v.wg;
    const ks = [...Array(K).keys()];
    const loadOwn = ks.map((k) => `let i${k} = base + ${k}u * WG + lidx; let p${k} = pos[min(i${k}, u.n - 1u)]; var a${k} = vec3f(0.0);`).join('\n');
    const store = ks.map((k) => `if (i${k} < u.n) {
        let v = vel[i${k}].xyz + a${k} * (u.G * u.dt);
        vel[i${k}] = vec4f(v, 0.0);
        posOut[i${k}] = vec4f(p${k}.xyz + v * u.dt, p${k}.w);
      }`).join('\n');
    const all = (indent) => ks.map((k) => indent + interact(k)).join('\n');
    let shared = '', body;
    if (v.mode === 'naive') {
      body = `
        ${loadOwn}
        for (var j = 0u; j < u.n; j++) { let b = pos[j];
${all('          ')}
        }`;
    } else if (v.mode === 'tiled') {
      shared = `var<workgroup> tile: array<vec4f, ${WG}>;`;
      body = `
        ${loadOwn}
        let tiles = (u.n + WG - 1u) / WG;
        for (var t = 0u; t < tiles; t++) {
          let j = t * WG + lidx;
          tile[lidx] = select(vec4f(0.0), pos[min(j, u.n - 1u)], j < u.n);
          workgroupBarrier();
          for (var q = 0u; q < WG; q += 4u) {
            ${[0, 1, 2, 3].map((r) => `{ let b = tile[q + ${r}u];\n${all('              ')}\n            }`).join('\n            ')}
          }
          workgroupBarrier();
        }`;
    } else {
      // subgroup broadcast: each lane holds one body; constant-index broadcast over 32 lanes
      body = `
        ${loadOwn}
        let chunks = (u.n + 31u) / 32u;
        for (var t = 0u; t < chunks; t++) {
          let j = t * 32u + sgi;
          let mine = select(vec4f(0.0), pos[min(j, u.n - 1u)], j < u.n);
          ${[...Array(32).keys()].map((l) => `{ let b = subgroupBroadcast(mine, ${l}u);\n${all('            ')}\n          }`).join('\n          ')}
        }`;
    }
    k = this.ctx.kernel({
      name: `nbody-${key}`, workgroupSize: WG, guard: false, subgroups: v.mode === 'sg',
      buffers: { pos: { type: 'vec4f', access: 'read' }, posOut: { type: 'vec4f' }, vel: { type: 'vec4f' } },
      uniforms: { eps2: 'f32', G: 'f32', dt: 'f32' },
      shared,
      code: `let base = wgi * (WG * ${K}u);\n${body}\n${store}`,
    });
    k.bodiesPerGroup = WG * K;
    this.kernels.set(key, k);
    return k;
  }

  /** Advance `steps` steps. */
  step(steps = 1, variant = this.variant) {
    const k = this.kernel(variant);
    const groups = Math.ceil(this.count / k.bodiesPerGroup);
    this.ctx.region('nbody', () => {
      for (let s = 0; s < steps; s++) {
        k.dispatch(groups, { n: this.count, eps2: this.softening ** 2, G: this.G, dt: this.dt },
          { pos: this.pos, posOut: this.pos2, vel: this.vel });
        [this.pos, this.pos2] = [this.pos2, this.pos];
      }
    });
  }

  /**
   * Time every variant on this device and keep the fastest. Returns
   * [{ variant, ms, gips }] sorted fastest first (gips = 1e9 interactions / s).
   */
  async autotune({ reps = 4, cache = true } = {}) {
    const caps = this.ctx.caps;
    const key = `si-nbody-v1-${caps.vendor}-${caps.architecture}-${Math.round(Math.log2(this.count))}`;
    if (cache && typeof localStorage !== 'undefined') {
      try { const c = JSON.parse(localStorage.getItem(key)); if (c) { this.variant = c.variant; return c.results; } } catch {}
    }
    const results = [];
    const n2 = this.count * this.count;
    for (const v of VARIANTS(caps)) {
      const k = this.kernel(v);
      try { await k.ready(); } catch { continue; }
      // choose a batch so each measurement is ~20 ms of GPU work
      const ms = await timeIt(this.ctx, () => this.step(1, v), reps, n2);
      results.push({ variant: v, ms, gips: n2 / (ms * 1e6) });
    }
    results.sort((a, b) => a.ms - b.ms);
    this.variant = results[0].variant;
    if (cache && typeof localStorage !== 'undefined') localStorage.setItem(key, JSON.stringify({ variant: this.variant, results }));
    return results;
  }

  // ---------------------------------------------------------------- initial conditions
  /** Two rotating disk galaxies on a collision course. */
  galaxies(o = {}) {
    const n = this.count, rnd = mulberry(o.seed ?? 7);
    const pos = new Float32Array(n * 4), vel = new Float32Array(n * 4);
    const gal = [
      { c: [-0.9, 0, 0.15], v: [0.18, 0, -0.06], axis: norm([0.15, 1, 0.2]), share: 0.55 },
      { c: [0.9, 0, -0.15], v: [-0.22, 0, 0.07], axis: norm([-0.3, 1, -0.1]), share: 0.45 },
    ];
    const mStar = 1 / n;
    let k = 0;
    for (const g of gal) {
      const count = g === gal[1] ? n - k : Math.floor(n * g.share);
      const M = count * mStar;
      const coreMass = M * 0.25;
      const [t1, t2] = perp(g.axis);
      for (let s = 0; s < count; s++, k++) {
        let x, y, z, vx, vy, vz, m = mStar;
        if (s === 0) { // massive core
          m = coreMass; x = 0; y = 0; z = 0; vx = vy = vz = 0;
        } else {
          const r = 0.03 + 0.5 * -Math.log(1 - rnd() * 0.95) * 0.35;      // exponential disk
          const a = rnd() * Math.PI * 2;
          const h = (rnd() - 0.5) * 0.02 * (1 + r * 2);
          const enclosed = coreMass + (M - coreMass) * (1 - Math.exp(-r / 0.175) * (1 + r / 0.175));
          const vc = Math.sqrt((this.G * enclosed) / Math.sqrt(r * r + this.softening ** 2));
          const ca = Math.cos(a), sa = Math.sin(a);
          x = (t1[0] * ca + t2[0] * sa) * r + g.axis[0] * h;
          y = (t1[1] * ca + t2[1] * sa) * r + g.axis[1] * h;
          z = (t1[2] * ca + t2[2] * sa) * r + g.axis[2] * h;
          // tangent = axis x radial
          const rad = [t1[0] * ca + t2[0] * sa, t1[1] * ca + t2[1] * sa, t1[2] * ca + t2[2] * sa];
          const tan = cross(g.axis, rad);
          const disp = 0.05 * vc;
          vx = tan[0] * vc + (rnd() - 0.5) * disp; vy = tan[1] * vc + (rnd() - 0.5) * disp; vz = tan[2] * vc + (rnd() - 0.5) * disp;
        }
        pos.set([g.c[0] + x, g.c[1] + y, g.c[2] + z, m], 4 * k);
        vel.set([g.v[0] + vx, g.v[1] + vy, g.v[2] + vz, 0], 4 * k);
      }
    }
    this.pos.write(pos); this.vel.write(vel);
    return this;
  }

  /** Plummer sphere in virial equilibrium (Aarseth, Hénon & Wielen 1974). */
  plummer(o = {}) {
    const n = this.count, rnd = mulberry(o.seed ?? 3), a = o.scale ?? 0.3;
    const pos = new Float32Array(n * 4), vel = new Float32Array(n * 4);
    for (let k = 0; k < n; k++) {
      const r = a / Math.sqrt(Math.pow(rnd() * 0.99 + 1e-6, -2 / 3) - 1);
      const d = randDir(rnd);
      let q, g;
      do { q = rnd(); g = rnd() * 0.1; } while (g > q * q * Math.pow(1 - q * q, 3.5));
      const ve = Math.SQRT2 * Math.pow(r * r + a * a, -0.25) * Math.sqrt(this.G);
      const v = q * ve, dv = randDir(rnd);
      pos.set([d[0] * r, d[1] * r, d[2] * r, 1 / n], 4 * k);
      vel.set([dv[0] * v, dv[1] * v, dv[2] * v, 0], 4 * k);
    }
    this.pos.write(pos); this.vel.write(vel);
    return this;
  }
}

/** Mean GPU wall time (ms) of fn() measured over batches of ~20 ms. */
async function timeIt(ctx, fn, reps = 4, workHint = 0) {
  const q = ctx.device.queue;
  fn(); await ctx.idle();                       // warm-up / compile
  let t0 = performance.now();
  fn(); await ctx.idle();
  let one = Math.max(0.05, performance.now() - t0);
  const batch = Math.max(1, Math.min(200, Math.round(20 / one)));
  let best = Infinity;
  for (let r = 0; r < reps; r++) {
    t0 = performance.now();
    for (let b = 0; b < batch; b++) fn();
    ctx.flush();
    await q.onSubmittedWorkDone();
    best = Math.min(best, (performance.now() - t0) / batch);
  }
  return best;
}

function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function perp(n) {
  const t = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const a = norm(cross(n, t));
  return [a, cross(n, a)];
}
function randDir(rnd) {
  const z = rnd() * 2 - 1, a = rnd() * Math.PI * 2, r = Math.sqrt(1 - z * z);
  return [r * Math.cos(a), r * Math.sin(a), z];
}
return { NBody, timeIt };
})();

// ---- src/render/points.js
const __m26 = (() => {
const RenderProgram = __m11.RenderProgram;
// Additive glow sprites for very large particle counts (galaxies, sparks,
// curl-noise flows). Each particle is a camera-facing quad with a Gaussian
// falloff, blended additively into the HDR target (no depth write), so millions
// of overlapping points integrate into smooth light — pair with View bloom.


class Points {
  /**
   * o: data (vec4f GArray), stride, position, velocity (-1 = none), count, size (world units),
   *    pixelSize (min size in px), color [r,g,b] (linear HDR), colorBy 'constant'|'speed'|'w', speedScale, intensity
   */
  constructor(ctx, o) {
    this.o = o;
    const stride = o.stride || 1, pos = o.position || 0, vel = o.velocity ?? -1;
    this.program = new RenderProgram(ctx, {
      name: 'points', imports: ['color'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { size: 'f32', minPixels: 'f32', intensity: 'f32', speedScale: 'f32', color: 'vec3f', colorBy: 'u32', scale: 'f32', offset: 'vec3f' },
      targets: [{ format: 'rgba16float', blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }],
      depth: { write: false, compare: 'less' },
      topology: 'triangle-strip',
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) q: vec2f, @location(1) @interpolate(flat) col: vec3f }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let p = P[ii * ${stride}u + ${pos}u];
          let world = p.xyz * u.scale + u.offset;
          let clip = cam.viewProj * vec4f(world, 1.0);
          let corner = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
          // world-space size, but never below minPixels so far points stay visible
          let pxPerUnit = cam.viewport.y / (clip.w * cam.pixelAngle * cam.viewport.y);
          let sizePx = max(u.size * pxPerUnit, u.minPixels);
          let k = sizePx / max(u.size * pxPerUnit, 1e-6);   // >1 when clamped: dim to conserve energy
          let ofs = corner * sizePx / cam.viewport * 2.0 * clip.w;
          var col = u.color;
          ${vel >= 0 ? `if (u.colorBy == 1u) { let v = P[ii * ${stride}u + ${vel}u].xyz; let c = turbo(0.15 + 0.8 * clamp(length(v) * u.speedScale, 0.0, 1.0)); col = c * c * 1.5; }` : ''}
          if (u.colorBy == 2u) { col = turbo(clamp(p.w, 0.0, 1.0)); col = col * col * 1.5; }
          col *= u.intensity / (k * k);
          return VO(vec4f(clip.xy + ofs, clip.zw), corner, col);
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let r2 = dot(in.q, in.q);
          if (r2 > 1.0) { discard; }
          let g = exp(-r2 * 4.0) - 0.0183;
          return vec4f(in.col * g, 0.0);
        }`,
    });
    this.values = { size: o.size ?? 0.01, minPixels: o.pixelSize ?? 1.5, intensity: o.intensity ?? 1, speedScale: o.speedScale ?? 1,
      color: o.color || [1, 0.8, 0.5], colorBy: { constant: 0, speed: 1, w: 2 }[o.colorBy || 'constant'], scale: o.scale ?? 1, offset: o.offset || [0, 0, 0] };
    this.program.set(this.values).bind({ P: o.data });
  }
  set(v) { this.program.set(v); return this; }
  bind(r) { this.program.bind(r); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.data.length; }
  draw(pass, view) { const n = this.count; if (n > 0) this.program.draw(pass, view, 4, n); }
}
return { Points };
})();

// ---- src/render/sdf.js
const __m27 = (() => {
const RenderProgram = __m11.RenderProgram;
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

class SdfScene {
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
return { SdfScene };
})();

// ---- src/render/glyphs.js
const __m28 = (() => {
const RenderProgram = __m11.RenderProgram;
// Oriented instanced glyphs (arrows / fish / darts) for agents: each instance
// is a tiny tetrahedral dart aligned with its velocity, lit with the PBR module.
// 12 vertices per instance (4 triangles), generated in the vertex shader.


class Glyphs {
  /** o: pos (vec4f GArray), vel (vec4f GArray), count, size, color, colorBy 'speed'|'constant'|'heading', speedScale */
  constructor(ctx, o) {
    this.o = o;
    this.program = new RenderProgram(ctx, {
      name: 'glyphs', imports: ['pbr', 'color', 'math'],
      buffers: { P: { type: 'vec4f' }, V: { type: 'vec4f' } },
      uniforms: { size: 'f32', color: 'vec3f', colorBy: 'u32', speedScale: 'f32', sun: 'vec3f' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) n: vec3f, @location(1) col: vec3f, @location(2) w: vec3f }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          // dart: tip at +z, base triangle at z=0 (unit sized)
          var corners = array<vec3f, 4>(vec3f(0.0, 0.0, 1.6), vec3f(0.0, 0.55, 0.0), vec3f(-0.5, -0.35, 0.0), vec3f(0.5, -0.35, 0.0));
          var tris = array<u32, 12>(0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u);
          let p = P[ii].xyz;
          let v = V[ii].xyz;
          let f = safeNormalize(v + vec3f(1e-6, 0.0, 0.0));
          let side = safeNormalize(cross(vec3f(0.0, 1.0, 0.0), f) + vec3f(0.0, 0.0, 1e-6));
          let up = cross(f, side);
          let tri = vi / 3u;
          let a = corners[tris[tri * 3u]]; let b = corners[tris[tri * 3u + 1u]]; let c = corners[tris[tri * 3u + 2u]];
          let lc = corners[tris[vi]];
          var ln = normalize(cross(b - a, c - a));
          if (dot(ln, (a + b + c) / 3.0 - vec3f(0.0, -0.04, 0.4)) < 0.0) { ln = -ln; }   // face normals point outwards
          let world = p + (side * lc.x + up * lc.y + f * lc.z) * u.size;
          let n = side * ln.x + up * ln.y + f * ln.z;
          var col = u.color;
          if (u.colorBy == 1u) { let t = turbo(clamp(length(v) * u.speedScale, 0.0, 1.0)); col = t * t; }
          if (u.colorBy == 2u) { col = srgbToLinear(palette(atan2(f.z, f.x) / TAU + 0.5, vec3f(0.5), vec3f(0.45), vec3f(1.0), vec3f(0.0, 0.33, 0.67))); }
          return VO(cam.viewProj * vec4f(world, 1.0), n, col, world);
        }
        @fragment fn fs(in: VO) -> @location(0) vec4f {
          let n = normalize(in.n);
          let v = normalize(cam.eye - in.w);
          return vec4f(shadePBR(n, v, in.col, 0.0, 0.5, u.sun, vec3f(2.4), 1.0, 0.9), 1.0);
        }`,
    });
    this.program.set({ size: o.size ?? 0.01, color: o.color || [0.85, 0.85, 0.9], colorBy: { constant: 0, speed: 1, heading: 2 }[o.colorBy || 'heading'],
      speedScale: o.speedScale ?? 1.5, sun: [0.42, 0.8, 0.42] });
  }
  set(v) { this.program.set(v); return this; }
  get count() { const c = this.o.count; return typeof c === 'function' ? c() : c ?? this.o.pos.length; }
  draw(pass, view) { this.program.draw(pass, view, 12, this.count, null, { P: this.o.pos, V: this.o.vel }); }
}
return { Glyphs };
})();

// ---- src/physics/boids.js
const __m29 = (() => {
const SpatialGrid = __m9.SpatialGrid;
// Boids (Reynolds 1987) at scale: separation, alignment and cohesion over a
// fixed perception radius, with neighbours found through the counting-sort
// uniform grid. Every frame the agents are re-ordered into cell order, so the
// neighbour loop reads contiguous memory (the arrays stay permuted — boids have
// no identity, so no index map is needed).


class Boids {
  /**
   * o: count, bounds (half-size of the cube, default 1), radius (perception), maxSpeed, minSpeed,
   *    separation, alignment, cohesion, maxNeighbors (cap on interactions per boid)
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.count = o.count || 65536;
    this.half = o.bounds ?? 1;
    this.params = {
      radius: o.radius ?? 0.06, maxSpeed: o.maxSpeed ?? 0.6, minSpeed: o.minSpeed ?? 0.25,
      separation: o.separation ?? 1.6, alignment: o.alignment ?? 1.0, cohesion: o.cohesion ?? 0.9,
      maxNeighbors: o.maxNeighbors ?? 48, wall: o.wall ?? 2.0,
      predator: [0, 0, 0], predatorRadius: 0, predatorForce: 6,
    };
    const n = this.count;
    this.pos = ctx.array('vec4f', n, { label: 'boids-pos' });
    this.vel = ctx.array('vec4f', n, { label: 'boids-vel' });
    this.sPos = ctx.array('vec4f', n, { label: 'boids-spos' });
    this.sVel = ctx.array('vec4f', n, { label: 'boids-svel' });
    const h = this.half;
    this.grid = new SpatialGrid(ctx, { maxCount: n, min: [-h, -h, -h], max: [h, h, h], cellSize: this.params.radius });
    this.updateK = ctx.kernel({
      name: 'boids-update', imports: ['grid'],
      buffers: { S: { type: 'vec4f', access: 'read' }, SV: { type: 'vec4f', access: 'read' }, start: { type: 'u32', access: 'read' },
        pos: { type: 'vec4f' }, vel: { type: 'vec4f' } },
      uniforms: { ...this.grid.uniforms(), dt: 'f32', r2: 'f32', maxSpeed: 'f32', minSpeed: 'f32', wSep: 'f32', wAli: 'f32', wCoh: 'f32',
        half: 'f32', wall: 'f32', maxNb: 'u32', predator: 'vec3f', predR: 'f32', predF: 'f32' },
      code: /* wgsl */`
        let p = S[i].xyz;
        let v = SV[i].xyz;
        var sep = vec3f(0.0); var ali = vec3f(0.0); var coh = vec3f(0.0);
        var cnt = 0u;
        ${SpatialGrid.neighbors('p', /* wgsl */`
            if (j != i && cnt < u.maxNb) {
              let d = S[j].xyz - p;
              let d2 = dot(d, d);
              if (d2 < u.r2) {
                sep -= d / max(d2, 1e-5);
                ali += SV[j].xyz;
                coh += d;
                cnt++;
              }
            }`)}
        var acc = vec3f(0.0);
        if (cnt > 0u) {
          let inv = 1.0 / f32(cnt);
          acc += u.wSep * sep * inv * u.r2 * 4.0;
          acc += u.wAli * (ali * inv - v);
          acc += u.wCoh * coh * inv * (4.0 / sqrt(u.r2));
        }
        // soft walls: steer back when close to the cube faces
        let edge = u.half * 0.85;
        acc += u.wall * (max(vec3f(0.0), -edge - p) - max(vec3f(0.0), p - edge)) * 8.0;
        // predator (pointer) repulsion
        let dp = p - u.predator;
        let dpl = length(dp);
        if (dpl < u.predR) { acc += u.predF * dp / max(dpl, 1e-3) * (1.0 - dpl / u.predR); }
        var nv = v + acc * u.dt;
        let sp = length(nv);
        nv = nv * clamp(sp, u.minSpeed, u.maxSpeed) / max(sp, 1e-6);
        pos[i] = vec4f(clamp(p + nv * u.dt, vec3f(-u.half), vec3f(u.half)), 1.0);
        vel[i] = vec4f(nv, 0.0);`,
    });
  }

  /** Scatter boids uniformly in the cube with random velocities. */
  randomize(seed = 1) {
    let s = seed >>> 0 || 1;
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
    const n = this.count, p = new Float32Array(n * 4), v = new Float32Array(n * 4), h = this.half * 0.8;
    for (let k = 0; k < n; k++) {
      p.set([(rnd() * 2 - 1) * h, (rnd() * 2 - 1) * h, (rnd() * 2 - 1) * h, 1], 4 * k);
      const z = rnd() * 2 - 1, a = rnd() * 6.283, r = Math.sqrt(1 - z * z), sp = this.params.minSpeed;
      v.set([r * Math.cos(a) * sp, r * Math.sin(a) * sp, z * sp, 0], 4 * k);
    }
    this.pos.write(p); this.vel.write(v);
    return this;
  }

  step(dt = 1 / 60) {
    const P = this.params, g = this.grid;
    this.ctx.region('boids', () => {
      g.build(this.pos, this.count);
      g.gather(this.pos, this.sPos);
      g.gather(this.vel, this.sVel);
      this.updateK.run(this.count, {
        ...g.values(), dt: Math.min(dt, 1 / 30), r2: P.radius * P.radius, maxSpeed: P.maxSpeed, minSpeed: P.minSpeed,
        wSep: P.separation, wAli: P.alignment, wCoh: P.cohesion, half: this.half, wall: P.wall, maxNb: P.maxNeighbors,
        predator: P.predator, predR: P.predatorRadius, predF: P.predatorForce,
      }, { S: this.sPos, SV: this.sVel, start: g.start, pos: this.pos, vel: this.vel });
    });
  }
}
return { Boids };
})();

// ---- src/physics/cloth.js
const __m30 = (() => {
// XPBD cloth (Macklin, Müller & Chentanez 2016) with "small steps" (Macklin et
// al. 2019): many substeps with a single constraint iteration each, which
// converges faster than many iterations of one big step and makes the
// compliance (α) behave like a real material stiffness independent of the
// iteration count.
//
// Parallel Gauss–Seidel through graph colouring: on a W×H particle grid every
// constraint family (stretch-x, stretch-y, two shear diagonals, bend-x, bend-y)
// splits into 2 colours whose constraints share no particle, so 12 dispatches
// update every constraint once with no atomics and no races.
//
// XPBD update for a distance constraint C = |x_a - x_b| - L (λ reset each substep):
//   Δλ = -C / (w_a + w_b + α/Δt²),  x_a += w_a Δλ n,  x_b -= w_b Δλ n
//
// Long-range attachments (Kim, Chentanez & Müller 2012): each particle is also
// tethered to its nearest pin by a unilateral constraint |x - x_pin| <= L_rest.
// Gauss–Seidel spreads a load only a few edges per substep, so a hanging sheet
// would otherwise sag/stretch under its own weight; one extra dispatch fixes it.

const FAMILIES = [
  // [dx, dy, kind] kind: 0 stretch, 1 shear, 2 bend
  [1, 0, 0], [0, 1, 0], [1, 1, 1], [-1, 1, 1], [2, 0, 2], [0, 2, 2],
];

class Cloth {
  /**
   * o: width/height (particles), size: [sx, sz] world size, origin [x,y,z] (top-left corner),
   *    pin: 'top' | 'corners' | 'none' | (i, j) => bool, substeps, stretch/shear/bend compliance,
   *    gravity, damping, wind, colliders
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    const W = (this.W = o.width || 128), H = (this.H = o.height || 128);
    this.count = W * H;
    this.size = o.size || [1.6, 1.6];
    this.origin = o.origin || [-0.8, 1.6, 0];
    this.substeps = o.substeps || 16;
    this.compliance = { stretch: o.stretch ?? 0, shear: o.shear ?? 1e-6, bend: o.bend ?? 2e-4 };
    this.gravity = o.gravity || [0, -9.81, 0];
    this.damping = o.damping ?? 0.4;
    this.wind = o.wind || [0, 0, 0];
    this.windNoise = o.windNoise ?? 0.6;
    this.colliders = o.colliders || [];
    this.friction = o.friction ?? 0.4;
    this.thickness = o.thickness ?? 0.01;
    this.time = 0;
    const n = this.count;
    this.pos = ctx.array('vec4f', n, { label: 'cloth-pos', usage: GPUBufferUsage.VERTEX });
    this.prev = ctx.array('vec4f', n, { label: 'cloth-prev' });
    this.vel = ctx.array('vec4f', n, { label: 'cloth-vel' });
    this.nrm = ctx.array('vec4f', n, { label: 'cloth-normal' });
    this.colBuf = ctx.array('vec4f', 16 * 2, { label: 'cloth-colliders' });
    this.tether = ctx.array('vec2u', n, { label: 'cloth-tether' });
    this._kernels();
    this.reset(o.pin || 'top');
    // triangle index buffer for rendering
    const idx = new Uint32Array((W - 1) * (H - 1) * 6);
    let k = 0;
    for (let j = 0; j < H - 1; j++)
      for (let i = 0; i < W - 1; i++) {
        const a = j * W + i, b = a + 1, c = a + W, d = c + 1;
        idx.set([a, c, b, b, c, d], k); k += 6;
      }
    this.indices = ctx.device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    ctx.device.queue.writeBuffer(this.indices, 0, idx);
    this.indexCount = idx.length;
  }

  /** Flat sheet in the xz... actually hanging in the xy plane, with pinned particles. */
  reset(pin = this.pinMode) {
    this.pinMode = pin;
    const { W, H } = this, n = this.count;
    const p = new Float32Array(n * 4);
    const pinFn = typeof pin === 'function' ? pin
      : pin === 'top' ? (i, j) => j === 0
      : pin === 'corners' ? (i, j) => j === 0 && (i === 0 || i === W - 1)
      : pin === 'top-sparse' ? (i, j) => j === 0 && i % 8 === 0
      : () => false;
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++) {
        const k = j * W + i;
        const x = this.origin[0] + (i / (W - 1)) * this.size[0];
        // horizontal sheet when nothing is pinned (drop test), vertical curtain otherwise
        const vertical = pin !== 'none';
        const y = vertical ? this.origin[1] - (j / (H - 1)) * this.size[1] : this.origin[1];
        const z = vertical ? this.origin[2] + 0.002 * Math.sin(i * 0.7) : this.origin[2] + (j / (H - 1)) * this.size[1] - this.size[1] / 2;
        p.set([x, y, z, pinFn(i, j) ? 0 : 1], 4 * k);
      }
    this.pos.write(p); this.prev.write(p);
    this.ctx.clear(this.vel);
    // tethers: nearest pinned particle and its rest distance (static pins)
    const pins = [];
    for (let k = 0; k < n; k++) if (p[4 * k + 3] === 0) pins.push(k);
    const teth = new Uint32Array(n * 2), f32 = new Float32Array(teth.buffer);
    for (let k = 0; k < n; k++) {
      let best = -1, bd = Infinity;
      for (const q of pins) {
        const d = Math.hypot(p[4 * k] - p[4 * q], p[4 * k + 1] - p[4 * q + 1], p[4 * k + 2] - p[4 * q + 2]);
        if (d < bd) { bd = d; best = q; }
      }
      teth[2 * k] = best < 0 || best === k ? 0xffffffff : best;
      f32[2 * k + 1] = bd;
    }
    this.tether.write(teth);
    this.restX = this.size[0] / (W - 1);
    this.restY = this.size[1] / (H - 1);
  }

  _kernels() {
    const ctx = this.ctx, W = this.W, H = this.H;
    const defines = { W: `${W}u`, H: `${H}u` };
    this.predictK = ctx.kernel({
      name: 'cloth-predict', imports: ['noise'], defines,
      buffers: { pos: { type: 'vec4f' }, prev: { type: 'vec4f' }, vel: { type: 'vec4f' }, nrm: { type: 'vec4f', access: 'read' } },
      uniforms: { dt: 'f32', gravity: 'vec3f', wind: 'vec3f', windNoise: 'f32', time: 'f32', damping: 'f32', drag: 'f32' },
      code: /* wgsl */`
        let p = pos[i];
        prev[i] = p;
        if (p.w == 0.0) { return; }
        var v = vel[i].xyz;
        // aerodynamic drag on the normal component of the relative wind (cheap lift/drag model)
        let gust = u.wind * (1.0 + u.windNoise * simplex3(p.xyz * 1.5 + vec3f(u.time * 0.7, 0.0, u.time * 0.4)));
        let n = nrm[i].xyz;
        let rel = gust - v;
        let a = u.gravity + n * dot(rel, n) * u.drag;
        v = (v + a * u.dt) * (1.0 - u.damping * u.dt);
        pos[i] = vec4f(p.xyz + v * u.dt, p.w);`,
    });
    // One kernel solves one colour of one constraint family; (dx, dy) is the
    // particle offset, the colour selects every other constraint along the
    // direction so no two threads touch the same particle.
    this.solveK = ctx.kernel({
      name: 'cloth-solve', defines,
      buffers: { pos: { type: 'vec4f' } },
      uniforms: { dx: 'i32', dy: 'i32', color: 'u32', rest: 'f32', alpha: 'f32', ncx: 'u32', ncy: 'u32' },
      code: /* wgsl */`
        // enumerate constraints of this colour: start particles (si, sj)
        let cx = i % u.ncx;
        let cy = i / u.ncx;
        if (cy >= u.ncy) { return; }
        // colour along the dominant direction: blocks of |d| particles alternate
        var si: i32; var sj: i32;
        let adx = abs(u.dx); let ady = abs(u.dy);
        if (ady == 0) {          // horizontal: x = block * 2|d| + color*|d| + r
          let blk = cx / u32(adx); let r = cx % u32(adx);
          si = i32(blk * u32(2 * adx) + u.color * u32(adx) + r); sj = i32(cy);
        } else {                 // vertical / diagonal: rows alternate
          let blk = cy / u32(ady); let r = cy % u32(ady);
          sj = i32(blk * u32(2 * ady) + u.color * u32(ady) + r); si = i32(cx) + select(0, -u.dx, u.dx < 0);
        }
        let ti = si + u.dx; let tj = sj + u.dy;
        if (si < 0 || sj < 0 || ti < 0 || tj < 0 || si >= i32(W) || ti >= i32(W) || sj >= i32(H) || tj >= i32(H)) { return; }
        let a = u32(sj) * W + u32(si);
        let b = u32(tj) * W + u32(ti);
        let pa = pos[a]; let pb = pos[b];
        let wsum = pa.w + pb.w;
        if (wsum == 0.0) { return; }
        let d = pa.xyz - pb.xyz;
        let len = length(d);
        if (len < 1e-9) { return; }
        let n = d / len;
        let C = len - u.rest;
        let dl = -C / (wsum + u.alpha);
        pos[a] = vec4f(pa.xyz + pa.w * dl * n, pa.w);
        pos[b] = vec4f(pb.xyz - pb.w * dl * n, pb.w);`,
    });
    this.collideK = ctx.kernel({
      name: 'cloth-collide', defines,
      buffers: { pos: { type: 'vec4f' }, prev: { type: 'vec4f', access: 'read' }, vel: { type: 'vec4f' }, COL: { type: 'vec4f', access: 'read' } },
      uniforms: { numColliders: 'u32', thickness: 'f32', friction: 'f32', dt: 'f32', floorY: 'f32' },
      code: /* wgsl */`
        var p = pos[i];
        if (p.w == 0.0) { vel[i] = vec4f(0.0); return; }
        var x = p.xyz;
        let x0 = prev[i].xyz;
        for (var k = 0u; k < u.numColliders; k++) {
          let c = COL[k * 2u]; let s = COL[k * 2u + 1u];
          if (s.w < 0.5) {        // sphere: c.xyz centre, c.w radius
            let d = x - c.xyz;
            let l = length(d);
            let r = c.w + u.thickness;
            if (l < r) {
              let n = d / max(l, 1e-6);
              x = c.xyz + n * r;
              // friction: remove part of the tangential displacement this substep
              let dx = x - x0;
              let tang = dx - dot(dx, n) * n;
              x -= tang * min(u.friction, 1.0);
            }
          } else {                 // axis-aligned box: c.xyz centre, s.xyz half-extent
            let q = x - c.xyz;
            let h = s.xyz + u.thickness;
            if (all(abs(q) < h)) {
              let pen = h - abs(q);
              if (pen.x < pen.y && pen.x < pen.z) { x.x = c.x + sign(q.x) * h.x; }
              else if (pen.y < pen.z) { x.y = c.y + sign(q.y) * h.y; }
              else { x.z = c.z + sign(q.z) * h.z; }
              let dx = x - x0;
              x -= vec3f(dx.x, 0.0, dx.z) * min(u.friction, 1.0) * select(0.0, 1.0, pen.y < pen.x && pen.y < pen.z);
            }
          }
        }
        if (x.y < u.floorY + u.thickness) {
          x.y = u.floorY + u.thickness;
          let dx = x - x0;
          x.x -= dx.x * u.friction; x.z -= dx.z * u.friction;
        }
        pos[i] = vec4f(x, p.w);
        vel[i] = vec4f((x - x0) / u.dt, 0.0);`,
    });
    this.tetherK = ctx.kernel({
      name: 'cloth-tether', buffers: { pos: { type: 'vec4f' }, T: { type: 'vec2u', access: 'read' } }, uniforms: { slack: 'f32' },
      code: /* wgsl */`
        let t = T[i];
        if (t.x == 0xffffffffu) { return; }
        let p = pos[i];
        if (p.w == 0.0) { return; }
        let a = pos[t.x].xyz;
        let L = bitcast<f32>(t.y) * u.slack;
        let d = p.xyz - a;
        let l = length(d);
        if (l > L) { pos[i] = vec4f(a + d * (L / l), p.w); }`,
    });
    this.normalK = ctx.kernel({
      name: 'cloth-normals', defines, buffers: { pos: { type: 'vec4f', access: 'read' }, nrm: { type: 'vec4f' } },
      code: /* wgsl */`
        let ii = i32(i % W); let jj = i32(i / W);
        let l = pos[u32(jj) * W + u32(max(ii - 1, 0))].xyz;
        let r = pos[u32(jj) * W + u32(min(ii + 1, i32(W) - 1))].xyz;
        let d = pos[u32(max(jj - 1, 0)) * W + u32(ii)].xyz;
        let t = pos[u32(min(jj + 1, i32(H) - 1)) * W + u32(ii)].xyz;
        nrm[i] = vec4f(normalize(cross(t - d, r - l) + vec3f(0.0, 1e-9, 0.0)), 0.0);`,
    });
  }

  _uploadColliders() {
    const data = new Float32Array(16 * 8);
    this.colliders.slice(0, 16).forEach((c, k) => {
      if (c.type === 'box') data.set([...c.center, 0, ...(c.half || [0.1, 0.1, 0.1]), 1], k * 8);
      else data.set([...c.center, c.radius ?? 0.2, 0, 0, 0, 0], k * 8);
    });
    this.colBuf.write(data);
  }

  /** Advance one frame of `frameDt` seconds. */
  step(frameDt = 1 / 60) {
    const ctx = this.ctx, W = this.W, H = this.H;
    this._uploadColliders();
    const dt = Math.min(frameDt, 1 / 30) / this.substeps;
    const dt2 = dt * dt;
    ctx.region('cloth', () => {
      for (let s = 0; s < this.substeps; s++) {
        this.time += dt;
        this.predictK.run(this.count, { dt, gravity: this.gravity, wind: this.wind, windNoise: this.windNoise, time: this.time,
          damping: this.damping, drag: 3.0 }, { pos: this.pos, prev: this.prev, vel: this.vel, nrm: this.nrm });
        for (const [dx, dy, kind] of FAMILIES) {
          const rest = Math.hypot(dx * this.restX, dy * this.restY);
          const alpha = [this.compliance.stretch, this.compliance.shear, this.compliance.bend][kind] / dt2;
          const adx = Math.abs(dx), ady = Math.abs(dy);
          for (let color = 0; color < 2; color++) {
            // number of constraint start particles per colour (upper bounds; the kernel discards out-of-range)
            let ncx, ncy;
            if (ady === 0) { ncx = Math.ceil(W / (2 * adx)) * adx; ncy = H; }
            else { ncx = W; ncy = Math.ceil(H / (2 * ady)) * ady; }
            this.solveK.run(ncx * ncy, { dx, dy, color, rest, alpha, ncx, ncy }, { pos: this.pos });
          }
        }
        if (this.tethers !== false) this.tetherK.run(this.count, { slack: 1.005 }, { pos: this.pos, T: this.tether });
        this.collideK.run(this.count, { numColliders: Math.min(16, this.colliders.length), thickness: this.thickness,
          friction: this.friction, dt, floorY: this.floorY ?? -10 }, { pos: this.pos, prev: this.prev, vel: this.vel, COL: this.colBuf });
      }
      this.normalK.run(this.count, null, { pos: this.pos, nrm: this.nrm });
    });
  }
}
return { Cloth };
})();

// ---- src/render/mesh.js
const __m31 = (() => {
const RenderProgram = __m11.RenderProgram;
// Surface mesh whose vertices live in GPU storage buffers (positions + normals
// written by a simulation), drawn with an index buffer. Two-sided PBR shading
// with separate front/back colours and an optional grid pattern from (W, H).


class SurfaceMesh {
  /**
   * o: pos, nrm (vec4f GArrays), indices (GPUBuffer, uint32), indexCount,
   *    grid: [W, H] for UV-based patterns, front, back (linear RGB), roughness, pattern: 'checker'|'stripes'|'none'
   */
  constructor(ctx, o) {
    this.o = o;
    this.program = new RenderProgram(ctx, {
      name: 'surface-mesh', imports: ['pbr'],
      buffers: { P: { type: 'vec4f' }, N: { type: 'vec4f' } },
      uniforms: { front: 'vec3f', back: 'vec3f', roughness: 'f32', grid: 'vec2u', pattern: 'u32', sun: 'vec3f', tiles: 'f32' },
      code: /* wgsl */`
        struct VO { @builtin(position) pos: vec4f, @location(0) w: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
          let p = P[vi].xyz;
          let uv = vec2f(f32(vi % u.grid.x), f32(vi / u.grid.x)) / vec2f(max(vec2u(1u), u.grid - 1u));
          return VO(cam.viewProj * vec4f(p, 1.0), p, N[vi].xyz, uv);
        }
        @fragment fn fs(in: VO, @builtin(front_facing) front: bool) -> @location(0) vec4f {
          var n = normalize(in.n);
          let v = normalize(cam.eye - in.w);
          if (dot(n, v) < 0.0) { n = -n; }
          var albedo = select(u.back, u.front, front);
          let g = in.uv * u.tiles;
          if (u.pattern == 1u) {
            let c = (floor(g.x) + floor(g.y)) % 2.0;
            albedo *= 0.75 + 0.25 * c;
          } else if (u.pattern == 2u) {
            albedo *= 0.8 + 0.2 * step(0.5, fract(g.x));
          }
          // wrapped diffuse approximates light scattering through thin fabric
          let c = shadePBR(n, v, albedo, 0.0, u.roughness, u.sun, vec3f(2.8), 1.0, 1.0);
          let trans = albedo * 0.25 * max(dot(-n, u.sun), 0.0);
          return vec4f(c + trans, 1.0);
        }`,
    });
    this.program.set({ front: o.front || [0.75, 0.12, 0.1], back: o.back || [0.85, 0.75, 0.6], roughness: o.roughness ?? 0.75,
      grid: o.grid || [2, 2], pattern: { none: 0, checker: 1, stripes: 2 }[o.pattern || 'checker'], sun: [0.4, 0.75, 0.52], tiles: o.tiles ?? 12 });
  }
  set(v) { this.program.set(v); return this; }
  draw(pass, view) {
    this.program.drawIndexed(pass, view, this.o.indices, this.o.indexCount, 1, null, { P: this.o.pos, N: this.o.nrm });
  }
}
return { SurfaceMesh };
})();

// ---- src/physics/smoke.js
const __m32 = (() => {
// Eulerian smoke / fire on a staggered MAC grid (Harlow & Welch 1965; Stam
// 1999; Fedkiw, Stam & Jensen 2001; Bridson, "Fluid Simulation for Computer
// Graphics").
//
// Why staggered: on a collocated grid, central-difference divergence and
// gradient compose into a *wide* Laplacian, while the cheap 7-point Poisson
// solve uses the compact one — the mismatch leaves ~40% of the divergence
// behind (measured). On a MAC grid velocity components live on cell faces,
// div∘grad is exactly the compact 7-point Laplacian, and the projection is exact
// up to the solver tolerance.
//
// Storage (3D rgba16float textures, hardware trilinear filtering):
//   vel texel (i,j,k) = (u at the -x face, v at the -y face, w at the -z face) of cell (i,j,k)
//   scal texel (i,j,k) = (density, temperature) at the cell centre
// Per step: advect faces and scalars (semi-Lagrangian) -> forces (vorticity
// confinement, buoyancy, sources) -> divergence -> red–black SOR -> projection.

class Smoke {
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.size = o.size || [1, 1.5, 1];
    const res = o.res || 64;
    const dx = Math.max(...this.size) / res;
    this.dx = dx;
    this.dims = this.size.map((s) => Math.max(8, Math.round(s / dx)));
    this.size = this.dims.map((d) => d * dx);
    this.origin = o.origin || [-this.size[0] / 2, 0, -this.size[2] / 2];
    this.iterations = o.iterations ?? 24;
    this.omega = o.omega ?? 1.8;
    this.vorticity = o.vorticity ?? 0.35;
    this.buoyancy = o.buoyancy ?? 1.6;
    this.weight = o.weight ?? 0.08;
    this.cooling = o.cooling ?? 0.9;
    this.dissipation = o.dissipation ?? 0.15;
    this.sources = o.sources || [{ center: [0, 0.12, 0], radius: 0.09, density: 3, temperature: 2, velocity: [0, 0.6, 0] }];
    this.time = 0;
    const TU = GPUTextureUsage;
    const usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_DST | TU.COPY_SRC;
    const tex = (format, label) => ctx.texture({ size: this.dims, format, usage, label, dimension: '3d' });
    this.velA = tex('rgba16float', 'smoke-vel-a');
    this.velB = tex('rgba16float', 'smoke-vel-b');
    this.scalA = tex('rgba16float', 'smoke-scal-a');
    this.scalB = tex('rgba16float', 'smoke-scal-b');
    this.curl = tex('rgba16float', 'smoke-curl');
    this.div = tex('r32float', 'smoke-div');
    this.pressure = tex('r32float', 'smoke-p');
    this.srcBuf = ctx.array('vec4f', 8 * 3, { label: 'smoke-sources' });
    this._kernels();
  }

  _kernels() {
    const ctx = this.ctx;
    const wg = [4, 4, 4];
    const W3 = { storage: 'rgba16float', dim: '3d' };
    const S3 = { sampleType: 'float', dim: '3d' };
    const U3 = { sampleType: 'unfilterable-float', dim: '3d' };
    // Sampling helpers in *grid coordinates* g (cell (i,j,k) spans [i, i+1]^3).
    // A face-centred component is sampled by shifting into its own lattice.
    const mac = /* wgsl */`
      fn dimsF() -> vec3f { return vec3f(u.size); }
      fn sU(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.5, 0.0, 0.0)) / dimsF(), 0.0).x; }
      fn sV(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.0, 0.5, 0.0)) / dimsF(), 0.0).y; }
      fn sW(vel: texture_3d<f32>, g: vec3f) -> f32 { return textureSampleLevel(vel, lin, (g + vec3f(0.0, 0.0, 0.5)) / dimsF(), 0.0).z; }
      fn sVel(vel: texture_3d<f32>, g: vec3f) -> vec3f { return vec3f(sU(vel, g), sV(vel, g), sW(vel, g)); }`;
    const common = { workgroupSize: wg, samplers: { lin: 'linear' } };
    this.advectVelK = ctx.kernel({
      ...common, name: 'smoke-advect-vel',
      textures: { vel: S3, dst: W3 }, uniforms: { dtg: 'f32', decay: 'f32' }, shared: mac,
      code: /* wgsl */`
        let c = vec3f(p);
        // face positions in grid coordinates
        let gu = c + vec3f(0.0, 0.5, 0.5);
        let gv = c + vec3f(0.5, 0.0, 0.5);
        let gw = c + vec3f(0.5, 0.5, 0.0);
        let nu = sU(vel, gu - u.dtg * sVel(vel, gu));
        let nv = sV(vel, gv - u.dtg * sVel(vel, gv));
        let nw = sW(vel, gw - u.dtg * sVel(vel, gw));
        textureStore(dst, p, vec4f(vec3f(nu, nv, nw) * u.decay, 0.0));`,
    });
    this.advectScalK = ctx.kernel({
      ...common, name: 'smoke-advect-scalar',
      textures: { vel: S3, src: S3, dst: W3 }, uniforms: { dtg: 'f32', decay: 'vec4f' }, shared: mac,
      code: /* wgsl */`
        let g = vec3f(p) + 0.5;
        let back = g - u.dtg * sVel(vel, g);
        textureStore(dst, p, textureSampleLevel(src, lin, back / dimsF(), 0.0) * u.decay);`,
    });
    this.curlK = ctx.kernel({
      ...common, name: 'smoke-curl', textures: { vel: S3, dst: W3 }, uniforms: { invDx: 'f32' }, shared: mac,
      code: /* wgsl */`
        let g = vec3f(p) + 0.5;
        let ex = vec3f(1.0, 0.0, 0.0); let ey = vec3f(0.0, 1.0, 0.0); let ez = vec3f(0.0, 0.0, 1.0);
        let dwdy = sW(vel, g + ey) - sW(vel, g - ey); let dvdz = sV(vel, g + ez) - sV(vel, g - ez);
        let dudz = sU(vel, g + ez) - sU(vel, g - ez); let dwdx = sW(vel, g + ex) - sW(vel, g - ex);
        let dvdx = sV(vel, g + ex) - sV(vel, g - ex); let dudy = sU(vel, g + ey) - sU(vel, g - ey);
        let w = 0.5 * u.invDx * vec3f(dwdy - dvdz, dudz - dwdx, dvdx - dudy);
        textureStore(dst, p, vec4f(w, length(w)));`,
    });
    this._forceKernel();
    this.divK = ctx.kernel({
      ...common, name: 'smoke-div', textures: { vel: U3, dst: { storage: 'r32float', dim: '3d' } }, uniforms: { invDx: 'f32' },
      code: /* wgsl */`
        let c = vec3i(p);
        let m = vec3i(u.size) - 1;
        let f = textureLoad(vel, c, 0).xyz;           // -x, -y, -z faces
        let ux = select(textureLoad(vel, c + vec3i(1, 0, 0), 0).x, 0.0, c.x == m.x);
        let vy = select(textureLoad(vel, c + vec3i(0, 1, 0), 0).y, 0.0, c.y == m.y);
        let wz = select(textureLoad(vel, c + vec3i(0, 0, 1), 0).z, 0.0, c.z == m.z);
        let fx = select(f.x, 0.0, c.x == 0); let fy = select(f.y, 0.0, c.y == 0); let fz = select(f.z, 0.0, c.z == 0);
        textureStore(dst, c, vec4f(u.invDx * ((ux - fx) + (vy - fy) + (wz - fz)), 0.0, 0.0, 0.0));`,
    });
    // Red–black SOR on the compact Laplacian with Neumann walls: a wall face removes
    // that neighbour from the stencil (the diagonal becomes the number of open faces).
    this.sorK = ctx.kernel({
      ...common, name: 'smoke-sor',
      textures: { P: { storage: 'r32float', dim: '3d', access: 'read_write' }, div: U3 },
      uniforms: { parity: 'u32', omega: 'f32', h2: 'f32' },
      code: /* wgsl */`
        if (((p.x + p.y + p.z) & 1u) != u.parity) { return; }
        let c = vec3i(p);
        let m = vec3i(u.size) - 1;
        var s = 0.0; var nb = 0.0;
        if (c.x > 0) { s += textureLoad(P, c - vec3i(1, 0, 0)).x; nb += 1.0; }
        if (c.x < m.x) { s += textureLoad(P, c + vec3i(1, 0, 0)).x; nb += 1.0; }
        if (c.y > 0) { s += textureLoad(P, c - vec3i(0, 1, 0)).x; nb += 1.0; }
        if (c.y < m.y) { s += textureLoad(P, c + vec3i(0, 1, 0)).x; nb += 1.0; }
        if (c.z > 0) { s += textureLoad(P, c - vec3i(0, 0, 1)).x; nb += 1.0; }
        if (c.z < m.z) { s += textureLoad(P, c + vec3i(0, 0, 1)).x; nb += 1.0; }
        let pc = textureLoad(P, c).x;
        let gs = (s - u.h2 * textureLoad(div, c, 0).x) / nb;
        textureStore(P, c, vec4f(mix(pc, gs, u.omega), 0.0, 0.0, 0.0));`,
    });
    this.projectK = ctx.kernel({
      ...common, name: 'smoke-project',
      textures: { vel: U3, P: U3, dst: W3 }, uniforms: { invDx: 'f32' },
      code: /* wgsl */`
        let c = vec3i(p);
        let pc = textureLoad(P, c, 0).x;
        var v = textureLoad(vel, c, 0).xyz;
        // face between cell c-1 and c: subtract the pressure difference across it
        if (c.x > 0) { v.x -= u.invDx * (pc - textureLoad(P, c - vec3i(1, 0, 0), 0).x); } else { v.x = 0.0; }
        if (c.y > 0) { v.y -= u.invDx * (pc - textureLoad(P, c - vec3i(0, 1, 0), 0).x); } else { v.y = 0.0; }
        if (c.z > 0) { v.z -= u.invDx * (pc - textureLoad(P, c - vec3i(0, 0, 1), 0).x); } else { v.z = 0.0; }
        textureStore(dst, c, vec4f(v, 0.0));`,
    });
  }

  /** Vorticity confinement + buoyancy + sources, evaluated on the faces each component lives on. */
  _forceKernel() {
    const ctx = this.ctx;
    const S3 = { sampleType: 'float', dim: '3d' };
    const U3 = { sampleType: 'unfilterable-float', dim: '3d' };
    const W3 = { storage: 'rgba16float', dim: '3d' };
    this.forceK = ctx.kernel({
      workgroupSize: [4, 4, 4], samplers: { lin: 'linear' }, name: 'smoke-forces', imports: ['noise'],
      textures: { vel: U3, curl: S3, scal: S3, velOut: W3, scalOut: W3 },
      buffers: { SRC: { type: 'vec4f', access: 'read' } },
      uniforms: { dt: 'f32', dx: 'f32', eps: 'f32', buoy: 'f32', weight: 'f32', origin: 'vec3f', nsrc: 'u32', cooling: 'f32', time: 'f32' },
      shared: /* wgsl */`
        fn dimsF() -> vec3f { return vec3f(u.size); }
        fn wmag(g: vec3f, D: vec3f) -> f32 { return textureSampleLevel(curl, lin, g / D, 0.0).w; }
        // N x ω with N = ∇|ω| / |∇|ω|| (Fedkiw et al. 2001), sampled at grid position g
        fn confAt(g: vec3f, D: vec3f) -> vec3f {
          let gr = vec3f(wmag(g + vec3f(1.0, 0.0, 0.0), D) - wmag(g - vec3f(1.0, 0.0, 0.0), D),
                         wmag(g + vec3f(0.0, 1.0, 0.0), D) - wmag(g - vec3f(0.0, 1.0, 0.0), D),
                         wmag(g + vec3f(0.0, 0.0, 1.0), D) - wmag(g - vec3f(0.0, 0.0, 1.0), D));
          let l = length(gr);
          if (l < 1e-6) { return vec3f(0.0); }
          return cross(gr / l, textureSampleLevel(curl, lin, g / D, 0.0).xyz);
        }`,
      code: /* wgsl */`
        let c = vec3i(p);
        let cf = vec3f(p);
        let D = dimsF();
        var v = textureLoad(vel, c, 0).xyz;
        let gu = cf + vec3f(0.0, 0.5, 0.5); let gv = cf + vec3f(0.5, 0.0, 0.5); let gw = cf + vec3f(0.5, 0.5, 0.0);
        v += u.dt * u.eps * u.dx * vec3f(confAt(gu, D).x, confAt(gv, D).y, confAt(gw, D).z);
        let sv = textureSampleLevel(scal, lin, gv / D, 0.0);
        v.y += u.dt * (u.buoy * sv.y - u.weight * sv.x);
        var s = textureLoad(scal, c, 0);
        let pw = u.origin + (cf + 0.5) * u.dx;
        for (var k = 0u; k < u.nsrc; k++) {
          let a = SRC[k * 3u]; let b = SRC[k * 3u + 1u]; let svel = SRC[k * 3u + 2u];
          let d = length(pw - a.xyz) / a.w;
          if (d < 1.0) {
            let f = (1.0 - d * d) * (0.75 + 0.5 * simplex3(pw * 9.0 + vec3f(0.0, -u.time * 2.0, u.time)));
            s.x += b.x * f * u.dt;
            s.y = max(s.y, b.y * f);
            v = mix(v, svel.xyz, clamp(f * 40.0 * u.dt, 0.0, 1.0));
          }
        }
        s.y *= exp(-u.cooling * u.dt);
        if (c.x == 0) { v.x = 0.0; }
        if (c.y == 0) { v.y = 0.0; }
        if (c.z == 0) { v.z = 0.0; }
        textureStore(velOut, c, vec4f(v, 0.0));
        textureStore(scalOut, c, s);`,
    });
  }

  _uploadSources() {
    const d = new Float32Array(8 * 12);
    this.sources.slice(0, 8).forEach((s, k) => {
      d.set([...s.center, s.radius ?? 0.1, s.density ?? 2, s.temperature ?? 1, 0, 0, ...(s.velocity || [0, 0, 0]), 0], k * 12);
    });
    this.srcBuf.write(d);
  }

  /** Clear all fields. */
  reset() {
    for (const t of [this.velA, this.velB, this.scalA, this.scalB, this.curl, this.div, this.pressure]) {
      const bytes = t.format === 'r32float' ? 4 : 8;
      const zero = new Uint8Array(t.width * t.height * bytes);
      for (let z = 0; z < t.depth; z++)
        this.ctx.device.queue.writeTexture({ texture: t.texture, origin: [0, 0, z] }, zero, { bytesPerRow: t.width * bytes, rowsPerImage: t.height }, [t.width, t.height, 1]);
    }
  }

  step(dt = 1 / 60) {
    dt = Math.min(dt, 1 / 30);
    this.time += dt;
    this._uploadSources();
    const D = this.dims, dx = this.dx;
    const dtg = dt / dx; // velocity (world/s) -> grid cells per step
    const dec = Math.exp(-this.dissipation * dt);
    this.ctx.region('smoke', () => {
      this.advectVelK.run(D, { dtg, decay: 0.999 }, { vel: this.velA, dst: this.velB });
      this.advectScalK.run(D, { dtg, decay: [dec, 1, 1, 1] }, { vel: this.velA, src: this.scalA, dst: this.scalB });
      this.curlK.run(D, { invDx: 1 / dx }, { vel: this.velB, dst: this.curl });
      this.forceK.run(D, { dt, dx, eps: this.vorticity * 10, buoy: this.buoyancy, weight: this.weight, origin: this.origin,
        nsrc: Math.min(8, this.sources.length), cooling: this.cooling, time: this.time },
      { vel: this.velB, curl: this.curl, scal: this.scalB, velOut: this.velA, scalOut: this.scalA, SRC: this.srcBuf });
      this.divK.run(D, { invDx: 1 / dx }, { vel: this.velA, dst: this.div });
      for (let k = 0; k < this.iterations; k++) {
        this.sorK.run(D, { parity: 0, omega: this.omega, h2: dx * dx }, { P: this.pressure, div: this.div });
        this.sorK.run(D, { parity: 1, omega: this.omega, h2: dx * dx }, { P: this.pressure, div: this.div });
      }
      this.projectK.run(D, { invDx: 1 / dx }, { vel: this.velA, P: this.pressure, dst: this.velB });
      [this.velA, this.velB] = [this.velB, this.velA];
    });
  }
}
return { Smoke };
})();

// ---- src/render/volume.js
const __m33 = (() => {
const RenderProgram = __m11.RenderProgram;
// Volumetric renderer for a 3D density/temperature texture (the Smoke solver's
// scalar field). Single scattering with precomputed light transmittance:
//   1. compute: for every voxel march toward the sun -> transmittance texture
//   2. fragment: march the view ray through the box (stopping at the scene's
//      depth buffer), accumulating in-scattered sun light (Henyey–Greenstein
//      phase), sky ambient and blackbody-ish emission from temperature.
// Composited over the HDR scene with premultiplied alpha after the opaque pass.


class Volume {
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
return { Volume };
})();

// ---- src/physics/rigid.js
const __m34 = (() => {
const SpatialGrid = __m9.SpatialGrid;
const Scan = __m7.Scan;
const BODY_VEC4 = __m21.BODY_VEC4;
const CONTACT_VEC4 = __m21.CONTACT_VEC4;
const JOINT_VEC4 = __m21.JOINT_VEC4;
const SHAPES = __m21.SHAPES;
const FLAGS = __m21.FLAGS;
const JOINT_TYPES = __m21.JOINT_TYPES;
const JOINT_FLAGS = __m21.JOINT_FLAGS;
const NARROW_WGSL = __m21.NARROW_WGSL;
const RAY_WGSL = __m21.RAY_WGSL;
// Rigid bodies with Augmented Vertex Block Descent (Giles, Diaz & Yuksel,
// SIGGRAPH 2025), building on Vertex Block Descent (Chen et al., 2024).
//
// Every body is one 6-DOF "vertex". Per step:
//   commands  game-side edits (poses, velocities, impulses, flags, joint motors) are
//             queued on the CPU and applied in order by one GPU thread.
//   predict   inertial target y = x₀ + v Δt + g Δt²; external impulses (the `ext`
//             buffer, written by commands, force fields, fluid coupling, neural nets)
//             are applied here. Iterations start at x₀ + vΔt + a gΔt², where a ∈ [0, 1]
//             is how much the body actually fell last step (VBD's adaptive guess).
//   pairs     broadphase on bounding spheres in a counting-sort grid sized for ordinary
//             bodies (the few oversized ones are tested against everything). Each pair is
//             owned by one body, which keeps its pairs sorted; a second pass gives every
//             body the sorted list of pairs it joins as the second body.
//   contacts  narrowphase, one thread per pair, run twice: count, prefix-sum, write. Box–box
//             (SAT over 15 axes, face clipping, ≤ 4 points), sphere, capsule and floor
//             manifolds, generated with a margin (speculative) so fast bodies do not tunnel.
//             Every point inherits λ and k from the nearest point of the same pair last step
//             (GPU hash table); a point that was sticking keeps its anchors.
//   colour    Jones–Plassmann + first fit over contact and joint edges, double-buffered.
//   iterate   per colour, each body takes one Newton step on its local energy
//               G(x) = ½|x − y|²_M / Δt² + Σ_c [ ½ k_c C_c² + λ_c C_c ]
//             (contacts linearised at the step start, joint rows evaluated exactly), solving
//             H Δ = −g as 3×3 blocks in registers with 8 lanes accumulating a body's rows;
//             then one dual update per contact and per joint (augmented Lagrangian):
//               λ ← k C + λ (clamped to f_n ≤ 0, |f_t| ≤ μ|λ_n|, motor limits),
//               k ← min(k_max, k + β|C|) while the constraint is active and not sliding.
//   velocity  v = (x − x₀)/Δt, ω from the orientation change.
//
// Determinism (confluence): no stage depends on the order in which GPU threads run.
// Pair lists are sorted, contact slots come from prefix sums rather than atomic counters,
// every body visits its contacts in a fixed order, colouring reads only the previous round,
// and the only atomics left are integer sums and hash-table slots (whose values do not
// depend on order). The same inputs on the same device give a bit-identical state, which
// is what lockstep and rollback netcode need (see hash(), snapshot(), net/).






const MAX_COLORS = 16;
const COLOR_ROUNDS = 20;
const MAX_LARGE = 256;    // bodies too big for the broadphase grid, tested against everything
const LANES = 8;          // threads per body in the primal step
const BODIES_PER_WG = 64 / LANES;
const MAXP = 48;          // pairs a body owns (kept: the lowest partner indices)
const MAXR = 64;          // pairs a body joins as the second body
const MAXRL = 1024;       // ... for a large dynamic body (only its pairs with contacts)
const CAND = Math.max(MAXP, MAXR);

// Where body b's reverse list lives. Small bodies have MAXR slots of their own; a large body's
// count word is m | (k + 1) << 16 for large-list entry k, and its MAXRL slots follow the counts.
const reverseList = (NB, b) => /* wgsl */`
  let rw = rlist[${NB * MAXR}u + ${b}];
  let rn = min(rw & 0xffffu, select(${MAXR}u, ${MAXRL}u, rw > 0xffffu));
  let rb = select(${b} * ${MAXR}u, ${NB * (MAXR + 1)}u + ((rw >> 16u) - 1u) * ${MAXRL}u, rw > 0xffffu);`;
const NONE = 0xffffffff;
const CMD = { pose: 1, velocity: 2, impulse: 3, flags: 4, filter: 5, kinematic: 6, motor: 7, jointFlags: 8, limits: 9, props: 10 };

// Reads only the members of a body a constraint needs (spliced into kernels that bind B).
const POSE_WGSL = /* wgsl */`
fn pose(i: u32) -> Pose { return Pose(B[i].x.xyz, B[i].x0.xyz, B[i].q, B[i].q0); }`;

class RigidWorld {
  /**
   * o: maxBodies, maxContacts, maxPairs, maxJoints, gravity, iterations, substeps,
   *    floor (y, or null for none), bounds [[min], [max]], deterministic (fixed colour budget),
   *    events (contact begin/end events). Solver knobs (defaults are the tested values):
   *    alpha 0.95 (fraction of an existing overlap kept per step), beta 30 (stiffness ramp,
   *    × m_eff/Δt² per metre), gamma 0.99 (stiffness decay between steps), kStart 1e-3 and
   *    kStartJoint 1 (× m_eff/Δt²), jointMassFloor 1 (kg: joint stiffness is scaled as if bodies weighed
   *    at least this, so light links carrying heavy loads stiffen fast enough — what AVBD's absolute β
   *    does for small bodies), contactMassFloor 0 (the same for contacts: helps a heavy box on light ones
   *    but makes piles of very light bodies jitter),
   *    contactTolerance 0 (opt-in: contact stiffness ≥ |λ|/ε; fixes heavy-on-light stacks of a few
   *    bodies but destabilises large piles), margin 0.02 (speculative distance), slop 5e-4,
   *    stickThreshold 0.01, postStabilize false.
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.maxBodies = Math.min(o.maxBodies || 16384, 65535);
    this.maxContacts = o.maxContacts || this.maxBodies * 16;
    this.maxPairs = o.maxPairs || this.maxBodies * 8;
    this.maxJoints = o.maxJoints || Math.max(64, this.maxBodies);
    this.maxCommands = o.maxCommands || 4096;
    this.gravity = o.gravity || [0, -9.81, 0];
    this.iterations = o.iterations ?? 8;
    this.substeps = o.substeps ?? 1;
    this.floor = o.floor === undefined ? 0 : o.floor;
    this.bounds = o.bounds || [[-40, -2, -40], [40, 40, 40]];
    this.deterministic = o.deterministic ?? false;
    this.postStabilize = o.postStabilize ?? false;
    this.alpha = o.alpha ?? 0.95;
    this.beta = o.beta ?? 30;
    this.gamma = o.gamma ?? 0.99;
    this.kStart = o.kStart ?? 1e-3;
    this.kStartJoint = o.kStartJoint ?? 1;
    this.jointMassFloor = o.jointMassFloor ?? 1;
    this.contactMassFloor = o.contactMassFloor ?? 0;
    this.contactTolerance = o.contactTolerance ?? 0;
    this.margin = o.margin ?? 0.02;
    this.slop = o.slop ?? 5e-4;
    this.stickThreshold = o.stickThreshold ?? 0.01;
    this.geoStiff = o.geoStiff ?? 1;
    this.fusedSolve = o.fusedSolve ?? 128;   // bodies up to which the whole solve is one dispatch (true / false force it)
    this.events = o.events ?? false;
    this.count = 0;
    this.uploaded = 0;
    this.cpu = [];
    this.free = [];
    this.dirty = new Set();
    this.joints = [];
    this.jointsUploaded = 0;
    this.jointsDirty = false;
    this.cmds = [];
    this.frame = 0;
    this._steps = 0;
    this.rSmall = 0.01;
    this.nLarge = 0;
    this.stats = { contacts: 0, colors: 0, pairs: 0 };
    this.forceFields = [];

    const A = (t, n, label, usage) => ctx.array(t, n, { label, usage });
    const tableSize = 1 << Math.ceil(Math.log2(this.maxContacts));
    this.tableSize = tableSize;
    this.B = A('vec4f', this.maxBodies * BODY_VEC4, 'rigid-bodies', GPUBufferUsage.VERTEX);
    // external impulses per body: (J, -) (L, -) in world units, applied and cleared each step
    this.ext = A('vec4f', this.maxBodies * 2, 'rigid-external-impulses');
    this.centers = A('vec4f', this.maxBodies, 'rigid-centers');
    this.pcount = A('u32', this.maxBodies + 1, 'rigid-pair-counts');
    this.poffset = A('u32', this.maxBodies + 1, 'rigid-pair-offsets');
    this.pairs = A('vec2u', this.maxPairs, 'rigid-pairs');
    this.pscratch = A('u32', this.maxBodies * MAXP, 'rigid-owned-partners');   // each body's sorted partners
    this.rlist = A('u32', this.maxBodies * (MAXR + 1) + MAX_LARGE * MAXRL, 'rigid-reverse-pairs');   // [lists | counts | large lists]
    this.ccount = A('u32', this.maxPairs + 1, 'rigid-contact-counts');
    this.coffset = A('u32', this.maxPairs + 1, 'rigid-contact-offsets');
    this.C = [A('vec4f', this.maxContacts * CONTACT_VEC4, 'rigid-contacts-0'), A('vec4f', this.maxContacts * CONTACT_VEC4, 'rigid-contacts-1')];
    // pair table: [keys (T) | (base, count) pairs (2T)] — one binding instead of two
    this.tbl = [A('u32', tableSize * 3, 'rigid-pairs-0'), A('u32', tableSize * 3, 'rigid-pairs-1')];
    this.J = A('vec4f', this.maxJoints * JOINT_VEC4, 'rigid-joints');
    this.jcsr = A('u32', this.maxBodies + 1 + 2 * this.maxJoints, 'rigid-joint-lists');
    this.jsetSize = 1 << Math.ceil(Math.log2(Math.max(16, 2 * this.maxJoints)));
    this.jset = A('u32', this.jsetSize * 2, 'rigid-jointed-pairs');
    this.large = A('u32', MAX_LARGE, 'rigid-large-bodies');
    this.color = [A('u32', this.maxBodies, 'rigid-color-0'), A('u32', this.maxBodies, 'rigid-color-1')];
    this.clist = A('u32', this.maxBodies, 'rigid-color-list');
    this.info = A('u32', MAX_COLORS * 4, 'rigid-color-info');
    // indirect args: one slot per colour, then the dual update, then the narrowphase
    this.args = A('u32', MAX_COLORS * 4 + 8, 'rigid-dispatch-args', GPUBufferUsage.INDIRECT);
    // counts: [1] contacts kept, [2] colours used, [3] candidate pairs, [4] events
    this.cnts = A('u32', 8, 'rigid-counts');
    this.cmdBuf = A('vec4f', this.maxCommands * 3, 'rigid-commands');
    this.maxEvents = o.maxEvents || 4096;
    this.eventBuf = A('vec4u', this.maxEvents, 'rigid-events');
    this.hashBuf = A('u32', 4, 'rigid-hash');
    this.pairScan = new Scan(ctx, this.maxBodies + 1);
    this.contactScan = new Scan(ctx, this.maxPairs + 1);
    this.grid = null;
    this._kernels();
  }

  // ------------------------------------------------------------------ bodies
  /** Add a box: { position, half: [hx,hy,hz], rotation: quat [x,y,z,w] | euler [rx,ry,rz], velocity, angularVelocity,
   *  density, friction, static, kinematic, sensor, fixedRotation, gravityScale, linearDamping, angularDamping, group, mask, userData } */
  addBox(o) { return this._add({ ...o, shape: SHAPES.box }); }
  /** Add a sphere: { position, radius, ... } */
  addSphere(o) { return this._add({ ...o, shape: SHAPES.sphere, half: [o.radius, o.radius, o.radius] }); }
  /** Add a capsule along its local y axis: { position, radius, halfHeight (of the straight part), ... } */
  addCapsule(o) { return this._add({ ...o, shape: SHAPES.capsule, half: [o.radius, o.halfHeight ?? o.radius, o.radius] }); }

  _add(o) {
    const id = this.free.length ? this.free.pop() : this.count;
    if (id >= this.maxBodies) return -1;
    const h = o.half || [0.5, 0.5, 0.5];
    const q = o.rotation ? (o.rotation.length === 4 ? o.rotation : eulerQuat(o.rotation)) : [0, 0, 0, 1];
    const rho = o.density ?? 1;
    let m, I, radius;
    if (o.shape === SHAPES.box) {
      m = rho * 8 * h[0] * h[1] * h[2];
      I = [(m / 3) * (h[1] ** 2 + h[2] ** 2), (m / 3) * (h[0] ** 2 + h[2] ** 2), (m / 3) * (h[0] ** 2 + h[1] ** 2)];
      radius = Math.hypot(...h);
    } else if (o.shape === SHAPES.capsule) {
      const r = h[0], H = 2 * h[1];
      const mc = rho * Math.PI * r * r * H, mh = rho * (2 / 3) * Math.PI * r ** 3;
      const Iax = mc * r * r / 2 + 2 * mh * 0.4 * r * r;
      const Ip = mc * (H * H / 12 + r * r / 4) + 2 * mh * (0.4 * r * r + H * H / 4 + 3 * H * r / 8);
      m = mc + 2 * mh; I = [Ip, Iax, Ip];
      radius = h[1] + r;
    } else {
      const r = h[0];
      m = (rho * 4 / 3) * Math.PI * r ** 3;
      I = [0.4 * m * r * r, 0.4 * m * r * r, 0.4 * m * r * r];
      radius = r;
    }
    let flags = 0;
    if (o.kinematic) flags |= FLAGS.kinematic;
    if (o.sensor) flags |= FLAGS.sensor;
    if (o.fixedRotation) flags |= FLAGS.fixedRotation;
    if (o.static || o.kinematic) { m = 0; I = [0, 0, 0]; }
    const p = o.position || [0, 0, 0], v = o.velocity || [0, 0, 0], w = o.angularVelocity || [0, 0, 0];
    const rec = new Float32Array(BODY_VEC4 * 4);
    const ru = new Uint32Array(rec.buffer);
    rec.set([p[0], p[1], p[2], radius], 0);
    rec.set(q, 4);
    rec.set([v[0], v[1], v[2], o.friction ?? 0.6], 8);
    rec.set([w[0], w[1], w[2], 0], 12);
    rec.set([p[0], p[1], p[2], radius], 16); rec.set(q, 20);
    rec.set([p[0], p[1], p[2], 0], 24); rec.set(q, 28);
    rec.set([h[0], h[1], h[2], o.shape], 32);
    rec.set([m, I[0], I[1], I[2]], 36);
    rec.set([o.gravityScale ?? 1, o.linearDamping ?? 0, o.angularDamping ?? 0, 0], 40);
    ru[44] = (o.group ?? 1) >>> 0; ru[45] = (o.mask ?? 0xffffffff) >>> 0; ru[46] = flags; ru[47] = (o.userData ?? 0) >>> 0;
    if (id === this.count) { this.cpu.push(rec); this.count++; }
    else {
      // a reused slot: drop queued edits aimed at its previous occupant (e.g. the removal itself)
      this.cmds = this.cmds.filter((c) => c.id !== id || (c.type >= CMD.motor && c.type <= CMD.limits));
      this.cpu[id] = rec; this.dirty.add(id);
    }
    this._pending = true;
    return id;
  }

  /** Remove a body (its slot is reused by later adds; joints attached to it are disabled). */
  remove(id) {
    this._cmd(CMD.flags, id, null, null, 0, FLAGS.disabled);
    this.free.push(id);
    this.joints.forEach((j, k) => { if (j.a === id || j.b === id) this.setJointEnabled(k, false); });
  }

  // ------------------------------------------------------------------ commands (applied in order, on the GPU)
  _cmd(type, id, a, c, hz = 0, hw = 0) {
    this.cmds.push({ type, id, a: a || [0, 0, 0, 0], c: c || [0, 0, 0, 0], hz, hw });
  }
  /** Teleport a body. */
  setPose(id, position, rotation) {
    const q = rotation ? (rotation.length === 4 ? rotation : eulerQuat(rotation)) : null;
    this._cmd(CMD.pose, id, [...position, q ? 1 : 0], q || [0, 0, 0, 1]);
  }
  setVelocity(id, v, w) { this._cmd(CMD.velocity, id, [...v, 1], w ? [...w, 1] : [0, 0, 0, 0]); }
  /** Impulse J (N·s) at an optional world point (default: the centre of mass). */
  applyImpulse(id, J, point) { this._cmd(CMD.impulse, id, [...J, 0], point ? [...point, 1] : [0, 0, 0, 0]); }
  /** Force F (N) applied for the next frame (dt), at an optional world point. */
  applyForce(id, F, point, dt = 1 / 60) { this.applyImpulse(id, F.map((f) => f * dt), point); }
  /** Angular impulse (N·m·s). */
  applyAngularImpulse(id, L) { this._cmd(CMD.impulse, id, [0, 0, 0, 0], [...L, 2]); }
  /** Move a kinematic body so it reaches the target pose at the end of this frame. */
  moveKinematic(id, position, rotation) {
    const q = rotation ? (rotation.length === 4 ? rotation : eulerQuat(rotation)) : [0, 0, 0, 0];
    this._cmd(CMD.kinematic, id, [...position, rotation ? 1 : 0], q);
  }
  setFilter(id, group, mask) { this._cmd(CMD.filter, id, null, null, group >>> 0, mask >>> 0); }
  setEnabled(id, on) { this._cmd(CMD.flags, id, null, null, on ? FLAGS.disabled : 0, on ? 0 : FLAGS.disabled); }
  setFlags(id, set = 0, clear = 0) { this._cmd(CMD.flags, id, null, null, clear, set); }
  /** (gravityScale, linearDamping, angularDamping) */
  setProps(id, { gravityScale = 1, linearDamping = 0, angularDamping = 0 } = {}) { this._cmd(CMD.props, id, [gravityScale, linearDamping, angularDamping, 0]); }

  // ------------------------------------------------------------------ joints
  /**
   * Add a joint between body a and body b (b: null or -1 for the world).
   * { type: 'ball' | 'hinge' | 'fixed' | 'distance' | 'slider', a, b, anchor: world point (default: b's
   *   centre, or a's for the world), anchorB (distance joints: the other end), axis: world axis
   *   (hinge, slider), limits: [lo, hi] (rad or m), motor: { speed, maxForce }, length (distance;
   *   default: current), rope (distance: only resists stretching), stiffness (N/m: a spring instead
   *   of a rigid joint), damping, breakForce, collideConnected (default false) }
   * Anchors and axes are given in world space and converted with the bodies' poses at the next step.
   */
  addJoint(o) {
    const k = this.joints.length;
    if (k >= this.maxJoints) return -1;
    const type = JOINT_TYPES[o.type || 'ball'];
    if (type === undefined) throw new Error(`RigidWorld: unknown joint type ${o.type}`);
    const a = o.a, b = o.b === undefined || o.b === null || o.b < 0 ? NONE : o.b;
    const pa = this.cpu[a] ? [...this.cpu[a].subarray(0, 3)] : [0, 0, 0];
    const anchor = o.anchor || (b === NONE ? pa : [...this.cpu[b].subarray(0, 3)]);
    const anchorB = o.anchorB || anchor;
    const axis = o.axis || [0, 1, 0];
    const al = Math.hypot(...axis) || 1;
    let flags = JOINT_FLAGS.worldAnchors;
    if (o.collideConnected) flags |= JOINT_FLAGS.collide;
    if (o.limits) flags |= JOINT_FLAGS.limit;
    if (o.motor) flags |= JOINT_FLAGS.motor;
    if (o.rope) flags |= JOINT_FLAGS.rope;
    const rec = new Float32Array(JOINT_VEC4 * 4);
    const ru = new Uint32Array(rec.buffer);
    ru[0] = a; ru[1] = b; ru[2] = type; ru[3] = flags;
    rec.set([anchor[0], anchor[1], anchor[2], o.breakForce ?? 0], 4);
    rec.set([anchorB[0], anchorB[1], anchorB[2], o.length ?? -1], 8);
    rec.set([0, 0, 0, 1], 12);
    rec.set([axis[0] / al, axis[1] / al, axis[2] / al, o.limits ? o.limits[0] : 0], 16);
    rec.set([axis[0] / al, axis[1] / al, axis[2] / al, o.limits ? o.limits[1] : 0], 20);
    rec.set([o.stiffness ?? 0, o.damping ?? 0, o.motor?.speed ?? 0, o.motor?.maxForce ?? 0], 24);
    this.joints.push({ a, b, type, flags, rec, enabled: true });
    this.jointsDirty = true;
    this._pending = true;
    return k;
  }
  /** Motor target speed (rad/s or m/s) and force/torque limit; speed null turns the motor off. */
  setMotor(j, speed, maxForce) { this._cmd(CMD.motor, j, [speed ?? 0, maxForce ?? 0, speed === null ? 0 : 1, 0]); }
  setLimits(j, lo, hi) { this._cmd(CMD.limits, j, [lo ?? 0, hi ?? 0, lo === null ? 0 : 1, 0]); }
  setJointEnabled(j, on) {
    this._cmd(CMD.jointFlags, j, null, null, on ? JOINT_FLAGS.disabled : 0, on ? 0 : JOINT_FLAGS.disabled);
    if (this.joints[j]) this.joints[j].enabled = on;
  }

  // ------------------------------------------------------------------ extension: force fields
  /**
   * A WGSL snippet run for every dynamic body before each frame: it sees `b` (Body), `i` and
   * the uniforms `u` (u.dt, u.time + your own) and adds to `force` / `torque` (world, N and N·m).
   *   world.addForceField({ code: 'force += vec3f(0.0, 0.0, u.wind) * b.mass.x;', uniforms: { wind: 'f32' }, values: { wind: 2 } })
   */
  addForceField({ code, uniforms = {}, imports = [], values = {} }) {
    const k = this.ctx.kernel({
      name: 'rigid-force-field', imports: ['rigid', ...imports],
      buffers: { B: { type: 'Body', access: 'read' }, ext: { type: 'vec4f' } },
      uniforms: { dt: 'f32', time: 'f32', ...uniforms },
      code: /* wgsl */`
        let b = B[i];
        if (b.mass.x <= 0.0 || (b.info.z & F_DISABLED) != 0u) { return; }
        var force = vec3f(0.0);
        var torque = vec3f(0.0);
        ${code}
        ext[2u * i] += vec4f(force * u.dt, 0.0);
        ext[2u * i + 1u] += vec4f(torque * u.dt, 0.0);`,
    });
    k.set(values);
    const field = { kernel: k, enabled: true, set: (v) => { k.set(v); return field; } };
    this.forceFields.push(field);
    return field;
  }

  // ------------------------------------------------------------------ upload
  /** Upload bodies and joints added since the last commit (called by step). */
  commit() {
    if (!this._pending) return;
    this.ctx.flush();   // queue writes land before unsubmitted work: submit that first
    const from = this.uploaded;
    if (this.count > from) {
      const data = new Float32Array((this.count - from) * BODY_VEC4 * 4);
      for (let k = from; k < this.count; k++) data.set(this.cpu[k], (k - from) * BODY_VEC4 * 4);
      this.B.write(data, from * BODY_VEC4);
    }
    for (const id of this.dirty) if (id < from) this.B.write(this.cpu[id], id * BODY_VEC4);
    this.dirty.clear();
    if (from === 0) { for (const t of this.tbl) this.ctx.clear(t); this.ctx.clear(this.ext); }
    this.uploaded = this.count;
    if (this.jointsDirty) this._uploadJoints();
    this._pending = false;
    // Grid cells are sized for ordinary bodies; the few much larger ones (static walls, a
    // boulder) go to a short list every body tests directly, so one big body cannot make the
    // cells huge and the broadphase quadratic.
    // Only true outliers are large: bodies of similar size (a pile of mixed boxes) all share the
    // grid, because a large dynamic body costs every body a test and a reverse-list scan.
    const radii = this.cpu.map((r) => r[3]).sort((a, b) => a - b);
    const maxLarge = Math.min(MAX_LARGE, Math.ceil(0.02 * radii.length));
    const typical = radii.length ? radii[Math.max(0, radii.length - 1 - maxLarge)] : 0.01;
    let top = radii.length - 1;
    while (top > 0 && radii[top] > 2 * typical) top--;
    this.rSmall = radii.length ? radii[top] : 0.01;
    const large = [];
    this.cpu.forEach((r, k) => { if (r[3] > this.rSmall) large.push(k); });
    this.nLarge = large.length;
    if (large.length) this.large.write(new Uint32Array(large));
    // cells fit the ordinary bodies, but the grid stays under ~4M cells (wide bounds, tiny bodies)
    const [lo, hi] = this.bounds;
    const volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    const cell = Math.max(0.05, 2 * this.rSmall + this.margin, 1.01 * Math.cbrt(volume / (1 << 22)));
    if (!this.grid || Math.abs(this.grid.cellSize - cell) > 1e-6) {
      this.grid = new SpatialGrid(this.ctx, { maxCount: this.maxBodies, min: this.bounds[0], max: this.bounds[1], cellSize: cell });
      this._gridKernels();
    }
  }

  _uploadJoints() {
    const n = this.joints.length;
    // new joint records (existing ones live on the GPU: they carry multipliers and broken flags)
    if (n > this.jointsUploaded) {
      const data = new Float32Array((n - this.jointsUploaded) * JOINT_VEC4 * 4);
      for (let k = this.jointsUploaded; k < n; k++) data.set(this.joints[k].rec, (k - this.jointsUploaded) * JOINT_VEC4 * 4);
      this.J.write(data, this.jointsUploaded * JOINT_VEC4);
      this.jointsUploaded = n;
    }
    // per-body joint lists (CSR) and the set of jointed pairs that must not collide
    const nb = this.maxBodies;
    const lists = Array.from({ length: this.count }, () => []);
    this.joints.forEach((j, k) => { lists[j.a]?.push(k); if (j.b !== NONE) lists[j.b]?.push(k); });
    const csr = new Uint32Array(nb + 1 + 2 * this.maxJoints);
    let off = 0;
    for (let b = 0; b <= nb; b++) { csr[b] = off; if (b < lists.length) for (const k of lists[b]) csr[nb + 1 + off++] = k; }
    this.jcsr.write(csr);
    const T = this.jsetSize, set = new Uint32Array(2 * T);
    const hash = (x) => { x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); return (x ^ (x >>> 16)) >>> 0; };
    this.joints.forEach((j, k) => {
      if (j.b === NONE || (j.flags & JOINT_FLAGS.collide)) return;
      const key = (Math.min(j.a, j.b) * (nb + 1) + Math.max(j.a, j.b) + 1) >>> 0;
      let s = hash(key) & (T - 1);
      while (set[s] !== 0 && set[s] !== key) s = (s + 1) & (T - 1);
      set[s] = key; set[T + s] = k;
    });
    this.jset.write(set);
    this.jointsDirty = false;
  }

  clear() {
    this.count = 0; this.uploaded = 0; this.cpu = []; this.free = []; this.dirty.clear();
    this.joints = []; this.jointsUploaded = 0; this.jointsDirty = true; this.cmds = [];
    this._pending = true; this.stats = { contacts: 0, colors: 0, pairs: 0 };
  }

  // Dispatching an empty colour still costs a pass barrier (~16 µs on Apple GPUs), so only the
  // colours recently needed (+2 headroom) are dispatched. A body that would need more shares the
  // last colour until the next stats read raises the budget. Deterministic worlds always
  // dispatch every colour, so the result never depends on when stats arrived.
  _colorBudget() {
    const c = this.stats.colors;
    return this.deterministic || !(c > 0) ? MAX_COLORS : Math.min(MAX_COLORS, c + 2);
  }

  // ------------------------------------------------------------------ kernels
  _kernels() {
    const ctx = this.ctx;
    const imports = ['rigid'];
    const NB = this.maxBodies;
    this.cmdK = ctx.kernel({
      name: 'rigid-commands', imports, workgroupSize: 1, guard: false,
      buffers: { B: { type: 'Body' }, ext: { type: 'vec4f' }, cmds: { type: 'vec4f', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { count: 'u32', dt: 'f32' },
      code: /* wgsl */`
        for (var k = 0u; k < u.count; k++) {
          let h = bitcast<vec4u>(cmds[k * 3u]);
          let a = cmds[k * 3u + 1u];
          let c = cmds[k * 3u + 2u];
          let id = h.y;
          if (h.x >= 7u && h.x <= 9u) {
            var j = J[id];
            if (h.x == 7u) { j.prm.z = a.x; j.prm.w = a.y; j.ab.w = (j.ab.w & ~JF_MOTOR) | select(0u, JF_MOTOR, a.z != 0.0); }
            if (h.x == 8u) { j.ab.w = (j.ab.w & ~h.z) | h.w; }
            if (h.x == 9u) { j.axa.w = a.x; j.axb.w = a.y; j.ab.w = (j.ab.w & ~JF_LIMIT) | select(0u, JF_LIMIT, a.z != 0.0); }
            J[id] = j;
            continue;
          }
          var b = B[id];
          switch h.x {
            case 1u: { b.x = vec4f(a.xyz, b.x.w); if (a.w != 0.0) { b.q = normalize(c); } }
            case 2u: { b.v = vec4f(a.xyz, b.v.w); if (c.w != 0.0) { b.w = vec4f(c.xyz, b.w.w); } }
            case 3u: {
              ext[2u * id] += vec4f(a.xyz, 0.0);
              if (c.w == 1.0) { ext[2u * id + 1u] += vec4f(cross(c.xyz - b.x.xyz, a.xyz), 0.0); }
              if (c.w == 2.0) { ext[2u * id + 1u] += vec4f(c.xyz, 0.0); }
            }
            case 4u: { b.info.z = (b.info.z & ~h.z) | h.w; }
            case 5u: { b.info.x = h.z; b.info.y = h.w; }
            case 6u: {
              b.v = vec4f((a.xyz - b.x.xyz) / u.dt, b.v.w);
              if (a.w != 0.0) { b.w = vec4f(rotvec(normalize(c), b.q) / u.dt, b.w.w); }
            }
            case 10u: { b.props = a; }
            default: {}
          }
          B[id] = b;
        }`,
    });
    this.predictK = ctx.kernel({
      name: 'rigid-predict', imports,
      buffers: { B: { type: 'Body' }, centers: { type: 'vec4f' }, ext: { type: 'vec4f' } },
      uniforms: { dt: 'f32', g: 'vec3f' },
      code: /* wgsl */`
        var b = B[i];
        b.x0 = b.x; b.q0 = b.q;
        let J = ext[2u * i].xyz;
        let L = ext[2u * i + 1u].xyz;
        ext[2u * i] = vec4f(0.0);
        ext[2u * i + 1u] = vec4f(0.0);
        if ((b.info.z & F_DISABLED) != 0u) { centers[i] = vec4f(1e9, 1e9, 1e9, 0.0); B[i] = b; return; }
        centers[i] = vec4f(b.x.xyz, b.x.w);
        if (b.mass.x > 0.0) {
          var v = b.v.xyz + J / b.mass.x;
          var w = b.w.xyz + invInertiaWorld(b) * L;
          v /= 1.0 + u.dt * b.props.y;
          w /= 1.0 + u.dt * b.props.z;
          if ((b.info.z & F_FIXED_ROT) != 0u) { w = vec3f(0.0); }
          let g = u.g * b.props.x;
          b.y = vec4f(b.x.xyz + v * u.dt + g * (u.dt * u.dt), 0.0);
          b.qy = qinteg(b.q, w, u.dt);
          // adaptive initial guess: as much of the gravity step as the body actually took last time
          let gl = length(g);
          var aw = 0.0;
          if (gl > 0.0) { aw = clamp((dot(b.v.xyz, g / gl) - b.w.w) / (u.dt * gl), 0.0, 1.0); }
          b.v = vec4f(v, b.v.w);
          b.w = vec4f(w, b.w.w);
          b.x = vec4f(b.x.xyz + v * u.dt + g * (aw * u.dt * u.dt), b.x.w);
          b.q = b.qy;
        } else if ((b.info.z & F_KINEMATIC) != 0u) {
          b.x = vec4f(b.x.xyz + b.v.xyz * u.dt, b.x.w);
          b.q = qinteg(b.q, b.w.xyz, u.dt);
          b.y = b.x; b.qy = b.q;
        } else { b.y = b.x; b.qy = b.q; }
        B[i] = b;`,
    });
    // Joints: convert world-space anchors, warm start, and record C at the start of the step.
    this.jointPrepK = ctx.kernel({
      name: 'rigid-joint-prep', imports,
      buffers: { B: { type: 'Body', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { gammaLam: 'f32', gammaK: 'f32', kStart: 'f32', ih2: 'f32', massFloor: 'f32' },
      shared: /* wgsl */`
        // rows that are positional (stiffness scaled by mass) rather than angular (by inertia)
        fn linearRows0(t: u32) -> vec4<bool> {
          if (t == J_DISTANCE) { return vec4<bool>(true, false, false, false); }
          if (t == J_SLIDER) { return vec4<bool>(true, true, false, false); }
          return vec4<bool>(true, true, true, false);
        }
        fn linearRows1(t: u32) -> vec4<bool> { return vec4<bool>(false, t == J_SLIDER, t == J_SLIDER, false); }`,
      code: /* wgsl */`
        var j = J[i];
        if ((j.ab.w & (JF_BROKEN | JF_DISABLED)) != 0u) { return; }
        let A = B[j.ab.x];
        let hasB = j.ab.y != STATIC;
        var Bb = A;
        if (hasB) { Bb = B[j.ab.y]; }
        if ((j.ab.w & JF_WORLD_ANCHORS) != 0u) {
          let wa = j.ra.xyz; let wb = j.rb.xyz; let ax = j.axa.xyz;
          j.ra = vec4f(qrot(qconj(A.q0), wa - A.x0.xyz), j.ra.w);
          j.axa = vec4f(qrot(qconj(A.q0), ax), j.axa.w);
          if (hasB) {
            j.rb = vec4f(qrot(qconj(Bb.q0), wb - Bb.x0.xyz), j.rb.w);
            j.axb = vec4f(qrot(qconj(Bb.q0), ax), j.axb.w);
            j.qrel = qmul(qconj(A.q0), Bb.q0);
          } else { j.qrel = qconj(A.q0); }
          if (j.rb.w < 0.0) { j.rb.w = length(wa - wb); }   // distance joints: rest length = current
          j.ab.w = j.ab.w & ~JF_WORLD_ANCHORS;
          j.lam0 = vec4f(0.0); j.lam1 = vec4f(0.0);
          j.k0 = vec4f(0.0); j.k1 = vec4f(0.0);
        }
        // warm start (AVBD eq. 19) and the minimum stiffness, in units of the effective mass / inertia
        var m = A.mass.x;
        var I = (A.mass.y + A.mass.z + A.mass.w) / 3.0;
        if (hasB && Bb.mass.x > 0.0 && (m == 0.0 || Bb.mass.x < m)) { m = Bb.mass.x; I = (Bb.mass.y + Bb.mass.z + Bb.mass.w) / 3.0; }
        let kl = u.kStart * max(m, u.massFloor) * u.ih2;
        let ka = u.kStart * max(I, u.massFloor * 0.01) * u.ih2;
        j.lam0 *= u.gammaLam; j.lam1 *= u.gammaLam;
        j.k0 = max(j.k0 * u.gammaK, select(vec4f(ka), vec4f(kl), linearRows0(j.ab.z)));
        j.k1 = max(j.k1 * u.gammaK, select(vec4f(ka), vec4f(kl), linearRows1(j.ab.z)));
        // constraint values at the start of the step
        let P0A = Pose(A.x0.xyz, A.x0.xyz, A.q0, A.q0);
        var P0B = P0A;
        if (hasB) { P0B = Pose(Bb.x0.xyz, Bb.x0.xyz, Bb.q0, Bb.q0); }
        let nrow = jointRows(j.ab.z);
        var c0 = vec4f(0.0); var c1 = vec4f(0.0);
        for (var row = 0u; row < nrow; row++) {
          let C = jointRow(j, P0A, P0B, hasB, row, 0.0).C;
          if (row < 4u) { c0[row] = C; } else { c1[row - 4u] = C; }
        }
        // motor rows keep the axis coordinate itself (angle or slide) at the start of the step
        if (j.ab.z == J_HINGE) { c1.y = -hingeAngle(j, A.q0, select(vec4f(0.0, 0.0, 0.0, 1.0), Bb.q0, hasB)); }
        if (j.ab.z == J_SLIDER) {
          let aw = qrot(A.q0, j.axa.xyz);
          var pb = j.rb.xyz;
          if (hasB) { pb = Bb.x0.xyz + qrot(Bb.q0, j.rb.xyz); }
          c1.y = dot(A.x0.xyz + qrot(A.q0, j.ra.xyz) - pb, aw);
        }
        j.c00 = c0; j.c01 = c1;
        J[i] = j;`,
    });
    this.colorInitK = ctx.kernel({ name: 'rigid-color-init', buffers: { a: { type: 'u32' }, b: { type: 'u32' } }, code: 'a[i] = 0xffffffffu; b[i] = 0xffffffffu;' });
    this.bucketK = ctx.kernel({
      name: 'rigid-bucket', imports,
      buffers: { B: { type: 'Body', access: 'read' }, color: { type: 'u32', access: 'read' }, info: { type: 'u32', atomic: true } },
      code: `if (B[i].mass.x <= 0.0 || (B[i].info.z & F_DISABLED) != 0u) { return; }
        atomicAdd(&info[min(color[i], ${MAX_COLORS - 1}u) * 4u], 1u);`,
    });
    this.offsetsK = ctx.kernel({
      name: 'rigid-offsets', workgroupSize: 1, guard: false,
      buffers: { info: { type: 'u32' }, args: { type: 'u32' }, counter: { type: 'u32' }, coffset: { type: 'u32', access: 'read' } },
      uniforms: { maxContacts: 'u32', maxPairs: 'u32' },
      code: /* wgsl */`
        var off = 0u; var used = 0u;
        for (var c = 0u; c < ${MAX_COLORS}u; c++) {
          let n = info[c * 4u];
          info[c * 4u + 1u] = off; info[c * 4u + 2u] = off;
          args[c * 4u] = (n + ${BODIES_PER_WG - 1}u) / ${BODIES_PER_WG}u; args[c * 4u + 1u] = 1u; args[c * 4u + 2u] = 1u;
          off += n;
          if (n > 0u) { used = c + 1u; }
        }
        let nc = min(coffset[u.maxPairs], u.maxContacts);
        counter[1] = nc; counter[2] = used;
        args[${MAX_COLORS}u * 4u] = (nc + 63u) / 64u; args[${MAX_COLORS}u * 4u + 1u] = 1u; args[${MAX_COLORS}u * 4u + 2u] = 1u;`,
    });
    this.scatterK = ctx.kernel({
      name: 'rigid-scatter', imports,
      buffers: { B: { type: 'Body', access: 'read' }, color: { type: 'u32', access: 'read' }, info: { type: 'u32', atomic: true }, clist: { type: 'u32' } },
      code: `if (B[i].mass.x <= 0.0 || (B[i].info.z & F_DISABLED) != 0u) { return; }
        let slot = atomicAdd(&info[min(color[i], ${MAX_COLORS - 1}u) * 4u + 2u], 1u); clist[slot] = i;`,
    });
    this.pairArgsK = ctx.kernel({
      name: 'rigid-pair-args', workgroupSize: 1, guard: false,
      buffers: { poffset: { type: 'u32', access: 'read' }, args: { type: 'u32' }, counter: { type: 'u32' } },
      uniforms: { count: 'u32', maxPairs: 'u32' },
      code: /* wgsl */`
        let np = min(poffset[u.count], u.maxPairs);
        counter[3] = np;
        let o = ${MAX_COLORS + 1}u * 4u;
        args[o] = (np + 63u) / 64u; args[o + 1u] = 1u; args[o + 2u] = 1u;`,
    });
    this.narrowCountK = ctx.kernel({
      name: 'rigid-narrow-count', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, counter: { type: 'u32', access: 'read' },
        ccount: { type: 'u32' } },
      uniforms: { floorY: 'f32', margin: 'f32' },
      shared: NARROW_WGSL,
      code: /* wgsl */`
        // grid-stride over the pairs: any dispatch size covers them all (sizes are estimates)
        for (var gi = i; gi < counter[3]; gi += nwg.x * nwg.y * WG) {
          let pr = pairs[gi];
          let A = B[pr.x];
          var nc = 0u;
          if (pr.y == STATIC) { nc = floorManifold(A, u.floorY, u.margin).count; }
          else {
            let Bo = B[pr.y];
            nc = pairManifold(A, Bo, u.margin).count;
            if (((A.info.z | Bo.info.z) & F_SENSOR) != 0u) { nc = min(nc, 1u); }
          }
          ccount[gi] = nc;
        }`,
    });
    this.narrowWriteK = ctx.kernel({
      name: 'rigid-narrow-write', imports, workgroupSize: 64, guard: false,
      buffers: {
        B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, counter: { type: 'u32', access: 'read' },
        coffset: { type: 'u32', access: 'read' }, C: { type: 'Contact' }, tbl: { type: 'u32', atomic: true },
        Cp: { type: 'Contact', access: 'read' }, tblP: { type: 'u32', access: 'read' },
      },
      uniforms: { floorY: 'f32', margin: 'f32', maxContacts: 'u32', tableMask: 'u32', nBodies: 'u32',
        gammaLam: 'f32', gammaK: 'f32', kStart: 'f32', ih2: 'f32', massFloor: 'f32', stickThreshold: 'f32' },
      shared: NARROW_WGSL + /* wgsl */`
        fn pairKey(a: u32, b: u32) -> u32 { return a * (u.nBodies + 1u) + select(b, u.nBodies, b == STATIC) + 1u; }
        fn lookupPrev(key: u32) -> vec2u {
          let T = u.tableMask + 1u;
          var s = pcg(key) & u.tableMask;
          for (var probe = 0u; probe < 32u; probe++) {
            let k = tblP[s];
            if (k == key) { return vec2u(tblP[T + 2u * s], tblP[T + 2u * s + 1u]); }
            if (k == 0u) { break; }
            s = (s + 1u) & u.tableMask;
          }
          return vec2u(0u, 0u);
        }
        fn insertPair(key: u32, base: u32, n: u32) {
          let T = u.tableMask + 1u;
          var s = pcg(key) & u.tableMask;
          for (var probe = 0u; probe < 64u; probe++) {
            let r = atomicCompareExchangeWeak(&tbl[s], 0u, key);
            if (r.exchanged || r.old_value == key) {
              atomicStore(&tbl[T + 2u * s], base); atomicStore(&tbl[T + 2u * s + 1u], n);
              return;
            }
            if (r.old_value != 0u) { s = (s + 1u) & u.tableMask; }
          }
        }`,
      code: /* wgsl */`
       for (var gi = i; gi < counter[3]; gi += nwg.x * nwg.y * WG) {
        let i = gi;   // the pair this iteration writes (grid-stride: any dispatch size covers all pairs)
        let base = coffset[i];
        let n = coffset[i + 1u] - base;
        if (n == 0u || base + n > u.maxContacts) { continue; }
        let pr = pairs[i];
        let a = pr.x; let b = pr.y;
        let A = B[a];
        var M: Manifold;
        var sensor = 0u;
        var Bb = A;
        if (b == STATIC) { M = floorManifold(A, u.floorY, u.margin); }
        else { Bb = B[b]; M = pairManifold(A, Bb, u.margin); if (((A.info.z | Bb.info.z) & F_SENSOR) != 0u) { sensor = 1u; } }
        let key = pairKey(a, b);
        insertPair(key, base, n);
        let prev = lookupPrev(key);
        var meff = A.mass.x;
        var mu = A.v.w;
        if (b != STATIC) {
          mu = sqrt(mu * Bb.v.w);
          if (Bb.mass.x > 0.0 && (meff == 0.0 || Bb.mass.x < meff)) { meff = Bb.mass.x; }
        }
        let mih = max(meff, u.massFloor) * u.ih2;
        for (var k = 0u; k < n; k++) {
          let pa = M.pa[k]; let pb = M.pb[k];
          var c: Contact;
          c.ab = vec4u(a, b, 0u, sensor);
          c.n = vec4f(M.nrm[k], mu);
          let ra = qrot(qconj(A.q0), pa - A.x0.xyz);
          var rb = vec4f(pb, 0.0);
          if (b != STATIC) { rb = vec4f(qrot(qconj(Bb.q0), pb - Bb.x0.xyz), 0.0); }
          c.ra = vec4f(ra, 0.0);
          c.rb = rb;
          c.lam = vec4f(0.0);
          c.k = vec4f(u.kStart * mih, u.kStart * mih, u.kStart * mih, mih);
          // Warm start from the nearest point of the same pair last step. A rolling contact moves
          // about |Δv|·Δt over the body each step, so the search radius grows with the relative
          // speed; a one-point manifold (sphere, capsule end) is always the same contact.
          let vrel = length(A.v.xyz - select(vec3f(0.0), Bb.v.xyz, b != STATIC));
          var bestD = max(0.05 * max(A.x.w, 0.05), 1.5 * vrel / sqrt(u.ih2));
          if (n == 1u && prev.y == 1u) { bestD = 1e30; }
          for (var q = 0u; q < prev.y; q++) {
            let pc = Cp[prev.x + q];
            let dd = length(pc.ra.xyz - ra);
            if (dd < bestD) {
              bestD = dd;
              c.lam = pc.lam * u.gammaLam;
              c.k = vec4f(max(pc.k.xyz * u.gammaK, vec3f(u.kStart * mih)), mih);
              // A sticking contact keeps its anchors (static friction remembers where it started) as
              // long as the remembered material point is still the touching one. A rolling body
              // brings a new point into contact every step; its old anchor rises off the ground,
              // and keeping it would let the body sink and feed energy into the roll.
              let still = length(A.x0.xyz + qrot(A.q0, pc.ra.xyz) - pa) < u.stickThreshold;
              if (pc.ab.z != 0u && still) { c.ra = vec4f(pc.ra.xyz, 0.0); c.rb = vec4f(pc.rb.xyz, 0.0); }
              else { c.ra = vec4f(ra, 0.0); c.rb = rb; }
            }
          }
          if (sensor != 0u) { c.lam = vec4f(0.0); }
          C[base + k] = c;
        }
       }`,
    });
    // One body per 8 lanes: each lane accumulates a share of the body's contacts and joint rows,
    // a shared-memory tree sums the 8 partial Hessians/gradients, and lane 0 solves. Colours are
    // often small (tens of bodies), so spreading a body over lanes keeps the GPU occupied.
    const PRIMAL_WGSL = /* wgsl */`
        var<workgroup> red: array<array<vec4f, 9>, WG>;   // one slot per invocation (WG: the kernel's workgroup size)
        var<workgroup> colorCount: u32;
        struct Acc { Hll: mat3x3f, Hla: mat3x3f, Haa: mat3x3f, gl: vec3f, ga: vec3f }
        fn addRow(acc: ptr<function, Acc>, jl: vec3f, ja: vec3f, f: f32, kk: f32) {
          (*acc).gl += f * jl; (*acc).ga += f * ja;
          (*acc).Hll += outer3(kk * jl, jl); (*acc).Hla += outer3(kk * jl, ja); (*acc).Haa += outer3(kk * ja, ja);
        }
        fn addContact(acc: ptr<function, Acc>, ci: u32, bi: u32, me: Pose) {
          let c = C[ci];
          if (c.ab.w != 0u) { return; }   // sensor: reports overlap, exerts nothing
          let isA = c.ab.x == bi;
          let hasB = c.ab.y != STATIC;
          var L: Lin;
          if (isA) { L = contactLin(c, me, pose(select(bi, c.ab.y, hasB)), hasB); }
          else { L = contactLin(c, pose(c.ab.x), me, true); }
          let nrm = c.n.xyz;
          let r = select(L.rb0, L.ra0, isA);
          let sg = select(-1.0, 1.0, isA);
          let T = basis(nrm);
          // Forces are clamped to their bounds (normal ≤ 0, |friction| ≤ μ|λ_n|) but the penalty
          // stiffness always enters the Hessian, as in AVBD: dropping an inactive contact from H lets
          // the next Newton step jump straight through it, and the iterations then 2-cycle.
          let fN = min(0.0, c.k.x * contactCn(L, nrm, u.alpha, u.slop) + c.lam.x);
          // isotropic Coulomb cone: the tangential force is clamped to a disc of radius μ|λ_n|
          let bound = c.n.w * abs(c.lam.x);
          var ft = vec2f(c.k.y * ((1.0 - u.alpha) * dot(T[0], L.gap) + dot(T[0], L.rel)) + c.lam.y,
                         c.k.z * ((1.0 - u.alpha) * dot(T[1], L.gap) + dot(T[1], L.rel)) + c.lam.z);
          let fl = length(ft);
          if (fl > bound) { ft *= bound / max(fl, 1e-30); }
          addRow(acc, nrm * sg, cross(r, nrm) * sg, fN, c.k.x);
          addRow(acc, T[0] * sg, cross(r, T[0]) * sg, ft.x, c.k.y);
          addRow(acc, T[1] * sg, cross(r, T[1]) * sg, ft.y, c.k.z);
          // geometric stiffness bound on the angular block keeps H well conditioned under large forces
          (*acc).Haa += diag3(vec3f(abs(fN) * length(r) * u.geoStiff));
        }
        fn addJoint(acc: ptr<function, Acc>, ji: u32, bi: u32, me: Pose) {
          let jt = J[ji];
          if ((jt.ab.w & (JF_BROKEN | JF_DISABLED | JF_WORLD_ANCHORS)) != 0u) { return; }
          let isA = jt.ab.x == bi;
          let hasB = jt.ab.y != STATIC;
          var PA = me; var PB = me;
          if (isA) { if (hasB) { PB = pose(jt.ab.y); } } else { PA = pose(jt.ab.x); }
          let soft = jt.prm.x > 0.0;
          let nrow = jointRows(jt.ab.z);
          var gsum = 0.0;
          for (var row = 0u; row < nrow; row++) {
            let jr = jointRow(jt, PA, PB, hasB, row, u.dt);
            if (jr.kind == JK_OFF) { continue; }
            let jl = select(jr.lb, jr.la, isA);
            let ja = select(jr.ab, jr.aa, isA);
            let lam = rowGet(jt.lam0, jt.lam1, row);
            let kp = rowGet(jt.k0, jt.k1, row);
            let c0 = rowGet(jt.c00, jt.c01, row);
            var f: f32; var kk = kp;
            if (jr.kind == JK_MOTOR) { f = clamp(kp * jr.C + lam, -jt.prm.w, jt.prm.w); }
            else if (soft) {
              // spring: f = k C + d (C − C₀)/Δt
              kk = jt.prm.x + jt.prm.y / u.dt;
              f = jt.prm.x * jr.C + jt.prm.y * (jr.C - c0) / u.dt;
              if (jr.kind == JK_GE) { f = min(f, 0.0); }
            } else {
              let Cs = jr.C - u.alpha * select(c0, min(c0, 0.0), jr.kind == JK_GE);
              f = kp * Cs + lam;
              if (jr.kind == JK_GE) { f = min(f, 0.0); }
            }
            addRow(acc, jl, ja, f, kk);
            // lumped geometric stiffness (AVBD §3.5): |f| times the curvature of C. Positional rows bend
            // with the body's rotation (≈ |r|); distance rows also curve along the line (1/len), which is
            // what keeps a tensioned rope from overshooting sideways.
            if (dot(jl, jl) > 0.0) { gsum += abs(f) * length(ja); }
            (*acc).Hll += diag3(vec3f(abs(f) * jr.geo * u.geoStiff));
          }
          (*acc).Haa += diag3(vec3f(gsum * u.geoStiff));
        }
        // One pass over BODIES_PER_WG bodies of a colour (slots base.., of n, listed from clist[first]):
        // 8 lanes per body accumulate contact and joint rows, a shared-memory tree sums them, lane 0
        // takes the Newton step. Shared by rigid-primal (one colour, many workgroups) and
        // rigid-solve-fused (everything in one workgroup); the storage barrier makes the new poses
        // visible to the rest of the workgroup before the next colour.
        fn primalPass(base: u32, n: u32, first: u32, lidx: u32) {
        let lane = lidx % ${LANES}u;
        let slot = base + lidx / ${LANES}u;
        var bi = 0u;
        var live = false;
        var acc: Acc;
        acc.Hll = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
        acc.Hla = acc.Hll; acc.Haa = acc.Hll;
        acc.gl = vec3f(0.0); acc.ga = vec3f(0.0);
        if (slot < n) {
          bi = clist[first + slot];
          live = B[bi].mass.x > 0.0;
        }
        if (live) {
          let me = pose(bi);
          // contacts of the pairs this body owns: one contiguous range, in pair order
          let c0 = min(coffset[poffset[bi]], u.maxContacts);
          let c1 = min(coffset[poffset[bi + 1u]], u.maxContacts);
          for (var k = c0 + lane; k < c1; k += ${LANES}u) { addContact(&acc, k, bi, me); }
          // pairs it joins as the second body, in owner order
          ${reverseList(NB, 'bi')}
          for (var r = lane; r < rn; r += ${LANES}u) {
            let p = rlist[rb + r];
            let e = min(coffset[p + 1u], u.maxContacts);
            for (var k = min(coffset[p], u.maxContacts); k < e; k++) { addContact(&acc, k, bi, me); }
          }
          // joints
          let j1 = jcsr[bi + 1u];
          for (var q = jcsr[bi] + lane; q < j1; q += ${LANES}u) { addJoint(&acc, jcsr[${NB + 1}u + q], bi, me); }
        }
        red[lidx] = array<vec4f, 9>(vec4f(acc.Hll[0], acc.gl.x), vec4f(acc.Hll[1], acc.gl.y), vec4f(acc.Hll[2], acc.gl.z),
                                    vec4f(acc.Hla[0], acc.ga.x), vec4f(acc.Hla[1], acc.ga.y), vec4f(acc.Hla[2], acc.ga.z),
                                    vec4f(acc.Haa[0], 0.0), vec4f(acc.Haa[1], 0.0), vec4f(acc.Haa[2], 0.0));
        workgroupBarrier();
        for (var s = ${LANES / 2}u; s > 0u; s >>= 1u) {
          if (lane < s) { for (var e = 0u; e < 9u; e++) { red[lidx][e] += red[lidx + s][e]; } }
          workgroupBarrier();
        }
        if (live && lane == 0u) {
          let t = red[lidx];
          var b = B[bi];
          let m = b.mass.x;
          let ih2 = 1.0 / (u.dt * u.dt);
          let R = qmat(b.q);
          let Iw = R * diag3(b.mass.yzw) * transpose(R);
          // inertial terms: ½|x − y|²_M / Δt² (and the rotational analogue)
          let gL = vec3f(t[0].w, t[1].w, t[2].w) + m * ih2 * (b.x.xyz - b.y.xyz);
          let HLL = mat3x3f(t[0].xyz, t[1].xyz, t[2].xyz) + diag3(vec3f(m * ih2));
          if ((b.info.z & F_FIXED_ROT) != 0u) {
            b.x = vec4f(b.x.xyz - cholSolve(chol3(HLL), gL), b.x.w);
          } else {
            let gA = vec3f(t[3].w, t[4].w, t[5].w) + ih2 * (Iw * rotvec(b.q, b.qy));
            let HLA = mat3x3f(t[3].xyz, t[4].xyz, t[5].xyz);
            let HAA = mat3x3f(t[6].xyz, t[7].xyz, t[8].xyz) + Iw * ih2;
            let d = solveBlock6(HLL, HLA, HAA, -gL, -gA);
            b.x = vec4f(b.x.xyz + d[0], b.x.w);
            b.q = qinteg(b.q, d[1], 1.0);
          }
          B[bi] = b;
        }
        workgroupBarrier();   // red[] is rewritten by the next pass
        storageBarrier();
        }
`;
    const DUAL_WGSL = /* wgsl */`
        // The augmented-Lagrangian update of contact i after a sweep (last: the final iteration).
        fn dualContact(i: u32, last: u32) {
        var c = C[i];
        if (c.ab.w != 0u) { return; }
        let hasB = c.ab.y != STATIC;
        let L = contactLin(c, pose(c.ab.x), pose(select(c.ab.x, c.ab.y, hasB)), hasB);
        let nrm = c.n.xyz;
        let Cn = contactCn(L, nrm, u.alpha, u.slop);
        let meff = c.k.w;
        let kmax = meff * u.kmaxScale;
        c.lam.x = min(0.0, c.k.x * Cn + c.lam.x);
        if (c.lam.x < 0.0) { c.k.x = min(kmax, c.k.x + u.beta * meff * abs(Cn)); }
        if (u.tolerance > 0.0) { c.k.x = max(c.k.x, -c.lam.x / u.tolerance); }
        // friction is projected onto the cone of the updated normal force, so every stored
        // (λ_n, λ_t) satisfies |λ_t| ≤ μ|λ_n| exactly (law "contacts" in LAWS.md)
        let bound = c.n.w * abs(c.lam.x);
        let T = basis(nrm);
        let ct = vec2f((1.0 - u.alpha) * dot(T[0], L.gap) + dot(T[0], L.rel), (1.0 - u.alpha) * dot(T[1], L.gap) + dot(T[1], L.rel));
        var lt = vec2f(c.k.y, c.k.z) * ct + c.lam.yz;
        let ll = length(lt);
        let stick = ll < bound;   // inside the friction cone: static friction, stiffen
        if (stick) {
          c.k.y = min(kmax, c.k.y + u.beta * meff * abs(ct.x));
          c.k.z = min(kmax, c.k.z + u.beta * meff * abs(ct.y));
        } else { lt *= bound / max(ll, 1e-30); }   // sliding: kinetic friction on the cone
        c.lam.y = lt.x;
        c.lam.z = lt.y;
        // static friction: the anchors are kept next step while the contact sticks and has not drifted far
        if (last != 0u) {
          let drift = L.gap - nrm * dot(nrm, L.gap);
          c.ab.z = select(0u, 1u, stick && length(drift) < u.stickThreshold);
        }
        C[i] = c;
        }
`;
    const JDUAL_WGSL = /* wgsl */`
        // The joint rows' multipliers and stiffness after a sweep; breaks joints past their strength.
        fn jointDualOne(i: u32) {
        var j = J[i];
        if ((j.ab.w & (JF_BROKEN | JF_DISABLED | JF_WORLD_ANCHORS)) != 0u || j.prm.x > 0.0) { return; }
        let hasB = j.ab.y != STATIC;
        let PA = pose(j.ab.x);
        var PB = PA;
        if (hasB) { PB = pose(j.ab.y); }
        let ma = B[j.ab.x].mass;
        var m = ma.x; var I = (ma.y + ma.z + ma.w) / 3.0;
        if (hasB) { let mb = B[j.ab.y].mass; if (mb.x > 0.0 && (m == 0.0 || mb.x < m)) { m = mb.x; I = (mb.y + mb.z + mb.w) / 3.0; } }
        let nrow = jointRows(j.ab.z);
        var lam0 = j.lam0; var lam1 = j.lam1; var k0 = j.k0; var k1 = j.k1;
        for (var row = 0u; row < nrow; row++) {
          let jr = jointRow(j, PA, PB, hasB, row, u.dt);
          var lam = rowGet(lam0, lam1, row);
          var k = rowGet(k0, k1, row);
          let c0 = rowGet(j.c00, j.c01, row);
          let scale = select(max(I, u.massFloor * 0.01), max(m, u.massFloor), dot(jr.la, jr.la) + dot(jr.lb, jr.lb) > 0.0) * u.ih2;
          let kmax = scale * u.kmaxScale;
          if (jr.kind == JK_OFF) { lam = 0.0; }
          else if (jr.kind == JK_MOTOR) {
            lam = clamp(k * jr.C + lam, -j.prm.w, j.prm.w);
            if (abs(lam) < j.prm.w) { k = min(kmax, k + u.beta * scale * abs(jr.C)); }
          } else {
            let Cs = jr.C - u.alpha * select(c0, min(c0, 0.0), jr.kind == JK_GE);
            lam = k * Cs + lam;
            if (jr.kind == JK_GE) { lam = min(lam, 0.0); }
            if (jr.kind == JK_EQ || lam < 0.0) { k = min(kmax, k + u.beta * scale * abs(Cs)); }
          }
          if (row < 4u) { lam0[row] = lam; k0[row] = k; } else { lam1[row - 4u] = lam; k1[row - 4u] = k; }
        }
        j.lam0 = lam0; j.lam1 = lam1; j.k0 = k0; j.k1 = k1;
        // breakable joints: the constraint force (λ is a force) exceeded its strength
        if (j.ra.w > 0.0) {
          var fmag = length(lam0.xyz);
          if (j.ab.z == J_DISTANCE) { fmag = abs(lam0.x); }
          if (j.ab.z == J_SLIDER) { fmag = length(lam0.xy); }
          if (fmag > j.ra.w) { j.ab.w = j.ab.w | JF_BROKEN; }
        }
        J[i] = j;
        }
`;
    this.primalK = ctx.kernel({
      name: 'rigid-primal', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body' }, C: { type: 'Contact', access: 'read' }, J: { type: 'Joint', access: 'read' },
        poffset: { type: 'u32', access: 'read' }, coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' },
        jcsr: { type: 'u32', access: 'read' }, clist: { type: 'u32', access: 'read' }, info: { type: 'u32', access: 'read' } },
      uniforms: { dt: 'f32', alpha: 'f32', colorIdx: 'u32', geoStiff: 'f32', slop: 'f32', maxContacts: 'u32' },
      shared: POSE_WGSL + PRIMAL_WGSL,
      code: /* wgsl */`
        // Grid-stride over the colour's bodies, BODIES_PER_WG at a time: any dispatch size covers
        // the colour, so the CPU sizes dispatches from (stale) counts without indirect dispatches,
        // which cost ~300 µs each on some Vulkan drivers. The count comes through a uniform load
        // so the loop (and its barriers) stays in uniform control flow.
        if (lidx == 0u) { colorCount = info[u.colorIdx * 4u]; }
        let n = workgroupUniformLoad(&colorCount);
        let first = info[u.colorIdx * 4u + 1u];
        for (var base = wgi * ${BODIES_PER_WG}u; base < n; base += nwg.x * nwg.y * ${BODIES_PER_WG}u) { primalPass(base, n, first, lidx); }`,
    });
    this.dualK = ctx.kernel({
      name: 'rigid-dual', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body', access: 'read' }, C: { type: 'Contact' }, counter: { type: 'u32', access: 'read' } },
      uniforms: { alpha: 'f32', beta: 'f32', kmaxScale: 'f32', slop: 'f32', last: 'u32', stickThreshold: 'f32', tolerance: 'f32' },
      shared: POSE_WGSL + DUAL_WGSL,
      code: /* wgsl */`
        for (var gi = i; gi < counter[1]; gi += nwg.x * nwg.y * WG) { dualContact(gi, u.last); }   // grid-stride over the contacts`,
    });
    this.jointDualK = ctx.kernel({
      name: 'rigid-joint-dual', imports,
      buffers: { B: { type: 'Body', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { dt: 'f32', alpha: 'f32', beta: 'f32', kmaxScale: 'f32', ih2: 'f32', massFloor: 'f32' },
      shared: POSE_WGSL + JDUAL_WGSL,
      code: 'jointDualOne(i);',
    });
    // The whole solve of a small world in one dispatch: one workgroup runs every iteration, colour
    // by colour, then the contact and joint duals, with barriers where the multi-dispatch path has
    // dispatch boundaries. Same functions, same order — the same results up to rounding: drivers
    // compile the shared functions a little differently inside the two shaders (1–4 ulp after a
    // step). Each path is deterministic, and which one runs depends only on the body count. Below a
    // few hundred bodies it replaces ~150 tiny dispatches, each with 4–15 µs of fixed overhead.
    this.solveFusedK = ctx.kernel({
      name: 'rigid-solve-fused', imports, workgroupSize: 128, guard: false,
      buffers: { B: { type: 'Body' }, C: { type: 'Contact' }, J: { type: 'Joint' },
        poffset: { type: 'u32', access: 'read' }, coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' },
        jcsr: { type: 'u32', access: 'read' }, clist: { type: 'u32', access: 'read' }, info: { type: 'u32', access: 'read' } },
      uniforms: { dt: 'f32', alpha: 'f32', beta: 'f32', kmaxScale: 'f32', slop: 'f32', stickThreshold: 'f32', tolerance: 'f32',
        geoStiff: 'f32', maxContacts: 'u32', ih2: 'f32', massFloor: 'f32', iterations: 'u32', nColors: 'u32', nJoints: 'u32', maxPairs: 'u32' },
      shared: POSE_WGSL + PRIMAL_WGSL + DUAL_WGSL + JDUAL_WGSL,
      code: /* wgsl */`
        let nc = min(coffset[u.maxPairs], u.maxContacts);
        for (var it = 0u; it < u.iterations; it++) {
          for (var c = 0u; c < u.nColors; c++) {
            workgroupBarrier();   // everyone has read the previous colour's count (an empty colour has no other barrier)
            if (lidx == 0u) { colorCount = info[c * 4u]; }
            let n = workgroupUniformLoad(&colorCount);
            let first = info[c * 4u + 1u];
            for (var base = 0u; base < n; base += WG / ${LANES}u) { primalPass(base, n, first, lidx); }
          }
          let last = select(0u, 1u, it + 1u == u.iterations);
          for (var ci = lidx; ci < nc; ci += WG) { dualContact(ci, last); }
          for (var ji = lidx; ji < u.nJoints; ji += WG) { jointDualOne(ji); }
          storageBarrier();
        }`,
    });
    this.velocityK = ctx.kernel({
      name: 'rigid-velocity', imports,
      buffers: { B: { type: 'Body' } },
      uniforms: { dt: 'f32', g: 'vec3f' },
      code: /* wgsl */`
        var b = B[i];
        if (b.mass.x == 0.0 || (b.info.z & F_DISABLED) != 0u) { return; }
        let g = u.g * b.props.x;
        let gl = length(g);
        let vg = select(0.0, dot(b.v.xyz, g / gl), gl > 0.0);   // for the adaptive initial guess
        b.v = vec4f((b.x.xyz - b.x0.xyz) / u.dt, b.v.w);
        b.w = vec4f(rotvec(b.q, b.q0) / u.dt, vg);
        B[i] = b;`,
    });
    // order-independent 64-bit state hash (sum of per-body mixes): equal states, equal hashes
    this.hashK = ctx.kernel({
      name: 'rigid-hash', imports,
      buffers: { B: { type: 'Body', access: 'read' }, out: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let b = B[i];
        var h = pcg(i ^ 0x9e3779b9u);
        let w0 = bitcast<vec4u>(b.x); let w1 = bitcast<vec4u>(b.q); let w2 = bitcast<vec4u>(b.v); let w3 = bitcast<vec4u>(b.w);
        h = pcg(h ^ w0.x); h = pcg(h ^ w0.y); h = pcg(h ^ w0.z);
        h = pcg(h ^ w1.x); h = pcg(h ^ w1.y); h = pcg(h ^ w1.z); h = pcg(h ^ w1.w);
        h = pcg(h ^ w2.x); h = pcg(h ^ w2.y); h = pcg(h ^ w2.z);
        h = pcg(h ^ w3.x); h = pcg(h ^ w3.y); h = pcg(h ^ w3.z);
        h = pcg(h ^ b.info.z);
        atomicAdd(&out[0], h);
        atomicAdd(&out[1], pcg(h + 0x632be59bu));`,
    });
    // contact events: pairs present now but not last step (begin) and the reverse (end)
    this.eventsK = ctx.kernel({
      name: 'rigid-events', imports: ['hash'],
      buffers: { cur: { type: 'u32', access: 'read' }, prev: { type: 'u32', access: 'read' }, events: { type: 'vec4u' }, counter: { type: 'u32', atomic: true } },
      uniforms: { tableMask: 'u32', kind: 'u32', maxEvents: 'u32' },
      code: /* wgsl */`
        let key = cur[i];
        if (key == 0u) { return; }
        var s = pcg(key) & u.tableMask;
        for (var probe = 0u; probe < 64u; probe++) {
          let k = prev[s];
          if (k == key) { return; }
          if (k == 0u) { break; }
          s = (s + 1u) & u.tableMask;
        }
        let slot = atomicAdd(&counter[4], 1u);
        if (slot < u.maxEvents) { events[slot] = vec4u(key, u.kind, 0u, 0u); }`,
    });
  }

  // Kernels that read the broadphase grid (rebuilt when the cell size changes).
  _gridKernels() {
    const ctx = this.ctx;
    const NB = this.maxBodies;
    const G = this.grid.uniforms();
    const shared = /* wgsl */`
      struct Lite { x: vec4f, m: f32, info: vec4u }
      fn lite(i: u32) -> Lite { return Lite(B[i].x0, B[i].mass.x, B[i].info); }
      fn jointed(a: u32, b: u32) -> bool {
        if (u.jsetMask == 0u) { return false; }
        let key = min(a, b) * (u.nBodies + 1u) + max(a, b) + 1u;
        var x = key;
        x = (x ^ (x >> 16u)) * 0x7feb352du; x = (x ^ (x >> 15u)) * 0x846ca68bu; x = x ^ (x >> 16u);
        var s = x & u.jsetMask;
        for (var probe = 0u; probe < 64u; probe++) {
          let k = jset[s];
          if (k == 0u) { return false; }
          if (k == key) { return (J[jset[u.jsetMask + 1u + s]].ab.w & (JF_BROKEN | JF_DISABLED)) == 0u; }
          s = (s + 1u) & u.jsetMask;
        }
        return false;
      }
      fn canCollide(a: u32, la: Lite, o: u32, lo: Lite) -> bool {
        if (((la.info.z | lo.info.z) & F_DISABLED) != 0u) { return false; }
        if (la.m <= 0.0 && lo.m <= 0.0) { return false; }
        if ((la.info.x & lo.info.y) == 0u || (lo.info.x & la.info.y) == 0u) { return false; }
        if (length(lo.x.xyz - la.x.xyz) >= la.x.w + lo.x.w + u.margin) { return false; }
        return !jointed(a, o);
      }
      // bounded sorted insert: keeps the cap smallest values whatever the visiting order
      fn insertSorted(arr: ptr<function, array<u32, ${CAND}>>, n: ptr<function, u32>, v: u32, cap: u32) {
        var k = *n;
        if (k == cap) { if (v >= (*arr)[cap - 1u]) { return; } k = cap - 1u; } else { *n = k + 1u; }
        while (k > 0u && (*arr)[k - 1u] > v) { (*arr)[k] = (*arr)[k - 1u]; k--; }
        (*arr)[k] = v;
      }
      // the pairs body i owns: small bodies own pairs with larger-index small bodies and with every
      // large body; large bodies own pairs with larger-index large bodies; dynamic bodies own the floor.
      // The same walk collects a dynamic small body's lower-index small owners (its reverse list).
      fn ownedPairs(i: u32, arr: ptr<function, array<u32, ${CAND}>>, rev: ptr<function, array<u32, ${CAND}>>, nr: ptr<function, u32>) -> u32 {
        var n = 0u;
        let li = lite(i);
        if ((li.info.z & F_DISABLED) != 0u) { return 0u; }
        let pi = li.x.xyz;
        let isLarge = li.x.w > u.rSmall;
        let dyn = li.m > 0.0;
        if (u.hasFloor != 0u && dyn && pi.y - li.x.w < u.floorY + u.margin) { insertSorted(arr, &n, STATIC, ${MAXP}u); }
        if (!isLarge) {
          ${SpatialGrid.neighbors('pi', /* wgsl */`
              let o = order[j];
              if (o != i && (o > i || dyn)) {
                let lo = lite(o);
                if (lo.x.w <= u.rSmall && canCollide(i, li, o, lo)) {
                  if (o > i) { insertSorted(arr, &n, o, ${MAXP}u); } else { insertSorted(rev, nr, o, ${MAXR}u); }
                }
              }`)}
        }
        for (var k = 0u; k < u.nLarge; k++) {
          let o = large[k];
          if (o == i || (isLarge && o < i)) { continue; }
          if (canCollide(i, li, o, lite(o))) { insertSorted(arr, &n, o, ${MAXP}u); }
        }
        return n;
      }`;
    const base = {
      imports: ['rigid', 'grid'],
      uniforms: { ...G, floorY: 'f32', hasFloor: 'u32', margin: 'f32', rSmall: 'f32', nLarge: 'u32', nBodies: 'u32', jsetMask: 'u32', count: 'u32', maxPairs: 'u32' },
      shared,
    };
    const bufs = { B: { type: 'Body', access: 'read' }, order: { type: 'u32', access: 'read' }, start: { type: 'u32', access: 'read' },
      large: { type: 'u32', access: 'read' }, jset: { type: 'u32', access: 'read' }, J: { type: 'Joint', access: 'read' } };
    this.pairCountK = ctx.kernel({
      ...base, name: 'rigid-pair-find',
      buffers: { ...bufs, pcount: { type: 'u32' }, pscratch: { type: 'u32' }, rlist: { type: 'u32' } },
      code: /* wgsl */`
        var arr: array<u32, ${CAND}>;
        var rev: array<u32, ${CAND}>;
        var nr = 0u;
        let n = ownedPairs(i, &arr, &rev, &nr);
        pcount[i] = n;
        for (var k = 0u; k < n; k++) { pscratch[i * ${MAXP}u + k] = arr[k]; }
        for (var k = 0u; k < nr; k++) { rlist[i * ${MAXR}u + k] = rev[k]; }
        rlist[${NB * MAXR}u + i] = nr;
        if (i == u.count - 1u) { pcount[u.count] = 0u; }`,
    });
    this.pairWriteK = ctx.kernel({
      name: 'rigid-pair-write',
      buffers: { poffset: { type: 'u32', access: 'read' }, pscratch: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u' } },
      uniforms: { maxPairs: 'u32' },
      code: /* wgsl */`
        let o = poffset[i];
        let n = poffset[i + 1u] - o;
        for (var k = 0u; k < n; k++) { if (o + k < u.maxPairs) { pairs[o + k] = vec2u(i, pscratch[i * ${MAXP}u + k]); } }`,
    });
    // For every dynamic body, the pairs it joins as the second body, sorted by owner. The
    // owner's list is sorted, so the pair index comes from a binary search.
    const findPair = /* wgsl */`
      fn findPair(o: u32, i: u32) -> u32 {
        let e = min(poffset[o + 1u], u.maxPairs);
        var lo = poffset[o];
        var hi = e;
        while (lo < hi) { let mid = (lo + hi) / 2u; if (pairs[mid].y < i) { lo = mid + 1u; } else { hi = mid; } }
        return select(0xffffffffu, lo, lo < e && pairs[lo].y == i);
      }`;
    const revBufs = { ...bufs, poffset: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, rlist: { type: 'u32' } };
    // small bodies: rigid-pair-find left the owners in the list; turn them into pair indices in place
    this.reverseK = ctx.kernel({
      name: 'rigid-reverse-pairs',
      buffers: { B: { type: 'Body', access: 'read' }, poffset: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, rlist: { type: 'u32' } },
      uniforms: { rSmall: 'f32', maxPairs: 'u32' }, imports: ['rigid'], shared: findPair,
      code: /* wgsl */`
        if (B[i].x0.w > u.rSmall) { return; }   // large bodies: rigid-reverse-large
        let n = min(rlist[${NB * MAXR}u + i], ${MAXR}u);
        var m = 0u;
        for (var k = 0u; k < n; k++) {
          let p = findPair(rlist[i * ${MAXR}u + k], i);
          if (p != 0xffffffffu) { rlist[i * ${MAXR}u + m] = p; m++; }
        }
        rlist[${NB * MAXR}u + i] = m;`,
    });
    // A large dynamic body's owners are every small body touching it and the lower-index large
    // ones. It runs after the narrowphase and keeps only pairs with contacts: a large body's
    // bounding sphere overlaps far more bodies than actually touch it. One workgroup per large
    // body tests all bodies 256 at a time; a workgroup prefix sum places the hits, so the list
    // comes out sorted by owner without depending on thread order.
    this.reverseLargeK = ctx.kernel({
      ...base, name: 'rigid-reverse-large', workgroupSize: 256, guard: false,
      buffers: { ...revBufs, coffset: { type: 'u32', access: 'read' } },
      shared: shared + findPair + 'var<workgroup> rscan: array<u32, 256>; var<workgroup> rlive: u32; var<workgroup> rtotal: u32;',
      code: /* wgsl */`
        let b = large[wid.x];
        let base = ${NB * (MAXR + 1)}u + wid.x * ${MAXRL}u;
        if (lidx == 0u) {
          let l0 = lite(b);
          rlive = select(0u, 1u, l0.m > 0.0 && (l0.info.z & F_DISABLED) == 0u);
          rtotal = 0u;
        }
        if (workgroupUniformLoad(&rlive) == 0u) {
          if (lidx == 0u) { rlist[${NB * MAXR}u + b] = 0u; }
          return;
        }
        let li = lite(b);
        for (var c = 0u; c < u.count; c += 256u) {
          let o = c + lidx;
          var p = 0xffffffffu;
          if (o < u.count && o != b) {
            let lo = lite(o);
            if ((lo.x.w <= u.rSmall || o < b) && canCollide(b, li, o, lo)) {
              p = findPair(o, b);
              if (p != 0xffffffffu && coffset[p + 1u] == coffset[p]) { p = 0xffffffffu; }
            }
          }
          let hit = select(0u, 1u, p != 0xffffffffu);
          rscan[lidx] = hit;
          workgroupBarrier();
          for (var s = 1u; s < 256u; s <<= 1u) {
            var add = 0u;
            if (lidx >= s) { add = rscan[lidx - s]; }
            workgroupBarrier();
            rscan[lidx] += add;
            workgroupBarrier();
          }
          let before = workgroupUniformLoad(&rtotal);
          let at = before + rscan[lidx] - hit;
          if (hit == 1u && at < ${MAXRL}u) { rlist[base + at] = p; }
          workgroupBarrier();
          if (lidx == 255u) { rtotal = before + rscan[255]; }
          if (workgroupUniformLoad(&rtotal) >= ${MAXRL}u) { break; }
        }
        if (lidx == 0u) { rlist[${NB * MAXR}u + b] = min(rtotal, ${MAXRL}u) | ((wid.x + 1u) << 16u); }`,
    });
    // Jones–Plassmann + first fit over contact and joint edges. Each round reads the previous
    // round's colours only (double-buffered), so the colouring is the same whatever the thread order.
    this.colorK = ctx.kernel({
      name: 'rigid-color', imports: ['rigid'],
      buffers: { B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, poffset: { type: 'u32', access: 'read' },
        coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' }, jcsr: { type: 'u32', access: 'read' },
        J: { type: 'Joint', access: 'read' }, cin: { type: 'u32', access: 'read' }, cout: { type: 'u32' } },
      uniforms: { last: 'u32', maxColor: 'u32', maxPairs: 'u32' },
      shared: /* wgsl */`
        struct Visit { used: u32, wait: bool }
        fn visit(v: ptr<function, Visit>, i: u32, pi: u32, o: u32, self_flags: u32) {
          if (o == STATIC) { return; }
          let bo = B[o];
          if (bo.mass.x <= 0.0 || ((bo.info.z | self_flags) & (F_SENSOR | F_DISABLED)) != 0u) { return; }
          let co = cin[o];
          if (co == 0xffffffffu) {
            let po = pcg(o * 9781u + 7u);
            if ((po > pi || (po == pi && o > i)) && u.last == 0u) { (*v).wait = true; }
          } else { (*v).used |= 1u << min(co, 31u); }
        }`,
      code: /* wgsl */`
        if (cin[i] != 0xffffffffu) { cout[i] = cin[i]; return; }
        let bi = B[i];
        if (bi.mass.x <= 0.0 || (bi.info.z & F_DISABLED) != 0u) { cout[i] = 0u; return; }
        let pi = pcg(i * 9781u + 7u);
        var v: Visit;
        v.used = 0u; v.wait = false;
        let p1 = min(poffset[i + 1u], u.maxPairs);
        for (var p = min(poffset[i], u.maxPairs); p < p1; p++) { if (coffset[p + 1u] > coffset[p]) { visit(&v, i, pi, pairs[p].y, bi.info.z); } }
        ${reverseList(NB, 'i')}
        for (var r = 0u; r < rn; r++) { let p = rlist[rb + r]; if (coffset[p + 1u] > coffset[p]) { visit(&v, i, pi, pairs[p].x, bi.info.z); } }
        let j1 = jcsr[i + 1u];
        for (var q = jcsr[i]; q < j1; q++) {
          let jt = J[jcsr[${NB + 1}u + q]];
          if ((jt.ab.w & (JF_BROKEN | JF_DISABLED)) != 0u) { continue; }
          visit(&v, i, pi, select(jt.ab.x, jt.ab.y, jt.ab.x == i), 0u);
        }
        if (v.wait) { cout[i] = 0xffffffffu; return; }
        cout[i] = min(countTrailingZeros(~v.used), u.maxColor);`,
    });
  }

  // ------------------------------------------------------------------ stepping
  step(frameDt = 1 / 60) {
    this.commit();
    if (!this.count) { this.cmds = []; return; }
    const ctx = this.ctx, n = this.count;
    const fdt = Math.min(frameDt, 1 / 30);
    const dt = fdt / this.substeps;
    const nColors = this._colorBudget();
    const post = this.postStabilize, alpha = post ? 1 : this.alpha;
    const gammaLam = post ? 1 : this.alpha * this.gamma;
    const nj = this.joints.length;
    const ih2 = 1 / (dt * dt);
    this._applyCommands(fdt);
    for (const f of this.forceFields) if (f.enabled) f.kernel.run(n, { dt: fdt, time: this.frame * dt }, { B: this.B, ext: this.ext });
    const gridVals = {
      ...this.grid.values(), floorY: this.floor ?? 0, hasFloor: this.floor === null ? 0 : 1, margin: this.margin,
      rSmall: this.rSmall, nLarge: this.nLarge, nBodies: this.maxBodies, jsetMask: nj ? this.jsetSize - 1 : 0, count: n,
      maxPairs: this.maxPairs,
    };
    const gridBufs = { B: this.B, order: this.grid.order, start: this.grid.start, large: this.large, jset: this.jset, J: this.J };
    for (let s = 0; s < this.substeps; s++) {
      const cur = this.frame & 1, prev = cur ^ 1;
      ctx.region('rigid: broadphase', () => {
        ctx.clear(this.cnts); ctx.clear(this.tbl[cur]); ctx.clear(this.ccount);
        this.predictK.run(n, { dt, g: this.gravity }, { B: this.B, centers: this.centers, ext: this.ext });
        if (nj) this.jointPrepK.run(nj, { gammaLam, gammaK: this.gamma, kStart: this.kStartJoint, ih2, massFloor: this.jointMassFloor }, { B: this.B, J: this.J });
        this.grid.build(this.centers, n);
        this.pairCountK.run(n, gridVals, { ...gridBufs, pcount: this.pcount, pscratch: this.pscratch, rlist: this.rlist });
        this.pairScan.run(this.pcount, this.poffset, n + 1);
        this.pairWriteK.run(n, { maxPairs: this.maxPairs }, { poffset: this.poffset, pscratch: this.pscratch, pairs: this.pairs });
        this.pairArgsK.dispatch(1, { count: n, maxPairs: this.maxPairs }, { poffset: this.poffset, args: this.args, counter: this.cnts });
        this.reverseK.run(n, { rSmall: this.rSmall, maxPairs: this.maxPairs }, { B: this.B, poffset: this.poffset, pairs: this.pairs, rlist: this.rlist });
      });
      const W = this._sizes(n);
      ctx.region('rigid: narrowphase', () => {
        this.narrowCountK.dispatch(W.pairs, { floorY: this.floor ?? 0, margin: this.margin },
          { B: this.B, pairs: this.pairs, counter: this.cnts, ccount: this.ccount });
        this.contactScan.run(this.ccount, this.coffset, this.maxPairs + 1);
        this.narrowWriteK.dispatch(W.pairs, {
          floorY: this.floor ?? 0, margin: this.margin, maxContacts: this.maxContacts, tableMask: this.tableSize - 1,
          nBodies: this.maxBodies, gammaLam, gammaK: this.gamma, kStart: this.kStart, ih2, massFloor: this.contactMassFloor,
          stickThreshold: this.stickThreshold,
        }, { B: this.B, pairs: this.pairs, counter: this.cnts, coffset: this.coffset, C: this.C[cur], tbl: this.tbl[cur], Cp: this.C[prev], tblP: this.tbl[prev] });
        if (this.nLarge) this.reverseLargeK.dispatch([this.nLarge, 1, 1], gridVals,
          { ...gridBufs, poffset: this.poffset, pairs: this.pairs, rlist: this.rlist, coffset: this.coffset });
      });
      ctx.region('rigid: colouring', () => {
        this.colorInitK.run(n, null, { a: this.color[0], b: this.color[1] });
        const cbufs = { B: this.B, pairs: this.pairs, poffset: this.poffset, coffset: this.coffset, rlist: this.rlist, jcsr: this.jcsr, J: this.J };
        for (let r = 0; r < COLOR_ROUNDS; r++)
          this.colorK.run(n, { last: r === COLOR_ROUNDS - 1 ? 1 : 0, maxColor: nColors - 1, maxPairs: this.maxPairs },
            { ...cbufs, cin: this.color[r & 1], cout: this.color[(r + 1) & 1] });
        const fin = this.color[COLOR_ROUNDS & 1];
        ctx.clear(this.info);
        this.bucketK.run(n, null, { B: this.B, color: fin, info: this.info });
        this.offsetsK.dispatch(1, { maxContacts: this.maxContacts, maxPairs: this.maxPairs }, { info: this.info, args: this.args, counter: this.cnts, coffset: this.coffset });
        this.scatterK.run(n, null, { B: this.B, color: fin, info: this.info, clist: this.clist });
      });
      ctx.region('rigid: solve', () => {
        const pbufs = { B: this.B, C: this.C[cur], J: this.J, poffset: this.poffset, coffset: this.coffset, rlist: this.rlist,
          jcsr: this.jcsr, clist: this.clist, info: this.info };
        const sweep = (a) => {
          for (let c = 0; c < nColors; c++)
            this.primalK.dispatch(W.color(c), { dt, alpha: a, colorIdx: c, geoStiff: this.geoStiff, slop: this.slop, maxContacts: this.maxContacts }, pbufs);
        };
        if (this._useFused(n) && !post) {
          this.solveFusedK.dispatch(1, {
            dt, alpha, beta: this.beta, kmaxScale: 1e6, slop: this.slop, stickThreshold: this.stickThreshold, tolerance: this.contactTolerance,
            geoStiff: this.geoStiff, maxContacts: this.maxContacts, ih2, massFloor: this.jointMassFloor, iterations: this.iterations,
            nColors, nJoints: nj, maxPairs: this.maxPairs,
          }, { ...pbufs, C: this.C[cur] });
        } else for (let it = 0; it < this.iterations; it++) {
          sweep(alpha);
          this.dualK.dispatch(W.contacts, {
            alpha, beta: this.beta, kmaxScale: 1e6, slop: this.slop, last: it === this.iterations - 1 ? 1 : 0, stickThreshold: this.stickThreshold,
            tolerance: this.contactTolerance,
          }, { B: this.B, C: this.C[cur], counter: this.cnts });
          if (nj) this.jointDualK.run(nj, { dt, alpha, beta: this.beta, kmaxScale: 1e6, ih2, massFloor: this.jointMassFloor }, { B: this.B, J: this.J });
        }
        this.velocityK.run(n, { dt, g: this.gravity }, { B: this.B });
        if (post) sweep(0);   // remove the remaining overlap / joint drift without touching velocities
      });
      if (this.events) {
        const T = this.tableSize;
        this.eventsK.run(T, { tableMask: T - 1, kind: 0, maxEvents: this.maxEvents }, { cur: this.tbl[cur], prev: this.tbl[prev], events: this.eventBuf, counter: this.cnts });
        this.eventsK.run(T, { tableMask: T - 1, kind: 1, maxEvents: this.maxEvents }, { cur: this.tbl[prev], prev: this.tbl[cur], events: this.eventBuf, counter: this.cnts });
      }
      this.frame++;
    }
    // refresh the stats (colour budget, HUD) every 30 steps without stalling
    if (++this._steps % 30 === 1 && !this._statsBusy) {
      this._statsBusy = true;
      this.readStats().finally(() => { this._statsBusy = false; });
    }
  }

  _applyCommands(dt) {
    if (!this.cmds.length) return;
    const cmds = this.cmds;
    this.cmds = [];
    for (let o = 0; o < cmds.length; o += this.maxCommands) {
      const chunk = cmds.slice(o, o + this.maxCommands);
      const data = new Float32Array(chunk.length * 12);
      const u = new Uint32Array(data.buffer);
      chunk.forEach((c, k) => {
        u[k * 12] = c.type; u[k * 12 + 1] = c.id; u[k * 12 + 2] = c.hz >>> 0; u[k * 12 + 3] = c.hw >>> 0;
        data.set(c.a, k * 12 + 4); data.set(c.c, k * 12 + 8);
      });
      this.ctx.flush();   // the buffer write must not overtake earlier unsubmitted work
      this.cmdBuf.write(data);
      this.cmdK.dispatch(1, { count: chunk.length, dt }, { B: this.B, ext: this.ext, cmds: this.cmdBuf, J: this.J });
    }
  }

  // ------------------------------------------------------------------ readback & queries
  /** Asynchronously refresh this.stats = { contacts, colors, pairs } (for HUDs). */
  async readStats() {
    const [c, info] = await Promise.all([this.cnts.read(8), this.info.read(MAX_COLORS * 4)]);
    this.stats = { contacts: c[1], colors: c[2], pairs: c[3] };
    this._est = { pairs: c[3], contacts: c[1], colors: Array.from({ length: MAX_COLORS }, (_, k) => info[k * 4]) };
    return this.stats;
  }

  // The fused single-workgroup solve pays off while one workgroup sweeps the world faster than
  // ~(colours + 2) × iterations dispatches can be issued: measured crossovers ~130 bodies on an M2 Max
  // (Metal, ~4 µs per dispatch) and ~230 on a GB10 (Vulkan, ~15 µs), hence 128 by default. It needs
  // 9 storage buffers in one shader stage (the WebGPU default limit is 8).
  _useFused(n) {
    const on = this.fusedSolve;
    if (on === false) return false;
    if (this.ctx.limits.maxStorageBuffersPerShaderStage < 9) return false;
    return on === true || n <= on;
  }

  // Workgroup counts for the grid-stride kernels, from the latest (asynchronously read) counts.
  // They only set how much of the GPU a dispatch occupies: every kernel loops until it has covered
  // the real count on the GPU, so a stale or missing estimate costs time, never correctness.
  _sizes(n) {
    const e = this._est;
    const fit = (x, max) => Math.max(1, Math.min(max, Math.ceil(x)));
    return {
      pairs: fit((e ? e.pairs : 6 * n) * 1.25 / 64, Math.ceil(this.maxPairs / 64)),
      contacts: fit((e ? e.contacts : 4 * n) * 1.25 / 64, Math.ceil(this.maxContacts / 64)),
      color: (c) => fit((e ? e.colors[c] : n) * 1.25 / BODIES_PER_WG + 1, Math.ceil(n / BODIES_PER_WG)),
    };
  }

  /** Body poses: Float32Array of [x, y, z, qx, qy, qz, qw] per body. */
  async readPoses() {
    const raw = await this.B.read(this.count * BODY_VEC4);
    const out = new Float32Array(this.count * 7);
    for (let k = 0; k < this.count; k++) {
      out.set(raw.subarray(k * 48, k * 48 + 3), k * 7);
      out.set(raw.subarray(k * 48 + 4, k * 48 + 8), k * 7 + 3);
    }
    return out;
  }

  /** Full body state: { position, rotation, velocity, angularVelocity, flags } typed arrays. */
  async readBodies() {
    const raw = await this.B.read(this.count * BODY_VEC4);
    const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
    const n = this.count;
    const out = { position: new Float32Array(3 * n), rotation: new Float32Array(4 * n), velocity: new Float32Array(3 * n),
      angularVelocity: new Float32Array(3 * n), flags: new Uint32Array(n) };
    for (let k = 0; k < n; k++) {
      const o = k * 48;
      out.position.set(raw.subarray(o, o + 3), 3 * k);
      out.rotation.set(raw.subarray(o + 4, o + 8), 4 * k);
      out.velocity.set(raw.subarray(o + 8, o + 11), 3 * k);
      out.angularVelocity.set(raw.subarray(o + 12, o + 15), 3 * k);
      out.flags[k] = u[o + 46];
    }
    return out;
  }

  /** Joint state: { broken: bool[], force: Float32Array (|λ| of the positional rows, N) }. */
  async readJoints() {
    const n = this.joints.length;
    if (!n) return { broken: [], force: new Float32Array(0) };
    const raw = await this.J.read(n * JOINT_VEC4);
    const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
    const broken = [], force = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const o = k * JOINT_VEC4 * 4;
      broken.push((u[o + 3] & JOINT_FLAGS.broken) !== 0);
      force[k] = Math.hypot(raw[o + 28], raw[o + 29], raw[o + 30]);
    }
    return { broken, force };
  }

  /** Contact begin/end events of the last step (needs { events: true }): { begin: [[a, b], ...], end: [...] }, b = -1 for the floor. */
  async readEvents() {
    const c = await this.cnts.read(8);
    const n = Math.min(c[4], this.maxEvents);
    const out = { begin: [], end: [] };
    if (!n) return out;
    const raw = await this.eventBuf.read(n);
    const N = this.maxBodies + 1;
    for (let k = 0; k < n; k++) {
      const key = raw[4 * k] - 1, a = Math.floor(key / N), b = key % N;
      const pair = b === this.maxBodies ? [a, -1] : [Math.min(a, b), Math.max(a, b)];
      (raw[4 * k + 1] ? out.end : out.begin).push(pair);
    }
    const order = (x, y) => x[0] - y[0] || x[1] - y[1];
    out.begin.sort(order); out.end.sort(order);
    return out;
  }

  /** A reusable GPU ray query (see RayQuery): rays can be written by the CPU or by other kernels. */
  rayQuery(maxRays = 1024) { return new RayQuery(this, maxRays); }

  /** Cast rays [{ origin, dir, maxDistance, mask }] → [{ hit, body, distance, point, normal }]. */
  async raycast(rays) {
    if (!this._rq || this._rq.maxRays < rays.length) this._rq = new RayQuery(this, Math.max(64, rays.length));
    this._rq.set(rays).run(rays.length);
    return this._rq.read(rays.length);
  }

  /** 64-bit hash of every body's pose and velocity (hex string); equal states give equal hashes. */
  async hash() {
    this.hashRecord();
    const h = await this.hashBuf.read(2);
    return h[1].toString(16).padStart(8, '0') + h[0].toString(16).padStart(8, '0');
  }
  /**
   * Record the hash computation now (state at this point of the command stream) into a u32×2
   * buffer — of the live bodies, or of a snapshot's.
   */
  hashRecord(target = this.hashBuf, snap = null) {
    const n = snap ? snap.meta.count : this.count;
    this.ctx.clear(target);
    if (n) this.hashK.run(n, null, { B: snap ? snap.B : this.B, out: target });
    return target;
  }

  /**
   * The complete simulation state as typed arrays (save games, network resync) — of the live
   * world, or of a snapshot (e.g. a rollback session's last confirmed tick):
   * { frame, count, bodies, joints, ext, contacts, slots, meta }. importState() restores it exactly;
   * the warm-start table travels as its occupied slots only.
   */
  async exportState(snap = null) {
    this.commit();
    // Read from a private copy, recorded now: the live world (or a snapshot in a rollback ring)
    // may move on while the reads are in flight.
    const f = snap ? this._cloneSnapshot(snap) : this.snapshot();
    const m = f.meta;
    try {
      const [cnt, bodies, joints, ext, table] = await Promise.all([
        f.cnt.read(8),
        m.count ? f.B.read(m.count * BODY_VEC4) : new Float32Array(0),
        m.joints.length ? f.J.read(m.joints.length * JOINT_VEC4) : new Float32Array(0),
        m.count ? f.ext.read(m.count * 2) : new Float32Array(0),
        f.tbl.read(this.tableSize * 3),
      ]);
      const nc = Math.min(cnt[1], this.maxContacts);
      const contacts = nc ? await f.C.read(nc * CONTACT_VEC4) : new Float32Array(0);
      const T = this.tableSize, slots = [];
      for (let k = 0; k < T; k++) if (table[k]) slots.push(k, table[k], table[T + 2 * k], table[T + 2 * k + 1]);
      return {
        version: 2, frame: m.frame, count: m.count, contactCount: nc, tableSize: T,
        bodies, joints, ext, contacts, slots: Uint32Array.from(slots),
        // records are copied with slice(): a byte copy, so u32 words stored in float arrays keep their bits
        meta: { free: m.free.slice(), joints: m.joints.map((j) => ({ a: j.a, b: j.b, type: j.type, flags: j.flags, rec: j.rec.slice(), enabled: j.enabled })) },
      };
    } finally {
      for (const b of [f.B, f.J, f.ext, f.C, f.tbl, f.cnt]) b.destroy();
    }
  }

  _cloneSnapshot(s) {
    const ctx = this.ctx, m = s.meta;
    const c = {
      B: ctx.array('vec4f', Math.max(1, m.count) * BODY_VEC4, { label: 'export-bodies' }),
      J: ctx.array('vec4f', Math.max(1, m.joints.length) * JOINT_VEC4, { label: 'export-joints' }),
      ext: ctx.array('vec4f', Math.max(1, m.count) * 2, { label: 'export-ext' }),
      C: ctx.array('vec4f', this.maxContacts * CONTACT_VEC4, { label: 'export-contacts' }),
      tbl: ctx.array('u32', this.tableSize * 3, { label: 'export-table' }),
      cnt: ctx.array('u32', 8, { label: 'export-counts' }),
      meta: m,
    };
    if (m.count) { ctx.copy(s.B, c.B, m.count * BODY_VEC4 * 16); ctx.copy(s.ext, c.ext, m.count * 32); }
    if (m.joints.length) ctx.copy(s.J, c.J, m.joints.length * JOINT_VEC4 * 16);
    ctx.copy(s.C, c.C);
    ctx.copy(s.tbl, c.tbl);
    ctx.copy(s.cnt, c.cnt, 32);
    return c;
  }

  importState(st) {
    const ctx = this.ctx;
    let table = st.table;
    if (st.slots) {
      if (st.tableSize !== this.tableSize) throw new Error(`RigidWorld.importState: table size ${st.tableSize} ≠ ${this.tableSize} (create both worlds with the same maxBodies / maxContacts)`);
      const T = this.tableSize;
      table = new Uint32Array(T * 3);
      for (let k = 0; k < st.slots.length; k += 4) { const i = st.slots[k]; table[i] = st.slots[k + 1]; table[T + 2 * i] = st.slots[k + 2]; table[T + 2 * i + 1] = st.slots[k + 3]; }
    }
    this.cmds = [];
    this.frame = st.frame; this.count = st.count; this.uploaded = st.count; this.free = st.meta.free.slice();
    this.cpu = [];
    for (let k = 0; k < st.count; k++) this.cpu.push(st.bodies.slice(k * BODY_VEC4 * 4, (k + 1) * BODY_VEC4 * 4));
    this.dirty.clear();
    this.joints = st.meta.joints.map((j) => ({ ...j, rec: j.rec instanceof Float32Array ? j.rec.slice() : Float32Array.from(j.rec) }));
    this.jointsUploaded = this.joints.length;
    const last = (this.frame + 1) & 1;
    ctx.clear(this.C[last]); ctx.clear(this.tbl[last]);
    ctx.flush();   // the clears (and earlier work) must run before the queue writes below
    if (st.count) { this.B.write(st.bodies); this.ext.write(st.ext); }
    if (this.joints.length) this.J.write(st.joints);
    if (st.contactCount) this.C[last].write(st.contacts);
    this.tbl[last].write(table);
    this.cnts.write(new Uint32Array([0, st.contactCount || 0, 0, 0, 0, 0, 0, 0]));
    this.jointsDirty = true;
    this._pending = true;
    this.uploaded = this.count;   // bodies are on the GPU already; commit() only rebuilds lists and the grid
  }

  /**
   * Capture everything the next step depends on: bodies, joints, pending impulses, last step's
   * contacts (warm start) and the CPU bookkeeping. Pass a previous snapshot to reuse its buffers.
   */
  snapshot(into) {
    this.commit();
    const ctx = this.ctx;
    const s = into || {
      B: ctx.array('vec4f', this.maxBodies * BODY_VEC4, { label: 'snap-bodies' }),
      J: ctx.array('vec4f', this.maxJoints * JOINT_VEC4, { label: 'snap-joints' }),
      ext: ctx.array('vec4f', this.maxBodies * 2, { label: 'snap-ext' }),
      C: ctx.array('vec4f', this.maxContacts * CONTACT_VEC4, { label: 'snap-contacts' }),
      tbl: ctx.array('u32', this.tableSize * 3, { label: 'snap-table' }),
      cnt: ctx.array('u32', 8, { label: 'snap-counts' }),
    };
    if (!s.cnt) s.cnt = ctx.array('u32', 8, { label: 'snap-counts' });
    const last = (this.frame + 1) & 1;   // the buffers the next step reads as "previous"
    if (this.count) {
      ctx.copy(this.B, s.B, this.count * BODY_VEC4 * 16);
      ctx.copy(this.ext, s.ext, this.count * 32);
    }
    if (this.joints.length) ctx.copy(this.J, s.J, this.joints.length * JOINT_VEC4 * 16);
    ctx.copy(this.C[last], s.C);
    ctx.copy(this.tbl[last], s.tbl);
    ctx.copy(this.cnts, s.cnt, 32);   // contact count of the step that produced those contacts (for exports)
    s.meta = { frame: this.frame, count: this.count, cpu: this.cpu.slice(), free: this.free.slice(), joints: this.joints.slice(),
      steps: this._steps, stats: { ...this.stats }, cmds: this.cmds.slice() };
    return s;
  }

  /** Return to a snapshot (bodies and joints added since are dropped). */
  restore(s) {
    const ctx = this.ctx, m = s.meta;
    this.commit();
    this.cmds = m.cmds.slice();
    this.frame = m.frame; this.count = m.count; this.uploaded = m.count; this.cpu = m.cpu.slice(); this.free = m.free.slice();
    this.dirty.clear();
    if (this.joints.length !== m.joints.length) { this.joints = m.joints.slice(); this.jointsUploaded = this.joints.length; this.jointsDirty = true; this._pending = true; }
    this._steps = m.steps; this.stats = { ...m.stats };
    const last = (this.frame + 1) & 1;
    if (this.count) {
      ctx.copy(s.B, this.B, this.count * BODY_VEC4 * 16);
      ctx.copy(s.ext, this.ext, this.count * 32);
    }
    if (this.joints.length) ctx.copy(s.J, this.J, this.joints.length * JOINT_VEC4 * 16);
    ctx.copy(s.C, this.C[last]);
    ctx.copy(s.tbl, this.tbl[last]);
    if (s.cnt) ctx.copy(s.cnt, this.cnts, 32);   // so an export right after this describes the restored contacts
    this._pending = true;   // re-derive the large-body list for the restored set of bodies
  }
}

/**
 * Batched GPU ray casts against a RigidWorld. Every (body, ray) pair is one thread; the nearest
 * hit per ray is found with two integer atomicMin passes (distance bits, then body id), so the
 * result does not depend on thread order.
 *   rays: vec4f × 2 per ray — (origin, maxDistance), (direction, mask bits)
 *   hits: vec4f × 2 per ray — (distance, or -1 for a miss; body index as u32 bits, -, -), (normal, -)
 */
class RayQuery {
  constructor(world, maxRays) {
    const ctx = world.ctx;
    this.world = world;
    this.maxRays = maxRays;
    this.rays = ctx.array('vec4f', maxRays * 2, { label: 'ray-query-rays' });
    this.hits = ctx.array('vec4f', maxRays * 2, { label: 'ray-query-hits' });
    this.best = ctx.array('u32', maxRays * 2, { label: 'ray-query-best' });
    const common = { imports: ['rigid'], shared: RAY_WGSL, uniforms: { phase: 'u32' } };
    this.testK = ctx.kernel({
      ...common, name: 'ray-test', dims: 2, workgroupSize: [64, 1, 1],
      buffers: { B: { type: 'Body', access: 'read' }, rays: { type: 'vec4f', access: 'read' }, best: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let bi = p.x; let ri = p.y;
        let b = B[bi];
        if ((b.info.z & (F_DISABLED | F_SENSOR)) != 0u) { return; }
        let o = rays[2u * ri]; let dr = rays[2u * ri + 1u];
        if ((b.info.x & bitcast<u32>(dr.w)) == 0u) { return; }
        // bounding sphere first
        let oc = o.xyz - b.x.xyz;
        let bb = dot(oc, dr.xyz);
        if (bb * bb - dot(oc, oc) + b.x.w * b.x.w < 0.0) { return; }
        let h = rayBody(o.xyz, dr.xyz, b);
        if (h.x < 0.0 || h.x > o.w) { return; }
        if (u.phase == 0u) { atomicMin(&best[2u * ri], bitcast<u32>(h.x)); }
        else if (atomicLoad(&best[2u * ri]) == bitcast<u32>(h.x)) { atomicMin(&best[2u * ri + 1u], bi); }`,
    });
    this.resolveK = ctx.kernel({
      ...common, name: 'ray-resolve',
      buffers: { B: { type: 'Body', access: 'read' }, rays: { type: 'vec4f', access: 'read' }, best: { type: 'u32', access: 'read' }, hits: { type: 'vec4f' } },
      code: /* wgsl */`
        let bi = best[2u * i + 1u];
        if (bi == 0xffffffffu) { hits[2u * i] = vec4f(-1.0, 0.0, 0.0, 0.0); hits[2u * i + 1u] = vec4f(0.0); return; }
        let o = rays[2u * i]; let dr = rays[2u * i + 1u];
        let h = rayBody(o.xyz, dr.xyz, B[bi]);
        hits[2u * i] = vec4f(h.x, bitcast<f32>(bi), 0.0, 0.0);
        hits[2u * i + 1u] = vec4f(h.yzw, 0.0);`,
    });
  }
  /** Upload rays [{ origin, dir, maxDistance, mask }]. */
  set(list) {
    const data = new Float32Array(list.length * 8);
    const u = new Uint32Array(data.buffer);
    list.forEach((r, k) => {
      const d = r.dir, l = Math.hypot(...d) || 1;
      data.set([...r.origin, r.maxDistance ?? 1e6], 8 * k);
      data.set([d[0] / l, d[1] / l, d[2] / l], 8 * k + 4);
      u[8 * k + 7] = (r.mask ?? 0xffffffff) >>> 0;
    });
    this.world.ctx.flush();
    this.rays.write(data);
    return this;
  }
  /** Record the query for `count` rays (rays may have been written by GPU kernels). */
  run(count = this.maxRays) {
    const w = this.world, ctx = w.ctx;
    if (!w.count || !count) return this;
    ctx.fill(this.best, 0xffffffff);
    const bufs = { B: w.B, rays: this.rays, best: this.best };
    this.testK.run([w.count, count], { phase: 0 }, bufs);
    this.testK.run([w.count, count], { phase: 1 }, bufs);
    this.resolveK.run(count, { phase: 0 }, { ...bufs, hits: this.hits });
    return this;
  }
  /** Read hits: [{ hit, body, distance, normal, point }]. */
  async read(count = this.maxRays) {
    const h = await this.hits.read(count * 2);
    const r = await this.rays.read(count * 2);
    const u = new Uint32Array(h.buffer, h.byteOffset, h.length);
    const out = [];
    for (let k = 0; k < count; k++) {
      const t = h[8 * k];
      if (!(t >= 0)) { out.push({ hit: false, body: -1, distance: Infinity }); continue; }
      const o = r.subarray(8 * k, 8 * k + 3), d = r.subarray(8 * k + 4, 8 * k + 7);
      out.push({ hit: true, body: u[8 * k + 1], distance: t, normal: [h[8 * k + 4], h[8 * k + 5], h[8 * k + 6]],
        point: [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t] });
    }
    return out;
  }
}

function eulerQuat([x, y, z]) {
  const cx = Math.cos(x / 2), sx = Math.sin(x / 2), cy = Math.cos(y / 2), sy = Math.sin(y / 2), cz = Math.cos(z / 2), sz = Math.sin(z / 2);
  return [sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz];
}
return { RIGID_SHAPES: SHAPES, RIGID_FLAGS: FLAGS, JOINT_TYPES, JOINT_FLAGS, RigidWorld, RayQuery, eulerQuat };
})();

// ---- src/render/bodies.js
const __m35 = (() => {
const RenderProgram = __m11.RenderProgram;
// Instanced renderer for rigid bodies stored in a RigidWorld body buffer
// (12 vec4 per body: x at 0, q at 1, size at 8, mass at 9, info at 11). Every body
// is drawn as its oriented bounding box (36 vertices); boxes shade the faces
// directly, spheres and capsules ray-cast the exact surface inside that box and
// write the true depth. Disabled bodies are not drawn.


class Bodies {
  constructor(ctx, world, o = {}) {
    this.world = world;
    this.program = new RenderProgram(ctx, {
      name: 'rigid-bodies', imports: ['pbr', 'color', 'math', 'hash'],
      buffers: { P: { type: 'vec4f' } },
      uniforms: { sun: 'vec3f', staticColor: 'vec3f', saturation: 'f32', kinematicColor: 'vec3f', groupMask: 'u32' },
      code: /* wgsl */`
        struct VO {
          @builtin(position) pos: vec4f,
          @location(0) w: vec3f,
          @location(1) n: vec3f,
          @location(2) @interpolate(flat) col: vec3f,
          @location(3) @interpolate(flat) c: vec4f,     // centre, radius
          @location(4) @interpolate(flat) q: vec4f,     // orientation
          @location(5) @interpolate(flat) sh: vec2f,    // (shape, half length)
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
          let x = P[ii * 12u];
          let q = P[ii * 12u + 1u];
          let sz = P[ii * 12u + 8u];
          let m = P[ii * 12u + 9u];
          let info = bitcast<vec4u>(P[ii * 12u + 11u]);
          let flags = info.z;
          var o: VO;
          // disabled bodies, and bodies outside the drawn collision groups, are not drawn
          if ((flags & 4u) != 0u || (info.x & u.groupMask) == 0u) { o.pos = vec4f(0.0, 0.0, -2.0, 1.0); return o; }
          let shape = u32(sz.w + 0.5);
          var ext = sz.xyz;
          if (shape == 2u) { ext = vec3f(sz.x, sz.y + sz.x, sz.x); }
          let f = vi / 6u;
          let a = f / 2u;
          let s = select(-1.0, 1.0, (f & 1u) == 0u);
          var uv = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
          let t = uv[vi % 6u];
          var lp = vec3f(0.0);
          lp[a] = s; lp[(a + 1u) % 3u] = t.x; lp[(a + 2u) % 3u] = t.y;
          var ln = vec3f(0.0); ln[a] = s;
          let world = x.xyz + qrot(q, lp * ext);
          var col = u.staticColor;
          if (m.x > 0.0) {
            let h = pcg(ii * 2654435761u + 17u);
            col = srgbToLinear(palette(u2f(h), vec3f(0.55), vec3f(0.35 * u.saturation), vec3f(1.0), vec3f(0.0, 0.33, 0.67)));
          } else if ((flags & 1u) != 0u) { col = u.kinematicColor; }
          o.pos = cam.viewProj * vec4f(world, 1.0);
          o.w = world; o.n = qrot(q, ln); o.col = col;
          o.c = vec4f(x.xyz, sz.x); o.q = q; o.sh = vec2f(f32(shape), sz.y);
          return o;
        }
        struct FO { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 }
        @fragment fn fs(in: VO) -> FO {
          var n = normalize(in.n);
          var p = in.w;
          let shape = u32(in.sh.x + 0.5);
          if (shape != 1u) {
            let rd = normalize(in.w - cam.eye);
            // ray in the body frame
            let iq = vec4f(-in.q.xyz, in.q.w);
            let ro = qrot(iq, cam.eye - in.c.xyz);
            let rl = qrot(iq, rd);
            let r = in.c.w;
            let h = select(0.0, in.sh.y, shape == 2u);
            // capsule (a sphere when h = 0): segment (0,-h,0)-(0,h,0), radius r (Quilez)
            let pa = vec3f(0.0, -h, 0.0);
            let ba = vec3f(0.0, 2.0 * h, 0.0);
            let oa = ro - pa;
            let baba = dot(ba, ba); let bard = dot(ba, rl); let baoa = dot(ba, oa);
            let rdoa = dot(rl, oa); let oaoa = dot(oa, oa);
            var t = -1.0;
            if (baba > 1e-12) {
              let a = baba - bard * bard;
              let b = baba * rdoa - baoa * bard;
              let c = baba * oaoa - baoa * baoa - r * r * baba;
              let disc = b * b - a * c;
              if (disc >= 0.0 && a > 1e-12) {
                let tc = (-b - sqrt(disc)) / a;
                let y = baoa + tc * bard;
                if (y > 0.0 && y < baba) { t = tc; }
              }
            }
            if (t < 0.0) {
              // end spheres (the sphere itself when h = 0): nearest hit in front of the eye
              var best = 1e30;
              for (var e = 0; e < 2; e++) {
                let oc = ro - select(pa + ba, pa, e == 0);
                let b = dot(rl, oc);
                let disc = b * b - dot(oc, oc) + r * r;
                if (disc >= 0.0) { let te = -b - sqrt(disc); if (te > 0.0 && te < best) { best = te; } }
              }
              if (best > 1e29) { discard; }
              t = best;
            }
            let lp = ro + rl * t;
            let hh = select(0.0, clamp(dot(lp - pa, ba) / baba, 0.0, 1.0), baba > 1e-12);
            let ln = (lp - pa - hh * ba) / r;
            p = cam.eye + rd * t;
            n = qrot(in.q, ln);
          }
          let v = normalize(cam.eye - p);
          let ao = 0.75 + 0.25 * clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
          let c = shadePBR(n, v, in.col, 0.0, 0.55, u.sun, vec3f(2.8), 1.0, ao);
          let clip = cam.viewProj * vec4f(p, 1.0);
          return FO(vec4f(c, 1.0), clip.z / clip.w);
        }`,
    });
    const sun = o.sun || [0.45, 0.8, 0.35], l = Math.hypot(...sun);
    this.program.set({ sun: sun.map((x) => x / l), staticColor: o.staticColor || [0.25, 0.26, 0.28], saturation: o.saturation ?? 1,
      kinematicColor: o.kinematicColor || [0.55, 0.42, 0.22], groupMask: (o.groups ?? 0xffffffff) >>> 0 });
  }
  set(v) { this.program.set(v); return this; }
  draw(pass, view) {
    const n = this.world.count;
    if (n) this.program.draw(pass, view, 36, n, null, { P: this.world.B });
  }
}
return { Bodies };
})();

// ---- src/nn/gemm.js
const __m37 = (() => {
// General matrix multiply on the GPU: C = act(α op(A) op(B) + β C + bias), f32, row-major.
//
// Each workgroup computes a BM×BN tile of C. The K dimension is walked in slices of BK:
// the A and B slices are staged in shared memory, then every thread accumulates a TM×TN
// block of outputs in registers (TM·TN FMAs per TM+TN shared-memory reads). Transposes are
// folded into the loads, so the backward pass (Xᵀ·dZ, dZ·Wᵀ) needs no transpose kernel.
// The epilogue fuses α, β·C, a bias row and the activation. Configurations are autotuned
// per device (the best tile depends on the GPU's register file and shared memory).
//
// The certificate is Freivalds' check: for a random vector r, A(Br) = Cr up to rounding.
// It costs three matrix–vector products, O(n²), against the O(n³) multiply, and a wrong C
// passes with probability ≤ 1/2 per random ±1 vector (see verify.js).

const ACTIVATIONS = {
  none: { f: 'x', df: '1.0' },
  relu: { f: 'max(x, 0.0)', df: 'select(0.0, 1.0, x > 0.0)' },
  leaky: { f: 'select(0.01 * x, x, x > 0.0)', df: 'select(0.01, 1.0, x > 0.0)' },
  tanh: { f: 'tanh(x)', df: '1.0 - tanh(x) * tanh(x)' },
  sigmoid: { f: '1.0 / (1.0 + exp(-x))', df: '(1.0 / (1.0 + exp(-x))) * (1.0 - 1.0 / (1.0 + exp(-x)))' },
  sin: { f: 'sin(x)', df: 'cos(x)' },
  gelu: { f: '0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))',
    df: '0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + 0.5 * x * (1.0 - tanh(0.7978845608 * (x + 0.044715 * x * x * x)) * tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.7978845608 * (1.0 + 0.134145 * x * x)' },
};

const GEMM_CONFIGS = [
  { BM: 64, BN: 64, BK: 16, TM: 4, TN: 4 },     // 3.16 TFLOP/s at 2048³ on an M2 Max (default)
  { BM: 64, BN: 64, BK: 16, TM: 8, TN: 4 },
  { BM: 128, BN: 64, BK: 16, TM: 8, TN: 4 },
  { BM: 64, BN: 128, BK: 16, TM: 4, TN: 8 },
  { BM: 64, BN: 32, BK: 16, TM: 4, TN: 4 },
  { BM: 128, BN: 128, BK: 8, TM: 8, TN: 8 },
];

const cache = new Map();
const R = (n) => Array.from({ length: n }, (_, k) => k);

/** The kernel for one (config, transA, transB, activation, bias, beta) combination. */
function gemmKernel(ctx, cfg, { transA = false, transB = false, act = 'none', bias = false, accumulate = false, split = false } = {}) {
  const key = `${cfg.BM}x${cfg.BN}x${cfg.BK}/${cfg.TM}x${cfg.TN}${cfg.prefetch ? 'p' : ''}/${+transA}${+transB}/${act}/${+bias}/${+accumulate}/${+split}`;
  const perCtx = cache.get(ctx) || new Map();
  cache.set(ctx, perCtx);
  if (perCtx.has(key)) return perCtx.get(key);
  const { BM, BN, BK, TM, TN } = cfg;
  const TX = BN / TN, TY = BM / TM, WG = TX * TY;
  // shared tiles are vec4 arrays (k-major), so a thread reads its TM (TN) values with TM/4 (TN/4) loads
  const SA = BM / 4 + 1, SB = BN / 4 + 1;          // row strides in vec4s (+1 against bank conflicts)
  // Each thread stages whole vec4s (4 consecutive m of A, 4 consecutive n of B); a thread
  // writing one component of a shared vec4 races with its neighbours on some GPUs.
  const VA = (BM * BK) / (4 * WG), VB = (BN * BK) / (4 * WG);
  const actf = ACTIVATIONS[act];
  if (!actf) throw new Error(`gemm: unknown activation ${act}`);
  // (k, g): slice row k and vec4 group g of the tile, chosen so global reads stay coalesced
  const idxA = transA ? `let g = e % ${BM / 4}u; let kk = e / ${BM / 4}u;` : `let kk = e % ${BK}u; let g = e / ${BK}u;`;
  const idxB = transB ? `let kk = e % ${BK}u; let g = e / ${BK}u;` : `let g = e % ${BN / 4}u; let kk = e / ${BN / 4}u;`;
  const ldA = (k0, c) => transA ? `A[(${k0} + kk) * u.lda + m0 + 4u * g + ${c}u]` : `A[(m0 + 4u * g + ${c}u) * u.lda + ${k0} + kk]`;
  const ldB = (k0, c) => transB ? `B[(n0 + 4u * g + ${c}u) * u.ldb + ${k0} + kk]` : `B[(${k0} + kk) * u.ldb + n0 + 4u * g + ${c}u]`;
  const okA = (k0, c) => `m0 + 4u * g + ${c}u < u.M && ${k0} + kk < kEnd`;
  const okB = (k0, c) => `n0 + 4u * g + ${c}u < u.N && ${k0} + kk < kEnd`;
  // global → registers for the slice starting at k0 (zero outside the matrix)
  const fetch = (k0) => `
        ${R(VA).map((l) => `{ let e = ${l * WG}u + lidx; ${idxA} ra${l} = vec4f(${R(4).map((c) => `select(0.0, ${ldA(k0, c)}, ${okA(k0, c)})`).join(', ')}); }`).join('\n        ')}
        ${R(VB).map((l) => `{ let e = ${l * WG}u + lidx; ${idxB} rb${l} = vec4f(${R(4).map((c) => `select(0.0, ${ldB(k0, c)}, ${okB(k0, c)})`).join(', ')}); }`).join('\n        ')}`;
  // registers → shared memory
  const stage = `
        ${R(VA).map((l) => `{ let e = ${l * WG}u + lidx; ${idxA} As[kk * ${SA}u + g] = ra${l}; }`).join('\n        ')}
        ${R(VB).map((l) => `{ let e = ${l * WG}u + lidx; ${idxB} Bs[kk * ${SB}u + g] = rb${l}; }`).join('\n        ')}`;
  const kernel = ctx.kernel({
    name: `gemm-${key}`, dims: 2, workgroupSize: [WG, 1, 1], guard: false,
    buffers: { A: { type: 'f32', access: 'read' }, B: { type: 'f32', access: 'read' }, C: { type: 'f32' },
      ...(bias ? { bias: { type: 'f32', access: 'read' } } : {}) },
    uniforms: { M: 'u32', N: 'u32', K: 'u32', lda: 'u32', ldb: 'u32', ldc: 'u32', alpha: 'f32', beta: 'f32', chunk: 'u32' },
    shared: /* wgsl */`
      var<workgroup> As: array<vec4f, ${BK * SA}>;
      var<workgroup> Bs: array<vec4f, ${BK * SB}>;
      fn act(x: f32) -> f32 { return ${actf.f}; }
      fn store(m: u32, n: u32, s: f32, z: u32) {
        ${split ? 'C[(z * u.M + m) * u.N + n] = s;   // raw partial sum of this K chunk' : `var x = u.alpha * s;
        ${accumulate ? 'x += u.beta * C[m * u.ldc + n];' : ''}
        ${bias ? 'x += bias[n];' : ''}
        C[m * u.ldc + n] = act(x);`}
      }`,
    code: /* wgsl */`
      let tile = wid.xy;
      let m0 = tile.y * ${BM}u;
      let n0 = tile.x * ${BN}u;
      let tx = lidx % ${TX}u;
      let ty = lidx / ${TX}u;
      let kBegin = wid.z * u.chunk;
      let kEnd = min(u.K, kBegin + u.chunk);
      ${R(TM).map((i) => R(TN).map((j) => `var c${i}_${j} = 0.0;`).join(' ')).join('\n      ')}
      ${R(VA).map((l) => `var ra${l} = vec4f(0.0);`).join(' ')}
      ${R(VB).map((l) => `var rb${l} = vec4f(0.0);`).join(' ')}
      ${fetch('kBegin')}
      for (var k0 = kBegin; k0 < kEnd; k0 += ${BK}u) {
        ${stage}
        workgroupBarrier();
        ${cfg.prefetch ? `if (k0 + ${BK}u < kEnd) { ${fetch(`(k0 + ${BK}u)`)} }   // next slice in flight while this one is used` : ''}
        for (var kk = 0u; kk < ${BK}u; kk++) {
          ${R(TM / 4).map((q) => `let a${q} = As[kk * ${SA}u + ty * ${TM / 4}u + ${q}u];`).join(' ')}
          ${R(TN / 4).map((q) => `let b${q} = Bs[kk * ${SB}u + tx * ${TN / 4}u + ${q}u];`).join(' ')}
          ${R(TM).map((i) => R(TN).map((j) => `c${i}_${j} = fma(a${i >> 2}[${i & 3}], b${j >> 2}[${j & 3}], c${i}_${j});`).join(' ')).join('\n          ')}
        }
        workgroupBarrier();
        ${cfg.prefetch ? '' : `if (k0 + ${BK}u < kEnd) { ${fetch(`(k0 + ${BK}u)`)} }`}
      }
      ${R(TM).map((i) => `{
        let m = m0 + ty * ${TM}u + ${i}u;
        if (m < u.M) {
          ${R(TN).map((j) => `{ let n = n0 + tx * ${TN}u + ${j}u; if (n < u.N) { store(m, n, c${i}_${j}, wid.z); } }`).join('\n          ')}
        }
      }`).join('\n      ')}`,
  });
  kernel.cfg = cfg;
  perCtx.set(key, kernel);
  return kernel;
}

/**
 * C = act(α op(A) op(B) + β C + bias). Shapes: op(A) is M×K, op(B) is K×N, C is M×N (row-major).
 * o: { M, N, K, transA, transB, alpha, beta, bias (GArray of N), act, cfg }
 */
function gemm(ctx, A, B, C, o) {
  const { M, N, K } = o;
  const cfg = o.cfg || ctx._gemmCfg || GEMM_CONFIGS[0];
  const accumulate = (o.beta ?? 0) !== 0;
  const lda = o.lda ?? (o.transA ? M : K);
  const ldb = o.ldb ?? (o.transB ? K : N);
  const ldc = o.ldc ?? N;
  const tiles = Math.ceil(N / cfg.BN) * Math.ceil(M / cfg.BM);
  // Split-K: a small output with a long K (e.g. a weight gradient Xᵀ dZ over a big batch) would
  // occupy only a few workgroups. Split K into chunks, write partial tiles, then sum them in a
  // fixed order (deterministic) while applying the epilogue.
  const splits = o.split ?? (tiles < 64 && K >= 2048 ? Math.min(64, Math.ceil(256 / tiles), Math.ceil(K / 512)) : 1);
  const vals = { M, N, K, lda, ldb, ldc, alpha: o.alpha ?? 1, beta: o.beta ?? 0 };
  if (splits <= 1) {
    const k = gemmKernel(ctx, cfg, { transA: o.transA, transB: o.transB, act: o.act || 'none', bias: !!o.bias, accumulate });
    k.dispatch([Math.ceil(N / cfg.BN), Math.ceil(M / cfg.BM), 1], { ...vals, chunk: K }, { A, B, C, ...(o.bias ? { bias: o.bias } : {}) });
    return C;
  }
  const chunk = Math.ceil(K / splits / cfg.BK) * cfg.BK;
  const S = Math.ceil(K / chunk);
  const need = S * M * N;
  let tmp = ctx._gemmTmp;
  if (!tmp || tmp.length < need) {
    if (tmp) { ctx.flush(); tmp.destroy(); }   // recorded work still uses the old buffer: submit it first
    tmp = ctx._gemmTmp = ctx.array('f32', need, { label: 'gemm-split-partials' });
  }
  const k = gemmKernel(ctx, cfg, { transA: o.transA, transB: o.transB, split: true });
  k.dispatch([Math.ceil(N / cfg.BN), Math.ceil(M / cfg.BM), S], { ...vals, chunk }, { A, B, C: tmp });
  reduceKernel(ctx, o.act || 'none', !!o.bias, accumulate).run(M * N, { ...vals, S }, { P: tmp, C, ...(o.bias ? { bias: o.bias } : {}) });
  return C;
}

function reduceKernel(ctx, act, bias, accumulate) {
  const key = `reduce/${act}/${+bias}/${+accumulate}`;
  const perCtx = cache.get(ctx) || new Map();
  cache.set(ctx, perCtx);
  if (!perCtx.has(key)) {
    perCtx.set(key, ctx.kernel({
      name: `gemm-${key}`,
      buffers: { P: { type: 'f32', access: 'read' }, C: { type: 'f32' }, ...(bias ? { bias: { type: 'f32', access: 'read' } } : {}) },
      uniforms: { M: 'u32', N: 'u32', K: 'u32', lda: 'u32', ldb: 'u32', ldc: 'u32', alpha: 'f32', beta: 'f32', S: 'u32' },
      shared: `fn act(x: f32) -> f32 { return ${ACTIVATIONS[act].f}; }`,
      code: /* wgsl */`
        let m = i / u.N; let n = i % u.N;
        var s = 0.0;
        for (var z = 0u; z < u.S; z++) { s += P[z * u.M * u.N + i]; }
        var x = u.alpha * s;
        ${accumulate ? 'x += u.beta * C[m * u.ldc + n];' : ''}
        ${bias ? 'x += bias[n];' : ''}
        C[m * u.ldc + n] = act(x);`,
    }));
  }
  return perCtx.get(key);
}

/** Time every configuration on this device for an n×n×n multiply; remember the fastest (per adapter). */
async function autotuneGemm(ctx, { n = 2048, reps = 5, cache: useCache = true } = {}) {
  const id = `si-gemm-${ctx.caps.vendor}-${ctx.caps.architecture}`;
  if (useCache && typeof localStorage !== 'undefined') {
    const hit = localStorage.getItem(id);
    if (hit) { ctx._gemmCfg = JSON.parse(hit); return { best: ctx._gemmCfg, cached: true }; }
  }
  const A = ctx.array('f32', n * n), B = ctx.array('f32', n * n), C = ctx.array('f32', n * n);
  const results = [];
  for (const cfg of GEMM_CONFIGS) {
    const run = () => gemm(ctx, A, B, C, { M: n, N: n, K: n, cfg });
    run(); await ctx.idle();
    const t0 = performance.now();
    for (let r = 0; r < reps; r++) run();
    await ctx.idle();
    const ms = (performance.now() - t0) / reps;
    results.push({ cfg, ms, tflops: (2 * n ** 3) / (ms * 1e9) });
  }
  results.sort((a, b) => a.ms - b.ms);
  ctx._gemmCfg = results[0].cfg;
  if (useCache && typeof localStorage !== 'undefined') localStorage.setItem(id, JSON.stringify(results[0].cfg));
  A.destroy(); B.destroy(); C.destroy();
  return { best: results[0].cfg, results };
}
return { ACTIVATIONS, GEMM_CONFIGS, gemmKernel, gemm, autotuneGemm };
})();

// ---- src/nn/nn.js
const __m38 = (() => {
const gemm = __m37.gemm;
const ACTIVATIONS = __m37.ACTIVATIONS;
const defineModule = __m5.defineModule;
// Neural networks on the GPU: dense layers with fused bias + activation (the GEMM epilogue),
// backpropagation by hand-derived adjoints, Fourier-feature encodings, MSE / softmax
// cross-entropy losses, SGD and Adam — enough to train MLPs (neural fields, policies,
// classifiers, learned physics) entirely on the GPU — plus Population, a batch of small
// per-agent networks evaluated in one dispatch and trained by evolution strategies.
//
// Layout: activations are row-major [batch × features]; a dense layer is Y = act(X W + b)
// with W [in × out]. Backward, for dY = ∂L/∂Y:
//   dZ = dY ⊙ act'(Z)        (elementwise; for relu/tanh/sigmoid act' comes from Y itself)
//   dW = Xᵀ dZ,  db = Σ_rows dZ,  dX = dZ Wᵀ
// Both products use the transposed GEMM variants, so no transpose is ever materialised.




// activation derivative written in terms of the output y (when possible) or the input z
const DERIV_FROM_Y = { none: '1.0', relu: 'select(0.0, 1.0, y > 0.0)', leaky: 'select(0.01, 1.0, y > 0.0)', tanh: '1.0 - y * y', sigmoid: 'y * (1.0 - y)' };

defineModule('nn', {
  deps: ['hash'],
  code: /* wgsl */`
/** Standard normal sample from two uniform hashes (Box–Muller). */
fn nnGauss(a: u32, b: u32) -> f32 {
  let u1 = max(u2f(pcg(a * 0x9E3779B9u + b)), 1e-7);
  let u2 = u2f(pcg(b * 0x85EBCA6Bu + a + 0x6A09E667u));
  return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}`,
});

let seedCounter = 1;

/** A dense layer Y = act(X W + b). */
class Dense {
  constructor(ctx, inF, outF, o = {}) {
    this.ctx = ctx;
    this.in = inF; this.out = outF;
    this.act = o.act || 'relu';
    if (!ACTIVATIONS[this.act]) throw new Error(`Dense: unknown activation ${this.act}`);
    this.W = ctx.array('f32', inF * outF, { label: 'nn-W' });
    this.b = ctx.array('f32', outF, { label: 'nn-b' });
    this.gW = ctx.array('f32', inF * outF, { label: 'nn-gW' });
    this.gb = ctx.array('f32', outF, { label: 'nn-gb' });
    this.init(o.init || (this.act === 'relu' || this.act === 'leaky' ? 'he' : this.act === 'sin' ? 'siren' : 'xavier'), o.seed ?? seedCounter++, o.first, o.omega);
  }
  get params() { return [{ w: this.W, g: this.gW }, { w: this.b, g: this.gb }]; }

  /** He / Xavier / SIREN initialisation (uniform), biases zero. */
  init(kind, seed, first = false, omega = 30) {
    const n = this.in * this.out;
    let bound;
    if (kind === 'he') bound = Math.sqrt(6 / this.in);
    else if (kind === 'siren') bound = first ? 1 / this.in : Math.sqrt(6 / this.in) / omega;
    else bound = Math.sqrt(6 / (this.in + this.out));
    const w = new Float32Array(n);
    let s = (seed * 2654435761) >>> 0 || 1;
    for (let k = 0; k < n; k++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; w[k] = ((s >>> 0) / 4294967296 * 2 - 1) * bound; }
    this.W.write(w);
    this.b.write(new Float32Array(this.out));
  }
}

/**
 * Fourier features: [x, sin(2^k π x), cos(2^k π x)] for k < bands, per input coordinate —
 * lets an MLP represent high frequencies (Tancik et al. 2020). in → in·(1 + 2·bands).
 */
class FourierFeatures {
  constructor(ctx, inF, bands = 8) {
    this.ctx = ctx; this.in = inF; this.bands = bands; this.out = inF * (1 + 2 * bands);
    this.k = ctx.kernel({
      name: 'nn-fourier', buffers: { X: { type: 'f32', access: 'read' }, Y: { type: 'f32' } },
      uniforms: { inF: 'u32', bands: 'u32' },
      code: /* wgsl */`
        let row = i / u.inF; let c = i % u.inF;
        let x = X[i];
        let o = row * u.inF * (1u + 2u * u.bands) + c * (1u + 2u * u.bands);
        Y[o] = x;
        var f = 3.14159265;
        for (var b = 0u; b < u.bands; b++) { Y[o + 1u + 2u * b] = sin(f * x); Y[o + 2u + 2u * b] = cos(f * x); f *= 2.0; }`,
    });
  }
  get params() { return []; }
  forward(X, Y, batch) { this.k.run(batch * this.in, { inF: this.in, bands: this.bands }, { X, Y }); return Y; }
}

/**
 * A multilayer perceptron with optional Fourier-feature input encoding.
 *   const net = new MLP(gpu, { layers: [2, 64, 64, 3], act: 'relu', out: 'sigmoid', fourier: 8, maxBatch: 65536 });
 *   net.forward(X, batch);            // net.output: [batch × 3]
 *   net.backward(dY, batch);          // gradients in each layer's gW / gb
 */
class MLP {
  constructor(ctx, o) {
    this.ctx = ctx;
    this.maxBatch = o.maxBatch || 16384;
    const sizes = o.layers.slice();
    this.encoder = o.fourier ? new FourierFeatures(ctx, sizes[0], o.fourier) : null;
    if (this.encoder) sizes[0] = this.encoder.out;
    this.layers = [];
    for (let k = 0; k + 1 < sizes.length; k++) {
      const last = k + 2 === sizes.length;
      this.layers.push(new Dense(ctx, sizes[k], sizes[k + 1], { act: last ? o.out || 'none' : o.act || 'relu', first: k === 0, seed: o.seed ? o.seed + k : undefined }));
    }
    const A = (n, label) => ctx.array('f32', n, { label });
    this.acts = [A(this.maxBatch * sizes[0], 'nn-a0'), ...this.layers.map((l, k) => A(this.maxBatch * l.out, `nn-a${k + 1}`))];
    this.grads = this.acts.map((a, k) => (k === 0 ? null : A(a.length, `nn-d${k}`)));
    this.dZ = A(this.maxBatch * Math.max(...this.layers.map((l) => l.out)), 'nn-dz');
    this.input = this.encoder ? A(this.maxBatch * o.layers[0], 'nn-x') : this.acts[0];
    this._derivK = new Map();
    this.colSumK = ctx.kernel({
      name: 'nn-colsum', workgroupSize: 64, buffers: { D: { type: 'f32', access: 'read' }, g: { type: 'f32' } },
      uniforms: { rows: 'u32', cols: 'u32' },
      shared: 'var<workgroup> part: array<f32, 64>;',
      guard: false,
      code: /* wgsl */`
        // one workgroup per column: strided partial sums, then a tree in shared memory
        let col = wgi;
        var s = 0.0;
        if (col < u.cols) { for (var r = lidx; r < u.rows; r += 64u) { s += D[r * u.cols + col]; } }
        part[lidx] = s;
        workgroupBarrier();
        for (var w = 32u; w > 0u; w >>= 1u) { if (lidx < w) { part[lidx] += part[lidx + w]; } workgroupBarrier(); }
        if (lidx == 0u && col < u.cols) { g[col] = part[0]; }`,
    });
  }
  get output() { return this.acts[this.acts.length - 1]; }
  get params() { return this.layers.flatMap((l) => l.params); }
  get parameterCount() { return this.layers.reduce((s, l) => s + l.in * l.out + l.out, 0); }

  /** Y = net(X) for `batch` rows (X: [batch × in], or this.input when omitted). */
  forward(X = this.input, batch = this.maxBatch) {
    if (batch > this.maxBatch) throw new Error(`MLP.forward: batch ${batch} > maxBatch ${this.maxBatch}`);
    let a = this.acts[0];
    if (this.encoder) this.encoder.forward(X, a, batch);
    else if (X !== a) this.ctx.copy(X, a, batch * this.layers[0].in * 4);
    this.layers.forEach((l, k) => {
      gemm(this.ctx, this.acts[k], l.W, this.acts[k + 1], { M: batch, N: l.out, K: l.in, bias: l.b, act: l.act });
    });
    return this.output;
  }

  _deriv(act) {
    if (!this._derivK.has(act)) {
      const fromY = DERIV_FROM_Y[act];
      if (!fromY) throw new Error(`MLP.backward: activation ${act} needs its pre-activation; use relu, leaky, tanh, sigmoid or none`);
      this._derivK.set(act, this.ctx.kernel({
        name: `nn-deriv-${act}`, buffers: { dY: { type: 'f32', access: 'read' }, Y: { type: 'f32', access: 'read' }, dZ: { type: 'f32' } },
        code: `let y = Y[i]; dZ[i] = dY[i] * (${fromY});`,
      }));
    }
    return this._derivK.get(act);
  }

  /** Backpropagate dY = ∂L/∂output ([batch × out]); fills every layer's gW, gb. */
  backward(dY, batch) {
    let d = dY;
    for (let k = this.layers.length - 1; k >= 0; k--) {
      const l = this.layers[k];
      this._deriv(l.act).run(batch * l.out, null, { dY: d, Y: this.acts[k + 1], dZ: this.dZ });
      gemm(this.ctx, this.acts[k], this.dZ, l.gW, { M: l.in, N: l.out, K: batch, transA: true });          // dW = Xᵀ dZ
      this.colSumK.dispatch(l.out, { rows: batch, cols: l.out }, { D: this.dZ, g: l.gb });                    // db = Σ dZ
      if (k > 0) {
        gemm(this.ctx, this.dZ, l.W, this.grads[k], { M: batch, N: l.in, K: l.out, transB: true });          // dX = dZ Wᵀ
        d = this.grads[k];
      }
    }
  }
}

/** Mean-squared error: writes dY = 2(Y − T)/(batch·cols) and (optionally) the loss into `loss[0]`. */
class MSELoss {
  constructor(ctx, maxElems) {
    this.ctx = ctx;
    this.grad = ctx.array('f32', maxElems, { label: 'mse-grad' });
    this.sum = ctx.array('f32', 1024, { label: 'mse-partials' });
    this.k = ctx.kernel({
      name: 'nn-mse', workgroupSize: 256, buffers: { Y: { type: 'f32', access: 'read' }, T: { type: 'f32', access: 'read' }, dY: { type: 'f32' }, part: { type: 'f32' } },
      uniforms: { count: 'u32', scale: 'f32' },
      shared: 'var<workgroup> s: array<f32, 256>;',
      guard: false,
      code: /* wgsl */`
        var acc = 0.0;
        for (var e = gid.x; e < u.count; e += nwg.x * 256u) { let d = Y[e] - T[e]; dY[e] = 2.0 * d * u.scale; acc += d * d; }
        s[lidx] = acc;
        workgroupBarrier();
        for (var w = 128u; w > 0u; w >>= 1u) { if (lidx < w) { s[lidx] += s[lidx + w]; } workgroupBarrier(); }
        if (lidx == 0u) { part[wgi] = s[0] * u.scale; }`,
    });
  }
  /** Returns this.grad; call readLoss() for the value. */
  run(Y, T, count) {
    this.groups = Math.min(1024, Math.ceil(count / 256));
    this.ctx.clear(this.sum);
    this.k.dispatch(this.groups, { count, scale: 1 / count }, { Y, T, dY: this.grad, part: this.sum });
    return this.grad;
  }
  async readLoss() { const p = await this.sum.read(this.groups); return p.reduce((a, b) => a + b, 0); }
}

/** Adam (Kingma & Ba 2015) with bias correction; optional weight decay (AdamW) and gradient clipping. */
class Adam {
  constructor(ctx, params, o = {}) {
    this.ctx = ctx;
    this.params = params;
    this.lr = o.lr ?? 1e-3; this.beta1 = o.beta1 ?? 0.9; this.beta2 = o.beta2 ?? 0.999; this.eps = o.eps ?? 1e-8;
    this.weightDecay = o.weightDecay ?? 0; this.clip = o.clip ?? 0;
    this.t = 0;
    this.state = params.map((p) => ({ m: ctx.array('f32', p.w.length), v: ctx.array('f32', p.w.length) }));
    this.k = ctx.kernel({
      name: 'nn-adam', buffers: { w: { type: 'f32' }, g: { type: 'f32', access: 'read' }, m: { type: 'f32' }, v: { type: 'f32' } },
      uniforms: { lr: 'f32', b1: 'f32', b2: 'f32', eps: 'f32', c1: 'f32', c2: 'f32', wd: 'f32', clip: 'f32' },
      code: /* wgsl */`
        var gi = g[i];
        if (u.clip > 0.0) { gi = clamp(gi, -u.clip, u.clip); }
        let mi = u.b1 * m[i] + (1.0 - u.b1) * gi;
        let vi = u.b2 * v[i] + (1.0 - u.b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        w[i] = w[i] - u.lr * ((mi / u.c1) / (sqrt(vi / u.c2) + u.eps) + u.wd * w[i]);`,
    });
  }
  step() {
    this.t++;
    const c1 = 1 - this.beta1 ** this.t, c2 = 1 - this.beta2 ** this.t;
    this.params.forEach((p, k) => this.k.run(p.w.length, { lr: this.lr, b1: this.beta1, b2: this.beta2, eps: this.eps, c1, c2, wd: this.weightDecay, clip: this.clip },
      { w: p.w, g: p.g, m: this.state[k].m, v: this.state[k].v }));
  }
}

/** Plain SGD with momentum. */
class SGD {
  constructor(ctx, params, o = {}) {
    this.params = params; this.lr = o.lr ?? 1e-2; this.momentum = o.momentum ?? 0.9;
    this.vel = params.map((p) => ctx.array('f32', p.w.length));
    this.k = ctx.kernel({
      name: 'nn-sgd', buffers: { w: { type: 'f32' }, g: { type: 'f32', access: 'read' }, v: { type: 'f32' } }, uniforms: { lr: 'f32', mu: 'f32' },
      code: 'let vi = u.mu * v[i] + g[i]; v[i] = vi; w[i] = w[i] - u.lr * vi;',
    });
  }
  step() { this.params.forEach((p, k) => this.k.run(p.w.length, { lr: this.lr, mu: this.momentum }, { w: p.w, g: p.g, v: this.vel[k] })); }
}

/**
 * A population of small MLPs — one per agent — that share a parameter vector θ and differ by
 * antithetic Gaussian perturbations: agent 2a uses θ + σ ε_a, agent 2a+1 uses θ − σ ε_a, with ε
 * regenerated from a hash (never stored). forward() evaluates every agent's network in one
 * dispatch; update(fitness) is OpenAI-style evolution strategies with centred ranks:
 *   θ ← θ + lr / (P σ) Σ_a rank(F_a) ε_a.
 *   const pop = new Population(gpu, { layers: [8, 32, 32, 2], count: 4096, sigma: 0.05 });
 *   pop.forward(obs, act);       // obs: [count × 8], act: [count × 2]
 *   await pop.update(fitness);   // fitness: GArray [count] (higher is better)
 */
class Population {
  constructor(ctx, o) {
    this.ctx = ctx;
    this.sizes = o.layers;
    this.count = o.count + (o.count & 1);
    this.sigma = o.sigma ?? 0.05;
    this.lr = o.lr ?? 0.03;
    this.act = o.act || 'tanh';
    this.out = o.out || 'tanh';
    this.generation = 0;
    this.seed = o.seed ?? 1234;
    let n = 0;
    this.offsets = [];
    for (let k = 0; k + 1 < this.sizes.length; k++) { this.offsets.push(n); n += this.sizes[k] * this.sizes[k + 1] + this.sizes[k + 1]; }
    this.parameterCount = n;
    this.theta = ctx.array('f32', n, { label: 'pop-theta' });
    this.weights = ctx.array('f32', this.count, { label: 'pop-rank-weights' });
    const init = new Float32Array(n);
    let s = this.seed;
    for (let k = 0; k + 1 < this.sizes.length; k++) {
      const a = this.sizes[k], b = this.sizes[k + 1], bound = Math.sqrt(6 / (a + b));
      for (let q = 0; q < a * b; q++) { s = (s * 16807) % 2147483647; init[this.offsets[k] + q] = (s / 2147483647 * 2 - 1) * bound; }
    }
    this.theta.write(init);
    const maxW = Math.max(...this.sizes);
    const layersWGSL = this.sizes.slice(0, -1).map((a, k) => {
      const b = this.sizes[k + 1], off = this.offsets[k], last = k + 2 === this.sizes.length;
      const f = ACTIVATIONS[last ? this.out : this.act].f;
      return /* wgsl */`
        for (var j = 0u; j < ${b}u; j++) {
          var z = param(${off + a * b}u + j);
          for (var q = 0u; q < ${a}u; q++) { z = fma(h[q], param(${off}u + q * ${b}u + j), z); }
          let x = z;
          nxt[j] = ${f};
        }
        for (var j = 0u; j < ${b}u; j++) { h[j] = nxt[j]; }`;
    }).join('\n');
    this.forwardK = ctx.kernel({
      name: 'pop-forward', imports: ['nn'],
      buffers: { theta: { type: 'f32', access: 'read' }, X: { type: 'f32', access: 'read' }, Y: { type: 'f32' } },
      uniforms: { sigma: 'f32', seed: 'u32' },
      shared: /* wgsl */`
        var<private> agent: u32;
        fn param(j: u32) -> f32 {
          let e = nnGauss(u.seed ^ (agent >> 1u), j);
          return theta[j] + select(u.sigma, -u.sigma, (agent & 1u) == 1u) * e;
        }`,
      code: /* wgsl */`
        agent = i;
        var h: array<f32, ${maxW}>;
        var nxt: array<f32, ${maxW}>;
        for (var q = 0u; q < ${this.sizes[0]}u; q++) { h[q] = X[i * ${this.sizes[0]}u + q]; }
        ${layersWGSL}
        for (var j = 0u; j < ${this.sizes[this.sizes.length - 1]}u; j++) { Y[i * ${this.sizes[this.sizes.length - 1]}u + j] = h[j]; }`,
    });
    // θ_j += lr/(Pσ) Σ_pairs (w_{2a} − w_{2a+1}) ε_{a,j}
    this.updateK = ctx.kernel({
      name: 'pop-update', imports: ['nn'],
      buffers: { theta: { type: 'f32' }, w: { type: 'f32', access: 'read' } },
      uniforms: { scale: 'f32', seed: 'u32', pairs: 'u32' },
      code: /* wgsl */`
        var s = 0.0;
        for (var a = 0u; a < u.pairs; a++) { s += (w[2u * a] - w[2u * a + 1u]) * nnGauss(u.seed ^ a, i); }
        theta[i] += u.scale * s;`,
    });
  }
  get seedNow() { return (this.seed * 2654435761 + this.generation * 40503) >>> 0; }
  /** Evaluate every agent's perturbed network: X [count × in] → Y [count × out]. */
  forward(X, Y, count = this.count) { this.forwardK.run(count, { sigma: this.sigma, seed: this.seedNow }, { theta: this.theta, X, Y }); return Y; }
  /** Evaluate the unperturbed θ for agent rows (σ = 0), e.g. to show the current policy. */
  forwardMean(X, Y, count = this.count) { this.forwardK.run(count, { sigma: 0, seed: this.seedNow }, { theta: this.theta, X, Y }); return Y; }
  /** One ES step from per-agent fitness (GArray or Float32Array, higher is better). Returns { mean, best }. */
  async update(fitness) {
    const f = fitness instanceof Float32Array ? fitness : await fitness.read(this.count);
    const P = this.count;
    const order = Array.from({ length: P }, (_, k) => k).sort((a, b) => f[a] - f[b]);
    const w = new Float32Array(P);
    order.forEach((k, r) => { w[k] = r / (P - 1) - 0.5; });   // centred ranks in [-0.5, 0.5]
    this.ctx.flush();
    this.weights.write(w);
    this.updateK.run(this.parameterCount, { scale: this.lr / (P * this.sigma), seed: this.seedNow, pairs: P / 2 }, { theta: this.theta, w: this.weights });
    this.generation++;
    let mean = 0, best = -Infinity;
    for (let k = 0; k < P; k++) { mean += f[k]; best = Math.max(best, f[k]); }
    return { mean: mean / P, best };
  }
}
return { Dense, FourierFeatures, MLP, MSELoss, Adam, SGD, Population };
})();

// ---- src/nn/index.js
const __m36 = (() => {
const gemm = __m37.gemm;
const gemmKernel = __m37.gemmKernel;
const autotuneGemm = __m37.autotuneGemm;
const GEMM_CONFIGS = __m37.GEMM_CONFIGS;
const ACTIVATIONS = __m37.ACTIVATIONS;
const MLP = __m38.MLP;
const Dense = __m38.Dense;
const FourierFeatures = __m38.FourierFeatures;
const MSELoss = __m38.MSELoss;
const Adam = __m38.Adam;
const SGD = __m38.SGD;
const Population = __m38.Population;
// Neural networks: GEMM + dense layers + losses + optimisers + evolution strategies.
return { gemm, gemmKernel, autotuneGemm, GEMM_CONFIGS, ACTIVATIONS, MLP, Dense, FourierFeatures, MSELoss, Adam, SGD, Population };
})();

// ---- src/interop/three.js
const __m39 = (() => {
// three.js interop (r170+, WebGPURenderer + TSL). ShaderIntel and three.js share one GPUDevice;
// simulation state is copied GPU→GPU into three.js storage attributes by ShaderIntel kernels,
// and node materials read it in the vertex stage — nothing round-trips through the CPU.
//
//   import * as THREE from 'three/webgpu';
//   import * as TSL from 'three/tsl';
//   const gpu = await SI.init();
//   const renderer = await SI.three.renderer(THREE, gpu, { antialias: true });
//   const bodies = SI.three.bodies({ THREE, TSL, renderer, world });   // world: RigidWorld
//   scene.add(bodies.object);
//   function frame() { world.step(1 / 60); bodies.update(); gpu.flush(); renderer.render(scene, camera); }
//
// This module never imports three.js: pass in the namespaces you use (no version lock-in).
// It reaches the GPU buffer behind a three.js storage attribute through the WebGPU backend
// (backend.get(attribute).buffer), which three.js has exposed since r160.

/** A WebGPURenderer that uses ShaderIntel's device (features and limits already negotiated). */
async function renderer(THREE, gpu, opts = {}) {
  const r = new THREE.WebGPURenderer({ device: gpu.device, ...opts });
  await r.init();
  return r;
}

/** The GPUBuffer three.js allocated for a storage attribute (created on demand). */
function bufferOf(renderer, attribute) {
  const backend = renderer.backend;
  if (!backend?.get || !backend.createStorageAttribute) throw new Error('ShaderIntel three interop: needs THREE.WebGPURenderer (WebGPU backend)');
  let data = backend.get(attribute);
  if (!data.buffer) { backend.createStorageAttribute(attribute); data = backend.get(attribute); }
  if (!data.buffer) throw new Error('ShaderIntel three interop: could not reach the storage buffer of an attribute');
  return data.buffer;
}

const SHAPES = { sphere: 0, box: 1, capsule: 2 };

/**
 * Instanced meshes for every body of a RigidWorld (boxes, spheres, capsules), lit by three.js.
 * o: { THREE, TSL, renderer, world, material(shape) → NodeMaterial (optional), castShadow, receiveShadow,
 *      colors: (k) → TSL color node (optional), groups (collision-group bits to draw) }
 * Returns { object: THREE.Group, update(), meshes }.
 */
function bodies(o) {
  const { THREE, TSL, renderer: r, world } = o;
  const gpu = world.ctx;
  const max = world.maxBodies;
  // per instance: (position, visible), (quaternion), (size, shape)
  const attr = new THREE.StorageInstancedBufferAttribute(max * 3, 4);
  const buf = bufferOf(r, attr);
  const pack = gpu.kernel({
    name: 'three-bodies-pack', buffers: { B: { type: 'vec4f', access: 'read' }, out: { type: 'vec4f' } },
    uniforms: { groups: 'u32' },
    code: /* wgsl */`
      let x = B[i * 12u]; let q = B[i * 12u + 1u]; let sz = B[i * 12u + 8u];
      let info = bitcast<vec4u>(B[i * 12u + 11u]);
      let visible = select(0.0, 1.0, (info.z & 4u) == 0u && (info.x & u.groups) != 0u);
      out[3u * i] = vec4f(x.xyz, visible);
      out[3u * i + 1u] = q;
      out[3u * i + 2u] = sz;`,
  });
  const { Fn, storage, instanceIndex, positionLocal, normalLocal, vec3, vec4, float, cross, clamp, select, hash, mix,
    transformNormalToView } = TSL;
  const data = storage(attr, 'vec4', max * 3).toReadOnly?.() ?? storage(attr, 'vec4', max * 3);
  const rot = (q, v) => { const t = cross(q.xyz, v).mul(2); return v.add(t.mul(q.w)).add(cross(q.xyz, t)); };
  const palette = o.colors || ((k) => {
    const h = hash(k.add(17));
    return mix(vec3(0.85, 0.45, 0.3), vec3(0.3, 0.55, 0.9), h).mul(mix(float(0.75), float(1.1), hash(k.add(91))));
  });
  const group = new THREE.Group();
  const meshes = {};
  const geometries = {
    box: new THREE.BoxGeometry(2, 2, 2),
    sphere: new THREE.SphereGeometry(1, 24, 16),
    capsule: new THREE.CapsuleGeometry(1, 2, 8, 16),
  };
  for (const [name, shape] of Object.entries(SHAPES)) {
    const material = o.material?.(name) || new THREE.MeshStandardNodeMaterial({ roughness: 0.55, metalness: 0.0 });
    const base = instanceIndex.mul(3);
    const pv = data.element(base);
    const q = data.element(base.add(1));
    const sz = data.element(base.add(2));
    const show = pv.w.mul(select(sz.w.round().equal(float(shape)), float(1), float(0)));
    const local = Fn(() => {
      const p = positionLocal;
      if (shape === SHAPES.box) return p.mul(sz.xyz);
      if (shape === SHAPES.sphere) return p.mul(sz.x);
      const c = clamp(p.y, -1, 1);   // capsule: stretch the straight part to h, scale the caps by r
      return vec3(p.x.mul(sz.x), c.mul(sz.y).add(p.y.sub(c).mul(sz.x)), p.z.mul(sz.x));
    })();
    material.positionNode = pv.xyz.add(rot(q, local)).mul(show);
    if (transformNormalToView) material.normalNode = transformNormalToView(rot(q, normalLocal));
    if (!o.material) material.colorNode = palette(instanceIndex);
    const mesh = new THREE.InstancedMesh(geometries[name], material, max);
    mesh.frustumCulled = false;
    mesh.castShadow = o.castShadow ?? true;
    mesh.receiveShadow = o.receiveShadow ?? true;
    mesh.count = 0;
    group.add(mesh);
    meshes[name] = mesh;
  }
  return {
    object: group,
    meshes,
    attribute: attr,
    /** Copy this frame's poses into three.js (GPU → GPU). Call after world.step(), before rendering. */
    update() {
      const n = world.count;
      for (const m of Object.values(meshes)) m.count = n;
      if (n) pack.run(n, { groups: (o.groups ?? 0xffffffff) >>> 0 }, { B: world.B, out: buf });
    },
  };
}

/**
 * Instanced particles from any ShaderIntel array of positions (MPM particles, boids, points).
 * o: { THREE, TSL, renderer, source: GArray, stride (vec4s per element), count: () => n, max,
 *      scale (multiply), offset (add), radius, color: TSL node or [r,g,b], material }
 */
function particles(o) {
  const { THREE, TSL, renderer: r } = o;
  const src0 = typeof o.source === 'function' ? o.source() : o.source;
  const gpu = o.gpu || src0.ctx;
  const max = o.max;
  const attr = new THREE.StorageInstancedBufferAttribute(max, 4);
  const buf = bufferOf(r, attr);
  const pack = gpu.kernel({
    name: 'three-particles-pack', buffers: { src: { type: 'vec4f', access: 'read' }, out: { type: 'vec4f' } },
    uniforms: { stride: 'u32', scale: 'f32', offset: 'vec3f' },
    code: 'out[i] = vec4f(src[i * u.stride].xyz * u.scale + u.offset, 1.0);',
  });
  const { storage, instanceIndex, positionLocal, vec3 } = TSL;
  const data = storage(attr, 'vec4', max);
  const material = o.material || new THREE.MeshStandardNodeMaterial({ roughness: 0.4 });
  material.positionNode = data.element(instanceIndex).xyz.add(positionLocal.mul(o.radius ?? 0.01));
  if (!o.material) material.colorNode = Array.isArray(o.color) ? vec3(...o.color) : (o.color || vec3(0.2, 0.45, 0.9));
  const mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), material, max);
  mesh.frustumCulled = false;
  mesh.count = 0;
  return {
    object: mesh,
    attribute: attr,
    update() {
      const n = Math.min(max, typeof o.count === 'function' ? o.count() : o.count);
      mesh.count = n;
      if (n) pack.run(n, { stride: o.stride ?? 1, scale: o.scale ?? 1, offset: o.offset ?? [0, 0, 0] }, { src: typeof o.source === 'function' ? o.source() : o.source, out: buf });
    },
  };
}
return { renderer, bufferOf, bodies, particles };
})();

// ---- src/net/transport.js
const __m41 = (() => {
// Transports for peer-to-peer sessions. Every transport hands out Channel objects:
//   channel.send(message)          message: plain objects, arrays, numbers, strings, typed arrays
//   channel.onmessage = (m) => {}
//   channel.close()
//
// On the wire a message is JSON in which every typed array travels as base64 of its bytes, so
// the bits arrive exactly: -0, NaN payloads and u32 words stored in float arrays (the rigid
// body records keep collision masks that way) survive, where JSON numbers would not. Messages
// longer than 64 KB are split into parts, under every browser's data-channel message limit.
//
// LoopbackNetwork  two (or more) endpoints in one page with simulated latency, jitter, loss and
//                  reordering, driven by a seeded RNG — for tests, demos and tuning netcode.
// RTCPeer          WebRTC data channels between browsers ("inputs": unordered, no retransmits;
//                  "state": reliable, ordered), with signalling through any Signaler:
// BroadcastSignaler  same-origin tabs (BroadcastChannel) — no server needed for local play/testing.
// WebSocketSignaler  your own relay server (it only forwards {room, from, payload} messages).

/** Seeded xorshift32 → [0, 1). */
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

/**
 * Simulated network. new LoopbackNetwork({ latency: 60, jitter: 15, loss: 0.05, seed: 1 }).
 * connect('a', 'b') returns [channelAtA, channelAtB]; each has .unreliable and .reliable.
 * latency/jitter in ms (one way), loss applies to unreliable channels only.
 */
class LoopbackNetwork {
  constructor(o = {}) {
    this.latency = o.latency ?? 40;
    this.jitter = o.jitter ?? 10;
    this.loss = o.loss ?? 0;
    this.duplicate = o.duplicate ?? 0;
    this.random = rng(o.seed ?? 1);
    // codec: send what WebRTC would (the wire encoding) instead of a structured clone, so a
    // message that survives the simulator survives the real transport
    this.codec = o.codec ?? true;
    this.stats = { sent: 0, dropped: 0, delivered: 0, bytes: 0 };
  }
  _link(deliver, reliable) {
    let lastArrival = 0;
    return {
      onmessage: null,
      send: (m) => {
        this.stats.sent++;
        if (!reliable && this.random() < this.loss) { this.stats.dropped++; return; }
        const copies = !reliable && this.random() < this.duplicate ? 2 : 1;
        const wire = this.codec ? encode(m) : null;
        if (wire) this.stats.bytes += wire.length;
        for (let c = 0; c < copies; c++) {
          let delay = this.latency + (this.random() * 2 - 1) * this.jitter;
          if (reliable) { delay = Math.max(delay, lastArrival - performance.now()); lastArrival = performance.now() + delay; }   // ordered
          const copy = wire ? decode(wire) : structuredClone(m);
          setTimeout(() => { this.stats.delivered++; deliver(copy); }, Math.max(0, delay));
        }
      },
      close() {},
    };
  }
  connect() {
    const a = { unreliable: null, reliable: null }, b = { unreliable: null, reliable: null };
    for (const kind of ['unreliable', 'reliable']) {
      const rel = kind === 'reliable';
      const ab = this._link((m) => b[kind].onmessage?.(m), rel);
      const ba = this._link((m) => a[kind].onmessage?.(m), rel);
      a[kind] = { send: ab.send, onmessage: null, close() {} };
      b[kind] = { send: ba.send, onmessage: null, close() {} };
    }
    return [a, b];
  }
}

/** Signalling over BroadcastChannel: peers in tabs of the same origin find each other by room name. */
class BroadcastSignaler {
  constructor(room) {
    this.id = Math.random().toString(36).slice(2);
    this.bc = new BroadcastChannel(`si-signal-${room}`);
    this.onmessage = null;
    this.bc.onmessage = (e) => { if (e.data.from !== this.id && (!e.data.to || e.data.to === this.id)) this.onmessage?.(e.data); };
  }
  send(payload, to) { this.bc.postMessage({ from: this.id, to, payload }); }
  close() { this.bc.close(); }
}

/** Signalling through a WebSocket relay that broadcasts { room, from, to, payload } to the room. */
class WebSocketSignaler {
  constructor(url, room) {
    this.id = Math.random().toString(36).slice(2);
    this.room = room;
    this.onmessage = null;
    this.ws = new WebSocket(url);
    this.ready = new Promise((res, rej) => { this.ws.onopen = () => res(); this.ws.onerror = rej; });
    this.ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.room === room && m.from !== this.id && (!m.to || m.to === this.id)) this.onmessage?.(m);
    };
  }
  async send(payload, to) { await this.ready; this.ws.send(JSON.stringify({ room: this.room, from: this.id, to, payload })); }
  close() { this.ws.close(); }
}

/**
 * A WebRTC peer. The initiator creates the data channels and the offer; the other side answers.
 *   const peer = new RTCPeer(new BroadcastSignaler('room-1'), { initiator: true });
 *   await peer.ready;  peer.channels.unreliable.send(...)
 * iceServers: e.g. [{ urls: 'stun:stun.l.google.com:19302' }] for peers on different networks.
 */
class RTCPeer {
  constructor(signaler, o = {}) {
    this.signaler = signaler;
    this.pc = new RTCPeerConnection({ iceServers: o.iceServers || [] });
    this.channels = { unreliable: dataChannel(null), reliable: dataChannel(null) };
    this.remote = null;
    const attach = (dc) => {
      dc.binaryType = 'arraybuffer';
      this.channels[dc.label === 'inputs' ? 'unreliable' : 'reliable'].attach(dc);
    };
    this.ready = new Promise((resolve) => {
      let open = 0;
      const check = (dc) => { dc.addEventListener('open', () => { if (++open === 2) resolve(this); }); };
      if (o.initiator) {
        const a = this.pc.createDataChannel('inputs', { ordered: false, maxRetransmits: 0 });
        const b = this.pc.createDataChannel('state', { ordered: true });
        attach(a); attach(b); check(a); check(b);
      } else {
        this.pc.ondatachannel = (e) => { attach(e.channel); check(e.channel); };
      }
    });
    this.pc.onicecandidate = (e) => { if (e.candidate) signaler.send({ ice: e.candidate.toJSON() }, this.remote); };
    // Signalling messages are handled one at a time, and candidates that arrive before the remote
    // description is applied wait for it (dropping them can leave the connection without a path).
    const pendingIce = [];
    let chain = Promise.resolve();
    const handle = async (m) => {
      const p = m.payload;
      if (p.hello) {
        if (!o.initiator || this.remote) return;   // the first guest to say hello is the peer
        this.remote = m.from;
        await this.pc.setLocalDescription(await this.pc.createOffer());
        signaler.send({ sdp: this.pc.localDescription.toJSON() }, this.remote);
      } else if (p.sdp) {
        if (this.remote && m.from !== this.remote) return;
        this.remote = m.from;
        clearInterval(this._hello);
        await this.pc.setRemoteDescription(p.sdp);
        if (p.sdp.type === 'offer') { await this.pc.setLocalDescription(await this.pc.createAnswer()); signaler.send({ sdp: this.pc.localDescription.toJSON() }, this.remote); }
        for (const c of pendingIce.splice(0)) await this.pc.addIceCandidate(c).catch(() => {});
      } else if (p.ice) {
        if (this.remote && m.from !== this.remote) return;
        if (!this.pc.remoteDescription) pendingIce.push(p.ice);
        else await this.pc.addIceCandidate(p.ice).catch(() => {});
      }
    };
    signaler.onmessage = (m) => { chain = chain.then(() => handle(m)).catch((e) => console.warn('RTCPeer signalling:', e)); };
    // The guest says hello until the host answers: the host may not be listening yet (a page that
    // is still initialising its GPU), and a single hello would then be lost for good.
    if (!o.initiator) {
      signaler.send({ hello: true });
      this._hello = setInterval(() => signaler.send({ hello: true }), 500);
    }
  }
  close() { clearInterval(this._hello); this.pc.close(); this.signaler.close?.(); }
}

const CHUNK = 65536;

/**
 * Wrap an RTCDataChannel (or anything with send(), onmessage and readyState) as a Channel: the
 * wire encoding above, with long messages split into parts and reassembled. Use it to run a
 * RollbackSession over a WebRTC connection you set up yourself.
 */
function dataChannel(dc) {
  const parts = new Map();
  let next = 1;
  const ch = {
    dc: null,
    onmessage: null,
    send(m) {
      if (this.dc?.readyState !== 'open') return;
      const s = encode(m);
      if (s.length <= CHUNK) { this.dc.send(s); return; }
      const id = next++, n = Math.ceil(s.length / CHUNK);
      for (let k = 0; k < n; k++) this.dc.send(JSON.stringify({ __part: id, k, n, d: s.slice(k * CHUNK, (k + 1) * CHUNK) }));
    },
    close() { this.dc?.close(); },
    attach(d) {
      this.dc = d;
      d.onmessage = (e) => {
        const text = typeof e.data === 'string' ? e.data : new TextDecoder().decode(e.data);
        if (!text.startsWith('{"__part"')) { this.onmessage?.(decode(text)); return; }
        const p = JSON.parse(text);
        const got = parts.get(p.__part) || { n: p.n, have: 0, d: [] };
        if (got.d[p.k] === undefined) { got.d[p.k] = p.d; got.have++; }
        parts.set(p.__part, got);
        if (got.have < got.n) return;
        parts.delete(p.__part);
        this.onmessage?.(decode(got.d.join('')));
      };
    },
  };
  if (dc) ch.attach(dc);
  return ch;
}

function toBase64(bytes) {
  let s = '';
  for (let k = 0; k < bytes.length; k += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(k, k + 0x8000));
  return btoa(s);
}
function fromBase64(b64) {
  const s = atob(b64), bytes = new Uint8Array(s.length);
  for (let k = 0; k < s.length; k++) bytes[k] = s.charCodeAt(k);
  return bytes;
}
/** The wire encoding: JSON, with typed arrays as { __ta: type, b: base64 of the bytes }. */
function encode(m) {
  return JSON.stringify(m, (k, v) => (ArrayBuffer.isView(v) ? { __ta: v.constructor.name, b: toBase64(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)) } : v));
}
function decode(s) {
  return JSON.parse(s, (k, v) => (v && typeof v === 'object' && typeof v.__ta === 'string' ? new globalThis[v.__ta](fromBase64(v.b).buffer) : v));
}
return { LoopbackNetwork, BroadcastSignaler, WebSocketSignaler, RTCPeer, dataChannel, encode, decode };
})();

// ---- src/net/session.js
const __m42 = (() => {
// Rollback netcode (in the style of GGPO) for deterministic GPU simulations.
//
// Every peer runs the whole simulation at a fixed tick. A player's input for tick t is
// applied at t + delay; inputs travel redundantly over an unreliable channel (the last
// `redundancy` ticks in every packet), so loss costs nothing until it exceeds that window.
// When a remote input has not arrived the session predicts it (the player's last input) and
// keeps going; when the real input differs from the prediction it restores the GPU snapshot
// taken before that tick and re-simulates up to the present — within one frame.
//
// This only works if the simulation is a pure function of (state, inputs): RigidWorld in
// deterministic mode is (see physics/rigid.js — confluence). Peers on different GPUs can still
// diverge in the last bits of floating point, so every `hashEvery` confirmed ticks the peers
// exchange 64-bit state hashes. On a mismatch the authority (player 0) sends the state of its
// newest fully confirmed tick, taken from its rollback ring; the other peer imports it at that
// tick and re-simulates to the present with the inputs it has (resync). Each applied state
// starts a new epoch, and hashes from an abandoned history (an older epoch) are ignored, so one
// divergence costs one state transfer.
//
// Proof-carrying transcript: the session folds every confirmed tick's inputs into a hash chain
// and records the state hash every `hashEvery` ticks. The transcript (inputs + state hashes +
// chain head) is a certificate of the match: anyone holding the same code can replay it
// (verifyTranscript) and check every state hash, and changing any input breaks the chain.

const EMPTY = new Uint8Array(1);

function sameBytes(a, b) {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
}

/** 64-bit FNV-1a-style mix over strings / bytes (two 32-bit lanes, as a hex string). */
function chainHash(prev, ...parts) {
  let a = parseInt(prev.slice(8), 16) >>> 0 || 0x811c9dc5, b = parseInt(prev.slice(0, 8), 16) >>> 0 || 0x01000193;
  const eat = (x) => { a = Math.imul(a ^ x, 0x01000193) >>> 0; b = Math.imul(b ^ (x + a), 0x5bd1e995) >>> 0; };
  for (const p of parts) {
    if (typeof p === 'number') { eat(p & 0xff); eat((p >>> 8) & 0xff); eat((p >>> 16) & 0xff); eat(p >>> 24); }
    else if (typeof p === 'string') for (let k = 0; k < p.length; k++) eat(p.charCodeAt(k));
    else for (let k = 0; k < p.length; k++) eat(p[k]);
  }
  return b.toString(16).padStart(8, '0') + a.toString(16).padStart(8, '0');
}

class RollbackSession {
  /**
   * o: sim (RigidWorld or any { snapshot(into), restore(s), step(dt), hashRecord(target, source) }),
   *    game ({ save() → state, load(state) } for game state kept outside the simulation),
   *    channel ({ unreliable, reliable } from a transport), players (default 2), localPlayer,
   *    applyInputs(tick, inputs: Uint8Array[]) — turns every player's input into simulation commands,
   *    inputSize (bytes, default 1), delay (ticks, default 2), maxRollback (default 8), dt (default 1/60),
   *    hashEvery (default 30), redundancy (default 8), onDesync(tick, local, remote),
   *    resync (default true: the authority sends its state on a mismatch), authority (default 0),
   *    inputHistory (ticks of confirmed inputs kept so a late state can still be replayed, default 600),
   *    maxInFlight (ticks of GPU work allowed in the queue before the session waits a tick, default 3)
   */
  constructor(o) {
    this.sim = o.sim;
    this.game = o.game || null;
    this.players = o.players ?? 2;
    this.local = o.localPlayer;
    this.channel = o.channel;
    this.applyInputs = o.applyInputs;
    this.inputSize = o.inputSize ?? 1;
    this.delay = o.delay ?? 2;
    this.maxRollback = o.maxRollback ?? 8;
    this.dt = o.dt ?? 1 / 60;
    this.hashEvery = o.hashEvery ?? 30;
    this.redundancy = o.redundancy ?? 8;
    this.onDesync = o.onDesync || null;
    this.resync = o.resync ?? true;
    this.authority = o.authority ?? 0;
    this.inputHistory = o.inputHistory ?? 600;
    // Backpressure: a loop that ticks faster than the GPU executes would grow the queue without
    // bound, and every read-back (hashes, a resync export) would wait behind it — 16 s once on a
    // slow-dispatch driver. Each tick is fenced; past maxInFlight unfinished ticks we wait.
    this.maxInFlight = o.maxInFlight ?? 3;
    this._inflight = 0;
    this.epoch = 0;              // states applied from the authority so far (this peer)
    this.remoteEpoch = 0;        // the newest epoch seen in the peer's hashes
    this.expectEpoch = 0;        // hashes from older epochs describe a history the peer has abandoned
    this.resyncWanted = false;
    this.pendingState = null;
    this.tick = 0;
    this.inputs = Array.from({ length: this.players }, () => new Map());
    this.predicted = Array.from({ length: this.players }, () => new Map());
    this.confirmed = new Array(this.players).fill(-1);   // highest tick t with every input ≤ t known
    this.ring = new Array(this.maxRollback + 2).fill(null);
    this.rollbackTo = Infinity;
    this.hashes = new Map();        // tick -> { local, remote }
    this.pendingHash = [];          // ticks whose state hash is still to be taken
    this.chain = '0000000000000000';
    this.chainTick = -1;
    this.log = { inputs: [], hashes: {} };
    this.stats = { rollbacks: 0, rolledBackTicks: 0, stalls: 0, desyncs: 0, resyncs: 0, statesSent: 0, statesReceived: 0, exportErrors: 0, gpuWaits: 0, hashesRecorded: 0, hashesDropped: 0, hashesStale: 0, ping: 0, inSyncTick: -1, sent: 0, received: 0 };
    const zero = new Uint8Array(this.inputSize);
    for (let t = 0; t < this.delay; t++) this.inputs[this.local].set(t, zero);
    this.confirmed[this.local] = this.delay - 1;
    this.channel.unreliable.onmessage = (m) => this._onInputs(m);
    this.channel.reliable.onmessage = (m) => this._onReliable(m);
  }

  /** Lowest tick every player has confirmed (ticks ≤ this are final). */
  get confirmedTick() { return Math.min(...this.confirmed); }

  /**
   * Advance by one tick with this frame's local input (Uint8Array of inputSize bytes).
   * Returns the number of ticks simulated (0 while waiting for a peer that is too far behind).
   */
  advance(localInput = EMPTY) {
    const t = this.tick + this.delay;
    if (!this.inputs[this.local].has(t)) {
      this.inputs[this.local].set(t, Uint8Array.from(localInput));
      this.confirmed[this.local] = t;
    }
    this._send();
    if (this.pendingState && this.pendingState.tick <= this.tick) this._applyState();
    if (this.rollbackTo < this.tick) this._rollback();
    if (this.resyncWanted) this._sendState();
    this._hashConfirmed();
    this._extendChain();
    if (this.tick - this.confirmedTick > this.maxRollback) { this.stats.stalls++; return 0; }
    if (this._inflight > this.maxInFlight) { this.stats.gpuWaits++; return 0; }
    this._simulate(this.tick);
    this.tick++;
    const ctx = this.sim.ctx;
    if (ctx?.device) {
      ctx.flush();
      this._inflight++;
      ctx.device.queue.onSubmittedWorkDone().then(() => this._inflight--);
    }
    return 1;
  }

  _inputFor(p, t) {
    let v = this.inputs[p].get(t);
    if (!v) {
      // prediction: the most recent input we have from that player
      for (let s = Math.min(t, this.confirmed[p]); s >= 0 && !v; s--) v = this.inputs[p].get(s);
      v = v || new Uint8Array(this.inputSize);
      this.predicted[p].set(t, v);
    } else this.predicted[p].delete(t);
    return v;
  }

  _simulate(t) {
    const slot = t % this.ring.length;
    const prev = this.ring[slot];
    this.ring[slot] = { tick: t, snap: this.sim.snapshot(prev?.snap), game: this.game ? structuredClone(this.game.save()) : null };
    const inputs = [];
    for (let p = 0; p < this.players; p++) inputs.push(this._inputFor(p, t));
    this.applyInputs(t, inputs);
    this.sim.step(this.dt);
    if (t % this.hashEvery === 0 && !this.pendingHash.includes(t)) this.pendingHash.push(t);
  }

  _rollback() {
    const from = this.rollbackTo;
    this.rollbackTo = Infinity;
    const e = this.ring[from % this.ring.length];
    if (!e || e.tick !== from) throw new Error(`RollbackSession: tick ${from} is outside the rollback window`);
    this.sim.restore(e.snap);
    if (this.game) this.game.load(structuredClone(e.game));
    const to = this.tick;
    for (let t = from; t < to; t++) this._simulate(t);
    this.stats.rollbacks++;
    this.stats.rolledBackTicks += to - from;
  }

  // The state after tick T is final once every input ≤ T is known; it is the snapshot taken
  // before tick T + 1 (still in the ring). Hash it on the GPU and share it.
  _hashConfirmed() {
    const c = this.confirmedTick;
    this.pendingHash = this.pendingHash.filter((T) => {
      if (T > c || T + 1 >= this.tick) return true;
      const e = this.ring[(T + 1) % this.ring.length];
      if (!e || e.tick !== T + 1) { this.stats.hashesDropped++; return false; }   // fell out of the window: skip this one
      const buf = this.sim.ctx.array('u32', 4);
      this.sim.hashRecord(buf, e.snap);
      this.stats.hashesRecorded++;
      const epoch = this.epoch;
      buf.read(2).then((h) => {
        buf.destroy();
        if (epoch !== this.epoch) { this.stats.hashesStale++; return; }   // a state arrived meanwhile: this hash belongs to the old history
        const hex = h[1].toString(16).padStart(8, '0') + h[0].toString(16).padStart(8, '0');
        this._gotHash(T, hex, 'local');
        this.channel.reliable.send({ type: 'hash', from: this.local, tick: T, hash: hex, epoch });
      });
      return false;
    });
  }

  _gotHash(T, hex, who, epoch = 0) {
    if (who === 'remote') {
      if (epoch < this.expectEpoch) return;
      this.remoteEpoch = Math.max(this.remoteEpoch, epoch);
    }
    const h = this.hashes.get(T) || {};
    h[who] = hex;
    this.hashes.set(T, h);
    if (who === 'local') this.log.hashes[T] = hex;
    if (h.local && h.remote) {
      if (h.local === h.remote) this.stats.inSyncTick = Math.max(this.stats.inSyncTick, T);
      else if (!h.reported) {
        h.reported = true;
        this.stats.desyncs++;
        this.onDesync?.(T, h.local, h.remote);
        if (this.resync && this.local === this.authority) this.resyncWanted = true;
      }
    }
  }

  // Authority: send the state before tick S, the newest one whose every input is final.
  _sendState() {
    this.resyncWanted = false;
    const S = Math.min(this.confirmedTick + 1, this.tick - 1);
    const e = this.ring[S % this.ring.length];
    if (!e || e.tick !== S || !this.sim.exportState) return;
    const epoch = this.remoteEpoch + 1;
    this.expectEpoch = epoch;
    this.stats.statesSent++;
    const t0 = performance.now();
    this.sim.exportState(e.snap).then((state) => {
      this.stats.lastExport = { tick: S, ms: Math.round(performance.now() - t0), sentAt: this.tick };
      this.channel.reliable.send({ type: 'state', from: this.local, tick: S, epoch, state, game: e.game });
    })
      .catch((err) => {   // the peer keeps asking (its hashes still differ): count, report, retry on the next mismatch
        this.stats.exportErrors++;
        this.expectEpoch = epoch - 1;
        console.warn('RollbackSession: state export failed', err);
      });
  }

  // Other peers: adopt the authority's state at its tick and re-simulate to the present.
  _applyState() {
    const m = this.pendingState;
    this.pendingState = null;
    const S = m.tick;
    // A state older than the inputs we kept cannot be replayed to the present: ask for a newer one
    // (the authority ignores our old-epoch hashes from now on, so it would not notice by itself).
    if (S < this.tick - this.inputHistory + this.ring.length) {
      this.channel.reliable.send({ type: 'resync', from: this.local, epoch: this.epoch });
      return;
    }
    this.sim.importState(m.state);
    if (this.game && m.game) this.game.load(structuredClone(m.game));
    this.epoch = m.epoch;
    this.stats.lastResync = { tick: S, appliedAt: this.tick };
    this.rollbackTo = Infinity;
    // hashes of the abandoned history are recomputed (and re-sent) from the new one
    for (const [T, h] of this.hashes) if (T >= S - 1) { delete h.local; h.reported = false; }
    for (let T = Math.ceil(Math.max(0, S - 1) / this.hashEvery) * this.hashEvery; T < this.tick; T += this.hashEvery)
      if (!this.pendingHash.includes(T)) this.pendingHash.push(T);
    const to = this.tick;
    for (let t = S; t < to; t++) this._simulate(t);
    this.stats.resyncs++;
  }

  // Fold confirmed ticks into the transcript's hash chain (inputs, then any state hash).
  _extendChain() {
    const c = this.confirmedTick;
    while (this.chainTick < c && this.chainTick + 1 < this.tick) {
      const t = this.chainTick + 1;
      const row = [];
      for (let p = 0; p < this.players; p++) row.push(this.inputs[p].get(t) || new Uint8Array(this.inputSize));
      this.log.inputs.push(row.map((r) => Array.from(r)));
      this.chain = chainHash(this.chain, t, ...row);
      this.chainTick = t;
      // forget what can never be replayed again (rollbacks need ring.length ticks, a late resync state more)
      const old = t - Math.max(this.ring.length + this.redundancy + 2, this.inputHistory);
      if (old >= 0) for (let p = 0; p < this.players; p++) { this.inputs[p].delete(old); this.predicted[p].delete(old); }
    }
  }

  _send() {
    const t1 = this.tick + this.delay;
    const t0 = Math.max(0, t1 - this.redundancy + 1);
    const list = [];
    for (let t = t0; t <= t1; t++) list.push(Array.from(this.inputs[this.local].get(t) || EMPTY));
    this.channel.unreliable.send({ from: this.local, start: t0, inputs: list, ack: Math.min(...this.confirmed.filter((_, p) => p !== this.local)), sent: performance.now(), echo: this._echo });
    this.stats.sent++;
  }

  _onInputs(m) {
    if (m.from === this.local) return;
    this.stats.received++;
    if (m.echo) this.stats.ping = Math.round(performance.now() - m.echo);
    this._echo = m.sent;
    const p = m.from;
    m.inputs.forEach((arr, k) => {
      const t = m.start + k;
      if (this.inputs[p].has(t)) return;
      const v = Uint8Array.from(arr);
      this.inputs[p].set(t, v);
      if (t < this.tick) {
        const pred = this.predicted[p].get(t);
        if (!sameBytes(pred, v)) this.rollbackTo = Math.min(this.rollbackTo, t);
        this.predicted[p].delete(t);
      }
    });
    while (this.inputs[p].has(this.confirmed[p] + 1)) this.confirmed[p]++;
  }

  _onReliable(m) {
    if (m.from === this.local) return;
    if (m.type === 'hash') this._gotHash(m.tick, m.hash, 'remote', m.epoch ?? 0);
    if (m.type === 'resync' && this.resync && this.local === this.authority) this.resyncWanted = true;
    if (m.type === 'state') {
      this.stats.statesReceived++;
      this.stats.lastStateReceivedAt = this.tick;
      if (this.stateHandler) this.stateHandler(m);   // a custom policy
      else if (this.resync && (!this.pendingState || m.epoch > this.pendingState.epoch)) this.pendingState = m;
    }
  }

  /** The certificate of the match so far: per-tick inputs, state hashes and the chain head. */
  transcript() {
    return { players: this.players, inputSize: this.inputSize, delay: this.delay, hashEvery: this.hashEvery, dt: this.dt,
      inputs: this.log.inputs.slice(), hashes: { ...this.log.hashes }, chain: this.chain, ticks: this.chainTick + 1 };
  }
}

/**
 * Re-execute a transcript from scratch and check every recorded state hash and the chain head.
 * makeSim() → a fresh simulation in the starting state; applyInputs(sim, tick, inputs) as in the
 * session (game state included). Returns { ok, ticks, checkedHashes, firstMismatch }.
 */
async function verifyTranscript(tr, makeSim, applyInputs) {
  const sim = await makeSim();
  let chain = '0000000000000000', checked = 0;
  for (let t = 0; t < tr.ticks; t++) {
    const row = tr.inputs[t].map((r) => Uint8Array.from(r));
    applyInputs(sim, t, row);
    sim.step(tr.dt);
    chain = chainHash(chain, t, ...row);
    if (tr.hashes[t] !== undefined) {
      const h = await sim.hash();
      checked++;
      if (h !== tr.hashes[t]) return { ok: false, ticks: t + 1, checkedHashes: checked, firstMismatch: { tick: t, expected: tr.hashes[t], got: h } };
    }
  }
  return { ok: chain === tr.chain, ticks: tr.ticks, checkedHashes: checked, chain, firstMismatch: chain === tr.chain ? null : { chain: true } };
}
return { chainHash, RollbackSession, verifyTranscript };
})();

// ---- src/net/index.js
const __m40 = (() => {
const LoopbackNetwork = __m41.LoopbackNetwork;
const RTCPeer = __m41.RTCPeer;
const BroadcastSignaler = __m41.BroadcastSignaler;
const WebSocketSignaler = __m41.WebSocketSignaler;
const dataChannel = __m41.dataChannel;
const encode = __m41.encode;
const decode = __m41.decode;
const RollbackSession = __m42.RollbackSession;
const verifyTranscript = __m42.verifyTranscript;
const chainHash = __m42.chainHash;
// Peer-to-peer play: transports and rollback sessions for deterministic GPU simulations.
return { LoopbackNetwork, RTCPeer, BroadcastSignaler, WebSocketSignaler, dataChannel, encode, decode, RollbackSession, verifyTranscript, chainHash };
})();

// ---- src/verify/index.js
const __m43 = (() => {
const MSELoss = __m38.MSELoss;
// Certificates: small, independent checkers for the results of the big, optimised kernels.
//
// This follows the design of proof assistants (Lean's small kernel; Bend 2's `--verdict`, which
// re-checks every proof with a separate kernel verified in Lean) and of proof-carrying code:
// the producer (an optimised GPU kernel — tiled, subgrouped, autotuned) is untrusted; each
// result comes with a cheap check written as plainly as possible in a few lines, sharing no
// code with the producer. A checker is either exact (prefix sums, colourings, sortedness),
// probabilistic with a stated error bound (Freivalds), or a physical law with a tolerance
// (complementarity, conservation, Coulomb's cone). The laws of the library are listed in
// LAWS.md and run by proveAll() — the analogue of `bend PROOF.bend` before a release.



const verdict = (ok, evidence) => ({ ok, ...evidence });

/** Exclusive prefix sum: out[0] = 0 and out[k+1] − out[k] = in[k] (u32, wrapping). Exact. */
async function checkScan(ctx, input, output, n) {
  const bad = ctx.array('u32', 4);
  const k = ctx.kernel({
    name: 'verify-scan', buffers: { a: { type: 'u32', access: 'read' }, s: { type: 'u32', access: 'read' }, bad: { type: 'u32', atomic: true } },
    code: `
      let ok = select(s[i + 1u] - s[i] == a[i], true, i + 1u >= u.n) && (i != 0u || s[0] == 0u);
      if (!ok) { atomicAdd(&bad[0], 1u); atomicMin(&bad[1], i); }`,
  });
  ctx.fill(bad, 0);
  const init = ctx.kernel({ name: 'verify-scan-init', buffers: { bad: { type: 'u32' } }, code: 'if (i == 1u) { bad[1] = 0xffffffffu; }' });
  init.run(4, null, { bad });
  k.run(n, null, { a: input, s: output, bad });
  const r = await bad.read(2);
  bad.destroy();
  return verdict(r[0] === 0, { violations: r[0], first: r[0] ? r[1] : -1 });
}

/** keys[k] ≤ keys[k+1] for k < n − 1 (u32). Exact. */
async function checkSorted(ctx, keys, n) {
  const bad = ctx.array('u32', 4);
  ctx.kernel({ name: 'verify-sorted', buffers: { a: { type: 'u32', access: 'read' }, bad: { type: 'u32', atomic: true } },
    code: 'if (i + 1u < u.n && a[i] > a[i + 1u]) { atomicAdd(&bad[0], 1u); }' }).run(n, null, { a: keys, bad });
  const r = await bad.read(1);
  bad.destroy();
  return verdict(r[0] === 0, { violations: r[0] });
}

/**
 * a and b hold the same multiset of u32 values: two order-independent fingerprints (sums of
 * independent hashes) must match. A false accept needs a collision in both 32-bit sums.
 */
async function checkPermutation(ctx, a, b, n) {
  const fp = ctx.array('u32', 4);
  const k = ctx.kernel({ name: 'verify-perm', imports: ['hash'], buffers: { x: { type: 'u32', access: 'read' }, fp: { type: 'u32', atomic: true } },
    uniforms: { side: 'u32' },
    code: `let h = x[i]; atomicAdd(&fp[2u * u.side], pcg(h ^ 0x9e3779b9u)); atomicAdd(&fp[2u * u.side + 1u], pcg(h * 0x85ebca6bu + 7u));` });
  k.run(n, { side: 0 }, { x: a, fp });
  k.run(n, { side: 1 }, { x: b, fp });
  const r = await fp.read(4);
  fp.destroy();
  return verdict(r[0] === r[2] && r[1] === r[3], { fingerprints: Array.from(r) });
}

/**
 * Freivalds' certificate for C = op(A) op(B): for random ±1 vectors r, compare A(Br) with Cr.
 * O(n²) per trial against the O(n³) product; a wrong C survives a trial with probability ≤ 1/2.
 * o: { M, N, K, transA, transB, trials (default 4), tol (relative to the row's L2 norm, default 1e-4) }
 */
async function freivalds(ctx, A, B, C, o) {
  const { M, N, K } = o;
  const trials = o.trials ?? 4, tol = o.tol ?? 1e-4;
  const r = ctx.array('f32', N), Br = ctx.array('f32', K), ABr = ctx.array('f32', M), Cr = ctx.array('f32', M), mag = ctx.array('f32', M);
  const matvec = (name, rows, cols, trans) => ctx.kernel({
    name, buffers: { X: { type: 'f32', access: 'read' }, v: { type: 'f32', access: 'read' }, y: { type: 'f32' } },
    uniforms: { cols: 'u32' },
    code: `var s = 0.0; for (var c = 0u; c < u.cols; c++) { s += X[${trans ? `c * u.n + i` : 'i * u.cols + c'}] * v[c]; } y[i] = s;`,
  });
  // scale of (Cr)_i for random ±1 r: the row's L2 norm (an L1 bound would hide single bad entries)
  const absvec = ctx.kernel({ name: 'verify-rownorm', buffers: { X: { type: 'f32', access: 'read' }, y: { type: 'f32' } }, uniforms: { cols: 'u32' },
    code: 'var s = 0.0; for (var c = 0u; c < u.cols; c++) { let x = X[i * u.cols + c]; s += x * x; } y[i] = sqrt(s);' });
  let maxRel = 0;
  for (let t = 0; t < trials; t++) {
    const rv = new Float32Array(N);
    let s = (0x9e3779b9 * (t + 1)) >>> 0;
    for (let k = 0; k < N; k++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; rv[k] = s & 1 ? 1 : -1; }
    ctx.flush();
    r.write(rv);
    // B r : B is K×N (or N×K transposed)
    matvec('verify-mv-b', K, N, !!o.transB).run(K, { cols: N }, { X: B, v: r, y: Br });
    matvec('verify-mv-a', M, K, !!o.transA).run(M, { cols: K }, { X: A, v: Br, y: ABr });
    matvec('verify-mv-c', M, N, false).run(M, { cols: N }, { X: C, v: r, y: Cr });
    absvec.run(M, { cols: N }, { X: C, y: mag });
    const [x, y, m] = [await ABr.read(), await Cr.read(), await mag.read()];
    for (let k = 0; k < M; k++) maxRel = Math.max(maxRel, Math.abs(x[k] - y[k]) / Math.max(1e-6, m[k], Math.abs(x[k])));
  }
  for (const b of [r, Br, ABr, Cr, mag]) b.destroy();
  return verdict(maxRel <= tol, { maxRelative: maxRel, trials, falseAcceptBound: 2 ** -trials });
}

/** No two dynamic bodies of one colour share a contact or a joint (RigidWorld, after a step). Exact. */
async function checkColoring(world) {
  const nc = (await world.cnts.read(8))[1];
  const colors = await world.color[20 & 1].read(world.count);
  const raw = await world.C[(world.frame - 1) & 1].read(Math.max(1, nc) * 6);
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  const bodies = await world.readBodies();
  const dyn = (b) => b !== 0xffffffff && b < world.count;
  const massOf = async () => { const B = await world.B.read(world.count * 12); return (k) => B[k * 48 + 36]; };
  const m = await massOf();
  let conflicts = 0;
  for (let k = 0; k < nc; k++) {
    const a = u[k * 24], b = u[k * 24 + 1], sensor = u[k * 24 + 3];
    if (sensor || !dyn(b) || m(a) <= 0 || m(b) <= 0) continue;
    if ((bodies.flags[a] | bodies.flags[b]) & 6) continue;
    if (colors[a] === colors[b]) conflicts++;
  }
  for (const j of world.joints) if (dyn(j.b) && m(j.a) > 0 && m(j.b) > 0 && colors[j.a] === colors[j.b]) conflicts++;
  return verdict(conflicts === 0, { conflicts, contacts: nc });
}

/**
 * Contact conditions of the last step (RigidWorld): force sign λ_n ≤ 0, isotropic Coulomb cone
 * |λ_t| ≤ μ|λ_n|, bounded penetration, and complementarity |λ_n|·gap ≈ 0 for separated contacts.
 * o: { maxPenetration (m, default 0.02), coneTol (default 1e-3 relative) }
 */
async function checkContacts(world, o = {}) {
  const nc = (await world.cnts.read(8))[1];
  if (!nc) return verdict(true, { contacts: 0 });
  const raw = await world.C[(world.frame - 1) & 1].read(nc * 6);
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  const B = await world.B.read(world.count * 12);
  const pose = (k) => ({ x: B.subarray(k * 48, k * 48 + 3), q: B.subarray(k * 48 + 4, k * 48 + 8) });
  const rot = (q, v) => {
    const [x, y, z, w] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
  };
  let signViol = 0, coneViol = 0, maxPen = 0, maxCompl = 0;
  for (let k = 0; k < nc; k++) {
    const o6 = k * 24;
    if (u[o6 + 3]) continue;   // sensor
    const a = u[o6], b = u[o6 + 1];
    const n = raw.subarray(o6 + 4, o6 + 7), mu = raw[o6 + 7];
    const lam = raw.subarray(o6 + 16, o6 + 19);
    if (lam[0] > 1e-6) signViol++;
    if (Math.hypot(lam[1], lam[2]) > mu * Math.abs(lam[0]) * (1 + (o.coneTol ?? 1e-3)) + 1e-6) coneViol++;
    // gap between the two anchors along the normal at the end of the step
    const A = pose(a), pa = rot(A.q, raw.subarray(o6 + 8, o6 + 11)).map((v, i) => v + A.x[i]);
    let pb = Array.from(raw.subarray(o6 + 12, o6 + 15));
    if (b !== 0xffffffff) { const P = pose(b); pb = rot(P.q, pb).map((v, i) => v + P.x[i]); }
    const gap = (pa[0] - pb[0]) * n[0] + (pa[1] - pb[1]) * n[1] + (pa[2] - pb[2]) * n[2];
    maxPen = Math.max(maxPen, -gap);
    if (gap > 0.005) maxCompl = Math.max(maxCompl, Math.abs(lam[0]) * gap);
  }
  const ok = signViol === 0 && coneViol === 0 && maxPen <= (o.maxPenetration ?? 0.02);
  return verdict(ok, { contacts: nc, signViolations: signViol, coneViolations: coneViol, maxPenetration: maxPen, complementarity: maxCompl });
}

/** Grid mass and momentum after P2G equal the particles' (MLS-MPM conservation). */
async function checkMPM(sim, tol = 1e-4) {
  const n = sim.count, raw = await sim.P.read(n * 8);
  sim.p2g();
  const g = new Int32Array((await sim.grid.read()).buffer);
  const FP = 2 ** 22;
  const mats = sim.materials;
  const vol = 1 / sim.ppcAxis ** 3;
  const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
  let pm = 0, px = 0, py = 0, pz = 0, gm = 0, gx = 0, gy = 0, gz = 0;
  for (let k = 0; k < n; k++) {
    const m = (mats[u[32 * k + 3]]?.density ?? 1) * vol;
    pm += m; px += m * raw[32 * k + 4]; py += m * raw[32 * k + 5]; pz += m * raw[32 * k + 6];
  }
  for (let k = 0; k < g.length; k += 4) { gx += g[k]; gy += g[k + 1]; gz += g[k + 2]; gm += g[k + 3]; }
  gm /= FP; gx /= FP; gy /= FP; gz /= FP;
  const massErr = Math.abs(gm - pm) / pm;
  const momErr = Math.hypot(gx - px, gy - py, gz - pz) / Math.max(1e-9, pm);
  return verdict(massErr < tol && momErr < tol + (n * 27 * 4) / FP / pm, { massError: massErr, momentumError: momErr });
}

/**
 * Backpropagation against central finite differences on random parameters (an MLP with
 * relu/tanh/sigmoid/none activations, MSE loss on X → T). Returns the worst relative error.
 */
async function checkGradient(net, X, T, batch, o = {}) {
  const ctx = net.ctx, outN = net.layers[net.layers.length - 1].out;
  const loss = new MSELoss(ctx, batch * outN);
  const lossAt = async () => { net.forward(X, batch); loss.run(net.output, T, batch * outN); return loss.readLoss(); };
  net.forward(X, batch);
  net.backward(loss.run(net.output, T, batch * outN), batch);
  const samples = o.samples ?? 12, eps = o.eps ?? 1e-2;
  let worst = 0, s = 12345;
  for (let q = 0; q < samples; q++) {
    s = (s * 16807) % 2147483647;
    const l = net.layers[s % net.layers.length];
    const W = await l.W.read(), G = await l.gW.read();
    const idx = s % W.length;
    const w0 = W[idx];
    W[idx] = w0 + eps; ctx.flush(); l.W.write(W); const lp = await lossAt();
    W[idx] = w0 - eps; ctx.flush(); l.W.write(W); const lm = await lossAt();
    W[idx] = w0; ctx.flush(); l.W.write(W);
    const fd = (lp - lm) / (2 * eps);
    worst = Math.max(worst, Math.abs(fd - G[idx]) / Math.max(1e-4, Math.abs(fd), Math.abs(G[idx])));
  }
  return verdict(worst < (o.tol ?? 0.05), { maxRelativeError: worst, samples });
}

/** The same build run twice gives bit-identical state hashes (needs a deterministic RigidWorld). */
async function checkDeterminism(makeWorld, steps = 120, every = 30) {
  const runs = [];
  for (let r = 0; r < 2; r++) {
    const w = await makeWorld();
    const h = [];
    for (let s = 0; s < steps; s++) { w.step(1 / 60); if (s % every === every - 1) h.push(await w.hash()); }
    runs.push(h);
  }
  return verdict(runs[0].join() === runs[1].join(), { hashes: runs });
}

/** snapshot → step n → restore → step n reproduces the same state hash. */
async function checkReplay(world, steps = 60) {
  const snap = world.snapshot();
  for (let s = 0; s < steps; s++) world.step(1 / 60);
  const a = await world.hash();
  world.restore(snap);
  for (let s = 0; s < steps; s++) world.step(1 / 60);
  const b = await world.hash();
  return verdict(a === b, { first: a, replay: b });
}

/**
 * The laws of the library (see LAWS.md), each an executable check against a fresh scene.
 * proveAll(gpu) runs them all and returns [{ name, statement, ok, evidence }].
 */
const LAWS = [
  { name: 'scan', statement: 'Scan.run computes an exclusive prefix sum: s[0] = 0, s[k+1] − s[k] = a[k].', async check(gpu, SI) {
    const n = 1 << 18, a = new Uint32Array(n).map((_, k) => (k * 2654435761) % 1000);
    const A = gpu.array('u32', a), S = gpu.array('u32', n + 1);
    new SI.Scan(gpu, n + 1).run(A, S, n);
    return checkScan(gpu, A, S, n);
  } },
  { name: 'gemm', statement: 'gemm computes C = A·B (Freivalds certificate, false-accept probability ≤ 2⁻⁸).', async check(gpu, SI) {
    const M = 777, N = 513, K = 1029;
    const rnd = (n, s) => { const x = new Float32Array(n); for (let k = 0; k < n; k++) { s = (s * 16807) % 2147483647; x[k] = s / 2147483647 - 0.5; } return x; };
    const A = gpu.array('f32', rnd(M * K, 3)), B = gpu.array('f32', rnd(K * N, 7)), C = gpu.array('f32', M * N);
    SI.nn.gemm(gpu, A, B, C, { M, N, K });
    return freivalds(gpu, A, B, C, { M, N, K, trials: 8 });
  } },
  { name: 'gemm-rejects', statement: 'Freivalds rejects a product with a single corrupted entry.', async check(gpu, SI) {
    const M = 300, N = 200, K = 400;
    const rnd = (n, s) => { const x = new Float32Array(n); for (let k = 0; k < n; k++) { s = (s * 16807) % 2147483647; x[k] = s / 2147483647 - 0.5; } return x; };
    const A = gpu.array('f32', rnd(M * K, 5)), B = gpu.array('f32', rnd(K * N, 9)), C = gpu.array('f32', M * N);
    SI.nn.gemm(gpu, A, B, C, { M, N, K });
    const c = await C.read(); c[12345] += 0.5; gpu.flush(); C.write(c);
    const r = await freivalds(gpu, A, B, C, { M, N, K, trials: 8 });
    return verdict(!r.ok, { detected: !r.ok, maxRelative: r.maxRelative });
  } },
  { name: 'colouring', statement: 'No two dynamic bodies of one colour share a contact or a joint.', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 2048 });
    let s = 7; const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    for (let k = 0; k < 1000; k++) w.addBox({ position: [(r() - 0.5) * 5, 0.3 + k * 0.02, (r() - 0.5) * 5], half: [0.2, 0.2, 0.2], rotation: [r(), r(), r()] });
    for (let k = 0; k < 120; k++) w.step(1 / 60);
    return checkColoring(w);
  } },
  { name: 'contacts', statement: 'Contact forces push (λ_n ≤ 0), stay in the Coulomb cone (|λ_t| ≤ μ|λ_n|), and a settled 600-body pile of 20–60 g bodies penetrates less than 2 cm (8 iterations; heavier stacks need substeps).', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 2048 });
    let s = 9; const r = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    for (let k = 0; k < 600; k++) (r() < 0.5 ? w.addBox({ position: [(r() - 0.5) * 4, 0.3 + k * 0.02, (r() - 0.5) * 4], half: [0.2, 0.15, 0.25], rotation: [r(), r(), r()] })
      : w.addSphere({ position: [(r() - 0.5) * 4, 0.3 + k * 0.02, (r() - 0.5) * 4], radius: 0.18 }));
    for (let k = 0; k < 360; k++) w.step(1 / 60);
    return checkContacts(w, { maxPenetration: 0.02 });
  } },
  { name: 'determinism', statement: 'A deterministic RigidWorld is a function of its inputs: two runs give bit-identical states.', async check(gpu) {
    return checkDeterminism(() => {
      const w = gpu.rigid({ maxBodies: 1024, deterministic: true });
      for (let k = 0; k < 300; k++) w.addBox({ position: [(k % 7) * 0.45 - 1.5, 0.3 + Math.floor(k / 7) * 0.45, (k % 5) * 0.1], half: [0.2, 0.2, 0.2], rotation: [k * 0.1, k * 0.2, 0] });
      return w;
    });
  } },
  { name: 'replay', statement: 'restore(snapshot) followed by the same steps reproduces the same state.', async check(gpu) {
    const w = gpu.rigid({ maxBodies: 1024, deterministic: true });
    for (let k = 0; k < 200; k++) w.addSphere({ position: [(k % 5) * 0.5, 0.5 + Math.floor(k / 5) * 0.5, (k % 3) * 0.3], radius: 0.2 });
    for (let k = 0; k < 30; k++) w.step(1 / 60);
    return checkReplay(w, 60);
  } },
  { name: 'mpm-conservation', statement: 'MLS-MPM P2G conserves mass and momentum exactly (fixed-point sums).', async check(gpu) {
    const sim = gpu.mpm({ grid: 32, maxParticles: 50000 });
    sim.add({ box: [[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]] }, { material: 'water', velocity: [0.3, 0.1, -0.2] });
    for (let k = 0; k < 10; k++) sim.step();
    return checkMPM(sim);
  } },
  { name: 'backprop', statement: 'MLP.backward equals the finite-difference gradient of the MSE loss.', async check(gpu, SI) {
    const net = new SI.nn.MLP(gpu, { layers: [3, 16, 16, 2], act: 'tanh', out: 'none', maxBatch: 64, seed: 4 });
    const X = gpu.array('f32', new Float32Array(64 * 3).map((_, k) => Math.sin(k * 0.37))), T = gpu.array('f32', new Float32Array(64 * 2).map((_, k) => Math.cos(k)));
    return checkGradient(net, X, T, 64);
  } },
];

async function proveAll(gpu, SI, only) {
  const out = [];
  for (const law of LAWS) {
    if (only && !only.includes(law.name)) continue;
    const t0 = performance.now();
    let r;
    try { r = await law.check(gpu, SI); } catch (e) { r = { ok: false, error: String(e.message || e) }; }
    out.push({ name: law.name, statement: law.statement, ok: !!r.ok, ms: Math.round(performance.now() - t0), evidence: r });
  }
  return out;
}
return { checkScan, checkSorted, checkPermutation, freivalds, checkColoring, checkContacts, checkMPM, checkGradient, checkDeterminism, checkReplay, LAWS, proveAll };
})();

// ---- src/shaders/noise.js
const __m45 = (() => {
const defineModule = __m5.defineModule;


// Procedural noise with analytic derivatives.
//
// Design (first principles): a noise evaluation is lattice hashing + gradient
// generation + kernel falloff. We make each part cheap:
//  * hashing: (x*K1) ^ (y*K2) ^ (z*K3) is linear in each coordinate mod 2^32, so
//    the corners' products are the base product plus a constant — selects and
//    adds instead of 12 multiplies; one lowbias32-style finaliser per corner.
//  * gradients: unpack4x8snorm(hash) gives a random vector in [-1,1]^3 in one
//    instruction; normalising it makes the gradient set isotropic.
//  * simplex kernels use r^2 = 0.5 so contributions vanish exactly at the
//    simplex neighbourhood boundary -> C2-continuous value and gradient
//    (curl noise and normal mapping need smooth derivatives).
// Output ranges are calibrated to roughly [-1, 1].

defineModule('noise', {
  deps: ['hash'],
  doc: 'Simplex 2D/3D/4D, Perlin, value and Worley noise, fBm, ridged, curl noise — all with analytic gradients where useful.',
  code: /* wgsl */`
const SI_K = vec4u(0x8da6b343u, 0xd8163841u, 0xcb1ab31fu, 0x9e3779b1u);

fn si_fin4(h0: vec4u) -> vec4u {
  var h = (h0 ^ (h0 >> vec4u(16u))) * 0x7feb352du;
  h = (h ^ (h >> vec4u(15u))) * 0x846ca68bu;
  return h ^ (h >> vec4u(16u));
}
fn si_fin(h0: u32) -> u32 {
  var h = (h0 ^ (h0 >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
fn si_g3(h: u32) -> vec3f {
  let g = unpack4x8snorm(h).xyz;
  return g * inverseSqrt(max(dot(g, g), 1e-4));
}
fn si_g2(h: u32) -> vec2f {
  let g = unpack2x16snorm(h);
  return g * inverseSqrt(max(dot(g, g), 1e-6));
}
fn si_g4(h: u32) -> vec4f {
  let g = unpack4x8snorm(h);
  return g * inverseSqrt(max(dot(g, g), 1e-4));
}

// ------------------------------------------------------------------ simplex 2D
/** 2D simplex noise: (value, d/dx, d/dy). */
fn simplex2d(p: vec2f) -> vec3f {
  const K1 = 0.366025403784439;  // (sqrt(3)-1)/2
  const K2 = 0.211324865405187;  // (3-sqrt(3))/6
  let s = floor(p + (p.x + p.y) * K1);
  let a = p - s + (s.x + s.y) * K2;
  let m = select(0.0, 1.0, a.x > a.y);
  let o = vec2f(m, 1.0 - m);
  let b = a - o + K2;
  let c = a - 1.0 + 2.0 * K2;
  let bx = u32(i32(s.x)) * SI_K.x;
  let by = u32(i32(s.y)) * SI_K.y;
  let ox = select(0u, SI_K.x, m > 0.5);
  let oy = select(SI_K.y, 0u, m > 0.5);
  let h = si_fin4(vec4u(bx ^ by, (bx + ox) ^ (by + oy), (bx + SI_K.x) ^ (by + SI_K.y), 0u));
  let ga = si_g2(h.x); let gb = si_g2(h.y); let gc = si_g2(h.z);
  let w = max(0.5 - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
  let w2 = w * w; let w4 = w2 * w2;
  let d = vec3f(dot(ga, a), dot(gb, b), dot(gc, c));
  let n = dot(w4, d);
  let t = w2 * w * d * 8.0;
  let g = w4.x * ga + w4.y * gb + w4.z * gc - (t.x * a + t.y * b + t.z * c);
  return vec3f(n, g) * 96.0;
}
fn simplex2(p: vec2f) -> f32 {
  const K1 = 0.366025403784439;
  const K2 = 0.211324865405187;
  let s = floor(p + (p.x + p.y) * K1);
  let a = p - s + (s.x + s.y) * K2;
  let m = select(0.0, 1.0, a.x > a.y);
  let b = a - vec2f(m, 1.0 - m) + K2;
  let c = a - 1.0 + 2.0 * K2;
  let bx = u32(i32(s.x)) * SI_K.x;
  let by = u32(i32(s.y)) * SI_K.y;
  let ox = select(0u, SI_K.x, m > 0.5);
  let oy = select(SI_K.y, 0u, m > 0.5);
  let h = si_fin4(vec4u(bx ^ by, (bx + ox) ^ (by + oy), (bx + SI_K.x) ^ (by + SI_K.y), 0u));
  let w = max(0.5 - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
  let w2 = w * w;
  return 96.0 * dot(w2 * w2, vec3f(dot(si_g2(h.x), a), dot(si_g2(h.y), b), dot(si_g2(h.z), c)));
}

// ------------------------------------------------------------------ simplex 3D
fn si_corners3(s: vec3f, i1: vec3f, i2: vec3f) -> vec4u {
  let b = vec3u(vec3i(s)) * SI_K.xyz;
  let o1 = select(vec3u(0u), SI_K.xyz, i1 > vec3f(0.5));
  let o2 = select(vec3u(0u), SI_K.xyz, i2 > vec3f(0.5));
  let c1 = b + o1; let c2 = b + o2; let c3 = b + SI_K.xyz;
  return si_fin4(vec4u(b.x ^ b.y ^ b.z, c1.x ^ c1.y ^ c1.z, c2.x ^ c2.y ^ c2.z, c3.x ^ c3.y ^ c3.z));
}
/** 3D simplex noise: (value, gradient). */
fn simplex3d(p: vec3f) -> vec4f {
  let s = floor(p + dot(p, vec3f(1.0 / 3.0)));
  let x0 = p - s + dot(s, vec3f(1.0 / 6.0));
  let e = step(x0.yzx, x0);
  let l = 1.0 - e;
  let i1 = min(e, l.zxy);
  let i2 = max(e, l.zxy);
  let x1 = x0 - i1 + 1.0 / 6.0;
  let x2 = x0 - i2 + 1.0 / 3.0;
  let x3 = x0 - 0.5;
  let h = si_corners3(s, i1, i2);
  let g0 = si_g3(h.x); let g1 = si_g3(h.y); let g2 = si_g3(h.z); let g3 = si_g3(h.w);
  let w = max(0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  let w2 = w * w; let w4 = w2 * w2;
  let d = vec4f(dot(g0, x0), dot(g1, x1), dot(g2, x2), dot(g3, x3));
  let t = w2 * w * d * 8.0;
  let g = w4.x * g0 + w4.y * g1 + w4.z * g2 + w4.w * g3 - (t.x * x0 + t.y * x1 + t.z * x2 + t.w * x3);
  return vec4f(dot(w4, d), g) * 104.0;
}
fn simplex3(p: vec3f) -> f32 {
  let s = floor(p + dot(p, vec3f(1.0 / 3.0)));
  let x0 = p - s + dot(s, vec3f(1.0 / 6.0));
  let e = step(x0.yzx, x0);
  let l = 1.0 - e;
  let i1 = min(e, l.zxy);
  let i2 = max(e, l.zxy);
  let x1 = x0 - i1 + 1.0 / 6.0;
  let x2 = x0 - i2 + 1.0 / 3.0;
  let x3 = x0 - 0.5;
  let h = si_corners3(s, i1, i2);
  let w = max(0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  let w2 = w * w;
  let d = vec4f(dot(si_g3(h.x), x0), dot(si_g3(h.y), x1), dot(si_g3(h.z), x2), dot(si_g3(h.w), x3));
  return 104.0 * dot(w2 * w2, d);
}

// ------------------------------------------------------------------ simplex 4D
/** 4D simplex noise (e.g. animated 3D noise: simplex4(vec4f(p, t))). */
fn simplex4(v: vec4f) -> f32 {
  const F4 = 0.309016994374947;
  const G4 = 0.138196601125011;
  let s = floor(v + dot(v, vec4f(F4)));
  let x0 = v - s + dot(s, vec4f(G4));
  // rank of each component -> simplex corners
  let isX = step(x0.yzw, x0.xxx);
  let isYZ = step(x0.zww, x0.yyz);
  var r = vec4f(isX.x + isX.y + isX.z, 1.0 - isX);
  r.y += isYZ.x + isYZ.y;
  r.z += 1.0 - isYZ.x;
  r.w += 1.0 - isYZ.y;
  r.z += isYZ.z;
  r.w += 1.0 - isYZ.z;
  let i3 = clamp(r, vec4f(0.0), vec4f(1.0));
  let i2 = clamp(r - 1.0, vec4f(0.0), vec4f(1.0));
  let i1 = clamp(r - 2.0, vec4f(0.0), vec4f(1.0));
  let x1 = x0 - i1 + G4;
  let x2 = x0 - i2 + 2.0 * G4;
  let x3 = x0 - i3 + 3.0 * G4;
  let x4 = x0 - 1.0 + 4.0 * G4;
  let b = vec4u(vec4i(s)) * SI_K;
  let c1 = b + select(vec4u(0u), SI_K, i1 > vec4f(0.5));
  let c2 = b + select(vec4u(0u), SI_K, i2 > vec4f(0.5));
  let c3 = b + select(vec4u(0u), SI_K, i3 > vec4f(0.5));
  let c4 = b + SI_K;
  let h = si_fin4(vec4u(b.x ^ b.y ^ b.z ^ b.w, c1.x ^ c1.y ^ c1.z ^ c1.w, c2.x ^ c2.y ^ c2.z ^ c2.w, c3.x ^ c3.y ^ c3.z ^ c3.w));
  let h4 = si_fin(c4.x ^ c4.y ^ c4.z ^ c4.w);
  let w0 = max(0.6 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3f(0.0));
  let w1 = max(0.6 - vec2f(dot(x3, x3), dot(x4, x4)), vec2f(0.0));
  let a0 = w0 * w0; let a1 = w1 * w1;
  return 45.0 * (dot(a0 * a0, vec3f(dot(si_g4(h.x), x0), dot(si_g4(h.y), x1), dot(si_g4(h.z), x2)))
               + dot(a1 * a1, vec2f(dot(si_g4(h.w), x3), dot(si_g4(h4), x4))));
}

// ------------------------------------------------------------------ Perlin / value
fn si_lat3(i: vec3f) -> array<u32, 8> {
  let b = vec3u(vec3i(i)) * SI_K.xyz;
  let c = b + SI_K.xyz;
  let lo = si_fin4(vec4u(b.x ^ b.y ^ b.z, c.x ^ b.y ^ b.z, b.x ^ c.y ^ b.z, c.x ^ c.y ^ b.z));
  let hi = si_fin4(vec4u(b.x ^ b.y ^ c.z, c.x ^ b.y ^ c.z, b.x ^ c.y ^ c.z, c.x ^ c.y ^ c.z));
  return array<u32, 8>(lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w);
}
/** 3D gradient (Perlin) noise with quintic fade: (value, gradient). */
fn perlin3d(x: vec3f) -> vec4f {
  let i = floor(x);
  let f = x - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let h = si_lat3(i);
  let ga = si_g3(h[0]); let gb = si_g3(h[1]); let gc = si_g3(h[2]); let gd = si_g3(h[3]);
  let ge = si_g3(h[4]); let gf = si_g3(h[5]); let gg = si_g3(h[6]); let gh = si_g3(h[7]);
  let va = dot(ga, f);
  let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
  let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
  let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
  let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
  let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
  let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
  let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
  let k = -va + vb + vc - vd + ve - vf - vg + vh;
  let v = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf) + k * u.x * u.y * u.z;
  let g = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
        + du * (vec3f(vb, vc, ve) - va
               + u.yzx * vec3f(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
               + u.zxy * vec3f(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
               + u.yzx * u.zxy * k);
  return vec4f(v, g) * 1.42;
}
fn perlin3(x: vec3f) -> f32 { return perlin3d(x).x; }

/** 3D value noise in [-1, 1] (cheapest lattice noise). */
fn value3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = x - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let h = si_lat3(i);
  let a = vec4f(u2f(h[0]), u2f(h[1]), u2f(h[2]), u2f(h[3]));
  let b = vec4f(u2f(h[4]), u2f(h[5]), u2f(h[6]), u2f(h[7]));
  let z = mix(a, b, u.z);
  let y = mix(z.xy, z.zw, u.y);
  return mix(y.x, y.y, u.x) * 2.0 - 1.0;
}

// ------------------------------------------------------------------ cellular
/** 2D Worley noise: (F1, F2) Euclidean distances to the nearest feature points. */
fn worley2(p: vec2f) -> vec2f {
  let i = floor(p);
  let f = p - i;
  var d = vec2f(8.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2f(f32(x), f32(y));
      let c = vec2i(i) + vec2i(x, y);
      let h = pcg2d(bitcast<vec2u>(c));
      let r = o + u2f2(h) - f;
      let dd = dot(r, r);
      if (dd < d.x) { d = vec2f(dd, d.x); } else if (dd < d.y) { d.y = dd; }
    }
  }
  return sqrt(d);
}
/** 3D Worley noise: (F1, F2). */
fn worley3(p: vec3f) -> vec2f {
  let i = floor(p);
  let f = p - i;
  var d = vec2f(8.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3f(f32(x), f32(y), f32(z));
        let h = pcg3d(bitcast<vec3u>(vec3i(i) + vec3i(x, y, z)));
        let r = o + u2f3(h) - f;
        let dd = dot(r, r);
        if (dd < d.x) { d = vec2f(dd, d.x); } else if (dd < d.y) { d.y = dd; }
      }
    }
  }
  return sqrt(d);
}

// ------------------------------------------------------------------ fractal sums
// Octaves are rotated (orthonormal, det=1) and scaled by ~2 to hide lattice alignment.
const SI_OCT = mat3x3f(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);

fn fbm2(p: vec2f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * simplex2(q);
    q = mat2x2f(1.6, 1.2, -1.2, 1.6) * q;
    a *= 0.5;
  }
  return s;
}
fn fbm3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * simplex3(q);
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}
/** fBm with exact gradient (chain rule through the octave rotations). */
fn fbm3d(p: vec3f, octaves: i32) -> vec4f {
  var q = p; var a = 0.5; var s = 0.0; var g = vec3f(0.0);
  var J = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // dq/dp
  for (var k = 0; k < octaves; k++) {
    let n = simplex3d(q);
    s += a * n.x;
    g += a * (transpose(J) * n.yzw);
    q = SI_OCT * q * 2.02;
    J = SI_OCT * J * 2.02;
    a *= 0.5;
  }
  return vec4f(s, g);
}
/** Ridged multifractal (sharp crests), roughly [0, 1]. */
fn ridged3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0; var prev = 1.0;
  for (var k = 0; k < octaves; k++) {
    var n = 1.0 - abs(simplex3(q));
    n = n * n;
    s += n * a * prev;
    prev = n;
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}
fn turbulence3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * abs(simplex3(q));
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}

// ------------------------------------------------------------------ curl noise
/**
 * Divergence-free 3D noise field (Bridson et al. 2007): curl of a vector
 * potential of three decorrelated simplex fields, using analytic gradients.
 */
fn curl3(p: vec3f) -> vec3f {
  let a = simplex3d(p);
  let b = simplex3d(p + vec3f(31.416, -47.853, 12.793));
  let c = simplex3d(p + vec3f(-233.145, -113.408, -185.31));
  return vec3f(c.z - b.w, a.w - c.y, b.y - a.z);
}
/** 2D curl noise: rotated gradient of a scalar stream function. */
fn curl2(p: vec2f) -> vec2f {
  let n = simplex2d(p);
  return vec2f(n.z, -n.y);
}
`,
});
return {  };
})();

// ---- src/shaders/sdf.js
const __m46 = (() => {
const defineModule = __m5.defineModule;


// Signed distance functions. Exact where possible (|∇d| = 1), otherwise a
// bound (marked). Conventions: primitives are centred at the origin; transform
// the query point, not the shape. Operators take/return distances.
defineModule('sdf', {
  deps: ['math'],
  doc: 'SDF primitives (3D/2D), boolean and smooth operators, domain operators.',
  code: /* wgsl */`
// ---------------------------------------------------------------- 3D primitives
fn sdSphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }
fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max3(q), 0.0);
}
fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
  let q = abs(p) - b + r;
  return length(max(q, vec3f(0.0))) + min(max3(q), 0.0) - r;
}
fn sdBoxFrame(p0: vec3f, b: vec3f, e: f32) -> f32 {
  let p = abs(p0) - b;
  let q = abs(p + e) - e;
  return min(min(
    length(max(vec3f(p.x, q.y, q.z), vec3f(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3f(q.x, p.y, q.z), vec3f(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3f(q.x, q.y, p.z), vec3f(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0));
}
/** Torus in the xz-plane: t = (major radius, minor radius). */
fn sdTorus(p: vec3f, t: vec2f) -> f32 { return length(vec2f(length(p.xz) - t.x, p.y)) - t.y; }
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
/** Vertical capped cylinder: half-height h, radius r. */
fn sdCylinder(p: vec3f, h: f32, r: f32) -> f32 {
  let d = abs(vec2f(length(p.xz), p.y)) - vec2f(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
/** Vertical capped cone: half-height h, bottom radius r1, top radius r2. */
fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
fn sdPlane(p: vec3f, n: vec3f, h: f32) -> f32 { return dot(p, n) + h; }
/** Octahedron |x|+|y|+|z| = s (bound). */
fn sdOctahedron(p: vec3f, s: f32) -> f32 {
  let q = abs(p);
  return (q.x + q.y + q.z - s) * 0.57735027;
}
/** Ellipsoid with radii r (bound, good near the surface). */
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}
/** Hexagonal prism along z: h = (radius, half-length). */
fn sdHexPrism(p0: vec3f, h: vec2f) -> f32 {
  let k = vec3f(-0.8660254, 0.5, 0.57735);
  var p = abs(p0);
  let t = 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
  p = vec3f(p.xy - t, p.z);
  let d = vec2f(length(p.xy - vec2f(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x), p.z - h.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
/** Chain link: half-length le, ring radius r1, thickness r2. */
fn sdLink(p: vec3f, le: f32, r1: f32, r2: f32) -> f32 {
  let q = vec3f(p.x, max(abs(p.y) - le, 0.0), p.z);
  return length(vec2f(length(q.xy) - r1, q.z)) - r2;
}
/** Infinite cylinder along y through c.xy (xz plane), radius c.z. */
fn sdInfCylinder(p: vec3f, c: vec3f) -> f32 { return length(p.xz - c.xy) - c.z; }

// ---------------------------------------------------------------- 2D primitives
fn sdCircle(p: vec2f, r: f32) -> f32 { return length(p) - r; }
fn sdBox2(p: vec2f, b: vec2f) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}
/** Rounded box with per-corner radii r = (top-right, bottom-right, top-left, bottom-left). */
fn sdRoundBox2(p: vec2f, b: vec2f, r4: vec4f) -> f32 {
  let r2 = select(r4.zw, r4.xy, p.x > 0.0);
  let r = select(r2.y, r2.x, p.y > 0.0);
  let q = abs(p) - b + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - r;
}
fn sdSegment2(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}
fn sdEquilateralTriangle(p0: vec2f, r: f32) -> f32 {
  let k = sqrt(3.0);
  var p = vec2f(abs(p0.x) - r, p0.y + r / k);
  if (p.x + k * p.y > 0.0) { p = vec2f(p.x - k * p.y, -k * p.x - p.y) / 2.0; }
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
fn sdHexagon(p0: vec2f, r: f32) -> f32 {
  let k = vec3f(-0.866025404, 0.5, 0.577350269);
  var p = abs(p0);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2f(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}

// ---------------------------------------------------------------- operators
fn opUnion(a: f32, b: f32) -> f32 { return min(a, b); }
/** a minus b. */
fn opSubtract(a: f32, b: f32) -> f32 { return max(a, -b); }
fn opIntersect(a: f32, b: f32) -> f32 { return max(a, b); }
/** Quadratic polynomial smooth minimum; k = blend radius in distance units. */
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let kk = 4.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  return min(a, b) - h * h * kk * 0.25;
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
/** Cubic smooth minimum (C2). */
fn sminCubic(a: f32, b: f32, k: f32) -> f32 {
  let kk = 6.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  return min(a, b) - h * h * h * kk * (1.0 / 6.0);
}
/** Smooth minimum that also returns the blend factor (0 -> a, 1 -> b) for mixing materials. */
fn sminMix(a: f32, b: f32, k: f32) -> vec2f {
  let kk = 4.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  let m = h * h * 0.5;
  let s = m * kk * 0.5;
  return select(vec2f(b - s, 1.0 - m), vec2f(a - s, m), a < b);
}
fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 { return smin(a, b, k); }
fn opSmoothSubtract(a: f32, b: f32, k: f32) -> f32 { return smax(a, -b, k); }
fn opSmoothIntersect(a: f32, b: f32, k: f32) -> f32 { return smax(a, b, k); }
fn opRound(d: f32, r: f32) -> f32 { return d - r; }
fn opOnion(d: f32, t: f32) -> f32 { return abs(d) - t; }
/** Union of two (distance, material) pairs. */
fn opU(a: vec2f, b: vec2f) -> vec2f { return select(b, a, a.x < b.x); }

// ---------------------------------------------------------------- domain operators
/** Infinite repetition with period s (returns the local coordinate). */
fn opRepeat(p: vec3f, s: vec3f) -> vec3f { return p - s * round(p / s); }
/** Repetition limited to cells -l..l. */
fn opRepeatLimited(p: vec3f, s: f32, l: vec3f) -> vec3f { return p - s * clamp(round(p / s), -l, l); }
fn opTwist(p: vec3f, k: f32) -> vec3f {
  let c = cos(k * p.y); let s = sin(k * p.y);
  let q = mat2x2f(c, s, -s, c) * p.xz;
  return vec3f(q.x, p.y, q.y);
}
fn opBend(p: vec3f, k: f32) -> vec3f {
  let c = cos(k * p.x); let s = sin(k * p.x);
  let q = mat2x2f(c, s, -s, c) * p.xy;
  return vec3f(q, p.z);
}
fn opElongate(p: vec3f, h: vec3f) -> vec3f { return p - clamp(p, -h, h); }
fn opSymX(p: vec3f) -> vec3f { return vec3f(abs(p.x), p.yz); }
/** Revolve a 2D profile around y at radius o: evaluate sd2D(opRevolve(p, o)). */
fn opRevolve(p: vec3f, o: f32) -> vec2f { return vec2f(length(p.xz) - o, p.y); }
/** Extrude a 2D distance d along z with half-depth h. */
fn opExtrude(p: vec3f, d: f32, h: f32) -> f32 {
  let w = vec2f(d, abs(p.z) - h);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
}
`,
});
return {  };
})();

// ---- src/shaders/raymarch.js
const __m47 = (() => {
const defineModule = __m5.defineModule;


// Sphere tracing for user scenes. The importing shader must define
//   fn map(p: vec3f) -> vec2f   // (signed distance, material id)
//
// The fast path is two-pass cone marching: neighbouring pixel rays share most
// of their empty space, so one conservative cone per T×T pixel tile walks that
// space once (coneMarch), and every pixel ray then starts at the tile's safe
// distance. A distance field is 1-Lipschitz, so if map(p) > r the whole disc of
// radius r around p is empty — the pre-pass can never skip a surface. On a
// typical scene this removes ~80% of per-pixel map() calls (bench: 'raymarch').
//
// raymarch() also implements over-relaxed sphere tracing (Keinert et al. 2014).
// Measured lesson: fewer steps per ray did not translate into less time on GPUs,
// because SIMD lanes wait for the slowest ray and the bookkeeping costs ALU.
defineModule('raymarch', {
  doc: 'Over-relaxed sphere tracing, tetrahedral normals, soft shadows, ambient occlusion. Requires fn map(p: vec3f) -> vec2f.',
  code: /* wgsl */`
struct Hit { t: f32, mat: f32, steps: u32, hit: bool }

/** Trace a ray against map(). pixelAngle ≈ tan(fov/2) * 2 / screenHeight. */
fn raymarch(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, pixelAngle: f32) -> Hit {
  var omega = 1.6;
  var t = tmin;
  var candT = tmin;
  var candErr = 1e32;
  var candMat = 0.0;
  var prevR = 0.0;
  var stepLen = 0.0;
  let fsign = select(1.0, -1.0, map(ro + rd * tmin).x < 0.0);
  var k = 0u;
  for (; k < 256u; k++) {
    let m = map(ro + rd * t);
    let signedR = fsign * m.x;
    let r = abs(signedR);
    let fail = omega > 1.0 && (r + prevR) < stepLen;
    if (fail) {
      stepLen -= omega * stepLen;
      omega = 1.0;
    } else {
      stepLen = signedR * omega;
      let err = r / max(t, 1e-4);
      if (err < candErr) { candT = t; candErr = err; candMat = m.y; }
      if (err < pixelAngle) { break; }
    }
    prevR = r;
    if (t > tmax) { break; }
    t += stepLen;
  }
  return Hit(candT, candMat, k, t <= tmax && candErr < pixelAngle * 4.0);
}

/**
 * Conservative cone march for a pixel tile. k = cone radius per unit distance
 * (≈ tile half-diagonal in pixels × pixel angle). Returns a distance that every
 * ray inside the cone can safely start from.
 */
fn coneMarch(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32 {
  var t = tmin;
  for (var s = 0; s < 160; s++) {
    let d = map(ro + rd * t).x;
    let r = k * t;
    if (d <= r * 1.5 || t > tmax) { break; }
    t += (d - r) / (1.0 + k);
  }
  return min(t, tmax);
}

/** Classic sphere tracing (reference / benchmark baseline). */
fn raymarchBasic(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, pixelAngle: f32) -> Hit {
  var t = tmin;
  var k = 0u;
  var mat = 0.0;
  for (; k < 256u; k++) {
    let m = map(ro + rd * t);
    mat = m.y;
    if (abs(m.x) < pixelAngle * t || t > tmax) { break; }
    t += m.x;
  }
  return Hit(t, mat, k, t <= tmax);
}

/** Surface normal from 4 map() samples (tetrahedral central differences). */
fn calcNormal(p: vec3f, eps: f32) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.5773 * eps;
  return normalize(e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x +
                   e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);
}

/** Soft shadow toward a light (Quilez's improved penumbra estimate). k = sharpness. */
fn softShadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32 {
  var res = 1.0;
  var t = tmin;
  var ph = 1e10;
  for (var i = 0; i < 64; i++) {
    let h = map(ro + rd * t).x;
    if (h < 1e-4) { return 0.0; }
    let y = h * h / (2.0 * ph);
    let d = sqrt(max(h * h - y * y, 0.0));
    res = min(res, k * d / max(1e-4, t - y));
    ph = h;
    t += clamp(h, 0.005, 0.5);
    if (t > tmax) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

/** Ambient occlusion from 5 samples along the normal. */
fn calcAO(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.01 + 0.12 * f32(i) / 4.0;
    let d = map(p + h * n).x;
    occ += (h - d) * sca;
    sca *= 0.95;
    if (occ > 0.35) { break; }
  }
  return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * n.y);
}
`,
});
return {  };
})();

// ---- src/shaders/color.js
const __m48 = (() => {
const defineModule = __m5.defineModule;


defineModule('color', {
  deps: ['math'],
  doc: 'sRGB transfer, OKLab/OKLCh, HSV, cosine palettes, scientific colormaps (turbo, viridis, inferno, magma).',
  code: /* wgsl */`
/** Exact sRGB EOTF / OETF (IEC 61966-2-1). */
fn srgbToLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn linearToSrgb(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, 12.92 * x, x <= vec3f(0.0031308));
}
fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// ---- OKLab (Björn Ottosson, 2020): perceptually uniform, great for gradients.
fn si_cbrt(x: vec3f) -> vec3f { return sign(x) * pow(abs(x), vec3f(1.0 / 3.0)); }
fn linearToOklab(c: vec3f) -> vec3f {
  let lms = vec3f(
    0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
    0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
    0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b);
  let q = si_cbrt(lms);
  return vec3f(
    0.2104542553 * q.x + 0.7936177850 * q.y - 0.0040720468 * q.z,
    1.9779984951 * q.x - 2.4285922050 * q.y + 0.4505937099 * q.z,
    0.0259040371 * q.x + 0.7827717662 * q.y - 0.8086757660 * q.z);
}
fn oklabToLinear(l: vec3f) -> vec3f {
  let q = vec3f(
    l.x + 0.3963377774 * l.y + 0.2158037573 * l.z,
    l.x - 0.1055613458 * l.y - 0.0638541728 * l.z,
    l.x - 0.0894841775 * l.y - 1.2914855480 * l.z);
  let lms = q * q * q;
  return vec3f(
     4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
    -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
    -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z);
}
/** (L, C, h[radians]) <-> OKLab. */
fn oklabToOklch(l: vec3f) -> vec3f { return vec3f(l.x, length(l.yz), atan2(l.z, l.y)); }
fn oklchToOklab(l: vec3f) -> vec3f { return vec3f(l.x, l.y * cos(l.z), l.y * sin(l.z)); }
/** Perceptual interpolation of two linear-RGB colours. */
fn mixOklab(a: vec3f, b: vec3f, t: f32) -> vec3f {
  return oklabToLinear(mix(linearToOklab(a), linearToOklab(b), t));
}

/** HSV (all in [0,1]) to linear-ish RGB, smooth variant. */
fn hsv2rgb(c: vec3f) -> vec3f {
  let rgb = clamp(abs((c.x * 6.0 + vec3f(0.0, 4.0, 2.0)) % 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
  let s = rgb * rgb * (3.0 - 2.0 * rgb);
  return c.z * mix(vec3f(1.0), s, c.y);
}
fn rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = select(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), c.g >= c.b);
  let q = select(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), c.r >= p.x);
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

/** Cosine palette a + b·cos(2π(c·t + d)) (Quilez). */
fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}

// ---- scientific colormaps (inputs clamped to [0,1], outputs sRGB-encoded)
fn turbo(x0: f32) -> vec3f {
  let x = clamp(x0, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return vec3f(
    dot(v4, vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396, 59.28637943)),
    dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2f(4.27729857, 2.82956604)),
    dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f(-89.90310912, 27.34824973)));
}
fn si_poly6(t: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f, c6: vec3f) -> vec3f {
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
fn viridis(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061),
    vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685),
    vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659),
    vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987),
    vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105),
    vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234),
    vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832));
}
fn inferno(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184),
    vec3f(0.1065134194856116, 0.5639564367884091, 3.932712388889277),
    vec3f(11.60249308247187, -3.972853965665698, -15.9423941062914),
    vec3f(-41.70399613139459, 17.43639888205313, 44.35414519872813),
    vec3f(77.162935699427, -33.40235894210092, -81.80730925738993),
    vec3f(-71.31942824499214, 32.62606426397723, 73.20951985803202),
    vec3f(25.13112622477341, -12.24266895238567, -23.07032500287172));
}
fn magma(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933),
    vec3f(0.2516605407371642, 0.6775232436837668, 2.494026599312351),
    vec3f(8.353717279216625, -3.577719514958484, 0.3144679030132573),
    vec3f(-27.66873308576866, 14.26473078096533, -13.64921318813922),
    vec3f(52.17613981234068, -27.94360607168351, 12.94416944238394),
    vec3f(-50.76852536473588, 29.04658282127291, 4.23415299384598),
    vec3f(18.65570506591883, -11.48977351997711, -5.601961508734096));
}
`,
});

defineModule('tonemap', {
  deps: ['color'],
  doc: 'HDR -> display: ACES (Narkowicz and Hill fits), AgX, Khronos PBR Neutral, Reinhard; blue-noise-free dithering.',
  code: /* wgsl */`
fn tonemapACES(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
/** ACES RRT+ODT fit (Stephen Hill), includes the sRGB->AP1 matrices. */
fn tonemapACESHill(c: vec3f) -> vec3f {
  let inM = mat3x3f(0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777);
  let outM = mat3x3f(1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276, 1.07602);
  let v = c * inM;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp((a / b) * outM, vec3f(0.0), vec3f(1.0));
}
/** AgX (Sobotka) with the minimal polynomial fit; returns display-linear RGB. */
fn tonemapAgX(c: vec3f) -> vec3f {
  let agxM = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                     0.0784335999999992, 0.878468636469772, 0.0784336,
                     0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let agxInv = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393; let maxEv = 4.026069;
  var v = agxM * max(c, vec3f(1e-10));
  v = clamp(log2(v), vec3f(minEv), vec3f(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  let x2 = v * v; let x4 = x2 * x2;
  v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
  v = agxInv * v;
  return pow(max(v, vec3f(0.0)), vec3f(2.2));
}
/** Khronos PBR Neutral (2024): hue-preserving, keeps base colours faithful. */
fn tonemapNeutral(c0: vec3f) -> vec3f {
  let start = 0.8 - 0.04;
  let desat = 0.15;
  var c = c0;
  let x = min(c.r, min(c.g, c.b));
  let off = select(0.04, x - 6.25 * x * x, x < 0.08);
  c -= off;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < start) { return c; }
  let d = 1.0 - start;
  let newPeak = 1.0 - d * d / (peak + d - start);
  c *= newPeak / peak;
  let g = 1.0 - 1.0 / (desat * (peak - newPeak) + 1.0);
  return mix(c, vec3f(newPeak), g);
}
/** Extended Reinhard on luminance with white point w. */
fn tonemapReinhard(c: vec3f, w: f32) -> vec3f {
  let l = luminance(c);
  let n = l * (1.0 + l / (w * w)) / (1.0 + l);
  return c * (n / max(l, 1e-6));
}
/** Final display encode with ±0.5 LSB triangular dither to kill 8-bit banding. */
fn encodeDisplay(linear: vec3f, px: vec2f) -> vec3f {
  let s = linearToSrgb(linear);
  let n = fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715)))) + fract(52.9829189 * fract(dot(px + 17.0, vec2f(0.06711056, 0.00583715)))) - 1.0;
  return s + n / 255.0;
}
`,
});
return {  };
})();

// ---- src/shaders/pbr.js
const __m49 = (() => {
const defineModule = __m5.defineModule;


// Physically based shading: GGX microfacet specular with the height-correlated
// Smith visibility term (Heitz 2014), Schlick Fresnel, Lambert diffuse, and an
// analytic sky for image-free ambient + reflections.
defineModule('pbr', {
  deps: ['math'],
  doc: 'Cook-Torrance GGX BRDF, Schlick Fresnel, analytic sky, environment BRDF approximation.',
  code: /* wgsl */`
fn D_GGX(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let f = (NoH * a2 - NoH) * NoH + 1.0;
  return a2 / (PI * f * f + 1e-7);
}
fn V_SmithGGX(NoV: f32, NoL: f32, a: f32) -> f32 {
  let a2 = a * a;
  let gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / (gv + gl + 1e-7);
}
fn F_Schlick(VoH: f32, f0: vec3f) -> vec3f {
  let f = pow(1.0 - VoH, 5.0);
  return f + f0 * (1.0 - f);
}
/** Direct light: returns outgoing radiance for one light of given radiance. */
fn brdfDirect(n: vec3f, v: vec3f, l: vec3f, albedo: vec3f, metallic: f32, roughness: f32, radiance: vec3f) -> vec3f {
  let h = normalize(v + l);
  let NoV = max(dot(n, v), 1e-4);
  let NoL = clamp(dot(n, l), 0.0, 1.0);
  let NoH = clamp(dot(n, h), 0.0, 1.0);
  let VoH = clamp(dot(v, h), 0.0, 1.0);
  let a = max(roughness * roughness, 2e-3);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let F = F_Schlick(VoH, f0);
  let spec = D_GGX(NoH, a) * V_SmithGGX(NoV, NoL, a) * F;
  let kd = (1.0 - F) * (1.0 - metallic);
  return (kd * albedo * INV_PI + spec) * radiance * NoL;
}
/** Split-sum environment BRDF approximation (Karis, mobile). */
fn envBRDF(f0: vec3f, NoV: f32, roughness: f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}
/** Analytic sky radiance (gradient + horizon glow + sun disc). */
fn skyColor(rd: vec3f, sun: vec3f) -> vec3f {
  let y = rd.y;
  var c = mix(vec3f(0.55, 0.62, 0.72), vec3f(0.20, 0.38, 0.75), clamp(y, 0.0, 1.0));
  c = mix(c, vec3f(0.18, 0.17, 0.16), clamp(-y * 4.0, 0.0, 1.0));
  c += vec3f(1.0, 0.75, 0.45) * 0.35 * pow(1.0 - abs(y), 8.0);
  let s = max(dot(rd, sun), 0.0);
  c += vec3f(1.0, 0.85, 0.6) * (0.25 * pow(s, 8.0) + 30.0 * pow(s, 1500.0));
  return c;
}
/** One-call shading: sun (with shadow factor) + sky ambient + reflection. */
fn shadePBR(n: vec3f, v: vec3f, albedo: vec3f, metallic: f32, roughness: f32, sun: vec3f, sunRadiance: vec3f, shadow: f32, ao: f32) -> vec3f {
  var c = brdfDirect(n, v, sun, albedo, metallic, roughness, sunRadiance) * shadow;
  let NoV = max(dot(n, v), 1e-4);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let spec = envBRDF(f0, NoV, roughness);
  let r = reflect(-v, n);
  let skyAmb = skyColor(normalize(n + vec3f(0.0, 1.0, 0.0)), sun) * 0.6;
  let refl = mix(skyColor(r, sun), skyAmb, roughness);
  c += ao * ((1.0 - spec) * (1.0 - metallic) * albedo * skyAmb + spec * refl);
  return c;
}
`,
});
return {  };
})();

// ---- src/shaders/index.js
const __m44 = (() => {
// Registers every shader-library module. Import for side effects.
return {  };
})();

// ---- src/index.js
const __m0 = (() => {
const init = __m1.init;
const Context = __m1.Context;
const SIError = __m1.SIError;
const GArray = __m2.GArray;
const Kernel = __m4.Kernel;
const defineModule = __m5.defineModule;
const link = __m5.link;
const moduleNames = __m5.moduleNames;
const moduleInfo = __m5.moduleInfo;
const literal = __m5.literal;
const layoutOf = __m3.layoutOf;
const structLayout = __m3.structLayout;
const toHalf = __m3.toHalf;
const fromHalf = __m3.fromHalf;
const Scan = __m7.Scan;
const Reduce = __m8.Reduce;
const SpatialGrid = __m9.SpatialGrid;
const View = __m10.View;
const OrbitCamera = __m10.OrbitCamera;
const Hud = __m10.Hud;
const RenderProgram = __m11.RenderProgram;
const Fullscreen = __m14.Fullscreen;
const Spheres = __m15.Spheres;
const m4 = __m13.m4;
const v3 = __m13.v3;
const Stage = __m16.Stage;
const Lines = __m16.Lines;
const panel = __m17.panel;
const MPM = __m18.MPM;
const MATERIALS = __m18.MATERIALS;
const NBody = __m25.NBody;
const timeIt = __m25.timeIt;
const Points = __m26.Points;
const Bloom = __m12.Bloom;
const SdfScene = __m27.SdfScene;
const Glyphs = __m28.Glyphs;
const Boids = __m29.Boids;
const Cloth = __m30.Cloth;
const SurfaceMesh = __m31.SurfaceMesh;
const Smoke = __m32.Smoke;
const Volume = __m33.Volume;
const FluidSurface = __m19.FluidSurface;
const RigidWorld = __m34.RigidWorld;
const RayQuery = __m34.RayQuery;
const eulerQuat = __m34.eulerQuat;
const RIGID_SHAPES = __m34.RIGID_SHAPES;
const RIGID_FLAGS = __m34.RIGID_FLAGS;
const JOINT_TYPES = __m34.JOINT_TYPES;
const JOINT_FLAGS = __m34.JOINT_FLAGS;
const Bodies = __m35.Bodies;
const FluidRigidCoupling = __m20.FluidRigidCoupling;
const gemm = __m36.gemm;
const autotuneGemm = __m36.autotuneGemm;
const MLP = __m36.MLP;
const Adam = __m36.Adam;
const Population = __m36.Population;
const three = __m39;
const nn = __m36;
const net = __m40;
const verify = __m43;
// ShaderIntel — WebGPU physics + shader library.
//
//   import * as SI from './src/index.js';
//   const gpu = await SI.init();
//
// Everything is plain ES modules with zero dependencies.









































// Shader library modules (registered on import).


// Convenience factories on the context: gpu.view(), gpu.fullscreen(), gpu.mpm() ...


















Object.assign(Context.prototype, {
  rigid(opts) { return new RigidWorld(this, opts); },
  bodies(world, opts) { return new Bodies(this, world, opts); },
  smoke(opts) { return new Smoke(this, opts); },
  volume(opts) { return new Volume(this, opts); },
  cloth(opts) { return new Cloth(this, opts); },
  mesh(opts) { return new SurfaceMesh(this, opts); },
  glyphs(opts) { return new Glyphs(this, opts); },
  boids(opts) { return new Boids(this, opts); },
  sdf(opts) { return new SdfScene(this, opts); },
  nbody(opts) { return new NBody(this, opts); },
  points(opts) { return new Points(this, opts); },
  stage(opts) { return new Stage(this, opts); },
  view(canvas, opts) { return new View(this, canvas, opts); },
  fullscreen(desc) { return new Fullscreen(this, desc); },
  program(desc) { return new RenderProgram(this, desc); },
  spheres(opts) { return new Spheres(this, opts); },
  mpm(opts) { return new MPM(this, opts); },
});
return { init, Context, SIError, GArray, Kernel, defineModule, link, moduleNames, moduleInfo, literal, layoutOf, structLayout, toHalf, fromHalf, Scan, Reduce, SpatialGrid, View, OrbitCamera, Hud, RenderProgram, Fullscreen, Spheres, m4, v3, Stage, Lines, panel, MPM, MATERIALS, NBody, timeIt, Points, Bloom, SdfScene, Glyphs, Boids, Cloth, SurfaceMesh, Smoke, Volume, FluidSurface, RigidWorld, RayQuery, eulerQuat, RIGID_SHAPES, RIGID_FLAGS, JOINT_TYPES, JOINT_FLAGS, Bodies, FluidRigidCoupling, gemm, autotuneGemm, MLP, Adam, Population, three, nn, net, verify };
})();

global.SI = __m0;
})(typeof globalThis !== 'undefined' ? globalThis : window);
