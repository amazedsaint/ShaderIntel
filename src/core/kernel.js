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
import { structLayout, writeField, layoutOf, roundUp } from './types.js';
import { link, constants } from './wgsl.js';
import { SIError } from './errors.js';

const USES_SUBGROUPS = /\bsgi\b|\bsgs\b|\bsgid\b|\bnsg\b|\bsubgroup[A-Z]\w*\s*\(|\bquad[A-Z]\w*\s*\(/;
const USES_SGID = /\bsgid\b|\bnsg\b/;

let kernelId = 0;

export class Kernel {
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
      throw new SIError(`ShaderIntel: kernel "${this.name}" uses subgroup operations but this device lacks the "subgroups" feature`);
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
    // subgroups: false makes capability-dependent modules (e.g. wgscan) emit their portable code
    this.code = link(src, desc.imports || [], desc.subgroups === false ? { ...caps, subgroups: false } : caps);

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
      if (!f) throw new SIError(`ShaderIntel: kernel "${this.name}" has no uniform "${k}"`);
      writeField(this.uview, f, v);
    }
    return this;
  }

  /** Persistently bind resources by name: k.bind({ pos: arrB }). */
  bind(resources) {
    if (!resources) return this;
    for (const [k, v] of Object.entries(resources)) {
      if (!this.slotByName.has(k)) throw new SIError(`ShaderIntel: kernel "${this.name}" has no binding "${k}"`);
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
      if (!r) throw new SIError(`ShaderIntel: kernel "${this.name}": binding "${s.name}" was never provided`);
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
    throw new SIError(`ShaderIntel: kernel "${kernel.name}", binding "${slot.name}": a buffer of ${bytes} bytes is not a whole ` +
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
  if (access === 'read' && atomic) throw new SIError(`ShaderIntel: atomic binding "${name}" must be read_write`);
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
