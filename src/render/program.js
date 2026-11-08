// RenderProgram: the render-pipeline twin of Kernel. Write @vertex/@fragment
// functions; bindings, the camera block and the uniform struct are generated.
//   binding 0  cam: Camera   (from the View)
//   binding 1  u:   Params   (uniform ring, dynamic offset)
//   binding 2+ buffers (read-only storage), textures, samplers in declaration order
import { structLayout, writeField } from '../core/types.js';
import { link, constants } from '../core/wgsl.js';

export const CAMERA_WGSL = /* wgsl */`
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
export const CAMERA_SIZE = 6 * 64 + 16 + 16 + 16;

let progId = 0;

export class RenderProgram {
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
