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
// gpu.flush() matters: ShaderIntel batches work until a flush, and three.js renders on its own
// queue submissions, so without it three.js draws stale poses.
//
// This module never imports three.js: pass in the namespaces you use (no version lock-in).
// It reaches the GPU buffer behind a three.js storage attribute through the WebGPU backend
// (backend.get(attribute).buffer), which three.js has exposed since r160.

import { SIError } from '../core/errors.js';

/** A WebGPURenderer that uses ShaderIntel's device (features and limits already negotiated). */
export async function renderer(THREE, gpu, opts = {}) {
  const r = new THREE.WebGPURenderer({ device: gpu.device, ...opts });
  await r.init();
  return r;
}

/** The GPUBuffer three.js allocated for a storage attribute (created on demand). */
export function bufferOf(renderer, attribute) {
  const backend = renderer.backend;
  if (!backend?.get || !backend.createStorageAttribute) throw new SIError('three interop: needs THREE.WebGPURenderer with the WebGPU backend', 'invalid-argument');
  let data = backend.get(attribute);
  if (!data.buffer) { backend.createStorageAttribute(attribute); data = backend.get(attribute); }
  if (!data.buffer) throw new SIError('three interop: could not reach the storage buffer of an attribute (three.js version not supported?)');
  return data.buffer;
}

const SHAPES = { sphere: 0, box: 1, capsule: 2 };

/**
 * Instanced meshes for every body of a RigidWorld (boxes, spheres, capsules), lit by three.js.
 * o: { THREE, TSL, renderer, world, material(shape) → NodeMaterial (optional), castShadow, receiveShadow,
 *      colors: (k) → TSL color node (optional), groups (collision-group bits to draw) }
 * Returns { object: THREE.Group, update(), meshes }.
 */
export function bodies(o) {
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
  const seen = new Set();
  for (const [name, shape] of Object.entries(SHAPES)) {
    // each shape needs its own positionNode: clone a material the callback hands out more than once
    let material = o.material?.(name) || new THREE.MeshStandardNodeMaterial({ roughness: 0.55, metalness: 0.0 });
    if (seen.has(material)) material = material.clone();
    seen.add(material);
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
    /** Copy this frame's poses into three.js (GPU → GPU). Call after world.step(), then gpu.flush(), before rendering. */
    update() {
      const n = world.count;
      for (const m of Object.values(meshes)) m.count = n;
      if (n) pack.run(n, { groups: (o.groups ?? 0xffffffff) >>> 0 }, { B: world.B, out: buf });
    },
    /** Remove the meshes from their parent and free their geometries and (default) materials. */
    dispose() {
      group.removeFromParent?.();
      for (const g of Object.values(geometries)) g.dispose();
      if (!o.material) for (const m of Object.values(meshes)) m.material.dispose();
    },
  };
}

/**
 * Instanced particles from any ShaderIntel array of positions (MPM particles, boids, points).
 * o: { THREE, TSL, renderer, source: GArray, stride (vec4s per element), count: () => n, max,
 *      scale (multiply), offset (add), radius, color: TSL node or [r,g,b], material }
 */
export function particles(o) {
  const { THREE, TSL, renderer: r } = o;
  const src0 = typeof o.source === 'function' ? o.source() : o.source;
  const gpu = o.gpu || src0.ctx;
  const max = o.max ?? Math.floor(src0.length / (o.stride ?? 1));
  const countOf = () => (o.count === undefined ? max : typeof o.count === 'function' ? o.count() : o.count);
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
      const n = Math.max(0, Math.min(max, countOf() | 0));
      mesh.count = n;
      if (n) pack.run(n, { stride: o.stride ?? 1, scale: o.scale ?? 1, offset: o.offset ?? [0, 0, 0] }, { src: typeof o.source === 'function' ? o.source() : o.source, out: buf });
    },
    /** Remove the mesh from its parent and free its geometry and (default) material. */
    dispose() {
      mesh.removeFromParent?.();
      mesh.geometry.dispose();
      if (!o.material) material.dispose();
    },
  };
}
