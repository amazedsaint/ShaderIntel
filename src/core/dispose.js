// destroyAll(a, b, [c, d], obj): destroy GPU resources and owned helper objects in one call.
// Accepts GArrays, texture handles, GPUBuffers / GPUTextures, anything with destroy(), arrays of
// those and null / undefined (skipped). Every destroy() in ShaderIntel is idempotent.
export function destroyAll(...list) {
  for (const r of list.flat(4)) {
    if (r && typeof r.destroy === 'function') {
      try { r.destroy(); } catch (e) { console.warn('ShaderIntel: destroy failed', e); }
    }
  }
}
