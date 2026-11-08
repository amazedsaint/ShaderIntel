// Uniform-grid neighbour search by counting sort — O(N), three dispatches + a scan:
//   count:   cell[i] = cellOf(p_i); rank[i] = atomicAdd(&count[cell], 1)
//   scan:    start = exclusiveScan(count)      (numCells + 1 entries)
//   scatter: order[start[cell[i]] + rank[i]] = i
// Cells are linearised x-fastest, so the three cells of a row (x-1..x+1) are
// contiguous in sorted order: a 27-cell neighbourhood is visited as 9 ranges.
import { defineModule } from '../core/wgsl.js';
import { Scan } from './scan.js';
import { destroyAll } from '../core/dispose.js';
import { SIError } from '../core/errors.js';

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

export class SpatialGrid {
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

  destroy() { destroyAll(this.cell, this.rank, this.order, this.start, this.scan); }

  /** Uniform declarations a kernel needs to evaluate grid cells. */
  uniforms() { return { gridOrigin: 'vec3f', gridInv: 'f32', gridDims: 'vec3u', gridCells: 'u32' }; }

  /** Matching uniform values. */
  values() {
    return { gridOrigin: this.origin, gridInv: 1 / this.cellSize, gridDims: this.dims, gridCells: this.numCells };
  }

  /** Sort `count` particles (pos: vec4f array) into cells. Fills order/start. */
  build(pos, count) {
    if (count > this.maxCount) throw new SIError(`SpatialGrid.build: ${count} points exceed maxCount = ${this.maxCount}`, 'invalid-argument');
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
