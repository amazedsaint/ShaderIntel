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

export class Reduce {
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
