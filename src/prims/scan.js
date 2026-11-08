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
import { defineModule } from '../core/wgsl.js';

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

export class Scan {
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
