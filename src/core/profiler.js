// GPU profiler built on timestamp queries. A profiled region is exactly one
// compute pass (timestamps can only be written at pass boundaries), so library
// code clears buffers with compute kernels instead of clearBuffer inside
// regions. Results are smoothed with an exponential moving average because
// browsers quantize timestamps (Chrome: 100 µs) unless developer flags are on.

export class Profiler {
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
