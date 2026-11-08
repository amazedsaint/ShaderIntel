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
import { gemm, ACTIVATIONS } from './gemm.js';
import { defineModule } from '../core/wgsl.js';
import '../shaders/hash.js';

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
export class Dense {
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
export class FourierFeatures {
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
export class MLP {
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
export class MSELoss {
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
export class Adam {
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
export class SGD {
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
export class Population {
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
