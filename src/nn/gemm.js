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

export const ACTIVATIONS = {
  none: { f: 'x', df: '1.0' },
  relu: { f: 'max(x, 0.0)', df: 'select(0.0, 1.0, x > 0.0)' },
  leaky: { f: 'select(0.01 * x, x, x > 0.0)', df: 'select(0.01, 1.0, x > 0.0)' },
  tanh: { f: 'tanh(x)', df: '1.0 - tanh(x) * tanh(x)' },
  sigmoid: { f: '1.0 / (1.0 + exp(-x))', df: '(1.0 / (1.0 + exp(-x))) * (1.0 - 1.0 / (1.0 + exp(-x)))' },
  sin: { f: 'sin(x)', df: 'cos(x)' },
  gelu: { f: '0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))',
    df: '0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + 0.5 * x * (1.0 - tanh(0.7978845608 * (x + 0.044715 * x * x * x)) * tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.7978845608 * (1.0 + 0.134145 * x * x)' },
};

export const GEMM_CONFIGS = [
  { BM: 64, BN: 64, BK: 16, TM: 4, TN: 4 },     // 3.16 TFLOP/s at 2048³ on an M2 Max (default)
  { BM: 64, BN: 64, BK: 16, TM: 8, TN: 4 },
  { BM: 128, BN: 64, BK: 16, TM: 8, TN: 4 },
  { BM: 64, BN: 128, BK: 16, TM: 4, TN: 8 },
  { BM: 64, BN: 32, BK: 16, TM: 4, TN: 4 },
  { BM: 128, BN: 128, BK: 8, TM: 8, TN: 8 },
];

const cache = new WeakMap();   // per Context: a destroyed context's kernels go with it
const R = (n) => Array.from({ length: n }, (_, k) => k);

/** The kernel for one (config, transA, transB, activation, bias, beta) combination. */
export function gemmKernel(ctx, cfg, { transA = false, transB = false, act = 'none', bias = false, accumulate = false, split = false } = {}) {
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
export function gemm(ctx, A, B, C, o) {
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
export async function autotuneGemm(ctx, { n = 2048, reps = 5, cache: useCache = true } = {}) {
  // versioned: a cached tile shape is only reused if it is still one of the configurations on offer
  const id = `si-gemm-v2-${ctx.caps.vendor}-${ctx.caps.architecture}`;
  const known = (c) => GEMM_CONFIGS.find((g) => c && ['BM', 'BN', 'BK', 'TM', 'TN'].every((k) => g[k] === c[k]) && !!g.prefetch === !!c.prefetch);
  if (useCache && typeof localStorage !== 'undefined') {
    try {
      const hit = known(JSON.parse(localStorage.getItem(id)));
      if (hit) { ctx._gemmCfg = hit; return { best: hit, cached: true }; }
    } catch { /* a corrupt cache entry is re-measured */ }
  }
  const A = ctx.array('f32', n * n), B = ctx.array('f32', n * n), C = ctx.array('f32', n * n);
  const results = [];
  try {
    for (const cfg of GEMM_CONFIGS) {
      const run = () => gemm(ctx, A, B, C, { M: n, N: n, K: n, cfg });
      run(); await ctx.idle();
      const t0 = performance.now();
      for (let r = 0; r < reps; r++) run();
      await ctx.idle();
      const ms = (performance.now() - t0) / reps;
      results.push({ cfg, ms, tflops: (2 * n ** 3) / (ms * 1e9) });
    }
  } finally {
    A.destroy(); B.destroy(); C.destroy();
  }
  results.sort((a, b) => a.ms - b.ms);
  ctx._gemmCfg = results[0].cfg;
  if (useCache && typeof localStorage !== 'undefined') localStorage.setItem(id, JSON.stringify(results[0].cfg));
  return { best: results[0].cfg, results };
}
