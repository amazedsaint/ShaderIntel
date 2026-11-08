# Learnings

Everything non-obvious that was measured, broke, or surprised us while building ShaderIntel.
Hardware for all numbers: Apple M2 Max (30-core GPU, Metal 3), Chrome 152 / Dawn, subgroup size 32.

## Environment and tooling

- **Headless Chrome could not run inside the macOS sandbox** used for development (`ProcessSingleton`
  cannot create its socket directory). Testing ran in the app's embedded Chromium instead, which
  exposes WebGPU with subgroups, `shader-f16` and timestamp queries.
- **`requestAnimationFrame` is throttled to ~1.5 Hz in an unfocused embedded browser.** FPS counters
  are meaningless there; every performance number in this repo comes from GPU work timed between
  `queue.onSubmittedWorkDone()` fences, never from rAF.
- **Timestamp queries are quantised to 100 µs** in Chrome without developer flags. Use them as
  smoothed per-region estimates (the profiler averages), and time benchmarks with batches of ≥ 20 ms.
- **GPU clock state shifts results by 10–30% between runs.** The benchmark warms the GPU for 400 ms and
  interleaves baseline/ours measurements, keeping the best batch of each.
- The npm registry and the DGX machines were not reachable from the sandbox, so the library has zero
  dependencies and was validated on the Mac's GPU only.
- **No TypeScript compiler was available** (npm blocked). `types/shaderintel.d.ts` was syntax-checked
  with Node 24's built-in TypeScript parser (type stripping), wrapped in a `declare module` block, and
  a planted error confirmed the check bites. Node's parser is more lenient than `tsc`: it accepted
  `export { a, b }` inside a `namespace`, which `tsc` rejects (TS1194) — re-exports in a namespace go
  through aliases (`export const gemm: typeof gemm_`). Later, from a terminal, `tsc` (strict) checked
  the declarations against `tests/types/usage.ts` through the package's `exports`: clean.
- **WebRTC between two tabs could not be tested in the sandbox**: Chrome advertises host candidates as
  mDNS `.local` names, which nothing there resolves. The netcode is tested over `LoopbackNetwork`
  (latency, jitter, loss, duplication in one page) instead. Later, headless on the DGX (Linux), two
  pages of `examples/p2p.html` connected over real data channels — which first exposed two signalling
  races (see Peer-to-peer netcode).
- **The three.js example was first blocked** (it imports three.js from a CDN, and the session's
  automatic permission mode refused that page). Run later in headless Chrome, it worked first time —
  `WebGPURenderer({ device })`, `StorageInstancedBufferAttribute`, `backend.get(attr).buffer` as read
  from the r180 sources — except for shadow acne: a directional light needs `shadow.bias` /
  `normalBias`.
- **Profiler regions are summed per submit.** Ten steps recorded before one flush report ten steps'
  time under each label. Flush every step (or divide) before reading per-step numbers.
- **Headless Chrome, driven over the DevTools protocol, is the test runner** (`tools/headless.mjs`,
  no dependencies: Node 22+ has a WebSocket). Headless pages are not throttled the way a backgrounded
  embedded preview is: the suite takes 29–42 s instead of minutes, and the two-peer test 4.5 s
  instead of 70–105 s. Three runner bugs worth knowing: clicking a button that exists in the HTML
  before the page's module has attached its handler does nothing (the benchmark page now has `?run`);
  pages opened as tabs of one window are background tabs whose animation frames stop (one window per
  page); `grep` into a pipe buffers its output (`--line-buffered`).
- **WebGPU on a DGX Spark (GB10, Vulkan, driver 580) works on the host but not in Docker.** In the
  NGC PyTorch image the NVIDIA Vulkan driver loads but cannot create an instance ("Could not get
  vkCreateInstance via vk_icdGetInstanceProcAddr"), even privileged, with /dev/dri, and without the
  image's CUDA forward-compatibility libraries. In plain Ubuntu 24.04 it needs the X client libraries
  it links against (libX11, libXext) and then Vulkan works (`vulkaninfo` lists the GB10) and Chrome's
  GPU process starts on ANGLE/Vulkan — yet `requestAdapter()` still returns null ("A valid external
  Instance reference no longer exists"), as root or not, with seccomp unconfined, with Chrome 155 or
  157. So the GB10 numbers here come from Chrome on the host; `tools/docker/` keeps the recipe.
- **nsys cannot see inside Chrome there**: Vulkan API tracing does not attach to Chrome's GPU
  process (a separate process), `--in-process-gpu` under nsys hung — and the hung Chrome kept the GPU
  busy, so later runs stalled until it was killed — and GPU metrics need admin rights
  (ERR_NVGPUCTRPERM). The library's own timestamp profiler, per kernel, did the job.
- **The DGX is shared** (CI runners, other users; load average ≈ 20 on 20 cores): its timings are
  upper bounds.
- **TypeScript's native compiler (tsgo) ships WebGPU types in lib.dom**; adding `@webgpu/types` on
  top produces hundreds of duplicate-identifier errors. The declarations resolve WebGPU types from the
  globals when present, so they compile with either.
- **macOS does not let a terminal-launched Chrome reach LAN addresses** without the Local Network
  permission, so two WebRTC peers in one headless browser on the Mac never connect, mDNS or not. On
  Linux (the DGX) the same test connects: 5/5 state hashes equal over real data channels.

## WebGPU / WGSL pitfalls

- **One `writeBuffer` per uniform buffer per submit is a bug magnet.** `queue.writeBuffer` runs before
  the command buffer, so ten dispatches recorded with ten different uniform values all see the last
  one. Fix: a uniform ring with one 256-byte slot per dispatch, bound with dynamic offsets, uploaded once
  per submit (`core/context.js`).
- **Binding one buffer twice as writable storage in a dispatch is a validation error** in Dawn
  ("Writable storage buffer binding aliasing"). In-place algorithms need a single binding (we generate a
  separate in-place pipeline).
- **Builtin names are a namespace.** A local `let nsg = …` in a kernel clashed with the injected
  `@builtin(num_subgroups) nsg`. Reserved names are documented in `core/kernel.js`.
- **WGSL forbids mixing bitwise and arithmetic operators without parentheses:** `a * b ^ c` is a parse
  error; write `(a * b) ^ c`.
- **Loops containing barriers need provably uniform control flow.** Values read from atomics are
  non-uniform to the analysis, so a look-back loop must route its decisions through
  `workgroupUniformLoad`. Derivatives (`fwidth`) must be taken before any non-uniform branch.
- **Workgroup memory is zero-initialised by the implementation** on every dispatch; a 27 KB tile costs
  ~27 stores per thread before your code runs. Size tiles to what is used.
- **Textures need `COPY_SRC` to be read back** — the first smoke "all zeros" readback was a failed copy,
  not a broken solver.
- **Relaxed atomics carry no ordering for ordinary stores.** A single-pass scan done *in place* read a
  predecessor's tile after its outputs were visible but before its "ready" flag was: wrong prefixes at
  tile boundaries. Out-of-place is safe (fallbacks only read the untouched source); in-place uses
  reduce-then-scan.
- `subgroupBroadcast` needs a constant lane id; `subgroupShuffle` takes a dynamic one. Subgroup
  operations must sit in uniform control flow (default `subgroup_uniformity` error), so code that
  "only subgroup 0 needs" is often run by every subgroup redundantly instead of branching.
- **WGSL reserves many ordinary words** for future use: `active`, `filter`, `shared`, `target`,
  `type`, `set`, `module`, `new`, `self`, … A local `var active` broke the rigid-body kernel at
  compile time. The kernel builder prints the offending line, so this costs seconds — once you know.
- **At most 8–10 storage buffers per stage is the portable budget.** The first rigid-body narrowphase
  needed 13; merging the hash table's keys and values into one buffer, and the contact counters with
  the per-body counts, brought it to 9.
- **Dynamically indexed private arrays live in memory, not registers.** A 6×6 Hessian as
  `array<f32, 36>` indexed in loops spilled on Apple GPUs; the same arithmetic on `mat3x3f` blocks
  (Schur complement, unrolled 3×3 Cholesky) runs entirely in registers.
- **Empty indirect dispatches are not free**: ~16 µs each in a pass on Apple/Chrome (one barrier). A
  solver that dispatches 16 colour slots × 8 iterations pays ~2 ms for nothing when 6 colours are
  used; the rigid solver now dispatches only the colours recently needed (+2), learnt from an
  asynchronous stats read. A deterministic world must not do that — the number of dispatches would
  depend on when the read-back landed — so it always dispatches the full budget (+0.4 ms on the pile).
- **A constant expression must not produce NaN or ∞.** `bitcast<f32>(0xffffffffu)` as a `const` or
  literal argument is a shader-creation error in WGSL; use a runtime value or another sentinel.
- **Writing different components of one `var<workgroup>` vec4 from different threads races**: a
  component store is a read-modify-write of the whole vector. The first GEMM staged A and B that way
  and was wrong by up to 9.8; each thread now writes whole vec4s.
- **`queue.writeBuffer` overtakes unsubmitted work.** A write lands before every command still being
  recorded, so code that records a dispatch, then writes the buffer it reads, sees the new data in
  the old dispatch. `importState` and the rigid command queue flush before writing.
- **Do not destroy a buffer that recorded-but-unsubmitted commands use** (validation error at submit):
  growing the split-K scratch buffer flushes first.
- **Locals must not shadow bindings.** In the Adam kernel `let m = …; m[i] = m` referred to the local,
  not the binding `m`; WGSL allows the shadowing, so the error surfaced as a type mismatch far away.
- **A workgroup-wide `return` needs a uniform condition.** The large-body reverse-list kernel decides
  whether to run through `workgroupUniformLoad` of a flag thread 0 computed; reading the body directly
  would be non-uniform to the analysis and the later barriers would not compile.
- **The bundler handles static imports only.** A dynamic `import()` in `verify/` produced a broken
  `dist/` bundle; the import is static now.
- **Indirect dispatches can cost 20× a direct one.** Measured per dispatch: M2 Max (Metal) 3.9 µs
  direct, 15 µs indirect; GB10 (Vulkan) 15 µs direct, **304 µs** indirect. A rigid step issued ~138
  indirect dispatches (one per colour per iteration), so a 38-body world took 24.6 ms per step on the
  GB10. Every solver kernel now loops grid-stride over a count read on the GPU, and the CPU sizes
  plain dispatches from counts read back asynchronously: a stale estimate costs time, never results
  (the determinism hashes did not change by a bit). 24.6 → 2.9 ms, then 0.9 ms with the fused solve.
- **A workgroup barrier does not order a workgroupUniformLoad against the next write.** In a loop
  `if (lidx == 0) { x = ...; } let n = workgroupUniformLoad(&x);` with no other barrier in an
  iteration (an empty colour), thread 0 can overwrite x before a slower thread has loaded it: add a
  barrier at the top of the iteration.
- **Storage writes become visible to the rest of the workgroup only after `storageBarrier()`**:
  `workgroupBarrier()` orders workgroup memory only. The fused solve needs one after every pass.
- **The same WGSL function compiles differently in two shaders.** The fused and multi-dispatch
  solves run identical functions in the same order, yet differ by 1–4 ulp after one step on both
  Metal and Vulkan (inlining and FMA contraction depend on the surrounding code), and the difference
  grows chaotically. Each path is deterministic; choose between them by state only (body count).
- **Read-back latency is a queue position, not a constant.** A read waits for everything submitted
  before it; a loop that records ticks faster than the GPU runs them grows the queue without limit
  (a resync export once took 16 s on the GB10). RollbackSession now fences each tick and waits when
  more than `maxInFlight` (3) are unfinished.

## Performance findings (what won, what lost)

| Experiment | Result |
|---|---|
| Scan tile loads: per-thread blocked → lane-contiguous through shared memory | **+15%** (reduce pass reaches the pure-read roofline) |
| Single-pass decoupled-look-back scan (serial, subgroup-parallel and single-thread look-back; spin limits 16–4096) | **1.4–2.5× slower** than reduce-then-scan; larger spin limits were slower still — successors poll while predecessors starve. Kept as `mode: 'single-pass'`, not default. |
| N-body register blocking K = 2/4 | +8% over K = 1; **K = 8 collapses 9×** (register spills) |
| N-body subgroup broadcast (no shared memory) | **33% slower** than shared-memory tiles on Apple |
| N-body best | tiles of 128 × K = 4: 207 G pair interactions/s ≈ 4.1 TFLOP/s |
| Noise: normalised `unpack4x8snorm` gradients | 1.38× faster than Ashima, 1.56× than psrdnoise |
| Noise: unnormalised gradients | 18% faster but visibly patchy contrast — rejected |
| Noise: 16-entry const gradient table / Perlin edge selection / w³ falloff | all slower (dynamic const-array indexing is expensive) |
| Over-relaxed sphere tracing (ω 1.2–1.95) | 19% fewer steps at best, **no faster** (lanes wait for the slowest ray) |
| Two-pass cone marching (4×4 tiles) | **2.2–2.7× faster**, per-pixel steps 21 → 4 |
| MPM P2G shared-memory tiles + Morton block sort | **2.8× faster**, bit-identical to global atomics |
| Grid neighbour search vs all-pairs | 122× at 65k particles |
| MPM simulation at 430k particles, 8 substeps | ~8 ms/frame GPU (P2G ≈ 0.6 ms/substep) |
| Rigid primal: 6×6 `array<f32,36>` LDLᵀ → 3×3 register blocks | **2× faster** (337 bodies: 13.2 → 6.6 ms/step) |
| Rigid narrowphase: thread per body walking neighbours → pair list + thread per pair | **33× faster** (6.3 → 0.19 ms, 6,004 bodies): SIMD lanes no longer wait for the body with most neighbours |
| Rigid broadphase: grid cell from the largest body → oversized bodies in a side list | 3.2 → 0.7 ms: four 7 m static walls had made 14 m cells and an all-pairs test |
| Rigid primal: one thread per body → 8 lanes per body + shared-memory reduction | per iteration 0.6 → 0.25 ms (6,004 bodies); small colours finally fill the GPU |
| Rigid total, 6,004 mixed bodies, 8 iterations | **12.1 → 3.2 ms per step** (before determinism) |
| Determinism rewrite (sorted pair lists, prefix-sum allocation, double-buffered colouring) | 3.2 → **7.5 ms**: re-measured weeks later, not assumed |
| … the "largest 2% are large bodies" rule | made 117 ordinary boxes "large"; each large body's reverse list was one thread scanning all 6,004 bodies: **2.8 ms of serial latency**. Only size outliers (> 2× the 98th percentile) are large now, and a large body's list is built by a whole workgroup with a prefix sum: 7.5 → 4.8 ms |
| … three neighbour walks (count, write, reverse) → one walk that records both lists | 4.8 → 4.4 ms; final state bit-identical to before |
| Rigid total today, 6,004 bodies, 28k contacts | **4.25 ms** (4.6 ms deterministic). Primal sweeps 2.1 ms, narrowphase 0.95 ms (it runs twice — count, then write — so contacts can be allocated exactly with a prefix sum), neighbour walk 0.43 ms, colouring 0.38 ms |
| GEMM 2048³: naive → 16×16 shared tiles → register-blocked vec4 tiles (autotuned) | 0.43 → 1.42 → **3.0–3.16 TFLOP/s** |
| GEMM split-K for weight gradients over 65,536 rows | **10.8×** at 128 × 128 (9.3 → 0.86 ms), 32× at 64 × 64 |
| Rigid indirect → direct grid-stride dispatches | duel (38 bodies): GB10 24.6 → 2.9 ms, M2 Max 2.0 → 1.24 ms per step; 6,004-body pile on the M2 Max 4.25 → 3.58 ms |
| Fused single-workgroup solve for small worlds | GB10 duel 2.9 → 0.9 ms; crossover vs multi-dispatch ~130 bodies (M2 Max), ~230 (GB10) with 128 invocations (with 64: ~60 / ~130) — default threshold 128 |
| 6,004-body pile, GB10 (Chrome, Vulkan) | 3.05 ms per step (3.2 deterministic) — on a shared, loaded machine |
| GEMM 2048³ on the GB10 | 11.1 TFLOP/s, 4.9× the 16×16 tiled kernel, 13.3× naive (fastest tile 128×64×16, 8×4) |
| Per-pixel hash on the GB10 | the sin hash is 1.7× faster than any well-mixed integer hash (hardware sine is cheap on NVIDIA; 32-bit multiplies are not) — and fails uniformity by four orders of magnitude |

## Physics and math findings

- **Collocated grids cannot project exactly.** Central-difference divergence ∘ gradient is a wide
  Laplacian; solving the compact 7-point one left 42% of the divergence. A staggered MAC grid makes the
  operators consistent: 0.4% left with the same solver.
- **Gauss–Seidel XPBD propagates load slowly down long chains**: a 48-row hanging sheet stretched 2.2%.
  Long-range attachments (tethers to the nearest pin) brought it to 0.8% for one extra dispatch.
- **Explicit MPM fluids trade compressibility for substeps.** In lattice units hydrostatic compression
  is `g h / c²` with `c ≤ 0.4 dx/Δt`; the solver sets stiffness to the stable maximum so materials
  cannot explode, and more substeps give stiffer water (mean J 0.954 at 8 substeps).
- **A linear EOS (τ = K (J − 1)) is the stable choice** for J-based fluids: Tait's γ = 7 makes the wave
  speed grow as J⁻⁴ under compression and breaks the CFL bound.
- **Neo-Hookean jelly needs no SVD** (`τ = μ (F Fᵀ − I) + λ ln J I`); only plasticity models (snow, sand)
  pay for a 3×3 SVD.
- **APIC + B-splines conserves linear momentum exactly** (Σᵢ wᵢₚ (xᵢ − xₚ) = 0), which makes a precise
  regression test: grid mass/momentum must equal particle mass/momentum to fixed-point precision.
- **Displaced SDFs must stay 1-Lipschitz.** Adding `a · noise(k p)` raises the gradient bound to
  `1 + a k |∇noise|`; dividing by that bound removed the "holes" sphere tracing punched into the blob.
- **Integer (fixed-point) accumulation makes GPU results deterministic** regardless of atomic order — a
  rare property for GPU scatter algorithms, and the reason two very different P2G kernels can be
  verified bit-for-bit.
- **Hash quality matters at scale**: `fract(sin(x)·43758.5453)` fails a 256-bucket χ² test by four
  orders of magnitude on large inputs (χ² ≈ 6.5M vs ≈ 330 expected); PCG passes at the same speed.
- **Test hashes for independence, not only uniformity.** pcg2d's x output passes the value test
  (χ² 326) but fails on horizontally adjacent pixels (χ² 1073 over 16 × 16 pair buckets): x skips the
  final mixing step that y gets. hash21 / ihash2 now use y (χ² 226), and hash22 derives its second
  value as pcg(x ^ y) (χ² 258); a test checks neighbour pairs.
- **Rigid bodies (AVBD): what made stacks stand still.** Every fix below was found by measuring the
  drift of a box resting across two others, a 10-box tower and a 20-row pyramid over 8 s:
  - *Keep the penalty in the Hessian when a contact force clamps to zero.* Dropping inactive contacts
    from `H` let the next Newton step jump through the contact; odd iterations launched a box, even
    ones dropped it (a 2-cycle). With `k` always in `H` a tilted box lands and rests (drift 4e-9 m).
  - *Linearise contacts at the start of the step* (AVBD's Taylor expansion). Re-evaluating geometry
    every iteration made large stiffness explode; with constant Jacobians each body's local problem is
    exactly quadratic and one Newton step solves it.
  - *Scale the stiffness ramp by the pair's effective mass.* β in absolute units drove λ to −4742 N
    where 2.45 N was needed; `k += β m_eff/Δt² |C|` with β ≈ 30 matches the reference's 1e5 for
    unit masses and stays right for heavy and light bodies.
  - *Do not start every step at the inertial prediction.* Recording λ after every iteration showed
    resting bodies starting 2.7 mm (one gravity step) inside their supports: iteration 1 pushed with
    6× the weight, later iterations relaxed, and every step ended with λ ≈ 0. Warm starting then
    carried nothing and the friction bound μ|λ_n| collapsed, so boxes slid (pyramid: 73 mm in 8 s).
    VBD's adaptive initial guess (start at `x₀ + vΔt + a gΔt²`, `a` = how much the body actually fell
    last step) fixed it outright: the pyramid settles once by 1.4 mm and then holds.
  - *Start stiffness low.* The reference demo starts at k = 1 N/m (≈ 3e-4 m/Δt²); starting at m/Δt²
    made the first dual update overshoot.
  - *Friction anchors need the rest of the recipe.* Keeping the anchor points of sticking contacts
    and correcting their full drift in every iteration double-counted with warm-started λ_t: creep got
    worse (5–37 cm). Correcting only the fraction (1 − α) per step, after the fixes above, works.
  - *Post-stabilisation did not port.* The 2D reference's extra α = 0 sweep let a tall 3D pyramid
    creep steadily (85 mm in 18 s at 8 iterations); it is kept as an option, off by default.
  - *Clamp friction to a disc, not per axis.* Per-axis clamping is a pyramid cone whose strength
    depends on the slide direction (up to √2 μ). With the disc projection a box slides down a 40°
    slope at 1.797 m/s², Coulomb's `g(sin θ − μ cos θ)` to four digits, and sticks at 20°.
- **`commit()` must upload only new bodies.** The first version re-uploaded every body from its
  initial state, so firing a ball reset the scene.

## Joints, contacts and rolling (rigid bodies, part 2)

- **A record stride that disagreed with its WGSL struct by one vec4** (`JOINT_VEC4 = 14` for a 13-field
  struct) shifted every joint after the first by 16 bytes: joint 0 worked, the rest read garbage (NaN
  chains, broken welds). This is what binding contracts now catch at bind time (see Verification).
- **Pick one sign convention and test it.** Hinge and slider motors and limits measure the motion of
  body a relative to body b; the first version mixed the two, so a +2 rad/s motor turned at −2.
- **Force-aware stiffness feeds back.** Raising a joint's stiffness to `|λ|/ε` inside the iterations
  diverged (NaN in the rope bridge and ragdolls); doing it once per step regressed the chain and drove
  ragdoll joints to 106–206 kN. A **mass floor** instead — stiffness scaled as if each body weighed at
  least 1 unit, which is what AVBD's absolute β does for small bodies — took a 347:1 chain's sag from
  145 mm to 4.7 mm without touching anything else.
- **The same floor on contacts launches light bodies** (piles of 20–60 g bodies flew off at
  40–83 m/s), and a contact tolerance rule (`k ≥ |λ|/ε`) fixes a short heavy-on-light stack but makes
  big piles creep or explode. Both stay opt-in. Heavy-on-light remains the known weak spot: at 8
  iterations a 100:1 stack sinks about 1 cm, 1000:1 about 3.5 cm; 4 substeps give 4.9 / 11 mm.
- **Distance joints need geometric stiffness.** A slack rope bridge went NaN until the lumped term
  `|f|/L` (the curvature of `|pa − pb|`) was added to the positional block.
- **Spawning bodies overlapping turns into enormous forces** at the first step; ragdolls are built with
  their parts apart.
- **Project friction with the normal force you just computed.** Clamping friction to the *previous*
  iteration's `μ|λ_n|` left 248 stored contacts outside the cone; the law checker found it.
- **A rolling ball sped up** (2.6 → 3.1 m/s over 7 s on flat ground) and rode 2 cm into the floor:
  sticking contacts kept their old anchor points, but on a rolling body the remembered material point
  has rotated away and up. Anchors are kept only while that point is still within `stickThreshold` of
  the new contact point.
- **Warm starts must find a rolling contact.** The contact matched last step's nearest point within 5%
  of the body radius; a ball rolling at 3 m/s moves its contact 5 cm per step, so a 0.3 m ball started
  every step from zero force and sank 11.5 mm. The radius now grows with `|Δv|Δt`, and one-point
  manifolds always match.
- **Small-angle rotation maps are a hidden brake.** Integrating `q + ½Δt ω⊗q` and recovering
  `ω = 2Δq.xyz/Δt` loses ≈ φ²/8 of the spin per step (φ = |ω|Δt): a 0.1 m ball rolling at 2 m/s lost
  58% of its speed in 10 s. The exact exponential and logarithm maps keep it at 2.000 m/s and a free
  body at 20.000 rad/s.
- **Caps hide in data structures.** A body's reverse pair list held 64 entries — fine for ordinary
  bodies, but a 4 × 2 × 4 m block resting on a bed of 0.4 m boxes touches 117 of them, and its own solve
  ignored the rest: it sat 7.6 cm into the boxes. Large bodies now get up to 1,024 entries, built after
  the narrowphase from pairs that actually touch.
- **Unbounded grids are a trap**: tiny bodies (5 cm) in the default 80 m bounds asked for 67M cells
  (268 MB) and threw inside `step()`. The cell size now grows to keep the grid under ~4M cells.

## Fluid ↔ rigid coupling

- **A floating cube is unstable upright for densities between 0.21 and 0.79** (its metacentric height
  `w²/12d − (H − d)/2` is negative). The first buoyancy test used cubes of density 0.25 / 0.5 / 0.75 and
  passed only because the small-angle rotation maps damped rotation; with exact maps the cubes tipped
  onto an edge, as real ones do. Boards (0.4 × 0.24 m) are stable and float with 0.20 / 0.52 / 0.87
  submerged — within about one 3.1 cm grid cell of Archimedes.
- **Measure the waterline where the body floats.** Sloshing tilts the surface by about a centimetre
  across a 3 m tank, which a 24 cm board turns into ±0.05 of submerged fraction; the test reads the
  level beside each board and averages over 3 s.
- **The boundary band sets the effective size of a body.** With 1.5 cells, bodies were too buoyant
  (the fluid saw them a cell thicker); 0.5 cells matches the waterline best.

## Neural networks

- **Tiles beat raw FLOPs here too.** The naive kernel streams A and B from global memory (0.43
  TFLOP/s); 16×16 shared tiles reach 1.42; computing a 4×4 or 8×4 block of outputs per thread from vec4
  tiles reaches 3.0–3.16 TFLOP/s. WebGPU exposes no matrix units, so this is plain FMA throughput.
- **Small outputs with long K starve the GPU**: a weight gradient `Xᵀ dZ` is a few tiles of output and
  a 65,536-long reduction. Splitting K over workgroups and summing the partial tiles in a fixed order
  keeps it deterministic and runs 10.8–32× faster.
- **Check gradients against finite differences in f32 with care**: in f32 a small step drowns in
  rounding, so the checker uses central differences with h = 10⁻² on 12 random weights and allows 5%
  relative error; the worst sampled component agrees to 2 × 10⁻⁴.
- **Freivalds needs the right norm.** Comparing `A(Br)` with `Cr` relative to the row's L1 norm can
  hide a single corrupted entry; relative to the L2 norm, one entry of 10⁻⁴ of the norm is always
  caught.
- **Evolution strategies fit the GPU well**: noise regenerated from a hash costs nothing to store, so
  4,096 agents share one parameter vector, and antithetic pairs with centred ranks make the update
  robust to the scale of the reward.

## Peer-to-peer netcode

- **Determinism is a property of the whole pipeline, not of the solver.** Anything that lets thread
  timing or a read-back's arrival change *what* is computed (atomic append order, a colour budget
  learnt asynchronously) breaks lockstep, even when every single kernel is deterministic.
- **Hash confirmed states without stalling**: the snapshot ring already holds the state after tick T
  (taken before tick T + 1), so the session hashes that snapshot on the GPU and reads the hash back
  asynchronously; peers compare 64-bit hashes every 30 ticks.
- **Redundancy beats retransmission for inputs**: every packet carries the last 8 ticks, so 10%
  random loss cost no rollbacks beyond those latency causes (two peers, 40 ± 15 ms: 13/13 hash
  checkpoints equal, 77 packets lost).
- **A transcript is a cheap certificate**: folding each confirmed tick's inputs into a hash chain
  and recording state hashes lets anyone replay a match and detect a single flipped input bit.
- **JSON corrupts GPU records.** Body records keep u32 words (group, collision mask, flags) inside
  `Float32Array`s; a mask of `0xffffffff` is a NaN bit pattern, which JSON writes as `null`, and JSON
  also drops the sign of −0. The data-channel encoding sent typed arrays as number lists and would have
  corrupted every exported state; it now sends their bytes as base64. The loopback simulator had used
  `structuredClone`, which is exact, so tests could not see it — the simulator now uses the real wire
  encoding, which is the point of a simulator. The same trap exists in JavaScript itself:
  `Float32Array.from(x)` converts through numbers, `x.slice()` copies bytes.
- **Resync from a snapshot, through a private copy.** The state to send is the authority's newest
  fully confirmed tick, which lives in the rollback ring; the ring is overwritten a few ticks later,
  while the GPU read-back is still in flight, so the export first records a GPU copy of the snapshot
  and reads from that. Counting resyncs as epochs, and tagging hashes with them, stops the old history's
  in-flight hashes from triggering a second transfer.
- **Data-channel messages have a size limit** (about 256 KB in Chrome); a world state is split into
  64 KB parts and reassembled by index, so parts may also arrive out of order.
- **A signalling handshake needs retries and ordering.** The first `RTCPeer` said hello once; a host
  page still initialising its GPU missed it and both pages waited forever. Candidates that arrived
  while the remote description was being applied were dropped. The guest now repeats hello until the
  host answers, and messages are handled in order with candidates queued until the description is set.
- **Make every resync path terminate.** A peer that drops a state as too old must say so: the
  authority ignores that peer's old-epoch hashes and would never resend by itself. And a peer must
  keep enough input history to replay a late state (now 10 s), or a slow export turns into a stuck
  session. Both happened on the GB10 and never on the faster Mac.
- **Different GPUs disagree in the last bits**: the same deterministic scene hashes `21ecaba2…` on an
  M2 Max and `79eb0e99…` on a GB10 (each bit-identical run to run). Cross-device lockstep needs the
  resync, not luck.

## Verification (learnings from proof-carrying systems)

- **Write the specification as executable laws** (Bend 2's `LAWS.bend`, an "AGENTS.md backed by
  proof"): `LAWS.md` lists every promise with its checker, and the test suite runs all of them.
- **Keep the trusted base small and separate** (Lean's kernel; Bend 2's `--verdict` re-checking proofs
  with a kernel verified in Lean): checkers are a few lines of plain WGSL or JavaScript and share no
  code with the optimised kernels they judge.
- **Say what kind of guarantee each check is**: exact (scan, colouring, determinism), probabilistic
  with a bound (Freivalds: ≤ 2⁻⁸ with 8 trials) or physical with a tolerance (penetration < 2 cm).
  The penetration law first claimed less than it could keep; the honest bound is written down and the
  measured value (12.7 mm, now 5.0 mm) reported next to it.
- **Types at the GPU boundary**: binding contracts — buffer sizes checked against the WGSL struct
  stride at `bind()` — turn the stride bug above from silent garbage into an error that names the
  kernel and binding.
- **Confluence by construction**: interaction-net runtimes (HVM for Bend) get order-independence from
  their semantics; here it comes from sorted lists, prefix-sum allocation, double buffering and
  integer atomics — and is then checked, not assumed.

## Production hardening (0.3.0)

Three independent reviews of the code (physics, rendering and primitives, networking / neural
networks / types) found about fifty defects that 114 passing tests had not; most were then confirmed
on the GPU and each fix got a regression test. What they had in common:

- **A ring buffer that flushes when it is full cannot flush inside a render pass.** The uniform ring
  submitted the batch when its 1 MB filled up — fine between compute dispatches, but a frame with
  thousands of draws would finish the encoder with the render pass still open. It grows by segments
  now (each bind group names its segment), which also avoids mid-batch submits.
- **`queue.writeBuffer` lands before everything not yet submitted** — the same pitfall as above, in
  new places: per-step collider and source uploads (several steps in one batch all saw the last
  data) and re-initialisation after a recorded step (`NBody.galaxies()` was overwritten by the step).
  `writeOrdered()` records the upload as a copy in the command stream; resets flush first.
- **Destroying a buffer that recorded work still uses fails the entire submit**, silently dropping
  every other dispatch in it (an unrelated n-body step vanished when a cloth was destroyed).
  `destroy()` submits first. Stepping a destroyed object recorded half a step and poisoned the batch
  the same way; it throws now, and layers skip sources that were destroyed.
- **Device limits in the wild are lower than on development machines.** The rigid solver binds 10
  storage buffers in one kernel (about 2% of devices allow fewer, a quarter of Firefox installs) and
  its fused solve needs 18 KB of workgroup memory (about 1% have 16 KB). Both used to fail silently,
  taking whole frames with them. Check `device.limits` against what each kernel needs, and fail with
  a message (web3dsurvey.com lists the distribution of every limit).
- **Subgroups are not always 32 wide.** Two-level subgroup reductions need
  workgroup / subgroupSize ≤ subgroupSize; Intel GPUs can run 8-wide subgroups, where both the
  reduction and the scan were wrong. Size from `subgroupMinSize`, or take the portable path.
- **Degenerate inputs are where NaN comes from**: Poisson's ratio 0.5 (λ through E is 0/0 — split
  the P-wave modulus directly), zero softening (a body's force on itself is 0·∞), `step(0)`, a cloth
  one particle wide, a central difference at the screen border (the clamped neighbour is the pixel
  itself). Bloom then spreads one NaN pixel over a blotch; its first pass drops non-finite values with
  an integer test (`x != x` may be folded away by compilers that assume no NaN).
- **Units leak through APIs**: MPM stores velocities in cells per substep, so changing `substeps`
  at run time changed every particle's speed until they were rescaled; densities written in kg/m³
  (1000) instead of relative to water overflow the 2²² fixed-point grid.
- **An invalid index is clamped, not rejected, on the GPU**: `setVelocity(-1, …)` (−1 is what a full
  world returns) moved the last body. Validate ids on the CPU.
- **A certificate with fixed randomness certifies nothing against a bad input**: Freivalds' check
  reused the same eight ±1 vectors, and with N > 2⁸ columns some always share a sign pattern — a
  product with two swapped columns passed. Fresh random bits per call restore the 2⁻ᵗ bound.
- **A checker that never looks somewhere cannot fail there**: the backprop law sampled only weights,
  so a zeroed bias gradient passed; sampling also tied the layer to the index. Sample every tensor
  with independent draws, and test the checker against a sabotaged implementation.
- **Data from the network is input, not state**: any peer could send a state that the authority
  adopted, and `decode` would construct any global a message named. Validate the sender, the epoch
  and the shape of every message; construct only whitelisted types.
- **`setTimeout` reorders messages whose delays differ by less than a millisecond**, so a
  "reliable, ordered" simulated channel was neither under load. Sequence numbers fix it.
- **String length is not byte length**: chunks sized in UTF-16 units exceeded the 64 KB data-channel
  part size once encoded (JSON escapes, non-ASCII). Encode to bytes, then split.
- **A tool that is wrong silently is worse than one that fails**: the regex bundler exported only
  the first name of `export const A = 0, B = 1`, and an import with a trailing comment broke the
  bundle. It now rejects every form it does not understand. `new URL(...).pathname` keeps `%20`
  (build scripts failed in folders with spaces: use `fileURLToPath`), and a path guard of
  `startsWith(root)` lets `root + '-other'` through.

## React integration

- **StrictMode mounts, unmounts and mounts again.** Creating GPU objects in effects would make every
  value `undefined` on the first render, and destroying in cleanup would destroy what the second
  mount uses. `useDisposable` creates during render (so a kernel can bind an array made one line
  earlier) and destroys a tick after cleanup, cancelled if the component mounts again at once; an
  object destroyed while a subtree was hidden is re-created when it shows.
- **Devices are pooled with a linger.** Two canvases with the same options share one device, which
  is destroyed 1.5 s after the last one unmounts — so StrictMode, route changes and fast refresh
  reuse it instead of creating and destroying devices (~100 ms each).
- **Children's effects run before their parent's**, so `<Physics>` creates its world during render;
  bodies add themselves in layout effects, joints wait for their bodies' ids (they may be written
  before them) and resolve before the next step.
- **A fixed-step solver must be stepped at a fixed rate.** `MPM.step()` advances 1/60 s; stepping
  it once per display frame ran fluids at twice the speed on 120 Hz screens, and a coupled fluid
  drifted out of lockstep with the world's fixed-step accumulator. A coupled fluid now steps inside
  the world's loop, once per world step.
- **Refs are set to null on unmount**: a test that checks objects after unmounting must capture them
  first. React's StrictMode double-invocation only exists in the development build, so the React
  test bundle uses it (and the examples use the production build).
- `react-dom/server` bundled as ESM for Node needs `require` (it loads `util`): an esbuild banner
  with `createRequire` provides it.

## Website

- **Every image is a real screenshot**: `tools/thumbnails.mjs` runs each example headless at
  1280×800, hides the UI overlays with injected CSS and saves 640×400 JPEGs (about 30 KB each).
- **Author styles beat the `hidden` attribute**: `.poster { display: block }` showed a hidden image
  (and its alt text) until `[hidden] { display: none !important }` was added. Likewise `.btn` declared
  after `.menu-btn` made a mobile-only button appear on desktop.
- **A live hero must not hijack the page**: no orbit controls (they capture the wheel), the pointer
  stirs the water instead, the loop pauses when scrolled out of view, and visitors without WebGPU or
  asking for reduced motion get a still image.

## Process

- **The runner waited 15 minutes on a syntax error**: a page whose module fails to load never sets
  its results. It now fails 5 s after the first page error. A comment inserted mid-line once commented
  out the rest of an object literal; the unit tests now parse every module (`node --check`).
- **Review by independent readers found what tests did not**: each reviewer read a third of the code
  with the instruction to confirm suspicions on the GPU and to list which resources every class owns
  — the inventory that made correct `destroy()` methods possible.
