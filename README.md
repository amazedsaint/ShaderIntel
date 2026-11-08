# ShaderIntel

A zero-dependency WebGPU library for **GPU physics**, **shaders**, **neural networks** and
**deterministic peer-to-peer games**, with a small JavaScript SDK that writes the WebGPU boilerplate
for you. Built from first principles: every solver states the math it implements, every optimisation
was measured on a real GPU, every benchmark verifies its result against a widely used baseline before
it reports a speed-up, and every promise the library makes is an executable law ([LAWS.md](LAWS.md))
checked by a small checker that shares no code with what it checks.

```js
import * as SI from './src/index.js';

const gpu  = await SI.init();                                   // device + capability detection
const view = gpu.view('#canvas', { bloom: true, hud: true });   // HDR target, orbit camera, loop
const pos  = gpu.array('vec4f', 1_000_000);                     // typed GPU array

gpu.kernel({ imports: ['hash'], buffers: { pos },               // write only the body of the kernel
  code: `var s = rngSeed(i, 1u); pos[i] = vec4f(randBall(&s), 1.0);` }).run(1_000_000);

const flow = gpu.kernel({ imports: ['noise'], buffers: { pos }, uniforms: { dt: 'f32' },
  code: `pos[i] += vec4f(curl3(pos[i].xyz) * u.dt, 0.0);` });     // divergence-free curl noise

view.add(gpu.points({ data: pos }));
view.run(({ dt }) => flow.run(1_000_000, { dt }));
```

## What is inside

| Area | What you get |
|---|---|
| **SDK core** | `init()` with feature/limit negotiation (subgroups, f16, timestamps); `gpu.array()` typed GPU arrays with exact WGSL layout; `gpu.kernel()` — write a WGSL body, get bindings, a byte-exact uniform struct, bounds guard and dispatch geometry (auto 2D split past 65,535 groups); a per-dispatch **uniform ring** (correct values for many dispatches in one submit); WGSL module linker; GPU profiler on timestamp queries; readable compile errors with source lines. |
| **Primitives** | Prefix sum (subgroup-accelerated, coalesced, 25–29 G elements/s on an M2 Max), reductions with map, counting-sort uniform grid (27-cell neighbourhood visited as 9 contiguous ranges). |
| **Physics** | **MLS-MPM** (water, oil, honey, jelly, rubber, snow, sand) in lattice units with a shared-memory tiled P2G; **rigid bodies** (boxes, spheres, capsules) with Augmented Vertex Block Descent — per-body 6×6 Newton steps in graph colours, augmented-Lagrangian contacts, isotropic Coulomb friction, GPU hash-table warm starting; **joints** (ball, hinge, weld, distance/rope/spring, slider) with limits, motors and break forces; kinematic bodies, sensors, collision filters, contact events, GPU ray casts, force fields; **two-way fluid ↔ rigid coupling** (bodies float, sink and splash in MPM water and sand); **XPBD cloth**; **Eulerian smoke & fire** on a staggered MAC grid; **N-body** gravity autotuned per GPU; **boids** at 100k–1M agents. |
| **Games & networking** | Deterministic rigid worlds (bit-identical replays from the same inputs, 64-bit GPU state hashes), GPU snapshots and save states; **rollback netcode** (GGPO-style: input delay, prediction, rollback through GPU snapshots, redundant inputs over an unreliable channel, desync detection and automatic resync from the authority's state) over WebRTC or a simulated lossy network, with a bit-exact wire encoding; match **transcripts** that anyone can re-verify. |
| **Neural networks** | Tiled, autotuned **GEMM** with fused bias + activation and split-K; MLPs with Fourier features, backprop, MSE, Adam/SGD; a **population** of per-agent networks trained by evolution strategies in one dispatch. |
| **Verification** | [LAWS.md](LAWS.md): the specification, every line executable (`SI.verify.proveAll`) — exact checkers (scan, sort, colouring, determinism, replay), Freivalds certificates for GEMM, physical laws with tolerances; WGSL **binding contracts** that reject mis-sized buffers at the call. |
| **Interop & typing** | three.js r180+ renders ShaderIntel bodies and particles GPU → GPU on one device; TypeScript declarations in [`types/`](types/shaderintel.d.ts). |
| **Shader library** (WGSL modules) | `hash` (PCG family, lowbias32, RNG, R2, IGN) · `noise` (simplex 2D/3D/4D with analytic gradients, Perlin, value, Worley, fBm with exact gradient, ridged, turbulence, curl) · `sdf` (primitives, smooth/boolean/domain operators) · `raymarch` (two-pass cone marching, normals, soft shadows, AO) · `pbr` (GGX, Smith, Schlick, sky) · `color` (sRGB, OKLab/OKLCh, HSV, palettes, turbo/viridis/inferno/magma) · `tonemap` (AgX, ACES, Khronos PBR Neutral, Reinhard, dithered encode) · `linalg` (3×3 Jacobi eigen, SVD, polar) · `math` (rotations, quaternions, bases). |
| **Rendering** | `View` (HDR, orbit camera, bloom, tonemapping, HUD), sphere impostors, glow points, oriented glyphs, surface meshes, SDF scenes, **screen-space water** (narrow-range filter), **volumetric smoke** (precomputed light transmittance), sky/ground stage. |

## Benchmarks (all results verified)

Each row runs the technique most code uses today and the ShaderIntel kernel on the same device and
inputs, checks that the results agree, and times both (best of interleaved batches of ≥ 20 ms GPU work).
Measured headless with `node tools/headless.mjs bench` on an Apple M2 Max (Chrome 157, Metal) and on
the NVIDIA GB10 of a DGX Spark (Chrome 155, Linux, Vulkan; a shared machine, so an upper bound).
Open [`bench/index.html`](bench/index.html) to reproduce on your GPU.

| Benchmark | Baseline (what most code does) | ShaderIntel | M2 Max | GB10 | Check |
|---|---|---|---|---|---|
| Prefix sum, 16M u32 | Blelloch scan, GPU Gems 3 ch. 39 | coalesced reduce-then-scan + subgroup scans | **2.3×** (31.8 G el/s) | **1.3×** (16.9 G el/s) | equals CPU scan |
| Prefix sum, 16M u32 | Hillis–Steele in global memory | same | **15×** | **17×** | equals CPU scan |
| 3D simplex noise, 16M | Ashima / webgl-noise `snoise` | integer lattice hash + `unpack4x8snorm` gradients | **1.32×** (20.6 G/s) | **1.21×** (38.5 G/s) | same range and σ |
| 3D simplex noise, 16M | psrdnoise (Gustavson & McEwan 2022) | same | **1.55×** | **1.20×** | — |
| 4-octave fBm, 16M | 4 × Ashima `snoise` | `fbm3` | **1.35×** | **1.26×** | — |
| Per-pixel hash, 16M | `fract(sin(dot(p, …)) · 43758.5453)` | `pcg2d`, fully mixed output | **1.00×** | **0.99×** | χ² 246 vs 6.5M (values), 226 vs 2.5M (neighbour pairs) |
| N-body, 65,536 bodies | one thread per body, global memory | tiles + register blocking, autotuned | **1.49×** (209 G pairs/s) | **1.34×** (650 G pairs/s) | bit-identical |
| Fixed-radius neighbours, 65,536 | all-pairs O(N²) (WebGPU "compute boids") | counting-sort grid | **197×** | **27×** | identical counts |
| MLS-MPM P2G, 262k particles | unsorted, global fixed-point atomics (WebGPU-Ocean / PB-MPM style) | Morton block sort + shared-memory tiles | **3.1×** (648 M p/s) | **2.9×** (595 M p/s) | bit-identical grids |
| SDF ray marching, 1080p | classic per-pixel sphere tracing | two-pass cone marching | **2.5×** (4.1 G rays/s) | **1.5×** (7.2 G rays/s) | 99.996% of pixels identical |
| Matrix multiply, 2048³ f32 | 16×16 shared-memory tiled kernel (CUDA guide / WebGPU tutorials) | register-blocked vec4 tiles, autotuned | **2.3×** (3.3 TFLOP/s) | **4.7×** (10.8 TFLOP/s) | Freivalds certificate + bit-identical |
| Matrix multiply, 2048³ f32 | naive, one thread per output | same | **7.1×** | **15.5×** | Freivalds certificate + bit-identical |

Solver accuracy is tested the same way: MPM conserves mass and momentum to fixed-point precision, the
smoke projection removes 99.6% of the divergence, a rigid box slides down a 40° slope at 1.801 m/s²
(Coulomb's `g(sin θ − μ cos θ)`: 1.797) while a 20-row box pyramid holds to 1.3 mm over 6 s, a
pendulum keeps its period to 0.23%, a chain of capsule links holds a box 347 times heavier than a link
with 4.7 mm of sag, and balls roll at constant speed. A pile of 6,004 mixed rigid bodies (28k contacts) steps in
**3.6 ms** on the M2 Max and **3.1 ms** on the GB10 with 8 solver iterations, and a 38-body two-player
world in about a millisecond: worlds of up to 128 bodies run their whole solve as one fused dispatch.

What did **not** win is documented too: over-relaxed sphere tracing, subgroup-broadcast N-body, a
single-pass decoupled-look-back scan and cheaper noise gradients all measured slower or worse on this
hardware — see [docs/LEARNINGS.md](docs/LEARNINGS.md).

## Examples

Serve the folder and open `http://localhost:8787/`:

```bash
node tools/serve.mjs
```

| Example | Shows |
|---|---|
| [`examples/quickstart.html`](examples/quickstart.html) | the SDK in 30 lines: 1M particles in curl noise |
| [`examples/mpm.html`](examples/mpm.html) | MLS-MPM water, jelly, rubber, snow, sand; screen-space water surface; push with a ball |
| [`examples/rigid.html`](examples/rigid.html) | rigid bodies: a 210-box pyramid, a brick wall hit by a cannonball, a 6,000-body pile; alt-click fires balls |
| [`examples/joints.html`](examples/joints.html) | ragdolls tumbling down stairs, a rope bridge, motorised machines (car, piston, paddle wheel), a wrecking ball through a breakable wall |
| [`examples/coupling.html`](examples/coupling.html) | rigid bodies floating, sinking and splashing in MPM water; a dam break against crates; balls in sand |
| [`examples/p2p.html`](examples/p2p.html) | two peers with their own GPU worlds over a simulated lossy network, rollback netcode, matching state hashes; `?webrtc=room` for two tabs |
| [`examples/neural.html`](examples/neural.html) | an MLP with Fourier features learns an image on the GPU: training and inference every frame |
| [`examples/three.html`](examples/three.html) | ShaderIntel physics rendered by three.js (shadows, materials) on the same GPU device, poses copied GPU → GPU; loads three.js from a CDN |
| [`examples/cloth.html`](examples/cloth.html) | XPBD cloth in gusty wind over sphere and box colliders |
| [`examples/smoke.html`](examples/smoke.html) | 3D smoke and fire, volumetric lighting |
| [`examples/nbody.html`](examples/nbody.html) | galaxy collision, autotuned O(N²) gravity |
| [`examples/boids.html`](examples/boids.html) | 131k–1M flocking boids with a predator |
| [`examples/sdf.html`](examples/sdf.html) | an SDF scene from one `map()` function, cone-marched with soft shadows and AO |
| [`examples/noise.html`](examples/noise.html) | domain warping, terrain lit by analytic noise gradients, Worley, curl flow, fire |
| [`bench/index.html`](bench/index.html) | the benchmark suite above |
| [`tests/index.html`](tests/index.html) | 114 tests: primitives vs CPU, noise gradients vs finite differences, conservation laws, rigid-body friction, stacking, joints and rolling, GEMM and backprop, buoyancy, two peers over a lossy network, every law in LAWS.md |

## Testing from a terminal

The test suite, the benchmarks and any experiment run headless in Chrome, driven over the DevTools
protocol by a zero-dependency script (Node 22+):

```bash
node tools/serve.mjs &
node tools/headless.mjs tests            # the 114 tests; exit code 1 on any failure
node tools/headless.mjs bench            # the verified benchmarks
node tools/headless.mjs info             # adapter, features and limits WebGPU reports
npm run typecheck                        # the TypeScript declarations, compiled against a typed usage file
```

On Linux with an NVIDIA GPU, `tools/docker/run.sh tests` does the same with Chrome on Vulkan; that
is how the suite runs on a DGX Spark (GB10). `tools/docker/Dockerfile` is the matching container
recipe — see docs/LEARNINGS.md for why Chrome's WebGPU currently has to run on the host there.

## Requirements

WebGPU: Chrome / Edge 113+, Safari 26+, or Firefox 141+ on Windows, served from `http://localhost` or
https. Subgroups, `shader-f16` and timestamp queries are used when present and fall back to portable
code paths when not (the test suite runs both).

## Layout

```
src/
  core/      context (device, batching, uniform ring), kernel builder + binding contracts, GPU arrays, WGSL linker, types, profiler
  prims/     scan, reduce, spatial grid
  shaders/   WGSL library modules: math, hash, noise, sdf, raymarch, color + tonemap, pbr, linalg
  physics/   mpm, rigid (AVBD: bodies, joints, narrowphase, rays), coupling (fluid ↔ rigid), cloth, smoke, nbody, boids
  render/    view, programs, spheres, points, glyphs, mesh, bodies, fluid (SSFR), volume, sdf, bloom, scene
  nn/        gemm (tiled, autotuned, split-K), MLP, losses, optimisers, evolution-strategies populations
  net/       rollback sessions, transcripts, transports (WebRTC, BroadcastChannel / WebSocket signalling, loopback)
  verify/    checkers and the executable laws
  interop/   three.js
  ui.js      tiny control panel for examples
types/       TypeScript declarations      LAWS.md   the specification, as executable laws
examples/    HTML demos                   bench/    benchmark suite + baselines
tests/       browser test suite           docs/     GUIDE.md (API + math), LEARNINGS.md
tools/       serve.mjs (static server), build.mjs (single-file bundle -> dist/)
```

## Documentation

- [docs/GUIDE.md](docs/GUIDE.md) — API reference and the first-principles derivations behind each solver.
- [LAWS.md](LAWS.md) — what the library promises, and the checker for each promise.
- [docs/LEARNINGS.md](docs/LEARNINGS.md) — what was measured, what failed, and WebGPU pitfalls.

`package.json` declares the package (`import * as SI from 'shaderintel'`, or `shaderintel/bundle`
for the single-file build); it is marked private until a license is chosen.
