# ShaderIntel guide

API reference plus the derivations behind every solver. Code lives in `src/`; everything is plain ES
modules with no dependencies.

- [1. Core SDK](#1-core-sdk)
- [2. Rendering](#2-rendering)
- [3. Parallel primitives](#3-parallel-primitives)
- [4. Physics](#4-physics) — MPM, cloth, smoke, N-body, boids, rigid bodies and joints, fluid ↔ rigid coupling
- [5. Neural networks](#5-neural-networks--sinn)
- [6. Peer-to-peer games](#6-peer-to-peer-games--sinet)
- [7. Verification](#7-verification--siverify-lawsmd)
- [8. three.js](#8-threejs--sithree)
- [9. Shader library reference](#9-shader-library-reference)
- [10. First-principles design notes](#10-first-principles-design-notes)
- [11. React](#11-react)

---

## 1. Core SDK

### 1.1 `init(opts)` → `Context`

```js
import * as SI from './src/index.js';
const gpu = await SI.init({ powerPreference: 'high-performance' });
gpu.caps // { subgroups, f16, timestamps, subgroupMin, subgroupMax, subgroupId, vendor, architecture, workgroupStorage, ... }
```

Requests every useful optional feature the adapter offers (`subgroups`, `shader-f16`,
`timestamp-query`, `float32-filterable`, …) and the adapter's maximum limits (storage buffer size,
workgroup storage, …). Pass `{ subgroups: false }` etc. to force the portable code paths.

`init()` rejects with an `SI.SIError` whose `code` says why: `'unsupported'` (no WebGPU in this
browser), `'no-adapter'` (WebGPU is there but no GPU is available) or `'no-device'`.
`SI.isSupported()` answers the first question without starting anything. Every error ShaderIntel
throws on purpose is an `SIError`; the codes `'compile'` (WGSL errors, with source lines) and
`'invalid-argument'` cover most of the rest.

`init({ device, adapter })` wraps a device created elsewhere (three.js's, say) instead of requesting
one; `destroy()` then leaves that device alive.

### 1.2 GPU arrays — `gpu.array(type, lengthOrData, opts)`

```js
const pos = gpu.array('vec4f', 100_000);          // zeroed
const ids = gpu.array('u32', new Uint32Array(n));  // uploaded
pos.write(float32Array, elementOffset);            // vec3 data is padded to its 16-byte stride automatically
const data = await pos.read();                     // typed array (keeps padding); readPacked() drops it
gpu.clear(pos); gpu.fill(ids, 7); gpu.fillFloat(pos, 1.5); gpu.copy(a, b);
```

`write()` throws if the data does not fit. It goes through `queue.writeBuffer`, which lands before
any work recorded but not yet submitted; `arr.writeOrdered(data)` instead records the upload in the
command stream, so dispatches recorded before it see the old contents and those after it the new
(simulations upload their colliders and sources this way, once per step).

Types follow WGSL host-shareable layout exactly (`f32 i32 u32 f16 vecNf/i/u/h matCxRf`). Arrays carry
`STORAGE | COPY_SRC | COPY_DST` (+ `opts.usage`). Any array can be bound under a *different* WGSL element
type in a kernel (it is just bytes) — e.g. the MPM particle buffer is a `vec4f` array bound as
`array<Particle>`.

### 1.3 Kernels — `gpu.kernel(desc)`

Write the body of a compute shader; ShaderIntel generates the rest.

```js
const k = gpu.kernel({
  name: 'integrate',                       // for errors and the profiler
  buffers: {                               // storage bindings, in declaration order
    pos,                                   //   GArray -> var<storage, read_write> pos: array<vec4f>
    vel: { array: velArr, access: 'read' },//   read-only binding
    cnt: { type: 'u32', atomic: true },    //   array<atomic<u32>>, bound later
    P:   { type: 'Particle' },             //   any struct type defined in imports / shared
  },
  textures: { src: { sampleType: 'float', dim: '3d' }, dst: { storage: 'rgba16float', dim: '3d', access: 'write' } },
  samplers: { lin: 'linear' },
  uniforms: { dt: 'f32', g: 'vec3f', M: 'mat4x4f' },   // struct Params, read as u.dt
  imports: ['noise', 'hash'],              // shader-library modules (also `#import noise` lines in code)
  defines: { K: 4, EPS: 1e-6 },            // const K = 4; const EPS = 0.000001;
  shared: 'var<workgroup> tile: array<vec4f, 256>; fn helper() {}',   // module-scope WGSL
  workgroupSize: 256,                      // or [8, 8] (2D) / [4, 4, 4] (3D)
  guard: true,                             // emit `if (i >= u.n) { return; }` (set false when using barriers)
  code: `vel[i] = vel[i] + vec4f(u.g * u.dt, 0.0); pos[i] = pos[i] + vel[i] * u.dt;`,
});
await k.ready();                           // optional: rejects with line-numbered WGSL errors
k.run(n, { dt: 1 / 60 }, { pos: otherArray }); // count, uniforms, binding overrides (all persistent)
k.run([W, H]);                             // 2D/3D kernels: invocation extent
k.dispatch(groups, uniforms, bindings);    // explicit workgroup counts
k.dispatchIndirect(argsArray, byteOffset);
```

Names available in `code`:

| name | meaning |
|---|---|
| `i` | linear invocation index (1D kernels), robust to the 2D split past 65,535 workgroups |
| `p` | `gid` (2D/3D kernels) |
| `u` | the uniform struct: `u.n` (1D count) or `u.size` (nD extent) + your uniforms |
| `gid lid lidx wid nwg wgi` | builtins; `wgi` = linear workgroup index |
| `sgi sgs` | subgroup lane / size (using them enables subgroups) |
| `sgid nsg` | subgroup index / count (native `subgroup_id` when available) |
| `WG`, `WG_SIZE` | workgroup size constants |

**Uniform ring.** `queue.writeBuffer` executes before the command buffer runs, so writing one uniform
buffer between dispatches of the same submit gives *every* dispatch the last value — a classic WebGPU
bug. Each dispatch therefore gets its own 256-byte slot in a ring bound with a dynamic offset;
identical consecutive values reuse their slot; the ring is uploaded once per submit.

### 1.4 Recording model

Everything recorded through the context (kernels, clears, copies, render passes) goes into one
command encoder; compute dispatches share one compute pass until something forces it to end.
`gpu.flush()` submits; `await gpu.idle()` submits and waits; `View.run` flushes once per frame.
`gpu.region('label', () => { … })` turns a block into one timestamped pass;
`gpu.profiler.report()` returns smoothed GPU milliseconds per label (Chrome quantises timestamps to
100 µs without developer flags, hence the averaging).

### 1.5 WGSL modules

```js
SI.defineModule('mylib', { deps: ['noise'], code: `fn wobble(p: vec3f) -> f32 { return simplex3(p * 2.0); }` });
SI.defineModule('fast', { code: (caps) => caps.subgroups ? `enable subgroups; …` : `…portable…` });
gpu.kernel({ imports: ['mylib'], … });
```

The linker resolves dependencies, emits each module once and hoists `enable` / `requires` /
`diagnostic` directives. WGSL allows module-scope declarations in any order, so no sorting is needed.

### 1.6 Reading back

`await arr.read()`, `await arr.readPacked()`, `await gpu.readTexture(tex)` (r32float, rgba16float,
rgba32float, rgba8unorm, … — half floats are expanded).

### 1.7 Lifecycle

Everything that allocates GPU memory has an idempotent `destroy()`: arrays, kernels' helpers,
primitives (`Scan`, `Reduce`, `SpatialGrid`), every simulation (`RigidWorld`, `MPM`, `Cloth`,
`Smoke`, `NBody`, `Boids`), every layer, every network and optimiser. An object frees what it
allocated itself, never what it was given: a `Points` layer does not free the array it draws, and
snapshots from `world.snapshot()` belong to the caller. `destroy()` first submits recorded work
that may still use the memory (a destroyed buffer would make the whole submit fail); using an
object after `destroy()` throws an `SIError`, and layers whose source was destroyed draw nothing.

`view.destroy()` stops the loop, removes its listeners, resize observer and HUD, unconfigures the
canvas and frees its textures; `view.destroy({ layers: true })` destroys the attached layers too.
`gpu.destroy()` releases the device and with it everything made on it.

---

## 2. Rendering

### 2.1 `gpu.view(canvas, opts)`

```js
const view = gpu.view('#c', {
  camera: { target: [0, 0.5, 0], distance: 3, yaw: 0.6, pitch: 0.35, fov: 45, autoRotate: 0 },
  hud: true, bloom: { strength: 0.06 }, tonemap: 'agx' /* aces | neutral | reinhard | none */,
  exposure: 1, background: [r, g, b, 1], controls: true,
});
view.add(layer);
view.run(({ dt, time, frame, view }) => { /* record simulation work */ });
const { origin, dir } = view.ray();       // world ray under the pointer; view.pointer has buttons/modifiers
view.stop(); view.run();                  // pause and resume (one loop per view)
view.setBloom({ strength: 0.1 }); view.setHud(false); view.setPixelRatio(1);
view.camera.set({ target: [0, 1, 0], distance: 5, autoRotate: 0.2 });
view.destroy();
```

`opts.onError(e)` receives errors thrown by the frame callback or rendering (the loop stops); by
default they are shown over the canvas. `view.onFrame` is the per-frame callback that `run(fn)` sets
and `view.step(dt)` calls, for driving frames by hand. Camera options also include `minPitch`,
`maxPitch`, `damping`, `rotateSpeed`, `zoomSpeed`, `panSpeed` and `enabled`. The device-pixel ratio
follows the screen (moving a window between displays) unless `pixelRatio` is fixed, and the
drawing buffer is capped at the device's texture limit.

Frame: `prepare(view)` hooks (compute) → HDR scene pass (`rgba16float` + `depth32float`), every
layer's `draw(pass, view)` → `afterScene(view)` hooks (screen-space effects) → bloom → tonemap with
triangular dither → `overlay(pass, view)`.

Every render program sees a `cam` uniform: `view proj viewProj invViewProj invView invProj eye time
viewport near far fovY pixelAngle frame dt`.

### 2.2 `gpu.program(desc)` — render pipelines with generated bindings

Same idea as kernels: declare `buffers / textures / samplers / uniforms / imports`, write `@vertex fn
vs` and `@fragment fn fs`, then `program.draw(pass, view, vertexCount, instanceCount, uniforms,
bindings)` or `drawIndexed(...)`.

### 2.3 Layers

| Factory | Use |
|---|---|
| `gpu.stage(opts)` | sky gradient + infinite anti-aliased grid floor (writes depth) |
| `SI.Lines.box(gpu, min, max, color)` | line geometry |
| `gpu.spheres({ data, stride, position, velocity, material, count, radius, scale, offset, palette, colorBy, mask })` | ray-cast sphere impostors (pixel-exact depth and normals at 4 vertices per particle) |
| `gpu.points({ data, size, pixelSize, colorBy, color, intensity })` | additive Gaussian glow sprites (energy-conserving minimum size) |
| `gpu.glyphs({ pos, vel, count, size, colorBy })` | velocity-aligned darts for agents |
| `gpu.mesh({ pos, nrm, indices, indexCount, grid, front, back })` | two-sided mesh from storage buffers |
| `gpu.bodies(world)` | rigid bodies straight from the solver's body buffer: instanced boxes, ray-cast spheres |
| `gpu.fullscreen({ code: 'fn shade(uv, px) -> vec4f', imports, uniforms })` | Shadertoy-style layer; `cameraRay(uv)` available |
| `gpu.sdf({ code: 'fn map(p) -> vec2f …', imports, uniforms, values })` | cone-marched SDF scene with shadows, AO, PBR |
| `sim.fluid()` (MPM) / `new SI.FluidSurface(gpu, …)` | screen-space water surface |
| `gpu.volume({ smoke })` | volumetric smoke/fire |

---

## 3. Parallel primitives

### Prefix sum — `new SI.Scan(gpu, maxN, { mode })`

`scan.run(src, dst = src, n)` — exclusive scan of u32. Tiles of 4096 values (256 threads × 16). Each
tile is loaded with lane-contiguous `vec4` reads into workgroup memory and handed to threads as
contiguous runs (+15% vs per-thread blocked loads). A thread scans its 16 values serially; the
workgroup scan is `subgroupInclusiveAdd` followed by every subgroup scanning the subgroup totals
redundantly and picking its offset with `subgroupShuffle` (one barrier). Multi-tile arrays use
reduce-then-scan (3N traffic, ~55% of memcpy speed). `mode: 'single-pass'` enables a chained scan with
decoupled look-back and fallback (2N traffic) — correct, but slower on Apple GPUs (see LEARNINGS).

### Reduction — `new SI.Reduce(gpu, { type, input, op, map, maxN })`

`op ∈ sum | min | max` over `f32 / u32 / i32 / vecN`; `map` is a WGSL expression of `x`, e.g. max
speed: `{ input: 'vec4f', type: 'f32', op: 'max', map: 'length(x.xyz)' }`. `run(src)` leaves the
result on the GPU (`result[0]`); `await read()` fetches it.

### Spatial grid — `new SI.SpatialGrid(gpu, { maxCount, min, max, cellSize })`

Counting sort into a uniform grid: `count` (atomic rank per cell) → `scan` (cell starts) → `scatter`
(sorted order). Cells are linearised x-fastest, so the three cells of a row are contiguous in sorted
order and a 27-cell neighbourhood is 9 contiguous ranges:

```js
grid.build(pos, n); grid.gather(pos, sortedPos);
gpu.kernel({ imports: ['grid'], uniforms: { ...grid.uniforms(), h2: 'f32' },
  buffers: { S: sortedPos, start: grid.start, out },
  code: `let pi = S[i].xyz; var c = 0u;
    ${SI.SpatialGrid.neighbors('pi', 'let d = S[j].xyz - pi; if (dot(d, d) < u.h2) { c++; }')}
    out[i] = c;` }).run(n, { ...grid.values(), h2: h * h });
```

---

## 4. Physics

### 4.1 MLS-MPM — `gpu.mpm(opts)`

```js
const sim = gpu.mpm({ grid: 64, size: [1, 1, 1], maxParticles: 1 << 20, substeps: 8, timeScale: 1, sortEvery: 4 });
sim.add({ box: [[x0, y0, z0], [x1, y1, z1]] }, { material: 'water', velocity: [0, 0, 0] });
sim.add({ sphere: [[cx, cy, cz], r] }, { material: 'jelly' });   // also cylinder, sdf + bounds, points
sim.setColliders([{ type: 'sphere', center, radius, velocity, friction }, { type: 'box', center, half }, { type: 'plane', center, normal }]);
sim.step();                                  // one frame = substeps × substep
view.add(sim.spheres({ solidsOnly: true })); view.add(sim.fluid());
```

Materials (`MATERIALS`): `water oil honey` (fluid), `jelly rubber` (Neo-Hookean), `snow`
(Stomakhin 2013), `sand` (Drucker–Prager, Klár 2016), or your own `{ kind, density, stiffness,
poisson, viscosity, hardening, compression, stretch, friction }`, where `kind` is 0–3 or a name from
`MPM_KINDS` (`'fluid'`, `'jelly'`, `'snow'`, `'sand'`). Densities are relative to water (at most 50:
the fixed-point grid would overflow near 300), Poisson's ratio is clamped to [0, 0.49] and viscosity
to its stable limit. At most 8 materials exist at once; `clear()` forgets them. `substeps` and
`timeScale` can change at any time: particle velocities are stored per substep, so they are
rescaled to keep their physical speed. `sim.decouple()` ends a coupling made by `couple()`.

**Math, in lattice units** (`dx = 1`, `Δt = 1`, positions in cells, velocities in cells/substep):

- Quadratic B-spline weights with `base = ⌊x − ½⌋`, `f = x − base`:
  `w₀ = ½(1.5 − f)²`, `w₁ = ¾ − (f − 1)²`, `w₂ = ½(f − ½)²`, and `D⁻¹ = 4`.
- P2G: `mᵢ += Σ w mₚ`, `(mv)ᵢ += Σ w [mₚ vₚ + Qₚ (xᵢ − xₚ)]` with
  `Qₚ = −D⁻¹ V⁰ₚ τₚ + mₚ Cₚ` (τ = P Fᵀ, the Kirchhoff stress). Because `Σᵢ wᵢₚ (xᵢ − xₚ) = 0` for
  B-splines, the affine and stress terms move momentum between nodes but never create it — the test
  suite checks mass and momentum conservation to 1e-7.
- Grid: `vᵢ = (mv)ᵢ / mᵢ + g'`, separating walls, floor friction, SDF colliders.
- G2P (APIC): `v = Σ w vᵢ`, `C = D⁻¹ Σ w vᵢ (xᵢ − xₚ)ᵀ`, `x += v`, `F ← (I + C) F`.
- Stress evaluated at the end of G2P and stored in the particle's spare `.w` lanes, so P2G is
  material-agnostic:
  - fluid: `J ← J (1 + tr C)`, `τ = K (J − 1) I + μ J (C + Cᵀ)` (linear EOS: constant wave speed);
  - Neo-Hookean: `τ = μ (F Fᵀ − I) + λ ln J I` (no SVD);
  - snow: SVD, clamp σ to `[1 − θc, 1 + θs]`, hardening `e^{ξ(1 − Jp)}`, fixed-corotated τ;
  - sand: SVD, Hencky strain `ε = ln σ`, Drucker–Prager return mapping, `τ = U diag(2με + λ tr ε) Uᵀ`.
- **Stability by construction.** In lattice units the P-wave speed is `c' = c Δt/dx`. Each material's
  stiffness is a fraction of `c'max = 0.4`, so `M = λ + 2μ = ρ c'²` and a material cannot be
  configured to explode; more substeps buy stiffer, less compressible materials. Gravity is
  `g' = g Δt²/dx`; particle speed is clamped below one cell per substep (CFL).
- **Fixed-point atomics.** WebGPU only has integer atomics. In lattice units every quantity of a
  stable simulation is O(1), so i32 with scale 2²² is exact enough and cannot overflow before the
  simulation itself would be unstable. Integer sums are order-independent, so the tiled and the
  global-atomic P2G produce *bit-identical* grids.
- **Tiled P2G.** Particles are counting-sorted every few substeps into 4³-cell blocks in Morton order.
  A workgroup takes 512 consecutive particles, computes the AABB of their stencils (subgroup min/max),
  accumulates into a shared-memory tile with workgroup atomics when it fits (≈27 KB budget), then
  flushes only non-empty nodes with global atomics: 2.8× faster than global atomics on unsorted
  particles.

### 4.2 XPBD cloth — `gpu.cloth(opts)`

```js
const cloth = gpu.cloth({ width: 128, height: 128, size: [1.6, 1.6], origin, pin: 'top' | 'top-sparse' | 'corners' | 'none' | (i, j) => bool,
  substeps: 20, stretch: 0, shear: 1e-6, bend: 2e-4, wind: [0, 0, 2], colliders: [...], friction: 0.4 });
cloth.floorY = 0; cloth.step(dt);
view.add(gpu.mesh({ pos: cloth.pos, nrm: cloth.nrm, indices: cloth.indices, indexCount: cloth.indexCount, grid: [128, 128] }));
```

- Distance constraint `C = |xₐ − x_b| − L`, gradient `n = (xₐ − x_b)/|xₐ − x_b|`, XPBD update with the
  multiplier reset every substep (one iteration): `Δλ = −C / (wₐ + w_b + α/Δt²)`,
  `xₐ += wₐ Δλ n`, `x_b −= w_b Δλ n`. Compliance α (m/N) is a physical material property independent
  of the iteration count — the point of XPBD.
- Small steps (Macklin et al. 2019): many substeps × 1 iteration converge faster than many iterations
  of one step, and damping falls out naturally.
- Parallel Gauss–Seidel: six families (stretch-x, stretch-y, two shears, bend-x, bend-y) split into two
  colours each such that no two constraints of one dispatch share a particle — 12 race-free
  dispatches per substep, no atomics.
- Long-range attachments (Kim et al. 2012): every particle is tethered to its nearest pin with a
  unilateral `|x − x_pin| ≤ L_rest` — measured peak strain 2.23% → 0.81% on a hanging sheet.
- Wind: drag on the normal component of the relative velocity, gusts from simplex noise. Colliders:
  spheres, boxes, floor, with friction.

### 4.3 Smoke & fire — `gpu.smoke(opts)` + `gpu.volume({ smoke })`

Staggered MAC grid on 3D `rgba16float` textures (velocity faces) with hardware trilinear sampling.
Per step: semi-Lagrangian advection of faces and scalars → vorticity confinement
`f = ε dx (N × ω)`, `N = ∇|ω| / |∇|ω||` → buoyancy `v_y += Δt (β T − κ ρ)` → sources →
divergence → pressure Poisson `∇²p = ∇·v` by red–black SOR (ω = 1.8, warm-started) → projection
`v −= ∇p`. On a MAC grid `div ∘ grad` *is* the compact 7-point Laplacian, so the projection is exact
up to solver tolerance: residual divergence ×0.004 (a collocated grid leaves ×0.42 — measured).

Volume rendering: per-voxel light transmittance toward the sun (compute), then a view-ray march
(Henyey–Greenstein phase, sky ambient, temperature emission) stopped at the scene depth, composited
with premultiplied alpha.

### 4.4 N-body — `gpu.nbody(opts)`

`a_i = G Σ_j m_j (x_j − x_i) / (|x_j − x_i|² + ε²)^{3/2}` (Plummer softening), symplectic Euler
(kick–drift), double-buffered positions. `await sim.autotune()` times every variant on the device
(naive, shared-memory tiles with workgroup 64/128/256 × register blocking K = 1/2/4, subgroup
broadcast) and caches the winner per adapter in `localStorage`. Initial conditions: `galaxies()`,
`plummer()`. Energy drift over 200 steps: 3.3e-4.

### 4.5 Boids — `gpu.boids(opts)`

Separation `Σ −d/|d|²`, alignment `⟨v⟩ − v`, cohesion `⟨d⟩` over a perception radius, soft walls,
optional predator; neighbours through the counting-sort grid, agents re-sorted every frame (the arrays
stay permuted — boids need no identity).

### 4.6 Rigid bodies — `gpu.rigid(opts)` + `gpu.bodies(world)`

```js
const world = gpu.rigid({ maxBodies: 16384, iterations: 8, substeps: 1, floor: 0, gravity: [0, -9.81, 0] });
const box = world.addBox({ position: [0, 1, 0], half: [0.5, 0.25, 0.5], rotation: [rx, ry, rz] /* or [qx, qy, qz, qw] */,
  velocity, angularVelocity, density: 1, friction: 0.6 });
const ball = world.addSphere({ position, radius: 0.3 });
const pill = world.addCapsule({ position, radius: 0.1, halfHeight: 0.3 });     // along its local y axis
world.addBox({ position: [0, -1, 0], half: [20, 1, 20], static: true });
world.step(1 / 60);              // uploads what was added since the last step, then simulates
view.add(gpu.bodies(world));     // instanced boxes, ray-cast spheres and capsules, straight from the body buffer
await world.readPoses();         // Float32Array of [x, y, z, qx, qy, qz, qw] per body
await world.readBodies();        // { position, rotation, velocity, angularVelocity, flags }
```

Every edit is a queued command applied on the GPU at the start of the next step, so it is cheap and
ordered with the simulation (an id that does not exist throws: on the GPU it would land on another
body): `setPose`, `setVelocity`, `applyImpulse(id, J, point)`, `applyForce`,
`applyAngularImpulse`, `moveKinematic(id, position, rotation)` (kinematic bodies follow a path and
push dynamic ones), `setFilter(id, group, mask)` (bodies collide when `a.group & b.mask` and
`b.group & a.mask`), `setEnabled`, `setProps({ gravityScale, linearDamping, angularDamping })`,
`remove(id)` (slots are reused). Body options also take `kinematic`, `sensor` (reports contacts,
exerts no force), `fixedRotation`, `group`, `mask` and `userData`.

**Joints.** `world.addJoint({ type, a, b, anchor, axis, ... })` with `b: null` for the world; anchors
and axes are given in world space.

| type | removes | options |
|---|---|---|
| `ball` | 3 translations | — |
| `hinge` | 3 translations + 2 rotations | `axis`, `limits: [lo, hi]` (rad), `motor: { speed, maxForce }` |
| `fixed` | all 6 | — (a weld; add `breakForce` for breakable walls) |
| `distance` | the distance | `length`, `anchorB`, `rope: true` (only resists stretching), `stiffness` / `damping` (a spring) |
| `slider` | 2 translations + 3 rotations | `axis`, `limits` (m), `motor` |

`breakForce` (N) on any joint, `collideConnected` (default false), and at run time
`setMotor(j, speed, maxForce?)` (the force limit defaults to the joint's current one; `speed` null
turns the motor off), `setLimits(j, lo, hi)` / `setLimits(j, null)`, `setJointEnabled`;
`readJoints()` → `{ broken, force }`.

**Queries and events.** `raycast([{ origin, dir, maxDistance, mask }])` → `{ hit, body, distance,
point, normal }` per ray, or `rayQuery(n)` for a reusable GPU query that other kernels can fill and
read (each (body, ray) pair is a thread; the nearest hit is chosen with two integer `atomicMin`
passes, so the answer does not depend on thread order). With `{ events: true }`, `readEvents()`
returns the contact pairs that began and ended in the last step (sensors included).

**Extension: force fields.** `world.addForceField({ code, uniforms, values })` runs a WGSL snippet
for every dynamic body before each frame; it sees the body `b` and adds to `force` / `torque`
(buoyancy, wind, magnets, a learned controller writing into a buffer...).

**Determinism, snapshots, save games.** With `{ deterministic: true }` a world is a pure function of
its inputs on a given device and browser: no result depends on the order in which GPU threads run
(see *Pipeline* below), and `await world.hash()` returns a 64-bit hash of every body's state.
`world.snapshot(old?)` copies everything the next step depends on on the GPU (bodies, joints,
pending impulses, last step's contacts for warm starting) and `world.restore(s)` returns to it — the
basis of rollback netcode (§6). `exportState()` / `importState()` do the same through the CPU for
save games and network resync.

A world needs 10 storage buffers per shader stage (98% of devices; the constructor throws an
`SIError` with code `'unsupported'` otherwise), and the fused solve for small worlds needs 18 KB of
workgroup memory (it falls back to the multi-dispatch solve below that). `step(0)` does nothing.

Solver options (defaults are the tested values): `iterations` 8, `substeps` 1, `alpha` 0.95
(fraction of an existing overlap kept per step), `beta` 30 (stiffness ramp, in units of m_eff/Δt²
per metre of error), `gamma` 0.99 (stiffness decay between steps), `kStart` 1e-3 and `kStartJoint`
1 (minimum stiffness × m_eff/Δt²), `jointMassFloor` 1 (kg), `contactMassFloor` 0, `contactTolerance`
0, `margin` 0.02 (speculative contact distance), `slop` 5e-4 (tolerated overlap), `stickThreshold`
0.01, `postStabilize` false, `deterministic` false, `events` false.

**Math — Augmented Vertex Block Descent** (Giles, Diaz & Yuksel 2025, building on Vertex Block
Descent, Chen et al. 2024). Every body is one 6-DOF "vertex"; a step minimises, body by body in
graph-colour order (block Gauss–Seidel), the local energy

`G(x) = ½ |x − y|²_M / Δt² + Σ_c [ ½ k_c C_c(x)² + λ_c C_c(x) ]`,  `y = x₀ + vΔt + gΔt²`.

- **Linearised contacts.** Each contact row (normal `n`, tangents `t₁ t₂`) is expanded once at the
  start of the step, `C(x) = C₀ + J·Δx` with `J = ±[d, r × d]`, so every local problem is exactly
  quadratic and a single Newton step `H Δ = −∇G` solves it:
  `H = M/Δt² + Σ k J Jᵀ`, `∇G = M/Δt² (x − y) + Σ f J`.
- **Forces and the friction cone.** `f_n = min(0, k C_n + λ_n)`; the tangential force
  `k C_t + λ_t` is projected onto the disc `|f_t| ≤ μ |λ_n|` (isotropic Coulomb) of the *updated*
  normal force, so every stored pair satisfies the cone exactly (law `contacts`). The penalty `k`
  stays in `H` even when a force is clamped — dropping it lets the next Newton step jump through the
  contact and the iteration 2-cycles.
- **Dual update** (augmented Lagrangian) after every sweep: `λ ← f` (clamped as above), and
  `k ← min(k_max, k + β|C|)` while a constraint is active and not sliding. Next step, `λ` is warm
  started as `αγλ` and `k` as `max(k_min, γk)` from the nearest contact point of the same body pair,
  found through a GPU hash table keyed by the pair. A rolling contact moves about `|Δv|·Δt` over the
  body each step, so the search radius grows with the relative speed, and a one-point manifold is
  always matched — otherwise a ball rolling at 3 m/s restarts from zero force every step and sinks.
- **Overlap and static friction.** A penetrating contact enforces `C − αC₀ ≥ 0`, removing `(1 − α)`
  of the overlap per step without injecting energy; speculative contacts (`C₀ > 0`, within the
  margin) count in full so fast bodies cannot tunnel. A contact that ends a step inside the friction
  cone keeps its two anchor points — static friction with positional memory, so stacks do not creep —
  but only while the remembered material point is still the touching one: a rolling body brings a new
  point into contact every step, and a stale anchor would let it sink and pump energy into the roll.
- **Joints** are rows `C(x)` evaluated nonlinearly every iteration, with Jacobians from the
  small-angle expansion around the satisfied state, stabilised like contacts (`C − αC₀`). A ball joint
  is 3 rows, a hinge 3 + 2 (the axes stay parallel) plus a limit or motor row on the hinge angle, a
  slider 2 + 3 plus a limit/motor row along the axis, a weld 6, a distance joint 1. Limits are
  one-sided rows active near the nearer bound; a motor is a row `C = (s − s₀) − v·Δt` whose force is
  clamped to `±maxForce`. Distance joints add AVBD's lumped geometric stiffness `|f|/L` on the
  positional block (without it a slack rope bridge diverged), angular rows `|f|·|J_a|`.
- **Mass floors.** A stiffness ramp proportional to `m/Δt²` grows too slowly for a 2.9 g chain link
  carrying a 1 kg box (347:1). Joint stiffness is scaled as if each body weighed at least
  `jointMassFloor` (1 mass unit) — the effect of AVBD's absolute β for small bodies — which brought the
  chain's sag from 145 mm to 4.7 mm. The contact equivalent (`contactMassFloor`) is off by default: it launches light
  bodies in large piles.
- **Rotations** use the exact exponential map to integrate (`q ← exp(½ ω Δt) q`) and the exact
  logarithm to recover the angular velocity. The usual small-angle forms lose a fraction ≈ φ²/8 of the
  spin per step (φ = |ω|Δt) — a ball of 10 cm radius rolling at 2 m/s lost 58% of its speed in 10 s on
  flat ground; now it keeps 2.000 m/s, and a free body spinning at 20 rad/s keeps 20.000.
- **Adaptive initial guess** (VBD). Iterations start at `x₀ + vΔt + a gΔt²` with
  `a = clamp((v − v_prev)·ĝ / (|g| Δt), 0, 1)`: a falling body starts at the full prediction, a
  resting body starts at rest. Starting resting bodies a full gravity step inside their supports made
  the first iterations overshoot (6× the weight), left the multipliers near zero at the end of every
  step and let a 20-row pyramid creep 73 mm in 8 s; with the adaptive guess it settles once by 1.4 mm.
- **The 6×6 solve** uses 3×3 blocks — Cholesky of the linear block and of the Schur complement
  `D − Bᵀ A⁻¹ B` — entirely in registers. Eight lanes share a body's contacts and reduce their partial
  Hessians through shared memory, so even a colour with a few dozen bodies occupies the GPU.

**GPU pipeline per step** — every stage is order-independent by construction:

1. *Predict* (commands, impulses, force fields, the adaptive guess) and *joint prep*.
2. *Broadphase.* Bounding spheres go into a counting-sort grid sized for ordinary bodies; true size
   outliers (static walls, a boulder — bodies more than twice the size of the 98th percentile) go to
   a short list every body tests directly. One walk over each body's 27 neighbouring cells collects,
   with a bounded sorted insertion, the pairs it *owns* (partners with a larger index) and, for
   dynamic bodies, its lower-index *owners*. A prefix sum gives every owner its slice of one pair
   array, and a binary search in each owner's sorted slice turns the owners into pair indices.
3. *Narrowphase*, one thread per pair, twice: once to count contact points, then — after a prefix sum
   allocates exactly — to write them (box–box SAT over 15 axes with a bias towards face axes,
   Sutherland–Hodgman clipping, reduction to ≤ 4 points; capsule–capsule, capsule–box, sphere and
   floor cases). A large dynamic body's reverse list is built afterwards from the pairs that actually
   touch it, by one workgroup per large body with a workgroup prefix sum (up to 1,024 pairs; a block
   resting on 117 boxes needs that).
4. *Colouring*: Jones–Plassmann with first fit over contact and joint edges, double-buffered so each
   round reads only the previous round's colours.
5. `iterations` × (one dispatch per colour + a contact dual + a joint dual), then the velocity
   update. Every kernel loops grid-stride over a count it reads on the GPU, so dispatch sizes come
   from counts read back asynchronously and only affect speed — no indirect dispatches, which cost
   up to 300 µs each on Vulkan. Only as many colours as recently needed are dispatched (deterministic
   worlds always dispatch the full budget, so the work does not depend on a read-back). Worlds of up
   to `fusedSolve` bodies (128) run the whole solve as **one dispatch**: a single workgroup loops over
   iterations and colours with barriers instead of dispatch boundaries — the same functions in the
   same order, deterministic, equal to the multi-dispatch solve up to rounding (one ulp per step).

All atomics are integer and only count; nothing a later stage reads depends on which thread ran
first. That is what makes `deterministic` hold and the `determinism` law check out.

**Measured** (M2 Max, 60 Hz, 8 iterations): a tilted box lands flat and drifts 0.003 mm in 2 s; a
10-box tower moves 0.24 mm and a 20-row pyramid 1.3 mm over 6 s (0.02 / 0.3 mm with 16 iterations);
a box sticks on a 20° slope and slides down a 40° one at 1.801 m/s² (Coulomb: 1.797); a pendulum's
period is within 0.23% of 2π√(I/mgL); balls and capsules roll 10 s at constant speed, 0.55 mm deep;
a pile of 6,004 mixed boxes and spheres (28k contacts) steps in **3.6 ms** on an M2 Max (3.7 ms
deterministic) and **3.1 ms** on a DGX Spark's GB10 (Chrome, Vulkan); a 38-body two-player world steps
in 1.2 ms / 0.7–0.9 ms.

**Known limit:** a heavy body on much lighter ones sinks into them at 8 iterations (100:1 about
1 cm, 1000:1 about 3.5 cm); substeps reduce it (4 substeps: 4.9 / 11 mm).

### 4.7 Fluid ↔ rigid coupling — `sim.couple(world, opts)`

```js
const sim = gpu.mpm({ grid: 96, size: [3, 1.2, 1.5], substeps: 8 });
const world = gpu.rigid({ bounds: [[-1, -1, -1], [4, 4, 3]] });
sim.couple(world, { friction: 0.2 });          // then step both every frame: sim.step(); world.step(dt);
```

Bodies are moving boundaries of the MPM grid. During every MPM substep each grid node inside a body
or within `band` cells of it (signed distance from the box / sphere / capsule SDF) has its velocity
relative to the body projected: the inward normal part is removed, `v_rel ← v_rel − min(0, v_rel·n) n`
(the fluid may leave the surface but not enter it), and Coulomb `friction` acts on the tangential
part. The momentum `m Δv` that adds to the node is taken from the body (Newton's third law):
`J = −Σ m Δv`, `L = −Σ r × m Δv`. Nodes accumulate that
momentum and its moment about each body's centre in shared-memory fixed point (2¹⁴ for momentum,
2¹⁰ for angular momentum), then in global integer atomics — order-independent — and one pass converts
lattice units to world impulses (`p · dx⁴ / Δt_sub`) and adds them to the body's next-step impulse.
Floating, sinking, drag and splashes follow; nothing is scripted. Bodies whose `group & mask` is zero
are ignored by the fluid (the example's invisible tank walls).

**Measured** at a 3.1 cm grid: boards of density 0.25 / 0.5 / 0.75 float with 0.20 / 0.52 / 0.87
of their height submerged (within about one grid cell of Archimedes), and a ball three times denser
than water sinks to the bottom. Cubes of density between 0.21 and 0.79 roll onto an edge, as real
ones do: upright, their metacentric height is negative.

---

## 5. Neural networks — `SI.nn`

### 5.1 Matrix multiply — `SI.nn.gemm(gpu, A, B, C, opts)`

`C = act(α op(A) op(B) + β C + bias)`, f32, row-major, with `{ M, N, K, transA, transB, alpha, beta,
bias, act, lda, ldb, ldc }`. Each workgroup computes a `BM × BN` tile of C; K is walked in slices of
`BK`, staged through shared memory as whole `vec4`s (one thread writes a whole vector — writing the
components of a shared `vec4` from different threads is a read-modify-write race), and every thread
accumulates a `TM × TN` block of outputs in scalar registers: `TM·TN` FMAs for `TM + TN` shared-memory
reads. Transposes are folded into the loads, so back-propagation needs no transpose kernel; bias and
activation are fused into the epilogue. `await SI.nn.autotuneGemm(gpu)` times six tile shapes on the
device and remembers the fastest (per adapter, in `localStorage`).

When the output is small and K long (a weight gradient `Xᵀ·dZ` over a large batch) only a few
workgroups would run, so K is split into chunks whose partial tiles are summed in a fixed order —
deterministic, and for a weight gradient over a 65,536-row batch 10.8× faster at 128 × 128
(9.3 → 0.86 ms) and 32× at 64 × 64.

**Certificate.** `SI.verify.freivalds(gpu, A, B, C, { M, N, K, trials })` checks a product in O(n²):
for random ±1 vectors r it compares `A(Br)` with `Cr`; a wrong C survives a trial with probability
≤ ½, so 8 trials bound a false accept by 2⁻⁸, and one corrupted entry of 10⁻⁴ of its row's norm is
always caught. The vectors are fresh random bits on every call (`seed` makes them reproducible, at
the price of the bound against adversarial inputs). The benchmark suite runs it before timing.

**Measured** (M2 Max, 2048³): 3.0 TFLOP/s, 2.1× the classic 16×16 shared-memory tiled kernel and
7.2× the naive one, bit-identical to both (same summation order). WebGPU exposes no matrix units
(no `simdgroup_matrix` / tensor cores), so this is plain FMA throughput.

### 5.2 Networks, losses, optimisers

```js
const net  = new SI.nn.MLP(gpu, { layers: [2, 128, 128, 3], act: 'relu', out: 'sigmoid', fourier: 8, maxBatch: 65536 });
const loss = new SI.nn.MSELoss(gpu, batch * 3);
const opt  = new SI.nn.Adam(gpu, net.params, { lr: 2e-3 });
net.input.write(x);                                   // or let a kernel fill net.input
net.forward(net.input, batch);                        // → net.output  [batch × 3]
net.backward(loss.run(net.output, targets, batch * 3), batch);   // fills every layer's gradients
opt.step();
await loss.readLoss();                                // only when you want to look
```

`Dense` layers fuse bias and activation (`none relu leaky tanh sigmoid sin gelu`) into the GEMM
epilogue (`sin` and `gelu` keep their pre-activation for the backward pass; SIREN initialisation
folds ω₀ into the weights); `backward` computes `dZ = dY ⊙ f′(Z)` from the stored outputs, `dW = Xᵀ dZ`, `db = Σ dZ`
(a per-column workgroup reduction) and `dX = dZ Wᵀ`. `fourier: B` prepends Fourier features
`[x, sin(2ᵏπx), cos(2ᵏπx)]` for `k < B` (Tancik et al. 2020), which lets a small MLP represent high
frequencies. `Adam` (with weight decay and gradient clipping) and `SGD` (momentum) update in place.
`SI.verify.checkGradient` compares backward against central finite differences (worst relative
error 2 × 10⁻⁴ in f32); an MLP fits a regression from loss 0.53 to 9 × 10⁻⁴ in 300 Adam steps, and the
neural-field example trains on and redraws a 256² image every frame in 13.8 ms.

### 5.3 Many agents at once — `SI.nn.Population`

```js
const pop = new SI.nn.Population(gpu, { layers: [8, 32, 32, 2], count: 4096, sigma: 0.05, lr: 0.03 });
pop.forward(observations, actions);        // every agent's own network in one dispatch
const { mean, best } = await pop.update(fitness);   // evolution strategies step
```

A population shares one parameter vector θ; agent `2a` runs `θ + σε_a` and agent `2a + 1` runs
`θ − σε_a` (antithetic sampling), with ε regenerated from a hash of (generation, pair, index) instead
of being stored — 4,096 agents cost one θ. `update` ranks the fitness, centres the ranks in
`[−½, ½]` (robust to outliers) and applies `θ ← θ + lr/(Pσ) Σ_a (w_2a − w_2a+1) ε_a` (OpenAI-style
evolution strategies) — gradient-free, so the reward can come from anything, including a physics
step. Observations and actions are GPU arrays, so a controller can read a world's state and write
forces (through a force field) without leaving the GPU.

## 6. Peer-to-peer games — `SI.net`

The same inputs give the same world (deterministic mode, §4.6), so peers only need to exchange
inputs.

```js
const world = gpu.rigid({ deterministic: true });
// ... build the level, identically on every peer
const rtc = new SI.net.RTCPeer(new SI.net.BroadcastSignaler('room-1'), { initiator: isHost });
await rtc.ready;
const session = new SI.net.RollbackSession({
  sim: world, game, channel: rtc.channels, players: 2, localPlayer: isHost ? 0 : 1,
  applyInputs: (tick, inputs) => { /* turn each player's input bytes into world commands */ },
  delay: 2, maxRollback: 8, hashEvery: 30,
});
// at a fixed 60 Hz: session.advance(new Uint8Array([buttons]));
```

**Rollback** (in the style of GGPO). A player's input for tick t is applied at t + `delay`. Packets go
over an unreliable, unordered channel and carry the last `redundancy` (8) ticks of input, so loss
costs nothing until a whole window is lost. When a remote input has not arrived, the session predicts
it (the player's last input) and keeps simulating; when the real input differs from the prediction,
it restores the GPU snapshot taken before that tick and re-simulates to the present within the same
frame. If a peer falls more than `maxRollback` ticks behind, the other waits (`stats.stalls`).
Game state that lives outside the world (scores, cooldowns) goes through `game.save()` /
`game.load()` and is snapshotted alongside it.

**Desync detection and resync.** Every `hashEvery` confirmed ticks, each peer hashes the snapshot of
that tick on the GPU (no stall) and sends the hash over the reliable channel; a mismatch calls
`onDesync`. Peers on different GPUs or browsers can differ in the last bits of floating point, so the
session repairs itself: the `authority` (player 0) exports the state of its newest fully confirmed
tick from its rollback ring — a private GPU copy is taken first, because the ring moves on while the
read-back is in flight — and sends it with the game state; the other peer imports it at that tick and
re-simulates to the present with the inputs it already has. Every adopted state starts a new *epoch*;
hashes carry their epoch, and those from an abandoned history are ignored, so one divergence costs
one transfer (in a duel of about 40 bodies: one state, and 132 KB on the wire for the whole 7-second match). Set
`resync: false` or a `stateHandler` to use your own policy. A peer keeps `inputHistory` (600) ticks of
inputs so a late state can still be replayed, and asks for a newer one if it cannot.

**Trust.** A peer's packets are validated before they touch the session (unknown players, malformed
inputs and hashes are dropped and counted in `stats.messagesRejected`). States are adopted only by
non-authorities, only from the authority and only when newer than the state held, so a peer cannot
overwrite the authoritative world or move its epoch back; resync requests are honoured at most once
a second. Game state crosses the wire in the same encoding as inputs, which carries NaN, ±Infinity,
-0, `undefined`, BigInt, Map, Set, Date, ArrayBuffer and typed arrays exactly — the same values
`structuredClone` keeps for rollbacks — and `decode` constructs nothing but those types.

**Backpressure.** Each tick's GPU work is fenced; when more than `maxInFlight` (3) ticks are still
executing, `advance()` waits a tick (`stats.gpuWaits`) instead of letting the queue — and with it the
latency of every hash and state read-back — grow without bound.

**Transcripts — proof-carrying matches.** The session folds every confirmed tick's inputs into a hash
chain and records the state hashes. `session.transcript()` is a certificate of the match:
`SI.net.verifyTranscript(tr, makeWorld, applyInputs)` replays it from the start and checks every
state hash and the chain head; changing one input bit anywhere is detected.

**Transports.** `RTCPeer` (an unreliable and a reliable data channel), signalled through
`BroadcastSignaler` (tabs of one browser) or `WebSocketSignaler(url, room)` (any relay that forwards
JSON); `SI.net.dataChannel(dc)` wraps a data channel you created yourself. On the wire a message is
JSON in which typed arrays travel as base64 of their bytes, so bits arrive exactly (plain JSON numbers
would turn the u32 collision masks stored in body records — NaN bit patterns as floats — into `null`),
and messages over 64 KB are split into parts. `LoopbackNetwork({ latency, jitter, loss, duplicate })`
simulates a network in one page and sends what WebRTC would (the same encoding) — the tests run two
peers over 40 ± 15 ms with 10% loss and check that every exchanged state hash agrees.

## 7. Verification — `SI.verify`, `LAWS.md`

What ShaderIntel promises is written down in [LAWS.md](../LAWS.md), and every line is executable:
`await SI.verify.proveAll(gpu, SI)` runs each law's checker and returns `{ name, statement, ok,
evidence }`. The idea comes from proof-carrying code and proof assistants (Lean's small trusted
kernel; Bend 2's `LAWS.bend` and `--verdict`): the optimised code is not trusted, a small checker that
shares no code with it is. Checkers are exact (scan, sortedness, permutation, graph colouring,
determinism, replay), probabilistic with a stated bound (Freivalds for GEMM), or physical with a
tolerance (contact sign and friction cone, penetration, MPM conservation, gradient vs finite
differences) — each law says which.

Use the checkers on your own kernels: `checkScan`, `checkSorted`, `checkPermutation` (a fingerprint:
probabilistic), `freivalds`, `checkColoring(world)`, `checkContacts(world)`, `checkMPM(sim)`,
`checkGradient(net, X, T, batch)` (every weight and bias tensor), `checkDeterminism(makeWorld)`,
`checkReplay(world)`.

**Binding contracts.** Every storage binding knows the WGSL type it is declared with — structs
included, parsed from the linked shader with WGSL's layout rules — and `bind()` / `run()` reject a
buffer that is not a whole number of elements, naming the kernel, the binding, the byte size and the
stride. A record written from JavaScript with one field too many (the bug that once shifted every
joint after the first by 16 bytes) is now an error at the call site.

## 8. three.js — `SI.three`

ShaderIntel can simulate while three.js (r180+, `three/webgpu`) renders, on one `GPUDevice`:

```js
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
const renderer = await SI.three.renderer(THREE, gpu, { canvas, antialias: true });  // shares gpu.device
const bodies = SI.three.bodies({ THREE, TSL, renderer, world, castShadow: true });
scene.add(bodies.object);
// every frame: world.step(dt); bodies.update(); gpu.flush(); renderer.render(scene, camera);
```

`gpu.flush()` matters: ShaderIntel batches its work until a flush, and three.js submits its own, so
without it three.js draws stale poses. `bodies.dispose()` / `particles.dispose()` remove the meshes
and free their geometries. To use three.js's device instead, `SI.init({ device })`.

`bodies.update()` runs one kernel that copies poses from the world's body buffer into three.js
storage attributes (GPU → GPU, no read-back); TSL position and normal nodes place the instanced box,
sphere and capsule meshes, so three.js lights, shadows and materials apply. `SI.three.particles`
does the same for any GPU array of points (MPM particles, boids), and `SI.three.bufferOf(renderer,
attribute)` exposes the `GPUBuffer` behind a three.js attribute for your own kernels.
`examples/three.html` loads three.js r180 from a CDN (the library itself has no dependencies) and is
checked headless with a screenshot (`node tools/headless.mjs shot --url …/examples/three.html`).

## 9. Shader library reference

Import modules with `imports: ['noise', ...]` (dependencies follow automatically).

| Module | Functions |
|---|---|
| `math` | `PI TAU INV_PI`, `sq max3 min3 dot2 dot3 remap smootherstep safeNormalize`, `rot2 rotAxis basis`, quaternions `qmul qconj qrot qaxis qmat` |
| `hash` | `pcg pcg2d pcg3d pcg4d lowbias32`, `u2f u2f2 u2f3 u2f4`, `hash11 hash21 hash22 hash31 hash33 hash44 ihash2 ihash3`, RNG `rngSeed randU32 rand rand2 rand3 randDir randBall randNormal2 randCosHemi`, `r2seq ign` |
| `noise` | `simplex2 simplex2d simplex3 simplex3d simplex4`, `perlin3 perlin3d value3`, `worley2 worley3`, `fbm2 fbm3 fbm3d ridged3 turbulence3`, `curl2 curl3` (the `…d` variants return `(value, gradient)`) |
| `sdf` | 3D `sdSphere sdBox sdRoundBox sdBoxFrame sdTorus sdCapsule sdCylinder sdCappedCone sdPlane sdOctahedron sdEllipsoid sdHexPrism sdLink sdInfCylinder`; 2D `sdCircle sdBox2 sdRoundBox2 sdSegment2 sdEquilateralTriangle sdHexagon`; ops `opUnion opSubtract opIntersect smin smax sminCubic sminMix opSmoothUnion opSmoothSubtract opSmoothIntersect opRound opOnion opU`; domain `opRepeat opRepeatLimited opTwist opBend opElongate opSymX opRevolve opExtrude` |
| `raymarch` | needs `fn map(p: vec3f) -> vec2f`: `coneMarch raymarchBasic raymarch calcNormal softShadow calcAO`, `struct Hit` |
| `pbr` | `D_GGX V_SmithGGX F_Schlick brdfDirect envBRDF skyColor shadePBR` |
| `color` | `srgbToLinear linearToSrgb luminance linearToOklab oklabToLinear oklabToOklch oklchToOklab mixOklab hsv2rgb rgb2hsv palette turbo viridis inferno magma` |
| `tonemap` | `tonemapACES tonemapACESHill tonemapAgX tonemapNeutral tonemapReinhard encodeDisplay` |
| `linalg` | `I3 outer3 trace3 diag3 eigSym3 svd3 polarR` |
| `grid` | `gridCoord gridIndex` |

---

## 10. First-principles design notes

1. **Bandwidth and latency, not FLOPs, bound most GPU physics.** Hence SoA/AoS choices per access
   pattern, coalesced loads through shared memory, sorting particles for cache locality, fewer passes
   per frame, and one compute pass per batch.
2. **Make the right thing the easy thing.** Kernels only contain the math; the SDK computes WGSL
   struct layouts, binding declarations, dispatch geometry and per-dispatch uniforms, which are where
   most WebGPU bugs live.
3. **Choose units in which the numbers are O(1).** Lattice units make MPM stability conditions
   explicit, bound every quantity for fixed-point atomics and make "stiffness" a fraction of what is
   stable.
4. **Exploit coherence.** Neighbouring pixel rays share empty space (cone marching); neighbouring
   particles share grid nodes (tiled P2G); cells are rows of contiguous particles (9-range neighbour
   loops).
5. **Discretise so the identities hold.** Staggered grids make `div∘grad = Laplacian`; APIC with
   B-splines conserves momentum exactly; integer accumulation makes results order-independent.
6. **Measure, verify, keep the honest result.** Every benchmark checks agreement before timing, and
   techniques that lost on real hardware are documented instead of shipped as defaults.
7. **Determinism by construction.** Results never depend on thread timing: sorted, bounded lists
   instead of append-with-atomics, prefix sums to allocate, double buffers for iterative passes,
   integer-only atomics. Rollback netcode, replays and the `determinism` law follow for free.
8. **Ship the checker with the code.** Every promise has an executable checker that is smaller than,
   and independent of, the code it judges (LAWS.md); types of GPU bindings are checked at the call.

---

## 11. React

`shaderintel-react` wraps all of this in components and hooks — `<Canvas>`, `useFrame`,
`useArray`, `useKernel`, `<Physics>` with `<Box>`, `<Sphere>`, `<Joint>`, `<Fluid>`, `<Cloth>`,
`<Smoke>`, `<Boids>`, `<NBody>`, `<SdfScene>` and more — with GPU memory tied to component
lifetimes. See [packages/react/README.md](../packages/react/README.md).
