# Changelog

## 0.3.0

The production release: a lifecycle for everything, about fifty fixes found by review and confirmed
on the GPU, a React package, a website with every example, and an MIT license.

### Added

- **`shaderintel-react`** (`packages/react`): `<Canvas>`, hooks (`useFrame`, `useArray`, `useKernel`,
  `useLayer`, `useResource`, `useGPU`, `useView`), layers (`<Stage>`, `<Points>`, `<Spheres>`,
  `<Glyphs>`, `<Fullscreen>`, `<SdfScene>`, `<BoxLines>`, `<Layer>`), rigid bodies (`<Physics>`,
  `<Box>`, `<Sphere>`, `<Capsule>`, `<RigidBody>`, `<Joint>`, `<ForceField>`) and simulations
  (`<Fluid>` + `<FluidBlock>`, `<Cloth>`, `<Smoke>`, `<Boids>`, `<NBody>`). StrictMode, fast refresh
  and server rendering are handled; canvases share one device.
- **Website**: a home page with a live, interactive simulation; an example browser with a
  searchable list, screenshots and source view; a docs viewer for the guide, React docs, laws,
  learnings and this changelog. Five React examples.
- `destroy()` on every simulation, layer, primitive and network, idempotent and safe in the
  middle of a batch; `View.destroy()` removes its listeners, observer, HUD and textures.
- `View`: `run()` resumes after `stop()`, `running`, `onFrame`, `onError`, `setBloom()`, `setHud()`,
  `setPixelRatio()`, device-pixel-ratio tracking, a size cap at the texture limit.
- `OrbitCamera`: `set()`, `detach()`, `enabled`, `minPitch` / `maxPitch`, rotate / zoom / pan speeds.
- `init({ device })` shares a device created elsewhere (three.js); `isSupported()`.
- `SIError` carries a `code` (`unsupported`, `no-adapter`, `no-device`, `compile`, `invalid-argument`).
- `Context.writeOrdered()` / `GArray.writeOrdered()`: uploads ordered with recorded work.
- `MPM`: `decouple()`, `MPM_KINDS`, material kinds by name, `substeps` and `timeScale` that can
  change at run time.
- `three.bodies()` / `three.particles()`: `dispose()`.
- CPU unit tests (`npm run test:unit`, run by CI), a React test suite on the GPU, robustness and
  regression tests (the browser suite grew from 114 to 138 tests, plus 13 React GPU tests and 17 CPU unit tests), a thumbnail generator, CI and Pages
  workflows.

### Fixed

Core and rendering

- A uniform-ring overflow flushed inside an open render pass (a frame with thousands of draws
  failed validation); the ring now grows by segments.
- `Reduce.run(src, 0)` looped forever; `n > maxN` silently returned wrong sums (now throws, as
  `Scan` and `SpatialGrid` do).
- Subgroup paths assumed subgroups of 32: `Reduce` and `Scan` were wrong on devices with 8-wide
  subgroups.
- A bad shader logged three errors; now one, with source lines.
- Screen-space water wrote NaN normals at the screen border, which bloom turned into white
  blotches; bloom now also drops NaN and infinity.
- `Stage` hid layers drawn before it; it is order-independent now.
- `SdfScene.set({ maxDistance })`, `Volume.set()`, `Bodies.set({ groups })` and
  `FluidSurface.set({ iterations })` used names the layer did not understand; `set()` translates
  them and rejects unknown names immediately instead of mid-frame.
- `GArray.write()` silently truncated data that did not fit.
- `SurfaceMesh` rejected `GArray` index buffers; `Lines` with no points threw; `bloom: { levels: 0 }`
  crashed.
- Destroying a buffer that recorded work still used made the whole submit fail; `step()` after
  `destroy()` poisoned the batch; layers kept drawing freed buffers.

Physics

- `RigidWorld` needs 10 storage buffers per shader stage and silently dropped every frame on
  devices with fewer; it now throws a clear error. The fused solve fell back on devices with 16 KB
  of workgroup memory instead of failing.
- Contact events were lost with `substeps > 1`.
- `step(0)` turned rigid worlds and cloth into NaN.
- Commands with an invalid body id (e.g. `-1` from a full world) moved another body; they throw now.
- `remove(id)` twice gave one slot to two bodies; `raycast` on an empty world reported hits;
  `addJoint` with a missing body crashed; a manifold straddling `maxContacts` left stale contacts
  that were still read; the replaced broadphase grid leaked.
- `setMotor(j, speed)` without a force wrote a zero force limit; it keeps the joint's limit now.
- CPU uploads overtook recorded steps: several steps in one batch all saw the last colliders, and
  `NBody.galaxies()` after a recorded step was lost.
- `NBody` with `softening: 0`, MPM materials with `poisson: 0.5`, densities in kg/m³ and very high
  viscosity produced NaN; they are clamped or rejected with a message.
- Changing MPM `substeps` at run time changed every particle's speed.
- `MPM.clear()` kept the old materials, so switching scenes hit the 8-material limit.
- A 1-particle-wide cloth was NaN; cloth floor friction above 1 pushed particles backwards.
- `Boids` missed neighbours after its radius changed.
- `NBody.autotune()` advanced the simulation it was timing, and reused a cached kernel that the
  device could not run.

Networking and verification

- `RollbackSession` adopted state from any peer, including one rewriting the authority's world or
  moving the epoch back; malformed packets could throw. States are now accepted only from the
  authority and only when newer; packets are validated; resync requests are rate-limited.
- A late input after a resync rolled back into the abandoned history; hashes queued before the
  resync described it.
- `decode()` would construct any global named by a remote peer; it constructs typed arrays only.
- Game state lost NaN, ±Infinity, -0, `undefined`, BigInt, Map, Set and Date in a resync; the wire
  encoding carries them now.
- The simulated reliable channel reordered messages less than a millisecond apart.
- `RTCPeer.ready` never settled on failure or `close()`; a lost offer or answer hung the
  connection; an unreachable relay raised an unhandled rejection every 500 ms.
- Long messages could exceed 64 KB per part; parts lost on an unreliable channel were kept forever.
- Freivalds' check reused the same random vectors every call (a product with swapped columns
  passed); it draws fresh ones now.
- The backprop law never checked bias gradients; the colouring law counted disabled joints; every
  law leaked its scene.

Neural networks and tooling

- `MLP` accepted `sin` and `gelu` but could not train them; SIREN initialisation assumed a factor
  the forward pass never applied.
- `Population` silently rounded an odd count up and read past the caller's buffers; seed 0 gave
  constant weights.
- The bundler mis-bundled multi-declaration exports and imports with trailing comments; it now
  fails the build on anything it does not understand. Build tools failed in paths with spaces.
- The dev server's path guard let through sibling folders that share the repository's name prefix.
- The examples: the neural-field reset kept the old loss and lost the learning rate; the three.js
  page fired a ball at the end of every orbit drag and stayed blank when the CDN was unreachable.

## 0.2.0

The first version in this repository: the SDK, primitives, physics, rendering, neural networks,
netcode, verification laws, examples, benchmarks and the browser test suite.
