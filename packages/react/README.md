# shaderintel-react

React components and hooks for [ShaderIntel](../../README.md): GPU rigid bodies and joints, MLS-MPM
fluids, cloth, smoke, flocks, N-body gravity, particles and WGSL shaders as declarative components,
plus hooks for writing your own compute kernels. Every component owns its GPU memory and frees it
when it unmounts; StrictMode, fast refresh and server rendering are handled.

```bash
npm install shaderintel shaderintel-react
```

React 18 or 19. The browser needs WebGPU (Chrome and Edge 113+, Safari 26+, Firefox 141+ on Windows),
served over https or from localhost.

## A first scene

```jsx
import { Canvas, Stage, Physics, Box, Sphere } from 'shaderintel-react';

export function App() {
  return (
    <div style={{ height: '100vh' }}>
      <Canvas camera={{ target: [0, 1, 0], distance: 8 }} hud>
        <Stage />
        <Physics>
          <Box position={[0, 4, 0]} size={[1, 1, 1]} />
          <Sphere position={[0.2, 7, 0]} radius={0.5} />
        </Physics>
      </Canvas>
    </div>
  );
}
```

`<Canvas>` fills its parent, so give the parent a size. Drag to orbit, right-drag or shift-drag to
pan, scroll or pinch to zoom.

## `<Canvas>`

Creates the GPU context and a view (HDR target, orbit camera, tonemapping, optional bloom and stats
overlay), runs the frame loop, and renders its children once the GPU is ready. DOM children are
rendered inside the canvas's wrapper, so control panels and labels can be positioned over it.

| Prop | |
|---|---|
| `camera` | `{ target, distance, yaw, pitch, fov, near, far, autoRotate, minDistance, maxDistance, ... }`, applied again whenever its values change |
| `controls` | orbit controls (default `true`) |
| `bloom` | `true`, `false` or `{ strength }` |
| `hud` | FPS and per-pass GPU times |
| `tonemap`, `exposure`, `background` | `'agx'` (default), `'aces'`, `'neutral'`, `'reinhard'`, `'none'`; linear RGBA clear colour |
| `pixelRatio`, `maxPixelRatio` | fixed device-pixel ratio, or the screen's capped at `maxPixelRatio` (2) |
| `paused` | stop the frame loop |
| `frameloop` | `'always'` (default) or `'never'`: render only when you call `advance()` |
| `fallback` | shown when WebGPU is missing or fails: a node, or `(error) => node` |
| `loading` | shown while the GPU initialises |
| `onCreated(root)`, `onError(error)` | the root is `{ gpu, view, canvas, advance }` |
| `init` | options for `SI.init()`, e.g. `{ powerPreference: 'low-power' }` or an external `{ device }` |
| `gpu` | use an existing ShaderIntel context instead |

Canvases with the same `init` options share one GPU device. It is released a moment after the
last of them unmounts, so a quick remount (StrictMode, route changes, fast refresh) reuses it.

Without WebGPU the default fallback says so and names the browsers that have it; pass `fallback`
to show your own (a screenshot or a video works well).

## Hooks

All hooks must be used inside `<Canvas>`.

- `useFrame(callback, priority = 0)`: runs `callback({ dt, time, frame, view, gpu })` before every
  frame. Lower priorities run first.
- `useGPU()`, `useView()`, `useShaderIntel()`: the GPU context, the view (camera, pointer, `ray()`),
  and the root state.
- `useArray(type, lengthOrData, opts?)`: a GPU array owned by the component. New data of the same
  length is uploaded; a different length re-creates the array.
- `useKernel(desc, deps?)`: a compute kernel, compiled again when `desc.code` or `deps` change.
- `useLayer(create, deps?)`: adds `create(gpu, view)` to the view, and removes and destroys it on
  unmount.
- `useResource(create, deps?, dispose?)`: any object tied to the component's life (destroyed with
  `destroy()` by default).

Your own GPU compute in a component:

```jsx
import { useLayoutEffect } from 'react';
import { Canvas, Points, useArray, useKernel, useFrame } from 'shaderintel-react';

function Flow({ count = 1_000_000 }) {
  const pos = useArray('vec4f', count);
  const seed = useKernel({ imports: ['hash'], buffers: { pos },
    code: 'var s = rngSeed(i, 1u); pos[i] = vec4f(randBall(&s), 1.0);' }, [pos]);
  const flow = useKernel({ imports: ['noise'], buffers: { pos }, uniforms: { dt: 'f32' },
    code: 'pos[i] += vec4f(curl3(pos[i].xyz) * u.dt * 0.1, 0.0);' }, [pos]);
  useLayoutEffect(() => { seed.run(count); }, [seed, count]);
  useFrame(({ dt }) => flow.run(count, { dt }));
  return <Points data={pos} size={0.002} colorBy="w" />;
}
```

## Render layers

Props compiled into a shader (code, imports, uniform declarations, data layout) re-create the layer
when they change; everything else is a uniform update, applied without recompiling. Each
component's `ref` is the ShaderIntel layer.

- `<Stage height cell fog sun ... />`: sky gradient and an infinite grid floor.
- `<Points data size pixelSize color colorBy intensity count ... />`: additive glowing sprites.
- `<Spheres data radius palette colorBy material mask count ... />`: ray-cast sphere impostors.
- `<Glyphs pos vel size colorBy />`: velocity-aligned darts for agents.
- `<Fullscreen code imports uniforms values writeDepth />`: a Shadertoy-style shader;
  `code` defines `fn shade(uv: vec2f, px: vec2f) -> vec4f`.
- `<SdfScene code imports uniforms values shadows ao maxDistance sun fog />`: write
  `fn map(p: vec3f) -> vec2f` and get cone marching, soft shadows, AO and PBR.
- `<BoxLines min max color />`: the outline of a box.
- `<Layer create={(gpu, view) => layer} deps={[...]} />`: anything else.

```jsx
<SdfScene
  uniforms={{ r: 'f32' }} values={{ r: radius }}
  code={`fn map(p: vec3f) -> vec2f {
    return opU(vec2f(p.y, 0.0), vec2f(sdSphere(p - vec3f(0.0, 1.0, 0.0), u.r), 1.0));
  }`} />
```

## Rigid bodies

`<Physics>` owns a RigidWorld (Augmented Vertex Block Descent on the GPU), steps it at a fixed
rate (`timeStep`, default 1/60 s, at most `maxSteps` per frame) and draws its bodies (`render`,
default true). Capacity options (`maxBodies`, `bounds`, `deterministic`, `events`, ...) re-create
the world; solver options (`gravity`, `iterations`, `substeps`, `floor`, ...) apply in place. Its
ref is the RigidWorld, and `onStep(world)` runs after every step.

Bodies are added when they mount and removed when they unmount:

```jsx
const ball = useRef();
<Physics gravity={[0, -9.81, 0]}>
  <Box position={[0, 2, 0]} size={[1, 0.5, 1]} density={2} friction={0.8} />
  <Sphere ref={ball} position={[0, 6, 0]} radius={0.3} />
  <Capsule position={[1, 3, 0]} radius={0.1} halfHeight={0.3} />
  <Box type="static" position={[0, -0.5, 0]} size={[20, 1, 20]} />
  <Box type="kinematic" position={[x, 1, 0]} size={[2, 0.2, 2]} />
</Physics>
// later: ball.current.applyImpulse([0, 5, 0]);
```

- Body props: `position`, `rotation` (Euler `[rx, ry, rz]` or quaternion `[x, y, z, w]`),
  `velocity`, `angularVelocity`, `density`, `friction`, `type` (`'dynamic'`, `'static'`,
  `'kinematic'`), `sensor`, `fixedRotation`, `gravityScale`, `linearDamping`, `angularDamping`,
  `group`, `mask`, `onAdded(id)`.
- `position`, `rotation` and the velocities are **initial** values: the simulation owns the pose
  afterwards. Kinematic bodies follow their `position` and `rotation` props and push dynamic ones.
- The ref API: `id`, `applyImpulse(J, point?)`, `applyForce(F, point?)`, `applyAngularImpulse(L)`,
  `setVelocity(v, w?)`, `setPose(p, q?)`, `moveKinematic(p, q?)`, `setEnabled(on)`,
  `setFilter(group, mask)`.
- Changing a body's shape, size or density adds a new body in place of the old one.

Joints connect body refs (or `b={null}` for the world), even when they are written before their
bodies:

```jsx
const a = useRef(), b = useRef();
<Joint type="hinge" a={a} b={b} anchor={[0, 2, 0]} axis={[0, 0, 1]}
       limits={[-1, 1]} motor={{ speed: 2, maxForce: 50 }} />
```

Types are `ball`, `hinge`, `fixed`, `distance` and `slider`. `motor`, `limits` and `enabled`
update live; other props are read when the joint is created.

`<ForceField code uniforms values />` runs WGSL for every body before each step (wind, buoyancy,
attractors): it reads the body `b` and adds to `force` and `torque`.

`useWorld()` gives the RigidWorld to any component inside `<Physics>`, for ray casts
(`world.raycast(...)`), events, snapshots and the rest of the
[RigidWorld API](../../docs/GUIDE.md#46-rigid-bodies--gpurigidopts--gpubodiesworld).

## Fluids, cloth, smoke, flocks and gravity

```jsx
<Physics>
  <Fluid grid={64} size={[3, 1.2, 1.5]} couple={{ friction: 0.2 }}>
    <FluidBlock box={[[0.05, 0.05, 0.05], [2.95, 0.45, 1.45]]} material="water" />
    <FluidBlock sphere={[[1.5, 0.9, 0.7], 0.15]} material="jelly" />
  </Fluid>
  <Box position={[1, 0.9, 0.7]} size={[0.2, 0.2, 0.2]} density={0.5} />
</Physics>
```

- `<Fluid>` is an MLS-MPM continuum filled by `<FluidBlock>` children (`box`, `sphere`, `cylinder`
  or `points`, with a `material`: `water`, `oil`, `honey`, `jelly`, `rubber`, `snow`, `sand` or a
  custom definition). Adding, removing or changing a block restarts the fluid. Inside `<Physics>`
  it is coupled both ways with the bodies and steps in lockstep with the world. `render` is
  `'surface'` (default), `'particles'` or `false`.
- `<Cloth width height size origin pin wind colliders substeps stretch shear bend floor />`:
  XPBD cloth.
- `<Smoke res size sources vorticity buoyancy cooling density emission tint />`: smoke and fire
  with volumetric light.
- `<Boids count radius separation alignment cohesion maxSpeed predator />`: a flock.
- `<NBody count G softening dt initial stepsPerFrame autotune />`: direct gravity drawn as glowing
  points.

Each ref is the simulation object, for anything the props don't cover.

## Server rendering

The package can be imported on a server: `<Canvas>` renders a sized `<div>` with a `<canvas>` and
your `loading` node, and creates the GPU context only in the browser. In Next.js, use it from a
client component (`'use client'`).

## Sharing a device with three.js

Pass the device three.js uses, so both libraries work on the same GPU memory:

```jsx
<Canvas init={{ device: renderer.backend.device }}>...</Canvas>
```

A device passed this way is never destroyed by ShaderIntel.

## Examples

Five React examples run in the [example browser](../../examples/#react-physics): a physics
playground, fluid with crates, a live shader editor, compute with hooks, and simulations that
mount and unmount. Their sources are in [`examples/react/src`](../../examples/react/src).

## License

MIT
