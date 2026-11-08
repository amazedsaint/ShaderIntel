// Particle and field simulations as components. Each owns its ShaderIntel simulation, steps it
// every frame and draws it; the ref is the simulation object for anything not covered by props.
import { createContext, forwardRef, useContext, useEffect, useImperativeHandle, useRef } from 'react';
import type { ReactNode } from 'react';
import type {
  Boids as SIBoids, BoidsOptions, Cloth as SICloth, ClothCollider, ClothOptions, CouplingOptions, MaterialDef, MPM, MPMOptions,
  NBody as SINBody, Smoke as SISmoke, SmokeOptions, SmokeSource, Uniforms,
} from 'shaderintel';
import { stableKey, useDisposable, useFrame, useIsomorphicLayoutEffect, useLayer, useResource } from './hooks.js';
import { useShaderIntel } from './context.js';
import { usePhysicsState } from './physics.js';

type V3 = [number, number, number] | number[];

// ------------------------------------------------------------------ MLS-MPM fluids and solids

interface FluidState { sim: MPM; blocks: Map<object, FluidBlockProps>; dirty: boolean }
const FluidContext = createContext<FluidState | null>(null);

export interface FluidProps extends MPMOptions {
  children?: ReactNode;
  /** "surface" (default): a screen-space water surface for liquids plus spheres for solids; "particles": every particle as a sphere; false: draw nothing. */
  render?: 'surface' | 'particles' | false;
  /** Radius multiplier of the particle spheres. */
  particleSize?: number;
  /** Kinematic colliders: { type: 'sphere' | 'box' | 'plane', center, radius | half | normal, velocity, friction }. */
  colliders?: Array<Record<string, unknown>>;
  /** Two-way coupling with the enclosing <Physics> world (default true when inside one). */
  couple?: boolean | CouplingOptions;
  paused?: boolean;
}

/**
 * An MLS-MPM continuum: water, oil, honey, jelly, rubber, snow and sand, filled by <FluidBlock>
 * children. Inside <Physics>, rigid bodies float, sink and splash in it (two-way coupling).
 * Grid options re-create it; changing the blocks restarts it from them.
 */
export const Fluid = forwardRef<MPM, FluidProps>(function Fluid(props, ref) {
  const { children, render = 'surface', particleSize, colliders, couple, paused = false, ...opts } = props;
  const { gpu } = useShaderIntel();
  const physics = usePhysicsState();
  const world = physics?.world ?? null;
  const state = useDisposable<FluidState>(() => ({ sim: gpu.mpm(opts), blocks: new Map(), dirty: true }), (s) => s.sim.destroy(),
    [gpu, stableKey([opts.grid, opts.size, opts.origin, opts.maxParticles, opts.ppcAxis])]);
  const sim = state.sim;
  useImperativeHandle(ref, () => sim, [sim]);
  useIsomorphicLayoutEffect(() => {
    if (opts.substeps !== undefined) sim.substeps = opts.substeps;
    if (opts.timeScale !== undefined) sim.timeScale = opts.timeScale;
    if (opts.gravity) sim.gravity = [...opts.gravity];
    if (opts.sortEvery !== undefined) sim.sortEvery = opts.sortEvery;
  }, [sim, opts.substeps, opts.timeScale, stableKey(opts.gravity ?? null), opts.sortEvery]);
  useIsomorphicLayoutEffect(() => { sim.setColliders(colliders ?? []); }, [sim, stableKey(colliders ?? [])]);

  const rebuild = () => {
    if (!state.dirty) return;
    state.dirty = false;
    sim.clear();
    for (const b of state.blocks.values()) {
      const { material, velocity, jitter, seed, box, sphere, cylinder, points } = b;
      sim.add({ box, sphere, cylinder, points } as Parameters<MPM['add']>[0], { material, velocity, jitter, seed });
    }
  };
  const latestPaused = useRef(paused);
  latestPaused.current = paused;

  // Coupled: step in lockstep with the world, once before every world step (MPM.step() is 1/60 s of
  // simulation; the world's default time step matches). Not coupled: a 1/60 s accumulator of its own.
  const coupled = !!world && couple !== false;
  const coupleKey = stableKey(couple ?? null);
  useEffect(() => {
    if (!coupled || !physics) return;
    const c = sim.couple(physics.world, typeof couple === 'object' ? couple : {});
    const stepFluid = () => { rebuild(); if (!latestPaused.current) sim.step(); };
    physics.beforeStep.add(stepFluid);
    return () => { physics.beforeStep.delete(stepFluid); if (sim.coupling === c) sim.decouple(); };
  }, [sim, physics, coupled, coupleKey]);
  const acc = useRef(0);
  useFrame(({ dt }) => {
    rebuild();
    if (coupled || paused) return;
    const h = 1 / 60;
    acc.current = Math.min(acc.current + dt, 4 * h);
    for (let n = 0; acc.current >= h * 0.999 && n < 4; n++) { sim.step(); acc.current -= h; }
  }, -20);

  return (
    <FluidContext.Provider value={state}>
      {render === 'surface' && <FluidSurfaceLayer sim={sim} particleSize={particleSize} />}
      {render === 'particles' && <FluidParticlesLayer sim={sim} particleSize={particleSize} solidsOnly={false} />}
      {children}
    </FluidContext.Provider>
  );
});

function FluidParticlesLayer({ sim, particleSize, solidsOnly }: { sim: MPM; particleSize?: number; solidsOnly: boolean }) {
  useLayer(() => sim.spheres({ solidsOnly, size: particleSize ?? 1 }), [sim, particleSize, solidsOnly]);
  return null;
}
function FluidSurfaceLayer({ sim, particleSize }: { sim: MPM; particleSize?: number }) {
  return (<><FluidParticlesLayer sim={sim} particleSize={particleSize} solidsOnly /><FluidWater sim={sim} /></>);
}
function FluidWater({ sim }: { sim: MPM }) {
  useLayer(() => sim.fluid(), [sim]);
  return null;
}

export interface FluidBlockProps {
  /** Axis-aligned box [[x0, y0, z0], [x1, y1, z1]] (world units). */
  box?: [V3, V3];
  /** [[cx, cy, cz], radius]. */
  sphere?: [V3, number];
  /** [[cx, cy, cz], radius, halfHeight] (vertical). */
  cylinder?: [V3, number, number];
  /** Explicit particle positions (xyz, world units). */
  points?: Float32Array;
  /** water | oil | honey | jelly | rubber | snow | sand, or a custom material definition. */
  material?: string | MaterialDef;
  /** Initial velocity (m/s). */
  velocity?: V3;
  jitter?: number;
  seed?: number;
}
/** A region of material inside <Fluid>. Adding, removing or changing blocks restarts the fluid. */
export function FluidBlock(props: FluidBlockProps) {
  const s = useContext(FluidContext);
  if (!s) throw new Error('shaderintel-react: <FluidBlock> must be inside <Fluid>.');
  const token = useRef({}).current;
  const key = stableKey(props);
  useIsomorphicLayoutEffect(() => {
    s.blocks.set(token, props);
    s.dirty = true;
    return () => { s.blocks.delete(token); s.dirty = true; };
  }, [s, key]);
  return null;
}

// ------------------------------------------------------------------ cloth

export interface ClothProps extends ClothOptions {
  /** Height of the floor the cloth rests on (default 0; null for none). */
  floor?: number | null;
  /** Surface look: 'checker' or 'plain', the number of checker tiles, front/back colours. */
  pattern?: 'checker' | 'plain';
  tiles?: number;
  front?: V3;
  back?: V3;
  paused?: boolean;
}
/**
 * XPBD cloth with wind, gusts and colliders. Resolution, size, origin and pinning re-create it;
 * wind, substeps, compliances, colliders and the floor update live.
 */
export const Cloth = forwardRef<SICloth, ClothProps>(function Cloth(props, ref) {
  const { floor = 0, pattern, tiles, front, back, paused = false, ...opts } = props;
  const cloth = useResource((gpu) => gpu.cloth(opts),
    [stableKey([opts.width, opts.height, opts.size, opts.origin, typeof opts.pin === 'function' ? String(opts.pin) : opts.pin, opts.thickness])]);
  useImperativeHandle(ref, () => cloth, [cloth]);
  useIsomorphicLayoutEffect(() => {
    cloth.floorY = floor;
    if (opts.wind) cloth.wind = [...opts.wind];
    if (opts.substeps !== undefined) cloth.substeps = opts.substeps;
    if (opts.colliders) cloth.colliders = opts.colliders as ClothCollider[];
    if (opts.gravity) cloth.gravity = [...opts.gravity];
    if (opts.damping !== undefined) cloth.damping = opts.damping;
    if (opts.friction !== undefined) cloth.friction = opts.friction;
    for (const k of ['stretch', 'shear', 'bend'] as const) if (opts[k] !== undefined) cloth.compliance[k] = opts[k] as number;
  }, [cloth, floor, stableKey([opts.wind, opts.substeps, opts.colliders, opts.gravity, opts.damping, opts.friction, opts.stretch, opts.shear, opts.bend])]);
  useLayer((gpu) => gpu.mesh({ pos: cloth.pos, nrm: cloth.nrm, indices: cloth.indices, indexCount: cloth.indexCount, grid: [cloth.W, cloth.H],
    pattern, tiles, front, back }), [cloth, pattern, tiles, stableKey([front ?? null, back ?? null])]);
  useFrame(({ dt }) => { if (!paused) cloth.step(dt); });
  return null;
});

// ------------------------------------------------------------------ smoke and fire

export interface SmokeProps extends SmokeOptions {
  /** Volume rendering: density scale, emission (fire glow), tint, light direction... */
  density?: number;
  emission?: number;
  tint?: V3;
  absorption?: number;
  paused?: boolean;
}
/**
 * Eulerian smoke and fire on a staggered 3D grid with volumetric lighting. `res`, `size` and
 * `origin` re-create it; the solver settings and `sources` update live.
 */
export const Smoke = forwardRef<SISmoke, SmokeProps>(function Smoke(props, ref) {
  const { density, emission, tint, paused = false, ...opts } = props;
  const smoke = useResource((gpu) => gpu.smoke(opts), [stableKey([opts.res, opts.size, opts.origin])]);
  useImperativeHandle(ref, () => smoke, [smoke]);
  useIsomorphicLayoutEffect(() => {
    for (const k of ['iterations', 'omega', 'vorticity', 'buoyancy', 'weight', 'cooling', 'dissipation'] as const)
      if (opts[k] !== undefined) smoke[k] = opts[k] as number;
    if (opts.sources) smoke.sources = opts.sources.map((s: SmokeSource) => ({ ...s }));
  }, [smoke, stableKey([opts.iterations, opts.omega, opts.vorticity, opts.buoyancy, opts.weight, opts.cooling, opts.dissipation, opts.sources])]);
  const volume = useLayer((gpu) => gpu.volume({ smoke }), [smoke]);
  useIsomorphicLayoutEffect(() => {
    const v: Uniforms = {};
    if (density !== undefined) v.density = density;
    if (emission !== undefined) v.emission = emission;
    if (tint) v.tint = [...tint];
    if (Object.keys(v).length) volume.set(v);
  }, [volume, density, emission, stableKey(tint ?? null)]);
  useFrame(({ dt }) => { if (!paused) smoke.step(dt); });
  return null;
});

// ------------------------------------------------------------------ flocking

export interface BoidsProps extends BoidsOptions {
  /** Glyph size (default 0.0085 × bounds). */
  size?: number;
  colorBy?: 'heading' | 'speed' | 'constant';
  color?: V3;
  /** A point the flock flees from (null for none), its radius and strength. */
  predator?: V3 | null;
  predatorRadius?: number;
  seed?: number;
  paused?: boolean;
}
/** Flocking agents (separation, alignment, cohesion) through a GPU neighbour grid; 100k–1M boids. */
export const Boids = forwardRef<SIBoids, BoidsProps>(function Boids(props, ref) {
  const { size, colorBy = 'heading', color, predator, predatorRadius = 0.35, seed = 1, paused = false, ...opts } = props;
  const flock = useResource((gpu) => gpu.boids(opts).randomize(seed), [stableKey([opts.count, opts.bounds, opts.radius, opts.maxNeighbors]), seed]);
  useImperativeHandle(ref, () => flock, [flock]);
  useIsomorphicLayoutEffect(() => {
    const P = flock.params as Record<string, unknown>;
    for (const k of ['maxSpeed', 'minSpeed', 'separation', 'alignment', 'cohesion', 'wall'] as const) if (opts[k] !== undefined) P[k] = opts[k];
    P.predatorRadius = predator ? predatorRadius : 0;
    if (predator) P.predator = [...predator];
  }, [flock, stableKey([opts.maxSpeed, opts.minSpeed, opts.separation, opts.alignment, opts.cohesion, opts.wall, predator ?? null, predatorRadius])]);
  const glyphs = useLayer((gpu) => gpu.glyphs({ pos: flock.pos, vel: flock.vel, count: flock.count, size: size ?? 0.0085 * (opts.bounds ?? 1), colorBy, color }), [flock]);
  useIsomorphicLayoutEffect(() => {
    glyphs.set({ colorBy: { constant: 0, speed: 1, heading: 2 }[colorBy], ...(size !== undefined ? { size } : {}), ...(color ? { color: [...color] } : {}) });
  }, [glyphs, colorBy, size, stableKey(color ?? null)]);
  useFrame(({ dt }) => { if (!paused) flock.step(dt); });
  return null;
});

// ------------------------------------------------------------------ gravity

export interface NBodyProps {
  count: number;
  G?: number;
  softening?: number;
  /** Simulation time per step. */
  dt?: number;
  /** Initial conditions: two colliding galaxies (default) or a Plummer sphere. */
  initial?: 'galaxies' | 'plummer';
  seed?: number;
  /** Steps per frame (default 1). */
  stepsPerFrame?: number;
  /** Time every kernel variant on this GPU first and use the fastest (cached per adapter). */
  autotune?: boolean;
  /** Point sprites: world size, minimum pixels, colour (linear HDR), brightness. */
  size?: number;
  pixelSize?: number;
  color?: V3;
  intensity?: number;
  paused?: boolean;
}
/** Direct O(N²) gravity, autotuned per GPU, drawn as glowing points (pair with bloom). */
export const NBody = forwardRef<SINBody, NBodyProps>(function NBody(props, ref) {
  const { count, G, softening, dt, initial = 'galaxies', seed = 1, stepsPerFrame = 1, autotune = true, size = 0.004, pixelSize = 1.2,
    color = [1.0, 0.72, 0.45], intensity = 0.9, paused = false } = props;
  const sim = useResource((gpu) => {
    const s = gpu.nbody({ count, G, softening, dt } as { count: number });
    if (initial === 'plummer') s.plummer({ seed }); else s.galaxies({ seed });
    return s;
  }, [count, initial, seed]);
  useImperativeHandle(ref, () => sim, [sim]);
  const tuned = useRef(false);
  useEffect(() => {
    if (!autotune || tuned.current) return;
    let live = true;
    sim.autotune({ cache: true }).then(() => { if (live) tuned.current = true; }, (e: unknown) => console.warn('shaderintel-react: N-body autotune failed', e));
    return () => { live = false; };
  }, [sim, autotune]);
  useIsomorphicLayoutEffect(() => {
    if (G !== undefined) sim.G = G;
    if (softening !== undefined) sim.softening = softening;
    if (dt !== undefined) sim.dt = dt;
  }, [sim, G, softening, dt]);
  const points = useLayer((gpu) => {
    const p = gpu.points({ data: sim.pos, size, pixelSize, color, intensity, colorBy: 'constant' });
    return { points: p, set: (v: Uniforms) => p.set(v), draw: (pass: unknown, v: unknown) => { p.bind({ P: sim.pos }).draw(pass as never, v as never); } };
  }, [sim]);
  useIsomorphicLayoutEffect(() => { points.set({ size, minPixels: pixelSize, color: [...color], intensity }); }, [points, size, pixelSize, stableKey(color), intensity]);
  useFrame(() => { if (!paused && stepsPerFrame > 0) sim.step(stepsPerFrame); });
  return null;
});
