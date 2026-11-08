// Rigid-body physics as components:
//
//   <Physics gravity={[0, -9.81, 0]}>
//     <Box position={[0, 4, 0]} size={[1, 1, 1]} />
//     <Sphere ref={ball} position={[0, 8, 0]} radius={0.5} />
//     <Box type="static" position={[0, -0.5, 0]} size={[20, 1, 20]} />
//   </Physics>
//
// <Physics> owns a RigidWorld, steps it at a fixed rate and draws its bodies. Bodies are added when
// they mount and removed when they unmount. Their position / rotation / velocity props are initial
// values (the simulation owns the pose afterwards) — except for kinematic bodies, which follow
// their position / rotation props. Use the ref API (applyImpulse, setVelocity, setPose, ...) to
// push bodies around.
import { createContext, forwardRef, useContext, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import type { ReactNode, RefObject } from 'react';
import type { BodiesOptions, BodyOptions, ForceField as SIForceField, JointOptions, RigidWorld, RigidWorldOptions, Rotation, Uniforms, WGSLType } from 'shaderintel';
import { useShaderIntel } from './context.js';
import { stableKey, useDisposable, useFrame, useIsomorphicLayoutEffect, useLayer } from './hooks.js';

type V3 = [number, number, number] | number[];

export interface PhysicsState {
  world: RigidWorld;
  /** Work queued until the next step (joints waiting for their bodies). */
  pending: Set<() => boolean>;
  /** Run before every world step, in lockstep with it (a coupled fluid steps here). */
  beforeStep: Set<(dt: number) => void>;
  timeStep: number | 'frame';
}
const PhysicsContext = createContext<PhysicsState | null>(null);

/** The RigidWorld of the enclosing <Physics>. */
export function useWorld(): RigidWorld {
  const s = useContext(PhysicsContext);
  if (!s) throw new Error('shaderintel-react: useWorld() / bodies / joints must be used inside <Physics>.');
  return s.world;
}
/** Same as useWorld(), but null outside <Physics> (for components that optionally couple to a world). */
export function useOptionalWorld(): RigidWorld | null { return useContext(PhysicsContext)?.world ?? null; }
/** The enclosing <Physics>' internals (world, lockstep hooks), or null. For simulations that couple to it. */
export function usePhysicsState(): PhysicsState | null { return useContext(PhysicsContext); }

export interface PhysicsProps extends RigidWorldOptions {
  children?: ReactNode;
  /** Seconds per simulation step (default 1/60), run as often as real time requires (at most maxSteps per frame); "frame" steps once per frame by the frame time. */
  timeStep?: number | 'frame';
  /** Cap on steps per frame, so a slow frame cannot snowball (default 4). */
  maxSteps?: number;
  /** Stop stepping (bodies keep their state). */
  paused?: boolean;
  /** Draw the bodies with ShaderIntel's renderer (default true). Pass false when drawing them yourself (e.g. with three.js). */
  render?: boolean;
  /** Options for the body renderer (colours, ...). */
  renderOptions?: BodiesOptions;
  /** Called after every step with the world (read stats, apply game logic). */
  onStep?: (world: RigidWorld) => void;
}

const WORLD_KEYS = ['maxBodies', 'maxContacts', 'maxPairs', 'maxJoints', 'maxCommands', 'maxEvents', 'bounds', 'deterministic', 'events',
  'fusedSolve'] as const;

/**
 * A GPU rigid-body world (Augmented Vertex Block Descent). Capacity options (maxBodies, bounds,
 * deterministic, events, ...) re-create the world when they change; solver options (gravity,
 * iterations, substeps, floor, ...) apply in place. The ref is the RigidWorld.
 */
export const Physics = forwardRef<RigidWorld, PhysicsProps>(function Physics(props, ref) {
  const { children, timeStep = 1 / 60, maxSteps = 4, paused = false, render = true, renderOptions, onStep, ...opts } = props;
  const { gpu } = useShaderIntel();
  const capacityKey = stableKey(WORLD_KEYS.map((k) => (opts as Record<string, unknown>)[k] ?? null));
  const state = useDisposable<PhysicsState>(() => ({ world: gpu.rigid(opts), pending: new Set(), beforeStep: new Set(), timeStep }),
    (s) => s.world.destroy(), [gpu, capacityKey]);
  state.timeStep = timeStep;
  const world = state.world;
  useImperativeHandle(ref, () => world, [world]);

  // solver options that can change at any time
  const live = { gravity: opts.gravity, iterations: opts.iterations, substeps: opts.substeps, floor: opts.floor, alpha: opts.alpha, beta: opts.beta,
    gamma: opts.gamma, margin: opts.margin, slop: opts.slop, stickThreshold: opts.stickThreshold, postStabilize: opts.postStabilize };
  useIsomorphicLayoutEffect(() => {
    for (const [k, v] of Object.entries(live)) if (v !== undefined) (world as unknown as Record<string, unknown>)[k] = Array.isArray(v) ? [...v] : v;
  }, [world, stableKey(live)]);

  const acc = useRef(0);
  useFrame(({ dt }) => {
    for (const f of state.pending) if (f()) state.pending.delete(f);
    if (paused) return;
    const stepOnce = (h: number) => {
      for (const f of state.beforeStep) f(h);
      world.step(h);
      onStep?.(world);
    };
    if (timeStep === 'frame') { stepOnce(dt); return; }
    acc.current = Math.min(acc.current + dt, timeStep * maxSteps);
    let n = 0;
    while (acc.current >= timeStep * 0.999 && n < maxSteps) { stepOnce(timeStep); acc.current -= timeStep; n++; }
  }, -10);

  return (
    <PhysicsContext.Provider value={state}>
      {render && <BodiesLayer world={world} options={renderOptions} />}
      {children}
    </PhysicsContext.Provider>
  );
});

function BodiesLayer({ world, options }: { world: RigidWorld; options?: BodiesOptions }) {
  useLayer((gpu) => gpu.bodies(world, options), [world, stableKey(options ?? null)]);
  return null;
}

// ------------------------------------------------------------------ bodies

/** The imperative handle of a body (a ref on <RigidBody>, <Box>, <Sphere>, <Capsule>). */
export interface RigidBodyApi {
  /** Slot of the body in the world (-1 before it is added or if the world is full). */
  readonly id: number;
  readonly world: RigidWorld;
  /** Impulse (N·s) at a world point (default: the centre of mass). */
  applyImpulse(impulse: V3, point?: V3): void;
  /** Force (N) for the next frame. */
  applyForce(force: V3, point?: V3): void;
  applyAngularImpulse(impulse: V3): void;
  setVelocity(linear: V3, angular?: V3): void;
  /** Teleport. */
  setPose(position: V3, rotation?: Rotation): void;
  /** Kinematic bodies: reach this pose at the end of the next step, pushing what is in the way. */
  moveKinematic(position: V3, rotation?: Rotation): void;
  setEnabled(on: boolean): void;
  setFilter(group: number, mask: number): void;
}

export type BodyType = 'dynamic' | 'static' | 'kinematic';

export interface RigidBodyProps extends Omit<BodyOptions, 'static' | 'kinematic'> {
  shape: 'box' | 'sphere' | 'capsule';
  /** dynamic (default), static (never moves) or kinematic (follows its position/rotation props, pushes others). */
  type?: BodyType;
  /** Boxes: half extents, or `size` (full extents). */
  half?: V3;
  size?: V3;
  /** Spheres and capsules. */
  radius?: number;
  /** Capsules: half length of the straight part (along local y). */
  halfHeight?: number;
  /** Called with the body's id once it is in the world. */
  onAdded?: (id: number) => void;
}

function bodyOptions(p: RigidBodyProps): BodyOptions & { half?: number[]; radius?: number; halfHeight?: number } {
  const { shape: _s, type, size, onAdded: _o, ...rest } = p;
  const o: Record<string, unknown> = { ...rest, static: type === 'static', kinematic: type === 'kinematic' };
  if (size && !rest.half) o.half = [size[0] / 2, size[1] / 2, size[2] / 2];
  for (const k of Object.keys(o)) if (o[k] === undefined) delete o[k];
  return o as BodyOptions;
}

/** A rigid body: shape="box" | "sphere" | "capsule". Prefer the <Box>, <Sphere> and <Capsule> shorthands. */
export const RigidBody = forwardRef<RigidBodyApi, RigidBodyProps>(function RigidBody(props, ref) {
  const world = useWorld();
  const idRef = useRef(-1);
  const api = useMemo<RigidBodyApi>(() => ({
    get id() { return idRef.current; },
    world,
    applyImpulse: (J, p) => { if (idRef.current >= 0) world.applyImpulse(idRef.current, [...J], p && [...p]); },
    applyForce: (F, p) => { if (idRef.current >= 0) world.applyForce(idRef.current, [...F], p && [...p]); },
    applyAngularImpulse: (L) => { if (idRef.current >= 0) world.applyAngularImpulse(idRef.current, [...L]); },
    setVelocity: (v, w) => { if (idRef.current >= 0) world.setVelocity(idRef.current, [...v], w && [...w]); },
    setPose: (p, q) => { if (idRef.current >= 0) world.setPose(idRef.current, [...p], q); },
    moveKinematic: (p, q) => { if (idRef.current >= 0) world.moveKinematic(idRef.current, [...p], q); },
    setEnabled: (on) => { if (idRef.current >= 0) world.setEnabled(idRef.current, on); },
    setFilter: (g, m) => { if (idRef.current >= 0) world.setFilter(idRef.current, g, m); },
  }), [world]);
  useImperativeHandle(ref, () => api, [api]);

  // shape and mass properties are fixed for a body's life: changing them adds a new body
  const shapeKey = stableKey([props.shape, props.type ?? 'dynamic', props.half ?? props.size ?? null, props.radius ?? null, props.halfHeight ?? null,
    props.density ?? null, props.sensor ?? false, props.fixedRotation ?? false]);
  const latest = useRef(props);
  latest.current = props;
  useIsomorphicLayoutEffect(() => {
    const p = latest.current;
    const o = bodyOptions(p);
    const id = p.shape === 'sphere' ? world.addSphere({ ...o, radius: p.radius ?? 0.5 })
      : p.shape === 'capsule' ? world.addCapsule({ ...o, radius: p.radius ?? 0.25, halfHeight: p.halfHeight })
        : world.addBox(o);
    if (id < 0) console.warn(`shaderintel-react: the world is full (maxBodies ${world.maxBodies}); a ${p.shape} was not added`);
    idRef.current = id;
    if (id >= 0) p.onAdded?.(id);
    return () => { if (id >= 0 && !world.destroyed) world.remove(id); idRef.current = -1; };
  }, [world, shapeKey]);

  // properties that can change on a live body
  const filterKey = stableKey([props.group ?? 1, props.mask ?? 0xffffffff]);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    if (idRef.current >= 0) world.setFilter(idRef.current, props.group ?? 1, props.mask ?? 0xffffffff);
  }, [world, filterKey]);
  const propsKey = stableKey([props.gravityScale ?? 1, props.linearDamping ?? 0, props.angularDamping ?? 0]);
  const firstProps = useRef(true);
  useEffect(() => {
    if (firstProps.current) { firstProps.current = false; return; }
    if (idRef.current >= 0) world.setProps(idRef.current, { gravityScale: props.gravityScale, linearDamping: props.linearDamping, angularDamping: props.angularDamping });
  }, [world, propsKey]);
  // kinematic bodies follow their props
  const poseKey = stableKey([props.position ?? null, props.rotation ?? null]);
  useEffect(() => {
    if (props.type === 'kinematic' && idRef.current >= 0 && props.position) world.moveKinematic(idRef.current, [...props.position], props.rotation);
  }, [world, poseKey, props.type]);
  return null;
});

type ShapeProps<K extends string> = Omit<RigidBodyProps, 'shape' | K>;
/** A box: size (full extents, default 1×1×1) or half extents. */
export const Box = forwardRef<RigidBodyApi, ShapeProps<'radius' | 'halfHeight'>>(function Box(props, ref) {
  return <RigidBody ref={ref} shape="box" {...props} />;
});
/** A sphere (radius, default 0.5). */
export const Sphere = forwardRef<RigidBodyApi, ShapeProps<'half' | 'size' | 'halfHeight'>>(function Sphere(props, ref) {
  return <RigidBody ref={ref} shape="sphere" {...props} />;
});
/** A capsule along its local y axis (radius, halfHeight of the straight part). */
export const Capsule = forwardRef<RigidBodyApi, ShapeProps<'half' | 'size'>>(function Capsule(props, ref) {
  return <RigidBody ref={ref} shape="capsule" {...props} />;
});

// ------------------------------------------------------------------ joints

type BodyRef = RefObject<RigidBodyApi | null> | number | null;
const idOf = (b: BodyRef | undefined): number | null =>
  b === null || b === undefined ? null : typeof b === 'number' ? b : b.current && b.current.id >= 0 ? b.current.id : -2;

export interface JointProps extends Omit<JointOptions, 'a' | 'b' | 'type'> {
  type: JointOptions['type'];
  /** The first body: a ref from <Box ref={...}> (or a body id). */
  a: BodyRef;
  /** The second body, or null / omitted for the world. */
  b?: BodyRef;
  /** false disables the joint (it can be enabled again). */
  enabled?: boolean;
  /** Called with the joint index once it exists. */
  onAdded?: (joint: number) => void;
}
/**
 * A joint between two bodies (or a body and the world): ball, hinge, fixed, distance, slider.
 * Anchors and axes are world-space at the moment the joint is created. Motors, limits and
 * `enabled` update live; other props are fixed (change the joint's key to rebuild it).
 */
export const Joint = forwardRef<number, JointProps>(function Joint(props, ref) {
  const s = useContext(PhysicsContext);
  if (!s) throw new Error('shaderintel-react: <Joint> must be inside <Physics>.');
  const world = s.world;
  const index = useRef(-1);
  useImperativeHandle(ref, () => index.current, [index.current]);
  const latest = useRef(props);
  latest.current = props;
  useIsomorphicLayoutEffect(() => {
    let alive = true;
    const tryAdd = () => {
      if (!alive) return true;
      const p = latest.current;
      const a = idOf(p.a), b = idOf(p.b);
      if (a === null || a === -2 || b === -2) return false;   // a body is not in the world yet: retry before the next step
      const { a: _a, b: _b, enabled, onAdded, ...rest } = p;
      const k = world.addJoint({ ...rest, a, b } as JointOptions);
      index.current = k;
      if (k >= 0) { if (enabled === false) world.setJointEnabled(k, false); onAdded?.(k); }
      else console.warn(`shaderintel-react: too many joints (maxJoints ${world.maxJoints})`);
      return true;
    };
    if (!tryAdd()) s.pending.add(tryAdd);
    return () => {
      alive = false;
      s.pending.delete(tryAdd);
      if (index.current >= 0 && !world.destroyed) world.setJointEnabled(index.current, false);
      index.current = -1;
    };
  }, [world]);
  const motorKey = stableKey(props.motor ?? null);
  useEffect(() => {
    if (index.current < 0) return;
    if (props.motor) world.setMotor(index.current, props.motor.speed, props.motor.maxForce);
  }, [world, motorKey]);
  const limitsKey = stableKey(props.limits ?? null);
  useEffect(() => {
    if (index.current < 0 || !props.limits) return;
    world.setLimits(index.current, props.limits[0], props.limits[1]);
  }, [world, limitsKey]);
  useEffect(() => {
    if (index.current >= 0 && props.enabled !== undefined) world.setJointEnabled(index.current, props.enabled);
  }, [world, props.enabled]);
  return null;
});

// ------------------------------------------------------------------ force fields

export interface ForceFieldProps {
  /** WGSL run for every dynamic body before each frame: read `b` (the Body), add to `force` / `torque`. */
  code: string;
  uniforms?: Record<string, WGSLType>;
  values?: Uniforms;
  imports?: string[];
  enabled?: boolean;
}
/** Custom forces on every body (wind, buoyancy, magnets, attractors) as a WGSL snippet. */
export const ForceField = forwardRef<SIForceField | null, ForceFieldProps>(function ForceField(props, ref) {
  const world = useWorld();
  const { code, uniforms, values, imports, enabled = true } = props;
  const field = useDisposable(() => world.addForceField({ code, uniforms, imports, values }), (f) => { f.enabled = false; },
    [world, code, stableKey(uniforms ?? {}), stableKey(imports ?? [])]);
  useImperativeHandle(ref, () => field, [field]);
  const valuesKey = stableKey(values ?? {});
  useIsomorphicLayoutEffect(() => { if (values) field.set(values); }, [field, valuesKey]);
  useIsomorphicLayoutEffect(() => { field.enabled = enabled; return () => { field.enabled = false; }; }, [field, enabled]);
  return null;
});
