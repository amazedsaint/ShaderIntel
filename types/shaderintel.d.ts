// Type declarations for ShaderIntel (src/index.js and dist/shaderintel.js).
//
// The library is plain JavaScript (ES modules, no build step); these declarations describe its
// public API for editors and TypeScript projects. WebGPU objects resolve to the @webgpu/types
// definitions when those are installed and to `unknown` otherwise, so the file compiles either way.

type G<K extends string> = typeof globalThis extends { [P in K]: { prototype: infer T } } ? T : unknown;
export type GPUDeviceT = G<'GPUDevice'>;
export type GPUAdapterT = G<'GPUAdapter'>;
export type GPUBufferT = G<'GPUBuffer'>;
export type GPUCommandEncoderT = G<'GPUCommandEncoder'>;
export type GPUComputePassEncoderT = G<'GPUComputePassEncoder'>;
export type GPURenderPassEncoderT = G<'GPURenderPassEncoder'>;
export type GPUTextureT = G<'GPUTexture'>;
export type GPUTextureViewT = G<'GPUTextureView'>;
export type GPUSamplerT = G<'GPUSampler'>;
export type GPUShaderModuleT = G<'GPUShaderModule'>;

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
/** A quaternion [x, y, z, w] or Euler angles [rx, ry, rz] in radians. */
export type Rotation = Quat | Vec3;
export type TypedArray = Float32Array | Uint32Array | Int32Array | Uint16Array | Uint8Array | Float64Array;

/** WGSL element types accepted by arrays and kernel bindings ('f32', 'vec4f', 'u32', a struct name, ...). */
export type WGSLType = 'f32' | 'u32' | 'i32' | 'f16' | 'vec2f' | 'vec3f' | 'vec4f' | 'vec2u' | 'vec3u' | 'vec4u'
  | 'vec2i' | 'vec3i' | 'vec4i' | 'vec2h' | 'vec4h' | 'mat3x3f' | 'mat4x4f' | (string & {});

// ---------------------------------------------------------------------------------- core

/** Why a ShaderIntel call failed, for programmatic handling. */
export type SIErrorCode = 'unsupported' | 'no-adapter' | 'no-device' | 'compile' | 'invalid-argument' | 'error';
/** The error class for every failure ShaderIntel reports on purpose. */
export class SIError extends Error {
  constructor(message: string, code?: SIErrorCode);
  readonly code: SIErrorCode;
}

export interface InitOptions {
  powerPreference?: 'high-performance' | 'low-power';
  /** false forces the portable path even where the feature exists (testing). */
  subgroups?: boolean;
  f16?: boolean;
  timestamps?: boolean;
  subgroupId?: boolean;
  /** Use a device created elsewhere (e.g. by three.js) instead of requesting one; it is not destroyed by Context.destroy(). */
  device?: GPUDeviceT;
  /** The adapter of `device`, when known (for caps.vendor / architecture). */
  adapter?: GPUAdapterT | null;
}

/**
 * Initialise WebGPU and return a Context. Rejects with an SIError whose code is 'unsupported'
 * (no WebGPU in this browser), 'no-adapter' or 'no-device'.
 */
export function init(opts?: InitOptions): Promise<Context>;
/** True when this environment exposes WebGPU (navigator.gpu); an adapter may still be unavailable. */
export function isSupported(): boolean;

export interface Caps {
  subgroups: boolean;
  f16: boolean;
  timestamps: boolean;
  float32Filterable: boolean;
  float32Blendable: boolean;
  subgroupMin: number;
  subgroupMax: number;
  subgroupId: boolean;
  vendor: string;
  architecture: string;
  workgroupStorage: number;
}

export interface ProfileEntry { label: string; ms: number; last: number; samples: number }

export class Context {
  /** null when the context was created from an external device without its adapter. */
  readonly adapter: GPUAdapterT | null;
  readonly device: GPUDeviceT;
  readonly caps: Caps;
  readonly limits: Record<string, number>;
  /** True after destroy() or once the device is lost. */
  readonly destroyed: boolean;
  /** False when the device was passed in with init({ device }). */
  readonly ownsDevice: boolean;
  /** Preferred canvas format. */
  readonly format: string;
  readonly stats: { dispatches: number; submits: number };
  readonly profiler: { enabled: boolean; report(): ProfileEntry[]; reset(): void };
  /** Resolves (with the loss info) if the device is lost. */
  readonly lost: Promise<unknown>;

  /** Typed GPU array: ctx.array('vec4f', n) or ctx.array('f32', float32Array). */
  array(type: WGSLType, lengthOrData: number | ArrayLike<number>, opts?: ArrayOptions): GArray;
  /** A compute kernel from the body of a WGSL function (see KernelDesc). */
  kernel(desc: KernelDesc): Kernel;
  sampler(filter?: 'linear' | 'nearest', wrap?: string): GPUSamplerT;
  texture(desc: { size: number[]; format?: string; dimension?: '1d' | '2d' | '3d'; usage?: number; sampleCount?: number; label?: string }): TextureHandle;
  /** The command encoder of the current batch (created on demand). */
  readonly encoder: GPUCommandEncoderT;
  computePass(): GPUComputePassEncoderT;
  endPass(): void;
  renderPass(desc: unknown): GPURenderPassEncoderT;
  /** Time a block of GPU work: it becomes one compute pass with timestamps (see profiler.report()). */
  region<T>(label: string, fn: () => T): T;
  /** Submit everything recorded so far. */
  flush(): void;
  /** Flush and wait until the GPU is idle. */
  idle(): Promise<void>;
  clear(arr: GArray): this;
  /** Fill every 32-bit word with `value` (u32 bits). */
  fill(arr: GArray, value?: number): this;
  fillFloat(arr: GArray, x: number): this;
  copy(src: GArray, dst: GArray, bytes?: number, srcOffset?: number, dstOffset?: number): void;
  read(arr: GArray, count?: number): Promise<TypedArray>;
  /** Read a texture back (needs COPY_SRC); half floats are expanded. */
  readTexture(tex: TextureHandle | GPUTextureT): Promise<{ width: number; height: number; depth: number; channels: number; data: Float32Array }>;
  /**
   * Write data into a buffer ordered with the recorded work: dispatches recorded before see the old
   * contents, those recorded after see the new (queue.writeBuffer would land before all of them).
   */
  writeOrdered(dst: GArray | GPUBufferT, data: ArrayBufferView | ArrayLike<number>, byteOffset?: number): this;
  shaderModule(code: string, label?: string): { module: GPUShaderModuleT; ready: Promise<void> };
  /**
   * Submit pending work and release the device — and with it every buffer, texture and pipeline made
   * on it. A device passed in with init({ device }) is left alive. Idempotent.
   */
  destroy(): void;

  // factories (also available as classes)
  view(canvas: string | HTMLCanvasElement, opts?: ViewOptions): View;
  rigid(opts?: RigidWorldOptions): RigidWorld;
  bodies(world: RigidWorld, opts?: BodiesOptions): Bodies;
  mpm(opts?: MPMOptions): MPM;
  nbody(opts: NBodyOptions): NBody;
  cloth(opts?: ClothOptions): Cloth;
  smoke(opts?: SmokeOptions): Smoke;
  boids(opts?: BoidsOptions): Boids;
  sdf(opts: SdfSceneOptions): SdfScene;
  volume(opts: VolumeOptions): Volume;
  mesh(opts: MeshOptions): SurfaceMesh;
  glyphs(opts: GlyphsOptions): Glyphs;
  points(opts: PointsOptions): Points;
  stage(opts?: StageOptions): Stage;
  fullscreen(desc: FullscreenDesc): Fullscreen;
  program(desc: Record<string, unknown>): RenderProgram;
  spheres(opts: SpheresOptions): Spheres;
}

export interface ArrayOptions { usage?: number; label?: string }

/** A texture with its default view (from Context.texture()). */
export interface TextureHandle {
  texture: GPUTextureT; view: GPUTextureViewT; id: string;
  width: number; height: number; depth: number; format: string;
  destroy(): void;
}

/** A typed GPU array (a storage buffer with an element type). */
export class GArray {
  readonly ctx: Context;
  readonly id: number;
  readonly type: string;
  /** Bytes per element (WGSL storage layout: a vec3f occupies 16). */
  readonly stride: number;
  readonly length: number;
  readonly byteLength: number;
  readonly label: string;
  readonly buffer: GPUBufferT;
  readonly destroyed?: boolean;
  /** Scalars per element as laid out in memory. */
  readonly slots: number;
  /**
   * Write elements starting at `elementOffset` (queue write: ordered before unsubmitted work — flush
   * first if that matters). Throws SIError('invalid-argument') if the data does not fit.
   */
  write(data: ArrayLike<number> | ArrayBufferView, elementOffset?: number): this;
  /** Like write(), but ordered with recorded work (see Context.writeOrdered). */
  writeOrdered(data: ArrayLike<number> | ArrayBufferView, elementOffset?: number): this;
  /** Zero the array, recorded in order with dispatches. */
  clear(): this;
  /** Read back (flushes); vec3 elements come back padded to 4 scalars. */
  read(count?: number): Promise<TypedArray>;
  /** Read back with vec3 padding removed. */
  readPacked(count?: number): Promise<TypedArray>;
  destroy(): void;
}

export interface BufferBinding {
  type?: WGSLType;
  /** An array to bind by default. */
  array?: GArray;
  access?: 'read' | 'read_write';
  atomic?: boolean;
  /** Bind as var<uniform> of `type` instead of a storage array. */
  uniform?: boolean;
}

export interface KernelDesc {
  name?: string;
  /** Storage bindings: a GArray (bound by default) or a declaration. Read in WGSL by name. */
  /** A GArray (bound by default), a declaration, or just its WGSL element type. */
  buffers?: Record<string, GArray | BufferBinding | WGSLType>;
  /** Uniforms: name → WGSL type; read in WGSL as u.name. */
  uniforms?: Record<string, WGSLType>;
  textures?: Record<string, unknown>;
  samplers?: Record<string, unknown>;
  /** Shader-library modules to link ('noise', 'sdf', 'rigid', ...). */
  imports?: string[];
  /** Module-scope WGSL (helper functions, var<workgroup> arrays). */
  shared?: string;
  /** The body of the entry point. Defined names: i (1D), p (nD), u, gid, lid, lidx, wid, nwg, wgi, and subgroup names when used. */
  code: string;
  /** 1 (default), 2 or 3. */
  dims?: 1 | 2 | 3;
  workgroupSize?: number | [number, number] | [number, number, number];
  /** Skip the generated bounds check (default true: guarded). */
  guard?: boolean;
  /** Module-scope constants: { K: 4 } → const K = 4; */
  defines?: Record<string, number | boolean | string | number[]>;
  /** Initial uniform values. */
  values?: Uniforms;
  /** false makes capability-dependent library modules emit their portable code. */
  subgroups?: boolean;
}

export type Uniforms = Record<string, number | ArrayLike<number> | boolean>;
export type Resources = Record<string, GArray | unknown>;

/**
 * A compute kernel. Storage bindings are checked against their WGSL types when bound:
 * a buffer that is not a whole number of elements is an error naming the kernel and binding.
 */
export class Kernel {
  readonly name: string;
  /** Resolves when compiled; rejects with formatted WGSL errors. */
  ready(): Promise<void>;
  /** Persistent uniform values. */
  set(values: Uniforms): this;
  /** Persistently bind resources by name. */
  bind(resources: Resources): this;
  /** Run over `count` invocations (1D) or an extent [x, y, z] (nD). */
  run(count: number | number[], uniforms?: Uniforms | null, resources?: Resources): this;
  /** Dispatch an explicit number of workgroups. */
  dispatch(groups: number | number[], uniforms?: Uniforms | null, resources?: Resources): this;
  /** Dispatch with workgroup counts read from a GPU buffer (3 × u32 at byte `offset`). */
  dispatchIndirect(args: GArray, offset?: number, uniforms?: Uniforms | null, resources?: Resources): this;
}

export interface ModuleDef {
  /** WGSL, or a function of the device capabilities returning WGSL (one name, several implementations). */
  code: string | ((caps: Caps) => string);
  /** Modules this one needs. */
  deps?: string[];
  doc?: string;
}
/** Register a WGSL module that kernels can import by name. */
export function defineModule(name: string, def: string | ModuleDef): void;
export function link(code: string, imports?: string[], caps?: Partial<Caps>): string;
export function moduleNames(): string[];
export function moduleInfo(name: string): unknown;
export function literal(x: number | boolean, type?: 'f32' | 'i32' | 'u32'): string;
export function layoutOf(type: string): { type: string; stride: number; components: number; scalar: string; [k: string]: unknown };
export function structLayout(name: string, members: Record<string, WGSLType>): {
  name: string; size: number; align: number; wgsl: string;
  fields: Array<{ name: string; type: string; offset: number; layout: unknown }>;
};
export function toHalf(x: number): number;
export function fromHalf(h: number): number;

// ---------------------------------------------------------------------------------- primitives

/** Exclusive prefix sum of u32 values (reduce-then-scan, or single-pass decoupled look-back). */
export class Scan {
  constructor(ctx: Context, maxN: number, o?: { mode?: 'reduce-then-scan' | 'single-pass'; lookback?: 'thread' | 'workgroup' | 'subgroup'; spins?: number });
  /** Returns dst. */
  run(src: GArray, dst?: GArray, n?: number): GArray;
  destroy(): void;
}

export class Reduce {
  constructor(ctx: Context, o?: { type?: WGSLType; input?: WGSLType; op?: 'sum' | 'min' | 'max'; map?: string; maxN?: number });
  /** The 1-element result array (on the GPU); n = 0 gives the identity; n > maxN throws. */
  run(src: GArray, n?: number): GArray;
  destroy(): void;
  read(): Promise<number | number[]>;
}

/** Uniform-grid counting sort for fixed-radius neighbour queries. */
export class SpatialGrid {
  constructor(ctx: Context, o: { maxCount: number; min: number[]; max: number[]; cellSize: number });
  readonly cellSize: number;
  readonly order: GArray;
  readonly start: GArray;
  uniforms(): Record<string, WGSLType>;
  values(): Uniforms;
  build(pos: GArray, count: number): void;
  gather(src: GArray, dst: GArray, count?: number): void;
  /** WGSL that runs `body` for every neighbour index j of point expression `P` (27 cells as 9 row ranges). */
  static neighbors(P: string, body: string, startName?: string): string;
}

/** Mean GPU wall time (ms) of fn(), measured over batches of ~20 ms between submit fences. */
export function timeIt(ctx: Context, fn: () => unknown, reps?: number, workHint?: number): Promise<number>;

// ---------------------------------------------------------------------------------- rendering

export type Tonemap = 'agx' | 'aces' | 'neutral' | 'reinhard' | 'none';

export interface CameraOptions {
  target?: number[];
  distance?: number;
  /** Radians around the y axis. */
  yaw?: number;
  /** Radians above the horizon. */
  pitch?: number;
  /** Vertical field of view in degrees (default 45). */
  fov?: number;
  near?: number;
  far?: number;
  minDistance?: number;
  maxDistance?: number;
  minPitch?: number;
  maxPitch?: number;
  /** Radians per second. */
  autoRotate?: number;
  /** Higher settles faster (default 8). */
  damping?: number;
  rotateSpeed?: number;
  zoomSpeed?: number;
  panSpeed?: number;
  /** false ignores input without detaching the listeners. */
  enabled?: boolean;
}

export interface ViewOptions {
  camera?: CameraOptions;
  /** Mouse / touch orbit controls (default true). */
  controls?: boolean;
  /** Stats overlay (FPS, per-region GPU times); { extra: () => lines, parent } to customise. */
  hud?: boolean | { extra?: () => string | string[]; parent?: HTMLElement };
  bloom?: boolean | { strength?: number; [k: string]: unknown };
  /** Linear RGBA clear colour. */
  background?: number[];
  exposure?: number;
  tonemap?: Tonemap;
  /** Fixed device-pixel ratio (default: the screen's, capped at maxPixelRatio, followed when it changes). */
  pixelRatio?: number;
  maxPixelRatio?: number;
  /** Receives errors thrown by the frame callback or rendering (the loop stops); default: an overlay on the canvas. */
  onError?: (error: Error) => void;
}

/** What View.run's callback receives. */
export interface FrameInfo { dt: number; time: number; frame: number; view: View }

/** Anything a View can draw. Every method is optional. */
export interface Layer {
  /** Record compute or off-screen work before the scene pass. */
  prepare?(view: View): void;
  /** Draw into the HDR scene pass (rgba16float + depth32float). */
  draw?(pass: GPURenderPassEncoderT, view: View): void;
  /** Screen-space effects after the scene (water surfaces, volumes). */
  afterScene?(view: View): void;
  /** Draw after tonemapping, straight into the canvas. */
  overlay?(pass: GPURenderPassEncoderT, view: View): void;
  resize?(view: View): void;
  attach?(view: View): void;
  detach?(view: View): void;
  destroy?(): void;
}

export interface Pointer {
  /** CSS pixels from the canvas's top-left corner. */
  x: number; y: number;
  /** Normalised device coordinates ([-1, 1], y up). */
  ndc: [number, number];
  down: boolean; buttons: number; inside: boolean;
  shift: boolean; alt: boolean; ctrl: boolean;
}

/** A canvas with an HDR scene target, an orbit camera, tonemapping, optional bloom and HUD, and a frame loop. */
export class View {
  constructor(ctx: Context, canvas: string | HTMLCanvasElement, opts?: ViewOptions);
  readonly ctx: Context;
  readonly id: string;
  readonly canvas: HTMLCanvasElement;
  readonly camera: OrbitCamera;
  readonly pointer: Pointer;
  readonly layers: ReadonlyArray<Layer>;
  /** Drawing-buffer size in device pixels. */
  readonly width: number;
  readonly height: number;
  readonly pixelRatio: number;
  readonly time: number;
  readonly frame: number;
  readonly dt: number;
  readonly hud: Hud | null;
  readonly bloom: Bloom | null;
  readonly destroyed: boolean;
  /** True while the frame loop is scheduled. */
  readonly running: boolean;
  exposure: number;
  tonemap: Tonemap;
  background: number[];
  onError: ((error: Error) => void) | null;
  /** Called before every frame (set by run(fn), or directly). */
  onFrame: ((frame: FrameInfo) => void) | null;
  readonly view: Float32Array; readonly proj: Float32Array; readonly viewProj: Float32Array; readonly invViewProj: Float32Array;
  /** Add a layer (idempotent); returns it. */
  add<L extends Layer | object>(layer: L): L;
  /** Remove a layer without destroying it; returns it. */
  remove<L>(layer: L): L;
  /** World-space ray through a point in normalised device coordinates (default: the pointer). */
  ray(ndc?: number[]): { origin: Vec3; dir: Vec3 };
  /** Record and submit one frame. */
  render(): this;
  /** Start or resume the frame loop; update runs before every render (run() without an argument keeps the current one). */
  run(update?: (frame: FrameInfo) => void): this;
  /** Pause the frame loop (run() resumes it). */
  stop(): this;
  /** Advance one frame by hand (tests, offline rendering, frameloop "never"). */
  step(dt?: number): this;
  setBloom(o: boolean | { strength?: number; [k: string]: unknown }): this;
  setHud(o: boolean | { extra?: () => string | string[]; parent?: HTMLElement }): this;
  setPixelRatio(ratio: number): this;
  /** Stop the loop, remove listeners and the HUD, unconfigure the canvas, free the view's textures; { layers: true } also destroys the layers. */
  destroy(opts?: { layers?: boolean }): void;
}

/** Orbit camera with damping: drag rotates, right/shift-drag pans, wheel/pinch zooms. */
export class OrbitCamera {
  constructor(el: HTMLElement, o?: CameraOptions);
  target: number[];
  distance: number;
  yaw: number;
  pitch: number;
  /** Vertical field of view in radians (fov: the same in degrees). */
  fovY: number;
  fov: number;
  near: number; far: number;
  minDistance: number; maxDistance: number; minPitch: number; maxPitch: number;
  autoRotate: number; damping: number; rotateSpeed: number; zoomSpeed: number; panSpeed: number;
  enabled: boolean;
  /** Apply any CameraOptions at once (resets the damping velocity). */
  set(o: CameraOptions): this;
  attach(): this;
  detach(): this;
  update(dt: number): void;
  eye(): Vec3;
}
export class Hud { extra: (() => string | string[]) | null; readonly el: HTMLElement; destroy(): void }

export interface BodiesOptions { groups?: number; [k: string]: unknown }
/** Draws every body of a RigidWorld (boxes, spheres and ray-cast capsules) straight from GPU memory. */
export class Bodies implements RenderLayer {
  constructor(ctx: Context, world: RigidWorld, o?: BodiesOptions);
  set(values: Uniforms): this;
  draw(pass: GPURenderPassEncoderT, view: View): void;
  destroy(): void;
}

/** Render layers: construct with a Context and options, then view.add(layer). destroy() frees what the layer allocated itself. */
export interface RenderLayer extends Layer { set(values: Uniforms): this; destroy(): void }
export class RenderProgram {
  constructor(ctx: Context, desc: Record<string, unknown>);
  ready(): Promise<void>;
  set(values: Uniforms): this;
  bind(resources: Resources): this;
  draw(pass: GPURenderPassEncoderT, view: View, vertexCount: number, instanceCount?: number, uniforms?: Uniforms | null, resources?: Resources): this;
  drawIndexed(pass: GPURenderPassEncoderT, view: View, indexBuffer: unknown, indexCount: number, instanceCount?: number, uniforms?: Uniforms | null, resources?: Resources): this;
}

export interface FullscreenDesc {
  /** WGSL defining fn shade(uv: vec2f, px: vec2f) -> vec4f (or -> FsFrag with writeDepth); cameraRay(uv), worldDepth(p), cam.* and u.* are available. */
  code: string;
  name?: string;
  imports?: string[];
  uniforms?: Record<string, WGSLType>;
  values?: Uniforms;
  defines?: Record<string, number | boolean | string | number[]>;
  buffers?: Record<string, GArray | BufferBinding>;
  textures?: Record<string, unknown>;
  samplers?: Record<string, unknown>;
  writeDepth?: boolean;
  targets?: unknown[];
}
export class Fullscreen implements RenderLayer {
  constructor(ctx: Context, desc: FullscreenDesc);
  ready(): Promise<void>; set(values: Uniforms): this; bind(resources: Resources): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void;
}

interface ParticleSource {
  /** vec4f array holding the particles. */
  data: GArray;
  /** vec4s per particle (default 1) and the vec4 index of the position (default 0). */
  stride?: number;
  position?: number;
  /** vec4 index of the velocity, or -1 (default). */
  velocity?: number;
  /** Particles to draw (default: the whole array); a function is read every frame. */
  count?: number | (() => number);
  scale?: number;
  offset?: number[];
}
export interface PointsOptions extends ParticleSource {
  /** World-space size (default 0.01), never below pixelSize pixels (default 1.5). */
  size?: number;
  pixelSize?: number;
  color?: number[];
  colorBy?: 'constant' | 'speed' | 'w';
  speedScale?: number;
  intensity?: number;
}
export class Points implements RenderLayer { constructor(ctx: Context, o: PointsOptions); readonly count: number; set(values: Uniforms): this; bind(resources: Resources): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export interface SpheresOptions extends ParticleSource {
  radius?: number;
  /** WGSL expression of px: vec4f giving a u32 palette index. */
  material?: string;
  /** WGSL boolean expression of px: draw the particle or not. */
  mask?: string;
  palette?: number[][];
  colorBy?: 'material' | 'speed' | 'height';
  speedScale?: number;
  velScale?: number;
  sun?: number[];
  ambient?: number;
  matMask?: number;
}
export class Spheres implements RenderLayer { constructor(ctx: Context, o: SpheresOptions); readonly count: number; set(values: Uniforms): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export interface GlyphsOptions { pos: GArray; vel: GArray; count?: number | (() => number); size?: number; colorBy?: 'constant' | 'speed' | 'heading'; color?: number[]; speedScale?: number }
export class Glyphs implements RenderLayer { constructor(ctx: Context, o: GlyphsOptions); readonly count: number; set(values: Uniforms): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export interface MeshOptions {
  /** vec4f positions and normals of a grid of vertices, and a u32 index buffer. */
  pos: GArray; nrm: GArray; indices: GPUBufferT; indexCount: number;
  /** [columns, rows] of the vertex grid (for the checker pattern). */
  grid?: number[];
  front?: number[]; back?: number[];
  pattern?: 'checker' | 'plain';
  tiles?: number;
  roughness?: number;
}
export class SurfaceMesh implements RenderLayer { constructor(ctx: Context, o: MeshOptions); set(values: Uniforms): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export interface StageOptions {
  /** Floor height (default 0). */
  height?: number;
  /** Grid cell size (default 0.1). */
  cell?: number;
  groundColor?: number[]; lineColor?: number[]; skyTop?: number[]; skyBottom?: number[];
  fog?: number;
  sun?: number[];
}
export class Stage implements RenderLayer { constructor(ctx: Context, o?: StageOptions); set(values: Uniforms): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export class Lines implements RenderLayer {
  constructor(ctx: Context, o: { points: Float32Array | number[]; color?: number[] });
  set(values: Uniforms): this; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void;
  /** The twelve edges of an axis-aligned box. */
  static box(ctx: Context, min: number[], max: number[], color?: number[]): Lines;
}
export interface SdfSceneOptions {
  /** WGSL defining fn map(p: vec3f) -> vec2f (distance, material id) and optionally fn material(id, p, n) -> Material. */
  code: string;
  imports?: string[];
  uniforms?: Record<string, WGSLType>;
  values?: Uniforms;
  /** Cone-marching tile in pixels (default 4). */
  tile?: number;
  shadows?: boolean;
  ao?: boolean;
  maxDistance?: number;
  sun?: number[];
  sunRadiance?: number[];
  fog?: number;
}
export class SdfScene implements RenderLayer { constructor(ctx: Context, o: SdfSceneOptions); set(values: Uniforms): this; prepare(view: View): void; draw(pass: GPURenderPassEncoderT, view: View): void; destroy(): void }
export interface FluidSurfaceOptions extends Omit<ParticleSource, 'data'> {
  data: GArray | (() => GArray);
  radius?: number;
  mask?: string;
  tint?: number[];
  absorb?: number[];
  refraction?: number;
  filterRadius?: number;
  iterations?: number;
  thicknessScale?: number;
  maxPixels?: number;
  sun?: number[];
}
/** Screen-space fluid surface (a post-scene layer). */
export class FluidSurface implements RenderLayer { constructor(ctx: Context, o: FluidSurfaceOptions); set(values: Uniforms): this; afterScene(view: View): void; destroy(): void }
export interface VolumeOptions { smoke: Smoke; density?: number; emission?: number; tint?: number[]; g?: number; ambient?: number; sun?: number[]; sunColor?: number[] }
/** Volumetric smoke rendering (a post-scene layer). */
export class Volume implements RenderLayer { constructor(ctx: Context, o: VolumeOptions); set(values: Uniforms): this; prepare(): void; afterScene(view: View): void; destroy(): void }
export class Bloom { constructor(ctx: Context, o?: { strength?: number; [k: string]: unknown }); strength: number; run(hdr: unknown): unknown; destroy(): void }

export interface PanelControl {
  label?: string;
  min?: number; max?: number; step?: number;
  value?: number | string | boolean;
  checked?: boolean;
  options?: string[];
  html?: string;
  onInput?: (v: number) => void;
  onChange?: (v: any) => void;
  onClick?: () => void;
}
/** A small control panel (sliders, selects, checkboxes, buttons). */
export function panel(title: string, controls?: PanelControl[], parent?: HTMLElement): { values: Record<string, number | string | boolean>; el: HTMLElement };

export const m4: Record<string, (...args: any[]) => any>;
export const v3: {
  add(a: number[], b: number[]): Vec3; sub(a: number[], b: number[]): Vec3; scale(a: number[], s: number): Vec3;
  dot(a: number[], b: number[]): number; cross(a: number[], b: number[]): Vec3; len(a: number[]): number; norm(a: number[]): Vec3;
  lerp(a: number[], b: number[], t: number): Vec3;
};

// ---------------------------------------------------------------------------------- rigid bodies

export interface RigidWorldOptions {
  /** ≤ 65535 (default 16384). */
  maxBodies?: number;
  maxContacts?: number;
  maxPairs?: number;
  maxJoints?: number;
  maxCommands?: number;
  gravity?: number[];
  /** Solver iterations per substep (default 8). */
  iterations?: number;
  substeps?: number;
  /** Height of an infinite floor plane, or null for none (default 0). */
  floor?: number | null;
  /** World bounds of the broadphase grid (default [[-40,-2,-40],[40,40,40]]). */
  bounds?: [number[], number[]];
  /** Fixed colour budget: a pure function of the inputs, for lockstep / rollback netcode. */
  deterministic?: boolean;
  /** Contact begin/end events (readEvents). */
  events?: boolean;
  maxEvents?: number;
  /** AVBD knobs; the defaults are the tested values. */
  alpha?: number; beta?: number; gamma?: number; kStart?: number; kStartJoint?: number;
  jointMassFloor?: number; contactMassFloor?: number; contactTolerance?: number;
  margin?: number; slop?: number; stickThreshold?: number; geoStiff?: number; postStabilize?: boolean;
  /** Worlds with at most this many bodies solve in one fused dispatch (default 128); true / false force it. */
  fusedSolve?: number | boolean;
}

export interface BodyOptions {
  position?: number[];
  rotation?: Rotation;
  velocity?: number[];
  angularVelocity?: number[];
  density?: number;
  friction?: number;
  static?: boolean;
  /** Moved by moveKinematic/setVelocity; pushes dynamic bodies, is not pushed. */
  kinematic?: boolean;
  /** Reports contacts (events) without responding to them. */
  sensor?: boolean;
  fixedRotation?: boolean;
  gravityScale?: number;
  linearDamping?: number;
  angularDamping?: number;
  /** Collision filter: bodies a and b collide when (a.group & b.mask) and (b.group & a.mask). */
  group?: number;
  mask?: number;
  userData?: number;
}

export type JointType = 'ball' | 'hinge' | 'fixed' | 'distance' | 'slider';
export interface JointOptions {
  type?: JointType;
  a: number;
  /** The other body, or null / -1 for the world. */
  b?: number | null;
  /** World-space anchor (default: b's centre, or a's for the world). */
  anchor?: number[];
  /** Distance joints: the other end. */
  anchorB?: number[];
  /** World-space axis (hinge, slider). */
  axis?: number[];
  /** [lower, upper] in radians (hinge) or metres (slider). */
  limits?: [number, number];
  motor?: { speed: number; maxForce: number };
  /** Distance joints (default: the current distance). */
  length?: number;
  /** Distance joints: resist stretching only. */
  rope?: boolean;
  /** N/m: a spring instead of a rigid joint. */
  stiffness?: number;
  damping?: number;
  /** The joint breaks when its force exceeds this (N). */
  breakForce?: number;
  collideConnected?: boolean;
}

/** A ray-cast result; point and normal are only present on a hit. */
export interface RayHit { hit: boolean; body: number; distance: number; point?: Vec3; normal?: Vec3 }
export interface RigidSnapshot { readonly [k: string]: unknown }
export interface RigidState {
  version: number; frame: number; count: number; contactCount: number; tableSize: number;
  bodies: Float32Array; joints: Float32Array; ext: Float32Array; contacts: Float32Array;
  /** Occupied warm-start table slots: [slot, key, first contact, count] × n. */
  slots: Uint32Array;
  meta: unknown;
}
export interface ForceField { readonly kernel: Kernel; enabled: boolean; set(values: Uniforms): ForceField }

/**
 * GPU rigid bodies (boxes, spheres, capsules) and joints, solved with Augmented Vertex Block
 * Descent. Every query is asynchronous; edits are queued commands applied at the next step.
 */
export class RigidWorld {
  constructor(ctx: Context, o?: RigidWorldOptions);
  readonly ctx: Context;
  readonly maxBodies: number;
  /** Number of body slots in use (removed slots are reused). */
  readonly count: number;
  readonly joints: ReadonlyArray<{ a: number; b: number; type: number; flags: number; enabled: boolean }>;
  gravity: number[];
  iterations: number;
  substeps: number;
  floor: number | null;
  deterministic: boolean;
  stats: { contacts: number; colors: number; pairs: number };
  /** Body state on the GPU: 12 vec4 per body (see BODY_VEC4 in physics/rigid-wgsl.js). */
  readonly B: GArray;

  addBox(o: BodyOptions & { half?: number[] }): number;
  addSphere(o: BodyOptions & { radius: number }): number;
  /** A capsule along its local y axis. */
  addCapsule(o: BodyOptions & { radius: number; halfHeight?: number }): number;
  remove(id: number): void;
  /** Teleport a body (ids that do not exist throw). */
  setPose(id: number, position: number[], rotation?: Rotation | null): void;
  setVelocity(id: number, v: number[], w?: number[]): void;
  applyImpulse(id: number, J: number[], point?: number[]): void;
  applyForce(id: number, F: number[], point?: number[], dt?: number): void;
  applyAngularImpulse(id: number, L: number[]): void;
  /** Move a kinematic body to a target pose over the next step (its velocity follows). */
  moveKinematic(id: number, position: number[], rotation?: Rotation): void;
  setFilter(id: number, group: number, mask: number): void;
  setEnabled(id: number, on: boolean): void;
  setFlags(id: number, set?: number, clear?: number): void;
  setProps(id: number, props?: { gravityScale?: number; linearDamping?: number; angularDamping?: number }): void;

  addJoint(o: JointOptions): number;
  /** Target speed (rad/s or m/s) and force limit; speed null turns the motor off. */
  /** Target speed (rad/s or m/s) and force limit (default: the joint's current limit); speed null turns the motor off. */
  setMotor(joint: number, speed: number | null, maxForce?: number): void;
  /** Limits in rad (hinge) or m (slider), lo ≤ hi. */
  setLimits(joint: number, lo: number, hi: number): void;
  /** Remove the limits. */
  setLimits(joint: number, lo: null): void;
  setJointEnabled(joint: number, on: boolean): void;

  /** WGSL run for every dynamic body before each frame; it adds to `force` / `torque` (world, N and N·m). */
  addForceField(o: { code: string; uniforms?: Record<string, WGSLType>; imports?: string[]; values?: Uniforms }): ForceField;

  /** Upload what was added since the last step (step() calls it). */
  commit(): void;
  /** Remove every body and joint. */
  clear(): void;
  step(frameDt?: number): void;

  readStats(): Promise<{ contacts: number; colors: number; pairs: number }>;
  /** [x, y, z, qx, qy, qz, qw] per body. */
  readPoses(): Promise<Float32Array>;
  readBodies(): Promise<{ position: Float32Array; rotation: Float32Array; velocity: Float32Array; angularVelocity: Float32Array; flags: Uint32Array }>;
  readJoints(): Promise<{ broken: boolean[]; force: Float32Array }>;
  /** Contact begin/end pairs of the last step (b = -1 for the floor). Needs { events: true }. */
  readEvents(): Promise<{ begin: Array<[number, number]>; end: Array<[number, number]> }>;
  rayQuery(maxRays?: number): RayQuery;
  raycast(rays: Array<{ origin: number[]; dir: number[]; maxDistance?: number; mask?: number }>): Promise<RayHit[]>;

  /** 64-bit hash of every body's pose and velocity (hex); equal states give equal hashes. */
  hash(): Promise<string>;
  /** Record the hash computation now, of the live state or of a snapshot, into a u32 array. */
  hashRecord(target?: GArray, snap?: RigidSnapshot | null): GArray;
  /** Capture everything the next step depends on (GPU copies; pass an old snapshot to reuse its buffers). */
  snapshot(into?: RigidSnapshot): RigidSnapshot;
  restore(s: RigidSnapshot): void;
  /** The complete state as typed arrays (save games, network resync) — of the live world or of a snapshot. */
  exportState(snap?: RigidSnapshot | null): Promise<RigidState>;
  /** Restore an exported state exactly (both worlds need the same maxBodies / maxContacts). */
  importState(st: RigidState): void;
  readonly destroyed: boolean;
  readonly maxJoints: number;
  /** Free every GPU buffer the world allocated (snapshots you took are yours to destroy). Idempotent. */
  destroy(): void;
}

/** Batched GPU ray casts; rays can be written by the CPU or by other kernels. */
export class RayQuery {
  constructor(world: RigidWorld, maxRays: number);
  /** vec4f × 2 per ray: (origin, maxDistance), (direction, mask bits). */
  readonly rays: GArray;
  /** vec4f × 2 per ray: (distance or -1, body bits, -, -), (normal, -). */
  readonly hits: GArray;
  set(list: Array<{ origin: number[]; dir: number[]; maxDistance?: number; mask?: number }>): this;
  run(count?: number): this;
  read(count?: number): Promise<RayHit[]>;
  destroy(): void;
}

export function eulerQuat(angles: number[]): Quat;
export const RIGID_SHAPES: { sphere: 0; box: 1; capsule: 2 };
export const RIGID_FLAGS: { kinematic: 1; sensor: 2; disabled: 4; fixedRotation: 8 };
export const JOINT_TYPES: { ball: 0; hinge: 1; fixed: 2; distance: 3; slider: 4 };
export const JOINT_FLAGS: { broken: 1; collide: 2; limit: 4; motor: 8; rope: 16; worldAnchors: 32; disabled: 64 };

// ---------------------------------------------------------------------------------- particles & fields

export interface MaterialDef {
  /** 0 fluid, 1 Neo-Hookean solid (jelly, rubber), 2 snow, 3 sand — or the name ('fluid', 'jelly', 'snow', 'sand'). */
  kind: number | 'fluid' | 'jelly' | 'elastic' | 'snow' | 'sand';
  density?: number;
  /** Fraction (0, 1] of the stiffest stable wave speed. */
  stiffness?: number;
  poisson?: number;
  viscosity?: number;
  hardening?: number; compression?: number; stretch?: number;
  /** Sand: friction angle in degrees. */
  friction?: number;
  /** Linear RGB for the renderers. */
  color?: number[];
  [k: string]: unknown;
}
export const MATERIALS: Record<'water' | 'oil' | 'honey' | 'jelly' | 'rubber' | 'snow' | 'sand', MaterialDef>;
/** Material kinds by name; MaterialDef.kind accepts the number or the name. */
export const MPM_KINDS: { fluid: 0; jelly: 1; elastic: 1; snow: 2; sand: 3 };

export interface MPMOptions {
  /** Particle capacity (default 262144). */
  maxParticles?: number;
  /** Cells along the longest axis (default 64). */
  grid?: number;
  /** World size [x, y, z] or a cube edge (default 1). */
  size?: number[] | number;
  origin?: number[];
  gravity?: number[];
  /** Substeps per 1/60 s frame (default 8): more give stiffer, less compressible materials. */
  substeps?: number;
  timeScale?: number;
  /** Re-sort particles every this many substeps (default 4). */
  sortEvery?: number;
  /** Particles per cell along each axis when filling shapes (default 2, i.e. 8 per cell). */
  ppcAxis?: number;
}

export type MPMShape =
  | { box: [number[], number[]]; material?: string | MaterialDef; velocity?: number[] }
  | { sphere: [number[], number]; material?: string | MaterialDef; velocity?: number[] }
  | { cylinder: [number[], number, number]; material?: string | MaterialDef; velocity?: number[] }
  | { sdf: (p: number[]) => number; bounds: [number[], number[]]; material?: string | MaterialDef; velocity?: number[] }
  | { points: Float32Array | number[]; material?: string | MaterialDef; velocity?: number[] };

/** MLS-MPM continuum simulation (water, oil, honey, jelly, rubber, snow, sand). */
export class MPM {
  constructor(ctx: Context, o?: MPMOptions);
  readonly count: number;
  readonly maxParticles: number;
  /** Cell size (world units) and grid dimensions. */
  readonly dx: number;
  readonly dims: number[];
  readonly size: number[];
  readonly origin: number[];
  readonly bound: number;
  substeps: number;
  timeScale: number;
  sortEvery: number;
  gravity: number[];
  readonly substepDt: number;
  /** The particle buffer (8 vec4 per particle, lattice units); it is swapped by the sort. */
  readonly P: GArray;
  readonly materials: ReadonlyArray<MaterialDef>;
  coupling: FluidRigidCoupling | null;
  /** Fill a region with particles; returns how many were added. */
  add(shape: MPMShape, opts?: { material?: string | MaterialDef; velocity?: number[]; jitter?: number; seed?: number }): number;
  /** Remove every particle (and forget the materials, so a new scene can use up to 8 others). */
  clear(): void;
  /** Two-way coupling with a RigidWorld: bodies are moving boundaries and receive the fluid's momentum. */
  couple(world: RigidWorld, o?: CouplingOptions): FluidRigidCoupling;
  /** End the coupling made by couple(). */
  decouple(): void;
  /** Kinematic colliders re-uploaded every step: { type: 'sphere' | 'box' | 'plane', center, radius | half | normal, velocity, friction }. */
  setColliders(list: Array<Record<string, unknown>>): void;
  /** One frame: `substeps` substeps. */
  step(): void;
  readPositions(): Promise<Float32Array>;
  palette(): number[][];
  fluidMask(): number;
  /** A screen-space water surface for the liquid materials. */
  fluid(opts?: Partial<FluidSurfaceOptions>): Layer & { surface: FluidSurface; set(values: Uniforms): unknown };
  /** Spheres for the particles; solidsOnly skips liquids (pair with fluid()). */
  spheres(opts?: Partial<SpheresOptions> & { solidsOnly?: boolean; size?: number }): Layer & { spheres: Spheres; set(values: Uniforms): unknown };
  destroy(): void;
}

export interface CouplingOptions {
  /** Tangential friction between fluid and bodies (default 0.2). */
  friction?: number;
  /** Boundary band in cells (default 0.5). */
  band?: number;
  /** Bodies with (group & mask) == 0 are ignored by the fluid. */
  mask?: number;
  densityScale?: number;
}
export class FluidRigidCoupling {
  constructor(ctx: Context, sim: MPM, world: RigidWorld, o?: CouplingOptions);
  destroy(): void;
}

export interface NBodyOptions {
  count: number;
  /** Gravitational constant (default 1), Plummer softening (default 0.02), time step (default 0.002). */
  G?: number; softening?: number; dt?: number;
  variant?: unknown;
}
export class NBody {
  constructor(ctx: Context, o: NBodyOptions);
  readonly count: number;
  /** Current positions (double-buffered: re-read it every frame). */
  readonly pos: GArray;
  readonly vel: GArray;
  G: number; softening: number; dt: number;
  galaxies(o?: { seed?: number; [k: string]: unknown }): this;
  plummer(o?: { seed?: number; scale?: number }): this;
  step(n?: number, variant?: unknown): void;
  /** Time every kernel variant on this GPU (cached per adapter); results sorted fastest first. */
  autotune(o?: { cache?: boolean; reps?: number }): Promise<Array<{ variant: { mode: string; wg: number; K: number }; ms: number; gips: number }>>;
  destroy(): void;
}

export interface ClothCollider { type: 'sphere' | 'box' | 'plane' | string; center?: number[]; radius?: number; half?: number[]; normal?: number[]; [k: string]: unknown }
export interface ClothOptions {
  /** Particles across and down (default 128 × 128). */
  width?: number; height?: number;
  /** World size [x, y] (default [1.6, 1.6]) and the top-left corner. */
  size?: number[]; origin?: number[];
  /** Pinned particles: 'top' (default), 'top-sparse', 'corners', 'none' or (i, j) => boolean. */
  pin?: 'top' | 'top-sparse' | 'corners' | 'none' | ((i: number, j: number) => boolean);
  substeps?: number;
  /** XPBD compliances (m/N): 0 is inextensible. */
  stretch?: number; shear?: number; bend?: number;
  gravity?: number[];
  damping?: number;
  wind?: number[];
  windNoise?: number;
  colliders?: ClothCollider[];
  friction?: number;
  thickness?: number;
}
export class Cloth {
  constructor(ctx: Context, o?: ClothOptions);
  readonly W: number; readonly H: number; readonly count: number;
  readonly pos: GArray; readonly nrm: GArray; readonly indices: GPUBufferT; readonly indexCount: number;
  substeps: number; wind: number[]; windNoise: number; gravity: number[]; damping: number; friction: number;
  colliders: ClothCollider[];
  compliance: { stretch: number; shear: number; bend: number };
  /** Floor height, or null for none. */
  floorY: number | null;
  origin: number[];
  reset(pin?: ClothOptions['pin']): void;
  step(dt?: number): void;
  destroy(): void;
}

export interface SmokeSource { center: number[]; radius: number; density?: number; temperature?: number; velocity?: number[] }
export interface SmokeOptions {
  /** Cells along the longest axis (default 64). */
  res?: number;
  size?: number[];
  origin?: number[];
  /** Pressure solver sweeps (default 24) and SOR factor (default 1.8). */
  iterations?: number;
  omega?: number;
  vorticity?: number;
  buoyancy?: number;
  weight?: number;
  cooling?: number;
  dissipation?: number;
  sources?: SmokeSource[];
}
export class Smoke {
  constructor(ctx: Context, o?: SmokeOptions);
  readonly dims: number[]; readonly dx: number; readonly size: number[]; readonly origin: number[];
  iterations: number; omega: number; vorticity: number; buoyancy: number; weight: number; cooling: number; dissipation: number;
  sources: SmokeSource[];
  reset(): void;
  step(dt?: number): void;
  destroy(): void;
}

export interface BoidsOptions {
  count?: number;
  /** Half-size of the cube they live in (default 1). */
  bounds?: number;
  /** Perception radius (default 0.06). */
  radius?: number;
  maxSpeed?: number; minSpeed?: number;
  separation?: number; alignment?: number; cohesion?: number;
  maxNeighbors?: number;
  wall?: number;
}
export class Boids {
  constructor(ctx: Context, o?: BoidsOptions);
  readonly count: number;
  readonly pos: GArray; readonly vel: GArray;
  params: { radius: number; maxSpeed: number; minSpeed: number; separation: number; alignment: number; cohesion: number; maxNeighbors: number;
    wall: number; predator: number[]; predatorRadius: number; predatorForce: number };
  /** Scatter the flock (positions and headings) from a seed. */
  randomize(seed?: number): this;
  step(dt?: number): void;
  destroy(): void;
}

// ---------------------------------------------------------------------------------- neural networks

export type Activation = 'none' | 'relu' | 'leaky' | 'tanh' | 'sigmoid' | 'sin' | 'gelu';
export interface GemmConfig { BM: number; BN: number; BK: number; TM: number; TN: number; prefetch?: boolean }
export interface GemmOptions {
  M: number; N: number; K: number;
  transA?: boolean; transB?: boolean;
  alpha?: number; beta?: number;
  /** Row vector added to every row of C. */
  bias?: GArray;
  act?: Activation;
  cfg?: GemmConfig;
  lda?: number; ldb?: number; ldc?: number;
  /** Split-K factor (default: automatic for small outputs with a long K). */
  split?: number;
}
/** C = act(α op(A) op(B) + β C + bias), f32, row-major. */
export function gemm(ctx: Context, A: GArray, B: GArray, C: GArray, o: GemmOptions): GArray;
/** Time every tile configuration on this GPU and remember the fastest (cached in localStorage). */
export function autotuneGemm(ctx: Context, o?: { n?: number; reps?: number; cache?: boolean }): Promise<{ best: GemmConfig; cached?: boolean; results?: Array<{ cfg: GemmConfig; ms: number; tflops: number }> }>;

export interface Param { w: GArray; g: GArray }
export interface MLPOptions {
  /** Layer widths including input and output, e.g. [2, 64, 64, 3]. */
  layers: number[];
  act?: Activation;
  out?: Activation;
  /** Fourier-feature bands for the input encoding. */
  fourier?: number;
  maxBatch?: number;
  seed?: number;
}
export class MLP {
  constructor(ctx: Context, o: MLPOptions);
  readonly ctx: Context;
  readonly maxBatch: number;
  readonly layers: ReadonlyArray<nn.Dense>;
  /** The Fourier-feature encoder, when `fourier` was given. */
  readonly encoder: nn.FourierFeatures | null;
  /** Write inputs here ([batch × layers[0]]). */
  readonly input: GArray;
  /** The last forward pass's outputs ([batch × out]). */
  readonly output: GArray;
  readonly params: Param[];
  readonly parameterCount: number;
  forward(X?: GArray, batch?: number): GArray;
  /** Back-propagate dLoss/dOutput; overwrites every layer's gradients (params[k].g). */
  backward(dY: GArray, batch: number): void;
  /** Free every buffer of the network. */
  destroy(): void;
}
export class Adam {
  constructor(ctx: Context, params: Param[], o?: { lr?: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number; clip?: number });
  lr: number; beta1: number; beta2: number; eps: number; weightDecay: number; clip: number;
  /** Steps taken (for bias correction). */
  readonly t: number;
  step(): void;
  /** Free the moment buffers (the parameters belong to the network). */
  destroy(): void;
}
export class Population {
  /** count must be even: agents come in antithetic pairs (θ + σε, θ − σε). */
  constructor(ctx: Context, o: { layers: number[]; count: number; sigma?: number; lr?: number; act?: Activation; out?: Activation; seed?: number });
  readonly count: number;
  /** The shared parameter vector. */
  readonly theta: GArray;
  sigma: number;
  lr: number;
  destroy(): void;
  readonly parameterCount: number;
  readonly generation: number;
  /** Every agent's network on its own observation row: X [count × in] → Y [count × out]. */
  forward(X: GArray, Y: GArray, count?: number): GArray;
  /** The unperturbed mean network for every row. */
  forwardMean(X: GArray, Y: GArray, count?: number): GArray;
  /** Evolution-strategies update from per-agent fitness (higher is better). */
  update(fitness: GArray | Float32Array): Promise<{ mean: number; best: number }>;
}

// nn re-exports the top-level names (a namespace cannot use `export { }`, so through aliases)
declare const gemm_: typeof gemm;
declare const autotuneGemm_: typeof autotuneGemm;
declare const MLP_: typeof MLP;
declare const Adam_: typeof Adam;
declare const Population_: typeof Population;
type MLP_ = MLP; type Adam_ = Adam; type Population_ = Population;
type Activation_ = Activation; type GemmConfig_ = GemmConfig; type GemmOptions_ = GemmOptions; type Param_ = Param;

export namespace nn {
  export const gemm: typeof gemm_;
  export const autotuneGemm: typeof autotuneGemm_;
  export const MLP: typeof MLP_; export type MLP = MLP_;
  export const Adam: typeof Adam_; export type Adam = Adam_;
  export const Population: typeof Population_; export type Population = Population_;
  export type Activation = Activation_; export type GemmConfig = GemmConfig_; export type GemmOptions = GemmOptions_; export type Param = Param_;
  export const ACTIVATIONS: Record<Activation, { f: string; df: string }>;
  export const GEMM_CONFIGS: GemmConfig[];
  export function gemmKernel(ctx: Context, cfg: GemmConfig, o?: { transA?: boolean; transB?: boolean; act?: Activation; bias?: boolean; accumulate?: boolean; split?: boolean }): Kernel;
  export class Dense {
    constructor(ctx: Context, inF: number, outF: number, o?: { act?: Activation; init?: 'he' | 'xavier' | 'siren'; seed?: number; first?: boolean; omega?: number });
    readonly in: number; readonly out: number; readonly act: Activation;
    readonly W: GArray; readonly b: GArray; readonly gW: GArray; readonly gb: GArray; readonly params: Param[];
    /** Re-initialise the weights (uniform He / Xavier / SIREN with ω₀ folded in) and zero the biases. */
    init(kind: 'he' | 'xavier' | 'siren', seed: number, first?: boolean, omega?: number): void;
    destroy(): void;
  }
  export class FourierFeatures {
    constructor(ctx: Context, inF: number, bands?: number);
    readonly in: number; readonly bands: number; readonly out: number;
    forward(X: GArray, Y: GArray, batch: number): GArray;
    destroy(): void;
  }
  export class MSELoss {
    constructor(ctx: Context, maxElems: number);
    /** dLoss/dY of the last run. */
    readonly grad: GArray;
    /** Records the loss and returns dLoss/dY (count ≤ maxElems). */
    run(Y: GArray, T: GArray, count: number): GArray;
    readLoss(): Promise<number>;
    destroy(): void;
  }
  export class SGD { constructor(ctx: Context, params: Param[], o?: { lr?: number; momentum?: number }); lr: number; momentum: number; step(): void; destroy(): void }
}

// ---------------------------------------------------------------------------------- networking

export interface Channel { send(message: unknown): void; onmessage: ((message: any) => void) | null; close(): void }
/** Anything that relays signalling messages between peers (see BroadcastSignaler, WebSocketSignaler). */
export interface Signaler { send(payload: unknown, to?: string): void | Promise<void>; onmessage: ((message: { from: string; to?: string; payload: any }) => void) | null; close?(): void }
export interface ChannelPair { unreliable: Channel; reliable: Channel }

export interface Transcript {
  players: number; inputSize: number; delay: number; hashEvery: number; dt: number;
  inputs: number[][][];
  /** tick → state hash after that tick. */
  hashes: Record<number, string>;
  /** Head of the input hash chain. */
  chain: string;
  ticks: number;
}

/** Anything a RollbackSession can drive (RigidWorld in deterministic mode qualifies). */
export interface RollbackSim {
  ctx: Context;
  snapshot(into?: any): any;
  restore(s: any): void;
  step(dt: number): void;
  hashRecord(target: GArray, snap: any): unknown;
  /** Needed for resync (the authority exports, the others import); RigidWorld has both. */
  exportState?(snap?: any): Promise<unknown>;
  importState?(state: any): void;
}

export interface RollbackOptions {
  sim: RollbackSim;
  /** Game state kept outside the simulation. */
  game?: { save(): unknown; load(state: any): void } | null;
  channel: ChannelPair;
  players?: number;
  localPlayer: number;
  /** Turn every player's input for a tick into simulation commands. */
  applyInputs(tick: number, inputs: Uint8Array[]): void;
  inputSize?: number;
  /** Input delay in ticks (default 2). */
  delay?: number;
  maxRollback?: number;
  dt?: number;
  hashEvery?: number;
  /** Inputs repeated in every packet (default 8). */
  redundancy?: number;
  onDesync?(tick: number, local: string, remote: string): void;
  /** On a hash mismatch the authority sends its last confirmed state and the other peer adopts it (default true). */
  resync?: boolean;
  /** The player whose state wins a resync (default 0). */
  authority?: number;
  /** Ticks of confirmed inputs kept so that a late resync state can still be replayed (default 600). */
  inputHistory?: number;
  /** Ticks of GPU work allowed in flight before advance() waits a tick (default 3). */
  maxInFlight?: number;
}

export namespace net {
  /** A simulated network for tests and demos: latency, jitter, loss and duplication. */
  export class LoopbackNetwork {
    /** codec (default true): messages go through the wire encoding, as over WebRTC, instead of a structured clone. */
    constructor(o?: { latency?: number; jitter?: number; loss?: number; duplicate?: number; seed?: number; codec?: boolean });
    latency: number; jitter: number; loss: number; duplicate: number;
    readonly stats: { sent: number; delivered: number; dropped: number; bytes: number };
    /** Two connected endpoints; channel.close() stops its deliveries. */
    connect(): [ChannelPair, ChannelPair];
  }
  /** Wrap an RTCDataChannel (or anything with send(), onmessage, readyState) as a Channel with the wire encoding and chunking. */
  export function dataChannel(dc: unknown): Channel & { attach(dc: unknown): void; close(): void };
  /** The wire encoding: JSON with typed arrays as base64 of their bytes (exact bits). */
  export function encode(message: unknown): string;
  export function decode(text: string): any;
  /** Signalling between tabs of one browser. */
  export class BroadcastSignaler implements Signaler { constructor(room: string); readonly id: string; onmessage: Signaler['onmessage']; send(payload: unknown, to?: string): void; close(): void }
  /** Signalling through a relay that forwards { room, from, to, payload } JSON messages. */
  export class WebSocketSignaler implements Signaler {
    constructor(url: string, room: string);
    readonly id: string; readonly ready: Promise<void>; onmessage: Signaler['onmessage'];
    send(payload: unknown, to?: string): Promise<void>; close(): void;
  }
  /** A WebRTC peer with an unreliable (unordered, no retransmits) and a reliable data channel. */
  export class RTCPeer {
    constructor(signaler: BroadcastSignaler | WebSocketSignaler, o?: { initiator?: boolean; iceServers?: unknown[] });
    /** Resolves (with the peer) when both channels are open; rejects if the connection fails or is closed first. */
    readonly ready: Promise<RTCPeer>;
    readonly pc: RTCPeerConnection;
    readonly channels: ChannelPair;
    close(): void;
  }
  /** GGPO-style rollback netcode with a proof-carrying transcript. */
  export class RollbackSession {
    constructor(o: RollbackOptions);
    readonly tick: number;
    /** Ticks ≤ this have every player's input (final). */
    readonly confirmedTick: number;
    readonly chain: string;
    readonly hashes: Map<number, { local?: string; remote?: string; reported?: boolean }>;
    readonly stats: { rollbacks: number; rolledBackTicks: number; stalls: number; desyncs: number; resyncs: number; statesSent: number; statesReceived: number;
      exportErrors: number; gpuWaits: number; hashesRecorded: number; hashesDropped: number; hashesStale: number; ping: number; inSyncTick: number;
      sent: number; received: number; [diagnostic: string]: unknown };
    /** States adopted from the authority so far. */
    readonly epoch: number;
    /** Replaces the built-in resync: called with { type: 'state', tick, epoch, state, game } messages. */
    stateHandler?: (message: any) => void;
    /** Advance one tick with this frame's local input; returns 0 while stalled for a lagging peer. */
    advance(localInput?: Uint8Array): number;
    transcript(): Transcript;
  }
  /** Re-execute a transcript and check every state hash and the chain head. */
  export function verifyTranscript<S extends { step(dt: number): void; hash(): Promise<string> }>(tr: Transcript, makeSim: () => S | Promise<S>, applyInputs: (sim: S, tick: number, inputs: Uint8Array[]) => void):
    Promise<{ ok: boolean; ticks: number; checkedHashes: number; chain?: string; firstMismatch: unknown }>;
  export function chainHash(prev: string, ...parts: Array<number | string | ArrayLike<number>>): string;
}

// ---------------------------------------------------------------------------------- verification

export interface Verdict { ok: boolean; [evidence: string]: unknown }
export interface Law { name: string; statement: string; check(gpu: Context, SI: unknown): Promise<Verdict> }

export namespace verify {
  export function checkScan(ctx: Context, input: GArray, output: GArray, n: number): Promise<Verdict>;
  export function checkSorted(ctx: Context, keys: GArray, n: number): Promise<Verdict>;
  export function checkPermutation(ctx: Context, a: GArray, b: GArray, n: number): Promise<Verdict>;
  /** Freivalds' certificate for C = op(A) op(B); false accept ≤ 2^-trials. */
  /** Fresh random vectors on every call (pass seed for reproducible vectors, which forfeits the bound against adversarial inputs). */
  export function freivalds(ctx: Context, A: GArray, B: GArray, C: GArray, o: { M: number; N: number; K: number; transA?: boolean; transB?: boolean; trials?: number; tol?: number; seed?: number }): Promise<Verdict & { maxRelative: number; trials: number; falseAcceptBound: number }>;
  export function checkColoring(world: RigidWorld): Promise<Verdict>;
  export function checkContacts(world: RigidWorld, o?: { maxPenetration?: number; coneTol?: number }): Promise<Verdict>;
  export function checkMPM(sim: MPM, tol?: number): Promise<Verdict>;
  export function checkGradient(net: MLP, X: GArray, T: GArray, batch: number, o?: { samples?: number; eps?: number; tol?: number; seed?: number }): Promise<Verdict>;
  export function checkDeterminism(makeWorld: () => RigidWorld | Promise<RigidWorld>, steps?: number, every?: number): Promise<Verdict>;
  export function checkReplay(world: RigidWorld, steps?: number): Promise<Verdict>;
  /** The executable specification (LAWS.md). */
  export const LAWS: Law[];
  export function proveAll(gpu: Context, SI: unknown, only?: string[]): Promise<Array<{ name: string; statement: string; ok: boolean; ms: number; evidence: Verdict }>>;
}

// ---------------------------------------------------------------------------------- three.js interop

export namespace three {
  /** A three.js WebGPURenderer sharing ShaderIntel's GPUDevice. */
  export function renderer(THREE: any, gpu: Context, opts?: Record<string, unknown>): Promise<any>;
  /** The GPUBuffer behind a three.js storage attribute. */
  export function bufferOf(renderer: any, attribute: any): GPUBufferT;
  /**
   * Instanced meshes of every rigid body, posed by a kernel straight from GPU memory each frame.
   * Call update() after world.step(), then gpu.flush(), then render.
   */
  export function bodies(o: { THREE: any; TSL: any; renderer: any; world: RigidWorld; material?: (shape: 'box' | 'sphere' | 'capsule') => any; castShadow?: boolean; receiveShadow?: boolean; colors?: (instance: any) => any; groups?: number }):
    { object: any; meshes: { box: any; sphere: any; capsule: any }; attribute: any; update(): void; dispose(): void };
  /** Instanced particles from any GArray of vec4 positions (or a function returning the current one). */
  export function particles(o: { THREE: any; TSL: any; renderer: any; source: GArray | (() => GArray); gpu?: Context; stride?: number; count?: number | (() => number); max?: number;
    scale?: number; offset?: number[]; radius?: number; color?: number[] | unknown; material?: any }):
    { object: any; attribute: any; update(): void; dispose(): void };
}

// only the names marked `export` above are part of the API
export {};
