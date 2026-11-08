// Declarative render layers. Props that are compiled into the shader (code, imports, uniform
// declarations, data layout) re-create the layer when they change; everything else is a uniform
// update applied in place. Every component's ref is the underlying ShaderIntel object.
import { forwardRef, useImperativeHandle } from 'react';
import type { DependencyList } from 'react';
import { Lines } from 'shaderintel';
import type {
  Context, FullscreenDesc, GArray, Glyphs as GlyphsLayer, Points as PointsLayer, SdfScene as SdfLayer, Spheres as SpheresLayer,
  Stage as StageLayer, Fullscreen as FullscreenLayer, Uniforms, View, WGSLType,
} from 'shaderintel';
import { defined, stableKey, useLayer, useLiveValues } from './hooks.js';

type V3 = [number, number, number] | number[];

// ------------------------------------------------------------------ generic

export interface LayerProps<L extends object> {
  /** Build the layer: (gpu, view) => anything with draw / prepare / afterScene / overlay. */
  create: (gpu: Context, view: View) => L;
  /** Re-create the layer when these change. */
  deps?: DependencyList;
}
/** Any custom layer: <Layer create={(gpu) => gpu.points({ data })} deps={[data]} /> */
export const Layer = forwardRef(function Layer<L extends object>({ create, deps = [] }: LayerProps<L>, ref: React.Ref<L>) {
  const layer = useLayer(create, deps);
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
}) as <L extends object>(props: LayerProps<L> & { ref?: React.Ref<L> }) => null;

// ------------------------------------------------------------------ stage

export interface StageProps {
  /** Floor height (default 0). */
  height?: number;
  /** Grid cell size (default 0.1; every 10th line is brighter). */
  cell?: number;
  groundColor?: V3; lineColor?: V3; skyTop?: V3; skyBottom?: V3;
  fog?: number;
  sun?: V3;
}
/** Sky gradient and an infinite anti-aliased grid floor. */
export const Stage = forwardRef<StageLayer, StageProps>(function Stage(props, ref) {
  const layer = useLayer((gpu) => gpu.stage(defined({ ...props })), []);
  useLiveValues(layer, defined({ ...props }));
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

// ------------------------------------------------------------------ particles

interface ParticleLayout {
  /** vec4f array holding the particles. Changing it re-creates the layer. */
  data: GArray;
  /** vec4s per particle (default 1) and the vec4 index of the position (default 0). */
  stride?: number;
  position?: number;
  /** vec4 index of the velocity (enables colorBy="speed"); -1 for none. */
  velocity?: number;
  /** Particles to draw: a number or a function read every frame (default: all of `data`). */
  count?: number | (() => number);
  /** World transform of the stored positions. */
  scale?: number;
  offset?: V3;
}

export interface PointsProps extends ParticleLayout {
  /** World-space size (default 0.01), never smaller than pixelSize pixels (default 1.5). */
  size?: number;
  pixelSize?: number;
  /** Linear HDR colour for colorBy="constant". */
  color?: V3;
  colorBy?: 'constant' | 'speed' | 'w';
  speedScale?: number;
  intensity?: number;
}
const COLOR_BY_POINTS = { constant: 0, speed: 1, w: 2 } as const;
/** Additive glowing sprites for very many particles (pair with bloom). */
export const Points = forwardRef<PointsLayer, PointsProps>(function Points(props, ref) {
  const { data, stride, position, velocity, count, size, pixelSize, color, colorBy, speedScale, intensity, scale, offset } = props;
  const layer = useLayer((gpu) => gpu.points(defined({ data, stride, position, velocity, count, size, pixelSize, color, colorBy, speedScale, intensity, scale, offset })),
    [data, stride, position, velocity]);
  (layer as unknown as { o: { count?: unknown } }).o.count = count;
  useLiveValues(layer, defined({ size, minPixels: pixelSize, color, colorBy: colorBy && COLOR_BY_POINTS[colorBy], speedScale, intensity, scale, offset }));
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

export interface SpheresProps extends ParticleLayout {
  radius?: number;
  /** WGSL expression of `px: vec4f` giving a palette index (u32), e.g. "u32(px.w)". */
  material?: string;
  /** WGSL boolean expression of `px` deciding whether a particle is drawn. */
  mask?: string;
  /** Up to 8 linear RGB colours. */
  palette?: number[][];
  colorBy?: 'material' | 'speed' | 'height';
  speedScale?: number;
  sun?: V3;
  ambient?: number;
}
const COLOR_BY_SPHERES = { material: 0, speed: 1, height: 2 } as const;
/** Ray-cast sphere impostors with exact depth and normals. */
export const Spheres = forwardRef<SpheresLayer, SpheresProps>(function Spheres(props, ref) {
  const { data, stride, position, velocity, count, material, mask, palette, colorBy, radius, scale, offset, speedScale, sun, ambient } = props;
  const layer = useLayer((gpu) => gpu.spheres(defined({ data, stride, position, velocity, count, material, mask, palette, colorBy, radius, scale, offset, speedScale, sun, ambient })),
    [data, stride, position, velocity, material, mask]);
  (layer as unknown as { o: { count?: unknown } }).o.count = count;
  const pal: Record<string, number[]> = {};
  palette?.slice(0, 8).forEach((c, k) => { pal['pal' + k] = [c[0], c[1], c[2], 1]; });
  useLiveValues(layer, defined({ radius, scale, offset, speedScale, ambient, colorBy: colorBy && COLOR_BY_SPHERES[colorBy], ...pal }));
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

export interface GlyphsProps {
  /** vec4f positions and velocities (e.g. from <Boids> or your own kernels). */
  pos: GArray;
  vel: GArray;
  count?: number | (() => number);
  size?: number;
  colorBy?: 'constant' | 'speed' | 'heading';
  color?: V3;
  speedScale?: number;
}
/** Velocity-aligned darts for agents (boids, crowds). */
export const Glyphs = forwardRef<GlyphsLayer, GlyphsProps>(function Glyphs(props, ref) {
  const { pos, vel, count, size, colorBy, color, speedScale } = props;
  const layer = useLayer((gpu) => gpu.glyphs(defined({ pos, vel, count, size, colorBy, color, speedScale })), [pos, vel]);
  (layer as unknown as { o: { count?: unknown } }).o.count = count;
  useLiveValues(layer, defined({ size, color, speedScale, colorBy: colorBy && { constant: 0, speed: 1, heading: 2 }[colorBy] }));
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

// ------------------------------------------------------------------ shaders

export interface FullscreenProps {
  /** WGSL defining `fn shade(uv: vec2f, px: vec2f) -> vec4f`; `cameraRay(uv)`, `cam.time` and `u.*` are available. */
  code: string;
  imports?: string[];
  /** Uniform declarations: name → WGSL type. */
  uniforms?: Record<string, WGSLType>;
  /** Uniform values (updated in place when they change). */
  values?: Uniforms;
  defines?: Record<string, number | boolean | string>;
  /** Write depth (shade returns FsFrag(color, depth)) so the layer can sit behind 3D layers. */
  writeDepth?: boolean;
}
/** A Shadertoy-style full-screen shader backed by the WGSL library. */
export const Fullscreen = forwardRef<FullscreenLayer, FullscreenProps>(function Fullscreen(props, ref) {
  const { code, imports, uniforms, values, defines, writeDepth } = props;
  const layer = useLayer((gpu) => gpu.fullscreen({ code, imports, uniforms, values, defines, writeDepth } as FullscreenDesc),
    [code, stableKey(imports ?? []), stableKey(uniforms ?? {}), stableKey(defines ?? {}), !!writeDepth]);
  useLiveValues(layer, values ?? {});
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

export interface SdfSceneProps {
  /** WGSL defining `fn map(p: vec3f) -> vec2f` (distance, material id) and optionally `fn material(id, p, n) -> Material`. */
  code: string;
  imports?: string[];
  uniforms?: Record<string, WGSLType>;
  values?: Uniforms;
  /** Cone-marching tile size in pixels (default 4). */
  tile?: number;
  shadows?: boolean;
  ao?: boolean;
  maxDistance?: number;
  sun?: V3;
  sunRadiance?: V3;
  fog?: number;
}
/** A signed-distance-field scene: write map(), get cone marching, soft shadows, AO and PBR. */
export const SdfScene = forwardRef<SdfLayer, SdfSceneProps>(function SdfScene(props, ref) {
  const { code, imports, uniforms, values, tile, shadows, ao, maxDistance, sun, sunRadiance, fog } = props;
  const layer = useLayer((gpu) => gpu.sdf(defined({ code, imports, uniforms, values, tile, shadows, ao, maxDistance, sun, sunRadiance, fog })),
    [code, stableKey(imports ?? []), stableKey(uniforms ?? {}), tile]);
  const norm = sun ? (() => { const l = Math.hypot(sun[0], sun[1], sun[2]) || 1; return [sun[0] / l, sun[1] / l, sun[2] / l]; })() : undefined;
  useLiveValues(layer, defined({ ...(values ?? {}), shadows: shadows === undefined ? undefined : +shadows, ao: ao === undefined ? undefined : +ao,
    maxDist: maxDistance, sun: norm, sunRadiance, fog }));
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});

// ------------------------------------------------------------------ helpers

export interface BoxLinesProps {
  min: V3;
  max: V3;
  /** Linear RGBA (default a soft grey). */
  color?: number[];
}
/** The outline of an axis-aligned box (bounds of a simulation, a tank, a room). */
export const BoxLines = forwardRef<object, BoxLinesProps>(function BoxLines({ min, max, color }, ref) {
  const layer = useLayer((gpu) => Lines.box(gpu, min, max, color), [stableKey([min, max, color ?? null])]);
  useImperativeHandle(ref, () => layer, [layer]);
  return null;
});
