// <Canvas>: a WebGPU canvas with a ShaderIntel View (HDR, orbit camera, tonemapping, bloom, HUD)
// and a frame loop. Children are rendered once the GPU is ready and can use every hook.
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import type { CSSProperties, HTMLAttributes, ReactNode } from 'react';
import type { CameraOptions, Context, InitOptions, Tonemap, View } from 'shaderintel';
import { ShaderIntelContext, acquireGPU, createSubscribers } from './context.js';
import type { FrameState, RootState } from './context.js';
import { stableKey, useIsomorphicLayoutEffect } from './hooks.js';

export interface CanvasProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'onError'> {
  children?: ReactNode;
  /** Use this GPU context (it is not destroyed on unmount). Default: a device shared by every <Canvas> with the same `init` options. */
  gpu?: Context;
  /** Options for SI.init(): powerPreference, subgroups, f16, timestamps, or an external { device }. */
  init?: InitOptions;
  /** Orbit camera: target, distance, yaw, pitch, fov, near, far, autoRotate, ... Applied again whenever its values change. */
  camera?: CameraOptions;
  /** Mouse / touch orbit controls (default true). */
  controls?: boolean;
  /** Bloom: true, false, or { strength }. */
  bloom?: boolean | { strength?: number; [k: string]: unknown };
  /** Stats overlay: FPS and per-region GPU times. */
  hud?: boolean;
  tonemap?: Tonemap;
  exposure?: number;
  /** Linear RGBA clear colour of the HDR scene. */
  background?: [number, number, number, number] | number[];
  /** Fixed device-pixel ratio (default: the screen's, capped by maxPixelRatio). */
  pixelRatio?: number;
  maxPixelRatio?: number;
  /** Pause the frame loop (frame callbacks and rendering). */
  paused?: boolean;
  /** "always" (default): render every animation frame. "never": only when you call advance() from useShaderIntel(). */
  frameloop?: 'always' | 'never';
  /** Rendered instead of the canvas when WebGPU is unavailable or fails: a node, or a function of the error. */
  fallback?: ReactNode | ((error: Error) => ReactNode);
  /** Rendered over the canvas while the GPU initialises. */
  loading?: ReactNode;
  /** Called once the GPU context and the view exist. */
  onCreated?: (root: RootState) => void;
  /** Called with initialisation errors and errors thrown by frame callbacks (the loop stops). */
  onError?: (error: Error) => void;
}

type Status = { kind: 'loading' } | { kind: 'ready'; root: RootState } | { kind: 'error'; error: Error };

const wrapStyle: CSSProperties = { position: 'relative', width: '100%', height: '100%', overflow: 'hidden' };
const canvasStyle: CSSProperties = { display: 'block', width: '100%', height: '100%', touchAction: 'none' };

function DefaultFallback({ error }: { error: Error }) {
  const unsupported = (error as { code?: string }).code === 'unsupported' || (error as { code?: string }).code === 'no-adapter';
  return (
    <div role="alert" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24, textAlign: 'center',
      font: '14px/1.5 system-ui, sans-serif', color: '#c9d1d9', background: '#0d1117' }}>
      <div>
        <strong>{unsupported ? 'WebGPU is not available' : 'The GPU scene failed'}</strong>
        <div style={{ opacity: 0.75, marginTop: 6, maxWidth: 560 }}>
          {unsupported ? 'Use Chrome or Edge 113+, Safari 26+ or Firefox 141+ (Windows), over https or localhost.' : error.message}
        </div>
      </div>
    </div>
  );
}

/**
 * A WebGPU canvas. It fills its parent (give the parent a size), creates the GPU context and a
 * View, runs the frame loop and renders `children` — components and hooks from this package —
 * once everything is ready. The ref exposes the root state ({ gpu, view, canvas, advance }).
 */
export const Canvas = forwardRef<RootState | null, CanvasProps>(function Canvas(props, ref) {
  const { children, gpu: gpuProp, init: initOpts, camera, controls = true, bloom, hud, tonemap, exposure, background,
    pixelRatio, maxPixelRatio, paused = false, frameloop = 'always', fallback, loading, onCreated, onError,
    style, ...rest } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<Status>({ kind: 'loading' });
  const latest = useRef(props);
  latest.current = props;
  const initKey = stableKey(initOpts ?? {});

  // ---- create the GPU context and the view (again only if the context or the init options change)
  useEffect(() => {
    let cancelled = false;
    let view: View | null = null;
    let release: (() => void) | null = null;
    setStatus((s) => (s.kind === 'loading' ? s : { kind: 'loading' }));
    const fail = (e: unknown) => {
      const error = e instanceof Error ? e : new Error(String(e));
      if (!cancelled) setStatus({ kind: 'error', error });
      latest.current.onError?.(error);
    };
    (async () => {
      let gpu: Context;
      if (gpuProp) gpu = gpuProp;
      else {
        const lease = acquireGPU(initOpts);
        release = lease.release;
        gpu = await lease.promise;
      }
      if (cancelled || !canvasRef.current) return;
      const p = latest.current;
      // bloom and the HUD are applied by the live-option effects below
      view = gpu.view(canvasRef.current, {
        camera: p.camera, controls: p.controls !== false, tonemap: p.tonemap, exposure: p.exposure,
        background: p.background, pixelRatio: p.pixelRatio, maxPixelRatio: p.maxPixelRatio,
        onError: (e: Error) => fail(e),
      });
      const subscribers = createSubscribers();
      const v = view;
      const tick = (s: { dt: number; time: number; frame: number }) => subscribers.run({ ...s, view: v, gpu } as FrameState);
      const root: RootState = {
        gpu, view: v, canvas: canvasRef.current,
        subscribe: subscribers.subscribe,
        advance(dt = 1 / 60) { v.step(dt); },
      };
      v.onFrame = tick;   // called by view.run()'s loop and by view.step()
      setStatus({ kind: 'ready', root });
      p.onCreated?.(root);
    })().catch(fail);
    return () => {
      cancelled = true;
      view?.destroy();
      release?.();
    };
  }, [gpuProp, initKey]);

  const root = status.kind === 'ready' ? status.root : null;
  useImperativeHandle(ref, () => root as RootState, [root]);

  // ---- live options: applied to the existing view, no re-creation
  const cameraKey = stableKey(camera ?? null);
  useIsomorphicLayoutEffect(() => { if (root && camera) root.view.camera.set(camera); }, [root, cameraKey]);
  useIsomorphicLayoutEffect(() => {
    if (!root) return;
    if (controls) root.view.camera.attach(); else root.view.camera.detach();
  }, [root, controls]);
  const bloomKey = stableKey(bloom ?? false);
  useIsomorphicLayoutEffect(() => { if (root) root.view.setBloom(bloom ?? false); }, [root, bloomKey]);
  useIsomorphicLayoutEffect(() => { if (root && !!root.view.hud !== !!hud) root.view.setHud(!!hud); }, [root, hud]);
  useIsomorphicLayoutEffect(() => {
    if (!root) return;
    const v = root.view;
    if (tonemap) v.tonemap = tonemap;
    if (exposure !== undefined) v.exposure = exposure;
    if (background) v.background = [...background];
  }, [root, tonemap, exposure, stableKey(background ?? null)]);
  useIsomorphicLayoutEffect(() => { if (root && pixelRatio) root.view.setPixelRatio(pixelRatio); }, [root, pixelRatio]);

  // ---- the frame loop
  useEffect(() => {
    if (!root) return;
    if (paused || frameloop === 'never') root.view.stop();
    else root.view.run();
    return () => { root.view.stop(); };
  }, [root, paused, frameloop]);

  const shownFallback = useMemo(() => {
    if (status.kind !== 'error') return null;
    if (typeof fallback === 'function') return fallback(status.error);
    return fallback ?? <DefaultFallback error={status.error} />;
  }, [status, fallback]);

  return (
    <div {...rest} style={{ ...wrapStyle, ...style }}>
      <canvas ref={canvasRef} style={canvasStyle} />
      {root && <ShaderIntelContext.Provider value={root}>{children}</ShaderIntelContext.Provider>}
      {status.kind === 'loading' && loading}
      {shownFallback}
    </div>
  );
});
