// The state every component inside a <Canvas> shares, the frame-callback registry, and the
// device pool that lets several canvases (and React StrictMode's mount → unmount → mount) reuse
// one GPUDevice instead of creating and destroying devices.
import { createContext, useContext } from 'react';
import { init } from 'shaderintel';
import type { Context, InitOptions, View } from 'shaderintel';

/** What a frame callback receives. `dt` is the frame time in seconds (clamped to 1/20). */
export interface FrameState {
  dt: number;
  time: number;
  frame: number;
  view: View;
  gpu: Context;
}
export type FrameCallback = (state: FrameState) => void;

/** The root of a <Canvas>: the GPU context, its view and the frame loop. */
export interface RootState {
  gpu: Context;
  view: View;
  canvas: HTMLCanvasElement;
  /** Register a frame callback (lower priority runs first). Returns the unsubscribe function. */
  subscribe(callback: { current: FrameCallback }, priority?: number): () => void;
  /** Run every frame callback and render one frame now (for frameloop="never"). */
  advance(dt?: number): void;
}

export const ShaderIntelContext = createContext<RootState | null>(null);

/** The root state of the enclosing <Canvas>; throws a helpful error outside one. */
export function useShaderIntel(): RootState {
  const root = useContext(ShaderIntelContext);
  if (!root) throw new Error('shaderintel-react: this hook or component must be used inside <Canvas>.');
  return root;
}

/** The GPU context (ShaderIntel's `gpu`) of the enclosing <Canvas>. */
export function useGPU(): Context { return useShaderIntel().gpu; }

/** The View (camera, pointer, layers) of the enclosing <Canvas>. */
export function useView(): View { return useShaderIntel().view; }

// ------------------------------------------------------------------ frame loop

interface Subscriber { callback: { current: FrameCallback }; priority: number; order: number }

/** The frame-callback list of one root, kept sorted by (priority, subscription order). */
export function createSubscribers() {
  let list: Subscriber[] = [];
  let order = 0;
  return {
    subscribe(callback: { current: FrameCallback }, priority = 0) {
      const s: Subscriber = { callback, priority, order: order++ };
      list = [...list, s].sort((a, b) => a.priority - b.priority || a.order - b.order);
      return () => { list = list.filter((x) => x !== s); };
    },
    run(state: FrameState) {
      for (const s of list) s.callback.current(state);
    },
  };
}

// ------------------------------------------------------------------ device pool

interface PoolEntry { promise: Promise<Context>; refs: number; timer: ReturnType<typeof setTimeout> | null }
const pool = new Map<string, PoolEntry>();

/** How long an unused shared device is kept before it is destroyed (re-mounts within it reuse it). */
export const DEVICE_LINGER_MS = 1500;

/**
 * Acquire a shared GPU context for these init options. Every acquire must be paired with one
 * release(); the device is destroyed DEVICE_LINGER_MS after the last release, unless re-acquired.
 */
export function acquireGPU(opts: InitOptions = {}): { promise: Promise<Context>; release(): void } {
  if ((opts as { device?: unknown }).device) {
    // an external device is never pooled (and init() never destroys it)
    const promise = init(opts);
    return { promise, release() { promise.then((g) => g.destroy(), () => {}); } };
  }
  const key = JSON.stringify(opts, Object.keys(opts).sort());
  let e = pool.get(key);
  if (!e) {
    const entry: PoolEntry = { promise: init(opts), refs: 0, timer: null };
    pool.set(key, entry);
    entry.promise.then(
      (gpu) => { gpu.lost.then(() => { if (pool.get(key) === entry) pool.delete(key); }); },
      () => { if (pool.get(key) === entry) pool.delete(key); },
    );
    e = entry;
  }
  const entry = e;
  if (entry.timer) { clearTimeout(entry.timer); entry.timer = null; }
  entry.refs++;
  let released = false;
  return {
    promise: entry.promise,
    release() {
      if (released) return;
      released = true;
      if (--entry.refs > 0) return;
      entry.timer = setTimeout(() => {
        entry.timer = null;
        if (entry.refs > 0 || pool.get(key) !== entry) return;
        pool.delete(key);
        entry.promise.then((g) => g.destroy(), () => {});
      }, DEVICE_LINGER_MS);
    },
  };
}
