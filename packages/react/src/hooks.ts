// Hooks that create GPU objects with the right lifetime: made during render (so the value is
// there on the first render and can feed other hooks), destroyed after unmount. Destruction is
// deferred by a tick and cancelled if the component re-mounts at once, which is what React
// StrictMode (mount → unmount → mount) and fast refresh do; an object destroyed while a subtree
// was hidden is re-created when it shows again.
import { useEffect, useLayoutEffect, useReducer, useRef } from 'react';
import type { DependencyList } from 'react';
import type { ArrayOptions, Context, GArray, Kernel, KernelDesc, View, WGSLType } from 'shaderintel';
import { useShaderIntel } from './context.js';
import type { FrameCallback } from './context.js';

/** useLayoutEffect in the browser, useEffect on the server (no warning during SSR). */
export const useIsomorphicLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

function sameDeps(a: DependencyList, b: DependencyList) {
  return a.length === b.length && a.every((x, k) => Object.is(x, b[k]));
}

interface Box<T> { value: T; deps: DependencyList; dead: boolean; timer: ReturnType<typeof setTimeout> | null }

/**
 * Create a value during render and dispose of it after the component unmounts or `deps` change.
 * The building block of every other hook here.
 */
export function useDisposable<T>(create: () => T, dispose: (value: T) => void, deps: DependencyList): T {
  const [, rerender] = useReducer((x: number) => x + 1, 0);
  const box = useRef<Box<T> | null>(null);
  if (box.current === null || box.current.dead || !sameDeps(box.current.deps, deps)) {
    box.current = { value: create(), deps, dead: false, timer: null };
  }
  const current = box.current;
  useEffect(() => {
    if (current.timer) { clearTimeout(current.timer); current.timer = null; }
    if (current.dead) { rerender(); return; }   // destroyed while hidden: render again to re-create it
    return () => {
      current.timer = setTimeout(() => {
        current.timer = null;
        current.dead = true;
        try { dispose(current.value); } catch (e) { console.error(e); }
      }, 0);
    };
  }, [current]);
  return current.value;
}

const destroyOf = (x: unknown) => { (x as { destroy?: () => void } | null)?.destroy?.(); };

/**
 * Create any ShaderIntel object (or anything with destroy()) bound to the component's lifetime:
 *   const grid = useResource((gpu) => new SpatialGrid(gpu, { ... }), [cellSize]);
 */
export function useResource<T>(create: (gpu: Context, view: View) => T, deps: DependencyList = [], dispose: (value: T) => void = destroyOf): T {
  const { gpu, view } = useShaderIntel();
  return useDisposable(() => create(gpu, view), dispose, [gpu, view, ...deps]);
}

/**
 * Run `callback` before every frame is rendered (simulation steps, kernel dispatches, camera
 * moves). Lower `priority` runs first. The callback may change between renders freely.
 */
export function useFrame(callback: FrameCallback, priority = 0): void {
  const root = useShaderIntel();
  const ref = useRef(callback);
  useIsomorphicLayoutEffect(() => { ref.current = callback; });
  useEffect(() => root.subscribe(ref, priority), [root, priority]);
}

/**
 * A typed GPU array that lives as long as the component. Passing data uploads it; passing new data
 * of the same length later uploads that (a different length re-creates the array).
 *   const pos = useArray('vec4f', 100_000);
 *   const ids = useArray('u32', myUint32Array);
 */
export function useArray(type: WGSLType, lengthOrData: number | ArrayLike<number>, opts?: ArrayOptions): GArray {
  const gpu = useShaderIntel().gpu;
  const isData = typeof lengthOrData !== 'number';
  const length = isData ? (lengthOrData as ArrayLike<number>).length : (lengthOrData as number);
  const uploaded = useRef<{ array: GArray | null; data: unknown }>({ array: null, data: null });
  const array = useDisposable(() => {
    const a = gpu.array(type, lengthOrData, opts);
    uploaded.current = { array: a, data: isData ? lengthOrData : null };
    return a;
  }, destroyOf, [gpu, type, length, opts?.usage ?? 0, opts?.label ?? '']);
  useEffect(() => {
    if (!isData) return;
    const u = uploaded.current;
    if (u.array === array && u.data === lengthOrData) return;
    array.write(lengthOrData as ArrayLike<number>);
    uploaded.current = { array, data: lengthOrData };
  }, [array, lengthOrData, isData]);
  return array;
}

/**
 * A compute kernel, compiled once (and again when `deps` change). Bind arrays from useArray in
 * `desc.buffers` and list them in `deps` if they can be re-created.
 *   const step = useKernel({ buffers: { pos }, uniforms: { dt: 'f32' }, code: '...' }, [pos]);
 *   useFrame(({ dt }) => step.run(N, { dt }));
 */
export function useKernel(desc: KernelDesc, deps: DependencyList = []): Kernel {
  const gpu = useShaderIntel().gpu;
  return useDisposable(() => gpu.kernel(desc), () => {}, [gpu, desc.code, desc.shared ?? '', ...deps]);
}

/**
 * Add a render layer to the view for the component's lifetime:
 *   useLayer((gpu) => gpu.points({ data: pos, size: 0.01 }), [pos]);
 * The layer is removed (and destroyed, if it has destroy()) on unmount or when deps change.
 */
export function useLayer<L extends object>(create: (gpu: Context, view: View) => L, deps: DependencyList = []): L {
  const { gpu, view } = useShaderIntel();
  const layer = useDisposable(() => create(gpu, view), destroyOf, [gpu, view, ...deps]);
  useIsomorphicLayoutEffect(() => {
    view.add(layer);
    return () => { if (!view.destroyed) view.remove(layer); };
  }, [view, layer]);
  return layer;
}

/** Apply `values` with `target.set(values)` whenever they change by value (not by identity). */
export function useLiveValues(target: { set(values: Record<string, unknown>): unknown } | null, values: Record<string, unknown>): void {
  const key = stableKey(values);
  useIsomorphicLayoutEffect(() => {
    if (target && Object.keys(values).length) target.set(values);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, key]);
}

/** A string that changes when `x` changes by value (numbers, strings, arrays, plain objects). */
export function stableKey(x: unknown): string {
  return JSON.stringify(x, (_k, v) => (typeof v === 'function' ? `fn:${v.toString().length}` : ArrayBuffer.isView(v) ? Array.from(v as unknown as ArrayLike<number>) : v)) ?? '';
}

/** Drop undefined entries so they don't overwrite a layer's defaults. */
export function defined<T extends object>(o: T): T {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) out[k] = v;
  return out as T;
}
