// shaderintel-react without a browser: server rendering must not touch WebGPU, and hooks used
// outside <Canvas> must say so. The package is bundled for Node with esbuild first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));

async function load() {
  const dir = mkdtempSync(join(tmpdir(), 'si-react-'));
  const entry = join(dir, 'entry.jsx');
  writeFileSync(entry, `
    import { renderToString } from 'react-dom/server';
    import { createElement } from 'react';
    import * as R from 'shaderintel-react';
    export { renderToString, createElement, R };`);
  await build({ entryPoints: [entry], outfile: join(dir, 'out.mjs'), bundle: true, format: 'esm', platform: 'node', jsx: 'automatic', logLevel: 'error',
    alias: { 'shaderintel-react': join(root, 'packages/react/src/index.ts'), shaderintel: join(root, 'src/index.js') },
    nodePaths: [join(root, 'node_modules')],   // React bundled in: one instance for elements and hooks
    banner: { js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);" } });
  return import(pathToFileURL(join(dir, 'out.mjs')).href);
}

test('<Canvas> renders on the server: a sized wrapper with a canvas and the loading node, no WebGPU calls', async () => {
  const { renderToString, R, createElement: h } = await load();
  const html = renderToString(h(R.Canvas, { className: 'scene', loading: h('p', null, 'Loading GPU…') }, h(R.Stage), h(R.Physics, null, h(R.Box))));
  assert.match(html, /^<div class="scene" style="position:relative;width:100%;height:100%;overflow:hidden">/);
  assert.match(html, /<canvas style="display:block;width:100%;height:100%;touch-action:none"><\/canvas>/);
  assert.match(html, /Loading GPU…/);
  assert.doesNotMatch(html, /Stage|Physics/, 'children render only once the GPU is ready');
});

test('hooks and components outside <Canvas> throw a clear error', async () => {
  const { renderToString, R, createElement: h } = await load();
  assert.throws(() => renderToString(h(R.Stage)), /must be used inside <Canvas>/);
  const Probe = () => { R.useWorld(); return null; };
  assert.throws(() => renderToString(h(Probe)), /inside <Physics>/);
});
