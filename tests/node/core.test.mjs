// CPU-only unit tests (no GPU): run with `node --test tests/node/` — this is what CI runs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as SI from '../../src/index.js';

test('the library imports without a browser (server rendering) and reports WebGPU as unsupported', async () => {
  assert.equal(SI.isSupported(), false);
  await assert.rejects(SI.init(), (e) => e instanceof SI.SIError && e.code === 'unsupported');
});

test('struct layouts follow the WGSL host-shareable rules', () => {
  const L = SI.structLayout('P', { n: 'u32', a: 'vec3f', b: 'f32', m: 'mat3x3f', c: 'vec2f', h: 'f16' });
  const off = Object.fromEntries(L.fields.map((f) => [f.name, f.offset]));
  assert.deepEqual(off, { n: 0, a: 16, b: 28, m: 32, c: 80, h: 88 });
  assert.equal(L.size, 96);
  assert.equal(SI.layoutOf('vec3f').stride, 16);
  assert.equal(SI.layoutOf('mat4x4f').size, 64);
  assert.throws(() => SI.layoutOf('vec5f'), SI.SIError);
});

test('f16 conversion round-trips and rounds to nearest even', () => {
  for (const v of [0, 1, -2.5, 65504, 6.1e-5, 0.333]) assert.ok(Math.abs(SI.fromHalf(SI.toHalf(v)) - v) <= Math.max(1e-3 * Math.abs(v), 1e-7));
  assert.equal(SI.toHalf(Infinity), 0x7c00);
  assert.ok(Number.isNaN(SI.fromHalf(SI.toHalf(NaN))));
});

test('WGSL literals and the module linker', () => {
  assert.equal(SI.literal(1, 'f32'), '1.0');
  assert.equal(SI.literal(3, 'u32'), '3u');
  assert.throws(() => SI.literal(NaN), SI.SIError);
  SI.defineModule('test-a', { code: 'fn a() -> f32 { return 1.0; }' });
  SI.defineModule('test-b', { deps: ['test-a'], code: 'enable f16;\nfn b() -> f32 { return a(); }' });
  const code = SI.link('// #import test-b\nfn main() {}');
  assert.ok(code.startsWith('enable f16;'), 'directives are hoisted');
  assert.equal(code.match(/fn a\(\)/g).length, 1, 'each module once');
  assert.ok(code.indexOf('fn a()') < code.indexOf('fn main()'));
  assert.throws(() => SI.link('', ['no-such-module']), (e) => e instanceof SI.SIError && /no-such-module/.test(e.message));
  assert.ok(SI.moduleNames().includes('noise') && SI.moduleNames().includes('sdf'));
});

test('math helpers: quaternions from Euler angles are unit length', () => {
  for (const e of [[0, 0, 0], [1, 2, 3], [-0.5, 0.25, 4]]) {
    const q = SI.eulerQuat(e);
    assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-12);
  }
  assert.deepEqual(SI.v3.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1]);
});
