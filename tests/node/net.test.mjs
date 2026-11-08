import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encode, decode, LoopbackNetwork, chainHash } from '../../src/net/index.js';

test('the wire encoding keeps bits and the values JSON drops', () => {
  const m = { nan: NaN, negZero: -0, inf: -Infinity, big: 2n ** 70n, map: new Map([[1, new Set(['a'])]]), date: new Date(12345),
    f32: new Float32Array([1, -0, NaN]), u32: new Uint32Array([0xffffffff]), view: new DataView(new ArrayBuffer(4)), buf: new ArrayBuffer(3),
    tricky: { __t: 'u', __ta: 'Float32Array' }, list: [1, undefined, 3] };
  const r = decode(encode(m));
  assert.ok(Number.isNaN(r.nan));
  assert.ok(Object.is(r.negZero, -0));
  assert.equal(r.inf, -Infinity);
  assert.equal(r.big, 2n ** 70n);
  assert.ok(r.map.get(1).has('a'));
  assert.equal(r.date.getTime(), 12345);
  assert.ok(Object.is(r.f32[1], -0) && Number.isNaN(r.f32[2]));
  assert.equal(r.u32[0], 0xffffffff);
  assert.equal(r.view.byteLength, 4);
  assert.equal(r.buf.byteLength, 3);
  assert.deepEqual(r.tricky, { __t: 'u', __ta: 'Float32Array' });
  assert.equal(r.list.length, 3);
  assert.equal(r.list[1], undefined);
});

test('decode refuses to construct anything but typed arrays', () => {
  assert.throws(() => decode('{"x":{"__ta":"BroadcastChannel","b":""}}'), /not a typed array/);
  assert.throws(() => decode('{"x":{"__ta":"Float32Array","b":"AAA="}}'), /whole number/);
});

test('the simulated reliable channel delivers in order even with sub-millisecond gaps', async () => {
  const net = new LoopbackNetwork({ latency: 5, jitter: 4, seed: 3 });
  const [a, b] = net.connect();
  const got = [];
  b.reliable.onmessage = (m) => got.push(m.k);
  for (let k = 0; k < 300; k++) a.reliable.send({ k });
  await new Promise((r) => setTimeout(r, 120));
  assert.equal(got.length, 300);
  assert.ok(got.every((v, k) => v === k), 'reordered');
});

test('closing a simulated channel stops its deliveries', async () => {
  const net = new LoopbackNetwork({ latency: 10, jitter: 0 });
  const [a, b] = net.connect();
  let n = 0;
  b.unreliable.onmessage = () => n++;
  a.unreliable.send({ x: 1 });
  a.unreliable.close();
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(n, 0);
});

test('the transcript hash chain changes with a single input bit', () => {
  const a = chainHash('0000000000000000', 1, new Uint8Array([0]));
  const b = chainHash('0000000000000000', 1, new Uint8Array([1]));
  assert.match(a, /^[0-9a-f]{16}$/);
  assert.notEqual(a, b);
});
