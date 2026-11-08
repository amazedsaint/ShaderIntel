import { defineModule } from '../core/wgsl.js';

// Integer hashing. Float "fract(sin(x) * 43758.5453)" hashes are both slower
// (sin is a multi-instruction SFU op) and break down at large inputs and on
// GPUs with low-precision sin. Integer hashes are exact, portable and fast.
// pcg / pcg3d / pcg4d: Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT 2020.
// lowbias32: Chris Wellons' hash-prospector (bias ~0.17, two multiplies).
defineModule('hash', {
  doc: 'PCG-family and lowbias32 integer hashes, float conversions, PCG random numbers, low-discrepancy sequences.',
  code: /* wgsl */`
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn pcg2d(p: vec2u) -> vec2u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  return v ^ (v >> vec2u(16u));
}
fn pcg3d(p: vec3u) -> vec3u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn pcg4d(p: vec4u) -> vec4u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  v = v ^ (v >> vec4u(16u));
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  return v;
}
fn lowbias32(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u; x *= 0x7feb352du;
  x ^= x >> 15u; x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}

/** u32 -> uniform float in [0, 1) using the top 24 bits. */
fn u2f(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }
fn u2f2(x: vec2u) -> vec2f { return vec2f(x >> vec2u(8u)) * (1.0 / 16777216.0); }
fn u2f3(x: vec3u) -> vec3f { return vec3f(x >> vec3u(8u)) * (1.0 / 16777216.0); }
fn u2f4(x: vec4u) -> vec4f { return vec4f(x >> vec4u(8u)) * (1.0 / 16777216.0); }

// Float-coordinate hashes in [0, 1). Inputs are hashed by their bit pattern.
// pcg2d's x output skips the final mixing step its y output gets: over horizontally adjacent
// pixels it is correlated (χ² 1073 on 256 pair buckets, where ≲ 330 is independent), so the 2D
// hashes use y, and hash22 derives its second value from both (χ² 226 and 258).
fn hash11(p: f32) -> f32 { return u2f(pcg(bitcast<u32>(p))); }
fn hash21(p: vec2f) -> f32 { return u2f(pcg2d(bitcast<vec2u>(p)).y); }
fn hash22(p: vec2f) -> vec2f { let v = pcg2d(bitcast<vec2u>(p)); return u2f2(vec2u(v.y, pcg(v.x ^ v.y))); }
fn hash31(p: vec3f) -> f32 { return u2f(pcg3d(bitcast<vec3u>(p)).x); }
fn hash33(p: vec3f) -> vec3f { return u2f3(pcg3d(bitcast<vec3u>(p))); }
fn hash44(p: vec4f) -> vec4f { return u2f4(pcg4d(bitcast<vec4u>(p))); }
// Integer-lattice hashes (for cells / ids).
fn ihash2(p: vec2i) -> u32 { return pcg2d(bitcast<vec2u>(p)).y; }
fn ihash3(p: vec3i) -> u32 { return pcg3d(bitcast<vec3u>(p)).x; }

// ---- PCG random number generator (state in a function-scope u32)
fn rngSeed(a: u32, b: u32) -> u32 { return pcg(a ^ pcg(b + 0x9e3779b9u)); }
fn randU32(s: ptr<function, u32>) -> u32 {
  let state = *s;
  *s = state * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand(s: ptr<function, u32>) -> f32 { return u2f(randU32(s)); }
fn rand2(s: ptr<function, u32>) -> vec2f { return vec2f(rand(s), rand(s)); }
fn rand3(s: ptr<function, u32>) -> vec3f { return vec3f(rand(s), rand(s), rand(s)); }
/** Uniform direction on the unit sphere. */
fn randDir(s: ptr<function, u32>) -> vec3f {
  let u = rand2(s);
  let z = 1.0 - 2.0 * u.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let a = 6.28318530718 * u.y;
  return vec3f(r * cos(a), r * sin(a), z);
}
/** Uniform point inside the unit ball. */
fn randBall(s: ptr<function, u32>) -> vec3f { return randDir(s) * pow(rand(s), 1.0 / 3.0); }
/** Standard normal pair (Box–Muller). */
fn randNormal2(s: ptr<function, u32>) -> vec2f {
  let u = rand2(s);
  let r = sqrt(-2.0 * log(max(u.x, 1e-12)));
  let a = 6.28318530718 * u.y;
  return r * vec2f(cos(a), sin(a));
}
/** Cosine-weighted hemisphere sample around +z. */
fn randCosHemi(s: ptr<function, u32>) -> vec3f {
  let u = rand2(s);
  let r = sqrt(u.x);
  let a = 6.28318530718 * u.y;
  return vec3f(r * cos(a), r * sin(a), sqrt(max(0.0, 1.0 - u.x)));
}

// ---- low-discrepancy sequences
/** Roberts R2 sequence: best-known 2D additive recurrence. */
fn r2seq(n: u32) -> vec2f { return fract(vec2f(0.5) + f32(n) * vec2f(0.754877666246693, 0.569840290998053)); }
/** Interleaved gradient noise (Jimenez 2014): cheap per-pixel dither. */
fn ign(px: vec2f) -> f32 { return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715)))); }
`,
});
