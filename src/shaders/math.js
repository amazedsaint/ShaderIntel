import { defineModule } from '../core/wgsl.js';

defineModule('math', {
  doc: 'Constants, rotations, quaternions, orthonormal bases, small helpers.',
  code: /* wgsl */`
const PI = 3.14159265358979;
const TAU = 6.28318530717959;
const INV_PI = 0.318309886183791;
const SQRT2 = 1.41421356237310;

fn sq(x: f32) -> f32 { return x * x; }
fn max3(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
fn min3(v: vec3f) -> f32 { return min(v.x, min(v.y, v.z)); }
fn dot2(v: vec2f) -> f32 { return dot(v, v); }
fn dot3(v: vec3f) -> f32 { return dot(v, v); }
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 { return c + (x - a) * (d - c) / (b - a); }
/** Quintic smoothstep (C2 continuous). */
fn smootherstep(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
fn safeNormalize(v: vec3f) -> vec3f { let l = dot(v, v); return select(vec3f(0.0), v * inverseSqrt(l), l > 1e-20); }

/** 2D rotation matrix (counter-clockwise). */
fn rot2(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }

/** Rotation about a unit axis (Rodrigues). */
fn rotAxis(axis: vec3f, a: f32) -> mat3x3f {
  let c = cos(a); let s = sin(a); let t = 1.0 - c;
  let x = axis.x; let y = axis.y; let z = axis.z;
  return mat3x3f(
    t * x * x + c,     t * x * y + s * z, t * x * z - s * y,
    t * x * y - s * z, t * y * y + c,     t * y * z + s * x,
    t * x * z + s * y, t * y * z - s * x, t * z * z + c);
}

/** Orthonormal basis around n (Duff et al. 2017, branchless). Columns: t, b, n. */
fn basis(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}

// ---- quaternions (x, y, z, w) with w the scalar part
fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn qconj(q: vec4f) -> vec4f { return vec4f(-q.xyz, q.w); }
/** Rotate v by unit quaternion q (15 mul/add form). */
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn qaxis(axis: vec3f, a: f32) -> vec4f { return vec4f(axis * sin(0.5 * a), cos(0.5 * a)); }
fn qmat(q: vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y));
}
`,
});
