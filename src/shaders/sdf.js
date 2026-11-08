import { defineModule } from '../core/wgsl.js';

// Signed distance functions. Exact where possible (|∇d| = 1), otherwise a
// bound (marked). Conventions: primitives are centred at the origin; transform
// the query point, not the shape. Operators take/return distances.
defineModule('sdf', {
  deps: ['math'],
  doc: 'SDF primitives (3D/2D), boolean and smooth operators, domain operators.',
  code: /* wgsl */`
// ---------------------------------------------------------------- 3D primitives
fn sdSphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }
fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max3(q), 0.0);
}
fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
  let q = abs(p) - b + r;
  return length(max(q, vec3f(0.0))) + min(max3(q), 0.0) - r;
}
fn sdBoxFrame(p0: vec3f, b: vec3f, e: f32) -> f32 {
  let p = abs(p0) - b;
  let q = abs(p + e) - e;
  return min(min(
    length(max(vec3f(p.x, q.y, q.z), vec3f(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3f(q.x, p.y, q.z), vec3f(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3f(q.x, q.y, p.z), vec3f(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0));
}
/** Torus in the xz-plane: t = (major radius, minor radius). */
fn sdTorus(p: vec3f, t: vec2f) -> f32 { return length(vec2f(length(p.xz) - t.x, p.y)) - t.y; }
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
/** Vertical capped cylinder: half-height h, radius r. */
fn sdCylinder(p: vec3f, h: f32, r: f32) -> f32 {
  let d = abs(vec2f(length(p.xz), p.y)) - vec2f(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
/** Vertical capped cone: half-height h, bottom radius r1, top radius r2. */
fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
fn sdPlane(p: vec3f, n: vec3f, h: f32) -> f32 { return dot(p, n) + h; }
/** Octahedron |x|+|y|+|z| = s (bound). */
fn sdOctahedron(p: vec3f, s: f32) -> f32 {
  let q = abs(p);
  return (q.x + q.y + q.z - s) * 0.57735027;
}
/** Ellipsoid with radii r (bound, good near the surface). */
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}
/** Hexagonal prism along z: h = (radius, half-length). */
fn sdHexPrism(p0: vec3f, h: vec2f) -> f32 {
  let k = vec3f(-0.8660254, 0.5, 0.57735);
  var p = abs(p0);
  let t = 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
  p = vec3f(p.xy - t, p.z);
  let d = vec2f(length(p.xy - vec2f(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x), p.z - h.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
/** Chain link: half-length le, ring radius r1, thickness r2. */
fn sdLink(p: vec3f, le: f32, r1: f32, r2: f32) -> f32 {
  let q = vec3f(p.x, max(abs(p.y) - le, 0.0), p.z);
  return length(vec2f(length(q.xy) - r1, q.z)) - r2;
}
/** Infinite cylinder along y through c.xy (xz plane), radius c.z. */
fn sdInfCylinder(p: vec3f, c: vec3f) -> f32 { return length(p.xz - c.xy) - c.z; }

// ---------------------------------------------------------------- 2D primitives
fn sdCircle(p: vec2f, r: f32) -> f32 { return length(p) - r; }
fn sdBox2(p: vec2f, b: vec2f) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}
/** Rounded box with per-corner radii r = (top-right, bottom-right, top-left, bottom-left). */
fn sdRoundBox2(p: vec2f, b: vec2f, r4: vec4f) -> f32 {
  let r2 = select(r4.zw, r4.xy, p.x > 0.0);
  let r = select(r2.y, r2.x, p.y > 0.0);
  let q = abs(p) - b + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - r;
}
fn sdSegment2(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}
fn sdEquilateralTriangle(p0: vec2f, r: f32) -> f32 {
  let k = sqrt(3.0);
  var p = vec2f(abs(p0.x) - r, p0.y + r / k);
  if (p.x + k * p.y > 0.0) { p = vec2f(p.x - k * p.y, -k * p.x - p.y) / 2.0; }
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
fn sdHexagon(p0: vec2f, r: f32) -> f32 {
  let k = vec3f(-0.866025404, 0.5, 0.577350269);
  var p = abs(p0);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2f(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}

// ---------------------------------------------------------------- operators
fn opUnion(a: f32, b: f32) -> f32 { return min(a, b); }
/** a minus b. */
fn opSubtract(a: f32, b: f32) -> f32 { return max(a, -b); }
fn opIntersect(a: f32, b: f32) -> f32 { return max(a, b); }
/** Quadratic polynomial smooth minimum; k = blend radius in distance units. */
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let kk = 4.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  return min(a, b) - h * h * kk * 0.25;
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
/** Cubic smooth minimum (C2). */
fn sminCubic(a: f32, b: f32, k: f32) -> f32 {
  let kk = 6.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  return min(a, b) - h * h * h * kk * (1.0 / 6.0);
}
/** Smooth minimum that also returns the blend factor (0 -> a, 1 -> b) for mixing materials. */
fn sminMix(a: f32, b: f32, k: f32) -> vec2f {
  let kk = 4.0 * k;
  let h = max(kk - abs(a - b), 0.0) / max(kk, 1e-9);
  let m = h * h * 0.5;
  let s = m * kk * 0.5;
  return select(vec2f(b - s, 1.0 - m), vec2f(a - s, m), a < b);
}
fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 { return smin(a, b, k); }
fn opSmoothSubtract(a: f32, b: f32, k: f32) -> f32 { return smax(a, -b, k); }
fn opSmoothIntersect(a: f32, b: f32, k: f32) -> f32 { return smax(a, b, k); }
fn opRound(d: f32, r: f32) -> f32 { return d - r; }
fn opOnion(d: f32, t: f32) -> f32 { return abs(d) - t; }
/** Union of two (distance, material) pairs. */
fn opU(a: vec2f, b: vec2f) -> vec2f { return select(b, a, a.x < b.x); }

// ---------------------------------------------------------------- domain operators
/** Infinite repetition with period s (returns the local coordinate). */
fn opRepeat(p: vec3f, s: vec3f) -> vec3f { return p - s * round(p / s); }
/** Repetition limited to cells -l..l. */
fn opRepeatLimited(p: vec3f, s: f32, l: vec3f) -> vec3f { return p - s * clamp(round(p / s), -l, l); }
fn opTwist(p: vec3f, k: f32) -> vec3f {
  let c = cos(k * p.y); let s = sin(k * p.y);
  let q = mat2x2f(c, s, -s, c) * p.xz;
  return vec3f(q.x, p.y, q.y);
}
fn opBend(p: vec3f, k: f32) -> vec3f {
  let c = cos(k * p.x); let s = sin(k * p.x);
  let q = mat2x2f(c, s, -s, c) * p.xy;
  return vec3f(q, p.z);
}
fn opElongate(p: vec3f, h: vec3f) -> vec3f { return p - clamp(p, -h, h); }
fn opSymX(p: vec3f) -> vec3f { return vec3f(abs(p.x), p.yz); }
/** Revolve a 2D profile around y at radius o: evaluate sd2D(opRevolve(p, o)). */
fn opRevolve(p: vec3f, o: f32) -> vec2f { return vec2f(length(p.xz) - o, p.y); }
/** Extrude a 2D distance d along z with half-depth h. */
fn opExtrude(p: vec3f, d: f32, h: f32) -> f32 {
  let w = vec2f(d, abs(p.z) - h);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0)));
}
`,
});
