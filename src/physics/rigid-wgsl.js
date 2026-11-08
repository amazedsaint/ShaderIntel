// WGSL for the rigid-body solver (physics/rigid.js): data layouts, linearised
// contacts, joint constraint rows, the register-resident 6×6 Newton solve,
// narrowphase manifolds (sphere, box, capsule, floor) and ray tests.
import { defineModule } from '../core/wgsl.js';
import '../shaders/math.js';
import '../shaders/linalg.js';
import '../shaders/hash.js';

export const BODY_VEC4 = 12;
export const CONTACT_VEC4 = 6;
export const JOINT_VEC4 = 13;
export const SHAPES = { sphere: 0, box: 1, capsule: 2 };
export const FLAGS = { kinematic: 1, sensor: 2, disabled: 4, fixedRotation: 8 };
export const JOINT_TYPES = { ball: 0, hinge: 1, fixed: 2, distance: 3, slider: 4 };
export const JOINT_FLAGS = { broken: 1, collide: 2, limit: 4, motor: 8, rope: 16, worldAnchors: 32, disabled: 64 };

defineModule('rigid', {
  deps: ['math', 'hash', 'linalg'],
  code: /* wgsl */`
struct Body {
  x: vec4f,      // position; w = bounding radius
  q: vec4f,      // orientation quaternion (x, y, z, w)
  v: vec4f,      // linear velocity; w = friction coefficient
  w: vec4f,      // angular velocity (world); w = previous velocity along gravity
  x0: vec4f,     // pose at the start of the step
  q0: vec4f,
  y: vec4f,      // inertial target
  qy: vec4f,
  size: vec4f,   // box: half extents; sphere: (r, r, r); capsule: (r, half length, r); w = shape
  mass: vec4f,   // (m, Ixx, Iyy, Izz) in the body frame; m = 0 -> static or kinematic
  props: vec4f,  // (gravity scale, linear damping, angular damping, -)
  info: vec4u,   // (collision group bits, collision mask bits, flags, user data)
}
const F_KINEMATIC = 1u;
const F_SENSOR = 2u;
const F_DISABLED = 4u;
const F_FIXED_ROT = 8u;
struct Contact {
  ab: vec4u,     // (body a, body b or STATIC, sticking, sensor)
  n: vec4f,      // world normal pointing from b to a; w = friction
  ra: vec4f,     // contact anchor on a, a's frame
  rb: vec4f,     // contact anchor on b, b's frame (world point if b is STATIC)
  lam: vec4f,    // (λn, λt1, λt2, -)
  k: vec4f,      // (kn, kt1, kt2, m_eff/Δt²)
}
struct Joint {
  ab: vec4u,     // (body a, body b or STATIC = world, type, flags)
  ra: vec4f,     // anchor on a, a's frame; w = break force (0: unbreakable)
  rb: vec4f,     // anchor on b, b's frame (world point if b is STATIC); w = length (distance joints)
  qrel: vec4f,   // rest orientation of b relative to a: conj(q_a) ⊗ q_b
  axa: vec4f,    // axis in a's frame (hinge, slider); w = lower limit
  axb: vec4f,    // axis in b's frame (world if b is STATIC); w = upper limit
  prm: vec4f,    // (stiffness: 0 = rigid, damping, motor speed, max motor force/torque)
  lam0: vec4f,   // multipliers, rows 0-3
  lam1: vec4f,   //              rows 4-7
  k0: vec4f,     // penalty stiffness, rows 0-3
  k1: vec4f,     //                    rows 4-7
  c00: vec4f,    // constraint values at the start of the step, rows 0-3
  c01: vec4f,    //                                             rows 4-7
}
const STATIC = 0xffffffffu;
const J_BALL = 0u;
const J_HINGE = 1u;
const J_FIXED = 2u;
const J_DISTANCE = 3u;
const J_SLIDER = 4u;
const JF_BROKEN = 1u;
const JF_COLLIDE = 2u;
const JF_LIMIT = 4u;
const JF_MOTOR = 8u;
const JF_ROPE = 16u;
const JF_WORLD_ANCHORS = 32u;
const JF_DISABLED = 64u;

fn shapeOf(b: Body) -> u32 { return u32(b.size.w + 0.5); }
fn unitv(k: u32) -> vec3f { return vec3f(select(0.0, 1.0, k == 0u), select(0.0, 1.0, k == 1u), select(0.0, 1.0, k == 2u)); }
fn qmulv(q: vec4f, w: vec3f) -> vec4f { return qmul(vec4f(w, 0.0), q); }
// Rotations use the exact exponential and logarithm maps. The small-angle forms
// (q + ½Δt ω⊗q, ω = 2Δq.xyz/Δt) lose a fraction ≈ φ²/8 of the spin every step (φ = |ω|Δt),
// which brakes anything that turns fast: a ball rolling at 2 m/s with r = 0.1 lost half its
// speed in 10 s on flat ground.
/** q turned by the world-frame rotation vector w·dt. */
fn qinteg(q: vec4f, w: vec3f, dt: f32) -> vec4f {
  let a = length(w) * dt;
  if (a < 1e-6) { return normalize(q + 0.5 * dt * qmulv(q, w)); }
  let h = 0.5 * a;
  return normalize(qmul(vec4f(w * (dt * sin(h) / a), cos(h)), q));
}
/** Rotation vector taking q0 to q (world frame). */
fn rotvec(q: vec4f, q0: vec4f) -> vec3f {
  var d = qmul(q, qconj(q0));
  if (d.w < 0.0) { d = -d; }
  let s = length(d.xyz);
  if (s < 1e-7) { return 2.0 * d.xyz; }
  return d.xyz * (2.0 * atan2(s, d.w) / s);
}
/** Inverse world inertia R diag(1/I) Rᵀ (zero for fixed rotation or static bodies). */
fn invInertiaWorld(b: Body) -> mat3x3f {
  if (b.mass.x <= 0.0 || (b.info.z & F_FIXED_ROT) != 0u) { return mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0)); }
  let R = qmat(b.q);
  return R * diag3(1.0 / max(b.mass.yzw, vec3f(1e-30))) * transpose(R);
}

// The part of a body a constraint needs (4 of its 12 vec4s).
struct Pose { x: vec3f, x0: vec3f, q: vec4f, q0: vec4f }

// Contacts are linearised at the start of the step (AVBD: first-order Taylor expansion of
// collision constraints): C(x) = C0 + J0·Δ with constant Jacobians, so every body's local
// problem is exactly quadratic and its Newton step is exact.
// gap: separation of the two anchors at the start of the step (normal part = C0,
//      tangential part = friction drift of a sticking contact);
// rel: displacement of the anchor on a relative to the anchor on b since the step began.
struct Lin { gap: vec3f, rel: vec3f, ra0: vec3f, rb0: vec3f }
fn contactLin(c: Contact, A: Pose, Bp: Pose, hasB: bool) -> Lin {
  var l: Lin;
  l.ra0 = qrot(A.q0, c.ra.xyz);
  let da = A.x - A.x0 + cross(rotvec(A.q, A.q0), l.ra0);
  var db = vec3f(0.0);
  var pb = c.rb.xyz;
  l.rb0 = vec3f(0.0);
  if (hasB) {
    l.rb0 = qrot(Bp.q0, c.rb.xyz);
    db = Bp.x - Bp.x0 + cross(rotvec(Bp.q, Bp.q0), l.rb0);
    pb = Bp.x0 + l.rb0;
  }
  l.gap = A.x0 + l.ra0 - pb;
  l.rel = da - db;
  return l;
}
// Normal constraint. Overlap present at the start of the step is kept with weight α
// (α = 1: only prevent further penetration; α = 0: remove it all); a speculative gap
// (c0 > 0) always counts in full so bodies can close it.
fn contactCn(L: Lin, nrm: vec3f, alpha: f32, slop: f32) -> f32 {
  let c0 = dot(nrm, L.gap) + slop;
  return select(c0 * (1.0 - alpha), c0, c0 > 0.0) + dot(nrm, L.rel);
}

// ------------------------------------------------------------------ joints
// One joint row: its value C at the current poses and the Jacobians for body a (la, aa)
// and body b (lb, ab). kind: 0 equality, 1 inequality C ≥ 0, 2 motor (force bounded by
// ±prm.w), 3 inactive. Jacobians of the angular rows use the small-angle approximation
// around the satisfied state, as AVBD does for rigid-body joints.
// geo: curvature of C along the linear directions (distance joints: 1/len, from H = (I − nnᵀ)/len),
// used for AVBD's lumped geometric-stiffness term |f|·geo on the positional block.
struct JRow { C: f32, la: vec3f, aa: vec3f, lb: vec3f, ab: vec3f, kind: u32, geo: f32 }
const JK_EQ = 0u;
const JK_GE = 1u;
const JK_MOTOR = 2u;
const JK_OFF = 3u;
fn jointRows(t: u32) -> u32 {
  switch t {
    case 0u: { return 3u; }     // ball: point
    case 1u: { return 7u; }     // hinge: point, 2 axis-alignment rows, motor, limit
    case 2u: { return 6u; }     // fixed: point, 3 orientation rows
    case 3u: { return 1u; }     // distance
    default: { return 7u; }     // slider: 2 perpendicular rows, 3 orientation rows, motor, limit
  }
}
fn wrapAngle(a: f32) -> f32 { return a - 6.283185307 * round(a / 6.283185307); }
/** Hinge angle of b relative to a about the hinge axis (exact, any magnitude); a relative to b is its negative. */
fn hingeAngle(j: Joint, qa: vec4f, qb: vec4f) -> f32 {
  let qd = qmul(qmul(qconj(qa), qb), qconj(j.qrel));
  return wrapAngle(2.0 * atan2(dot(qd.xyz, j.axa.xyz), qd.w));
}
/** Small rotation vector taking b's rest orientation (relative to a) to its current one. */
fn orientErr(j: Joint, qa: vec4f, qb: vec4f) -> vec3f {
  var e = qmul(qb, qconj(qmul(qa, j.qrel)));
  if (e.w < 0.0) { e = -e; }
  return 2.0 * e.xyz;
}
fn jointRow(j: Joint, A: Pose, B: Pose, hasB: bool, row: u32, dt: f32) -> JRow {
  var r: JRow;
  r.kind = JK_EQ;
  let t = j.ab.z;
  let raw = qrot(A.q, j.ra.xyz);
  let pa = A.x + raw;
  var rbw = vec3f(0.0);
  var pb = j.rb.xyz;
  var qb = vec4f(0.0, 0.0, 0.0, 1.0);
  var qb0 = qb;
  if (hasB) { rbw = qrot(B.q, j.rb.xyz); pb = B.x + rbw; qb = B.q; qb0 = B.q0; }
  let d = pa - pb;
  if (t == J_DISTANCE) {
    let len = length(d);
    var n = select(vec3f(0.0, 1.0, 0.0), d / len, len > 1e-9);
    r.C = len - j.rb.w;
    r.geo = 1.0 / max(len, 1e-3);
    if ((j.ab.w & JF_ROPE) != 0u) { r.C = -r.C; n = -n; r.kind = JK_GE; }
    r.la = n; r.aa = cross(raw, n); r.lb = -n; r.ab = -cross(rbw, n);
    return r;
  }
  if (t != J_SLIDER && row < 3u) {           // point: anchors coincide
    let e = unitv(row);
    r.C = d[row]; r.la = e; r.aa = cross(raw, e); r.lb = -e; r.ab = -cross(rbw, e);
    return r;
  }
  let aw = qrot(A.q, j.axa.xyz);
  if (t == J_FIXED || (t == J_SLIDER && row >= 2u && row < 5u)) {   // orientation locked
    let k = select(row - 3u, row - 2u, t == J_SLIDER);
    let e = unitv(k);
    r.C = orientErr(j, A.q, qb)[k]; r.aa = -e; r.ab = e;
    return r;
  }
  let T = basis(aw);
  if (t == J_HINGE && row < 5u) {            // hinge axes stay aligned
    var bw = j.axb.xyz;
    if (hasB) { bw = qrot(qb, j.axb.xyz); }
    let tk = select(T[1], T[0], row == 3u);
    r.C = dot(cross(aw, bw), tk); r.aa = -tk; r.ab = tk;
    return r;
  }
  if (t == J_SLIDER && row < 2u) {           // no motion across the slide axis
    let tk = select(T[1], T[0], row == 0u);
    r.C = dot(tk, d); r.la = tk; r.aa = cross(pb - A.x, tk); r.lb = -tk; r.ab = -cross(rbw, tk);
    return r;
  }
  // motor (row 5) and limit (row 6) along the axis: the angle (hinge) or slide (slider) of a
  // relative to b, so for a body attached to the world they are simply its own angle / travel
  var s: f32;
  var la = vec3f(0.0); var aa = aw; var lb = vec3f(0.0); var ab = -aw;
  if (t == J_HINGE) { s = -hingeAngle(j, A.q, qb); }
  else { s = dot(pa - pb, aw); la = aw; aa = cross(aw, A.x - pb); lb = -aw; ab = cross(aw, rbw); }
  if (row == 5u) {
    if ((j.ab.w & JF_MOTOR) == 0u) { r.kind = JK_OFF; return r; }
    let s0 = j.c01.y;                         // value at the start of the step (stored by the prep pass)
    var ds = s - s0;
    if (t == J_HINGE) { ds = wrapAngle(ds); }
    r.C = ds - j.prm.z * dt; r.la = la; r.aa = aa; r.lb = lb; r.ab = ab; r.kind = JK_MOTOR;
    return r;
  }
  if ((j.ab.w & JF_LIMIT) == 0u) { r.kind = JK_OFF; return r; }
  let lo = s - j.axa.w;
  let hi = j.axb.w - s;
  r.kind = JK_GE;
  if (lo < hi) { r.C = lo; r.la = la; r.aa = aa; r.lb = lb; r.ab = ab; }
  else { r.C = hi; r.la = -la; r.aa = -aa; r.lb = -lb; r.ab = -ab; }
  if (r.C > 0.25) { r.kind = JK_OFF; }      // far from both limits: inactive
  return r;
}
fn rowGet(a: vec4f, b: vec4f, row: u32) -> f32 { return select(b[(row - 4u) & 3u], a[row & 3u], row < 4u); }

// ------------------------------------------------------------------ 6x6 solve
// 3x3 Cholesky factor (lower triangle of a symmetric positive-definite matrix) and solve.
struct Chol3 { d: vec3f, l10: f32, l20: f32, l21: f32 }
fn chol3(M: mat3x3f) -> Chol3 {
  var c: Chol3;
  c.d.x = sqrt(max(M[0][0], 1e-30));
  c.l10 = M[0][1] / c.d.x;
  c.l20 = M[0][2] / c.d.x;
  c.d.y = sqrt(max(M[1][1] - c.l10 * c.l10, 1e-30));
  c.l21 = (M[1][2] - c.l20 * c.l10) / c.d.y;
  c.d.z = sqrt(max(M[2][2] - c.l20 * c.l20 - c.l21 * c.l21, 1e-30));
  return c;
}
fn cholSolve(c: Chol3, v: vec3f) -> vec3f {
  let z0 = v.x / c.d.x;
  let z1 = (v.y - c.l10 * z0) / c.d.y;
  let z2 = (v.z - c.l20 * z0 - c.l21 * z1) / c.d.z;
  let x2 = z2 / c.d.z;
  let x1 = (z1 - c.l21 * x2) / c.d.y;
  return vec3f((z0 - c.l10 * x1 - c.l20 * x2) / c.d.x, x1, x2);
}
// Solve the SPD block system [[A, B], [Bᵀ, D]] (x; y) = (a; b) through the Schur complement
// S = D − Bᵀ A⁻¹ B. Same arithmetic as a 6x6 LDLᵀ, but everything stays in registers
// (a dynamically indexed array<f32, 36> spills to memory on most GPUs).
fn solveBlock6(A: mat3x3f, B: mat3x3f, D: mat3x3f, a: vec3f, b: vec3f) -> array<vec3f, 2> {
  let LA = chol3(A);
  let AiB = mat3x3f(cholSolve(LA, B[0]), cholSolve(LA, B[1]), cholSolve(LA, B[2]));
  let Aia = cholSolve(LA, a);
  let y = cholSolve(chol3(D - transpose(B) * AiB), b - transpose(B) * Aia);
  return array<vec3f, 2>(Aia - AiB * y, y);
}
`,
});

// Narrowphase geometry. Spliced into the narrowphase kernels. Manifolds carry a normal
// per point (capsule contacts can have several), always pointing from body b to body a.
export const NARROW_WGSL = /* wgsl */`
struct Manifold { count: u32, nrm: array<vec3f, 8>, pa: array<vec3f, 8>, pb: array<vec3f, 8> }

fn projBox(L: vec3f, R: mat3x3f, h: vec3f) -> f32 {
  return h.x * abs(dot(L, R[0])) + h.y * abs(dot(L, R[1])) + h.z * abs(dot(L, R[2]));
}
fn segClosest(p1: vec3f, d1: vec3f, e1: f32, p2: vec3f, d2: vec3f, e2: f32) -> array<vec3f, 2> {
  let r = p1 - p2;
  let b = dot(d1, d2); let c = dot(d1, r); let f = dot(d2, r);
  let den = 1.0 - b * b;
  var s = select(0.0, (b * f - c) / den, den > 1e-6);
  s = clamp(s, -e1, e1);
  let t = clamp(b * s + f, -e2, e2);
  s = clamp(b * t - c, -e1, e1);
  return array<vec3f, 2>(p1 + d1 * s, p2 + d2 * t);
}
// Append a point unless one already sits within dmin of it.
fn addPoint(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, pa: vec3f, pb: vec3f, n: vec3f, s: f32, dmin: f32) {
  let c = (*M).count;
  if (c >= 8u) { return; }
  for (var k = 0u; k < c; k++) { if (length((*M).pa[k] - pa) < dmin) { return; } }
  (*M).pa[c] = pa; (*M).pb[c] = pb; (*M).nrm[c] = n; (*sep)[c] = s;
  (*M).count = c + 1u;
}
// Keep at most 4 points: deepest, farthest from it, and the two maximising the spanned area.
fn reduce4(m: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, axis: vec3f) {
  let n = (*m).count;
  if (n <= 4u) { return; }
  var i0 = 0u;
  for (var k = 1u; k < n; k++) { if ((*sep)[k] < (*sep)[i0]) { i0 = k; } }
  var i1 = i0; var best = -1.0;
  for (var k = 0u; k < n; k++) { let d = length((*m).pb[k] - (*m).pb[i0]); if (d > best) { best = d; i1 = k; } }
  var i2 = i0; var i3 = i0; var amax = 0.0; var amin = 0.0;
  for (var k = 0u; k < n; k++) {
    let a = dot(cross((*m).pb[i1] - (*m).pb[i0], (*m).pb[k] - (*m).pb[i0]), axis);
    if (a > amax) { amax = a; i2 = k; }
    if (a < amin) { amin = a; i3 = k; }
  }
  var idx = array<u32, 4>(i0, i1, i2, i3);
  var out: Manifold; out.count = 0u;
  var outSep: array<f32, 8>;
  for (var q = 0u; q < 4u; q++) {
    var dup = false;
    for (var r = 0u; r < q; r++) { if (idx[r] == idx[q]) { dup = true; } }
    if (dup) { continue; }
    out.pa[out.count] = (*m).pa[idx[q]]; out.pb[out.count] = (*m).pb[idx[q]]; out.nrm[out.count] = (*m).nrm[idx[q]];
    outSep[out.count] = (*sep)[idx[q]];
    out.count += 1u;
  }
  *m = out; *sep = outSep;
}
fn flipManifold(M: Manifold) -> Manifold {
  var o = M;
  for (var k = 0u; k < M.count; k++) { o.nrm[k] = -M.nrm[k]; o.pa[k] = M.pb[k]; o.pb[k] = M.pa[k]; }
  return o;
}

fn boxBox(A: Body, Bb: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  let Ra = qmat(A.q0); let Rb = qmat(Bb.q0);
  let ca = A.x0.xyz; let cb = Bb.x0.xyz; let ha = A.size.xyz; let hb = Bb.size.xyz;
  let d = cb - ca;
  var best = 1e30; var axis = vec3f(0.0, 1.0, 0.0); var kind = 0u; var ai = 0u; var bj = 0u;
  for (var k = 0u; k < 3u; k++) {
    let L = Ra[k];
    let pen = ha[k] + projBox(L, Rb, hb) - abs(dot(d, L));
    if (pen < -margin) { return M; }
    if (pen < best) { best = pen; axis = L; kind = 0u; ai = k; }
  }
  for (var k = 0u; k < 3u; k++) {
    let L = Rb[k];
    let pen = projBox(L, Ra, ha) + hb[k] - abs(dot(d, L));
    if (pen < -margin) { return M; }
    if (pen < best * 0.98 - 1e-4) { best = pen; axis = L; kind = 1u; ai = k; }
  }
  for (var i = 0u; i < 3u; i++) {
    for (var j = 0u; j < 3u; j++) {
      var L = cross(Ra[i], Rb[j]);
      let len = length(L);
      if (len < 1e-4) { continue; }
      L = L / len;
      let pen = projBox(L, Ra, ha) + projBox(L, Rb, hb) - abs(dot(d, L));
      if (pen < -margin) { return M; }
      // edge axes must be clearly better than face axes (face manifolds are far more stable)
      if (pen < best * 0.9 - 2e-3) { best = pen; axis = L; kind = 2u; ai = i; bj = j; }
    }
  }
  var n = axis;
  if (dot(n, d) > 0.0) { n = -n; }   // n points from b towards a
  var sep: array<f32, 8>;
  if (kind == 2u) {
    // edge–edge: supporting edge of a (toward b, i.e. -n) and of b (toward a, +n)
    var pa = ca; var pb = cb;
    for (var k = 0u; k < 3u; k++) {
      if (k != ai) { pa += Ra[k] * ha[k] * select(1.0, -1.0, dot(Ra[k], n) > 0.0); }
      if (k != bj) { pb += Rb[k] * hb[k] * select(-1.0, 1.0, dot(Rb[k], n) > 0.0); }
    }
    let cp = segClosest(pa, Ra[ai], ha[ai], pb, Rb[bj], hb[bj]);
    M.pa[0] = cp[0]; M.pb[0] = cp[1]; M.nrm[0] = n; M.count = 1u;
    return M;
  }
  // face contact: the axis owner is the reference box
  let refA = kind == 0u;
  var Rr = Ra; var cr = ca; var hr = ha; var Ri = Rb; var cI = cb; var hi = hb;
  if (!refA) { Rr = Rb; cr = cb; hr = hb; Ri = Ra; cI = ca; hi = ha; }
  let toInc = select(n, -n, refA);                    // reference face normal, pointing at the incident box
  let sk = select(-1.0, 1.0, dot(Rr[ai], toInc) > 0.0);
  let rn = Rr[ai] * sk;
  let rc = cr + rn * hr[ai];
  let k1 = (ai + 1u) % 3u; let k2 = (ai + 2u) % 3u;
  // incident face: most anti-parallel to the reference normal
  var ij = 0u; var bd = -1.0;
  for (var k = 0u; k < 3u; k++) { let v = abs(dot(Ri[k], rn)); if (v > bd) { bd = v; ij = k; } }
  let si = select(1.0, -1.0, dot(Ri[ij], rn) > 0.0);
  let ic = cI + Ri[ij] * hi[ij] * si;
  let j1 = (ij + 1u) % 3u; let j2 = (ij + 2u) % 3u;
  let e1 = Ri[j1] * hi[j1]; let e2 = Ri[j2] * hi[j2];
  var poly: array<vec3f, 8>;
  poly[0] = ic + e1 + e2; poly[1] = ic - e1 + e2; poly[2] = ic - e1 - e2; poly[3] = ic + e1 - e2;
  var np = 4u;
  // Sutherland–Hodgman against the 4 side planes of the reference face
  for (var pl = 0u; pl < 4u; pl++) {
    let ax = select(Rr[k2], Rr[k1], pl < 2u);
    let ext = select(hr[k2], hr[k1], pl < 2u);
    let sgn = select(-1.0, 1.0, (pl & 1u) == 0u);
    var outp: array<vec3f, 8>;
    var no = 0u;
    for (var v = 0u; v < np; v++) {
      let a = poly[v];
      let b = poly[(v + 1u) % np];
      let da = sgn * dot(a - cr, ax) - ext;
      let db = sgn * dot(b - cr, ax) - ext;
      if (da <= 0.0 && no < 8u) { outp[no] = a; no += 1u; }
      if ((da <= 0.0) != (db <= 0.0) && no < 8u) { outp[no] = a + (b - a) * (da / (da - db)); no += 1u; }
    }
    poly = outp; np = no;
    if (np == 0u) { return M; }
  }
  for (var v = 0u; v < np; v++) {
    let p = poly[v];
    let s = dot(p - rc, rn);
    if (s > margin) { continue; }
    let onRef = p - rn * s;
    if (refA) { M.pa[M.count] = onRef; M.pb[M.count] = p; } else { M.pa[M.count] = p; M.pb[M.count] = onRef; }
    M.nrm[M.count] = n;
    sep[M.count] = s;
    M.count += 1u;
  }
  reduce4(&M, &sep, rn);
  return M;
}

fn boxFloor(A: Body, floorY: f32, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  let R = qmat(A.q0);
  var sep: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3f(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
    let p = A.x0.xyz + R * (s * A.size.xyz);
    let sp = p.y - floorY;
    if (sp < margin) { M.pa[M.count] = p; M.pb[M.count] = vec3f(p.x, floorY, p.z); M.nrm[M.count] = vec3f(0.0, 1.0, 0.0); sep[M.count] = sp; M.count += 1u; }
  }
  reduce4(&M, &sep, vec3f(0.0, 1.0, 0.0));
  return M;
}

fn capsuleAxis(b: Body) -> vec3f { return qrot(b.q0, vec3f(0.0, 1.0, 0.0)); }

fn capsuleFloor(A: Body, floorY: f32, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let ax = capsuleAxis(A) * A.size.y;
  let r = A.size.x;
  for (var e = 0u; e < 2u; e++) {
    let c = A.x0.xyz + select(-ax, ax, e == 0u);
    let s = c.y - r - floorY;
    if (s < margin) { addPoint(&M, &sep, c - vec3f(0.0, r, 0.0), vec3f(c.x, floorY, c.z), vec3f(0.0, 1.0, 0.0), s, 0.05 * r); }
  }
  return M;
}

// sphere (centre c, radius r) of body a against box Bx (body b)
fn sphereBoxInto(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, c: vec3f, r: f32, Bx: Body, margin: f32) {
  let R = qmat(Bx.q0);
  let lc = transpose(R) * (c - Bx.x0.xyz);
  let h = Bx.size.xyz;
  let qc = clamp(lc, -h, h);
  var n: vec3f; var pb: vec3f; var dist: f32;
  if (any(lc != qc)) {
    let dl = lc - qc;
    dist = length(dl);
    n = R * (dl / dist);
    pb = Bx.x0.xyz + R * qc;
  } else {
    // centre inside the box: push out through the nearest face
    let gap = h - abs(lc);
    var ax = 0u;
    if (gap.y < gap[ax]) { ax = 1u; }
    if (gap.z < gap[ax]) { ax = 2u; }
    var ln = vec3f(0.0); ln[ax] = select(-1.0, 1.0, lc[ax] >= 0.0);
    n = R * ln;
    var fp = lc; fp[ax] = h[ax] * ln[ax];
    pb = Bx.x0.xyz + R * fp;
    dist = -gap[ax];
  }
  let s = dist - r;
  if (s > margin) { return; }
  addPoint(M, sep, c - n * r, pb, n, s, 0.05 * r);
}
fn sphereSphereInto(M: ptr<function, Manifold>, sep: ptr<function, array<f32, 8>>, ca: vec3f, ra: f32, cb: vec3f, rb: f32, margin: f32) {
  let d = ca - cb;
  let dist = length(d);
  let s = dist - ra - rb;
  if (s > margin) { return; }
  let n = select(vec3f(0.0, 1.0, 0.0), d / dist, dist > 1e-6);
  addPoint(M, sep, ca - n * ra, cb + n * rb, n, s, 0.05 * min(ra, rb));
}

fn capsuleSphere(Cp: Body, S: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let u = capsuleAxis(Cp);
  let t = clamp(dot(S.x0.xyz - Cp.x0.xyz, u), -Cp.size.y, Cp.size.y);
  sphereSphereInto(&M, &sep, Cp.x0.xyz + u * t, Cp.size.x, S.x0.xyz, S.size.x, margin);
  return M;
}

fn capsuleCapsule(A: Body, Bb: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let ua = capsuleAxis(A); let ub = capsuleAxis(Bb);
  let ha = A.size.y; let hb = Bb.size.y; let ra = A.size.x; let rb = Bb.size.x;
  let ca = A.x0.xyz; let cb = Bb.x0.xyz;
  let cp = segClosest(ca, ua, ha, cb, ub, hb);
  sphereSphereInto(&M, &sep, cp[0], ra, cp[1], rb, margin);
  if (abs(dot(ua, ub)) > 0.95) {   // nearly parallel: the segment ends too, for a stable line contact
    for (var e = 0u; e < 2u; e++) {
      let pa = ca + ua * select(-ha, ha, e == 0u);
      sphereSphereInto(&M, &sep, pa, ra, cb + ub * clamp(dot(pa - cb, ub), -hb, hb), rb, margin);
      let pb = cb + ub * select(-hb, hb, e == 0u);
      sphereSphereInto(&M, &sep, ca + ua * clamp(dot(pb - ca, ua), -ha, ha), ra, pb, rb, margin);
    }
  }
  if (M.count > 0u) { reduce4(&M, &sep, M.nrm[0]); }
  return M;
}

// capsule (body a) against box (body b): both segment ends plus the segment point closest to
// the box (alternating projections between two convex sets converge to the closest pair)
fn capsuleBox(Cp: Body, Bx: Body, margin: f32) -> Manifold {
  var M: Manifold; M.count = 0u;
  var sep: array<f32, 8>;
  let u = capsuleAxis(Cp); let h = Cp.size.y; let r = Cp.size.x; let c = Cp.x0.xyz;
  sphereBoxInto(&M, &sep, c + u * h, r, Bx, margin);
  sphereBoxInto(&M, &sep, c - u * h, r, Bx, margin);
  let R = qmat(Bx.q0);
  var p = c;
  for (var it = 0; it < 6; it++) {
    let q = Bx.x0.xyz + R * clamp(transpose(R) * (p - Bx.x0.xyz), -Bx.size.xyz, Bx.size.xyz);
    p = c + u * clamp(dot(q - c, u), -h, h);
  }
  sphereBoxInto(&M, &sep, p, r, Bx, margin);
  if (M.count > 0u) { reduce4(&M, &sep, M.nrm[0]); }
  return M;
}

fn floorManifold(A: Body, floorY: f32, margin: f32) -> Manifold {
  let s = shapeOf(A);
  if (s == 1u) { return boxFloor(A, floorY, margin); }
  if (s == 2u) { return capsuleFloor(A, floorY, margin); }
  var M: Manifold; M.count = 0u;
  let p = A.x0.xyz;
  if (p.y - A.size.x - floorY < margin) {
    M.count = 1u; M.nrm[0] = vec3f(0.0, 1.0, 0.0);
    M.pa[0] = p - vec3f(0.0, A.size.x, 0.0); M.pb[0] = vec3f(p.x, floorY, p.z);
  }
  return M;
}

/** Contact manifold between bodies a and b (normals from b to a). */
fn pairManifold(A: Body, Bb: Body, margin: f32) -> Manifold {
  let sa = shapeOf(A); let sb = shapeOf(Bb);
  if (sa == 1u && sb == 1u) { return boxBox(A, Bb, margin); }
  if (sa == 0u && sb == 0u) {
    var M: Manifold; M.count = 0u; var sep: array<f32, 8>;
    sphereSphereInto(&M, &sep, A.x0.xyz, A.size.x, Bb.x0.xyz, Bb.size.x, margin);
    return M;
  }
  if (sa == 0u && sb == 1u) { var M: Manifold; M.count = 0u; var sep: array<f32, 8>; sphereBoxInto(&M, &sep, A.x0.xyz, A.size.x, Bb, margin); return M; }
  if (sa == 1u && sb == 0u) { var M: Manifold; M.count = 0u; var sep: array<f32, 8>; sphereBoxInto(&M, &sep, Bb.x0.xyz, Bb.size.x, A, margin); return flipManifold(M); }
  if (sa == 2u && sb == 2u) { return capsuleCapsule(A, Bb, margin); }
  if (sa == 2u && sb == 0u) { return capsuleSphere(A, Bb, margin); }
  if (sa == 0u && sb == 2u) { return flipManifold(capsuleSphere(Bb, A, margin)); }
  if (sa == 2u && sb == 1u) { return capsuleBox(A, Bb, margin); }
  return flipManifold(capsuleBox(Bb, A, margin));   // box a, capsule b
}
`;

// Ray tests against a body at its current pose: (t, normal); t < 0 is a miss.
export const RAY_WGSL = /* wgsl */`
fn rayBody(o: vec3f, d: vec3f, b: Body) -> vec4f {
  let s = shapeOf(b);
  let R = qmat(b.q);
  let lo = transpose(R) * (o - b.x.xyz);
  let ld = transpose(R) * d;
  if (s == 0u) {
    let r = b.size.x;
    let bb = dot(lo, ld);
    let h = bb * bb - dot(lo, lo) + r * r;
    if (h < 0.0) { return vec4f(-1.0); }
    let t = -bb - sqrt(h);
    if (t < 0.0) { return vec4f(-1.0); }
    return vec4f(t, R * ((lo + ld * t) / r));
  }
  if (s == 1u) {
    let h = b.size.xyz;
    let inv = 1.0 / select(ld, vec3f(1e-20), abs(ld) < vec3f(1e-20));
    let t1 = (-h - lo) * inv;
    let t2 = (h - lo) * inv;
    let tn = min(t1, t2);
    let tf = max(t1, t2);
    let tnear = max(max(tn.x, tn.y), tn.z);
    let tfar = min(min(tf.x, tf.y), tf.z);
    if (tnear > tfar || tnear < 0.0) { return vec4f(-1.0); }
    var ln = vec3f(0.0);
    if (tnear == tn.x) { ln.x = -sign(ld.x); } else if (tnear == tn.y) { ln.y = -sign(ld.y); } else { ln.z = -sign(ld.z); }
    return vec4f(tnear, R * ln);
  }
  // capsule (Quilez): segment from (0,-h,0) to (0,h,0), radius r
  let r = b.size.x;
  let pa = vec3f(0.0, -b.size.y, 0.0);
  let ba = vec3f(0.0, 2.0 * b.size.y, 0.0);
  let oa = lo - pa;
  let baba = dot(ba, ba); let bard = dot(ba, ld); let baoa = dot(ba, oa);
  let rdoa = dot(ld, oa); let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  var bq = baba * rdoa - baoa * bard;
  var c = baba * oaoa - baoa * baoa - r * r * baba;
  var h = bq * bq - a * c;
  if (h < 0.0) { return vec4f(-1.0); }
  var t = (-bq - sqrt(h)) / a;
  let y = baoa + t * bard;
  if (!(y > 0.0 && y < baba && a > 1e-12)) {
    let oc = select(lo - (pa + ba), oa, y <= 0.0);
    bq = dot(ld, oc);
    c = dot(oc, oc) - r * r;
    h = bq * bq - c;
    if (h <= 0.0) { return vec4f(-1.0); }
    t = -bq - sqrt(h);
  }
  if (t < 0.0) { return vec4f(-1.0); }
  let p = lo + ld * t;
  let hh = clamp(dot(p - pa, ba) / baba, 0.0, 1.0);
  return vec4f(t, R * ((p - pa - hh * ba) / r));
}
`;
