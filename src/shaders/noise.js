import { defineModule } from '../core/wgsl.js';

// Procedural noise with analytic derivatives.
//
// Design (first principles): a noise evaluation is lattice hashing + gradient
// generation + kernel falloff. We make each part cheap:
//  * hashing: (x*K1) ^ (y*K2) ^ (z*K3) is linear in each coordinate mod 2^32, so
//    the corners' products are the base product plus a constant — selects and
//    adds instead of 12 multiplies; one lowbias32-style finaliser per corner.
//  * gradients: unpack4x8snorm(hash) gives a random vector in [-1,1]^3 in one
//    instruction; normalising it makes the gradient set isotropic.
//  * simplex kernels use r^2 = 0.5 so contributions vanish exactly at the
//    simplex neighbourhood boundary -> C2-continuous value and gradient
//    (curl noise and normal mapping need smooth derivatives).
// Output ranges are calibrated to roughly [-1, 1].

defineModule('noise', {
  deps: ['hash'],
  doc: 'Simplex 2D/3D/4D, Perlin, value and Worley noise, fBm, ridged, curl noise — all with analytic gradients where useful.',
  code: /* wgsl */`
const SI_K = vec4u(0x8da6b343u, 0xd8163841u, 0xcb1ab31fu, 0x9e3779b1u);

fn si_fin4(h0: vec4u) -> vec4u {
  var h = (h0 ^ (h0 >> vec4u(16u))) * 0x7feb352du;
  h = (h ^ (h >> vec4u(15u))) * 0x846ca68bu;
  return h ^ (h >> vec4u(16u));
}
fn si_fin(h0: u32) -> u32 {
  var h = (h0 ^ (h0 >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
fn si_g3(h: u32) -> vec3f {
  let g = unpack4x8snorm(h).xyz;
  return g * inverseSqrt(max(dot(g, g), 1e-4));
}
fn si_g2(h: u32) -> vec2f {
  let g = unpack2x16snorm(h);
  return g * inverseSqrt(max(dot(g, g), 1e-6));
}
fn si_g4(h: u32) -> vec4f {
  let g = unpack4x8snorm(h);
  return g * inverseSqrt(max(dot(g, g), 1e-4));
}

// ------------------------------------------------------------------ simplex 2D
/** 2D simplex noise: (value, d/dx, d/dy). */
fn simplex2d(p: vec2f) -> vec3f {
  const K1 = 0.366025403784439;  // (sqrt(3)-1)/2
  const K2 = 0.211324865405187;  // (3-sqrt(3))/6
  let s = floor(p + (p.x + p.y) * K1);
  let a = p - s + (s.x + s.y) * K2;
  let m = select(0.0, 1.0, a.x > a.y);
  let o = vec2f(m, 1.0 - m);
  let b = a - o + K2;
  let c = a - 1.0 + 2.0 * K2;
  let bx = u32(i32(s.x)) * SI_K.x;
  let by = u32(i32(s.y)) * SI_K.y;
  let ox = select(0u, SI_K.x, m > 0.5);
  let oy = select(SI_K.y, 0u, m > 0.5);
  let h = si_fin4(vec4u(bx ^ by, (bx + ox) ^ (by + oy), (bx + SI_K.x) ^ (by + SI_K.y), 0u));
  let ga = si_g2(h.x); let gb = si_g2(h.y); let gc = si_g2(h.z);
  let w = max(0.5 - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
  let w2 = w * w; let w4 = w2 * w2;
  let d = vec3f(dot(ga, a), dot(gb, b), dot(gc, c));
  let n = dot(w4, d);
  let t = w2 * w * d * 8.0;
  let g = w4.x * ga + w4.y * gb + w4.z * gc - (t.x * a + t.y * b + t.z * c);
  return vec3f(n, g) * 96.0;
}
fn simplex2(p: vec2f) -> f32 {
  const K1 = 0.366025403784439;
  const K2 = 0.211324865405187;
  let s = floor(p + (p.x + p.y) * K1);
  let a = p - s + (s.x + s.y) * K2;
  let m = select(0.0, 1.0, a.x > a.y);
  let b = a - vec2f(m, 1.0 - m) + K2;
  let c = a - 1.0 + 2.0 * K2;
  let bx = u32(i32(s.x)) * SI_K.x;
  let by = u32(i32(s.y)) * SI_K.y;
  let ox = select(0u, SI_K.x, m > 0.5);
  let oy = select(SI_K.y, 0u, m > 0.5);
  let h = si_fin4(vec4u(bx ^ by, (bx + ox) ^ (by + oy), (bx + SI_K.x) ^ (by + SI_K.y), 0u));
  let w = max(0.5 - vec3f(dot(a, a), dot(b, b), dot(c, c)), vec3f(0.0));
  let w2 = w * w;
  return 96.0 * dot(w2 * w2, vec3f(dot(si_g2(h.x), a), dot(si_g2(h.y), b), dot(si_g2(h.z), c)));
}

// ------------------------------------------------------------------ simplex 3D
fn si_corners3(s: vec3f, i1: vec3f, i2: vec3f) -> vec4u {
  let b = vec3u(vec3i(s)) * SI_K.xyz;
  let o1 = select(vec3u(0u), SI_K.xyz, i1 > vec3f(0.5));
  let o2 = select(vec3u(0u), SI_K.xyz, i2 > vec3f(0.5));
  let c1 = b + o1; let c2 = b + o2; let c3 = b + SI_K.xyz;
  return si_fin4(vec4u(b.x ^ b.y ^ b.z, c1.x ^ c1.y ^ c1.z, c2.x ^ c2.y ^ c2.z, c3.x ^ c3.y ^ c3.z));
}
/** 3D simplex noise: (value, gradient). */
fn simplex3d(p: vec3f) -> vec4f {
  let s = floor(p + dot(p, vec3f(1.0 / 3.0)));
  let x0 = p - s + dot(s, vec3f(1.0 / 6.0));
  let e = step(x0.yzx, x0);
  let l = 1.0 - e;
  let i1 = min(e, l.zxy);
  let i2 = max(e, l.zxy);
  let x1 = x0 - i1 + 1.0 / 6.0;
  let x2 = x0 - i2 + 1.0 / 3.0;
  let x3 = x0 - 0.5;
  let h = si_corners3(s, i1, i2);
  let g0 = si_g3(h.x); let g1 = si_g3(h.y); let g2 = si_g3(h.z); let g3 = si_g3(h.w);
  let w = max(0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  let w2 = w * w; let w4 = w2 * w2;
  let d = vec4f(dot(g0, x0), dot(g1, x1), dot(g2, x2), dot(g3, x3));
  let t = w2 * w * d * 8.0;
  let g = w4.x * g0 + w4.y * g1 + w4.z * g2 + w4.w * g3 - (t.x * x0 + t.y * x1 + t.z * x2 + t.w * x3);
  return vec4f(dot(w4, d), g) * 104.0;
}
fn simplex3(p: vec3f) -> f32 {
  let s = floor(p + dot(p, vec3f(1.0 / 3.0)));
  let x0 = p - s + dot(s, vec3f(1.0 / 6.0));
  let e = step(x0.yzx, x0);
  let l = 1.0 - e;
  let i1 = min(e, l.zxy);
  let i2 = max(e, l.zxy);
  let x1 = x0 - i1 + 1.0 / 6.0;
  let x2 = x0 - i2 + 1.0 / 3.0;
  let x3 = x0 - 0.5;
  let h = si_corners3(s, i1, i2);
  let w = max(0.5 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
  let w2 = w * w;
  let d = vec4f(dot(si_g3(h.x), x0), dot(si_g3(h.y), x1), dot(si_g3(h.z), x2), dot(si_g3(h.w), x3));
  return 104.0 * dot(w2 * w2, d);
}

// ------------------------------------------------------------------ simplex 4D
/** 4D simplex noise (e.g. animated 3D noise: simplex4(vec4f(p, t))). */
fn simplex4(v: vec4f) -> f32 {
  const F4 = 0.309016994374947;
  const G4 = 0.138196601125011;
  let s = floor(v + dot(v, vec4f(F4)));
  let x0 = v - s + dot(s, vec4f(G4));
  // rank of each component -> simplex corners
  let isX = step(x0.yzw, x0.xxx);
  let isYZ = step(x0.zww, x0.yyz);
  var r = vec4f(isX.x + isX.y + isX.z, 1.0 - isX);
  r.y += isYZ.x + isYZ.y;
  r.z += 1.0 - isYZ.x;
  r.w += 1.0 - isYZ.y;
  r.z += isYZ.z;
  r.w += 1.0 - isYZ.z;
  let i3 = clamp(r, vec4f(0.0), vec4f(1.0));
  let i2 = clamp(r - 1.0, vec4f(0.0), vec4f(1.0));
  let i1 = clamp(r - 2.0, vec4f(0.0), vec4f(1.0));
  let x1 = x0 - i1 + G4;
  let x2 = x0 - i2 + 2.0 * G4;
  let x3 = x0 - i3 + 3.0 * G4;
  let x4 = x0 - 1.0 + 4.0 * G4;
  let b = vec4u(vec4i(s)) * SI_K;
  let c1 = b + select(vec4u(0u), SI_K, i1 > vec4f(0.5));
  let c2 = b + select(vec4u(0u), SI_K, i2 > vec4f(0.5));
  let c3 = b + select(vec4u(0u), SI_K, i3 > vec4f(0.5));
  let c4 = b + SI_K;
  let h = si_fin4(vec4u(b.x ^ b.y ^ b.z ^ b.w, c1.x ^ c1.y ^ c1.z ^ c1.w, c2.x ^ c2.y ^ c2.z ^ c2.w, c3.x ^ c3.y ^ c3.z ^ c3.w));
  let h4 = si_fin(c4.x ^ c4.y ^ c4.z ^ c4.w);
  let w0 = max(0.6 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3f(0.0));
  let w1 = max(0.6 - vec2f(dot(x3, x3), dot(x4, x4)), vec2f(0.0));
  let a0 = w0 * w0; let a1 = w1 * w1;
  return 45.0 * (dot(a0 * a0, vec3f(dot(si_g4(h.x), x0), dot(si_g4(h.y), x1), dot(si_g4(h.z), x2)))
               + dot(a1 * a1, vec2f(dot(si_g4(h.w), x3), dot(si_g4(h4), x4))));
}

// ------------------------------------------------------------------ Perlin / value
fn si_lat3(i: vec3f) -> array<u32, 8> {
  let b = vec3u(vec3i(i)) * SI_K.xyz;
  let c = b + SI_K.xyz;
  let lo = si_fin4(vec4u(b.x ^ b.y ^ b.z, c.x ^ b.y ^ b.z, b.x ^ c.y ^ b.z, c.x ^ c.y ^ b.z));
  let hi = si_fin4(vec4u(b.x ^ b.y ^ c.z, c.x ^ b.y ^ c.z, b.x ^ c.y ^ c.z, c.x ^ c.y ^ c.z));
  return array<u32, 8>(lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w);
}
/** 3D gradient (Perlin) noise with quintic fade: (value, gradient). */
fn perlin3d(x: vec3f) -> vec4f {
  let i = floor(x);
  let f = x - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let h = si_lat3(i);
  let ga = si_g3(h[0]); let gb = si_g3(h[1]); let gc = si_g3(h[2]); let gd = si_g3(h[3]);
  let ge = si_g3(h[4]); let gf = si_g3(h[5]); let gg = si_g3(h[6]); let gh = si_g3(h[7]);
  let va = dot(ga, f);
  let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
  let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
  let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
  let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
  let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
  let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
  let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
  let k = -va + vb + vc - vd + ve - vf - vg + vh;
  let v = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf) + k * u.x * u.y * u.z;
  let g = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
        + du * (vec3f(vb, vc, ve) - va
               + u.yzx * vec3f(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
               + u.zxy * vec3f(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
               + u.yzx * u.zxy * k);
  return vec4f(v, g) * 1.42;
}
fn perlin3(x: vec3f) -> f32 { return perlin3d(x).x; }

/** 3D value noise in [-1, 1] (cheapest lattice noise). */
fn value3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = x - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let h = si_lat3(i);
  let a = vec4f(u2f(h[0]), u2f(h[1]), u2f(h[2]), u2f(h[3]));
  let b = vec4f(u2f(h[4]), u2f(h[5]), u2f(h[6]), u2f(h[7]));
  let z = mix(a, b, u.z);
  let y = mix(z.xy, z.zw, u.y);
  return mix(y.x, y.y, u.x) * 2.0 - 1.0;
}

// ------------------------------------------------------------------ cellular
/** 2D Worley noise: (F1, F2) Euclidean distances to the nearest feature points. */
fn worley2(p: vec2f) -> vec2f {
  let i = floor(p);
  let f = p - i;
  var d = vec2f(8.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2f(f32(x), f32(y));
      let c = vec2i(i) + vec2i(x, y);
      let h = pcg2d(bitcast<vec2u>(c));
      let r = o + u2f2(h) - f;
      let dd = dot(r, r);
      if (dd < d.x) { d = vec2f(dd, d.x); } else if (dd < d.y) { d.y = dd; }
    }
  }
  return sqrt(d);
}
/** 3D Worley noise: (F1, F2). */
fn worley3(p: vec3f) -> vec2f {
  let i = floor(p);
  let f = p - i;
  var d = vec2f(8.0);
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3f(f32(x), f32(y), f32(z));
        let h = pcg3d(bitcast<vec3u>(vec3i(i) + vec3i(x, y, z)));
        let r = o + u2f3(h) - f;
        let dd = dot(r, r);
        if (dd < d.x) { d = vec2f(dd, d.x); } else if (dd < d.y) { d.y = dd; }
      }
    }
  }
  return sqrt(d);
}

// ------------------------------------------------------------------ fractal sums
// Octaves are rotated (orthonormal, det=1) and scaled by ~2 to hide lattice alignment.
const SI_OCT = mat3x3f(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);

fn fbm2(p: vec2f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * simplex2(q);
    q = mat2x2f(1.6, 1.2, -1.2, 1.6) * q;
    a *= 0.5;
  }
  return s;
}
fn fbm3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * simplex3(q);
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}
/** fBm with exact gradient (chain rule through the octave rotations). */
fn fbm3d(p: vec3f, octaves: i32) -> vec4f {
  var q = p; var a = 0.5; var s = 0.0; var g = vec3f(0.0);
  var J = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // dq/dp
  for (var k = 0; k < octaves; k++) {
    let n = simplex3d(q);
    s += a * n.x;
    g += a * (transpose(J) * n.yzw);
    q = SI_OCT * q * 2.02;
    J = SI_OCT * J * 2.02;
    a *= 0.5;
  }
  return vec4f(s, g);
}
/** Ridged multifractal (sharp crests), roughly [0, 1]. */
fn ridged3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0; var prev = 1.0;
  for (var k = 0; k < octaves; k++) {
    var n = 1.0 - abs(simplex3(q));
    n = n * n;
    s += n * a * prev;
    prev = n;
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}
fn turbulence3(p: vec3f, octaves: i32) -> f32 {
  var q = p; var a = 0.5; var s = 0.0;
  for (var k = 0; k < octaves; k++) {
    s += a * abs(simplex3(q));
    q = SI_OCT * q * 2.02;
    a *= 0.5;
  }
  return s;
}

// ------------------------------------------------------------------ curl noise
/**
 * Divergence-free 3D noise field (Bridson et al. 2007): curl of a vector
 * potential of three decorrelated simplex fields, using analytic gradients.
 */
fn curl3(p: vec3f) -> vec3f {
  let a = simplex3d(p);
  let b = simplex3d(p + vec3f(31.416, -47.853, 12.793));
  let c = simplex3d(p + vec3f(-233.145, -113.408, -185.31));
  return vec3f(c.z - b.w, a.w - c.y, b.y - a.z);
}
/** 2D curl noise: rotated gradient of a scalar stream function. */
fn curl2(p: vec2f) -> vec2f {
  let n = simplex2d(p);
  return vec2f(n.z, -n.y);
}
`,
});
