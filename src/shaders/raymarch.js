import { defineModule } from '../core/wgsl.js';

// Sphere tracing for user scenes. The importing shader must define
//   fn map(p: vec3f) -> vec2f   // (signed distance, material id)
//
// The fast path is two-pass cone marching: neighbouring pixel rays share most
// of their empty space, so one conservative cone per T×T pixel tile walks that
// space once (coneMarch), and every pixel ray then starts at the tile's safe
// distance. A distance field is 1-Lipschitz, so if map(p) > r the whole disc of
// radius r around p is empty — the pre-pass can never skip a surface. On a
// typical scene this removes ~80% of per-pixel map() calls (bench: 'raymarch').
//
// raymarch() also implements over-relaxed sphere tracing (Keinert et al. 2014).
// Measured lesson: fewer steps per ray did not translate into less time on GPUs,
// because SIMD lanes wait for the slowest ray and the bookkeeping costs ALU.
defineModule('raymarch', {
  doc: 'Over-relaxed sphere tracing, tetrahedral normals, soft shadows, ambient occlusion. Requires fn map(p: vec3f) -> vec2f.',
  code: /* wgsl */`
struct Hit { t: f32, mat: f32, steps: u32, hit: bool }

/** Trace a ray against map(). pixelAngle ≈ tan(fov/2) * 2 / screenHeight. */
fn raymarch(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, pixelAngle: f32) -> Hit {
  var omega = 1.6;
  var t = tmin;
  var candT = tmin;
  var candErr = 1e32;
  var candMat = 0.0;
  var prevR = 0.0;
  var stepLen = 0.0;
  let fsign = select(1.0, -1.0, map(ro + rd * tmin).x < 0.0);
  var k = 0u;
  for (; k < 256u; k++) {
    let m = map(ro + rd * t);
    let signedR = fsign * m.x;
    let r = abs(signedR);
    let fail = omega > 1.0 && (r + prevR) < stepLen;
    if (fail) {
      stepLen -= omega * stepLen;
      omega = 1.0;
    } else {
      stepLen = signedR * omega;
      let err = r / max(t, 1e-4);
      if (err < candErr) { candT = t; candErr = err; candMat = m.y; }
      if (err < pixelAngle) { break; }
    }
    prevR = r;
    if (t > tmax) { break; }
    t += stepLen;
  }
  return Hit(candT, candMat, k, t <= tmax && candErr < pixelAngle * 4.0);
}

/**
 * Conservative cone march for a pixel tile. k = cone radius per unit distance
 * (≈ tile half-diagonal in pixels × pixel angle). Returns a distance that every
 * ray inside the cone can safely start from.
 */
fn coneMarch(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32 {
  var t = tmin;
  for (var s = 0; s < 160; s++) {
    let d = map(ro + rd * t).x;
    let r = k * t;
    if (d <= r * 1.5 || t > tmax) { break; }
    t += (d - r) / (1.0 + k);
  }
  return min(t, tmax);
}

/** Classic sphere tracing (reference / benchmark baseline). */
fn raymarchBasic(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, pixelAngle: f32) -> Hit {
  var t = tmin;
  var k = 0u;
  var mat = 0.0;
  for (; k < 256u; k++) {
    let m = map(ro + rd * t);
    mat = m.y;
    if (abs(m.x) < pixelAngle * t || t > tmax) { break; }
    t += m.x;
  }
  return Hit(t, mat, k, t <= tmax);
}

/** Surface normal from 4 map() samples (tetrahedral central differences). */
fn calcNormal(p: vec3f, eps: f32) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.5773 * eps;
  return normalize(e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x +
                   e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);
}

/** Soft shadow toward a light (Quilez's improved penumbra estimate). k = sharpness. */
fn softShadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32 {
  var res = 1.0;
  var t = tmin;
  var ph = 1e10;
  for (var i = 0; i < 64; i++) {
    let h = map(ro + rd * t).x;
    if (h < 1e-4) { return 0.0; }
    let y = h * h / (2.0 * ph);
    let d = sqrt(max(h * h - y * y, 0.0));
    res = min(res, k * d / max(1e-4, t - y));
    ph = h;
    t += clamp(h, 0.005, 0.5);
    if (t > tmax) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

/** Ambient occlusion from 5 samples along the normal. */
fn calcAO(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i++) {
    let h = 0.01 + 0.12 * f32(i) / 4.0;
    let d = map(p + h * n).x;
    occ += (h - d) * sca;
    sca *= 0.95;
    if (occ > 0.35) { break; }
  }
  return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * n.y);
}
`,
});
