import { defineModule } from '../core/wgsl.js';

defineModule('color', {
  deps: ['math'],
  doc: 'sRGB transfer, OKLab/OKLCh, HSV, cosine palettes, scientific colormaps (turbo, viridis, inferno, magma).',
  code: /* wgsl */`
/** Exact sRGB EOTF / OETF (IEC 61966-2-1). */
fn srgbToLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn linearToSrgb(c: vec3f) -> vec3f {
  let x = max(c, vec3f(0.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, 12.92 * x, x <= vec3f(0.0031308));
}
fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// ---- OKLab (Björn Ottosson, 2020): perceptually uniform, great for gradients.
fn si_cbrt(x: vec3f) -> vec3f { return sign(x) * pow(abs(x), vec3f(1.0 / 3.0)); }
fn linearToOklab(c: vec3f) -> vec3f {
  let lms = vec3f(
    0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
    0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
    0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b);
  let q = si_cbrt(lms);
  return vec3f(
    0.2104542553 * q.x + 0.7936177850 * q.y - 0.0040720468 * q.z,
    1.9779984951 * q.x - 2.4285922050 * q.y + 0.4505937099 * q.z,
    0.0259040371 * q.x + 0.7827717662 * q.y - 0.8086757660 * q.z);
}
fn oklabToLinear(l: vec3f) -> vec3f {
  let q = vec3f(
    l.x + 0.3963377774 * l.y + 0.2158037573 * l.z,
    l.x - 0.1055613458 * l.y - 0.0638541728 * l.z,
    l.x - 0.0894841775 * l.y - 1.2914855480 * l.z);
  let lms = q * q * q;
  return vec3f(
     4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
    -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
    -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z);
}
/** (L, C, h[radians]) <-> OKLab. */
fn oklabToOklch(l: vec3f) -> vec3f { return vec3f(l.x, length(l.yz), atan2(l.z, l.y)); }
fn oklchToOklab(l: vec3f) -> vec3f { return vec3f(l.x, l.y * cos(l.z), l.y * sin(l.z)); }
/** Perceptual interpolation of two linear-RGB colours. */
fn mixOklab(a: vec3f, b: vec3f, t: f32) -> vec3f {
  return oklabToLinear(mix(linearToOklab(a), linearToOklab(b), t));
}

/** HSV (all in [0,1]) to linear-ish RGB, smooth variant. */
fn hsv2rgb(c: vec3f) -> vec3f {
  let rgb = clamp(abs((c.x * 6.0 + vec3f(0.0, 4.0, 2.0)) % 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
  let s = rgb * rgb * (3.0 - 2.0 * rgb);
  return c.z * mix(vec3f(1.0), s, c.y);
}
fn rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = select(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), c.g >= c.b);
  let q = select(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), c.r >= p.x);
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

/** Cosine palette a + b·cos(2π(c·t + d)) (Quilez). */
fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}

// ---- scientific colormaps (inputs clamped to [0,1], outputs sRGB-encoded)
fn turbo(x0: f32) -> vec3f {
  let x = clamp(x0, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return vec3f(
    dot(v4, vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396, 59.28637943)),
    dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2f(4.27729857, 2.82956604)),
    dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f(-89.90310912, 27.34824973)));
}
fn si_poly6(t: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f, c6: vec3f) -> vec3f {
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
fn viridis(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061),
    vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685),
    vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659),
    vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987),
    vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105),
    vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234),
    vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832));
}
fn inferno(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184),
    vec3f(0.1065134194856116, 0.5639564367884091, 3.932712388889277),
    vec3f(11.60249308247187, -3.972853965665698, -15.9423941062914),
    vec3f(-41.70399613139459, 17.43639888205313, 44.35414519872813),
    vec3f(77.162935699427, -33.40235894210092, -81.80730925738993),
    vec3f(-71.31942824499214, 32.62606426397723, 73.20951985803202),
    vec3f(25.13112622477341, -12.24266895238567, -23.07032500287172));
}
fn magma(x: f32) -> vec3f {
  return si_poly6(clamp(x, 0.0, 1.0),
    vec3f(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933),
    vec3f(0.2516605407371642, 0.6775232436837668, 2.494026599312351),
    vec3f(8.353717279216625, -3.577719514958484, 0.3144679030132573),
    vec3f(-27.66873308576866, 14.26473078096533, -13.64921318813922),
    vec3f(52.17613981234068, -27.94360607168351, 12.94416944238394),
    vec3f(-50.76852536473588, 29.04658282127291, 4.23415299384598),
    vec3f(18.65570506591883, -11.48977351997711, -5.601961508734096));
}
`,
});

defineModule('tonemap', {
  deps: ['color'],
  doc: 'HDR -> display: ACES (Narkowicz and Hill fits), AgX, Khronos PBR Neutral, Reinhard; blue-noise-free dithering.',
  code: /* wgsl */`
fn tonemapACES(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
/** ACES RRT+ODT fit (Stephen Hill), includes the sRGB->AP1 matrices. */
fn tonemapACESHill(c: vec3f) -> vec3f {
  let inM = mat3x3f(0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777);
  let outM = mat3x3f(1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276, 1.07602);
  let v = c * inM;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp((a / b) * outM, vec3f(0.0), vec3f(1.0));
}
/** AgX (Sobotka) with the minimal polynomial fit; returns display-linear RGB. */
fn tonemapAgX(c: vec3f) -> vec3f {
  let agxM = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                     0.0784335999999992, 0.878468636469772, 0.0784336,
                     0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let agxInv = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393; let maxEv = 4.026069;
  var v = agxM * max(c, vec3f(1e-10));
  v = clamp(log2(v), vec3f(minEv), vec3f(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  let x2 = v * v; let x4 = x2 * x2;
  v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
  v = agxInv * v;
  return pow(max(v, vec3f(0.0)), vec3f(2.2));
}
/** Khronos PBR Neutral (2024): hue-preserving, keeps base colours faithful. */
fn tonemapNeutral(c0: vec3f) -> vec3f {
  let start = 0.8 - 0.04;
  let desat = 0.15;
  var c = c0;
  let x = min(c.r, min(c.g, c.b));
  let off = select(0.04, x - 6.25 * x * x, x < 0.08);
  c -= off;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < start) { return c; }
  let d = 1.0 - start;
  let newPeak = 1.0 - d * d / (peak + d - start);
  c *= newPeak / peak;
  let g = 1.0 - 1.0 / (desat * (peak - newPeak) + 1.0);
  return mix(c, vec3f(newPeak), g);
}
/** Extended Reinhard on luminance with white point w. */
fn tonemapReinhard(c: vec3f, w: f32) -> vec3f {
  let l = luminance(c);
  let n = l * (1.0 + l / (w * w)) / (1.0 + l);
  return c * (n / max(l, 1e-6));
}
/** Final display encode with ±0.5 LSB triangular dither to kill 8-bit banding. */
fn encodeDisplay(linear: vec3f, px: vec2f) -> vec3f {
  let s = linearToSrgb(linear);
  let n = fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715)))) + fract(52.9829189 * fract(dot(px + 17.0, vec2f(0.06711056, 0.00583715)))) - 1.0;
  return s + n / 255.0;
}
`,
});
