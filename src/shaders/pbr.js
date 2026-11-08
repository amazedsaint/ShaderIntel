import { defineModule } from '../core/wgsl.js';

// Physically based shading: GGX microfacet specular with the height-correlated
// Smith visibility term (Heitz 2014), Schlick Fresnel, Lambert diffuse, and an
// analytic sky for image-free ambient + reflections.
defineModule('pbr', {
  deps: ['math'],
  doc: 'Cook-Torrance GGX BRDF, Schlick Fresnel, analytic sky, environment BRDF approximation.',
  code: /* wgsl */`
fn D_GGX(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let f = (NoH * a2 - NoH) * NoH + 1.0;
  return a2 / (PI * f * f + 1e-7);
}
fn V_SmithGGX(NoV: f32, NoL: f32, a: f32) -> f32 {
  let a2 = a * a;
  let gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / (gv + gl + 1e-7);
}
fn F_Schlick(VoH: f32, f0: vec3f) -> vec3f {
  let f = pow(1.0 - VoH, 5.0);
  return f + f0 * (1.0 - f);
}
/** Direct light: returns outgoing radiance for one light of given radiance. */
fn brdfDirect(n: vec3f, v: vec3f, l: vec3f, albedo: vec3f, metallic: f32, roughness: f32, radiance: vec3f) -> vec3f {
  let h = normalize(v + l);
  let NoV = max(dot(n, v), 1e-4);
  let NoL = clamp(dot(n, l), 0.0, 1.0);
  let NoH = clamp(dot(n, h), 0.0, 1.0);
  let VoH = clamp(dot(v, h), 0.0, 1.0);
  let a = max(roughness * roughness, 2e-3);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let F = F_Schlick(VoH, f0);
  let spec = D_GGX(NoH, a) * V_SmithGGX(NoV, NoL, a) * F;
  let kd = (1.0 - F) * (1.0 - metallic);
  return (kd * albedo * INV_PI + spec) * radiance * NoL;
}
/** Split-sum environment BRDF approximation (Karis, mobile). */
fn envBRDF(f0: vec3f, NoV: f32, roughness: f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}
/** Analytic sky radiance (gradient + horizon glow + sun disc). */
fn skyColor(rd: vec3f, sun: vec3f) -> vec3f {
  let y = rd.y;
  var c = mix(vec3f(0.55, 0.62, 0.72), vec3f(0.20, 0.38, 0.75), clamp(y, 0.0, 1.0));
  c = mix(c, vec3f(0.18, 0.17, 0.16), clamp(-y * 4.0, 0.0, 1.0));
  c += vec3f(1.0, 0.75, 0.45) * 0.35 * pow(1.0 - abs(y), 8.0);
  let s = max(dot(rd, sun), 0.0);
  c += vec3f(1.0, 0.85, 0.6) * (0.25 * pow(s, 8.0) + 30.0 * pow(s, 1500.0));
  return c;
}
/** One-call shading: sun (with shadow factor) + sky ambient + reflection. */
fn shadePBR(n: vec3f, v: vec3f, albedo: vec3f, metallic: f32, roughness: f32, sun: vec3f, sunRadiance: vec3f, shadow: f32, ao: f32) -> vec3f {
  var c = brdfDirect(n, v, sun, albedo, metallic, roughness, sunRadiance) * shadow;
  let NoV = max(dot(n, v), 1e-4);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let spec = envBRDF(f0, NoV, roughness);
  let r = reflect(-v, n);
  let skyAmb = skyColor(normalize(n + vec3f(0.0, 1.0, 0.0)), sun) * 0.6;
  let refl = mix(skyColor(r, sun), skyAmb, roughness);
  c += ao * ((1.0 - spec) * (1.0 - metallic) * albedo * skyAmb + spec * refl);
  return c;
}
`,
});
