import { defineModule } from '../core/wgsl.js';

// Small dense linear algebra for physics kernels: outer products, 3x3 symmetric
// eigen-decomposition (cyclic Jacobi) and SVD built on it, polar decomposition.
defineModule('linalg', {
  doc: '3x3 helpers: outer, identity, symmetric eigen-decomposition (Jacobi), SVD, polar rotation.',
  code: /* wgsl */`
const I3 = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
/** a ⊗ b = a bᵀ */
fn outer3(a: vec3f, b: vec3f) -> mat3x3f { return mat3x3f(a * b.x, a * b.y, a * b.z); }
fn trace3(m: mat3x3f) -> f32 { return m[0].x + m[1].y + m[2].z; }
fn diag3(d: vec3f) -> mat3x3f { return mat3x3f(d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z); }

fn si_jacobiRotate(A: ptr<function, mat3x3f>, V: ptr<function, mat3x3f>, p: i32, q: i32) {
  let apq = (*A)[q][p];
  if (abs(apq) < 1e-30) { return; }
  let theta = ((*A)[q][q] - (*A)[p][p]) / (2.0 * apq);
  let t = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
  let c = inverseSqrt(t * t + 1.0);
  let s = t * c;
  var J = I3;
  J[p][p] = c; J[q][q] = c; J[q][p] = s; J[p][q] = -s;
  *A = transpose(J) * (*A) * J;
  *V = (*V) * J;
}

struct Eig3 { V: mat3x3f, e: vec3f }
/** Symmetric A = V diag(e) Vᵀ (4 cyclic Jacobi sweeps: ~1e-6 relative accuracy in f32). */
fn eigSym3(A0: mat3x3f) -> Eig3 {
  var A = A0;
  var V = I3;
  for (var sweep = 0; sweep < 4; sweep++) {
    si_jacobiRotate(&A, &V, 0, 1);
    si_jacobiRotate(&A, &V, 0, 2);
    si_jacobiRotate(&A, &V, 1, 2);
  }
  return Eig3(V, vec3f(A[0].x, A[1].y, A[2].z));
}

struct SVD3 { U: mat3x3f, s: vec3f, V: mat3x3f }
/**
 * F = U diag(s) Vᵀ with U, V rotations (det = +1); s may have a negative last
 * entry if det(F) < 0. Built from the eigenvectors of FᵀF; U columns are
 * F v_i / s_i, re-orthogonalised for robustness.
 */
fn svd3(F: mat3x3f) -> SVD3 {
  let E = eigSym3(transpose(F) * F);
  // sort eigenpairs by decreasing eigenvalue
  var e = E.e; var V = E.V;
  if (e.x < e.y) { e = e.yxz; V = mat3x3f(V[1], V[0], -V[2]); }
  if (e.x < e.z) { e = e.zyx; V = mat3x3f(V[2], -V[1], V[0]); }
  if (e.y < e.z) { e = e.xzy; V = mat3x3f(-V[0], V[2], V[1]); }
  if (determinant(V) < 0.0) { V[2] = -V[2]; }
  var s = sqrt(max(e, vec3f(0.0)));
  var u0 = F * V[0];
  let l0 = length(u0);
  u0 = select(vec3f(1.0, 0.0, 0.0), u0 / l0, l0 > 1e-12);
  var u1 = F * V[1];
  u1 = u1 - dot(u1, u0) * u0;
  let l1 = length(u1);
  if (l1 > 1e-12) { u1 = u1 / l1; } else {
    u1 = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(u0.x) < 0.9);
    u1 = normalize(u1 - dot(u1, u0) * u0);
  }
  let u2 = cross(u0, u1);
  // sign of the smallest singular value carries det(F)
  if (dot(F * V[2], u2) < 0.0) { s.z = -s.z; }
  return SVD3(mat3x3f(u0, u1, u2), s, V);
}

/** Rotation part of F (polar decomposition) via the SVD: R = U Vᵀ. */
fn polarR(F: mat3x3f) -> mat3x3f {
  let d = svd3(F);
  return d.U * transpose(d.V);
}
`,
});
