// ShaderIntel — WebGPU physics + shader library.
//
//   import * as SI from './src/index.js';
//   const gpu = await SI.init();
//
// Everything is plain ES modules with zero dependencies.
export { init, Context, SIError } from './core/context.js';
export { GArray } from './core/buffer.js';
export { Kernel } from './core/kernel.js';
export { defineModule, link, moduleNames, moduleInfo, literal } from './core/wgsl.js';
export { layoutOf, structLayout, toHalf, fromHalf } from './core/types.js';
export { Scan } from './prims/scan.js';
export { Reduce } from './prims/reduce.js';
export { SpatialGrid } from './prims/grid.js';

export { View, OrbitCamera, Hud } from './render/view.js';
export { RenderProgram } from './render/program.js';
export { Fullscreen } from './render/fullscreen.js';
export { Spheres } from './render/spheres.js';
export { m4, v3 } from './render/math.js';
export { Stage, Lines } from './render/scene.js';
export { panel } from './ui.js';
export { MPM, MATERIALS } from './physics/mpm.js';
export { NBody, timeIt } from './physics/nbody.js';
export { Points } from './render/points.js';
export { Bloom } from './render/bloom.js';
export { SdfScene } from './render/sdf.js';
export { Glyphs } from './render/glyphs.js';
export { Boids } from './physics/boids.js';
export { Cloth } from './physics/cloth.js';
export { SurfaceMesh } from './render/mesh.js';
export { Smoke } from './physics/smoke.js';
export { Volume } from './render/volume.js';
export { FluidSurface } from './render/fluid.js';
export { RigidWorld, RayQuery, eulerQuat, RIGID_SHAPES, RIGID_FLAGS, JOINT_TYPES, JOINT_FLAGS } from './physics/rigid.js';
export { Bodies } from './render/bodies.js';
export { FluidRigidCoupling } from './physics/coupling.js';
import * as three from './interop/three.js';
export { three };
export { gemm, autotuneGemm, MLP, Adam, Population } from './nn/index.js';
import * as nn from './nn/index.js';
export { nn };
import * as net from './net/index.js';
export { net };
import * as verify from './verify/index.js';
export { verify };

// Shader library modules (registered on import).
import './shaders/index.js';

// Convenience factories on the context: gpu.view(), gpu.fullscreen(), gpu.mpm() ...
import { Context } from './core/context.js';
import { View } from './render/view.js';
import { Fullscreen } from './render/fullscreen.js';
import { Spheres } from './render/spheres.js';
import { RenderProgram } from './render/program.js';
import { MPM } from './physics/mpm.js';
import { Stage } from './render/scene.js';
import { NBody } from './physics/nbody.js';
import { Points } from './render/points.js';
import { SdfScene } from './render/sdf.js';
import { Glyphs } from './render/glyphs.js';
import { Boids } from './physics/boids.js';
import { Cloth } from './physics/cloth.js';
import { SurfaceMesh } from './render/mesh.js';
import { Smoke } from './physics/smoke.js';
import { Volume } from './render/volume.js';
import { RigidWorld } from './physics/rigid.js';
import { Bodies } from './render/bodies.js';
Object.assign(Context.prototype, {
  rigid(opts) { return new RigidWorld(this, opts); },
  bodies(world, opts) { return new Bodies(this, world, opts); },
  smoke(opts) { return new Smoke(this, opts); },
  volume(opts) { return new Volume(this, opts); },
  cloth(opts) { return new Cloth(this, opts); },
  mesh(opts) { return new SurfaceMesh(this, opts); },
  glyphs(opts) { return new Glyphs(this, opts); },
  boids(opts) { return new Boids(this, opts); },
  sdf(opts) { return new SdfScene(this, opts); },
  nbody(opts) { return new NBody(this, opts); },
  points(opts) { return new Points(this, opts); },
  stage(opts) { return new Stage(this, opts); },
  view(canvas, opts) { return new View(this, canvas, opts); },
  fullscreen(desc) { return new Fullscreen(this, desc); },
  program(desc) { return new RenderProgram(this, desc); },
  spheres(opts) { return new Spheres(this, opts); },
  mpm(opts) { return new MPM(this, opts); },
});
