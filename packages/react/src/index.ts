// shaderintel-react — React components and hooks for ShaderIntel.
//
//   import { Canvas, Stage, Physics, Box, Sphere } from 'shaderintel-react';
//
//   <Canvas camera={{ distance: 12, target: [0, 2, 0] }} hud>
//     <Stage />
//     <Physics>
//       <Box position={[0, 4, 0]} size={[1, 1, 1]} />
//       <Sphere position={[0.3, 7, 0]} radius={0.5} />
//     </Physics>
//   </Canvas>
export { Canvas } from './Canvas.js';
export type { CanvasProps } from './Canvas.js';
export { useShaderIntel, useGPU, useView, acquireGPU, DEVICE_LINGER_MS, ShaderIntelContext } from './context.js';
export type { RootState, FrameState, FrameCallback } from './context.js';
export { useFrame, useArray, useKernel, useLayer, useResource, useDisposable, useLiveValues } from './hooks.js';
export { Layer, Stage, Points, Spheres, Glyphs, Fullscreen, SdfScene, BoxLines } from './layers.js';
export type { LayerProps, StageProps, PointsProps, SpheresProps, GlyphsProps, FullscreenProps, SdfSceneProps, BoxLinesProps } from './layers.js';
export { Physics, RigidBody, Box, Sphere, Capsule, Joint, ForceField, useWorld, useOptionalWorld } from './physics.js';
export type { PhysicsProps, RigidBodyProps, RigidBodyApi, BodyType, JointProps, ForceFieldProps } from './physics.js';
export { Fluid, FluidBlock, Cloth, Smoke, Boids, NBody } from './simulations.js';
export type { FluidProps, FluidBlockProps, ClothProps, SmokeProps, BoidsProps, NBodyProps } from './simulations.js';
