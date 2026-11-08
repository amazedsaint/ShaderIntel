// Type-level test of types/shaderintel.d.ts: `npm run typecheck` compiles this file against the
// declarations (through package.json "exports", as a user's project would import them). Never run.
import * as SI from 'shaderintel';

export async function usage(canvas: HTMLCanvasElement): Promise<void> {
  const gpu: SI.Context = await SI.init({ powerPreference: 'high-performance' });
  const caps: boolean = gpu.caps.subgroups;

  // core
  const pos = gpu.array('vec4f', 1024);
  const k = gpu.kernel({ buffers: { pos }, uniforms: { dt: 'f32' }, imports: ['noise'], code: 'pos[i] += vec4f(curl3(pos[i].xyz) * u.dt, 0.0);' });
  await k.ready();
  k.run(1024, { dt: 1 / 60 });
  const data = await pos.read();
  new SI.Scan(gpu, 1024).run(gpu.array('u32', 1024));

  // rendering
  const view = gpu.view(canvas, { hud: true, camera: { distance: 10 } });
  view.add(gpu.stage({ height: 0 }));
  view.add(gpu.points({ data: pos }));
  view.run(({ dt }) => k.run(1024, { dt }));

  // rigid bodies, joints, queries
  const world = gpu.rigid({ maxBodies: 1024, deterministic: true, events: true, floor: 0 });
  const box = world.addBox({ position: [0, 1, 0], half: [0.5, 0.5, 0.5], rotation: [0, 0.3, 0], friction: 0.5 });
  const ball = world.addSphere({ position: [1, 3, 0], radius: 0.25, group: 2, mask: 1 });
  const pill = world.addCapsule({ position: [-1, 2, 0], radius: 0.1, halfHeight: 0.3, kinematic: true });
  const hinge = world.addJoint({ type: 'hinge', a: box, b: null, axis: [0, 0, 1], limits: [-1, 1], motor: { speed: 1, maxForce: 50 } });
  world.addJoint({ type: 'distance', a: ball, b: box, rope: true, length: 2, breakForce: 100 });
  world.setMotor(hinge, null);
  world.applyImpulse(ball, [0, 1, 0], [1, 3, 0]);
  world.moveKinematic(pill, [-1, 2.1, 0]);
  world.addForceField({ code: 'force += vec3f(u.wind, 0.0, 0.0) * b.mass.x;', uniforms: { wind: 'f32' }, values: { wind: 2 } }).set({ wind: 3 });
  world.step(1 / 60);
  view.add(gpu.bodies(world, { groups: 1 }));
  const poses: Float32Array = await world.readPoses();
  const { begin } = await world.readEvents();
  const hits = await world.raycast([{ origin: [0, 5, 0], dir: [0, -1, 0], maxDistance: 10 }]);
  const firstHit: number = hits[0].distance;
  const hash: string = await world.hash();
  const snap = world.snapshot();
  world.restore(snap);
  const state: SI.RigidState = await world.exportState(snap);
  world.importState(state);

  // fluid + coupling
  const sim = gpu.mpm({ grid: 64, size: [2, 1, 1], substeps: 8 });
  sim.add({ box: [[0, 0, 0], [1, 0.5, 1]] }, { material: 'water' });
  sim.couple(world, { friction: 0.2 });
  sim.step();

  // neural networks
  const net = new SI.nn.MLP(gpu, { layers: [2, 64, 3], act: 'relu', out: 'sigmoid', fourier: 8, maxBatch: 4096 });
  const loss = new SI.nn.MSELoss(gpu, 4096 * 3);
  const opt = new SI.Adam(gpu, net.params, { lr: 2e-3 });
  net.forward(net.input, 4096);
  net.backward(loss.run(net.output, gpu.array('f32', 4096 * 3), 4096 * 3), 4096);
  opt.step();
  SI.gemm(gpu, net.input, net.input, net.output, { M: 2, N: 2, K: 2, act: 'gelu', transB: true });
  const tuned = await SI.autotuneGemm(gpu, { n: 1024 });
  const tile: number = tuned.best.BM;
  const pop = new SI.Population(gpu, { layers: [8, 16, 2], count: 256 });
  const fit: { mean: number; best: number } = await pop.update(new Float32Array(256));

  // networking
  const [ca, cb] = new SI.net.LoopbackNetwork({ latency: 40, loss: 0.05 }).connect();
  const session = new SI.net.RollbackSession({
    sim: world, channel: ca, localPlayer: 0, players: 2,
    applyInputs: (tick: number, inputs: Uint8Array[]) => { if (inputs[1][0] & 1) world.applyImpulse(box, [0, 1, 0]); },
    onDesync: (tick, local, remote) => console.warn(tick, local, remote),
  });
  session.advance(new Uint8Array([1]));
  const tr: SI.Transcript = session.transcript();
  const ok: boolean = (await SI.net.verifyTranscript(tr, () => gpu.rigid({ deterministic: true }), (w, t, i) => {})).ok;
  const wire: string = SI.net.encode({ state });
  const back = SI.net.decode(wire);
  cb.reliable.onmessage = (m) => console.log(m);

  // verification
  const laws = await SI.verify.proveAll(gpu, SI, ['scan', 'gemm']);
  const cert = await SI.verify.freivalds(gpu, net.input, net.input, net.output, { M: 2, N: 2, K: 2, trials: 8 });
  const bound: number = cert.falseAcceptBound;

  // UI
  SI.panel('demo', [{ label: 'speed', min: 0, max: 2, step: 0.1, value: 1, onInput: (v) => console.log(v * 2) }, { label: 'reset', onClick: () => world.clear() }]);
  void [caps, data, poses, begin, firstHit, hash, tile, fit, tr, ok, back, laws, bound];
}
