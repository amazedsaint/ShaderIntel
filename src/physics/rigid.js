// Rigid bodies with Augmented Vertex Block Descent (Giles, Diaz & Yuksel,
// SIGGRAPH 2025), building on Vertex Block Descent (Chen et al., 2024).
//
// Every body is one 6-DOF "vertex". Per step:
//   commands  game-side edits (poses, velocities, impulses, flags, joint motors) are
//             queued on the CPU and applied in order by one GPU thread.
//   predict   inertial target y = x₀ + v Δt + g Δt²; external impulses (the `ext`
//             buffer, written by commands, force fields, fluid coupling, neural nets)
//             are applied here. Iterations start at x₀ + vΔt + a gΔt², where a ∈ [0, 1]
//             is how much the body actually fell last step (VBD's adaptive guess).
//   pairs     broadphase on bounding spheres in a counting-sort grid sized for ordinary
//             bodies (the few oversized ones are tested against everything). Each pair is
//             owned by one body, which keeps its pairs sorted; a second pass gives every
//             body the sorted list of pairs it joins as the second body.
//   contacts  narrowphase, one thread per pair, run twice: count, prefix-sum, write. Box–box
//             (SAT over 15 axes, face clipping, ≤ 4 points), sphere, capsule and floor
//             manifolds, generated with a margin (speculative) so fast bodies do not tunnel.
//             Every point inherits λ and k from the nearest point of the same pair last step
//             (GPU hash table); a point that was sticking keeps its anchors.
//   colour    Jones–Plassmann + first fit over contact and joint edges, double-buffered.
//   iterate   per colour, each body takes one Newton step on its local energy
//               G(x) = ½|x − y|²_M / Δt² + Σ_c [ ½ k_c C_c² + λ_c C_c ]
//             (contacts linearised at the step start, joint rows evaluated exactly), solving
//             H Δ = −g as 3×3 blocks in registers with 8 lanes accumulating a body's rows;
//             then one dual update per contact and per joint (augmented Lagrangian):
//               λ ← k C + λ (clamped to f_n ≤ 0, |f_t| ≤ μ|λ_n|, motor limits),
//               k ← min(k_max, k + β|C|) while the constraint is active and not sliding.
//   velocity  v = (x − x₀)/Δt, ω from the orientation change.
//
// Determinism (confluence): no stage depends on the order in which GPU threads run.
// Pair lists are sorted, contact slots come from prefix sums rather than atomic counters,
// every body visits its contacts in a fixed order, colouring reads only the previous round,
// and the only atomics left are integer sums and hash-table slots (whose values do not
// depend on order). The same inputs on the same device give a bit-identical state, which
// is what lockstep and rollback netcode need (see hash(), snapshot(), net/).
import { SpatialGrid } from '../prims/grid.js';
import { Scan } from '../prims/scan.js';
import { BODY_VEC4, CONTACT_VEC4, JOINT_VEC4, SHAPES, FLAGS, JOINT_TYPES, JOINT_FLAGS, NARROW_WGSL, RAY_WGSL } from './rigid-wgsl.js';

export { SHAPES as RIGID_SHAPES, FLAGS as RIGID_FLAGS, JOINT_TYPES, JOINT_FLAGS };

const MAX_COLORS = 16;
const COLOR_ROUNDS = 20;
const MAX_LARGE = 256;    // bodies too big for the broadphase grid, tested against everything
const LANES = 8;          // threads per body in the primal step
const BODIES_PER_WG = 64 / LANES;
const MAXP = 48;          // pairs a body owns (kept: the lowest partner indices)
const MAXR = 64;          // pairs a body joins as the second body
const MAXRL = 1024;       // ... for a large dynamic body (only its pairs with contacts)
const CAND = Math.max(MAXP, MAXR);

// Where body b's reverse list lives. Small bodies have MAXR slots of their own; a large body's
// count word is m | (k + 1) << 16 for large-list entry k, and its MAXRL slots follow the counts.
const reverseList = (NB, b) => /* wgsl */`
  let rw = rlist[${NB * MAXR}u + ${b}];
  let rn = min(rw & 0xffffu, select(${MAXR}u, ${MAXRL}u, rw > 0xffffu));
  let rb = select(${b} * ${MAXR}u, ${NB * (MAXR + 1)}u + ((rw >> 16u) - 1u) * ${MAXRL}u, rw > 0xffffu);`;
const NONE = 0xffffffff;
const CMD = { pose: 1, velocity: 2, impulse: 3, flags: 4, filter: 5, kinematic: 6, motor: 7, jointFlags: 8, limits: 9, props: 10 };

// Reads only the members of a body a constraint needs (spliced into kernels that bind B).
const POSE_WGSL = /* wgsl */`
fn pose(i: u32) -> Pose { return Pose(B[i].x.xyz, B[i].x0.xyz, B[i].q, B[i].q0); }`;

export class RigidWorld {
  /**
   * o: maxBodies, maxContacts, maxPairs, maxJoints, gravity, iterations, substeps,
   *    floor (y, or null for none), bounds [[min], [max]], deterministic (fixed colour budget),
   *    events (contact begin/end events). Solver knobs (defaults are the tested values):
   *    alpha 0.95 (fraction of an existing overlap kept per step), beta 30 (stiffness ramp,
   *    × m_eff/Δt² per metre), gamma 0.99 (stiffness decay between steps), kStart 1e-3 and
   *    kStartJoint 1 (× m_eff/Δt²), jointMassFloor 1 (kg: joint stiffness is scaled as if bodies weighed
   *    at least this, so light links carrying heavy loads stiffen fast enough — what AVBD's absolute β
   *    does for small bodies), contactMassFloor 0 (the same for contacts: helps a heavy box on light ones
   *    but makes piles of very light bodies jitter),
   *    contactTolerance 0 (opt-in: contact stiffness ≥ |λ|/ε; fixes heavy-on-light stacks of a few
   *    bodies but destabilises large piles), margin 0.02 (speculative distance), slop 5e-4,
   *    stickThreshold 0.01, postStabilize false.
   */
  constructor(ctx, o = {}) {
    this.ctx = ctx;
    this.maxBodies = Math.min(o.maxBodies || 16384, 65535);
    this.maxContacts = o.maxContacts || this.maxBodies * 16;
    this.maxPairs = o.maxPairs || this.maxBodies * 8;
    this.maxJoints = o.maxJoints || Math.max(64, this.maxBodies);
    this.maxCommands = o.maxCommands || 4096;
    this.gravity = o.gravity || [0, -9.81, 0];
    this.iterations = o.iterations ?? 8;
    this.substeps = o.substeps ?? 1;
    this.floor = o.floor === undefined ? 0 : o.floor;
    this.bounds = o.bounds || [[-40, -2, -40], [40, 40, 40]];
    this.deterministic = o.deterministic ?? false;
    this.postStabilize = o.postStabilize ?? false;
    this.alpha = o.alpha ?? 0.95;
    this.beta = o.beta ?? 30;
    this.gamma = o.gamma ?? 0.99;
    this.kStart = o.kStart ?? 1e-3;
    this.kStartJoint = o.kStartJoint ?? 1;
    this.jointMassFloor = o.jointMassFloor ?? 1;
    this.contactMassFloor = o.contactMassFloor ?? 0;
    this.contactTolerance = o.contactTolerance ?? 0;
    this.margin = o.margin ?? 0.02;
    this.slop = o.slop ?? 5e-4;
    this.stickThreshold = o.stickThreshold ?? 0.01;
    this.geoStiff = o.geoStiff ?? 1;
    this.fusedSolve = o.fusedSolve ?? 128;   // bodies up to which the whole solve is one dispatch (true / false force it)
    this.events = o.events ?? false;
    this.count = 0;
    this.uploaded = 0;
    this.cpu = [];
    this.free = [];
    this.dirty = new Set();
    this.joints = [];
    this.jointsUploaded = 0;
    this.jointsDirty = false;
    this.cmds = [];
    this.frame = 0;
    this._steps = 0;
    this.rSmall = 0.01;
    this.nLarge = 0;
    this.stats = { contacts: 0, colors: 0, pairs: 0 };
    this.forceFields = [];

    const A = (t, n, label, usage) => ctx.array(t, n, { label, usage });
    const tableSize = 1 << Math.ceil(Math.log2(this.maxContacts));
    this.tableSize = tableSize;
    this.B = A('vec4f', this.maxBodies * BODY_VEC4, 'rigid-bodies', GPUBufferUsage.VERTEX);
    // external impulses per body: (J, -) (L, -) in world units, applied and cleared each step
    this.ext = A('vec4f', this.maxBodies * 2, 'rigid-external-impulses');
    this.centers = A('vec4f', this.maxBodies, 'rigid-centers');
    this.pcount = A('u32', this.maxBodies + 1, 'rigid-pair-counts');
    this.poffset = A('u32', this.maxBodies + 1, 'rigid-pair-offsets');
    this.pairs = A('vec2u', this.maxPairs, 'rigid-pairs');
    this.pscratch = A('u32', this.maxBodies * MAXP, 'rigid-owned-partners');   // each body's sorted partners
    this.rlist = A('u32', this.maxBodies * (MAXR + 1) + MAX_LARGE * MAXRL, 'rigid-reverse-pairs');   // [lists | counts | large lists]
    this.ccount = A('u32', this.maxPairs + 1, 'rigid-contact-counts');
    this.coffset = A('u32', this.maxPairs + 1, 'rigid-contact-offsets');
    this.C = [A('vec4f', this.maxContacts * CONTACT_VEC4, 'rigid-contacts-0'), A('vec4f', this.maxContacts * CONTACT_VEC4, 'rigid-contacts-1')];
    // pair table: [keys (T) | (base, count) pairs (2T)] — one binding instead of two
    this.tbl = [A('u32', tableSize * 3, 'rigid-pairs-0'), A('u32', tableSize * 3, 'rigid-pairs-1')];
    this.J = A('vec4f', this.maxJoints * JOINT_VEC4, 'rigid-joints');
    this.jcsr = A('u32', this.maxBodies + 1 + 2 * this.maxJoints, 'rigid-joint-lists');
    this.jsetSize = 1 << Math.ceil(Math.log2(Math.max(16, 2 * this.maxJoints)));
    this.jset = A('u32', this.jsetSize * 2, 'rigid-jointed-pairs');
    this.large = A('u32', MAX_LARGE, 'rigid-large-bodies');
    this.color = [A('u32', this.maxBodies, 'rigid-color-0'), A('u32', this.maxBodies, 'rigid-color-1')];
    this.clist = A('u32', this.maxBodies, 'rigid-color-list');
    this.info = A('u32', MAX_COLORS * 4, 'rigid-color-info');
    // indirect args: one slot per colour, then the dual update, then the narrowphase
    this.args = A('u32', MAX_COLORS * 4 + 8, 'rigid-dispatch-args', GPUBufferUsage.INDIRECT);
    // counts: [1] contacts kept, [2] colours used, [3] candidate pairs, [4] events
    this.cnts = A('u32', 8, 'rigid-counts');
    this.cmdBuf = A('vec4f', this.maxCommands * 3, 'rigid-commands');
    this.maxEvents = o.maxEvents || 4096;
    this.eventBuf = A('vec4u', this.maxEvents, 'rigid-events');
    this.hashBuf = A('u32', 4, 'rigid-hash');
    this.pairScan = new Scan(ctx, this.maxBodies + 1);
    this.contactScan = new Scan(ctx, this.maxPairs + 1);
    this.grid = null;
    this._kernels();
  }

  // ------------------------------------------------------------------ bodies
  /** Add a box: { position, half: [hx,hy,hz], rotation: quat [x,y,z,w] | euler [rx,ry,rz], velocity, angularVelocity,
   *  density, friction, static, kinematic, sensor, fixedRotation, gravityScale, linearDamping, angularDamping, group, mask, userData } */
  addBox(o) { return this._add({ ...o, shape: SHAPES.box }); }
  /** Add a sphere: { position, radius, ... } */
  addSphere(o) { return this._add({ ...o, shape: SHAPES.sphere, half: [o.radius, o.radius, o.radius] }); }
  /** Add a capsule along its local y axis: { position, radius, halfHeight (of the straight part), ... } */
  addCapsule(o) { return this._add({ ...o, shape: SHAPES.capsule, half: [o.radius, o.halfHeight ?? o.radius, o.radius] }); }

  _add(o) {
    const id = this.free.length ? this.free.pop() : this.count;
    if (id >= this.maxBodies) return -1;
    const h = o.half || [0.5, 0.5, 0.5];
    const q = o.rotation ? (o.rotation.length === 4 ? o.rotation : eulerQuat(o.rotation)) : [0, 0, 0, 1];
    const rho = o.density ?? 1;
    let m, I, radius;
    if (o.shape === SHAPES.box) {
      m = rho * 8 * h[0] * h[1] * h[2];
      I = [(m / 3) * (h[1] ** 2 + h[2] ** 2), (m / 3) * (h[0] ** 2 + h[2] ** 2), (m / 3) * (h[0] ** 2 + h[1] ** 2)];
      radius = Math.hypot(...h);
    } else if (o.shape === SHAPES.capsule) {
      const r = h[0], H = 2 * h[1];
      const mc = rho * Math.PI * r * r * H, mh = rho * (2 / 3) * Math.PI * r ** 3;
      const Iax = mc * r * r / 2 + 2 * mh * 0.4 * r * r;
      const Ip = mc * (H * H / 12 + r * r / 4) + 2 * mh * (0.4 * r * r + H * H / 4 + 3 * H * r / 8);
      m = mc + 2 * mh; I = [Ip, Iax, Ip];
      radius = h[1] + r;
    } else {
      const r = h[0];
      m = (rho * 4 / 3) * Math.PI * r ** 3;
      I = [0.4 * m * r * r, 0.4 * m * r * r, 0.4 * m * r * r];
      radius = r;
    }
    let flags = 0;
    if (o.kinematic) flags |= FLAGS.kinematic;
    if (o.sensor) flags |= FLAGS.sensor;
    if (o.fixedRotation) flags |= FLAGS.fixedRotation;
    if (o.static || o.kinematic) { m = 0; I = [0, 0, 0]; }
    const p = o.position || [0, 0, 0], v = o.velocity || [0, 0, 0], w = o.angularVelocity || [0, 0, 0];
    const rec = new Float32Array(BODY_VEC4 * 4);
    const ru = new Uint32Array(rec.buffer);
    rec.set([p[0], p[1], p[2], radius], 0);
    rec.set(q, 4);
    rec.set([v[0], v[1], v[2], o.friction ?? 0.6], 8);
    rec.set([w[0], w[1], w[2], 0], 12);
    rec.set([p[0], p[1], p[2], radius], 16); rec.set(q, 20);
    rec.set([p[0], p[1], p[2], 0], 24); rec.set(q, 28);
    rec.set([h[0], h[1], h[2], o.shape], 32);
    rec.set([m, I[0], I[1], I[2]], 36);
    rec.set([o.gravityScale ?? 1, o.linearDamping ?? 0, o.angularDamping ?? 0, 0], 40);
    ru[44] = (o.group ?? 1) >>> 0; ru[45] = (o.mask ?? 0xffffffff) >>> 0; ru[46] = flags; ru[47] = (o.userData ?? 0) >>> 0;
    if (id === this.count) { this.cpu.push(rec); this.count++; }
    else {
      // a reused slot: drop queued edits aimed at its previous occupant (e.g. the removal itself)
      this.cmds = this.cmds.filter((c) => c.id !== id || (c.type >= CMD.motor && c.type <= CMD.limits));
      this.cpu[id] = rec; this.dirty.add(id);
    }
    this._pending = true;
    return id;
  }

  /** Remove a body (its slot is reused by later adds; joints attached to it are disabled). */
  remove(id) {
    this._cmd(CMD.flags, id, null, null, 0, FLAGS.disabled);
    this.free.push(id);
    this.joints.forEach((j, k) => { if (j.a === id || j.b === id) this.setJointEnabled(k, false); });
  }

  // ------------------------------------------------------------------ commands (applied in order, on the GPU)
  _cmd(type, id, a, c, hz = 0, hw = 0) {
    this.cmds.push({ type, id, a: a || [0, 0, 0, 0], c: c || [0, 0, 0, 0], hz, hw });
  }
  /** Teleport a body. */
  setPose(id, position, rotation) {
    const q = rotation ? (rotation.length === 4 ? rotation : eulerQuat(rotation)) : null;
    this._cmd(CMD.pose, id, [...position, q ? 1 : 0], q || [0, 0, 0, 1]);
  }
  setVelocity(id, v, w) { this._cmd(CMD.velocity, id, [...v, 1], w ? [...w, 1] : [0, 0, 0, 0]); }
  /** Impulse J (N·s) at an optional world point (default: the centre of mass). */
  applyImpulse(id, J, point) { this._cmd(CMD.impulse, id, [...J, 0], point ? [...point, 1] : [0, 0, 0, 0]); }
  /** Force F (N) applied for the next frame (dt), at an optional world point. */
  applyForce(id, F, point, dt = 1 / 60) { this.applyImpulse(id, F.map((f) => f * dt), point); }
  /** Angular impulse (N·m·s). */
  applyAngularImpulse(id, L) { this._cmd(CMD.impulse, id, [0, 0, 0, 0], [...L, 2]); }
  /** Move a kinematic body so it reaches the target pose at the end of this frame. */
  moveKinematic(id, position, rotation) {
    const q = rotation ? (rotation.length === 4 ? rotation : eulerQuat(rotation)) : [0, 0, 0, 0];
    this._cmd(CMD.kinematic, id, [...position, rotation ? 1 : 0], q);
  }
  setFilter(id, group, mask) { this._cmd(CMD.filter, id, null, null, group >>> 0, mask >>> 0); }
  setEnabled(id, on) { this._cmd(CMD.flags, id, null, null, on ? FLAGS.disabled : 0, on ? 0 : FLAGS.disabled); }
  setFlags(id, set = 0, clear = 0) { this._cmd(CMD.flags, id, null, null, clear, set); }
  /** (gravityScale, linearDamping, angularDamping) */
  setProps(id, { gravityScale = 1, linearDamping = 0, angularDamping = 0 } = {}) { this._cmd(CMD.props, id, [gravityScale, linearDamping, angularDamping, 0]); }

  // ------------------------------------------------------------------ joints
  /**
   * Add a joint between body a and body b (b: null or -1 for the world).
   * { type: 'ball' | 'hinge' | 'fixed' | 'distance' | 'slider', a, b, anchor: world point (default: b's
   *   centre, or a's for the world), anchorB (distance joints: the other end), axis: world axis
   *   (hinge, slider), limits: [lo, hi] (rad or m), motor: { speed, maxForce }, length (distance;
   *   default: current), rope (distance: only resists stretching), stiffness (N/m: a spring instead
   *   of a rigid joint), damping, breakForce, collideConnected (default false) }
   * Anchors and axes are given in world space and converted with the bodies' poses at the next step.
   */
  addJoint(o) {
    const k = this.joints.length;
    if (k >= this.maxJoints) return -1;
    const type = JOINT_TYPES[o.type || 'ball'];
    if (type === undefined) throw new Error(`RigidWorld: unknown joint type ${o.type}`);
    const a = o.a, b = o.b === undefined || o.b === null || o.b < 0 ? NONE : o.b;
    const pa = this.cpu[a] ? [...this.cpu[a].subarray(0, 3)] : [0, 0, 0];
    const anchor = o.anchor || (b === NONE ? pa : [...this.cpu[b].subarray(0, 3)]);
    const anchorB = o.anchorB || anchor;
    const axis = o.axis || [0, 1, 0];
    const al = Math.hypot(...axis) || 1;
    let flags = JOINT_FLAGS.worldAnchors;
    if (o.collideConnected) flags |= JOINT_FLAGS.collide;
    if (o.limits) flags |= JOINT_FLAGS.limit;
    if (o.motor) flags |= JOINT_FLAGS.motor;
    if (o.rope) flags |= JOINT_FLAGS.rope;
    const rec = new Float32Array(JOINT_VEC4 * 4);
    const ru = new Uint32Array(rec.buffer);
    ru[0] = a; ru[1] = b; ru[2] = type; ru[3] = flags;
    rec.set([anchor[0], anchor[1], anchor[2], o.breakForce ?? 0], 4);
    rec.set([anchorB[0], anchorB[1], anchorB[2], o.length ?? -1], 8);
    rec.set([0, 0, 0, 1], 12);
    rec.set([axis[0] / al, axis[1] / al, axis[2] / al, o.limits ? o.limits[0] : 0], 16);
    rec.set([axis[0] / al, axis[1] / al, axis[2] / al, o.limits ? o.limits[1] : 0], 20);
    rec.set([o.stiffness ?? 0, o.damping ?? 0, o.motor?.speed ?? 0, o.motor?.maxForce ?? 0], 24);
    this.joints.push({ a, b, type, flags, rec, enabled: true });
    this.jointsDirty = true;
    this._pending = true;
    return k;
  }
  /** Motor target speed (rad/s or m/s) and force/torque limit; speed null turns the motor off. */
  setMotor(j, speed, maxForce) { this._cmd(CMD.motor, j, [speed ?? 0, maxForce ?? 0, speed === null ? 0 : 1, 0]); }
  setLimits(j, lo, hi) { this._cmd(CMD.limits, j, [lo ?? 0, hi ?? 0, lo === null ? 0 : 1, 0]); }
  setJointEnabled(j, on) {
    this._cmd(CMD.jointFlags, j, null, null, on ? JOINT_FLAGS.disabled : 0, on ? 0 : JOINT_FLAGS.disabled);
    if (this.joints[j]) this.joints[j].enabled = on;
  }

  // ------------------------------------------------------------------ extension: force fields
  /**
   * A WGSL snippet run for every dynamic body before each frame: it sees `b` (Body), `i` and
   * the uniforms `u` (u.dt, u.time + your own) and adds to `force` / `torque` (world, N and N·m).
   *   world.addForceField({ code: 'force += vec3f(0.0, 0.0, u.wind) * b.mass.x;', uniforms: { wind: 'f32' }, values: { wind: 2 } })
   */
  addForceField({ code, uniforms = {}, imports = [], values = {} }) {
    const k = this.ctx.kernel({
      name: 'rigid-force-field', imports: ['rigid', ...imports],
      buffers: { B: { type: 'Body', access: 'read' }, ext: { type: 'vec4f' } },
      uniforms: { dt: 'f32', time: 'f32', ...uniforms },
      code: /* wgsl */`
        let b = B[i];
        if (b.mass.x <= 0.0 || (b.info.z & F_DISABLED) != 0u) { return; }
        var force = vec3f(0.0);
        var torque = vec3f(0.0);
        ${code}
        ext[2u * i] += vec4f(force * u.dt, 0.0);
        ext[2u * i + 1u] += vec4f(torque * u.dt, 0.0);`,
    });
    k.set(values);
    const field = { kernel: k, enabled: true, set: (v) => { k.set(v); return field; } };
    this.forceFields.push(field);
    return field;
  }

  // ------------------------------------------------------------------ upload
  /** Upload bodies and joints added since the last commit (called by step). */
  commit() {
    if (!this._pending) return;
    this.ctx.flush();   // queue writes land before unsubmitted work: submit that first
    const from = this.uploaded;
    if (this.count > from) {
      const data = new Float32Array((this.count - from) * BODY_VEC4 * 4);
      for (let k = from; k < this.count; k++) data.set(this.cpu[k], (k - from) * BODY_VEC4 * 4);
      this.B.write(data, from * BODY_VEC4);
    }
    for (const id of this.dirty) if (id < from) this.B.write(this.cpu[id], id * BODY_VEC4);
    this.dirty.clear();
    if (from === 0) { for (const t of this.tbl) this.ctx.clear(t); this.ctx.clear(this.ext); }
    this.uploaded = this.count;
    if (this.jointsDirty) this._uploadJoints();
    this._pending = false;
    // Grid cells are sized for ordinary bodies; the few much larger ones (static walls, a
    // boulder) go to a short list every body tests directly, so one big body cannot make the
    // cells huge and the broadphase quadratic.
    // Only true outliers are large: bodies of similar size (a pile of mixed boxes) all share the
    // grid, because a large dynamic body costs every body a test and a reverse-list scan.
    const radii = this.cpu.map((r) => r[3]).sort((a, b) => a - b);
    const maxLarge = Math.min(MAX_LARGE, Math.ceil(0.02 * radii.length));
    const typical = radii.length ? radii[Math.max(0, radii.length - 1 - maxLarge)] : 0.01;
    let top = radii.length - 1;
    while (top > 0 && radii[top] > 2 * typical) top--;
    this.rSmall = radii.length ? radii[top] : 0.01;
    const large = [];
    this.cpu.forEach((r, k) => { if (r[3] > this.rSmall) large.push(k); });
    this.nLarge = large.length;
    if (large.length) this.large.write(new Uint32Array(large));
    // cells fit the ordinary bodies, but the grid stays under ~4M cells (wide bounds, tiny bodies)
    const [lo, hi] = this.bounds;
    const volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    const cell = Math.max(0.05, 2 * this.rSmall + this.margin, 1.01 * Math.cbrt(volume / (1 << 22)));
    if (!this.grid || Math.abs(this.grid.cellSize - cell) > 1e-6) {
      this.grid = new SpatialGrid(this.ctx, { maxCount: this.maxBodies, min: this.bounds[0], max: this.bounds[1], cellSize: cell });
      this._gridKernels();
    }
  }

  _uploadJoints() {
    const n = this.joints.length;
    // new joint records (existing ones live on the GPU: they carry multipliers and broken flags)
    if (n > this.jointsUploaded) {
      const data = new Float32Array((n - this.jointsUploaded) * JOINT_VEC4 * 4);
      for (let k = this.jointsUploaded; k < n; k++) data.set(this.joints[k].rec, (k - this.jointsUploaded) * JOINT_VEC4 * 4);
      this.J.write(data, this.jointsUploaded * JOINT_VEC4);
      this.jointsUploaded = n;
    }
    // per-body joint lists (CSR) and the set of jointed pairs that must not collide
    const nb = this.maxBodies;
    const lists = Array.from({ length: this.count }, () => []);
    this.joints.forEach((j, k) => { lists[j.a]?.push(k); if (j.b !== NONE) lists[j.b]?.push(k); });
    const csr = new Uint32Array(nb + 1 + 2 * this.maxJoints);
    let off = 0;
    for (let b = 0; b <= nb; b++) { csr[b] = off; if (b < lists.length) for (const k of lists[b]) csr[nb + 1 + off++] = k; }
    this.jcsr.write(csr);
    const T = this.jsetSize, set = new Uint32Array(2 * T);
    const hash = (x) => { x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); return (x ^ (x >>> 16)) >>> 0; };
    this.joints.forEach((j, k) => {
      if (j.b === NONE || (j.flags & JOINT_FLAGS.collide)) return;
      const key = (Math.min(j.a, j.b) * (nb + 1) + Math.max(j.a, j.b) + 1) >>> 0;
      let s = hash(key) & (T - 1);
      while (set[s] !== 0 && set[s] !== key) s = (s + 1) & (T - 1);
      set[s] = key; set[T + s] = k;
    });
    this.jset.write(set);
    this.jointsDirty = false;
  }

  clear() {
    this.count = 0; this.uploaded = 0; this.cpu = []; this.free = []; this.dirty.clear();
    this.joints = []; this.jointsUploaded = 0; this.jointsDirty = true; this.cmds = [];
    this._pending = true; this.stats = { contacts: 0, colors: 0, pairs: 0 };
  }

  // Dispatching an empty colour still costs a pass barrier (~16 µs on Apple GPUs), so only the
  // colours recently needed (+2 headroom) are dispatched. A body that would need more shares the
  // last colour until the next stats read raises the budget. Deterministic worlds always
  // dispatch every colour, so the result never depends on when stats arrived.
  _colorBudget() {
    const c = this.stats.colors;
    return this.deterministic || !(c > 0) ? MAX_COLORS : Math.min(MAX_COLORS, c + 2);
  }

  // ------------------------------------------------------------------ kernels
  _kernels() {
    const ctx = this.ctx;
    const imports = ['rigid'];
    const NB = this.maxBodies;
    this.cmdK = ctx.kernel({
      name: 'rigid-commands', imports, workgroupSize: 1, guard: false,
      buffers: { B: { type: 'Body' }, ext: { type: 'vec4f' }, cmds: { type: 'vec4f', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { count: 'u32', dt: 'f32' },
      code: /* wgsl */`
        for (var k = 0u; k < u.count; k++) {
          let h = bitcast<vec4u>(cmds[k * 3u]);
          let a = cmds[k * 3u + 1u];
          let c = cmds[k * 3u + 2u];
          let id = h.y;
          if (h.x >= 7u && h.x <= 9u) {
            var j = J[id];
            if (h.x == 7u) { j.prm.z = a.x; j.prm.w = a.y; j.ab.w = (j.ab.w & ~JF_MOTOR) | select(0u, JF_MOTOR, a.z != 0.0); }
            if (h.x == 8u) { j.ab.w = (j.ab.w & ~h.z) | h.w; }
            if (h.x == 9u) { j.axa.w = a.x; j.axb.w = a.y; j.ab.w = (j.ab.w & ~JF_LIMIT) | select(0u, JF_LIMIT, a.z != 0.0); }
            J[id] = j;
            continue;
          }
          var b = B[id];
          switch h.x {
            case 1u: { b.x = vec4f(a.xyz, b.x.w); if (a.w != 0.0) { b.q = normalize(c); } }
            case 2u: { b.v = vec4f(a.xyz, b.v.w); if (c.w != 0.0) { b.w = vec4f(c.xyz, b.w.w); } }
            case 3u: {
              ext[2u * id] += vec4f(a.xyz, 0.0);
              if (c.w == 1.0) { ext[2u * id + 1u] += vec4f(cross(c.xyz - b.x.xyz, a.xyz), 0.0); }
              if (c.w == 2.0) { ext[2u * id + 1u] += vec4f(c.xyz, 0.0); }
            }
            case 4u: { b.info.z = (b.info.z & ~h.z) | h.w; }
            case 5u: { b.info.x = h.z; b.info.y = h.w; }
            case 6u: {
              b.v = vec4f((a.xyz - b.x.xyz) / u.dt, b.v.w);
              if (a.w != 0.0) { b.w = vec4f(rotvec(normalize(c), b.q) / u.dt, b.w.w); }
            }
            case 10u: { b.props = a; }
            default: {}
          }
          B[id] = b;
        }`,
    });
    this.predictK = ctx.kernel({
      name: 'rigid-predict', imports,
      buffers: { B: { type: 'Body' }, centers: { type: 'vec4f' }, ext: { type: 'vec4f' } },
      uniforms: { dt: 'f32', g: 'vec3f' },
      code: /* wgsl */`
        var b = B[i];
        b.x0 = b.x; b.q0 = b.q;
        let J = ext[2u * i].xyz;
        let L = ext[2u * i + 1u].xyz;
        ext[2u * i] = vec4f(0.0);
        ext[2u * i + 1u] = vec4f(0.0);
        if ((b.info.z & F_DISABLED) != 0u) { centers[i] = vec4f(1e9, 1e9, 1e9, 0.0); B[i] = b; return; }
        centers[i] = vec4f(b.x.xyz, b.x.w);
        if (b.mass.x > 0.0) {
          var v = b.v.xyz + J / b.mass.x;
          var w = b.w.xyz + invInertiaWorld(b) * L;
          v /= 1.0 + u.dt * b.props.y;
          w /= 1.0 + u.dt * b.props.z;
          if ((b.info.z & F_FIXED_ROT) != 0u) { w = vec3f(0.0); }
          let g = u.g * b.props.x;
          b.y = vec4f(b.x.xyz + v * u.dt + g * (u.dt * u.dt), 0.0);
          b.qy = qinteg(b.q, w, u.dt);
          // adaptive initial guess: as much of the gravity step as the body actually took last time
          let gl = length(g);
          var aw = 0.0;
          if (gl > 0.0) { aw = clamp((dot(b.v.xyz, g / gl) - b.w.w) / (u.dt * gl), 0.0, 1.0); }
          b.v = vec4f(v, b.v.w);
          b.w = vec4f(w, b.w.w);
          b.x = vec4f(b.x.xyz + v * u.dt + g * (aw * u.dt * u.dt), b.x.w);
          b.q = b.qy;
        } else if ((b.info.z & F_KINEMATIC) != 0u) {
          b.x = vec4f(b.x.xyz + b.v.xyz * u.dt, b.x.w);
          b.q = qinteg(b.q, b.w.xyz, u.dt);
          b.y = b.x; b.qy = b.q;
        } else { b.y = b.x; b.qy = b.q; }
        B[i] = b;`,
    });
    // Joints: convert world-space anchors, warm start, and record C at the start of the step.
    this.jointPrepK = ctx.kernel({
      name: 'rigid-joint-prep', imports,
      buffers: { B: { type: 'Body', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { gammaLam: 'f32', gammaK: 'f32', kStart: 'f32', ih2: 'f32', massFloor: 'f32' },
      shared: /* wgsl */`
        // rows that are positional (stiffness scaled by mass) rather than angular (by inertia)
        fn linearRows0(t: u32) -> vec4<bool> {
          if (t == J_DISTANCE) { return vec4<bool>(true, false, false, false); }
          if (t == J_SLIDER) { return vec4<bool>(true, true, false, false); }
          return vec4<bool>(true, true, true, false);
        }
        fn linearRows1(t: u32) -> vec4<bool> { return vec4<bool>(false, t == J_SLIDER, t == J_SLIDER, false); }`,
      code: /* wgsl */`
        var j = J[i];
        if ((j.ab.w & (JF_BROKEN | JF_DISABLED)) != 0u) { return; }
        let A = B[j.ab.x];
        let hasB = j.ab.y != STATIC;
        var Bb = A;
        if (hasB) { Bb = B[j.ab.y]; }
        if ((j.ab.w & JF_WORLD_ANCHORS) != 0u) {
          let wa = j.ra.xyz; let wb = j.rb.xyz; let ax = j.axa.xyz;
          j.ra = vec4f(qrot(qconj(A.q0), wa - A.x0.xyz), j.ra.w);
          j.axa = vec4f(qrot(qconj(A.q0), ax), j.axa.w);
          if (hasB) {
            j.rb = vec4f(qrot(qconj(Bb.q0), wb - Bb.x0.xyz), j.rb.w);
            j.axb = vec4f(qrot(qconj(Bb.q0), ax), j.axb.w);
            j.qrel = qmul(qconj(A.q0), Bb.q0);
          } else { j.qrel = qconj(A.q0); }
          if (j.rb.w < 0.0) { j.rb.w = length(wa - wb); }   // distance joints: rest length = current
          j.ab.w = j.ab.w & ~JF_WORLD_ANCHORS;
          j.lam0 = vec4f(0.0); j.lam1 = vec4f(0.0);
          j.k0 = vec4f(0.0); j.k1 = vec4f(0.0);
        }
        // warm start (AVBD eq. 19) and the minimum stiffness, in units of the effective mass / inertia
        var m = A.mass.x;
        var I = (A.mass.y + A.mass.z + A.mass.w) / 3.0;
        if (hasB && Bb.mass.x > 0.0 && (m == 0.0 || Bb.mass.x < m)) { m = Bb.mass.x; I = (Bb.mass.y + Bb.mass.z + Bb.mass.w) / 3.0; }
        let kl = u.kStart * max(m, u.massFloor) * u.ih2;
        let ka = u.kStart * max(I, u.massFloor * 0.01) * u.ih2;
        j.lam0 *= u.gammaLam; j.lam1 *= u.gammaLam;
        j.k0 = max(j.k0 * u.gammaK, select(vec4f(ka), vec4f(kl), linearRows0(j.ab.z)));
        j.k1 = max(j.k1 * u.gammaK, select(vec4f(ka), vec4f(kl), linearRows1(j.ab.z)));
        // constraint values at the start of the step
        let P0A = Pose(A.x0.xyz, A.x0.xyz, A.q0, A.q0);
        var P0B = P0A;
        if (hasB) { P0B = Pose(Bb.x0.xyz, Bb.x0.xyz, Bb.q0, Bb.q0); }
        let nrow = jointRows(j.ab.z);
        var c0 = vec4f(0.0); var c1 = vec4f(0.0);
        for (var row = 0u; row < nrow; row++) {
          let C = jointRow(j, P0A, P0B, hasB, row, 0.0).C;
          if (row < 4u) { c0[row] = C; } else { c1[row - 4u] = C; }
        }
        // motor rows keep the axis coordinate itself (angle or slide) at the start of the step
        if (j.ab.z == J_HINGE) { c1.y = -hingeAngle(j, A.q0, select(vec4f(0.0, 0.0, 0.0, 1.0), Bb.q0, hasB)); }
        if (j.ab.z == J_SLIDER) {
          let aw = qrot(A.q0, j.axa.xyz);
          var pb = j.rb.xyz;
          if (hasB) { pb = Bb.x0.xyz + qrot(Bb.q0, j.rb.xyz); }
          c1.y = dot(A.x0.xyz + qrot(A.q0, j.ra.xyz) - pb, aw);
        }
        j.c00 = c0; j.c01 = c1;
        J[i] = j;`,
    });
    this.colorInitK = ctx.kernel({ name: 'rigid-color-init', buffers: { a: { type: 'u32' }, b: { type: 'u32' } }, code: 'a[i] = 0xffffffffu; b[i] = 0xffffffffu;' });
    this.bucketK = ctx.kernel({
      name: 'rigid-bucket', imports,
      buffers: { B: { type: 'Body', access: 'read' }, color: { type: 'u32', access: 'read' }, info: { type: 'u32', atomic: true } },
      code: `if (B[i].mass.x <= 0.0 || (B[i].info.z & F_DISABLED) != 0u) { return; }
        atomicAdd(&info[min(color[i], ${MAX_COLORS - 1}u) * 4u], 1u);`,
    });
    this.offsetsK = ctx.kernel({
      name: 'rigid-offsets', workgroupSize: 1, guard: false,
      buffers: { info: { type: 'u32' }, args: { type: 'u32' }, counter: { type: 'u32' }, coffset: { type: 'u32', access: 'read' } },
      uniforms: { maxContacts: 'u32', maxPairs: 'u32' },
      code: /* wgsl */`
        var off = 0u; var used = 0u;
        for (var c = 0u; c < ${MAX_COLORS}u; c++) {
          let n = info[c * 4u];
          info[c * 4u + 1u] = off; info[c * 4u + 2u] = off;
          args[c * 4u] = (n + ${BODIES_PER_WG - 1}u) / ${BODIES_PER_WG}u; args[c * 4u + 1u] = 1u; args[c * 4u + 2u] = 1u;
          off += n;
          if (n > 0u) { used = c + 1u; }
        }
        let nc = min(coffset[u.maxPairs], u.maxContacts);
        counter[1] = nc; counter[2] = used;
        args[${MAX_COLORS}u * 4u] = (nc + 63u) / 64u; args[${MAX_COLORS}u * 4u + 1u] = 1u; args[${MAX_COLORS}u * 4u + 2u] = 1u;`,
    });
    this.scatterK = ctx.kernel({
      name: 'rigid-scatter', imports,
      buffers: { B: { type: 'Body', access: 'read' }, color: { type: 'u32', access: 'read' }, info: { type: 'u32', atomic: true }, clist: { type: 'u32' } },
      code: `if (B[i].mass.x <= 0.0 || (B[i].info.z & F_DISABLED) != 0u) { return; }
        let slot = atomicAdd(&info[min(color[i], ${MAX_COLORS - 1}u) * 4u + 2u], 1u); clist[slot] = i;`,
    });
    this.pairArgsK = ctx.kernel({
      name: 'rigid-pair-args', workgroupSize: 1, guard: false,
      buffers: { poffset: { type: 'u32', access: 'read' }, args: { type: 'u32' }, counter: { type: 'u32' } },
      uniforms: { count: 'u32', maxPairs: 'u32' },
      code: /* wgsl */`
        let np = min(poffset[u.count], u.maxPairs);
        counter[3] = np;
        let o = ${MAX_COLORS + 1}u * 4u;
        args[o] = (np + 63u) / 64u; args[o + 1u] = 1u; args[o + 2u] = 1u;`,
    });
    this.narrowCountK = ctx.kernel({
      name: 'rigid-narrow-count', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, counter: { type: 'u32', access: 'read' },
        ccount: { type: 'u32' } },
      uniforms: { floorY: 'f32', margin: 'f32' },
      shared: NARROW_WGSL,
      code: /* wgsl */`
        // grid-stride over the pairs: any dispatch size covers them all (sizes are estimates)
        for (var gi = i; gi < counter[3]; gi += nwg.x * nwg.y * WG) {
          let pr = pairs[gi];
          let A = B[pr.x];
          var nc = 0u;
          if (pr.y == STATIC) { nc = floorManifold(A, u.floorY, u.margin).count; }
          else {
            let Bo = B[pr.y];
            nc = pairManifold(A, Bo, u.margin).count;
            if (((A.info.z | Bo.info.z) & F_SENSOR) != 0u) { nc = min(nc, 1u); }
          }
          ccount[gi] = nc;
        }`,
    });
    this.narrowWriteK = ctx.kernel({
      name: 'rigid-narrow-write', imports, workgroupSize: 64, guard: false,
      buffers: {
        B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, counter: { type: 'u32', access: 'read' },
        coffset: { type: 'u32', access: 'read' }, C: { type: 'Contact' }, tbl: { type: 'u32', atomic: true },
        Cp: { type: 'Contact', access: 'read' }, tblP: { type: 'u32', access: 'read' },
      },
      uniforms: { floorY: 'f32', margin: 'f32', maxContacts: 'u32', tableMask: 'u32', nBodies: 'u32',
        gammaLam: 'f32', gammaK: 'f32', kStart: 'f32', ih2: 'f32', massFloor: 'f32', stickThreshold: 'f32' },
      shared: NARROW_WGSL + /* wgsl */`
        fn pairKey(a: u32, b: u32) -> u32 { return a * (u.nBodies + 1u) + select(b, u.nBodies, b == STATIC) + 1u; }
        fn lookupPrev(key: u32) -> vec2u {
          let T = u.tableMask + 1u;
          var s = pcg(key) & u.tableMask;
          for (var probe = 0u; probe < 32u; probe++) {
            let k = tblP[s];
            if (k == key) { return vec2u(tblP[T + 2u * s], tblP[T + 2u * s + 1u]); }
            if (k == 0u) { break; }
            s = (s + 1u) & u.tableMask;
          }
          return vec2u(0u, 0u);
        }
        fn insertPair(key: u32, base: u32, n: u32) {
          let T = u.tableMask + 1u;
          var s = pcg(key) & u.tableMask;
          for (var probe = 0u; probe < 64u; probe++) {
            let r = atomicCompareExchangeWeak(&tbl[s], 0u, key);
            if (r.exchanged || r.old_value == key) {
              atomicStore(&tbl[T + 2u * s], base); atomicStore(&tbl[T + 2u * s + 1u], n);
              return;
            }
            if (r.old_value != 0u) { s = (s + 1u) & u.tableMask; }
          }
        }`,
      code: /* wgsl */`
       for (var gi = i; gi < counter[3]; gi += nwg.x * nwg.y * WG) {
        let i = gi;   // the pair this iteration writes (grid-stride: any dispatch size covers all pairs)
        let base = coffset[i];
        let n = coffset[i + 1u] - base;
        if (n == 0u || base + n > u.maxContacts) { continue; }
        let pr = pairs[i];
        let a = pr.x; let b = pr.y;
        let A = B[a];
        var M: Manifold;
        var sensor = 0u;
        var Bb = A;
        if (b == STATIC) { M = floorManifold(A, u.floorY, u.margin); }
        else { Bb = B[b]; M = pairManifold(A, Bb, u.margin); if (((A.info.z | Bb.info.z) & F_SENSOR) != 0u) { sensor = 1u; } }
        let key = pairKey(a, b);
        insertPair(key, base, n);
        let prev = lookupPrev(key);
        var meff = A.mass.x;
        var mu = A.v.w;
        if (b != STATIC) {
          mu = sqrt(mu * Bb.v.w);
          if (Bb.mass.x > 0.0 && (meff == 0.0 || Bb.mass.x < meff)) { meff = Bb.mass.x; }
        }
        let mih = max(meff, u.massFloor) * u.ih2;
        for (var k = 0u; k < n; k++) {
          let pa = M.pa[k]; let pb = M.pb[k];
          var c: Contact;
          c.ab = vec4u(a, b, 0u, sensor);
          c.n = vec4f(M.nrm[k], mu);
          let ra = qrot(qconj(A.q0), pa - A.x0.xyz);
          var rb = vec4f(pb, 0.0);
          if (b != STATIC) { rb = vec4f(qrot(qconj(Bb.q0), pb - Bb.x0.xyz), 0.0); }
          c.ra = vec4f(ra, 0.0);
          c.rb = rb;
          c.lam = vec4f(0.0);
          c.k = vec4f(u.kStart * mih, u.kStart * mih, u.kStart * mih, mih);
          // Warm start from the nearest point of the same pair last step. A rolling contact moves
          // about |Δv|·Δt over the body each step, so the search radius grows with the relative
          // speed; a one-point manifold (sphere, capsule end) is always the same contact.
          let vrel = length(A.v.xyz - select(vec3f(0.0), Bb.v.xyz, b != STATIC));
          var bestD = max(0.05 * max(A.x.w, 0.05), 1.5 * vrel / sqrt(u.ih2));
          if (n == 1u && prev.y == 1u) { bestD = 1e30; }
          for (var q = 0u; q < prev.y; q++) {
            let pc = Cp[prev.x + q];
            let dd = length(pc.ra.xyz - ra);
            if (dd < bestD) {
              bestD = dd;
              c.lam = pc.lam * u.gammaLam;
              c.k = vec4f(max(pc.k.xyz * u.gammaK, vec3f(u.kStart * mih)), mih);
              // A sticking contact keeps its anchors (static friction remembers where it started) as
              // long as the remembered material point is still the touching one. A rolling body
              // brings a new point into contact every step; its old anchor rises off the ground,
              // and keeping it would let the body sink and feed energy into the roll.
              let still = length(A.x0.xyz + qrot(A.q0, pc.ra.xyz) - pa) < u.stickThreshold;
              if (pc.ab.z != 0u && still) { c.ra = vec4f(pc.ra.xyz, 0.0); c.rb = vec4f(pc.rb.xyz, 0.0); }
              else { c.ra = vec4f(ra, 0.0); c.rb = rb; }
            }
          }
          if (sensor != 0u) { c.lam = vec4f(0.0); }
          C[base + k] = c;
        }
       }`,
    });
    // One body per 8 lanes: each lane accumulates a share of the body's contacts and joint rows,
    // a shared-memory tree sums the 8 partial Hessians/gradients, and lane 0 solves. Colours are
    // often small (tens of bodies), so spreading a body over lanes keeps the GPU occupied.
    const PRIMAL_WGSL = /* wgsl */`
        var<workgroup> red: array<array<vec4f, 9>, WG>;   // one slot per invocation (WG: the kernel's workgroup size)
        var<workgroup> colorCount: u32;
        struct Acc { Hll: mat3x3f, Hla: mat3x3f, Haa: mat3x3f, gl: vec3f, ga: vec3f }
        fn addRow(acc: ptr<function, Acc>, jl: vec3f, ja: vec3f, f: f32, kk: f32) {
          (*acc).gl += f * jl; (*acc).ga += f * ja;
          (*acc).Hll += outer3(kk * jl, jl); (*acc).Hla += outer3(kk * jl, ja); (*acc).Haa += outer3(kk * ja, ja);
        }
        fn addContact(acc: ptr<function, Acc>, ci: u32, bi: u32, me: Pose) {
          let c = C[ci];
          if (c.ab.w != 0u) { return; }   // sensor: reports overlap, exerts nothing
          let isA = c.ab.x == bi;
          let hasB = c.ab.y != STATIC;
          var L: Lin;
          if (isA) { L = contactLin(c, me, pose(select(bi, c.ab.y, hasB)), hasB); }
          else { L = contactLin(c, pose(c.ab.x), me, true); }
          let nrm = c.n.xyz;
          let r = select(L.rb0, L.ra0, isA);
          let sg = select(-1.0, 1.0, isA);
          let T = basis(nrm);
          // Forces are clamped to their bounds (normal ≤ 0, |friction| ≤ μ|λ_n|) but the penalty
          // stiffness always enters the Hessian, as in AVBD: dropping an inactive contact from H lets
          // the next Newton step jump straight through it, and the iterations then 2-cycle.
          let fN = min(0.0, c.k.x * contactCn(L, nrm, u.alpha, u.slop) + c.lam.x);
          // isotropic Coulomb cone: the tangential force is clamped to a disc of radius μ|λ_n|
          let bound = c.n.w * abs(c.lam.x);
          var ft = vec2f(c.k.y * ((1.0 - u.alpha) * dot(T[0], L.gap) + dot(T[0], L.rel)) + c.lam.y,
                         c.k.z * ((1.0 - u.alpha) * dot(T[1], L.gap) + dot(T[1], L.rel)) + c.lam.z);
          let fl = length(ft);
          if (fl > bound) { ft *= bound / max(fl, 1e-30); }
          addRow(acc, nrm * sg, cross(r, nrm) * sg, fN, c.k.x);
          addRow(acc, T[0] * sg, cross(r, T[0]) * sg, ft.x, c.k.y);
          addRow(acc, T[1] * sg, cross(r, T[1]) * sg, ft.y, c.k.z);
          // geometric stiffness bound on the angular block keeps H well conditioned under large forces
          (*acc).Haa += diag3(vec3f(abs(fN) * length(r) * u.geoStiff));
        }
        fn addJoint(acc: ptr<function, Acc>, ji: u32, bi: u32, me: Pose) {
          let jt = J[ji];
          if ((jt.ab.w & (JF_BROKEN | JF_DISABLED | JF_WORLD_ANCHORS)) != 0u) { return; }
          let isA = jt.ab.x == bi;
          let hasB = jt.ab.y != STATIC;
          var PA = me; var PB = me;
          if (isA) { if (hasB) { PB = pose(jt.ab.y); } } else { PA = pose(jt.ab.x); }
          let soft = jt.prm.x > 0.0;
          let nrow = jointRows(jt.ab.z);
          var gsum = 0.0;
          for (var row = 0u; row < nrow; row++) {
            let jr = jointRow(jt, PA, PB, hasB, row, u.dt);
            if (jr.kind == JK_OFF) { continue; }
            let jl = select(jr.lb, jr.la, isA);
            let ja = select(jr.ab, jr.aa, isA);
            let lam = rowGet(jt.lam0, jt.lam1, row);
            let kp = rowGet(jt.k0, jt.k1, row);
            let c0 = rowGet(jt.c00, jt.c01, row);
            var f: f32; var kk = kp;
            if (jr.kind == JK_MOTOR) { f = clamp(kp * jr.C + lam, -jt.prm.w, jt.prm.w); }
            else if (soft) {
              // spring: f = k C + d (C − C₀)/Δt
              kk = jt.prm.x + jt.prm.y / u.dt;
              f = jt.prm.x * jr.C + jt.prm.y * (jr.C - c0) / u.dt;
              if (jr.kind == JK_GE) { f = min(f, 0.0); }
            } else {
              let Cs = jr.C - u.alpha * select(c0, min(c0, 0.0), jr.kind == JK_GE);
              f = kp * Cs + lam;
              if (jr.kind == JK_GE) { f = min(f, 0.0); }
            }
            addRow(acc, jl, ja, f, kk);
            // lumped geometric stiffness (AVBD §3.5): |f| times the curvature of C. Positional rows bend
            // with the body's rotation (≈ |r|); distance rows also curve along the line (1/len), which is
            // what keeps a tensioned rope from overshooting sideways.
            if (dot(jl, jl) > 0.0) { gsum += abs(f) * length(ja); }
            (*acc).Hll += diag3(vec3f(abs(f) * jr.geo * u.geoStiff));
          }
          (*acc).Haa += diag3(vec3f(gsum * u.geoStiff));
        }
        // One pass over BODIES_PER_WG bodies of a colour (slots base.., of n, listed from clist[first]):
        // 8 lanes per body accumulate contact and joint rows, a shared-memory tree sums them, lane 0
        // takes the Newton step. Shared by rigid-primal (one colour, many workgroups) and
        // rigid-solve-fused (everything in one workgroup); the storage barrier makes the new poses
        // visible to the rest of the workgroup before the next colour.
        fn primalPass(base: u32, n: u32, first: u32, lidx: u32) {
        let lane = lidx % ${LANES}u;
        let slot = base + lidx / ${LANES}u;
        var bi = 0u;
        var live = false;
        var acc: Acc;
        acc.Hll = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
        acc.Hla = acc.Hll; acc.Haa = acc.Hll;
        acc.gl = vec3f(0.0); acc.ga = vec3f(0.0);
        if (slot < n) {
          bi = clist[first + slot];
          live = B[bi].mass.x > 0.0;
        }
        if (live) {
          let me = pose(bi);
          // contacts of the pairs this body owns: one contiguous range, in pair order
          let c0 = min(coffset[poffset[bi]], u.maxContacts);
          let c1 = min(coffset[poffset[bi + 1u]], u.maxContacts);
          for (var k = c0 + lane; k < c1; k += ${LANES}u) { addContact(&acc, k, bi, me); }
          // pairs it joins as the second body, in owner order
          ${reverseList(NB, 'bi')}
          for (var r = lane; r < rn; r += ${LANES}u) {
            let p = rlist[rb + r];
            let e = min(coffset[p + 1u], u.maxContacts);
            for (var k = min(coffset[p], u.maxContacts); k < e; k++) { addContact(&acc, k, bi, me); }
          }
          // joints
          let j1 = jcsr[bi + 1u];
          for (var q = jcsr[bi] + lane; q < j1; q += ${LANES}u) { addJoint(&acc, jcsr[${NB + 1}u + q], bi, me); }
        }
        red[lidx] = array<vec4f, 9>(vec4f(acc.Hll[0], acc.gl.x), vec4f(acc.Hll[1], acc.gl.y), vec4f(acc.Hll[2], acc.gl.z),
                                    vec4f(acc.Hla[0], acc.ga.x), vec4f(acc.Hla[1], acc.ga.y), vec4f(acc.Hla[2], acc.ga.z),
                                    vec4f(acc.Haa[0], 0.0), vec4f(acc.Haa[1], 0.0), vec4f(acc.Haa[2], 0.0));
        workgroupBarrier();
        for (var s = ${LANES / 2}u; s > 0u; s >>= 1u) {
          if (lane < s) { for (var e = 0u; e < 9u; e++) { red[lidx][e] += red[lidx + s][e]; } }
          workgroupBarrier();
        }
        if (live && lane == 0u) {
          let t = red[lidx];
          var b = B[bi];
          let m = b.mass.x;
          let ih2 = 1.0 / (u.dt * u.dt);
          let R = qmat(b.q);
          let Iw = R * diag3(b.mass.yzw) * transpose(R);
          // inertial terms: ½|x − y|²_M / Δt² (and the rotational analogue)
          let gL = vec3f(t[0].w, t[1].w, t[2].w) + m * ih2 * (b.x.xyz - b.y.xyz);
          let HLL = mat3x3f(t[0].xyz, t[1].xyz, t[2].xyz) + diag3(vec3f(m * ih2));
          if ((b.info.z & F_FIXED_ROT) != 0u) {
            b.x = vec4f(b.x.xyz - cholSolve(chol3(HLL), gL), b.x.w);
          } else {
            let gA = vec3f(t[3].w, t[4].w, t[5].w) + ih2 * (Iw * rotvec(b.q, b.qy));
            let HLA = mat3x3f(t[3].xyz, t[4].xyz, t[5].xyz);
            let HAA = mat3x3f(t[6].xyz, t[7].xyz, t[8].xyz) + Iw * ih2;
            let d = solveBlock6(HLL, HLA, HAA, -gL, -gA);
            b.x = vec4f(b.x.xyz + d[0], b.x.w);
            b.q = qinteg(b.q, d[1], 1.0);
          }
          B[bi] = b;
        }
        workgroupBarrier();   // red[] is rewritten by the next pass
        storageBarrier();
        }
`;
    const DUAL_WGSL = /* wgsl */`
        // The augmented-Lagrangian update of contact i after a sweep (last: the final iteration).
        fn dualContact(i: u32, last: u32) {
        var c = C[i];
        if (c.ab.w != 0u) { return; }
        let hasB = c.ab.y != STATIC;
        let L = contactLin(c, pose(c.ab.x), pose(select(c.ab.x, c.ab.y, hasB)), hasB);
        let nrm = c.n.xyz;
        let Cn = contactCn(L, nrm, u.alpha, u.slop);
        let meff = c.k.w;
        let kmax = meff * u.kmaxScale;
        c.lam.x = min(0.0, c.k.x * Cn + c.lam.x);
        if (c.lam.x < 0.0) { c.k.x = min(kmax, c.k.x + u.beta * meff * abs(Cn)); }
        if (u.tolerance > 0.0) { c.k.x = max(c.k.x, -c.lam.x / u.tolerance); }
        // friction is projected onto the cone of the updated normal force, so every stored
        // (λ_n, λ_t) satisfies |λ_t| ≤ μ|λ_n| exactly (law "contacts" in LAWS.md)
        let bound = c.n.w * abs(c.lam.x);
        let T = basis(nrm);
        let ct = vec2f((1.0 - u.alpha) * dot(T[0], L.gap) + dot(T[0], L.rel), (1.0 - u.alpha) * dot(T[1], L.gap) + dot(T[1], L.rel));
        var lt = vec2f(c.k.y, c.k.z) * ct + c.lam.yz;
        let ll = length(lt);
        let stick = ll < bound;   // inside the friction cone: static friction, stiffen
        if (stick) {
          c.k.y = min(kmax, c.k.y + u.beta * meff * abs(ct.x));
          c.k.z = min(kmax, c.k.z + u.beta * meff * abs(ct.y));
        } else { lt *= bound / max(ll, 1e-30); }   // sliding: kinetic friction on the cone
        c.lam.y = lt.x;
        c.lam.z = lt.y;
        // static friction: the anchors are kept next step while the contact sticks and has not drifted far
        if (last != 0u) {
          let drift = L.gap - nrm * dot(nrm, L.gap);
          c.ab.z = select(0u, 1u, stick && length(drift) < u.stickThreshold);
        }
        C[i] = c;
        }
`;
    const JDUAL_WGSL = /* wgsl */`
        // The joint rows' multipliers and stiffness after a sweep; breaks joints past their strength.
        fn jointDualOne(i: u32) {
        var j = J[i];
        if ((j.ab.w & (JF_BROKEN | JF_DISABLED | JF_WORLD_ANCHORS)) != 0u || j.prm.x > 0.0) { return; }
        let hasB = j.ab.y != STATIC;
        let PA = pose(j.ab.x);
        var PB = PA;
        if (hasB) { PB = pose(j.ab.y); }
        let ma = B[j.ab.x].mass;
        var m = ma.x; var I = (ma.y + ma.z + ma.w) / 3.0;
        if (hasB) { let mb = B[j.ab.y].mass; if (mb.x > 0.0 && (m == 0.0 || mb.x < m)) { m = mb.x; I = (mb.y + mb.z + mb.w) / 3.0; } }
        let nrow = jointRows(j.ab.z);
        var lam0 = j.lam0; var lam1 = j.lam1; var k0 = j.k0; var k1 = j.k1;
        for (var row = 0u; row < nrow; row++) {
          let jr = jointRow(j, PA, PB, hasB, row, u.dt);
          var lam = rowGet(lam0, lam1, row);
          var k = rowGet(k0, k1, row);
          let c0 = rowGet(j.c00, j.c01, row);
          let scale = select(max(I, u.massFloor * 0.01), max(m, u.massFloor), dot(jr.la, jr.la) + dot(jr.lb, jr.lb) > 0.0) * u.ih2;
          let kmax = scale * u.kmaxScale;
          if (jr.kind == JK_OFF) { lam = 0.0; }
          else if (jr.kind == JK_MOTOR) {
            lam = clamp(k * jr.C + lam, -j.prm.w, j.prm.w);
            if (abs(lam) < j.prm.w) { k = min(kmax, k + u.beta * scale * abs(jr.C)); }
          } else {
            let Cs = jr.C - u.alpha * select(c0, min(c0, 0.0), jr.kind == JK_GE);
            lam = k * Cs + lam;
            if (jr.kind == JK_GE) { lam = min(lam, 0.0); }
            if (jr.kind == JK_EQ || lam < 0.0) { k = min(kmax, k + u.beta * scale * abs(Cs)); }
          }
          if (row < 4u) { lam0[row] = lam; k0[row] = k; } else { lam1[row - 4u] = lam; k1[row - 4u] = k; }
        }
        j.lam0 = lam0; j.lam1 = lam1; j.k0 = k0; j.k1 = k1;
        // breakable joints: the constraint force (λ is a force) exceeded its strength
        if (j.ra.w > 0.0) {
          var fmag = length(lam0.xyz);
          if (j.ab.z == J_DISTANCE) { fmag = abs(lam0.x); }
          if (j.ab.z == J_SLIDER) { fmag = length(lam0.xy); }
          if (fmag > j.ra.w) { j.ab.w = j.ab.w | JF_BROKEN; }
        }
        J[i] = j;
        }
`;
    this.primalK = ctx.kernel({
      name: 'rigid-primal', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body' }, C: { type: 'Contact', access: 'read' }, J: { type: 'Joint', access: 'read' },
        poffset: { type: 'u32', access: 'read' }, coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' },
        jcsr: { type: 'u32', access: 'read' }, clist: { type: 'u32', access: 'read' }, info: { type: 'u32', access: 'read' } },
      uniforms: { dt: 'f32', alpha: 'f32', colorIdx: 'u32', geoStiff: 'f32', slop: 'f32', maxContacts: 'u32' },
      shared: POSE_WGSL + PRIMAL_WGSL,
      code: /* wgsl */`
        // Grid-stride over the colour's bodies, BODIES_PER_WG at a time: any dispatch size covers
        // the colour, so the CPU sizes dispatches from (stale) counts without indirect dispatches,
        // which cost ~300 µs each on some Vulkan drivers. The count comes through a uniform load
        // so the loop (and its barriers) stays in uniform control flow.
        if (lidx == 0u) { colorCount = info[u.colorIdx * 4u]; }
        let n = workgroupUniformLoad(&colorCount);
        let first = info[u.colorIdx * 4u + 1u];
        for (var base = wgi * ${BODIES_PER_WG}u; base < n; base += nwg.x * nwg.y * ${BODIES_PER_WG}u) { primalPass(base, n, first, lidx); }`,
    });
    this.dualK = ctx.kernel({
      name: 'rigid-dual', imports, workgroupSize: 64, guard: false,
      buffers: { B: { type: 'Body', access: 'read' }, C: { type: 'Contact' }, counter: { type: 'u32', access: 'read' } },
      uniforms: { alpha: 'f32', beta: 'f32', kmaxScale: 'f32', slop: 'f32', last: 'u32', stickThreshold: 'f32', tolerance: 'f32' },
      shared: POSE_WGSL + DUAL_WGSL,
      code: /* wgsl */`
        for (var gi = i; gi < counter[1]; gi += nwg.x * nwg.y * WG) { dualContact(gi, u.last); }   // grid-stride over the contacts`,
    });
    this.jointDualK = ctx.kernel({
      name: 'rigid-joint-dual', imports,
      buffers: { B: { type: 'Body', access: 'read' }, J: { type: 'Joint' } },
      uniforms: { dt: 'f32', alpha: 'f32', beta: 'f32', kmaxScale: 'f32', ih2: 'f32', massFloor: 'f32' },
      shared: POSE_WGSL + JDUAL_WGSL,
      code: 'jointDualOne(i);',
    });
    // The whole solve of a small world in one dispatch: one workgroup runs every iteration, colour
    // by colour, then the contact and joint duals, with barriers where the multi-dispatch path has
    // dispatch boundaries. Same functions, same order — the same results up to rounding: drivers
    // compile the shared functions a little differently inside the two shaders (1–4 ulp after a
    // step). Each path is deterministic, and which one runs depends only on the body count. Below a
    // few hundred bodies it replaces ~150 tiny dispatches, each with 4–15 µs of fixed overhead.
    this.solveFusedK = ctx.kernel({
      name: 'rigid-solve-fused', imports, workgroupSize: 128, guard: false,
      buffers: { B: { type: 'Body' }, C: { type: 'Contact' }, J: { type: 'Joint' },
        poffset: { type: 'u32', access: 'read' }, coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' },
        jcsr: { type: 'u32', access: 'read' }, clist: { type: 'u32', access: 'read' }, info: { type: 'u32', access: 'read' } },
      uniforms: { dt: 'f32', alpha: 'f32', beta: 'f32', kmaxScale: 'f32', slop: 'f32', stickThreshold: 'f32', tolerance: 'f32',
        geoStiff: 'f32', maxContacts: 'u32', ih2: 'f32', massFloor: 'f32', iterations: 'u32', nColors: 'u32', nJoints: 'u32', maxPairs: 'u32' },
      shared: POSE_WGSL + PRIMAL_WGSL + DUAL_WGSL + JDUAL_WGSL,
      code: /* wgsl */`
        let nc = min(coffset[u.maxPairs], u.maxContacts);
        for (var it = 0u; it < u.iterations; it++) {
          for (var c = 0u; c < u.nColors; c++) {
            workgroupBarrier();   // everyone has read the previous colour's count (an empty colour has no other barrier)
            if (lidx == 0u) { colorCount = info[c * 4u]; }
            let n = workgroupUniformLoad(&colorCount);
            let first = info[c * 4u + 1u];
            for (var base = 0u; base < n; base += WG / ${LANES}u) { primalPass(base, n, first, lidx); }
          }
          let last = select(0u, 1u, it + 1u == u.iterations);
          for (var ci = lidx; ci < nc; ci += WG) { dualContact(ci, last); }
          for (var ji = lidx; ji < u.nJoints; ji += WG) { jointDualOne(ji); }
          storageBarrier();
        }`,
    });
    this.velocityK = ctx.kernel({
      name: 'rigid-velocity', imports,
      buffers: { B: { type: 'Body' } },
      uniforms: { dt: 'f32', g: 'vec3f' },
      code: /* wgsl */`
        var b = B[i];
        if (b.mass.x == 0.0 || (b.info.z & F_DISABLED) != 0u) { return; }
        let g = u.g * b.props.x;
        let gl = length(g);
        let vg = select(0.0, dot(b.v.xyz, g / gl), gl > 0.0);   // for the adaptive initial guess
        b.v = vec4f((b.x.xyz - b.x0.xyz) / u.dt, b.v.w);
        b.w = vec4f(rotvec(b.q, b.q0) / u.dt, vg);
        B[i] = b;`,
    });
    // order-independent 64-bit state hash (sum of per-body mixes): equal states, equal hashes
    this.hashK = ctx.kernel({
      name: 'rigid-hash', imports,
      buffers: { B: { type: 'Body', access: 'read' }, out: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let b = B[i];
        var h = pcg(i ^ 0x9e3779b9u);
        let w0 = bitcast<vec4u>(b.x); let w1 = bitcast<vec4u>(b.q); let w2 = bitcast<vec4u>(b.v); let w3 = bitcast<vec4u>(b.w);
        h = pcg(h ^ w0.x); h = pcg(h ^ w0.y); h = pcg(h ^ w0.z);
        h = pcg(h ^ w1.x); h = pcg(h ^ w1.y); h = pcg(h ^ w1.z); h = pcg(h ^ w1.w);
        h = pcg(h ^ w2.x); h = pcg(h ^ w2.y); h = pcg(h ^ w2.z);
        h = pcg(h ^ w3.x); h = pcg(h ^ w3.y); h = pcg(h ^ w3.z);
        h = pcg(h ^ b.info.z);
        atomicAdd(&out[0], h);
        atomicAdd(&out[1], pcg(h + 0x632be59bu));`,
    });
    // contact events: pairs present now but not last step (begin) and the reverse (end)
    this.eventsK = ctx.kernel({
      name: 'rigid-events', imports: ['hash'],
      buffers: { cur: { type: 'u32', access: 'read' }, prev: { type: 'u32', access: 'read' }, events: { type: 'vec4u' }, counter: { type: 'u32', atomic: true } },
      uniforms: { tableMask: 'u32', kind: 'u32', maxEvents: 'u32' },
      code: /* wgsl */`
        let key = cur[i];
        if (key == 0u) { return; }
        var s = pcg(key) & u.tableMask;
        for (var probe = 0u; probe < 64u; probe++) {
          let k = prev[s];
          if (k == key) { return; }
          if (k == 0u) { break; }
          s = (s + 1u) & u.tableMask;
        }
        let slot = atomicAdd(&counter[4], 1u);
        if (slot < u.maxEvents) { events[slot] = vec4u(key, u.kind, 0u, 0u); }`,
    });
  }

  // Kernels that read the broadphase grid (rebuilt when the cell size changes).
  _gridKernels() {
    const ctx = this.ctx;
    const NB = this.maxBodies;
    const G = this.grid.uniforms();
    const shared = /* wgsl */`
      struct Lite { x: vec4f, m: f32, info: vec4u }
      fn lite(i: u32) -> Lite { return Lite(B[i].x0, B[i].mass.x, B[i].info); }
      fn jointed(a: u32, b: u32) -> bool {
        if (u.jsetMask == 0u) { return false; }
        let key = min(a, b) * (u.nBodies + 1u) + max(a, b) + 1u;
        var x = key;
        x = (x ^ (x >> 16u)) * 0x7feb352du; x = (x ^ (x >> 15u)) * 0x846ca68bu; x = x ^ (x >> 16u);
        var s = x & u.jsetMask;
        for (var probe = 0u; probe < 64u; probe++) {
          let k = jset[s];
          if (k == 0u) { return false; }
          if (k == key) { return (J[jset[u.jsetMask + 1u + s]].ab.w & (JF_BROKEN | JF_DISABLED)) == 0u; }
          s = (s + 1u) & u.jsetMask;
        }
        return false;
      }
      fn canCollide(a: u32, la: Lite, o: u32, lo: Lite) -> bool {
        if (((la.info.z | lo.info.z) & F_DISABLED) != 0u) { return false; }
        if (la.m <= 0.0 && lo.m <= 0.0) { return false; }
        if ((la.info.x & lo.info.y) == 0u || (lo.info.x & la.info.y) == 0u) { return false; }
        if (length(lo.x.xyz - la.x.xyz) >= la.x.w + lo.x.w + u.margin) { return false; }
        return !jointed(a, o);
      }
      // bounded sorted insert: keeps the cap smallest values whatever the visiting order
      fn insertSorted(arr: ptr<function, array<u32, ${CAND}>>, n: ptr<function, u32>, v: u32, cap: u32) {
        var k = *n;
        if (k == cap) { if (v >= (*arr)[cap - 1u]) { return; } k = cap - 1u; } else { *n = k + 1u; }
        while (k > 0u && (*arr)[k - 1u] > v) { (*arr)[k] = (*arr)[k - 1u]; k--; }
        (*arr)[k] = v;
      }
      // the pairs body i owns: small bodies own pairs with larger-index small bodies and with every
      // large body; large bodies own pairs with larger-index large bodies; dynamic bodies own the floor.
      // The same walk collects a dynamic small body's lower-index small owners (its reverse list).
      fn ownedPairs(i: u32, arr: ptr<function, array<u32, ${CAND}>>, rev: ptr<function, array<u32, ${CAND}>>, nr: ptr<function, u32>) -> u32 {
        var n = 0u;
        let li = lite(i);
        if ((li.info.z & F_DISABLED) != 0u) { return 0u; }
        let pi = li.x.xyz;
        let isLarge = li.x.w > u.rSmall;
        let dyn = li.m > 0.0;
        if (u.hasFloor != 0u && dyn && pi.y - li.x.w < u.floorY + u.margin) { insertSorted(arr, &n, STATIC, ${MAXP}u); }
        if (!isLarge) {
          ${SpatialGrid.neighbors('pi', /* wgsl */`
              let o = order[j];
              if (o != i && (o > i || dyn)) {
                let lo = lite(o);
                if (lo.x.w <= u.rSmall && canCollide(i, li, o, lo)) {
                  if (o > i) { insertSorted(arr, &n, o, ${MAXP}u); } else { insertSorted(rev, nr, o, ${MAXR}u); }
                }
              }`)}
        }
        for (var k = 0u; k < u.nLarge; k++) {
          let o = large[k];
          if (o == i || (isLarge && o < i)) { continue; }
          if (canCollide(i, li, o, lite(o))) { insertSorted(arr, &n, o, ${MAXP}u); }
        }
        return n;
      }`;
    const base = {
      imports: ['rigid', 'grid'],
      uniforms: { ...G, floorY: 'f32', hasFloor: 'u32', margin: 'f32', rSmall: 'f32', nLarge: 'u32', nBodies: 'u32', jsetMask: 'u32', count: 'u32', maxPairs: 'u32' },
      shared,
    };
    const bufs = { B: { type: 'Body', access: 'read' }, order: { type: 'u32', access: 'read' }, start: { type: 'u32', access: 'read' },
      large: { type: 'u32', access: 'read' }, jset: { type: 'u32', access: 'read' }, J: { type: 'Joint', access: 'read' } };
    this.pairCountK = ctx.kernel({
      ...base, name: 'rigid-pair-find',
      buffers: { ...bufs, pcount: { type: 'u32' }, pscratch: { type: 'u32' }, rlist: { type: 'u32' } },
      code: /* wgsl */`
        var arr: array<u32, ${CAND}>;
        var rev: array<u32, ${CAND}>;
        var nr = 0u;
        let n = ownedPairs(i, &arr, &rev, &nr);
        pcount[i] = n;
        for (var k = 0u; k < n; k++) { pscratch[i * ${MAXP}u + k] = arr[k]; }
        for (var k = 0u; k < nr; k++) { rlist[i * ${MAXR}u + k] = rev[k]; }
        rlist[${NB * MAXR}u + i] = nr;
        if (i == u.count - 1u) { pcount[u.count] = 0u; }`,
    });
    this.pairWriteK = ctx.kernel({
      name: 'rigid-pair-write',
      buffers: { poffset: { type: 'u32', access: 'read' }, pscratch: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u' } },
      uniforms: { maxPairs: 'u32' },
      code: /* wgsl */`
        let o = poffset[i];
        let n = poffset[i + 1u] - o;
        for (var k = 0u; k < n; k++) { if (o + k < u.maxPairs) { pairs[o + k] = vec2u(i, pscratch[i * ${MAXP}u + k]); } }`,
    });
    // For every dynamic body, the pairs it joins as the second body, sorted by owner. The
    // owner's list is sorted, so the pair index comes from a binary search.
    const findPair = /* wgsl */`
      fn findPair(o: u32, i: u32) -> u32 {
        let e = min(poffset[o + 1u], u.maxPairs);
        var lo = poffset[o];
        var hi = e;
        while (lo < hi) { let mid = (lo + hi) / 2u; if (pairs[mid].y < i) { lo = mid + 1u; } else { hi = mid; } }
        return select(0xffffffffu, lo, lo < e && pairs[lo].y == i);
      }`;
    const revBufs = { ...bufs, poffset: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, rlist: { type: 'u32' } };
    // small bodies: rigid-pair-find left the owners in the list; turn them into pair indices in place
    this.reverseK = ctx.kernel({
      name: 'rigid-reverse-pairs',
      buffers: { B: { type: 'Body', access: 'read' }, poffset: { type: 'u32', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, rlist: { type: 'u32' } },
      uniforms: { rSmall: 'f32', maxPairs: 'u32' }, imports: ['rigid'], shared: findPair,
      code: /* wgsl */`
        if (B[i].x0.w > u.rSmall) { return; }   // large bodies: rigid-reverse-large
        let n = min(rlist[${NB * MAXR}u + i], ${MAXR}u);
        var m = 0u;
        for (var k = 0u; k < n; k++) {
          let p = findPair(rlist[i * ${MAXR}u + k], i);
          if (p != 0xffffffffu) { rlist[i * ${MAXR}u + m] = p; m++; }
        }
        rlist[${NB * MAXR}u + i] = m;`,
    });
    // A large dynamic body's owners are every small body touching it and the lower-index large
    // ones. It runs after the narrowphase and keeps only pairs with contacts: a large body's
    // bounding sphere overlaps far more bodies than actually touch it. One workgroup per large
    // body tests all bodies 256 at a time; a workgroup prefix sum places the hits, so the list
    // comes out sorted by owner without depending on thread order.
    this.reverseLargeK = ctx.kernel({
      ...base, name: 'rigid-reverse-large', workgroupSize: 256, guard: false,
      buffers: { ...revBufs, coffset: { type: 'u32', access: 'read' } },
      shared: shared + findPair + 'var<workgroup> rscan: array<u32, 256>; var<workgroup> rlive: u32; var<workgroup> rtotal: u32;',
      code: /* wgsl */`
        let b = large[wid.x];
        let base = ${NB * (MAXR + 1)}u + wid.x * ${MAXRL}u;
        if (lidx == 0u) {
          let l0 = lite(b);
          rlive = select(0u, 1u, l0.m > 0.0 && (l0.info.z & F_DISABLED) == 0u);
          rtotal = 0u;
        }
        if (workgroupUniformLoad(&rlive) == 0u) {
          if (lidx == 0u) { rlist[${NB * MAXR}u + b] = 0u; }
          return;
        }
        let li = lite(b);
        for (var c = 0u; c < u.count; c += 256u) {
          let o = c + lidx;
          var p = 0xffffffffu;
          if (o < u.count && o != b) {
            let lo = lite(o);
            if ((lo.x.w <= u.rSmall || o < b) && canCollide(b, li, o, lo)) {
              p = findPair(o, b);
              if (p != 0xffffffffu && coffset[p + 1u] == coffset[p]) { p = 0xffffffffu; }
            }
          }
          let hit = select(0u, 1u, p != 0xffffffffu);
          rscan[lidx] = hit;
          workgroupBarrier();
          for (var s = 1u; s < 256u; s <<= 1u) {
            var add = 0u;
            if (lidx >= s) { add = rscan[lidx - s]; }
            workgroupBarrier();
            rscan[lidx] += add;
            workgroupBarrier();
          }
          let before = workgroupUniformLoad(&rtotal);
          let at = before + rscan[lidx] - hit;
          if (hit == 1u && at < ${MAXRL}u) { rlist[base + at] = p; }
          workgroupBarrier();
          if (lidx == 255u) { rtotal = before + rscan[255]; }
          if (workgroupUniformLoad(&rtotal) >= ${MAXRL}u) { break; }
        }
        if (lidx == 0u) { rlist[${NB * MAXR}u + b] = min(rtotal, ${MAXRL}u) | ((wid.x + 1u) << 16u); }`,
    });
    // Jones–Plassmann + first fit over contact and joint edges. Each round reads the previous
    // round's colours only (double-buffered), so the colouring is the same whatever the thread order.
    this.colorK = ctx.kernel({
      name: 'rigid-color', imports: ['rigid'],
      buffers: { B: { type: 'Body', access: 'read' }, pairs: { type: 'vec2u', access: 'read' }, poffset: { type: 'u32', access: 'read' },
        coffset: { type: 'u32', access: 'read' }, rlist: { type: 'u32', access: 'read' }, jcsr: { type: 'u32', access: 'read' },
        J: { type: 'Joint', access: 'read' }, cin: { type: 'u32', access: 'read' }, cout: { type: 'u32' } },
      uniforms: { last: 'u32', maxColor: 'u32', maxPairs: 'u32' },
      shared: /* wgsl */`
        struct Visit { used: u32, wait: bool }
        fn visit(v: ptr<function, Visit>, i: u32, pi: u32, o: u32, self_flags: u32) {
          if (o == STATIC) { return; }
          let bo = B[o];
          if (bo.mass.x <= 0.0 || ((bo.info.z | self_flags) & (F_SENSOR | F_DISABLED)) != 0u) { return; }
          let co = cin[o];
          if (co == 0xffffffffu) {
            let po = pcg(o * 9781u + 7u);
            if ((po > pi || (po == pi && o > i)) && u.last == 0u) { (*v).wait = true; }
          } else { (*v).used |= 1u << min(co, 31u); }
        }`,
      code: /* wgsl */`
        if (cin[i] != 0xffffffffu) { cout[i] = cin[i]; return; }
        let bi = B[i];
        if (bi.mass.x <= 0.0 || (bi.info.z & F_DISABLED) != 0u) { cout[i] = 0u; return; }
        let pi = pcg(i * 9781u + 7u);
        var v: Visit;
        v.used = 0u; v.wait = false;
        let p1 = min(poffset[i + 1u], u.maxPairs);
        for (var p = min(poffset[i], u.maxPairs); p < p1; p++) { if (coffset[p + 1u] > coffset[p]) { visit(&v, i, pi, pairs[p].y, bi.info.z); } }
        ${reverseList(NB, 'i')}
        for (var r = 0u; r < rn; r++) { let p = rlist[rb + r]; if (coffset[p + 1u] > coffset[p]) { visit(&v, i, pi, pairs[p].x, bi.info.z); } }
        let j1 = jcsr[i + 1u];
        for (var q = jcsr[i]; q < j1; q++) {
          let jt = J[jcsr[${NB + 1}u + q]];
          if ((jt.ab.w & (JF_BROKEN | JF_DISABLED)) != 0u) { continue; }
          visit(&v, i, pi, select(jt.ab.x, jt.ab.y, jt.ab.x == i), 0u);
        }
        if (v.wait) { cout[i] = 0xffffffffu; return; }
        cout[i] = min(countTrailingZeros(~v.used), u.maxColor);`,
    });
  }

  // ------------------------------------------------------------------ stepping
  step(frameDt = 1 / 60) {
    this.commit();
    if (!this.count) { this.cmds = []; return; }
    const ctx = this.ctx, n = this.count;
    const fdt = Math.min(frameDt, 1 / 30);
    const dt = fdt / this.substeps;
    const nColors = this._colorBudget();
    const post = this.postStabilize, alpha = post ? 1 : this.alpha;
    const gammaLam = post ? 1 : this.alpha * this.gamma;
    const nj = this.joints.length;
    const ih2 = 1 / (dt * dt);
    this._applyCommands(fdt);
    for (const f of this.forceFields) if (f.enabled) f.kernel.run(n, { dt: fdt, time: this.frame * dt }, { B: this.B, ext: this.ext });
    const gridVals = {
      ...this.grid.values(), floorY: this.floor ?? 0, hasFloor: this.floor === null ? 0 : 1, margin: this.margin,
      rSmall: this.rSmall, nLarge: this.nLarge, nBodies: this.maxBodies, jsetMask: nj ? this.jsetSize - 1 : 0, count: n,
      maxPairs: this.maxPairs,
    };
    const gridBufs = { B: this.B, order: this.grid.order, start: this.grid.start, large: this.large, jset: this.jset, J: this.J };
    for (let s = 0; s < this.substeps; s++) {
      const cur = this.frame & 1, prev = cur ^ 1;
      ctx.region('rigid: broadphase', () => {
        ctx.clear(this.cnts); ctx.clear(this.tbl[cur]); ctx.clear(this.ccount);
        this.predictK.run(n, { dt, g: this.gravity }, { B: this.B, centers: this.centers, ext: this.ext });
        if (nj) this.jointPrepK.run(nj, { gammaLam, gammaK: this.gamma, kStart: this.kStartJoint, ih2, massFloor: this.jointMassFloor }, { B: this.B, J: this.J });
        this.grid.build(this.centers, n);
        this.pairCountK.run(n, gridVals, { ...gridBufs, pcount: this.pcount, pscratch: this.pscratch, rlist: this.rlist });
        this.pairScan.run(this.pcount, this.poffset, n + 1);
        this.pairWriteK.run(n, { maxPairs: this.maxPairs }, { poffset: this.poffset, pscratch: this.pscratch, pairs: this.pairs });
        this.pairArgsK.dispatch(1, { count: n, maxPairs: this.maxPairs }, { poffset: this.poffset, args: this.args, counter: this.cnts });
        this.reverseK.run(n, { rSmall: this.rSmall, maxPairs: this.maxPairs }, { B: this.B, poffset: this.poffset, pairs: this.pairs, rlist: this.rlist });
      });
      const W = this._sizes(n);
      ctx.region('rigid: narrowphase', () => {
        this.narrowCountK.dispatch(W.pairs, { floorY: this.floor ?? 0, margin: this.margin },
          { B: this.B, pairs: this.pairs, counter: this.cnts, ccount: this.ccount });
        this.contactScan.run(this.ccount, this.coffset, this.maxPairs + 1);
        this.narrowWriteK.dispatch(W.pairs, {
          floorY: this.floor ?? 0, margin: this.margin, maxContacts: this.maxContacts, tableMask: this.tableSize - 1,
          nBodies: this.maxBodies, gammaLam, gammaK: this.gamma, kStart: this.kStart, ih2, massFloor: this.contactMassFloor,
          stickThreshold: this.stickThreshold,
        }, { B: this.B, pairs: this.pairs, counter: this.cnts, coffset: this.coffset, C: this.C[cur], tbl: this.tbl[cur], Cp: this.C[prev], tblP: this.tbl[prev] });
        if (this.nLarge) this.reverseLargeK.dispatch([this.nLarge, 1, 1], gridVals,
          { ...gridBufs, poffset: this.poffset, pairs: this.pairs, rlist: this.rlist, coffset: this.coffset });
      });
      ctx.region('rigid: colouring', () => {
        this.colorInitK.run(n, null, { a: this.color[0], b: this.color[1] });
        const cbufs = { B: this.B, pairs: this.pairs, poffset: this.poffset, coffset: this.coffset, rlist: this.rlist, jcsr: this.jcsr, J: this.J };
        for (let r = 0; r < COLOR_ROUNDS; r++)
          this.colorK.run(n, { last: r === COLOR_ROUNDS - 1 ? 1 : 0, maxColor: nColors - 1, maxPairs: this.maxPairs },
            { ...cbufs, cin: this.color[r & 1], cout: this.color[(r + 1) & 1] });
        const fin = this.color[COLOR_ROUNDS & 1];
        ctx.clear(this.info);
        this.bucketK.run(n, null, { B: this.B, color: fin, info: this.info });
        this.offsetsK.dispatch(1, { maxContacts: this.maxContacts, maxPairs: this.maxPairs }, { info: this.info, args: this.args, counter: this.cnts, coffset: this.coffset });
        this.scatterK.run(n, null, { B: this.B, color: fin, info: this.info, clist: this.clist });
      });
      ctx.region('rigid: solve', () => {
        const pbufs = { B: this.B, C: this.C[cur], J: this.J, poffset: this.poffset, coffset: this.coffset, rlist: this.rlist,
          jcsr: this.jcsr, clist: this.clist, info: this.info };
        const sweep = (a) => {
          for (let c = 0; c < nColors; c++)
            this.primalK.dispatch(W.color(c), { dt, alpha: a, colorIdx: c, geoStiff: this.geoStiff, slop: this.slop, maxContacts: this.maxContacts }, pbufs);
        };
        if (this._useFused(n) && !post) {
          this.solveFusedK.dispatch(1, {
            dt, alpha, beta: this.beta, kmaxScale: 1e6, slop: this.slop, stickThreshold: this.stickThreshold, tolerance: this.contactTolerance,
            geoStiff: this.geoStiff, maxContacts: this.maxContacts, ih2, massFloor: this.jointMassFloor, iterations: this.iterations,
            nColors, nJoints: nj, maxPairs: this.maxPairs,
          }, { ...pbufs, C: this.C[cur] });
        } else for (let it = 0; it < this.iterations; it++) {
          sweep(alpha);
          this.dualK.dispatch(W.contacts, {
            alpha, beta: this.beta, kmaxScale: 1e6, slop: this.slop, last: it === this.iterations - 1 ? 1 : 0, stickThreshold: this.stickThreshold,
            tolerance: this.contactTolerance,
          }, { B: this.B, C: this.C[cur], counter: this.cnts });
          if (nj) this.jointDualK.run(nj, { dt, alpha, beta: this.beta, kmaxScale: 1e6, ih2, massFloor: this.jointMassFloor }, { B: this.B, J: this.J });
        }
        this.velocityK.run(n, { dt, g: this.gravity }, { B: this.B });
        if (post) sweep(0);   // remove the remaining overlap / joint drift without touching velocities
      });
      if (this.events) {
        const T = this.tableSize;
        this.eventsK.run(T, { tableMask: T - 1, kind: 0, maxEvents: this.maxEvents }, { cur: this.tbl[cur], prev: this.tbl[prev], events: this.eventBuf, counter: this.cnts });
        this.eventsK.run(T, { tableMask: T - 1, kind: 1, maxEvents: this.maxEvents }, { cur: this.tbl[prev], prev: this.tbl[cur], events: this.eventBuf, counter: this.cnts });
      }
      this.frame++;
    }
    // refresh the stats (colour budget, HUD) every 30 steps without stalling
    if (++this._steps % 30 === 1 && !this._statsBusy) {
      this._statsBusy = true;
      this.readStats().finally(() => { this._statsBusy = false; });
    }
  }

  _applyCommands(dt) {
    if (!this.cmds.length) return;
    const cmds = this.cmds;
    this.cmds = [];
    for (let o = 0; o < cmds.length; o += this.maxCommands) {
      const chunk = cmds.slice(o, o + this.maxCommands);
      const data = new Float32Array(chunk.length * 12);
      const u = new Uint32Array(data.buffer);
      chunk.forEach((c, k) => {
        u[k * 12] = c.type; u[k * 12 + 1] = c.id; u[k * 12 + 2] = c.hz >>> 0; u[k * 12 + 3] = c.hw >>> 0;
        data.set(c.a, k * 12 + 4); data.set(c.c, k * 12 + 8);
      });
      this.ctx.flush();   // the buffer write must not overtake earlier unsubmitted work
      this.cmdBuf.write(data);
      this.cmdK.dispatch(1, { count: chunk.length, dt }, { B: this.B, ext: this.ext, cmds: this.cmdBuf, J: this.J });
    }
  }

  // ------------------------------------------------------------------ readback & queries
  /** Asynchronously refresh this.stats = { contacts, colors, pairs } (for HUDs). */
  async readStats() {
    const [c, info] = await Promise.all([this.cnts.read(8), this.info.read(MAX_COLORS * 4)]);
    this.stats = { contacts: c[1], colors: c[2], pairs: c[3] };
    this._est = { pairs: c[3], contacts: c[1], colors: Array.from({ length: MAX_COLORS }, (_, k) => info[k * 4]) };
    return this.stats;
  }

  // The fused single-workgroup solve pays off while one workgroup sweeps the world faster than
  // ~(colours + 2) × iterations dispatches can be issued: measured crossovers ~130 bodies on an M2 Max
  // (Metal, ~4 µs per dispatch) and ~230 on a GB10 (Vulkan, ~15 µs), hence 128 by default. It needs
  // 9 storage buffers in one shader stage (the WebGPU default limit is 8).
  _useFused(n) {
    const on = this.fusedSolve;
    if (on === false) return false;
    if (this.ctx.limits.maxStorageBuffersPerShaderStage < 9) return false;
    return on === true || n <= on;
  }

  // Workgroup counts for the grid-stride kernels, from the latest (asynchronously read) counts.
  // They only set how much of the GPU a dispatch occupies: every kernel loops until it has covered
  // the real count on the GPU, so a stale or missing estimate costs time, never correctness.
  _sizes(n) {
    const e = this._est;
    const fit = (x, max) => Math.max(1, Math.min(max, Math.ceil(x)));
    return {
      pairs: fit((e ? e.pairs : 6 * n) * 1.25 / 64, Math.ceil(this.maxPairs / 64)),
      contacts: fit((e ? e.contacts : 4 * n) * 1.25 / 64, Math.ceil(this.maxContacts / 64)),
      color: (c) => fit((e ? e.colors[c] : n) * 1.25 / BODIES_PER_WG + 1, Math.ceil(n / BODIES_PER_WG)),
    };
  }

  /** Body poses: Float32Array of [x, y, z, qx, qy, qz, qw] per body. */
  async readPoses() {
    const raw = await this.B.read(this.count * BODY_VEC4);
    const out = new Float32Array(this.count * 7);
    for (let k = 0; k < this.count; k++) {
      out.set(raw.subarray(k * 48, k * 48 + 3), k * 7);
      out.set(raw.subarray(k * 48 + 4, k * 48 + 8), k * 7 + 3);
    }
    return out;
  }

  /** Full body state: { position, rotation, velocity, angularVelocity, flags } typed arrays. */
  async readBodies() {
    const raw = await this.B.read(this.count * BODY_VEC4);
    const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
    const n = this.count;
    const out = { position: new Float32Array(3 * n), rotation: new Float32Array(4 * n), velocity: new Float32Array(3 * n),
      angularVelocity: new Float32Array(3 * n), flags: new Uint32Array(n) };
    for (let k = 0; k < n; k++) {
      const o = k * 48;
      out.position.set(raw.subarray(o, o + 3), 3 * k);
      out.rotation.set(raw.subarray(o + 4, o + 8), 4 * k);
      out.velocity.set(raw.subarray(o + 8, o + 11), 3 * k);
      out.angularVelocity.set(raw.subarray(o + 12, o + 15), 3 * k);
      out.flags[k] = u[o + 46];
    }
    return out;
  }

  /** Joint state: { broken: bool[], force: Float32Array (|λ| of the positional rows, N) }. */
  async readJoints() {
    const n = this.joints.length;
    if (!n) return { broken: [], force: new Float32Array(0) };
    const raw = await this.J.read(n * JOINT_VEC4);
    const u = new Uint32Array(raw.buffer, raw.byteOffset, raw.length);
    const broken = [], force = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const o = k * JOINT_VEC4 * 4;
      broken.push((u[o + 3] & JOINT_FLAGS.broken) !== 0);
      force[k] = Math.hypot(raw[o + 28], raw[o + 29], raw[o + 30]);
    }
    return { broken, force };
  }

  /** Contact begin/end events of the last step (needs { events: true }): { begin: [[a, b], ...], end: [...] }, b = -1 for the floor. */
  async readEvents() {
    const c = await this.cnts.read(8);
    const n = Math.min(c[4], this.maxEvents);
    const out = { begin: [], end: [] };
    if (!n) return out;
    const raw = await this.eventBuf.read(n);
    const N = this.maxBodies + 1;
    for (let k = 0; k < n; k++) {
      const key = raw[4 * k] - 1, a = Math.floor(key / N), b = key % N;
      const pair = b === this.maxBodies ? [a, -1] : [Math.min(a, b), Math.max(a, b)];
      (raw[4 * k + 1] ? out.end : out.begin).push(pair);
    }
    const order = (x, y) => x[0] - y[0] || x[1] - y[1];
    out.begin.sort(order); out.end.sort(order);
    return out;
  }

  /** A reusable GPU ray query (see RayQuery): rays can be written by the CPU or by other kernels. */
  rayQuery(maxRays = 1024) { return new RayQuery(this, maxRays); }

  /** Cast rays [{ origin, dir, maxDistance, mask }] → [{ hit, body, distance, point, normal }]. */
  async raycast(rays) {
    if (!this._rq || this._rq.maxRays < rays.length) this._rq = new RayQuery(this, Math.max(64, rays.length));
    this._rq.set(rays).run(rays.length);
    return this._rq.read(rays.length);
  }

  /** 64-bit hash of every body's pose and velocity (hex string); equal states give equal hashes. */
  async hash() {
    this.hashRecord();
    const h = await this.hashBuf.read(2);
    return h[1].toString(16).padStart(8, '0') + h[0].toString(16).padStart(8, '0');
  }
  /**
   * Record the hash computation now (state at this point of the command stream) into a u32×2
   * buffer — of the live bodies, or of a snapshot's.
   */
  hashRecord(target = this.hashBuf, snap = null) {
    const n = snap ? snap.meta.count : this.count;
    this.ctx.clear(target);
    if (n) this.hashK.run(n, null, { B: snap ? snap.B : this.B, out: target });
    return target;
  }

  /**
   * The complete simulation state as typed arrays (save games, network resync) — of the live
   * world, or of a snapshot (e.g. a rollback session's last confirmed tick):
   * { frame, count, bodies, joints, ext, contacts, slots, meta }. importState() restores it exactly;
   * the warm-start table travels as its occupied slots only.
   */
  async exportState(snap = null) {
    this.commit();
    // Read from a private copy, recorded now: the live world (or a snapshot in a rollback ring)
    // may move on while the reads are in flight.
    const f = snap ? this._cloneSnapshot(snap) : this.snapshot();
    const m = f.meta;
    try {
      const [cnt, bodies, joints, ext, table] = await Promise.all([
        f.cnt.read(8),
        m.count ? f.B.read(m.count * BODY_VEC4) : new Float32Array(0),
        m.joints.length ? f.J.read(m.joints.length * JOINT_VEC4) : new Float32Array(0),
        m.count ? f.ext.read(m.count * 2) : new Float32Array(0),
        f.tbl.read(this.tableSize * 3),
      ]);
      const nc = Math.min(cnt[1], this.maxContacts);
      const contacts = nc ? await f.C.read(nc * CONTACT_VEC4) : new Float32Array(0);
      const T = this.tableSize, slots = [];
      for (let k = 0; k < T; k++) if (table[k]) slots.push(k, table[k], table[T + 2 * k], table[T + 2 * k + 1]);
      return {
        version: 2, frame: m.frame, count: m.count, contactCount: nc, tableSize: T,
        bodies, joints, ext, contacts, slots: Uint32Array.from(slots),
        // records are copied with slice(): a byte copy, so u32 words stored in float arrays keep their bits
        meta: { free: m.free.slice(), joints: m.joints.map((j) => ({ a: j.a, b: j.b, type: j.type, flags: j.flags, rec: j.rec.slice(), enabled: j.enabled })) },
      };
    } finally {
      for (const b of [f.B, f.J, f.ext, f.C, f.tbl, f.cnt]) b.destroy();
    }
  }

  _cloneSnapshot(s) {
    const ctx = this.ctx, m = s.meta;
    const c = {
      B: ctx.array('vec4f', Math.max(1, m.count) * BODY_VEC4, { label: 'export-bodies' }),
      J: ctx.array('vec4f', Math.max(1, m.joints.length) * JOINT_VEC4, { label: 'export-joints' }),
      ext: ctx.array('vec4f', Math.max(1, m.count) * 2, { label: 'export-ext' }),
      C: ctx.array('vec4f', this.maxContacts * CONTACT_VEC4, { label: 'export-contacts' }),
      tbl: ctx.array('u32', this.tableSize * 3, { label: 'export-table' }),
      cnt: ctx.array('u32', 8, { label: 'export-counts' }),
      meta: m,
    };
    if (m.count) { ctx.copy(s.B, c.B, m.count * BODY_VEC4 * 16); ctx.copy(s.ext, c.ext, m.count * 32); }
    if (m.joints.length) ctx.copy(s.J, c.J, m.joints.length * JOINT_VEC4 * 16);
    ctx.copy(s.C, c.C);
    ctx.copy(s.tbl, c.tbl);
    ctx.copy(s.cnt, c.cnt, 32);
    return c;
  }

  importState(st) {
    const ctx = this.ctx;
    let table = st.table;
    if (st.slots) {
      if (st.tableSize !== this.tableSize) throw new Error(`RigidWorld.importState: table size ${st.tableSize} ≠ ${this.tableSize} (create both worlds with the same maxBodies / maxContacts)`);
      const T = this.tableSize;
      table = new Uint32Array(T * 3);
      for (let k = 0; k < st.slots.length; k += 4) { const i = st.slots[k]; table[i] = st.slots[k + 1]; table[T + 2 * i] = st.slots[k + 2]; table[T + 2 * i + 1] = st.slots[k + 3]; }
    }
    this.cmds = [];
    this.frame = st.frame; this.count = st.count; this.uploaded = st.count; this.free = st.meta.free.slice();
    this.cpu = [];
    for (let k = 0; k < st.count; k++) this.cpu.push(st.bodies.slice(k * BODY_VEC4 * 4, (k + 1) * BODY_VEC4 * 4));
    this.dirty.clear();
    this.joints = st.meta.joints.map((j) => ({ ...j, rec: j.rec instanceof Float32Array ? j.rec.slice() : Float32Array.from(j.rec) }));
    this.jointsUploaded = this.joints.length;
    const last = (this.frame + 1) & 1;
    ctx.clear(this.C[last]); ctx.clear(this.tbl[last]);
    ctx.flush();   // the clears (and earlier work) must run before the queue writes below
    if (st.count) { this.B.write(st.bodies); this.ext.write(st.ext); }
    if (this.joints.length) this.J.write(st.joints);
    if (st.contactCount) this.C[last].write(st.contacts);
    this.tbl[last].write(table);
    this.cnts.write(new Uint32Array([0, st.contactCount || 0, 0, 0, 0, 0, 0, 0]));
    this.jointsDirty = true;
    this._pending = true;
    this.uploaded = this.count;   // bodies are on the GPU already; commit() only rebuilds lists and the grid
  }

  /**
   * Capture everything the next step depends on: bodies, joints, pending impulses, last step's
   * contacts (warm start) and the CPU bookkeeping. Pass a previous snapshot to reuse its buffers.
   */
  snapshot(into) {
    this.commit();
    const ctx = this.ctx;
    const s = into || {
      B: ctx.array('vec4f', this.maxBodies * BODY_VEC4, { label: 'snap-bodies' }),
      J: ctx.array('vec4f', this.maxJoints * JOINT_VEC4, { label: 'snap-joints' }),
      ext: ctx.array('vec4f', this.maxBodies * 2, { label: 'snap-ext' }),
      C: ctx.array('vec4f', this.maxContacts * CONTACT_VEC4, { label: 'snap-contacts' }),
      tbl: ctx.array('u32', this.tableSize * 3, { label: 'snap-table' }),
      cnt: ctx.array('u32', 8, { label: 'snap-counts' }),
    };
    if (!s.cnt) s.cnt = ctx.array('u32', 8, { label: 'snap-counts' });
    const last = (this.frame + 1) & 1;   // the buffers the next step reads as "previous"
    if (this.count) {
      ctx.copy(this.B, s.B, this.count * BODY_VEC4 * 16);
      ctx.copy(this.ext, s.ext, this.count * 32);
    }
    if (this.joints.length) ctx.copy(this.J, s.J, this.joints.length * JOINT_VEC4 * 16);
    ctx.copy(this.C[last], s.C);
    ctx.copy(this.tbl[last], s.tbl);
    ctx.copy(this.cnts, s.cnt, 32);   // contact count of the step that produced those contacts (for exports)
    s.meta = { frame: this.frame, count: this.count, cpu: this.cpu.slice(), free: this.free.slice(), joints: this.joints.slice(),
      steps: this._steps, stats: { ...this.stats }, cmds: this.cmds.slice() };
    return s;
  }

  /** Return to a snapshot (bodies and joints added since are dropped). */
  restore(s) {
    const ctx = this.ctx, m = s.meta;
    this.commit();
    this.cmds = m.cmds.slice();
    this.frame = m.frame; this.count = m.count; this.uploaded = m.count; this.cpu = m.cpu.slice(); this.free = m.free.slice();
    this.dirty.clear();
    if (this.joints.length !== m.joints.length) { this.joints = m.joints.slice(); this.jointsUploaded = this.joints.length; this.jointsDirty = true; this._pending = true; }
    this._steps = m.steps; this.stats = { ...m.stats };
    const last = (this.frame + 1) & 1;
    if (this.count) {
      ctx.copy(s.B, this.B, this.count * BODY_VEC4 * 16);
      ctx.copy(s.ext, this.ext, this.count * 32);
    }
    if (this.joints.length) ctx.copy(s.J, this.J, this.joints.length * JOINT_VEC4 * 16);
    ctx.copy(s.C, this.C[last]);
    ctx.copy(s.tbl, this.tbl[last]);
    if (s.cnt) ctx.copy(s.cnt, this.cnts, 32);   // so an export right after this describes the restored contacts
    this._pending = true;   // re-derive the large-body list for the restored set of bodies
  }
}

/**
 * Batched GPU ray casts against a RigidWorld. Every (body, ray) pair is one thread; the nearest
 * hit per ray is found with two integer atomicMin passes (distance bits, then body id), so the
 * result does not depend on thread order.
 *   rays: vec4f × 2 per ray — (origin, maxDistance), (direction, mask bits)
 *   hits: vec4f × 2 per ray — (distance, or -1 for a miss; body index as u32 bits, -, -), (normal, -)
 */
export class RayQuery {
  constructor(world, maxRays) {
    const ctx = world.ctx;
    this.world = world;
    this.maxRays = maxRays;
    this.rays = ctx.array('vec4f', maxRays * 2, { label: 'ray-query-rays' });
    this.hits = ctx.array('vec4f', maxRays * 2, { label: 'ray-query-hits' });
    this.best = ctx.array('u32', maxRays * 2, { label: 'ray-query-best' });
    const common = { imports: ['rigid'], shared: RAY_WGSL, uniforms: { phase: 'u32' } };
    this.testK = ctx.kernel({
      ...common, name: 'ray-test', dims: 2, workgroupSize: [64, 1, 1],
      buffers: { B: { type: 'Body', access: 'read' }, rays: { type: 'vec4f', access: 'read' }, best: { type: 'u32', atomic: true } },
      code: /* wgsl */`
        let bi = p.x; let ri = p.y;
        let b = B[bi];
        if ((b.info.z & (F_DISABLED | F_SENSOR)) != 0u) { return; }
        let o = rays[2u * ri]; let dr = rays[2u * ri + 1u];
        if ((b.info.x & bitcast<u32>(dr.w)) == 0u) { return; }
        // bounding sphere first
        let oc = o.xyz - b.x.xyz;
        let bb = dot(oc, dr.xyz);
        if (bb * bb - dot(oc, oc) + b.x.w * b.x.w < 0.0) { return; }
        let h = rayBody(o.xyz, dr.xyz, b);
        if (h.x < 0.0 || h.x > o.w) { return; }
        if (u.phase == 0u) { atomicMin(&best[2u * ri], bitcast<u32>(h.x)); }
        else if (atomicLoad(&best[2u * ri]) == bitcast<u32>(h.x)) { atomicMin(&best[2u * ri + 1u], bi); }`,
    });
    this.resolveK = ctx.kernel({
      ...common, name: 'ray-resolve',
      buffers: { B: { type: 'Body', access: 'read' }, rays: { type: 'vec4f', access: 'read' }, best: { type: 'u32', access: 'read' }, hits: { type: 'vec4f' } },
      code: /* wgsl */`
        let bi = best[2u * i + 1u];
        if (bi == 0xffffffffu) { hits[2u * i] = vec4f(-1.0, 0.0, 0.0, 0.0); hits[2u * i + 1u] = vec4f(0.0); return; }
        let o = rays[2u * i]; let dr = rays[2u * i + 1u];
        let h = rayBody(o.xyz, dr.xyz, B[bi]);
        hits[2u * i] = vec4f(h.x, bitcast<f32>(bi), 0.0, 0.0);
        hits[2u * i + 1u] = vec4f(h.yzw, 0.0);`,
    });
  }
  /** Upload rays [{ origin, dir, maxDistance, mask }]. */
  set(list) {
    const data = new Float32Array(list.length * 8);
    const u = new Uint32Array(data.buffer);
    list.forEach((r, k) => {
      const d = r.dir, l = Math.hypot(...d) || 1;
      data.set([...r.origin, r.maxDistance ?? 1e6], 8 * k);
      data.set([d[0] / l, d[1] / l, d[2] / l], 8 * k + 4);
      u[8 * k + 7] = (r.mask ?? 0xffffffff) >>> 0;
    });
    this.world.ctx.flush();
    this.rays.write(data);
    return this;
  }
  /** Record the query for `count` rays (rays may have been written by GPU kernels). */
  run(count = this.maxRays) {
    const w = this.world, ctx = w.ctx;
    if (!w.count || !count) return this;
    ctx.fill(this.best, 0xffffffff);
    const bufs = { B: w.B, rays: this.rays, best: this.best };
    this.testK.run([w.count, count], { phase: 0 }, bufs);
    this.testK.run([w.count, count], { phase: 1 }, bufs);
    this.resolveK.run(count, { phase: 0 }, { ...bufs, hits: this.hits });
    return this;
  }
  /** Read hits: [{ hit, body, distance, normal, point }]. */
  async read(count = this.maxRays) {
    const h = await this.hits.read(count * 2);
    const r = await this.rays.read(count * 2);
    const u = new Uint32Array(h.buffer, h.byteOffset, h.length);
    const out = [];
    for (let k = 0; k < count; k++) {
      const t = h[8 * k];
      if (!(t >= 0)) { out.push({ hit: false, body: -1, distance: Infinity }); continue; }
      const o = r.subarray(8 * k, 8 * k + 3), d = r.subarray(8 * k + 4, 8 * k + 7);
      out.push({ hit: true, body: u[8 * k + 1], distance: t, normal: [h[8 * k + 4], h[8 * k + 5], h[8 * k + 6]],
        point: [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t] });
    }
    return out;
  }
}

export function eulerQuat([x, y, z]) {
  const cx = Math.cos(x / 2), sx = Math.sin(x / 2), cy = Math.cos(y / 2), sy = Math.sin(y / 2), cz = Math.cos(z / 2), sz = Math.sin(z / 2);
  return [sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz];
}
