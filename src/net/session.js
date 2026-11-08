// Rollback netcode (in the style of GGPO) for deterministic GPU simulations.
//
// Every peer runs the whole simulation at a fixed tick. A player's input for tick t is
// applied at t + delay; inputs travel redundantly over an unreliable channel (the last
// `redundancy` ticks in every packet), so loss costs nothing until it exceeds that window.
// When a remote input has not arrived the session predicts it (the player's last input) and
// keeps going; when the real input differs from the prediction it restores the GPU snapshot
// taken before that tick and re-simulates up to the present — within one frame.
//
// This only works if the simulation is a pure function of (state, inputs): RigidWorld in
// deterministic mode is (see physics/rigid.js — confluence). Peers on different GPUs can still
// diverge in the last bits of floating point, so every `hashEvery` confirmed ticks the peers
// exchange 64-bit state hashes. On a mismatch the authority (player 0) sends the state of its
// newest fully confirmed tick, taken from its rollback ring; the other peer imports it at that
// tick and re-simulates to the present with the inputs it has (resync). Each applied state
// starts a new epoch, and hashes from an abandoned history (an older epoch) are ignored, so one
// divergence costs one state transfer.
//
// Proof-carrying transcript: the session folds every confirmed tick's inputs into a hash chain
// and records the state hash every `hashEvery` ticks. The transcript (inputs + state hashes +
// chain head) is a certificate of the match: anyone holding the same code can replay it
// (verifyTranscript) and check every state hash, and changing any input breaks the chain.

const EMPTY = new Uint8Array(1);

function sameBytes(a, b) {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
}

/** 64-bit FNV-1a-style mix over strings / bytes (two 32-bit lanes, as a hex string). */
export function chainHash(prev, ...parts) {
  let a = parseInt(prev.slice(8), 16) >>> 0 || 0x811c9dc5, b = parseInt(prev.slice(0, 8), 16) >>> 0 || 0x01000193;
  const eat = (x) => { a = Math.imul(a ^ x, 0x01000193) >>> 0; b = Math.imul(b ^ (x + a), 0x5bd1e995) >>> 0; };
  for (const p of parts) {
    if (typeof p === 'number') { eat(p & 0xff); eat((p >>> 8) & 0xff); eat((p >>> 16) & 0xff); eat(p >>> 24); }
    else if (typeof p === 'string') for (let k = 0; k < p.length; k++) eat(p.charCodeAt(k));
    else for (let k = 0; k < p.length; k++) eat(p[k]);
  }
  return b.toString(16).padStart(8, '0') + a.toString(16).padStart(8, '0');
}

export class RollbackSession {
  /**
   * o: sim (RigidWorld or any { snapshot(into), restore(s), step(dt), hashRecord(target, source) }),
   *    game ({ save() → state, load(state) } for game state kept outside the simulation),
   *    channel ({ unreliable, reliable } from a transport), players (default 2), localPlayer,
   *    applyInputs(tick, inputs: Uint8Array[]) — turns every player's input into simulation commands,
   *    inputSize (bytes, default 1), delay (ticks, default 2), maxRollback (default 8), dt (default 1/60),
   *    hashEvery (default 30), redundancy (default 8), onDesync(tick, local, remote),
   *    resync (default true: the authority sends its state on a mismatch), authority (default 0),
   *    inputHistory (ticks of confirmed inputs kept so a late state can still be replayed, default 600),
   *    maxInFlight (ticks of GPU work allowed in the queue before the session waits a tick, default 3)
   */
  constructor(o) {
    this.sim = o.sim;
    this.game = o.game || null;
    this.players = o.players ?? 2;
    this.local = o.localPlayer;
    this.channel = o.channel;
    this.applyInputs = o.applyInputs;
    this.inputSize = o.inputSize ?? 1;
    this.delay = o.delay ?? 2;
    this.maxRollback = o.maxRollback ?? 8;
    this.dt = o.dt ?? 1 / 60;
    this.hashEvery = o.hashEvery ?? 30;
    this.redundancy = o.redundancy ?? 8;
    this.onDesync = o.onDesync || null;
    this.resync = o.resync ?? true;
    this.authority = o.authority ?? 0;
    this.inputHistory = o.inputHistory ?? 600;
    // Backpressure: a loop that ticks faster than the GPU executes would grow the queue without
    // bound, and every read-back (hashes, a resync export) would wait behind it — 16 s once on a
    // slow-dispatch driver. Each tick is fenced; past maxInFlight unfinished ticks we wait.
    this.maxInFlight = o.maxInFlight ?? 3;
    this._inflight = 0;
    this.epoch = 0;              // states applied from the authority so far (this peer)
    this.remoteEpoch = 0;        // the newest epoch seen in the peer's hashes
    this.expectEpoch = 0;        // hashes from older epochs describe a history the peer has abandoned
    this.resyncWanted = false;
    this.pendingState = null;
    this.tick = 0;
    this.inputs = Array.from({ length: this.players }, () => new Map());
    this.predicted = Array.from({ length: this.players }, () => new Map());
    this.confirmed = new Array(this.players).fill(-1);   // highest tick t with every input ≤ t known
    this.ring = new Array(this.maxRollback + 2).fill(null);
    this.rollbackTo = Infinity;
    this.hashes = new Map();        // tick -> { local, remote }
    this.pendingHash = [];          // ticks whose state hash is still to be taken
    this.chain = '0000000000000000';
    this.chainTick = -1;
    this.log = { inputs: [], hashes: {} };
    this.stats = { rollbacks: 0, rolledBackTicks: 0, stalls: 0, desyncs: 0, resyncs: 0, statesSent: 0, statesReceived: 0, exportErrors: 0, gpuWaits: 0, hashesRecorded: 0, hashesDropped: 0, hashesStale: 0, ping: 0, inSyncTick: -1, sent: 0, received: 0 };
    const zero = new Uint8Array(this.inputSize);
    for (let t = 0; t < this.delay; t++) this.inputs[this.local].set(t, zero);
    this.confirmed[this.local] = this.delay - 1;
    this.channel.unreliable.onmessage = (m) => this._onInputs(m);
    this.channel.reliable.onmessage = (m) => this._onReliable(m);
  }

  /** Lowest tick every player has confirmed (ticks ≤ this are final). */
  get confirmedTick() { return Math.min(...this.confirmed); }

  /**
   * Advance by one tick with this frame's local input (Uint8Array of inputSize bytes).
   * Returns the number of ticks simulated (0 while waiting for a peer that is too far behind).
   */
  advance(localInput = EMPTY) {
    const t = this.tick + this.delay;
    if (!this.inputs[this.local].has(t)) {
      this.inputs[this.local].set(t, Uint8Array.from(localInput));
      this.confirmed[this.local] = t;
    }
    this._send();
    if (this.pendingState && this.pendingState.tick <= this.tick) this._applyState();
    if (this.rollbackTo < this.tick) this._rollback();
    if (this.resyncWanted) this._sendState();
    this._hashConfirmed();
    this._extendChain();
    if (this.tick - this.confirmedTick > this.maxRollback) { this.stats.stalls++; return 0; }
    if (this._inflight > this.maxInFlight) { this.stats.gpuWaits++; return 0; }
    this._simulate(this.tick);
    this.tick++;
    const ctx = this.sim.ctx;
    if (ctx?.device) {
      ctx.flush();
      this._inflight++;
      ctx.device.queue.onSubmittedWorkDone().then(() => this._inflight--);
    }
    return 1;
  }

  _inputFor(p, t) {
    let v = this.inputs[p].get(t);
    if (!v) {
      // prediction: the most recent input we have from that player
      for (let s = Math.min(t, this.confirmed[p]); s >= 0 && !v; s--) v = this.inputs[p].get(s);
      v = v || new Uint8Array(this.inputSize);
      this.predicted[p].set(t, v);
    } else this.predicted[p].delete(t);
    return v;
  }

  _simulate(t) {
    const slot = t % this.ring.length;
    const prev = this.ring[slot];
    this.ring[slot] = { tick: t, snap: this.sim.snapshot(prev?.snap), game: this.game ? structuredClone(this.game.save()) : null };
    const inputs = [];
    for (let p = 0; p < this.players; p++) inputs.push(this._inputFor(p, t));
    this.applyInputs(t, inputs);
    this.sim.step(this.dt);
    if (t % this.hashEvery === 0 && !this.pendingHash.includes(t)) this.pendingHash.push(t);
  }

  _rollback() {
    const from = this.rollbackTo;
    this.rollbackTo = Infinity;
    const e = this.ring[from % this.ring.length];
    if (!e || e.tick !== from) throw new Error(`RollbackSession: tick ${from} is outside the rollback window`);
    this.sim.restore(e.snap);
    if (this.game) this.game.load(structuredClone(e.game));
    const to = this.tick;
    for (let t = from; t < to; t++) this._simulate(t);
    this.stats.rollbacks++;
    this.stats.rolledBackTicks += to - from;
  }

  // The state after tick T is final once every input ≤ T is known; it is the snapshot taken
  // before tick T + 1 (still in the ring). Hash it on the GPU and share it.
  _hashConfirmed() {
    const c = this.confirmedTick;
    this.pendingHash = this.pendingHash.filter((T) => {
      if (T > c || T + 1 >= this.tick) return true;
      const e = this.ring[(T + 1) % this.ring.length];
      if (!e || e.tick !== T + 1) { this.stats.hashesDropped++; return false; }   // fell out of the window: skip this one
      const buf = this.sim.ctx.array('u32', 4);
      this.sim.hashRecord(buf, e.snap);
      this.stats.hashesRecorded++;
      const epoch = this.epoch;
      buf.read(2).then((h) => {
        buf.destroy();
        if (epoch !== this.epoch) { this.stats.hashesStale++; return; }   // a state arrived meanwhile: this hash belongs to the old history
        const hex = h[1].toString(16).padStart(8, '0') + h[0].toString(16).padStart(8, '0');
        this._gotHash(T, hex, 'local');
        this.channel.reliable.send({ type: 'hash', from: this.local, tick: T, hash: hex, epoch });
      });
      return false;
    });
  }

  _gotHash(T, hex, who, epoch = 0) {
    if (who === 'remote') {
      if (epoch < this.expectEpoch) return;
      this.remoteEpoch = Math.max(this.remoteEpoch, epoch);
    }
    const h = this.hashes.get(T) || {};
    h[who] = hex;
    this.hashes.set(T, h);
    if (who === 'local') this.log.hashes[T] = hex;
    if (h.local && h.remote) {
      if (h.local === h.remote) this.stats.inSyncTick = Math.max(this.stats.inSyncTick, T);
      else if (!h.reported) {
        h.reported = true;
        this.stats.desyncs++;
        this.onDesync?.(T, h.local, h.remote);
        if (this.resync && this.local === this.authority) this.resyncWanted = true;
      }
    }
  }

  // Authority: send the state before tick S, the newest one whose every input is final.
  _sendState() {
    this.resyncWanted = false;
    const S = Math.min(this.confirmedTick + 1, this.tick - 1);
    const e = this.ring[S % this.ring.length];
    if (!e || e.tick !== S || !this.sim.exportState) return;
    const epoch = this.remoteEpoch + 1;
    this.expectEpoch = epoch;
    this.stats.statesSent++;
    const t0 = performance.now();
    this.sim.exportState(e.snap).then((state) => {
      this.stats.lastExport = { tick: S, ms: Math.round(performance.now() - t0), sentAt: this.tick };
      this.channel.reliable.send({ type: 'state', from: this.local, tick: S, epoch, state, game: e.game });
    })
      .catch((err) => {   // the peer keeps asking (its hashes still differ): count, report, retry on the next mismatch
        this.stats.exportErrors++;
        this.expectEpoch = epoch - 1;
        console.warn('RollbackSession: state export failed', err);
      });
  }

  // Other peers: adopt the authority's state at its tick and re-simulate to the present.
  _applyState() {
    const m = this.pendingState;
    this.pendingState = null;
    const S = m.tick;
    // A state older than the inputs we kept cannot be replayed to the present: ask for a newer one
    // (the authority ignores our old-epoch hashes from now on, so it would not notice by itself).
    if (S < this.tick - this.inputHistory + this.ring.length) {
      this.channel.reliable.send({ type: 'resync', from: this.local, epoch: this.epoch });
      return;
    }
    this.sim.importState(m.state);
    if (this.game && m.game) this.game.load(structuredClone(m.game));
    this.epoch = m.epoch;
    this.stats.lastResync = { tick: S, appliedAt: this.tick };
    this.rollbackTo = Infinity;
    // hashes of the abandoned history are recomputed (and re-sent) from the new one
    for (const [T, h] of this.hashes) if (T >= S - 1) { delete h.local; h.reported = false; }
    for (let T = Math.ceil(Math.max(0, S - 1) / this.hashEvery) * this.hashEvery; T < this.tick; T += this.hashEvery)
      if (!this.pendingHash.includes(T)) this.pendingHash.push(T);
    const to = this.tick;
    for (let t = S; t < to; t++) this._simulate(t);
    this.stats.resyncs++;
  }

  // Fold confirmed ticks into the transcript's hash chain (inputs, then any state hash).
  _extendChain() {
    const c = this.confirmedTick;
    while (this.chainTick < c && this.chainTick + 1 < this.tick) {
      const t = this.chainTick + 1;
      const row = [];
      for (let p = 0; p < this.players; p++) row.push(this.inputs[p].get(t) || new Uint8Array(this.inputSize));
      this.log.inputs.push(row.map((r) => Array.from(r)));
      this.chain = chainHash(this.chain, t, ...row);
      this.chainTick = t;
      // forget what can never be replayed again (rollbacks need ring.length ticks, a late resync state more)
      const old = t - Math.max(this.ring.length + this.redundancy + 2, this.inputHistory);
      if (old >= 0) for (let p = 0; p < this.players; p++) { this.inputs[p].delete(old); this.predicted[p].delete(old); }
    }
  }

  _send() {
    const t1 = this.tick + this.delay;
    const t0 = Math.max(0, t1 - this.redundancy + 1);
    const list = [];
    for (let t = t0; t <= t1; t++) list.push(Array.from(this.inputs[this.local].get(t) || EMPTY));
    this.channel.unreliable.send({ from: this.local, start: t0, inputs: list, ack: Math.min(...this.confirmed.filter((_, p) => p !== this.local)), sent: performance.now(), echo: this._echo });
    this.stats.sent++;
  }

  _onInputs(m) {
    if (m.from === this.local) return;
    this.stats.received++;
    if (m.echo) this.stats.ping = Math.round(performance.now() - m.echo);
    this._echo = m.sent;
    const p = m.from;
    m.inputs.forEach((arr, k) => {
      const t = m.start + k;
      if (this.inputs[p].has(t)) return;
      const v = Uint8Array.from(arr);
      this.inputs[p].set(t, v);
      if (t < this.tick) {
        const pred = this.predicted[p].get(t);
        if (!sameBytes(pred, v)) this.rollbackTo = Math.min(this.rollbackTo, t);
        this.predicted[p].delete(t);
      }
    });
    while (this.inputs[p].has(this.confirmed[p] + 1)) this.confirmed[p]++;
  }

  _onReliable(m) {
    if (m.from === this.local) return;
    if (m.type === 'hash') this._gotHash(m.tick, m.hash, 'remote', m.epoch ?? 0);
    if (m.type === 'resync' && this.resync && this.local === this.authority) this.resyncWanted = true;
    if (m.type === 'state') {
      this.stats.statesReceived++;
      this.stats.lastStateReceivedAt = this.tick;
      if (this.stateHandler) this.stateHandler(m);   // a custom policy
      else if (this.resync && (!this.pendingState || m.epoch > this.pendingState.epoch)) this.pendingState = m;
    }
  }

  /** The certificate of the match so far: per-tick inputs, state hashes and the chain head. */
  transcript() {
    return { players: this.players, inputSize: this.inputSize, delay: this.delay, hashEvery: this.hashEvery, dt: this.dt,
      inputs: this.log.inputs.slice(), hashes: { ...this.log.hashes }, chain: this.chain, ticks: this.chainTick + 1 };
  }
}

/**
 * Re-execute a transcript from scratch and check every recorded state hash and the chain head.
 * makeSim() → a fresh simulation in the starting state; applyInputs(sim, tick, inputs) as in the
 * session (game state included). Returns { ok, ticks, checkedHashes, firstMismatch }.
 */
export async function verifyTranscript(tr, makeSim, applyInputs) {
  const sim = await makeSim();
  let chain = '0000000000000000', checked = 0;
  for (let t = 0; t < tr.ticks; t++) {
    const row = tr.inputs[t].map((r) => Uint8Array.from(r));
    applyInputs(sim, t, row);
    sim.step(tr.dt);
    chain = chainHash(chain, t, ...row);
    if (tr.hashes[t] !== undefined) {
      const h = await sim.hash();
      checked++;
      if (h !== tr.hashes[t]) return { ok: false, ticks: t + 1, checkedHashes: checked, firstMismatch: { tick: t, expected: tr.hashes[t], got: h } };
    }
  }
  return { ok: chain === tr.chain, ticks: tr.ticks, checkedHashes: checked, chain, firstMismatch: chain === tr.chain ? null : { chain: true } };
}
