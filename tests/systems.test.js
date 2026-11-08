import * as SI from '../src/index.js';
import { test, section, note, assert } from './harness.js';
import { createDuel, bot } from './duel.js';

const rnd = (n, seed) => { const a = new Float32Array(n); let s = seed; for (let k = 0; k < n; k++) { s = (s * 16807) % 2147483647; a[k] = s / 2147483647 - 0.5; } return a; };

export async function systemsTests(gpu) {
  section('neural networks');
  await test('gemm equals a CPU product for every transpose combination, bias + activation, and split-K', async () => {
    const cases = [[157, 93, 211, false, false], [157, 93, 211, true, false], [64, 129, 77, false, true], [33, 17, 3000, true, true], [96, 70, 5000, true, false]];
    let worst = 0;
    for (const [M, N, K, tA, tB] of cases) {
      const Ah = rnd(M * K, 3), Bh = rnd(K * N, 7), bh = rnd(N, 11);
      const A = gpu.array('f32', Ah), B = gpu.array('f32', Bh), C = gpu.array('f32', M * N), bias = gpu.array('f32', bh);
      SI.nn.gemm(gpu, A, B, C, { M, N, K, transA: tA, transB: tB, bias, act: 'relu' });
      const c = await C.read();
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        let s = bh[n];
        for (let k = 0; k < K; k++) s += (tA ? Ah[k * M + m] : Ah[m * K + k]) * (tB ? Bh[n * K + k] : Bh[k * N + n]);
        worst = Math.max(worst, Math.abs(Math.max(0, s) - c[m * N + n]) / Math.max(1, Math.sqrt(K) * 0.1));
      }
      for (const b of [A, B, C, bias]) b.destroy();
    }
    note(`worst scaled error ${worst.toExponential(2)} over ${cases.length} shapes (incl. K = 5000 via split-K)`);
    assert(worst < 1e-4, 'gemm differs from the CPU product');
  });
  await test('Freivalds certifies a 1024³ product and rejects one corrupted entry', async () => {
    const n = 1024;
    const A = gpu.array('f32', rnd(n * n, 5)), B = gpu.array('f32', rnd(n * n, 9)), C = gpu.array('f32', n * n);
    SI.nn.gemm(gpu, A, B, C, { M: n, N: n, K: n });
    const good = await SI.verify.freivalds(gpu, A, B, C, { M: n, N: n, K: n, trials: 6 });
    const c = await C.read(); c[123457] += 0.25; gpu.flush(); C.write(c);
    const bad = await SI.verify.freivalds(gpu, A, B, C, { M: n, N: n, K: n, trials: 6 });
    note(`correct: max relative ${good.maxRelative.toExponential(2)} (accepted) · corrupted: ${bad.maxRelative.toExponential(2)} (${bad.ok ? 'ACCEPTED' : 'rejected'})`);
    assert(good.ok && !bad.ok, 'certificate');
  });
  await test('backpropagation equals finite differences; an MLP fits sin(3x)·cos(2y) (loss ÷ 100)', async () => {
    const net = new SI.nn.MLP(gpu, { layers: [3, 16, 16, 2], act: 'tanh', out: 'none', maxBatch: 64, seed: 4 });
    const X = gpu.array('f32', new Float32Array(64 * 3).map((_, k) => Math.sin(k * 0.37))), T = gpu.array('f32', new Float32Array(64 * 2).map((_, k) => Math.cos(k)));
    const g = await SI.verify.checkGradient(net, X, T, 64);
    const N = 4096;
    const fit = new SI.nn.MLP(gpu, { layers: [2, 64, 64, 1], act: 'relu', out: 'none', fourier: 4, maxBatch: N, seed: 9 });
    const xs = new Float32Array(N * 2), ts = new Float32Array(N);
    for (let k = 0; k < N; k++) { const x = (k % 64) / 32 - 1, y = Math.floor(k / 64) / 32 - 1; xs[2 * k] = x; xs[2 * k + 1] = y; ts[k] = Math.sin(3 * x) * Math.cos(2 * y); }
    fit.input.write(xs);
    const T2 = gpu.array('f32', ts), loss = new SI.nn.MSELoss(gpu, N), opt = new SI.nn.Adam(gpu, fit.params, { lr: 3e-3 });
    let first = 0, last = 0;
    for (let it = 0; it < 300; it++) {
      fit.forward(fit.input, N);
      fit.backward(loss.run(fit.output, T2, N), N);
      opt.step();
      if (it === 0) first = await loss.readLoss();
    }
    last = await loss.readLoss();
    note(`gradient check: worst relative error ${g.maxRelativeError.toExponential(2)} · regression loss ${first.toExponential(2)} → ${last.toExponential(2)} in 300 Adam steps`);
    assert(g.ok, 'backprop disagrees with finite differences');
    assert(last < first / 100, 'MLP did not learn');
  });
  await test('evolution strategies: a population of 512 networks improves its mean fitness 5×', async () => {
    const pop = new SI.nn.Population(gpu, { layers: [4, 16, 1], count: 512, sigma: 0.1, lr: 0.05 });
    const obs = gpu.array('f32', new Float32Array(512 * 4).map((_, k) => Math.sin(k))), act = gpu.array('f32', 512);
    let first = 0, last = 0;
    for (let g = 0; g < 40; g++) {
      pop.forward(obs, act);
      const a = await act.read();
      const r = await pop.update(new Float32Array(512).map((_, k) => -((a[k] - 0.5) ** 2)));
      if (g === 0) first = r.mean;
      last = r.mean;
    }
    note(`mean fitness ${first.toExponential(2)} → ${last.toExponential(2)} over 40 generations`);
    assert(last > first / 5, 'no improvement');
  });

  section('fluid ↔ rigid coupling');
  await test('Archimedes: boards of density 0.25 / 0.5 / 0.75 float level with that fraction submerged (±0.15)', async () => {
    // Boards, not cubes: a floating cube is unstable upright for 0.21 < ρ < 0.79 (metacentric
    // height < 0) and really does tip over. A 0.4 × 0.24 board is stable upright at all three.
    // The waterline is read beside each board and averaged over the last 3 s: sloshing moves
    // and tilts the surface by about a centimetre. At dx = 3.1 cm the coupled boards float within
    // about one grid cell of Archimedes' waterline (deeper drafts sink a little too far).
    const size = [3, 1.2, 1.5];
    const sim = gpu.mpm({ grid: 96, size, maxParticles: 900000, substeps: 8 });
    const world = gpu.rigid({ maxBodies: 64, bounds: [[-1, -1, -1], [4, 4, 3]] });
    sim.couple(world, { friction: 0.2 });
    const inset = sim.dx * sim.bound;
    const wall = (p, h) => world.addBox({ position: p, half: h, static: true, group: 2, mask: 1 });
    wall([-0.25 + inset, 1, 0.75], [0.25, 1, 1.5]); wall([3.25 - inset, 1, 0.75], [0.25, 1, 1.5]);
    wall([1.5, 1, -0.25 + inset], [3, 1, 0.25]); wall([1.5, 1, 1.75 - inset], [3, 1, 0.25]);
    sim.add({ box: [[0.05, 0.05, 0.05], [2.95, 0.45, 1.45]] }, { material: 'water' });
    const hh = 0.12;
    const dens = [0.25, 0.5, 0.75], ids = dens.map((d, k) => world.addBox({ position: [0.6 + k * 0.9, 0.75, 0.75], half: [0.2, hh, 0.2], density: d, group: 1 }));
    const heavy = world.addSphere({ position: [1.5, 1.0, 0.35], radius: 0.1, density: 3, group: 1 });
    const frac = [0, 0, 0], tilt = [0, 0, 0];
    let level = 0, samples = 0;
    for (let f = 0; f < 420; f++) {
      sim.step(); world.step(1 / 60);
      if (f % 30 === 29) await gpu.idle();
      if (f < 240 || f % 30 !== 29) continue;
      // the waterline beside each board (sloshing tilts the surface across the tank)
      const xp = await sim.readPositions();
      const P = await world.readPoses();
      const near = ids.map(() => []);
      for (let k = 0; k < sim.count; k++) {
        const x = xp[3 * k], z = xp[3 * k + 2];
        if (Math.abs(z - 0.75) > 0.2) continue;
        ids.forEach((id, b) => { const d = Math.abs(x - P[id * 7]); if (d > 0.27 && d < 0.4) near[b].push(xp[3 * k + 1]); });
      }
      const lvs = near.map((ys) => { ys.sort((a, b) => a - b); return ys[Math.floor(ys.length * 0.98)]; });
      const lv = lvs.reduce((a, b) => a + b, 0) / lvs.length;
      ids.forEach((id, k) => {
        frac[k] += (lvs[k] - (P[id * 7 + 1] - hh)) / (2 * hh);
        // tilt: angle between the board's local y axis and the vertical
        const [x, , z] = P.slice(id * 7 + 3, id * 7 + 6);
        tilt[k] = Math.max(tilt[k], Math.acos(Math.min(1, Math.abs(1 - 2 * (x * x + z * z)))) * 180 / Math.PI);
      });
      level += lv; samples++;
    }
    frac.forEach((_, k) => (frac[k] /= samples));
    level /= samples;
    const P = await world.readPoses();
    note(`mean water level ${level.toFixed(3)} m · submerged fractions ${frac.map((f) => f.toFixed(3)).join(' / ')} (0.25 / 0.5 / 0.75) · tilt ${tilt.map((t) => t.toFixed(1)).join(' / ')}° · dense ball at y = ${P[heavy * 7 + 1].toFixed(3)} (rests on the bottom)`);
    assert(frac.every((f, k) => Math.abs(f - dens[k]) < 0.15) && tilt.every((t) => t < 10), 'buoyancy');
    assert(P[heavy * 7 + 1] < 0.15, 'a ball 3× denser than water must sink');
  });

  section('peer-to-peer netcode');
  await test('two peers over a lossy network (40 ± 15 ms, 10% loss) stay bit-identical through rollbacks', async () => {
    const net = new SI.net.LoopbackNetwork({ latency: 40, jitter: 15, loss: 0.1, seed: 5 });
    const [ca, cb] = net.connect();
    const A = createDuel(gpu), B = createDuel(gpu);
    const mk = (D, ch, p) => new SI.net.RollbackSession({ sim: D.world, game: D.game, channel: ch, players: 2, localPlayer: p, applyInputs: D.applyInputs, delay: 2, maxRollback: 10, hashEvery: 30 });
    const sa = mk(A, ca, 0), sb = mk(B, cb, 1), ba = bot(11), bb = bot(29);
    while (sa.tick < 360 || sb.tick < 360) { sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick)); await new Promise((r) => setTimeout(r, 8)); }
    await new Promise((r) => setTimeout(r, 400));
    for (let k = 0; k < 10; k++) { sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick)); await new Promise((r) => setTimeout(r, 30)); }
    await new Promise((r) => setTimeout(r, 300));
    const checked = [...sa.hashes.values()].filter((h) => h.local && h.remote);
    const same = checked.filter((h) => h.local === h.remote).length;
    // A peer may end a tick or two behind the other (it waits whenever its GPU is behind), so the
    // transcripts are compared over the ticks both have confirmed: inputs and the chain over them.
    const [ta, tb] = [sa.transcript(), sb.transcript()];
    const common = Math.min(ta.ticks, tb.ticks);
    const head = (tr) => tr.inputs.slice(0, common).reduce((h, row, t) => SI.net.chainHash(h, t, ...row.map((r) => Uint8Array.from(r))), '0000000000000000');
    const agree = common > 300 && head(ta) === head(tb);
    note(`${sa.stats.rollbacks} + ${sb.stats.rollbacks} rollbacks (${sa.stats.rolledBackTicks + sb.stats.rolledBackTicks} ticks re-simulated), ${net.stats.dropped} packets lost; state hashes equal at ${same}/${checked.length} checkpoints; ` +
      `transcripts ${agree ? 'identical' : 'DIFFERENT'} over ${common} common ticks${sa.stats.gpuWaits + sb.stats.gpuWaits ? ` (${sa.stats.gpuWaits + sb.stats.gpuWaits} ticks waited for the GPU)` : ''}`);
    assert(checked.length >= 8 && same === checked.length && sa.stats.desyncs === 0 && sb.stats.desyncs === 0, 'peers diverged');
    assert(sa.stats.rollbacks > 0, 'the test should exercise rollbacks');
    assert(agree, 'transcripts differ');
  });
  await test('a diverged peer is resynchronised by the authority\'s state (exact bits through the wire codec)', async () => {
    // Peer B starts with one tower box nudged — what different GPU rounding would do over time.
    // 80 ms each way: the state arrives many ticks after the authority picked it (a slower GPU's
    // read-back adds more), so the peer must still hold the inputs to replay from there.
    const net = new SI.net.LoopbackNetwork({ latency: 80, jitter: 10, loss: 0.05, seed: 9 });
    const [ca, cb] = net.connect();
    const A = createDuel(gpu), B = createDuel(gpu);
    B.world.applyImpulse(3, [0.02, 0, 0]);
    const mk = (D, ch, p) => new SI.net.RollbackSession({ sim: D.world, game: D.game, channel: ch, players: 2, localPlayer: p, applyInputs: D.applyInputs, delay: 2, maxRollback: 10, hashEvery: 30 });
    const sa = mk(A, ca, 0), sb = mk(B, cb, 1), ba = bot(5), bb = bot(17);
    while (sa.tick < 420 || sb.tick < 420) { sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick)); await new Promise((r) => setTimeout(r, 8)); }
    await new Promise((r) => setTimeout(r, 400));
    for (let k = 0; k < 10; k++) { sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick)); await new Promise((r) => setTimeout(r, 30)); }
    await new Promise((r) => setTimeout(r, 300));
    const both = [...sa.hashes.entries()].filter(([, h]) => h.local && h.remote).sort((x, y) => x[0] - y[0]);
    const tail = both.slice(-3);
    note(`mismatch detected at ${sa.stats.desyncs} checkpoint(s), ${sa.stats.statesSent} state(s) sent (${(net.stats.bytes / 1024).toFixed(0)} KB on the wire in total), ${sb.stats.resyncs} applied; ` +
      `last checkpoints ${tail.map(([t, h]) => `${t} ${h.local === h.remote ? '=' : '≠'}`).join(', ')}`);
    if (!(tail.length === 3 && tail.every(([, h]) => h.local === h.remote))) {
      const view = (s) => [...s.hashes.entries()].sort((x, y) => x[0] - y[0]).map(([t, h]) => `${t}:${h.local ? h.local.slice(0, 4) : '-'}/${h.remote ? h.remote.slice(0, 4) : '-'}`).join(' ');
      note(`A tick ${sa.tick} confirmed ${sa.confirmedTick} expectEpoch ${sa.expectEpoch} remoteEpoch ${sa.remoteEpoch} stats ${JSON.stringify(sa.stats)} · ${view(sa)}`);
      note(`B tick ${sb.tick} confirmed ${sb.confirmedTick} epoch ${sb.epoch} stats ${JSON.stringify(sb.stats)} · ${view(sb)}`);
    }
    assert(sa.stats.desyncs >= 1 && sb.stats.resyncs >= 1, 'the divergence was not detected and repaired');
    assert(tail.length === 3 && tail.every(([, h]) => h.local === h.remote), 'peers still differ after the resync');
  });
  await test('a match transcript replays to the same state hashes; a forged input breaks it', async () => {
    const net = new SI.net.LoopbackNetwork({ latency: 30, jitter: 10, loss: 0.05, seed: 9 });
    const [ca, cb] = net.connect();
    const A = createDuel(gpu), B = createDuel(gpu);
    const mk = (D, ch, p) => new SI.net.RollbackSession({ sim: D.world, game: D.game, channel: ch, players: 2, localPlayer: p, applyInputs: D.applyInputs, hashEvery: 30 });
    const sa = mk(A, ca, 0), sb = mk(B, cb, 1), ba = bot(3), bb = bot(8);
    while (sa.tick < 240) { sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick)); await new Promise((r) => setTimeout(r, 6)); }
    await new Promise((r) => setTimeout(r, 400));
    sa.advance(ba(sa.tick)); sb.advance(bb(sb.tick));
    await new Promise((r) => setTimeout(r, 300));
    const tr = sa.transcript();
    let D = null;
    const ok = await SI.net.verifyTranscript(tr, () => (D = createDuel(gpu)).world, (sim, t, inputs) => D.applyInputs(t, inputs));
    const bad = JSON.parse(JSON.stringify(tr));
    bad.inputs[100][1][0] ^= 1;
    let D2 = null;
    const rej = await SI.net.verifyTranscript(bad, () => (D2 = createDuel(gpu)).world, (sim, t, inputs) => D2.applyInputs(t, inputs));
    note(`${tr.ticks} ticks, ${ok.checkedHashes} state hashes replayed: ${ok.ok ? 'verified' : 'FAILED'} · forged input at tick 100: ${rej.ok ? 'ACCEPTED' : 'rejected'}`);
    assert(ok.ok && !rej.ok, 'transcript certificate');
  });

  section('laws (LAWS.md)');
  const laws = await SI.verify.proveAll(gpu, SI);
  for (const law of laws) {
    await test(`law ${law.name}: ${law.statement}`, async () => {
      note(JSON.stringify(law.evidence).slice(0, 200));
      assert(law.ok, law.evidence.error || 'law violated');
    });
  }
}
