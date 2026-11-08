// "Tower duel": the tiny two-player game used by the P2P example and the netcode tests.
// Inputs are one byte per player per tick: bit 0 move −z, bit 1 move +z, bit 2 fire.
export function createDuel(gpu, o = {}) {
  const world = gpu.rigid({ maxBodies: o.maxBodies || 1024, iterations: 8, deterministic: true, bounds: [[-20, -2, -12], [20, 20, 12]] });
  const cannons = [];
  for (const side of [-1, 1]) {
    for (let y = 0; y < 6; y++) for (let k = 0; k < 3; k++)
      world.addBox({ position: [side * 5, 0.4 + y * 0.8, (k - 1) * 0.85], half: [0.4, 0.4, 0.4] });
    cannons.push(world.addBox({ position: [side * 11, 1, 0], half: [0.6, 0.4, 0.6], kinematic: true }));
  }
  const game = {
    z: [0, 0], cooldown: [0, 0], shots: [0, 0],
    save() { return { z: this.z.slice(), cooldown: this.cooldown.slice(), shots: this.shots.slice() }; },
    load(s) { this.z = s.z.slice(); this.cooldown = s.cooldown.slice(); this.shots = s.shots.slice(); },
  };
  function applyInputs(tick, inputs) {
    for (let p = 0; p < 2; p++) {
      const b = inputs[p][0] || 0, side = p === 0 ? -1 : 1;
      game.z[p] = Math.max(-4, Math.min(4, game.z[p] + ((b >> 1) & 1) * 0.08 - (b & 1) * 0.08));
      world.moveKinematic(cannons[p], [side * 11, 1, game.z[p]]);
      if (game.cooldown[p] > 0) game.cooldown[p]--;
      else if (b & 4) {
        world.addSphere({ position: [side * 9.8, 1.4, game.z[p]], radius: 0.3, density: 6, velocity: [-side * 16, 3.5, 0] });
        game.cooldown[p] = 24;
        game.shots[p]++;
      }
    }
  }
  return { world, game, applyInputs, cannons };
}

/** A deterministic scripted player: sweeps, fires in bursts (different per seed). */
export function bot(seed) {
  let s = seed >>> 0 || 1;
  let dir = 1, hold = 0;
  return (tick) => {
    if (hold-- <= 0) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; dir = (s & 3) === 0 ? 0 : (s & 1 ? 1 : 2); hold = 10 + (s >>> 8) % 30; }
    return new Uint8Array([dir | ((tick + seed) % 37 < 3 ? 4 : 0)]);
  };
}
