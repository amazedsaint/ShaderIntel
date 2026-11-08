// Home page: the example mosaic (from examples/files.json), code highlighting, the mobile menu,
// and the hero — a live ShaderIntel scene (MLS-MPM water two-way coupled with rigid crates) that
// the pointer stirs. Without WebGPU, or with reduced motion requested, a still image stands in.
import { highlightAll } from './highlight.js';
import { setupMenu, setupCopyButtons } from './chrome.js';

setupMenu();
highlightAll();
setupCopyButtons();
buildMosaic();
startHero();

async function buildMosaic() {
  const root = document.getElementById('mosaic');
  try {
    const { categories } = await (await fetch('examples/files.json')).json();
    const featured = categories.flatMap((c) => c.examples).filter((e) => e.featured).sort((a, b) => a.featured - b.featured);
    root.innerHTML = '';
    featured.forEach((e, k) => {
      const a = document.createElement('a');
      a.className = k === 0 ? 'tile lead' : 'tile';
      a.href = `examples/#${e.id}`;
      a.title = e.description;
      const img = document.createElement('img');
      img.src = `examples/screenshots/${e.id}.jpg`;
      img.alt = '';
      img.loading = k < 4 ? 'eager' : 'lazy';
      img.width = 640; img.height = 400;
      const label = document.createElement('span');
      label.textContent = e.title;
      a.append(img, label);
      root.appendChild(a);
    });
  } catch {
    root.innerHTML = '<p class="text">The example list could not be loaded. <a href="examples/">Open the example browser</a>.</p>';
  }
}

// ------------------------------------------------------------------ hero

async function startHero() {
  const canvas = document.getElementById('hero-canvas');
  const poster = document.getElementById('hero-poster');
  const status = document.getElementById('hero-status');
  const still = (message) => {
    canvas.hidden = true;
    poster.hidden = false;
    status.innerHTML = message;
  };
  const SI = await import('../src/index.js');
  if (!SI.isSupported()) return still('This browser has no WebGPU, so this is a still image of the live scene.');
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    still('You asked for reduced motion, so this is a still image. ');
    const run = document.createElement('button');
    run.className = 'btn small';
    run.type = 'button';
    run.textContent = 'Run the live scene';
    run.onclick = () => { status.textContent = ''; poster.hidden = true; canvas.hidden = false; live(SI, canvas, status).catch((e) => still(failure(e))); };
    status.appendChild(document.createElement('br'));
    status.appendChild(run);
    return;
  }
  try { await live(SI, canvas, status); } catch (e) { still(failure(e)); }
}

const failure = (e) => (e?.code === 'no-adapter' || e?.code === 'no-device'
  ? 'WebGPU is turned off or unavailable on this device, so this is a still image of the live scene.'
  : `The live scene could not start (${String(e?.message || e).replace(/^ShaderIntel: /, '')}), so this is a still image.`);

async function live(SI, canvas, status) {
  const gpu = await SI.init();
  const small = Math.min(innerWidth, innerHeight * 1.6) < 820 || (navigator.hardwareConcurrency || 8) < 6;
  const view = gpu.view(canvas, {
    controls: false, maxPixelRatio: small ? 1 : 1.5, tonemap: 'agx',
    // framed so the tank sits up and to the right, leaving the lower left to the headline
    camera: small ? { target: [1.5, 0.05, 0.75], distance: 4.3, yaw: 0.6, pitch: 0.55, fov: 40 }
      : { target: [0.84, -0.22, 1.28], distance: 4.1, yaw: 0.68, pitch: 0.42, fov: 40 },
    onError: (e) => { canvas.hidden = true; document.getElementById('hero-poster').hidden = false; status.textContent = failure(e); },
  });
  view.add(gpu.stage({ cell: 0.25 }));

  // the tank: 3 × 1.2 × 1.5 m of MPM grid, a rigid world, coupled both ways
  const size = [3, 1.2, 1.5];
  const sim = gpu.mpm({ grid: small ? 44 : 60, size, maxParticles: 1 << 19, substeps: 8 });
  const world = gpu.rigid({ maxBodies: 64, bounds: [[-1, -1, -1], [4, 4, 3]] });
  sim.couple(world, { friction: 0.2 });
  const inset = sim.dx * sim.bound;
  const wall = (p, h) => world.addBox({ position: p, half: h, static: true, group: 2, mask: 1 });
  wall([-0.25 + inset, 1, size[2] / 2], [0.25, 1, size[2]]);
  wall([size[0] + 0.25 - inset, 1, size[2] / 2], [0.25, 1, size[2]]);
  wall([size[0] / 2, 1, -0.25 + inset], [size[0], 1, 0.25]);
  wall([size[0] / 2, 1, size[2] + 0.25 - inset], [size[0], 1, 0.25]);
  sim.add({ box: [[0.05, 0.05, 0.05], [2.95, 0.42, 1.45]] }, { material: 'water' });
  const crates = [
    world.addBox({ position: [0.6, 0.75, 0.6], half: [0.12, 0.12, 0.12], density: 0.3, group: 1, rotation: [0.2, 0.4, 0] }),
    world.addBox({ position: [1.5, 0.9, 0.9], half: [0.16, 0.1, 0.12], density: 0.5, group: 1, rotation: [0, 0.8, 0.1] }),
    world.addBox({ position: [2.4, 0.8, 0.55], half: [0.12, 0.12, 0.12], density: 0.7, group: 1 }),
    world.addCapsule({ position: [1.0, 0.85, 1.1], radius: 0.05, halfHeight: 0.16, rotation: [0, 0.3, Math.PI / 2], density: 0.6, group: 1 }),
    world.addBox({ position: [2.0, 1.1, 1.05], half: [0.2, 0.05, 0.11], density: 0.45, group: 1, rotation: [0.1, -0.5, 0] }),
  ];
  const ball = world.addSphere({ position: [1.25, 1.3, 0.45], radius: 0.09, density: 2.5, group: 1 });
  view.add(sim.spheres({ solidsOnly: true }));
  view.add(sim.fluid());
  view.add(gpu.bodies(world, { groups: 1 }));   // the tank walls (group 2) stay invisible

  // the pointer is a sphere moving through the water (no click needed; on touch, drag)
  const stir = { type: 'sphere', center: [1.5, 0.3, 0.75], radius: 0.13, velocity: [0, 0, 0], friction: 0.1 };
  let stirring = false, lastMove = 0;
  const onMove = () => { stirring = true; lastMove = performance.now(); };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerdown', onMove);
  canvas.addEventListener('pointerleave', () => { stirring = false; });

  // every few seconds the crate that has been still longest is dropped back in from above
  let drop = 0, dropAt = 6;
  const rnd = (() => { let s = 12345; return () => ((s = (s * 16807) % 2147483647) / 2147483647); })();

  let acc = 0;
  view.run(({ dt, time }) => {
    const p = view.pointer;
    const active = stirring && p.inside && performance.now() - lastMove < 1500;
    if (active) {
      const r = view.ray();
      const t = (0.28 - r.origin[1]) / r.dir[1];
      if (t > 0) {
        const target = [r.origin[0] + r.dir[0] * t, 0.28, r.origin[2] + r.dir[2] * t];
        target[0] = Math.min(2.85, Math.max(0.15, target[0]));
        target[2] = Math.min(1.35, Math.max(0.15, target[2]));
        stir.velocity = target.map((x, a) => (x - stir.center[a]) / Math.max(dt, 1e-3));
        stir.center = target;
      }
    }
    sim.setColliders(active ? [stir] : []);
    if (time > dropAt) {
      const id = crates[drop++ % crates.length];
      world.setPose(id, [0.4 + rnd() * 2.2, 1.15, 0.3 + rnd() * 0.9], [rnd() * 3, rnd() * 3, rnd() * 3]);
      world.setVelocity(id, [0, -1.5, 0], [rnd() * 4 - 2, rnd() * 4 - 2, 0]);
      dropAt = time + 5 + rnd() * 3;
    }
    // fixed 1/60 s steps: the fluid then the bodies, in lockstep (MPM.step() is one 1/60 s step)
    acc = Math.min(acc + dt, 3 / 60);
    while (acc >= 1 / 60 - 1e-6) { sim.step(); world.step(1 / 60); acc -= 1 / 60; }
  });
  status.innerHTML = `<span class="live" aria-hidden="true"></span>Running live on this GPU: <b>${Math.round(sim.count / 1000)},000 particles of water</b> ` +
    `and six bodies, coupled both ways. Move the pointer through the water to stir it.`;

  // pause while the hero is scrolled out of view
  new IntersectionObserver(([entry]) => { if (entry.isIntersecting) view.run(); else view.stop(); }, { threshold: 0.02 }).observe(canvas);
  Object.assign(window, { gpu, view, sim, world, ball });
}
