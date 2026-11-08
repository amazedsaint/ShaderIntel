// Screenshots for the site: every entry of examples/files.json → examples/screenshots/<id>.jpg
// (640×400), and the home page's live hero → site/hero.jpg (its still fallback). Runs the pages
// in headless Chrome on this machine's GPU, with the UI overlays hidden.
//   node tools/thumbnails.mjs [--only rigid,mpm] [--chrome PATH] [--wait 5]
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, existsSync, rmSync, mkdirSync } from 'node:fs';
import { tmpdir, platform } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const arg = (name, def) => { const k = process.argv.indexOf(`--${name}`); return k > 0 ? process.argv[k + 1] : def; };
const only = arg('only', '') ? new Set(arg('only').split(',')) : null;
const defaultWait = +arg('wait', 5);
const chromePath = arg('chrome', { darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', linux: '/opt/google/chrome/chrome',
  win32: 'C:/Program Files/Google/Chrome/Application/chrome.exe' }[platform()]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const HIDE = '.si-panel, .rp, .info, .si-hud, #si-error, .si-back { display: none !important; }';

// a private static server and a private Chrome
const port = 8900 + Math.floor(Math.random() * 90);
const server = spawn(process.execPath, [join(root, 'tools/serve.mjs'), String(port)], { stdio: 'ignore' });
const dir = mkdtempSync(join(tmpdir(), 'si-thumbs-'));
const chrome = spawn(chromePath, ['--headless=new', `--user-data-dir=${dir}`, '--remote-debugging-port=0', '--no-first-run', '--enable-unsafe-webgpu',
  '--enable-gpu', '--ignore-gpu-blocklist', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--hide-scrollbars',
  ...(platform() === 'linux' ? ['--enable-features=Vulkan', '--use-vulkan', '--use-angle=vulkan'] : []), 'about:blank'], { stdio: 'ignore' });
const portFile = join(dir, 'DevToolsActivePort');
for (let k = 0; k < 200 && !existsSync(portFile); k++) await sleep(100);
const [cdpPort, path] = readFileSync(portFile, 'utf8').trim().split('\n');
const ws = new WebSocket(`ws://127.0.0.1:${cdpPort}${path}`);
await new Promise((r) => { ws.onopen = r; });
let id = 0;
const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result); } };
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => { const msg = { id: ++id, method, params }; if (sessionId) msg.sessionId = sessionId; pending.set(msg.id, { resolve, reject }); ws.send(JSON.stringify(msg)); });

async function shoot(url, out, { wait = defaultWait, width = 1280, height = 800, scale = 0.5, clipSelector = null, extraCss = '' } = {}) {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank', newWindow: true });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const S = (m, p) => send(m, p, sessionId);
  await S('Page.enable');
  await S('Runtime.enable');
  await S('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  await S('Page.navigate', { url });
  await sleep(wait * 1000);
  await S('Runtime.evaluate', { expression: `(() => { const s = document.createElement('style'); s.textContent = ${JSON.stringify(HIDE + extraCss)}; document.head.appendChild(s); })()` });
  await sleep(400);   // let a frame render without the overlays
  let clip = { x: 0, y: 0, width, height, scale };
  if (clipSelector) {
    const r = await S('Runtime.evaluate', { expression: `JSON.stringify(document.querySelector(${JSON.stringify(clipSelector)}).getBoundingClientRect())`, returnByValue: true });
    const b = JSON.parse(r.result.value);
    clip = { x: b.x, y: b.y, width: b.width, height: b.height, scale };
  }
  const { data } = await S('Page.captureScreenshot', { format: 'jpeg', quality: 84, clip });
  writeFileSync(out, Buffer.from(data, 'base64'));
  await send('Target.closeTarget', { targetId });
}

try {
  const base = `http://localhost:${port}`;
  await sleep(500);
  const { categories } = JSON.parse(readFileSync(join(root, 'examples/files.json'), 'utf8'));
  mkdirSync(join(root, 'examples/screenshots'), { recursive: true });
  for (const e of categories.flatMap((c) => c.examples)) {
    if (only && !only.has(e.id)) continue;
    const out = join(root, 'examples/screenshots', `${e.id}.jpg`);
    const t0 = Date.now();
    await shoot(`${base}/examples/${e.path}`, out, { wait: e.wait ?? (e.heavy ? 3 : defaultWait) });
    console.log(`${e.id.padEnd(16)} ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
  if (!only || only.has('hero')) {
    await shoot(`${base}/`, join(root, 'site/hero.jpg'), { wait: 9, width: 1600, height: 1000, scale: 1, clipSelector: '.hero',
      extraCss: '.hero-copy, .hero-status { display: none !important; } .hero::after { display: none !important; }' });
    console.log('hero             site/hero.jpg');
  }
} finally {
  ws.close();
  chrome.kill();
  server.kill();
  try { rmSync(dir, { recursive: true, force: true }); } catch { /* Chrome may still hold it */ }
}
