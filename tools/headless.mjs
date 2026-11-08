// Zero-dependency headless runner: drives Chrome over the DevTools protocol (Node ≥ 22 has
// a global WebSocket), so the test suite, the benchmarks or any expression can run from a
// terminal, a CI job or a GPU container.
//
//   node tools/headless.mjs tests [--url http://localhost:8787/tests/] [--chrome PATH | --connect http://127.0.0.1:9222]
//   node tools/headless.mjs bench [--only gemm,scan]
//   node tools/headless.mjs eval  --url http://localhost:8787/tests/lab.html --expr 'await SI.init().then(g => g.caps)'
//   node tools/headless.mjs info  (adapter, features, limits)
//   node tools/headless.mjs lab --fn hashBench  (an experiment from tests/lab.js: fn(gpu, SI))
//   node tools/headless.mjs gpuinfo  (Chrome's GPU status: why WebGPU is or is not available)
//   node tools/headless.mjs shot --url <page> [--wait 5] [--out shot.png]
//   node tools/headless.mjs webrtc [--wait 20]   (two p2p.html peers over real WebRTC)
//
// Options: --flags "<extra chrome flags>", --timeout <s> (default 900), --out <file.json>.
// Exit code: 0 when everything passed, 1 on failures, 2 on errors.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir, platform } from 'node:os';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const mode = argv[0] && !argv[0].startsWith('--') ? argv.shift() : 'tests';
const opt = {};
const VALUED = new Set(['url', 'base', 'chrome', 'connect', 'flags', 'out', 'expr', 'only', 'timeout', 'wait', 'fn']);
for (let k = 0; k < argv.length; k++) {
  if (!argv[k].startsWith('--')) continue;
  const name = argv[k].slice(2);
  opt[name] = VALUED.has(name) ? argv[++k] : true;   // values may start with "--" (Chrome flags)
}
const base = opt.base || 'http://localhost:8787';
const url = opt.url || { tests: `${base}/tests/`, bench: `${base}/bench/?run${opt.only ? `&only=${opt.only}` : ''}`, info: `${base}/tests/lab.html`,
  eval: `${base}/tests/lab.html`, lab: `${base}/tests/lab.html`, gpuinfo: 'about:blank' }[mode];
const timeout = Number(opt.timeout || 900) * 1000;

const defaultChrome = {
  darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  linux: '/opt/google/chrome/chrome',
  win32: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
}[platform()];

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const log = (...a) => process.stdout.write(a.join(' ') + '\n');

async function launch() {
  if (opt.connect) {
    const v = await (await fetch(`${opt.connect}/json/version`)).json();
    return { ws: v.webSocketDebuggerUrl, close() {} };
  }
  const dir = mkdtempSync(join(tmpdir(), 'si-chrome-'));
  const flags = [
    '--headless=new', `--user-data-dir=${dir}`, '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check',
    '--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    ...(platform() === 'linux' ? ['--enable-features=Vulkan', '--use-vulkan', '--use-angle=vulkan', '--disable-vulkan-surface'] : []),
    // two peers in one browser: plain local addresses instead of mDNS names, which a browser
    // without the OS's local-network permission cannot resolve
    ...(mode === 'webrtc' ? ['--disable-features=WebRtcHideLocalIpsWithMdns'] : []),
    ...(opt.flags ? String(opt.flags).split(/\s+/).filter(Boolean) : []),
    'about:blank',
  ];
  const proc = spawn(opt.chrome || defaultChrome, flags, { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  proc.stderr.on('data', (d) => { stderr += d; if (opt.stderr) process.stderr.write(d); });
  const portFile = join(dir, 'DevToolsActivePort');
  for (let k = 0; k < 200 && !existsSync(portFile); k++) await sleep(100);
  if (!existsSync(portFile)) throw new Error(`Chrome did not start:\n${stderr.slice(-2000)}`);
  const [port, path] = readFileSync(portFile, 'utf8').trim().split('\n');
  return { ws: `ws://127.0.0.1:${port}${path}`, close() { proc.kill(); try { rmSync(dir, { recursive: true, force: true }); } catch { /* still in use */ } } };
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map(), handlers = [];
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? reject(new Error(m.error.message)) : resolve(m.result);
    } else for (const h of handlers) h(m);
  };
  const open = new Promise((r, j) => { ws.onopen = r; ws.onerror = () => j(new Error(`cannot connect to ${wsUrl}`)); });
  return {
    open,
    on: (h) => handlers.push(h),
    send(method, params = {}, sessionId) {
      const msg = { id: ++id, method, params };
      if (sessionId) msg.sessionId = sessionId;
      ws.send(JSON.stringify(msg));
      return new Promise((resolve, reject) => pending.set(msg.id, { resolve, reject }));
    },
    close: () => ws.close(),
  };
}

const chrome = await launch();
const cdp = connect(chrome.ws);
let code = 2;
try {
  await cdp.open;
  // one page = one target with its own session; errors and console output are collected per page
  const openPage = async (pageUrl) => {
    // a window per page: a background tab's animation frames stop, which would freeze a peer
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank', newWindow: true });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    const S = (method, params) => cdp.send(method, params, sessionId);
    const errors = [];
    cdp.on((m) => {
      if (m.sessionId !== sessionId) return;
      if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text);
      if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || opt.console)) {
        const text = m.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
        if (m.params.type === 'error') errors.push(text);
        if (opt.console) log(`[console.${m.params.type}]`, text);
      }
    });
    await S('Runtime.enable');
    await S('Page.enable');
    const evaluate = async (expression) => {
      const r = await S('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
      return r.result.value;
    };
    await S('Page.navigate', { url: pageUrl });
    // a page that throws (a module that fails to load, a syntax error) before the condition holds
    // fails the run after a short grace period instead of waiting for the full timeout
    const until = async (expr, every = 500) => {
      let firstError = 0;
      while (Date.now() - t0 < timeout) {
        if (await evaluate(expr).catch(() => false)) return true;
        if (errors.length && !firstError) firstError = Date.now();
        if (firstError && Date.now() - firstError > 5000) throw new Error(`the page failed before ${expr}: ${String(errors[0]).split('\n')[0]}`);
        await sleep(every);
      }
      throw new Error(`timed out after ${timeout / 1000} s waiting for ${expr}`);
    };
    return { S, evaluate, until, errors };
  };
  const t0 = Date.now();
  const page = mode === 'webrtc' ? null : await openPage(url);
  const { evaluate, until, errors } = page || { errors: [] };
  if (mode === 'tests') {
    await until('!!window.__results');
    let shown = 0;
    for (;;) {
      const r = await evaluate('({ log: window.__results.log.slice(), done: !!window.__results.done, passed: window.__results.passed, failed: window.__results.failed, failures: window.__results.failures })');
      for (; shown < r.log.length; shown++) log(r.log[shown]);
      if (r.done) {
        if (opt.out) writeFileSync(opt.out, JSON.stringify(r, null, 2));
        log(`\n${r.passed} passed, ${r.failed} failed in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
        for (const f of r.failures) log(`FAIL ${f.name}: ${f.error}`);
        code = r.failed ? 1 : 0;
        break;
      }
      if (Date.now() - t0 > timeout) throw new Error('timed out');
      await sleep(1000);
    }
  } else if (mode === 'bench') {
    // ?run starts the suite once the page has initialised (a click before that would be lost)
    // print the page's status line as it changes, so a slow or stuck benchmark is visible
    let last = '';
    while (!(await evaluate('!!window.__bench?.done').catch(() => false))) {
      const st = await evaluate("document.getElementById('status')?.textContent || ''").catch(() => '');
      if (st && st !== last) { log(`  [${((Date.now() - t0) / 1000).toFixed(0)} s] ${st}`); last = st; }
      if (Date.now() - t0 > timeout) throw new Error(`timed out (last status: ${last})`);
      await sleep(1000);
    }
    const rows = await evaluate('window.__bench.rows');
    const caps = await evaluate('window.__bench.caps');
    log(`adapter: ${caps.vendor} ${caps.architecture} · subgroups ${caps.subgroupMin}–${caps.subgroupMax}`);
    for (const r of rows) log(r.error ? `${r.id}: ERROR ${r.error}` :
      `${r.id.padEnd(12)} ${r.speedup.toFixed(2).padStart(7)}×  ${r.throughput.toFixed(2)} ${r.unit} (baseline ${r.baselineThroughput.toFixed(2)})  ${r.verified ? 'verified' : 'NOT VERIFIED'} — ${r.verifyNote}${r.note ? ` · ${r.note}` : ''}`);
    if (opt.out) writeFileSync(opt.out, JSON.stringify({ caps, rows }, null, 2));
    code = rows.every((r) => !r.error && r.verified) ? 0 : 1;
  } else if (mode === 'gpuinfo') {
    // Chrome's own diagnostics: GPU devices, driver, and the status of each feature (webgpu, vulkan, ...)
    await sleep(2000);
    const info = await cdp.send('SystemInfo.getInfo');
    const g = info.gpu || {};
    log(JSON.stringify({ devices: g.devices, featureStatus: g.featureStatus, driverBugWorkarounds: (g.driverBugWorkarounds || []).slice(0, 20),
      aux: Object.fromEntries(Object.entries(g.auxAttributes || {}).filter(([k]) => /vulkan|gl|angle|driver|gpu|sandbox|optimus|amd|nvidia|webgpu|dawn/i.test(k))) }, null, 2));
    code = 0;
  } else if (mode === 'shot') {
    // screenshot of any page after --wait seconds (default 5), plus the errors it raised
    await until('document.readyState === "complete"');
    await sleep(Number(opt.wait || 5) * 1000);
    const { data } = await page.S('Page.captureScreenshot', { format: 'png' });
    writeFileSync(opt.out || 'shot.png', Buffer.from(data, 'base64'));
    log(`screenshot: ${opt.out || 'shot.png'}`);
    code = errors.length ? 1 : 0;
  } else if (mode === 'webrtc') {
    // two pages of examples/p2p.html, one per peer, connected over real WebRTC data channels
    // (signalling through BroadcastChannel, which pages of one browser share)
    const room = `si-${Date.now()}`;
    const host = await openPage(`${base}/examples/p2p.html?webrtc=${room}&host`);
    const guest = await openPage(`${base}/examples/p2p.html?webrtc=${room}`);
    const probe = 'window.peers?.[0]?.session ? (() => { const s = window.peers[0].session; const both = [...s.hashes.values()].filter((h) => h.local && h.remote); ' +
      'return { tick: s.tick, confirmed: s.confirmedTick, checkpoints: both.length, equal: both.filter((h) => h.local === h.remote).length, rollbacks: s.stats.rollbacks, ping: s.stats.ping, received: s.stats.received }; })() : null';
    let a = null, b = null;
    const until2 = Date.now() + Number(opt.wait || 20) * 1000;
    while (Date.now() < until2) {
      [a, b] = [await host.evaluate(probe).catch(() => null), await guest.evaluate(probe).catch(() => null)];
      if (a && b && a.checkpoints >= 5 && b.checkpoints >= 5) break;
      await sleep(1000);
    }
    log(JSON.stringify({ host: a, guest: b }, null, 2));
    if (!a || !b) {
      const diag = 'window.rtc ? { connection: rtc.pc.connectionState, ice: rtc.pc.iceConnectionState, gathering: rtc.pc.iceGatheringState, signaling: rtc.pc.signalingState, remote: rtc.remote, channels: [rtc.channels.unreliable.dc?.readyState, rtc.channels.reliable.dc?.readyState] } : "no RTCPeer yet"';
      log('connection state:', JSON.stringify({ host: await host.evaluate(diag).catch((e) => e.message), guest: await guest.evaluate(diag).catch((e) => e.message) }));
    }
    errors.push(...host.errors, ...guest.errors);
    code = a && b && a.received > 0 && a.checkpoints >= 3 && a.equal === a.checkpoints && b.equal === b.checkpoints ? 0 : 1;
  } else {
    await until('document.readyState === "complete"');
    if (mode === 'lab' || url.includes('/tests/lab.html')) await until('window.ready === true');   // lab.html sets it once WebGPU is up
    const expr = mode === 'info'
      ? `(async () => { const SI = await import('/src/index.js'); const g = await SI.init(); const a = g.adapter.info || {};
          return { userAgent: navigator.userAgent, adapter: { vendor: a.vendor, architecture: a.architecture, device: a.device, description: a.description },
            caps: g.caps, features: [...g.device.features], limits: { maxComputeWorkgroupStorageSize: g.limits.maxComputeWorkgroupStorageSize,
            maxStorageBuffersPerShaderStage: g.limits.maxStorageBuffersPerShaderStage, maxBufferSize: g.limits.maxBufferSize,
            maxComputeInvocationsPerWorkgroup: g.limits.maxComputeInvocationsPerWorkgroup } }; })()`
      : mode === 'lab'
        ? `(async () => { const L = await import('/tests/lab.js'); return L[${JSON.stringify(opt.fn || '')}](window.gpu, window.SI); })()`
        : `(async () => { ${/\breturn\b/.test(opt.expr) ? opt.expr : `return (${opt.expr});`} })()`;
    const v = await evaluate(expr);
    log(JSON.stringify(v, null, 2));
    if (opt.out) writeFileSync(opt.out, JSON.stringify(v, null, 2));
    code = 0;
  }
  if (errors.length) { log(`\n${errors.length} page error(s):`); for (const e of errors.slice(0, 20)) log('  ' + String(e).split('\n').slice(0, 3).join(' | ')); }
} catch (e) {
  log(`error: ${e.message}`);
  code = 2;
} finally {
  cdp.close();
  chrome.close();
}
process.exit(code);
