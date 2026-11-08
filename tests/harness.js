// Minimal in-browser test runner. Results land in window.__results so an
// automated driver can read them; the page shows a readable log as well.
const log = [];
const results = { passed: 0, failed: 0, done: false, failures: [], log };
window.__results = results;

let out;
function print(line, cls = '') {
  log.push(line);
  if (!out) out = document.getElementById('log');
  if (out) {
    const div = document.createElement('div');
    div.textContent = line;
    if (cls) div.className = cls;
    out.appendChild(div);
  }
}

// ?grep=text runs only the tests whose name contains it (with ?only=suite to skip whole suites)
const grep = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('grep') : null;

export async function test(name, fn) {
  if (grep && !name.toLowerCase().includes(grep.toLowerCase())) return;
  const t0 = performance.now();
  try {
    await fn();
    results.passed++;
    print(`PASS ${name} (${(performance.now() - t0).toFixed(1)} ms)`, 'pass');
  } catch (e) {
    results.failed++;
    results.failures.push({ name, error: String(e && e.message || e) });
    print(`FAIL ${name}: ${e && e.message || e}`, 'fail');
    console.error(e);
  }
}

export function section(name) { print(`— ${name}`, 'section'); }
export function note(msg) { print(`  ${msg}`, 'note'); }

export function finish() {
  results.done = true;
  print(`${results.passed} passed, ${results.failed} failed`, results.failed ? 'fail' : 'pass');
}

export function assert(cond, msg = 'assertion failed') { if (!cond) throw new Error(msg); }

export function assertClose(a, b, tol = 1e-5, msg = '') {
  const d = Math.abs(a - b);
  if (!(d <= tol * Math.max(1, Math.abs(a), Math.abs(b)))) throw new Error(`${msg} expected ${b}, got ${a} (|Δ|=${d})`);
}

export function assertArrayEq(got, want, n = want.length, msg = '') {
  for (let k = 0; k < n; k++) {
    if (got[k] !== want[k]) throw new Error(`${msg} mismatch at ${k}: got ${got[k]}, want ${want[k]}`);
  }
}

/** Deterministic PRNG (mulberry32) so failures are reproducible. */
export function rng(seed = 1) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
