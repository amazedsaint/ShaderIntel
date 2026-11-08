// Repository health: every module parses, dist/ matches a fresh build, the examples registry is
// consistent with the files, and the TypeScript declarations cover the runtime exports.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync, readdirSync, existsSync, mkdtempSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const walk = (dir) => readdirSync(dir).flatMap((f) => { const p = join(dir, f); return statSync(p).isDirectory() ? walk(p) : [p]; });

test('every JavaScript module parses', () => {
  const files = [...walk(join(root, 'src')), ...walk(join(root, 'tools')).filter((f) => /\.m?js$/.test(f)), ...walk(join(root, 'site')).filter((f) => f.endsWith('.js')),
    ...walk(join(root, 'tests')).filter((f) => /\.m?js$/.test(f)), ...walk(join(root, 'bench')).filter((f) => f.endsWith('.js'))];
  for (const f of files) {
    const r = spawnSync(process.execPath, ['--check', f], { encoding: 'utf8' });
    assert.equal(r.status, 0, `${f}\n${r.stderr}`);
  }
});

test('dist/ is what tools/build.mjs produces from src/ (rebuild and commit it with your change)', () => {
  const out = mkdtempSync(join(tmpdir(), 'si-dist-'));
  const r = spawnSync(process.execPath, [join(root, 'tools/build.mjs'), '--out', out], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  for (const f of ['shaderintel.js', 'shaderintel.global.js'])
    assert.ok(readFileSync(join(out, f)).equals(readFileSync(join(root, 'dist', f))), `dist/${f} is stale: run npm run build:core`);
});

test('the bundle exports exactly what src/index.js exports', async () => {
  const src = await import(join(root, 'src/index.js'));
  const dist = await import(join(root, 'dist/shaderintel.js'));
  assert.deepEqual(Object.keys(dist).sort(), Object.keys(src).sort());
});

test('every runtime export is declared in types/shaderintel.d.ts', async () => {
  const src = await import(join(root, 'src/index.js'));
  const dts = readFileSync(join(root, 'types/shaderintel.d.ts'), 'utf8');
  const missing = Object.keys(src).filter((name) => !new RegExp(`export (?:declare )?(?:class|function|const|namespace|interface|type) ${name}\\b`).test(dts));
  assert.deepEqual(missing, [], `not declared in types/shaderintel.d.ts: ${missing.join(', ')}`);
});

test('examples/files.json points at real pages, sources and screenshots, with unique ids', () => {
  const { categories } = JSON.parse(readFileSync(join(root, 'examples/files.json'), 'utf8'));
  const ids = new Set();
  for (const c of categories) for (const e of c.examples) {
    assert.ok(!ids.has(e.id), `duplicate id ${e.id}`);
    ids.add(e.id);
    assert.ok(existsSync(join(root, 'examples', e.path)), `${e.id}: ${e.path} missing`);
    if (e.source) assert.ok(existsSync(join(root, 'examples', e.source)), `${e.id}: ${e.source} missing`);
    assert.ok(existsSync(join(root, 'examples/screenshots', `${e.id}.jpg`)), `${e.id}: screenshot missing (npm run thumbnails)`);
    assert.ok(e.title && e.description, `${e.id}: title and description`);
  }
});
