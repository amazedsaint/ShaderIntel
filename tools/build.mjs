// Zero-dependency bundler for ShaderIntel's own sources.
//   node tools/build.mjs
// Produces
//   dist/shaderintel.js         one ES module (import * as SI from './shaderintel.js')
//   dist/shaderintel.global.js  classic script that defines window.SI (works from file://)
// Each source module is wrapped in its own function scope (so private helpers with
// the same name in different files never collide) and returns its export object.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));   // (URL.pathname would keep %20 for spaces)
const entry = resolve(root, 'src/index.js');
const modules = new Map();
const order = [];

const TRAIL = String.raw`;?[ \t]*(?:\/\/[^\n]*)?$`;   // optional semicolon and trailing comment
const IMPORT_FROM = new RegExp(String.raw`^import\s+(\{[^}]*\}|\*\s+as\s+[\w$]+|[\w$]+)\s+from\s+['"]([^'"]+)['"]` + TRAIL, 'gm');
const IMPORT_BARE = new RegExp(String.raw`^import\s+['"]([^'"]+)['"]` + TRAIL, 'gm');
const EXPORT_FROM = /^export\s+\{([^}]*)\}\s+from\s+['"]([^'"]+)['"];?[ \t]*$/gm;
const EXPORT_LIST = /^export\s+\{([^}]*)\};?[ \t]*$/gm;
const EXPORT_DECL = /^export\s+(async\s+function|function|class|const|let|var)\s+([A-Za-z_$][\w$]*)/gm;

function specList(list) {
  return list.split(',').map((s) => s.trim()).filter(Boolean).map((s) => {
    const m = /^([\w$]+)(?:\s+as\s+([\w$]+))?$/.exec(s);
    if (!m) throw new Error(`cannot parse specifier "${s}"`);
    return { local: m[1], exported: m[2] || m[1] };
  });
}

function load(path) {
  if (modules.has(path)) return modules.get(path);
  const mod = { id: `__m${modules.size}`, path, exports: [], body: '' };
  modules.set(path, mod);
  let src = readFileSync(path, 'utf8');
  const dep = (spec) => load(resolve(dirname(path), spec));
  const head = [];
  const declared = new Map(); // a name imported twice (e.g. re-exported and imported) is declared once
  const bind = (name, expr) => {
    if (declared.has(name)) { if (declared.get(name) !== expr) throw new Error(`${relative(root, path)}: conflicting bindings for ${name}`); return; }
    declared.set(name, expr);
    head.push(`const ${name} = ${expr};`);
  };
  src = src.replace(EXPORT_FROM, (_, list, spec) => {
    const d = dep(spec);
    for (const { local, exported } of specList(list)) {
      bind(exported, `${d.id}.${local}`);
      mod.exports.push(exported);
    }
    return '';
  });
  src = src.replace(IMPORT_FROM, (_, what, spec) => {
    const d = dep(spec);
    what = what.trim();
    if (what.startsWith('*')) { bind(what.split(/\s+as\s+/)[1].trim(), d.id); return ''; }
    const inner = /^\{([\s\S]*)\}$/.exec(what);
    if (!inner) throw new Error(`${relative(root, path)}: default imports are not supported (${what})`);
    for (const { local, exported } of specList(inner[1])) bind(exported, `${d.id}.${local}`);
    return '';
  });
  src = src.replace(IMPORT_BARE, (_, spec) => { dep(spec); return ''; });
  src = src.replace(EXPORT_LIST, (_, list) => {
    for (const { local, exported } of specList(list)) mod.exports.push(exported === local ? local : `${exported}: ${local}`);
    return '';
  });
  // forms this bundler does not understand must fail the build, not ship a broken bundle
  const multi = /^export\s+(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=[^;\n]*,\s*[A-Za-z_$][\w$]*\s*=/m.exec(src);
  if (multi) throw new Error(`${relative(root, path)}: one export per declaration, please ("${multi[0].slice(0, 60)}…")`);
  src = src.replace(EXPORT_DECL, (_, kind, name) => { mod.exports.push(name); return `${kind} ${name}`; });
  const left = /^(?:export|import)\s[^\n]*/m.exec(src.replace(/`(?:\\.|[^`\\])*`/g, '``'));
  if (left) throw new Error(`${relative(root, path)}: unsupported module syntax for tools/build.mjs: "${left[0].slice(0, 80)}"`);
  mod.body = `${head.join('\n')}\n${src}`;
  order.push(mod); // post-order: dependencies are pushed first
  return mod;
}

const top = load(entry);
const wrapped = order.map((m) =>
  `// ---- ${relative(root, m.path)}\nconst ${m.id} = (() => {\n${m.body.trim()}\nreturn { ${[...new Set(m.exports)].join(', ')} };\n})();`).join('\n\n');
const banner = `// ShaderIntel — WebGPU physics + shader library. Single-file build of src/ (${order.length} modules).\n`;
const names = [...new Set(top.exports.map((e) => e.split(':')[0].trim()))];

// --out <dir> writes somewhere else (the unit tests build into a temporary folder and compare with dist/)
const outArg = process.argv.indexOf('--out');
const out = outArg > 0 ? resolve(process.argv[outArg + 1]) : resolve(root, 'dist');
mkdirSync(out, { recursive: true });
writeFileSync(resolve(out, 'shaderintel.js'), `${banner}${wrapped}\n\nexport const { ${names.join(', ')} } = ${top.id};\n`);
writeFileSync(resolve(out, 'shaderintel.global.js'),
  `${banner}(function (global) {\n'use strict';\n${wrapped}\n\nglobal.SI = ${top.id};\n})(typeof globalThis !== 'undefined' ? globalThis : window);\n`);
const kb = (f) => (readFileSync(resolve(out, f)).length / 1024).toFixed(0);
console.log(`bundled ${order.length} modules -> ${relative(root, out) || '.'}/shaderintel.js (${kb('shaderintel.js')} KB), shaderintel.global.js (${kb('shaderintel.global.js')} KB)`);
