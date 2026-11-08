// Zero-dependency bundler for ShaderIntel's own sources.
//   node tools/build.mjs
// Produces
//   dist/shaderintel.js         one ES module (import * as SI from './shaderintel.js')
//   dist/shaderintel.global.js  classic script that defines window.SI (works from file://)
// Each source module is wrapped in its own function scope (so private helpers with
// the same name in different files never collide) and returns its export object.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';

const root = resolve(new URL('..', import.meta.url).pathname);
const entry = resolve(root, 'src/index.js');
const modules = new Map();
const order = [];

const IMPORT_FROM = /^import\s+(\{[^}]*\}|\*\s+as\s+[\w$]+|[\w$]+)\s+from\s+['"]([^'"]+)['"];?[ \t]*$/gm;
const IMPORT_BARE = /^import\s+['"]([^'"]+)['"];?[ \t]*$/gm;
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
  src = src.replace(EXPORT_DECL, (_, kind, name) => { mod.exports.push(name); return `${kind} ${name}`; });
  mod.body = `${head.join('\n')}\n${src}`;
  order.push(mod); // post-order: dependencies are pushed first
  return mod;
}

const top = load(entry);
const wrapped = order.map((m) =>
  `// ---- ${relative(root, m.path)}\nconst ${m.id} = (() => {\n${m.body.trim()}\nreturn { ${[...new Set(m.exports)].join(', ')} };\n})();`).join('\n\n');
const banner = `// ShaderIntel — WebGPU physics + shader library. Single-file build of src/ (${order.length} modules).\n`;
const names = [...new Set(top.exports.map((e) => e.split(':')[0].trim()))];

mkdirSync(resolve(root, 'dist'), { recursive: true });
writeFileSync(resolve(root, 'dist/shaderintel.js'), `${banner}${wrapped}\n\nexport const { ${names.join(', ')} } = ${top.id};\n`);
writeFileSync(resolve(root, 'dist/shaderintel.global.js'),
  `${banner}(function (global) {\n'use strict';\n${wrapped}\n\nglobal.SI = ${top.id};\n})(typeof globalThis !== 'undefined' ? globalThis : window);\n`);
const kb = (f) => (readFileSync(resolve(root, f)).length / 1024).toFixed(0);
console.log(`bundled ${order.length} modules -> dist/shaderintel.js (${kb('dist/shaderintel.js')} KB), dist/shaderintel.global.js (${kb('dist/shaderintel.global.js')} KB)`);
