// WGSL module system. WGSL has no #include, so ShaderIntel ships its shader
// library as named modules that a tiny linker stitches together:
//   - dependencies are resolved transitively and each module is emitted once;
//   - `enable` / `requires` / `diagnostic` directives are hoisted to the top
//     (WGSL requires directives before any declaration);
//   - modules may be functions of the device capabilities, so one name can
//     resolve to a subgroup-accelerated or a portable body.
// WGSL allows module-scope declarations in any order, so no topological sort
// of the bodies is needed — only de-duplication.
import { SIError } from './errors.js';

const registry = new Map();

/**
 * Register a WGSL module.
 * @param {string} name   e.g. 'noise'
 * @param {object} def    { deps?: string[], code: string | (caps) => string, doc?: string }
 */
export function defineModule(name, def) {
  if (typeof def === 'string') def = { code: def };
  registry.set(name, { name, deps: def.deps || [], code: def.code, doc: def.doc || '' });
}

export const hasModule = (name) => registry.has(name);
export const moduleNames = () => [...registry.keys()];
export const moduleInfo = (name) => registry.get(name);

const IMPORT_RE = /^[ \t]*(?:\/\/)?[ \t]*#import[ \t]+([^\n]+)$/gm;
const DIRECTIVE_RE = /^[ \t]*(enable|requires)[ \t]+([^;]+);[ \t]*$/gm;
const DIAG_RE = /^[ \t]*diagnostic[ \t]*\([^)]*\)[ \t]*;[ \t]*$/gm;

/** Pull `#import a, b` lines out of user code. */
export function parseImports(code) {
  const names = [];
  const stripped = code.replace(IMPORT_RE, (_, list) => {
    for (const n of list.split(/[\s,]+/)) if (n) names.push(n);
    return '';
  });
  return { code: stripped, imports: names };
}

function extractDirectives(code, enables, requires, diags) {
  code = code.replace(DIRECTIVE_RE, (_, kind, list) => {
    const set = kind === 'enable' ? enables : requires;
    for (const n of list.split(',')) if (n.trim()) set.add(n.trim());
    return '';
  });
  return code.replace(DIAG_RE, (d) => { diags.add(d.trim()); return ''; });
}

/**
 * Link user code with library modules.
 * @returns {string} complete WGSL source
 */
export function link(userCode, imports = [], caps = {}) {
  const parsed = parseImports(userCode || '');
  const order = [];
  const seen = new Set();
  const visit = (name, from) => {
    if (seen.has(name)) return;
    const mod = registry.get(name);
    if (!mod) throw new SIError(`ShaderIntel: unknown WGSL module "${name}"${from ? ` (imported by ${from})` : ''}. Known: ${moduleNames().join(', ')}`);
    seen.add(name);
    for (const d of mod.deps) visit(d, name);
    order.push(mod);
  };
  for (const n of [...imports, ...parsed.imports]) visit(n);

  const enables = new Set(), requires = new Set(), diags = new Set();
  const bodies = order.map((m) => {
    const src = typeof m.code === 'function' ? m.code(caps) : m.code;
    return `// ---- module: ${m.name}\n` + extractDirectives(src, enables, requires, diags).trim();
  });
  const user = extractDirectives(parsed.code, enables, requires, diags);
  const head = [
    ...[...enables].map((e) => `enable ${e};`),
    ...[...requires].map((r) => `requires ${r};`),
    ...diags,
  ].join('\n');
  return `${head}\n${bodies.join('\n\n')}\n\n${user}`.trim() + '\n';
}

/** Format a JS number as a WGSL literal of the requested type. */
export function literal(value, type = 'f32') {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (type === 'u32') return `${Math.max(0, Math.round(value)) >>> 0}u`;
  if (type === 'i32') return `${Math.round(value) | 0}i`;
  if (!Number.isFinite(value)) throw new SIError(`ShaderIntel: non-finite WGSL literal ${value}`);
  const s = String(value);
  return /[.eE]/.test(s) ? s : s + '.0';
}

/** Turn { NAME: value } into `const NAME = value;` lines (type inferred: integers -> i32 unless suffixed). */
export function constants(defs = {}) {
  return Object.entries(defs).map(([k, v]) => {
    if (typeof v === 'string') return `const ${k} = ${v};`;
    if (typeof v === 'boolean') return `const ${k} = ${v};`;
    if (Array.isArray(v)) return `const ${k} = vec${v.length}f(${v.map((x) => literal(x)).join(', ')});`;
    return `const ${k} = ${Number.isInteger(v) ? `${v}` : literal(v)};`;
  }).join('\n');
}
