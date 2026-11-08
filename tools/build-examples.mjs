// Bundles the React examples and the React test page with esbuild, so they run from static files
// (no CDN, no dev server): JSX → ESM, React and shaderintel-react bundled into shared chunks,
// `shaderintel` itself left as a bare import that each page maps to the live sources (src/).
//   node tools/build-examples.mjs           production examples + development test bundle
//   node tools/build-examples.mjs --watch   rebuild on change
import { build, context } from 'esbuild';
import { readdirSync, rmSync, mkdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const watch = process.argv.includes('--watch');
const alias = { 'shaderintel-react': join(root, 'packages/react/src/index.ts') };

const jobs = [
  {
    name: 'react examples',
    entryPoints: readdirSync(join(root, 'examples/react/src')).filter((f) => f.endsWith('.jsx') && !f.startsWith('_')).map((f) => join(root, 'examples/react/src', f)),
    outdir: join(root, 'examples/react/build'),
    // production React: smaller and faster; StrictMode checks are exercised by the test bundle
    define: { 'process.env.NODE_ENV': '"production"' },
    minify: true,
  },
  {
    name: 'react tests',
    entryPoints: [join(root, 'packages/react/test/react.test.jsx')],
    outdir: join(root, 'packages/react/test/build'),
    define: { 'process.env.NODE_ENV': '"development"' },
    minify: false,
  },
];

for (const job of jobs) {
  rmSync(job.outdir, { recursive: true, force: true });
  mkdirSync(job.outdir, { recursive: true });
  const options = {
    entryPoints: job.entryPoints, outdir: job.outdir, bundle: true, splitting: true, format: 'esm', target: 'es2022',
    jsx: 'automatic', alias, external: ['shaderintel'], define: job.define, minify: job.minify, sourcemap: false,
    chunkNames: 'chunks/[name]-[hash]', legalComments: 'none', logLevel: 'warning',
  };
  if (watch) {
    const ctx = await context(options);
    await ctx.watch();
    console.log(`watching ${job.name}`);
  } else {
    const r = await build({ ...options, metafile: true });
    const bytes = Object.values(r.metafile.outputs).reduce((a, o) => a + o.bytes, 0);
    console.log(`${job.name}: ${job.entryPoints.length} entr${job.entryPoints.length === 1 ? 'y' : 'ies'} -> ${job.outdir.slice(root.length + 1)} (${(bytes / 1024).toFixed(0)} KB)`);
  }
}
