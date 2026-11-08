// Docs viewer: ?page=guide renders docs/GUIDE.md (and so on) with a table of contents. Links
// between the Markdown files become links between pages; links to source files go to GitHub.
import { render } from './markdown.js';
import { setupMenu } from './chrome.js';

setupMenu();

const REPO = 'https://github.com/amazedsaint/ShaderIntel/blob/main/';
const PAGES = [
  { id: 'guide', title: 'Guide', file: 'GUIDE.md', blurb: 'The API and the math behind each solver' },
  { id: 'react', title: 'React', file: '../packages/react/README.md', blurb: 'Components and hooks' },
  { id: 'laws', title: 'Laws', file: '../LAWS.md', blurb: 'What the library promises, and how it is checked' },
  { id: 'learnings', title: 'Learnings', file: 'LEARNINGS.md', blurb: 'What was measured, and what did not work' },
  { id: 'changelog', title: 'Changelog', file: '../CHANGELOG.md', blurb: 'What changed in each release' },
];
const root = new URL('../', location.href);
const params = new URLSearchParams(location.search);
const page = PAGES.find((p) => p.id === params.get('page')) || PAGES[0];

const pages = document.getElementById('pages');
for (const p of PAGES) {
  const li = document.createElement('li');
  const a = document.createElement('a');
  a.href = `?page=${p.id}`;
  a.innerHTML = `${p.title}<small></small>`;
  a.querySelector('small').textContent = p.blurb;
  if (p === page) a.setAttribute('aria-current', 'page');
  li.appendChild(a);
  pages.appendChild(li);
}

function rewrite(href, base) {
  if (/^(https?:|mailto:|#)/.test(href)) return href;
  const url = new URL(href, base);
  const path = url.pathname.slice(root.pathname.length);
  const target = PAGES.find((p) => new URL(p.file, location.href).pathname === url.pathname);
  if (target) return `?page=${target.id}${url.hash}`;
  if (/\.(html)$/.test(path) || path.endsWith('/')) return url.href;   // pages of this site
  return REPO + path + url.hash;                                    // source files: on GitHub
}

async function show() {
  const doc = document.getElementById('doc');
  const base = new URL(page.file, location.href);
  try {
    const res = await fetch(base);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const { html, headings } = render(await res.text(), { link: (h) => rewrite(h, base) });
    doc.innerHTML = html;
    document.title = `${headings.find((h) => h.level === 1)?.text || page.title} — ShaderIntel`;
    const toc = document.getElementById('toc');
    for (const h of headings.filter((x) => x.level === 2 || x.level === 3)) {
      const a = document.createElement('a');
      a.href = `#${h.id}`;
      a.textContent = h.text.replace(/\*\*/g, '');
      if (h.level === 3) a.className = 'h3';
      toc.appendChild(a);
    }
    if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
    // mark the section being read in the table of contents
    const links = new Map([...toc.querySelectorAll('a')].map((a) => [a.hash.slice(1), a]));
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting && links.has(e.target.id)) {
        for (const a of links.values()) a.classList.remove('here');
        links.get(e.target.id).classList.add('here');
      }
    }, { rootMargin: '0px 0px -70% 0px' });
    for (const h of doc.querySelectorAll('h2, h3')) io.observe(h);
  } catch (err) {
    doc.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'error';
    p.textContent = `This page could not be loaded (${err.message}). Read ${page.file.replace(/^\.\.\//, '')} on GitHub instead.`;
    doc.appendChild(p);
  }
}
show();
