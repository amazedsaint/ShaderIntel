// Example browser: builds the list from examples/files.json, runs the selected example in the
// iframe (the URL hash names it, so links like examples/#coupling work), filters the list and
// shows each example's source.
import { highlight } from './highlight.js';
import { setupMenu } from './chrome.js';

setupMenu();

const $ = (id) => document.getElementById(id);
const list = $('list'), frame = $('frame'), filter = $('filter');
const side = $('side');
let examples = [];
let current = null;

const norm = (s) => s.toLowerCase();

async function main() {
  const { categories } = await (await fetch('files.json')).json();
  examples = categories.flatMap((c) => c.examples.map((e) => ({ ...e, category: c.name })));
  render(categories);
  filter.addEventListener('input', () => render(categories));
  addEventListener('hashchange', () => open(location.hash.slice(1)));
  open(location.hash.slice(1) || examples[0].id, { replace: true });
}

function render(categories) {
  const q = norm(filter.value.trim());
  const match = (e, c) => !q || [e.title, e.description, e.id, c.name, ...(e.tags || [])].some((s) => norm(s).includes(q));
  list.innerHTML = '';
  let shown = 0;
  for (const c of categories) {
    const items = c.examples.filter((e) => match(e, c));
    if (!items.length) continue;
    shown += items.length;
    const section = document.createElement('section');
    const h = document.createElement('h2');
    h.textContent = c.name;
    const ul = document.createElement('ul');
    for (const e of items) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.className = 'item';
      a.href = `#${e.id}`;
      a.dataset.id = e.id;
      if (current && current.id === e.id) a.setAttribute('aria-current', 'page');
      const img = document.createElement('img');
      img.src = `screenshots/${e.id}.jpg`;
      img.alt = '';
      img.loading = 'lazy';
      img.width = 84; img.height = 52;
      const text = document.createElement('span');
      const b = document.createElement('b');
      b.textContent = e.title;
      const small = document.createElement('small');
      small.textContent = e.description;
      text.append(b, small);
      a.append(img, text);
      a.addEventListener('click', () => { if (innerWidth <= 860) closeMenu(); });
      li.appendChild(a);
      ul.appendChild(li);
    }
    section.append(h, ul);
    list.appendChild(section);
  }
  if (!shown) {
    const p = document.createElement('p');
    p.className = 'none';
    p.textContent = `No example matches “${filter.value.trim()}”. Try a simpler word, such as fluid or react.`;
    list.appendChild(p);
  }
}

function closeMenu() {
  side.classList.remove('open');
  const btn = side.querySelector('.menu-btn');
  btn.setAttribute('aria-expanded', 'false');
  btn.textContent = 'Menu';
}

function open(id, { replace = false } = {}) {
  const e = examples.find((x) => x.id === id) || examples[0];
  if (current === e) return;
  current = e;
  if (replace && location.hash.slice(1) !== e.id) history.replaceState(null, '', `#${e.id}`);
  frame.src = e.path;
  frame.title = e.title;
  $('title').textContent = e.title;
  $('desc').textContent = e.description;
  $('open-tab').href = e.path;
  document.title = `${e.title} — ShaderIntel examples`;
  for (const a of list.querySelectorAll('.item')) {
    if (a.dataset.id === e.id) { a.setAttribute('aria-current', 'page'); a.scrollIntoView({ block: 'nearest' }); }
    else a.removeAttribute('aria-current');
  }
  if (!$('source').hidden) showSource();
}

// ---------------------------------------------------------------- source drawer
const drawer = $('source'), toggle = $('show-source');
async function showSource() {
  const path = current.source || current.path;
  $('source-path').textContent = `examples/${path}`.replace(/examples\/\.\.\//, '');
  const pre = $('source-code');
  pre.textContent = 'Loading…';
  drawer.hidden = false;
  toggle.setAttribute('aria-expanded', 'true');
  toggle.textContent = 'Hide source';
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const text = await res.text();
    pre.innerHTML = highlight(text, path.endsWith('.html') ? 'html' : 'jsx');
    pre.scrollTop = 0;
  } catch (err) {
    pre.textContent = `The source could not be loaded (${err.message}). Open the example in a new tab and use the browser's view-source.`;
  }
}
function hideSource() {
  drawer.hidden = true;
  toggle.setAttribute('aria-expanded', 'false');
  toggle.textContent = 'View source';
  toggle.focus();
}
toggle.addEventListener('click', () => (drawer.hidden ? showSource() : hideSource()));
$('close-source').addEventListener('click', hideSource);
addEventListener('keydown', (e) => { if (e.key === 'Escape' && !drawer.hidden) hideSource(); });
drawer.querySelector('.copy-src').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  try { await navigator.clipboard.writeText($('source-code').textContent); btn.textContent = 'Copied'; }
  catch { btn.textContent = 'Select and copy'; }
  setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
});

main().catch((err) => {
  list.innerHTML = '';
  const p = document.createElement('p');
  p.className = 'none';
  p.textContent = `The example list could not be loaded (${err.message}). Serve the repository over http, e.g. node tools/serve.mjs.`;
  list.appendChild(p);
});
