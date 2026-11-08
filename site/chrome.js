// Shared page chrome: the narrow-screen menu and copy buttons on code blocks.

/** On narrow screens the sidebar collapses to its header; the Menu button opens it. */
export function setupMenu() {
  const side = document.getElementById('side');
  const btn = side?.querySelector('.menu-btn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const open = side.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(open));
    btn.textContent = open ? 'Close' : 'Menu';
  });
}

/** Every .copy button copies the <pre> of its .code block. */
export function setupCopyButtons(root = document) {
  for (const btn of root.querySelectorAll('.code .copy')) {
    btn.addEventListener('click', async () => {
      const pre = btn.closest('.code').querySelector('pre');
      try {
        await navigator.clipboard.writeText(pre.textContent);
        btn.textContent = 'Copied';
      } catch {
        btn.textContent = 'Select and copy';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
    });
  }
}
