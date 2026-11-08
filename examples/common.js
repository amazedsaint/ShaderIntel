// Shared by the examples: show any startup or runtime error on the page instead of a blank canvas.
function show(msg) {
  let el = document.getElementById('si-error');
  if (!el) {
    el = document.createElement('div');
    el.id = 'si-error';
    Object.assign(el.style, { position: 'fixed', inset: '0', padding: '28px', background: 'rgba(16,4,4,0.92)', color: '#ffb4a9',
      font: '13px/1.5 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap', overflow: 'auto', zIndex: 100 });
    document.body.appendChild(el);
  }
  el.textContent = String(msg);
}
window.addEventListener('error', (e) => show(e.error?.stack || e.message));
window.addEventListener('unhandledrejection', (e) => show(e.reason?.stack || e.reason?.message || e.reason));
if (!navigator.gpu) show('This browser does not expose WebGPU.\nUse Chrome / Edge 113+, Safari 26+ or Firefox 141+ (Windows), from http://localhost or https.');

// Opened on its own (not inside the example browser): offer a way back to the list.
if (window.top === window) {
  addEventListener('DOMContentLoaded', () => {
    const file = location.pathname.split('/').pop().replace(/\.html$/, '') || 'index';
    const inReact = location.pathname.includes('/examples/react/');
    const a = document.createElement('a');
    a.className = 'si-back';
    a.href = `${inReact ? '../' : './'}#${inReact ? 'react-' : ''}${file}`;
    a.textContent = 'All examples';
    Object.assign(a.style, { position: 'fixed', right: '12px', bottom: '12px', zIndex: 50, padding: '6px 11px', borderRadius: '6px',
      background: 'rgba(14,17,22,0.82)', color: '#d8dee9', font: '12px/1.2 system-ui, sans-serif', textDecoration: 'none',
      border: '1px solid rgba(255,255,255,0.1)' });
    document.body.appendChild(a);
  });
}
