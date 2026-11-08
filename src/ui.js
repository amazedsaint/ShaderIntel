// Tiny zero-dependency control panel for examples.
//   const ui = panel('Fluid', [
//     { label: 'substeps', min: 1, max: 16, step: 1, value: 8, onInput: (v) => (sim.substeps = v) },
//     { label: 'reset', onClick: reset },
//     { label: 'material', options: ['water', 'jelly'], value: 'water', onChange: (v) => ... },
//     { label: 'show grid', checked: true, onChange: (v) => ... },
//   ]);

const CSS = `
.si-panel { position: absolute; right: 12px; top: 12px; width: 236px; padding: 10px 12px 12px; z-index: 10;
  background: rgba(14,17,22,0.82); color: #d8dee9; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
  font: 12px/1.4 system-ui, -apple-system, sans-serif; backdrop-filter: blur(6px); }
.si-panel h3 { margin: 0 0 8px; font-size: 13px; font-weight: 600; color: #fff; }
.si-panel .row { display: grid; grid-template-columns: 82px 1fr 38px; align-items: center; gap: 6px; margin: 5px 0; }
.si-panel .row.wide { grid-template-columns: 82px 1fr; }
.si-panel input[type=range] { width: 100%; accent-color: #6cb6ff; }
.si-panel .val { text-align: right; color: #9fb1c7; font-variant-numeric: tabular-nums; }
.si-panel button { width: 100%; margin-top: 6px; padding: 6px; background: #22303f; color: #e6edf3; border: 1px solid #344558;
  border-radius: 5px; cursor: pointer; font: inherit; }
.si-panel button:hover { background: #2b3d51; }
.si-panel select { width: 100%; background: #1b232d; color: #e6edf3; border: 1px solid #344558; border-radius: 4px; padding: 3px; font: inherit; }
.si-panel .note { color: #8b9bb0; margin-top: 8px; font-size: 11px; }
.si-panel a { color: #6cb6ff; }
`;

let styled = false;

export function panel(title, controls = [], parent = document.body) {
  if (!styled) { const s = document.createElement('style'); s.textContent = CSS; document.head.appendChild(s); styled = true; }
  const el = document.createElement('div');
  el.className = 'si-panel';
  if (title) { const h = document.createElement('h3'); h.textContent = title; el.appendChild(h); }
  const api = { el, values: {} };
  for (const c of controls) {
    if (c.onClick) {
      const b = document.createElement('button');
      b.textContent = c.label;
      b.addEventListener('click', () => c.onClick());
      el.appendChild(b);
      continue;
    }
    if (c.html !== undefined) {
      const d = document.createElement('div'); d.className = 'note'; d.innerHTML = c.html; el.appendChild(d);
      continue;
    }
    const row = document.createElement('div');
    row.className = 'row';
    const lab = document.createElement('label');
    lab.textContent = c.label;
    row.appendChild(lab);
    if (c.options) {
      row.classList.add('wide');
      const sel = document.createElement('select');
      for (const o of c.options) { const opt = document.createElement('option'); opt.value = opt.textContent = o; sel.appendChild(opt); }
      sel.value = c.value ?? c.options[0];
      api.values[c.label] = sel.value;
      sel.addEventListener('change', () => { api.values[c.label] = sel.value; c.onChange?.(sel.value); });
      row.appendChild(sel);
    } else if (c.checked !== undefined || typeof c.value === 'boolean') {
      row.classList.add('wide');
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = c.checked ?? c.value;
      api.values[c.label] = cb.checked;
      cb.addEventListener('change', () => { api.values[c.label] = cb.checked; c.onChange?.(cb.checked); });
      row.appendChild(cb);
    } else {
      const r = document.createElement('input');
      r.type = 'range'; r.min = c.min ?? 0; r.max = c.max ?? 1; r.step = c.step ?? 0.01; r.value = c.value ?? 0;
      const val = document.createElement('span');
      val.className = 'val';
      // as many decimals as the step has (0.25 → 2, 2.5 → 1, 5 → 0)
      const stepStr = String(c.step ?? 0.01);
      const dec = /e-(\d+)$/.test(stepStr) ? Math.min(6, +RegExp.$1) : Math.min(6, (stepStr.split('.')[1] || '').length);
      const fmt = (v) => (+v).toFixed(dec);
      val.textContent = fmt(r.value);
      api.values[c.label] = +r.value;
      r.addEventListener('input', () => { val.textContent = fmt(r.value); api.values[c.label] = +r.value; c.onInput?.(+r.value); });
      row.appendChild(r); row.appendChild(val);
    }
    el.appendChild(row);
  }
  parent.appendChild(el);
  return api;
}
