// A tiny React control panel for the examples (the same look as the vanilla examples' SI.panel).
import { createRoot } from 'react-dom/client';

const css = `
.rp { position: absolute; right: 12px; top: 12px; width: 240px; padding: 10px 12px 12px; z-index: 10;
  background: rgba(14,17,22,0.82); color: #d8dee9; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
  font: 12px/1.4 system-ui, -apple-system, sans-serif; backdrop-filter: blur(6px); }
.rp h3 { margin: 0 0 8px; font-size: 13px; font-weight: 600; color: #fff; }
.rp .row { display: grid; grid-template-columns: 84px 1fr 40px; align-items: center; gap: 6px; margin: 5px 0; }
.rp .row.wide { grid-template-columns: 84px 1fr; }
.rp input[type=range] { width: 100%; accent-color: #6cb6ff; }
.rp .val { text-align: right; color: #9fb1c7; font-variant-numeric: tabular-nums; }
.rp button { width: 100%; margin-top: 6px; padding: 6px; background: #22303f; color: #e6edf3; border: 1px solid #344558;
  border-radius: 5px; cursor: pointer; font: inherit; }
.rp button:hover { background: #2b3d51; }
.rp select, .rp textarea { width: 100%; background: #1b232d; color: #e6edf3; border: 1px solid #344558; border-radius: 4px; padding: 3px; font: inherit; }
.rp textarea { font: 11px/1.45 ui-monospace, Menlo, monospace; resize: vertical; }
.rp .note { color: #8b9bb0; margin-top: 8px; font-size: 11px; }
.rp .err { color: #ffb4a9; white-space: pre-wrap; font: 11px/1.4 ui-monospace, Menlo, monospace; margin-top: 6px; max-height: 160px; overflow: auto; }
.rp a { color: #6cb6ff; }
.info { position: absolute; left: 12px; bottom: 12px; max-width: 520px; padding: 8px 12px; background: rgba(10,12,16,0.72);
  border-radius: 8px; color: #b9c4d0; font: 12px/1.5 system-ui, sans-serif; pointer-events: none; z-index: 5; }
.info b { color: #fff; }
.info code { color: #ffd38a; }
`;

export function mountApp(App) {
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
  createRoot(document.getElementById('root')).render(<App />);
}

export function Panel({ title, children }) {
  return <div className="rp" onPointerDown={(e) => e.stopPropagation()}>{title && <h3>{title}</h3>}{children}</div>;
}

export function Slider({ label, value, min, max, step = 0.01, onChange, format }) {
  const dec = Math.min(6, (String(step).split('.')[1] || '').length);
  return (
    <div className="row">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      <span className="val">{format ? format(value) : (+value).toFixed(dec)}</span>
    </div>
  );
}

export function Select({ label, value, options, onChange }) {
  return (
    <div className="row wide">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((o) => <option key={o} value={o}>{o}</option>)}</select>
    </div>
  );
}

export function Toggle({ label, value, onChange }) {
  return (
    <div className="row wide">
      <label>{label}</label>
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
    </div>
  );
}

export function Button({ children, onClick }) { return <button onClick={onClick}>{children}</button>; }
export function Note({ children }) { return <div className="note">{children}</div>; }
export function Info({ children }) { return <div className="info">{children}</div>; }
