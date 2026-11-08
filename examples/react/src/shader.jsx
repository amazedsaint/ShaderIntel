// shaderintel-react: a live SDF shader playground. Edit map(), press Apply: the new code is compiled
// off-screen first and only replaces the scene if it compiles, so a typo never breaks the frame.
import { useState } from 'react';
import { Canvas, SdfScene, useGPU } from 'shaderintel-react';
import { mountApp, Panel, Slider, Toggle, Button, Note, Info } from './_ui.jsx';

const PRESETS = {
  'melting blobs': /* wgsl */`fn map(p: vec3f) -> vec2f {
  let t = u.time;
  var d = vec2f(p.y, 0.0);                                   // ground, material 0
  let a = vec3f(sin(t * 0.7) * 0.9, 1.0 + 0.3 * sin(t * 1.3), 0.0);
  let b = vec3f(-0.8, 0.9 + 0.25 * cos(t), 0.3);
  var blob = smin(sdSphere(p - a, 0.55), sdSphere(p - b, 0.5), u.blend);
  blob += u.bump * simplex3(p * 4.0 + vec3f(0.0, t * 0.5, 0.0));
  d = opU(d, vec2f(blob / (1.0 + 11.0 * u.bump), 1.0));     // stay 1-Lipschitz
  let q = opTwist(p - vec3f(2.2, 1.2, -0.6), u.twist);
  d = opU(d, vec2f(sdRoundBox(q, vec3f(0.35, 1.2, 0.35), 0.06), 2.0));
  return d;
}`,
  'pillars': /* wgsl */`fn map(p: vec3f) -> vec2f {
  var d = vec2f(p.y, 0.0);
  let r = opRepeatLimited(p - vec3f(0.0, 1.0, 0.0), 1.1, vec3f(3.0, 0.0, 3.0));
  let q = opTwist(r, u.twist * 0.5);
  d = opU(d, vec2f(sdRoundBox(q, vec3f(0.22, 1.0, 0.22), 0.04 + u.blend * 0.1), 3.0));
  d = opU(d, vec2f(sdSphere(p - vec3f(0.0, 2.6 + 0.2 * sin(u.time), 0.0), 0.45 + u.bump * 4.0), 5.0));
  return d;
}`,
  'torus knot-ish': /* wgsl */`fn map(p: vec3f) -> vec2f {
  var d = vec2f(p.y + 0.2, 0.0);
  let q = rotAxis(vec3f(0.0, 1.0, 0.0), u.time * 0.3) * (p - vec3f(0.0, 1.2, 0.0));
  let tor = sdTorus(opTwist(q, u.twist).xzy, vec2f(0.9, 0.22 + u.bump * 3.0));
  d = opU(d, vec2f(smin(tor, sdSphere(q, 0.45), u.blend), 4.0));
  return d;
}`,
};
const UNIFORMS = { blend: 'f32', twist: 'f32', bump: 'f32' };
const IMPORTS = ['noise', 'math'];

// Compile candidate code with the scene's exact settings, without drawing it.
function useCompiler() {
  const gpu = useGPU();
  return async (code) => {
    const probe = gpu.sdf({ code, uniforms: UNIFORMS, imports: IMPORTS });
    try { await Promise.all([probe.program.ready(), probe.coneK.ready()]); return null; }
    catch (e) { return e.message.replace(/^ShaderIntel: /, ''); }
    finally { probe.destroy(); }
  };
}

function Editor({ code, onApply }) {
  const compile = useCompiler();
  const [text, setText] = useState(code);
  const [error, setError] = useState(null);
  const apply = async () => { const err = await compile(text); setError(err); if (!err) onApply(text); };
  return (<>
    <textarea rows={14} value={text} spellCheck={false} onChange={(e) => setText(e.target.value)} />
    <Button onClick={apply}>Apply</Button>
    {error && <div className="err">{error}</div>}
  </>);
}

function App() {
  const [preset, setPreset] = useState('melting blobs');
  const [code, setCode] = useState(PRESETS[preset]);
  const [blend, setBlend] = useState(0.35);
  const [twist, setTwist] = useState(1.2);
  const [bump, setBump] = useState(0.03);
  const [shadows, setShadows] = useState(true);
  const [ao, setAo] = useState(true);
  return (
    <Canvas camera={{ target: [0.4, 0.8, 0], distance: 6.5, pitch: 0.25, yaw: 0.6 }} background={[0.05, 0.06, 0.08, 1]} hud>
      <SdfScene code={code} imports={IMPORTS} uniforms={UNIFORMS} values={{ blend, twist, bump }} shadows={shadows} ao={ao} />
      <Panel title="Live SDF shader">
        <div className="row wide"><label>preset</label>
          <select value={preset} onChange={(e) => { setPreset(e.target.value); setCode(PRESETS[e.target.value]); }}>
            {Object.keys(PRESETS).map((k) => <option key={k}>{k}</option>)}
          </select>
        </div>
        <Slider label="blend" value={blend} min={0} max={1} step={0.01} onChange={setBlend} />
        <Slider label="twist" value={twist} min={-3} max={3} step={0.01} onChange={setTwist} />
        <Slider label="bumps" value={bump} min={0} max={0.1} step={0.005} onChange={setBump} />
        <Toggle label="soft shadows" value={shadows} onChange={setShadows} />
        <Toggle label="ambient occl." value={ao} onChange={setAo} />
        <Editor key={preset} code={code} onApply={setCode} />
        <Note>Available: the <code>sdf</code>, <code>noise</code> and <code>math</code> modules, <code>u.time</code>, <code>u.blend</code>, <code>u.twist</code>, <code>u.bump</code>. Return (distance, material id 0–5).</Note>
      </Panel>
      <Info><b>SDF playground</b> — one <code>map()</code> function, cone-marched in two passes with soft shadows, AO and PBR.
        Sliders are uniforms (no recompile); <b>Apply</b> recompiles, and keeps the old shader if the new one has errors.</Info>
    </Canvas>
  );
}

mountApp(App);
