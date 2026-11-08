import{A as C,B as x,C as i,E as m,F as A,G as T,H as P,a as u,b as I,c as w,h as y,i as g,m as S}from"./chunks/chunk-HS5P52HZ.js";var t=u(I(),1);var e=u(y(),1),f={"melting blobs":`fn map(p: vec3f) -> vec2f {
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
}`,pillars:`fn map(p: vec3f) -> vec2f {
  var d = vec2f(p.y, 0.0);
  let r = opRepeatLimited(p - vec3f(0.0, 1.0, 0.0), 1.1, vec3f(3.0, 0.0, 3.0));
  let q = opTwist(r, u.twist * 0.5);
  d = opU(d, vec2f(sdRoundBox(q, vec3f(0.22, 1.0, 0.22), 0.04 + u.blend * 0.1), 3.0));
  d = opU(d, vec2f(sdSphere(p - vec3f(0.0, 2.6 + 0.2 * sin(u.time), 0.0), 0.45 + u.bump * 4.0), 5.0));
  return d;
}`,"torus knot-ish":`fn map(p: vec3f) -> vec2f {
  var d = vec2f(p.y + 0.2, 0.0);
  let q = rotAxis(vec3f(0.0, 1.0, 0.0), u.time * 0.3) * (p - vec3f(0.0, 1.2, 0.0));
  let tor = sdTorus(opTwist(q, u.twist).xzy, vec2f(0.9, 0.22 + u.bump * 3.0));
  d = opU(d, vec2f(smin(tor, sdSphere(q, 0.45), u.blend), 4.0));
  return d;
}`},R={blend:"f32",twist:"f32",bump:"f32"},k=["noise","math"];function N(){let n=w();return async c=>{let o=n.sdf({code:c,uniforms:R,imports:k});try{return await Promise.all([o.program.ready(),o.coneK.ready()]),null}catch(s){return s.message.replace(/^ShaderIntel: /,"")}finally{o.destroy()}}}function E({code:n,onApply:c}){let o=N(),[s,l]=(0,t.useState)(n),[d,p]=(0,t.useState)(null);return(0,e.jsxs)(e.Fragment,{children:[(0,e.jsx)("textarea",{rows:14,value:s,spellCheck:!1,onChange:a=>l(a.target.value)}),(0,e.jsx)(A,{onClick:async()=>{let a=await o(s);p(a),a||c(s)},children:"Apply"}),d&&(0,e.jsx)("div",{className:"err",children:d})]})}function O(){let[n,c]=(0,t.useState)("melting blobs"),[o,s]=(0,t.useState)(f[n]),[l,d]=(0,t.useState)(.35),[p,v]=(0,t.useState)(1.2),[a,q]=(0,t.useState)(.03),[b,B]=(0,t.useState)(!0),[h,U]=(0,t.useState)(!0);return(0,e.jsxs)(g,{camera:{target:[.4,.8,0],distance:6.5,pitch:.25,yaw:.6},background:[.05,.06,.08,1],hud:!0,children:[(0,e.jsx)(S,{code:o,imports:k,uniforms:R,values:{blend:l,twist:p,bump:a},shadows:b,ao:h}),(0,e.jsxs)(x,{title:"Live SDF shader",children:[(0,e.jsxs)("div",{className:"row wide",children:[(0,e.jsx)("label",{children:"preset"}),(0,e.jsx)("select",{value:n,onChange:r=>{c(r.target.value),s(f[r.target.value])},children:Object.keys(f).map(r=>(0,e.jsx)("option",{children:r},r))})]}),(0,e.jsx)(i,{label:"blend",value:l,min:0,max:1,step:.01,onChange:d}),(0,e.jsx)(i,{label:"twist",value:p,min:-3,max:3,step:.01,onChange:v}),(0,e.jsx)(i,{label:"bumps",value:a,min:0,max:.1,step:.005,onChange:q}),(0,e.jsx)(m,{label:"soft shadows",value:b,onChange:B}),(0,e.jsx)(m,{label:"ambient occl.",value:h,onChange:U}),(0,e.jsx)(E,{code:o,onApply:s},n),(0,e.jsxs)(T,{children:["Available: the ",(0,e.jsx)("code",{children:"sdf"}),", ",(0,e.jsx)("code",{children:"noise"})," and ",(0,e.jsx)("code",{children:"math"})," modules, ",(0,e.jsx)("code",{children:"u.time"}),", ",(0,e.jsx)("code",{children:"u.blend"}),", ",(0,e.jsx)("code",{children:"u.twist"}),", ",(0,e.jsx)("code",{children:"u.bump"}),". Return (distance, material id 0\u20135)."]})]}),(0,e.jsxs)(P,{children:[(0,e.jsx)("b",{children:"SDF playground"})," \u2014 one ",(0,e.jsx)("code",{children:"map()"})," function, cone-marched in two passes with soft shadows, AO and PBR. Sliders are uniforms (no recompile); ",(0,e.jsx)("b",{children:"Apply"})," recompiles, and keeps the old shader if the new one has errors."]})]})}C(O);
