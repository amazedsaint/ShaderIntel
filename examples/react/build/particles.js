import{A as w,B as z,C as m,D as v,F as x,G as B,H as k,a as f,b as A,e as g,f as b,g as h,h as y,i as S,k as C}from"./chunks/chunk-HS5P52HZ.js";var a=f(A(),1);var e=f(y(),1);function P({count:s,speed:i,scale:r,shape:p,seed:t}){let o=b("vec4f",s),n=h({imports:["hash"],buffers:{pos:o},uniforms:{seed:"u32",shape:"u32"},code:`
      var s = rngSeed(i, u.seed);
      var p = randBall(&s);
      if (u.shape == 1u) { p = vec3f(p.x, p.y * 0.08, p.z); }                         // a disc
      if (u.shape == 2u) { p = normalize(p) * (0.9 + 0.1 * rand(&s)); }              // a shell
      pos[i] = vec4f(p, length(p));`},[o]),c=h({imports:["hash","noise"],buffers:{pos:o},uniforms:{time:"f32",dt:"f32",frame:"u32",speed:"f32",scale:"f32"},code:`
      var p = pos[i];
      let v = curl3(p.xyz * u.scale + vec3f(0.0, u.time * 0.05, 0.0));   // divergence-free: no clumping
      p = vec4f(p.xyz + v * (u.speed * u.dt), p.w);
      if (dot(p.xyz, p.xyz) > 1.8) {                                    // respawn what drifts away
        var s = rngSeed(i, u.frame);
        let q = randBall(&s);
        p = vec4f(q * 0.5, length(q));
      }
      pos[i] = p;`},[o]);return(0,a.useLayoutEffect)(()=>{n.run(s,{seed:t,shape:{ball:0,disc:1,shell:2}[p]})},[n,s,p,t]),g(({dt:u,time:d,frame:l})=>c.run(s,{time:d,dt:u,frame:l,speed:i,scale:r})),(0,e.jsx)(C,{data:o,size:.002,pixelSize:1,colorBy:"w",intensity:Math.min(.6,.22*1e6/s)})}function q(){let[s,i]=(0,a.useState)(1e6),[r,p]=(0,a.useState)(.12),[t,o]=(0,a.useState)(.9),[n,c]=(0,a.useState)("ball"),[u,d]=(0,a.useState)(1);return(0,e.jsxs)(S,{camera:{distance:4.2,pitch:.3,autoRotate:.08},bloom:{strength:.08},background:[0,0,0,1],hud:!0,children:[(0,e.jsx)(P,{count:s,speed:r,scale:t,shape:n,seed:u}),(0,e.jsxs)(z,{title:"Compute particles (hooks)",children:[(0,e.jsx)(v,{label:"particles",value:String(s),options:["100000","1000000","4000000"],onChange:l=>i(+l)}),(0,e.jsx)(v,{label:"start shape",value:n,options:["ball","disc","shell"],onChange:c}),(0,e.jsx)(m,{label:"speed",value:r,min:0,max:.5,step:.01,onChange:p}),(0,e.jsx)(m,{label:"noise scale",value:t,min:.2,max:3,step:.05,onChange:o}),(0,e.jsx)(x,{onClick:()=>d(l=>l+1),children:"Respawn"}),(0,e.jsxs)(B,{children:[s.toLocaleString()," particles advected through curl noise, one compute dispatch per frame."]})]}),(0,e.jsxs)(k,{children:[(0,e.jsx)("b",{children:"Hooks"})," \u2014 ",(0,e.jsx)("code",{children:"useArray"})," + ",(0,e.jsx)("code",{children:"useKernel"})," + ",(0,e.jsx)("code",{children:"useFrame"}),": a WGSL body is all you write; bindings, uniforms and dispatch sizes are generated. Changing the count re-allocates the GPU array; the old one is freed."]})]})}w(q);
