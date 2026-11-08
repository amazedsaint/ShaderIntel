// Transports for peer-to-peer sessions. Every transport hands out Channel objects:
//   channel.send(message)          message: plain objects, arrays, numbers, strings, typed arrays
//   channel.onmessage = (m) => {}
//   channel.close()
//
// On the wire a message is JSON in which every typed array travels as base64 of its bytes, so
// the bits arrive exactly: -0, NaN payloads and u32 words stored in float arrays (the rigid
// body records keep collision masks that way) survive, where JSON numbers would not. Messages
// longer than 64 KB are split into parts, under every browser's data-channel message limit.
//
// LoopbackNetwork  two (or more) endpoints in one page with simulated latency, jitter, loss and
//                  reordering, driven by a seeded RNG — for tests, demos and tuning netcode.
// RTCPeer          WebRTC data channels between browsers ("inputs": unordered, no retransmits;
//                  "state": reliable, ordered), with signalling through any Signaler:
// BroadcastSignaler  same-origin tabs (BroadcastChannel) — no server needed for local play/testing.
// WebSocketSignaler  your own relay server (it only forwards {room, from, payload} messages).

/** Seeded xorshift32 → [0, 1). */
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

/**
 * Simulated network. new LoopbackNetwork({ latency: 60, jitter: 15, loss: 0.05, seed: 1 }).
 * connect('a', 'b') returns [channelAtA, channelAtB]; each has .unreliable and .reliable.
 * latency/jitter in ms (one way), loss applies to unreliable channels only.
 */
export class LoopbackNetwork {
  constructor(o = {}) {
    this.latency = o.latency ?? 40;
    this.jitter = o.jitter ?? 10;
    this.loss = o.loss ?? 0;
    this.duplicate = o.duplicate ?? 0;
    this.random = rng(o.seed ?? 1);
    // codec: send what WebRTC would (the wire encoding) instead of a structured clone, so a
    // message that survives the simulator survives the real transport
    this.codec = o.codec ?? true;
    this.stats = { sent: 0, dropped: 0, delivered: 0, bytes: 0 };
  }
  _link(deliver, reliable) {
    let lastArrival = 0;
    return {
      onmessage: null,
      send: (m) => {
        this.stats.sent++;
        if (!reliable && this.random() < this.loss) { this.stats.dropped++; return; }
        const copies = !reliable && this.random() < this.duplicate ? 2 : 1;
        const wire = this.codec ? encode(m) : null;
        if (wire) this.stats.bytes += wire.length;
        for (let c = 0; c < copies; c++) {
          let delay = this.latency + (this.random() * 2 - 1) * this.jitter;
          if (reliable) { delay = Math.max(delay, lastArrival - performance.now()); lastArrival = performance.now() + delay; }   // ordered
          const copy = wire ? decode(wire) : structuredClone(m);
          setTimeout(() => { this.stats.delivered++; deliver(copy); }, Math.max(0, delay));
        }
      },
      close() {},
    };
  }
  connect() {
    const a = { unreliable: null, reliable: null }, b = { unreliable: null, reliable: null };
    for (const kind of ['unreliable', 'reliable']) {
      const rel = kind === 'reliable';
      const ab = this._link((m) => b[kind].onmessage?.(m), rel);
      const ba = this._link((m) => a[kind].onmessage?.(m), rel);
      a[kind] = { send: ab.send, onmessage: null, close() {} };
      b[kind] = { send: ba.send, onmessage: null, close() {} };
    }
    return [a, b];
  }
}

/** Signalling over BroadcastChannel: peers in tabs of the same origin find each other by room name. */
export class BroadcastSignaler {
  constructor(room) {
    this.id = Math.random().toString(36).slice(2);
    this.bc = new BroadcastChannel(`si-signal-${room}`);
    this.onmessage = null;
    this.bc.onmessage = (e) => { if (e.data.from !== this.id && (!e.data.to || e.data.to === this.id)) this.onmessage?.(e.data); };
  }
  send(payload, to) { this.bc.postMessage({ from: this.id, to, payload }); }
  close() { this.bc.close(); }
}

/** Signalling through a WebSocket relay that broadcasts { room, from, to, payload } to the room. */
export class WebSocketSignaler {
  constructor(url, room) {
    this.id = Math.random().toString(36).slice(2);
    this.room = room;
    this.onmessage = null;
    this.ws = new WebSocket(url);
    this.ready = new Promise((res, rej) => { this.ws.onopen = () => res(); this.ws.onerror = rej; });
    this.ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.room === room && m.from !== this.id && (!m.to || m.to === this.id)) this.onmessage?.(m);
    };
  }
  async send(payload, to) { await this.ready; this.ws.send(JSON.stringify({ room: this.room, from: this.id, to, payload })); }
  close() { this.ws.close(); }
}

/**
 * A WebRTC peer. The initiator creates the data channels and the offer; the other side answers.
 *   const peer = new RTCPeer(new BroadcastSignaler('room-1'), { initiator: true });
 *   await peer.ready;  peer.channels.unreliable.send(...)
 * iceServers: e.g. [{ urls: 'stun:stun.l.google.com:19302' }] for peers on different networks.
 */
export class RTCPeer {
  constructor(signaler, o = {}) {
    this.signaler = signaler;
    this.pc = new RTCPeerConnection({ iceServers: o.iceServers || [] });
    this.channels = { unreliable: dataChannel(null), reliable: dataChannel(null) };
    this.remote = null;
    const attach = (dc) => {
      dc.binaryType = 'arraybuffer';
      this.channels[dc.label === 'inputs' ? 'unreliable' : 'reliable'].attach(dc);
    };
    this.ready = new Promise((resolve) => {
      let open = 0;
      const check = (dc) => { dc.addEventListener('open', () => { if (++open === 2) resolve(this); }); };
      if (o.initiator) {
        const a = this.pc.createDataChannel('inputs', { ordered: false, maxRetransmits: 0 });
        const b = this.pc.createDataChannel('state', { ordered: true });
        attach(a); attach(b); check(a); check(b);
      } else {
        this.pc.ondatachannel = (e) => { attach(e.channel); check(e.channel); };
      }
    });
    this.pc.onicecandidate = (e) => { if (e.candidate) signaler.send({ ice: e.candidate.toJSON() }, this.remote); };
    // Signalling messages are handled one at a time, and candidates that arrive before the remote
    // description is applied wait for it (dropping them can leave the connection without a path).
    const pendingIce = [];
    let chain = Promise.resolve();
    const handle = async (m) => {
      const p = m.payload;
      if (p.hello) {
        if (!o.initiator || this.remote) return;   // the first guest to say hello is the peer
        this.remote = m.from;
        await this.pc.setLocalDescription(await this.pc.createOffer());
        signaler.send({ sdp: this.pc.localDescription.toJSON() }, this.remote);
      } else if (p.sdp) {
        if (this.remote && m.from !== this.remote) return;
        this.remote = m.from;
        clearInterval(this._hello);
        await this.pc.setRemoteDescription(p.sdp);
        if (p.sdp.type === 'offer') { await this.pc.setLocalDescription(await this.pc.createAnswer()); signaler.send({ sdp: this.pc.localDescription.toJSON() }, this.remote); }
        for (const c of pendingIce.splice(0)) await this.pc.addIceCandidate(c).catch(() => {});
      } else if (p.ice) {
        if (this.remote && m.from !== this.remote) return;
        if (!this.pc.remoteDescription) pendingIce.push(p.ice);
        else await this.pc.addIceCandidate(p.ice).catch(() => {});
      }
    };
    signaler.onmessage = (m) => { chain = chain.then(() => handle(m)).catch((e) => console.warn('RTCPeer signalling:', e)); };
    // The guest says hello until the host answers: the host may not be listening yet (a page that
    // is still initialising its GPU), and a single hello would then be lost for good.
    if (!o.initiator) {
      signaler.send({ hello: true });
      this._hello = setInterval(() => signaler.send({ hello: true }), 500);
    }
  }
  close() { clearInterval(this._hello); this.pc.close(); this.signaler.close?.(); }
}

const CHUNK = 65536;

/**
 * Wrap an RTCDataChannel (or anything with send(), onmessage and readyState) as a Channel: the
 * wire encoding above, with long messages split into parts and reassembled. Use it to run a
 * RollbackSession over a WebRTC connection you set up yourself.
 */
export function dataChannel(dc) {
  const parts = new Map();
  let next = 1;
  const ch = {
    dc: null,
    onmessage: null,
    send(m) {
      if (this.dc?.readyState !== 'open') return;
      const s = encode(m);
      if (s.length <= CHUNK) { this.dc.send(s); return; }
      const id = next++, n = Math.ceil(s.length / CHUNK);
      for (let k = 0; k < n; k++) this.dc.send(JSON.stringify({ __part: id, k, n, d: s.slice(k * CHUNK, (k + 1) * CHUNK) }));
    },
    close() { this.dc?.close(); },
    attach(d) {
      this.dc = d;
      d.onmessage = (e) => {
        const text = typeof e.data === 'string' ? e.data : new TextDecoder().decode(e.data);
        if (!text.startsWith('{"__part"')) { this.onmessage?.(decode(text)); return; }
        const p = JSON.parse(text);
        const got = parts.get(p.__part) || { n: p.n, have: 0, d: [] };
        if (got.d[p.k] === undefined) { got.d[p.k] = p.d; got.have++; }
        parts.set(p.__part, got);
        if (got.have < got.n) return;
        parts.delete(p.__part);
        this.onmessage?.(decode(got.d.join('')));
      };
    },
  };
  if (dc) ch.attach(dc);
  return ch;
}

function toBase64(bytes) {
  let s = '';
  for (let k = 0; k < bytes.length; k += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(k, k + 0x8000));
  return btoa(s);
}
function fromBase64(b64) {
  const s = atob(b64), bytes = new Uint8Array(s.length);
  for (let k = 0; k < s.length; k++) bytes[k] = s.charCodeAt(k);
  return bytes;
}
/** The wire encoding: JSON, with typed arrays as { __ta: type, b: base64 of the bytes }. */
export function encode(m) {
  return JSON.stringify(m, (k, v) => (ArrayBuffer.isView(v) ? { __ta: v.constructor.name, b: toBase64(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)) } : v));
}
export function decode(s) {
  return JSON.parse(s, (k, v) => (v && typeof v === 'object' && typeof v.__ta === 'string' ? new globalThis[v.__ta](fromBase64(v.b).buffer) : v));
}
