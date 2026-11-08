// Transports for peer-to-peer sessions. Every transport hands out Channel objects:
//   channel.send(message)          message: plain objects, arrays, numbers, strings, typed arrays
//   channel.onmessage = (m) => {}
//   channel.close()
//
// On the wire a message is JSON in which every typed array travels as base64 of its bytes, so
// the bits arrive exactly: -0, NaN payloads and u32 words stored in float arrays (the rigid
// body records keep collision masks that way) survive, where JSON numbers would not. Numbers
// that JSON cannot represent (NaN, ±Infinity, -0), undefined, BigInt, Map, Set, Date and
// ArrayBuffer are tagged too, so game state crosses the wire the way structuredClone copies it.
// Decoding only ever constructs those types. Messages longer than 64 KB travel as binary parts
// of at most 64 KB, under every browser's data-channel message limit.
//
// LoopbackNetwork  two (or more) endpoints in one page with simulated latency, jitter, loss and
//                  reordering, driven by a seeded RNG — for tests, demos and tuning netcode.
// RTCPeer          WebRTC data channels between browsers ("inputs": unordered, no retransmits;
//                  "state": reliable, ordered), with signalling through any Signaler:
// BroadcastSignaler  same-origin tabs (BroadcastChannel) — no server needed for local play/testing.
// WebSocketSignaler  your own relay server (it only forwards {room, from, payload} messages).

import { SIError } from '../core/errors.js';

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
    // Reliable links deliver strictly in order: every message gets a sequence number and waits
    // for its predecessors (timers alone reorder messages whose delays differ by under 1 ms).
    let seqOut = 0, seqIn = 0, closed = false;
    const ready = new Map();
    const link = {
      send: (m) => {
        if (closed) return;
        this.stats.sent++;
        if (!reliable && this.random() < this.loss) { this.stats.dropped++; return; }
        const copies = !reliable && this.random() < this.duplicate ? 2 : 1;
        const wire = this.codec ? encode(m) : null;
        if (wire) this.stats.bytes += wire.length;
        for (let c = 0; c < copies; c++) {
          const delay = Math.max(0, this.latency + (this.random() * 2 - 1) * this.jitter);
          const copy = wire ? decode(wire) : structuredClone(m);
          if (!reliable) { setTimeout(() => { if (!closed) { this.stats.delivered++; deliver(copy); } }, delay); continue; }
          const seq = seqOut++;
          setTimeout(() => {
            ready.set(seq, copy);
            while (!closed && ready.has(seqIn)) { const x = ready.get(seqIn); ready.delete(seqIn++); this.stats.delivered++; deliver(x); }
          }, delay);
        }
      },
      close() { closed = true; ready.clear(); },
    };
    return link;
  }
  /** Two connected endpoints [a, b], each { unreliable, reliable } (channel.close() stops its deliveries). */
  connect() {
    const a = { unreliable: null, reliable: null }, b = { unreliable: null, reliable: null };
    for (const kind of ['unreliable', 'reliable']) {
      const rel = kind === 'reliable';
      const ab = this._link((m) => b[kind].onmessage?.(m), rel);
      const ba = this._link((m) => a[kind].onmessage?.(m), rel);
      a[kind] = { send: ab.send, onmessage: null, close() { ab.close(); ba.close(); } };
      b[kind] = { send: ba.send, onmessage: null, close() { ab.close(); ba.close(); } };
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
    this.ready = new Promise((res, rej) => {
      this.ws.onopen = () => res();
      this.ws.onerror = () => rej(new SIError(`WebSocketSignaler: cannot connect to ${url}`, 'error'));
    });
    this.ready.catch(() => {});   // reported to whoever awaits it (send), not as an unhandled rejection
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
    this.closed = false;
    this.pc = new RTCPeerConnection({ iceServers: o.iceServers || [] });
    this.channels = { unreliable: dataChannel(null), reliable: dataChannel(null) };
    this.remote = null;
    let warned = false;
    // a signaler that cannot send (an unreachable relay) is reported once, not every retry
    const send = (payload, to) => {
      try { Promise.resolve(signaler.send(payload, to)).catch((e) => { if (!warned) { warned = true; console.warn('RTCPeer signalling:', e?.message || e); } }); }
      catch (e) { if (!warned) { warned = true; console.warn('RTCPeer signalling:', e?.message || e); } }
    };
    const attach = (dc) => {
      dc.binaryType = 'arraybuffer';
      this.channels[dc.label === 'inputs' ? 'unreliable' : 'reliable'].attach(dc);
    };
    let fail;
    this.ready = new Promise((resolve, reject) => {
      fail = reject;
      let open = 0;
      const check = (dc) => { dc.addEventListener('open', () => { if (++open === 2) { clearInterval(this._hello); resolve(this); } }); };
      if (o.initiator) {
        const a = this.pc.createDataChannel('inputs', { ordered: false, maxRetransmits: 0 });
        const b = this.pc.createDataChannel('state', { ordered: true });
        attach(a); attach(b); check(a); check(b);
      } else {
        this.pc.ondatachannel = (e) => { attach(e.channel); check(e.channel); };
      }
    });
    this._fail = (e) => fail(e);
    this.ready.catch(() => {});   // callers that await it see the rejection
    this.pc.onconnectionstatechange = () => {
      if (this.pc.connectionState === 'failed')
        fail(new SIError('RTCPeer: the connection failed — the peers found no network path (add a TURN server to iceServers)'));
    };
    this.pc.onicecandidate = (e) => { if (e.candidate) send({ ice: e.candidate.toJSON() }, this.remote); };
    // Signalling messages are handled one at a time, and candidates that arrive before the remote
    // description is applied wait for it (dropping them can leave the connection without a path).
    // Lost messages are recovered: the guest says hello until the channels open; the host answers a
    // repeated hello by re-sending its offer, and the guest a repeated offer by re-sending its answer.
    const pendingIce = [];
    let chain = Promise.resolve();
    const handle = async (m) => {
      if (this.closed) return;
      const p = m.payload;
      if (p.hello) {
        if (!o.initiator) return;
        if (this.remote && m.from !== this.remote) return;   // the first guest to say hello is the peer
        if (!this.remote) {
          this.remote = m.from;
          await this.pc.setLocalDescription(await this.pc.createOffer());
        } else if (this.pc.signalingState !== 'have-local-offer') return;   // answered already: nothing was lost
        send({ sdp: this.pc.localDescription.toJSON() }, this.remote);
      } else if (p.sdp) {
        if (this.remote && m.from !== this.remote) return;
        this.remote = m.from;
        if (p.sdp.type === 'offer' && this.pc.signalingState === 'stable' && this.pc.localDescription?.type === 'answer') {
          send({ sdp: this.pc.localDescription.toJSON() }, this.remote);   // our answer was lost: repeat it
          return;
        }
        if (p.sdp.type === 'answer' && this.pc.signalingState !== 'have-local-offer') return;   // a duplicate answer
        await this.pc.setRemoteDescription(p.sdp);
        if (p.sdp.type === 'offer') { await this.pc.setLocalDescription(await this.pc.createAnswer()); send({ sdp: this.pc.localDescription.toJSON() }, this.remote); }
        for (const c of pendingIce.splice(0)) await this.pc.addIceCandidate(c).catch(() => {});
      } else if (p.ice) {
        if (this.remote && m.from !== this.remote) return;
        if (!this.pc.remoteDescription) pendingIce.push(p.ice);
        else await this.pc.addIceCandidate(p.ice).catch(() => {});
      }
    };
    signaler.onmessage = (m) => { chain = chain.then(() => handle(m)).catch((e) => console.warn('RTCPeer signalling:', e)); };
    // The guest says hello until the channels open: the host may not be listening yet (a page that
    // is still initialising its GPU), and a single hello — or the reply to it — could be lost.
    if (!o.initiator) {
      send({ hello: true });
      this._hello = setInterval(() => send({ hello: true }), 500);
    }
  }
  /** Close the connection and the signaler (the peer owns it); a pending ready rejects. */
  close() {
    if (this.closed) return;
    this.closed = true;
    clearInterval(this._hello);
    this.signaler.onmessage = null;
    this.signaler.close?.();
    this.pc.close();
    this._fail(new SIError('RTCPeer: closed before the connection opened'));
  }
}

const CHUNK = 65536;          // bytes per data-channel message, under every browser's limit
const PART = 0x53495054;      // 'SIPT': a binary part of a long message
const PART_TIMEOUT = 15000;   // incomplete messages are forgotten after this (a part lost on an unreliable channel)

/**
 * Wrap an RTCDataChannel (or anything with send(), onmessage and readyState) as a Channel: the
 * wire encoding above, with long messages split into binary parts of at most 64 KB and
 * reassembled. Use it to run a RollbackSession over a WebRTC connection you set up yourself.
 */
export function dataChannel(dc) {
  const parts = new Map();
  let next = 1;
  const utf8 = new TextEncoder(), text8 = new TextDecoder();
  const ch = {
    dc: null,
    onmessage: null,
    send(m) {
      if (this.dc?.readyState !== 'open') return;
      const s = encode(m);
      if (s.length * 3 <= CHUNK) { this.dc.send(s); return; }   // surely small enough as UTF-8
      const bytes = utf8.encode(s);
      if (bytes.length <= CHUNK) { this.dc.send(s); return; }
      const body = CHUNK - 12, id = next++ >>> 0, n = Math.ceil(bytes.length / body);
      for (let k = 0; k < n; k++) {
        const chunk = bytes.subarray(k * body, (k + 1) * body);
        const msg = new Uint8Array(12 + chunk.length);
        const h = new DataView(msg.buffer);
        h.setUint32(0, PART); h.setUint32(4, id); h.setUint16(8, k); h.setUint16(10, n);
        msg.set(chunk, 12);
        this.dc.send(msg.buffer);
      }
    },
    close() { this.dc?.close(); parts.clear(); },
    attach(d) {
      this.dc = d;
      d.onmessage = (e) => {
        let text;
        if (typeof e.data === 'string') text = e.data;
        else {
          const bytes = new Uint8Array(e.data);
          const h = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          if (bytes.length < 12 || h.getUint32(0) !== PART) text = text8.decode(bytes);
          else {
            const id = h.getUint32(4), k = h.getUint16(8), n = h.getUint16(10), now = performance.now();
            for (const [key, g] of parts) if (now - g.t > PART_TIMEOUT) parts.delete(key);
            const got = parts.get(id) || { n, have: 0, d: [], t: now };
            if (got.d[k] === undefined && k < got.n) { got.d[k] = bytes.slice(12); got.have++; }
            parts.set(id, got);
            if (got.have < got.n) return;
            parts.delete(id);
            const all = new Uint8Array(got.d.reduce((a, c) => a + c.length, 0));
            let o = 0;
            for (const c of got.d) { all.set(c, o); o += c.length; }
            text = text8.decode(all);
          }
        }
        let m;
        try { m = decode(text); } catch (err) { console.warn('dataChannel: dropped a malformed message:', err.message); return; }
        this.onmessage?.(m);
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
// Typed arrays the decoder may construct (a remote message must not name arbitrary globals).
const TYPED = Object.fromEntries(['Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array',
  'Float16Array', 'Float32Array', 'Float64Array', 'BigInt64Array', 'BigUint64Array', 'DataView']
  .filter((n) => typeof globalThis[n] === 'function').map((n) => [n, globalThis[n]]));

/**
 * The wire encoding: JSON in which typed arrays travel as { __ta: type, b: base64 of the bytes },
 * and values JSON cannot carry are tagged { __t: kind, v }: NaN, ±Infinity and -0, undefined,
 * BigInt, Map, Set, Date, ArrayBuffer (and objects that happen to have a __t / __ta key).
 */
export function encode(m) {
  return JSON.stringify(m, function (k, v) {
    const raw = this[k];   // the value before toJSON (a Date would already be a string)
    if (ArrayBuffer.isView(raw)) return { __ta: raw.constructor.name, b: toBase64(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength)) };
    if (raw instanceof ArrayBuffer) return { __t: 'ab', b: toBase64(new Uint8Array(raw)) };
    if (typeof raw === 'number') return Number.isFinite(raw) && !Object.is(raw, -0) ? raw : { __t: 'num', v: Object.is(raw, -0) ? '-0' : String(raw) };
    if (typeof raw === 'bigint') return { __t: 'big', v: raw.toString() };
    if (raw === undefined) return k === '' ? undefined : { __t: 'u' };
    if (raw instanceof Date) return { __t: 'date', v: String(raw.getTime()) };
    if (raw instanceof Map) return { __t: 'map', v: [...raw] };
    if (raw instanceof Set) return { __t: 'set', v: [...raw] };
    if (raw && typeof raw === 'object' && !Array.isArray(raw) && v === raw && ('__t' in raw || '__ta' in raw)) return { __t: 'obj', v: Object.entries(raw) };
    return v;
  });
}

/** Inverse of encode(). Throws SIError on a malformed message (unknown typed array, bad sizes). */
export function decode(s) {
  return JSON.parse(s, (k, v) => {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return v;
    if (typeof v.__ta === 'string') {
      const Ctor = TYPED[v.__ta];
      if (!Ctor || typeof v.b !== 'string') throw new SIError(`decode: "${v.__ta}" is not a typed array`, 'invalid-argument');
      const bytes = fromBase64(v.b);
      if (Ctor === DataView) return new DataView(bytes.buffer);
      if (bytes.byteLength % Ctor.BYTES_PER_ELEMENT) throw new SIError(`decode: ${bytes.byteLength} bytes are not a whole number of ${v.__ta} elements`, 'invalid-argument');
      return new Ctor(bytes.buffer);
    }
    switch (v.__t) {
      case 'num': return Number(v.v);
      case 'u': return undefined;
      case 'big': return BigInt(v.v);
      case 'date': return new Date(Number(v.v));
      case 'map': return new Map(v.v);
      case 'set': return new Set(v.v);
      case 'ab': return fromBase64(v.b).buffer;
      case 'obj': return Object.fromEntries(v.v);
      default: return v;
    }
  });
}
