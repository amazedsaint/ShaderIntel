// The error class for every failure ShaderIntel reports on purpose (no imports, so any module can use it).
//   try { ... } catch (e) { if (e instanceof SI.SIError && e.code === 'unsupported') showFallback(); }
// Codes: 'unsupported' (no WebGPU), 'no-adapter', 'no-device', 'compile' (WGSL errors),
// 'invalid-argument', 'error' (anything else).
export class SIError extends Error {
  constructor(msg, code = 'error') {
    super(msg.startsWith('ShaderIntel') ? msg : `ShaderIntel: ${msg}`);
    this.name = 'ShaderIntelError';
    this.code = code;
  }
}
