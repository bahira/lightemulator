// ============================================================================
//  spear_search.mjs v2 â€” RECHERCHE SYMBOLIQUE Ã‰VOLUTIONNAIRE (500 it/cible)
//  exp : PadÃ© [3/2] exact en seed + raffinement Ã©volutionnaire
//  rsqrt : normalisation d'exposant par bits (branchless) + Newton sans division
// ============================================================================
import { writeFileSync } from 'node:fs';

const ITER = 500;

// ---------------------------------------------------------------------------
//  Cible 1 : exp(x) sur [âˆ’12,0]
//  x = kÂ·ln2 + r, râˆˆ[âˆ’ln2/2, ln2/2] ; e^r â‰ˆ P/Q, P=1+p1r+p2rÂ²+p3rÂ³, Q=1+q1r+q2rÂ²
//  Seed = PadÃ© [3/2] de e^r : P = 1+r/2+rÂ²/10+rÂ³/120, Q = 1âˆ’r/2+rÂ²/10 (Lâˆž~1e-4)
// ---------------------------------------------------------------------------
{
  const grid = [];
  for (let i = 0; i <= 4096; i++) grid.push(-12 + (12 * i) / 4096);
  const ref = grid.map(Math.exp);
  function errOf(c) {
    const [p1, p2, p3, q1, q2] = c;
    let worst = 0;
    for (let i = 0; i < grid.length; i++) {
      const x = grid[i];
      const k = Math.round(x * 1.4426950408889634);
      const r = x - k * 0.6931471805599453;
      const approx = Math.pow(2, k) * ((1 + r * (p1 + r * (p2 + r * p3))) / (1 + r * (q1 + r * q2)));
      const e = Math.abs(approx - ref[i]) / ref[i];
      if (e > worst) worst = e;
    }
    return worst;
  }
  let best = [0.5, 0.1, 1 / 120, -0.5, 0.1];
  let bestErr = errOf(best);
  console.log(`[exp]  seed PadÃ© [3/2] : Lâˆž = ${bestErr.toExponential(3)}`);
  const rng = (() => { let s = 1234567; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
  let accepted = 0;
  const t0 = Date.now();
  for (let it = 0; it < ITER; it++) {
    const scale = bestErr * (3 + 9 * (1 - it / ITER)) + 1e-9;
    const cand = best.map((v) => v + (rng() * 2 - 1) * scale * (rng() < 0.5 ? 1 : 0.25));
    const e = errOf(cand);
    if (e < bestErr) { bestErr = e; best = cand; accepted++; }
  }
  console.log(`[exp]  ${ITER} itÃ©rations (${Date.now() - t0} ms, ${accepted} acceptÃ©es) â†’ Lâˆž = ${bestErr.toExponential(3)} (ancien kernel 3.4e-3)`);

  // kernel final + bench honnÃªte (2M appels)
  const [p1, p2, p3, q1, q2] = best;
  const buf = new Float64Array(1), bits = new Uint32Array(buf.buffer);
  function spearExp(x) {
    const k = Math.round(x * 1.4426950408889634);
    const r = x - k * 0.6931471805599453;
    const num = 1 + r * (p1 + r * (p2 + r * p3));
    const den = 1 + r * (q1 + r * q2);
    buf[0] = 1; bits[1] = (1023 + k) << 20;   // 2^k par l'exposant IEEE754 (ordre : valeur PUIS bits)
    return buf[0] * (num / den);
  }
  // vÃ©rif du kernel FINAL (avec ldexp bits, pas Math.pow)
  let wv = 0;
  for (let i = 0; i <= 4096; i++) { const x = -12 + (12 * i) / 4096; const e = Math.abs(spearExp(x) - Math.exp(x)) / Math.exp(x); if (e > wv) wv = e; }
  console.log(`[exp]  kernel final (ldexp bits) Lâˆž = ${wv.toExponential(3)}`);

  function bench(fn) {
    const args = new Float64Array(20000);
    let s = 42;
    for (let i = 0; i < args.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; args[i] = -12 * (s / 4294967296); }
    let sink = 0;
    for (let w = 0; w < 5; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const t = process.hrtime.bigint();
    for (let w = 0; w < 200; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const dt = Number(process.hrtime.bigint() - t) / 1e6;
    globalThis.__sink = sink;
    return (dt * 1e6) / (200 * args.length);
  }
  const tRef = bench(Math.exp), tNew = bench(spearExp);
  console.log(`[exp]  JS : Math.exp ${tRef.toFixed(2)} ns vs spear ${tNew.toFixed(2)} ns â†’ Ã—${(tRef / tNew).toFixed(1)}`);
  globalThis.__exp = { coeffs: best, linf: wv, jsSpeedup: tRef / tNew };
}

// ---------------------------------------------------------------------------
//  Cible 2 : rsqrt sans division, normalisation d'exposant branchless par bits
//  x = mÂ²Â·y avec yâˆˆ[1,4) ; 1/âˆšx = inv(m)Â·g(y) ; seed linÃ©aire + 2 Newton
// ---------------------------------------------------------------------------
{
  const grid = [];
  for (let i = 0; i <= 4096; i++) grid.push(1 + (3 * i) / 4096);
  const ref = grid.map((y) => 1 / Math.sqrt(y));
  function errOf(c) {
    const [a, b] = c;
    let worst = 0;
    for (let i = 0; i < grid.length; i++) {
      const y = grid[i];
      let g = a + b * y;
      g *= 0.5 * (3 - y * g * g);
      g *= 0.5 * (3 - y * g * g);
      const e = Math.abs(g - ref[i]) / ref[i];
      if (e > worst) worst = e;
    }
    return worst;
  }
  let best = [1.1, -0.15];
  let bestErr = errOf(best);
  const rng = (() => { let s = 7654321; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
  let accepted = 0;
  const t0 = Date.now();
  for (let it = 0; it < ITER; it++) {
    const scale = bestErr * (4 + 12 * (1 - it / ITER)) + 1e-10;
    const cand = best.map((v) => v + (rng() * 2 - 1) * scale * (rng() < 0.5 ? 1 : 0.3));
    const e = errOf(cand);
    if (e < bestErr) { bestErr = e; best = cand; accepted++; }
  }
  console.log(`[rsqrt] ${ITER} itÃ©rations (${Date.now() - t0} ms, ${accepted} acceptÃ©es) â†’ Lâˆž [1,4) = ${bestErr.toExponential(3)}`);

  const [A, B] = best;
  const buf = new Float64Array(1), bits = new Uint32Array(buf.buffer);
  function spearRsqrt(x) {
    buf[0] = x;
    const E = (((bits[1] >>> 20) & 0x7ff) - 1023) >> 1;        // âŒŠE/2âŒ‹ arithmÃ©tique
    bits[1] = (1023 - 2 * E) << 20; const y = x * buf[0];      // y = xÂ·2^(âˆ’2E) âˆˆ [1,4)
    bits[1] = (1023 - E) << 20; const invm = buf[0];           // 2^(âˆ’E)
    let g = A + B * y;
    g *= 0.5 * (3 - y * g * g);
    g *= 0.5 * (3 - y * g * g);
    return g * invm;
  }
  let wv = 0;
  for (let i = 0; i <= 20000; i++) { const x = 1e-3 + 1000 * (i / 20000); const e = Math.abs(spearRsqrt(x) - 1 / Math.sqrt(x)) * Math.sqrt(x); if (e > wv) wv = e; }
  console.log(`[rsqrt] kernel final Lâˆž relative sur [1e-3, 1000] = ${wv.toExponential(3)}`);

  function bench(fn) {
    const args = new Float64Array(20000);
    let s = 99;
    for (let i = 0; i < args.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; args[i] = 0.01 + 100 * (s / 4294967296); }
    let sink = 0;
    for (let w = 0; w < 5; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const t = process.hrtime.bigint();
    for (let w = 0; w < 200; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const dt = Number(process.hrtime.bigint() - t) / 1e6;
    globalThis.__sink = sink;
    return (dt * 1e6) / (200 * args.length);
  }
  const tRef = bench((x) => 1 / Math.sqrt(x)), tNew = bench(spearRsqrt);
  console.log(`[rsqrt] JS : 1/sqrt ${tRef.toFixed(2)} ns vs no-div ${tNew.toFixed(2)} ns â†’ Ã—${(tRef / tNew).toFixed(1)} (le verdict final se mesure en C)`);
  globalThis.__rsqrt = { coeffs: best, linf: wv };
}

writeFileSync('lm_c/kernels_found.json', JSON.stringify({ iterations: ITER, exp: globalThis.__exp, rsqrt: globalThis.__rsqrt }, null, 2));
console.log('\nâ†’ lm_c/kernels_found.json Ã©crit');

