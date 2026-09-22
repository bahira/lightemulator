// ============================================================================
//  spear_search2.mjs — MUR #1 : exp sous ×3.
//  Hypothèse : la division du rationnel [3/2] coûte ~4 cycles de throughput.
//  Attaque : polynôme PUR (Horner, zéro division) après réduction de portée.
//    exp(x) = 2^k · poly(r),  r = x − k·ln2 ∈ [−ln2/2, ln2/2]
//  Deux variantes cherchées (500 itérations chacune, minimax rel) :
//    poly4 : 1 + c1r + c2r² + c3r³ + c4r⁴   (~4 FMA)
//    poly3 : 1 + c1r + c2r² + c3r³          (~3 FMA)
//  Verdict final : bench C (lm_bench_exp) — le JS n'est qu'un indicatif.
// ============================================================================
import { writeFileSync } from 'node:fs';

const ITER = 500;
const LN2 = Math.LN2;

function makeSearch(deg) {
  const grid = [];
  for (let i = 0; i <= 8192; i++) grid.push(-14 + 16 * i / 8192);
  const ref = grid.map(Math.exp);
  function errOf(c) {
    let worst = 0;
    for (let i = 0; i < grid.length; i++) {
      const x = grid[i];
      const k = Math.round(x / LN2);
      const r = x - k * LN2;
      let p = 0;
      for (let j = deg; j >= 1; j--) p = (p + c[j - 1]) * r;
      const approx = Math.pow(2, k) * (1 + p);
      const e = Math.abs(approx - ref[i]) / ref[i];
      if (e > worst) worst = e;
    }
    return worst;
  }
  // seed : Taylor tronqué
  const fact = [1, 1, 2, 6, 24];
  const seed = [];
  for (let j = 1; j <= deg; j++) seed.push(1 / fact[j]);
  let best = seed, bestErr = errOf(seed);
  const rng = (() => { let s = 0xC0FFEE + deg; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
  let accepted = 0;
  const t0 = Date.now();
  for (let it = 0; it < ITER; it++) {
    const scale = bestErr * (2 + 8 * (1 - it / ITER)) + 1e-11;
    const cand = best.map(v => v + (rng() * 2 - 1) * scale * (rng() < 0.5 ? 1 : 0.2));
    const e = errOf(cand);
    if (e < bestErr) { bestErr = e; best = cand; accepted++; }
  }
  return { deg, best, bestErr, ms: Date.now() - t0, accepted };
}

const results = [];
for (const deg of [4, 3]) {
  const r = makeSearch(deg);
  results.push(r);
  console.log(`[poly${deg}] ${ITER} it (${r.ms} ms, ${r.accepted} acc) → L∞ rel = ${r.bestErr.toExponential(3)} sur [−14,+2]`);
  console.log(`          c = [${r.best.map(v => v.toFixed(9)).join('f, ')}f]`);
}

// bench JS indicatif (le verdict final se mesure en C)
{
  const buf = new Float64Array(1), bits = new Uint32Array(buf.buffer);
  function mkExp(c, deg) {
    return function (x) {
      const k = Math.round(x * 1.4426950408889634);
      const r = x - k * 0.6931471805599453;
      let p = 0;
      for (let j = deg; j >= 1; j--) p = (p + c[j - 1]) * r;
      buf[0] = 1; bits[1] = (1023 + k) << 20;
      return buf[0] * (1 + p);
    };
  }
  function bench(fn) {
    const args = new Float64Array(20000);
    let s = 42;
    for (let i = 0; i < args.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; args[i] = -12 * (s / 4294967296); }
    let sink = 0;
    for (let w = 0; w < 5; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const t = process.hrtime.bigint();
    for (let w = 0; w < 200; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    globalThis.__sink = sink;
    return Number(process.hrtime.bigint() - t) / 1e6 / (200 * args.length);
  }
  const tRef = bench(Math.exp);
  for (const r of results) {
    const tNew = bench(mkExp(r.best, r.deg));
    console.log(`[poly${r.deg}] JS indicatif : Math.exp ${tRef.toFixed(2)} ns vs ${tNew.toFixed(2)} ns → ×${(tRef / tNew).toFixed(2)}`);
  }
}

writeFileSync('lm_c/kernels_found2.json', JSON.stringify({
  note: 'poly sans division — coefficients minimax 500 itérations',
  variants: results.map(r => ({ deg: r.deg, coeffs: r.best, linf: r.bestErr })),
}, null, 2));
console.log('→ lm_c/kernels_found2.json');
