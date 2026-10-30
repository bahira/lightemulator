/*
 * bench_wasm_gemm.mjs — attribue le gain du module wasm entre « largeur SIMD »
 * et « blocage 4 lignes M », en générant les trois variantes depuis la même
 * source (lm_c/lm_wasm.c) :
 *
 *   scalaire        -DLM_NO_WASM_SIMD   (chemin 1 ligne historique)
 *   v128 1 ligne    -DLM_WASM_1ROW      (16 voies ? non : 4 voies, 1 ligne M)
 *   v128 4 lignes   défaut              (4 voies × 4 lignes M)
 *
 * Protocole : mêmes 120 tokens greedy, instances alternées image par image,
 * médiane sur 5 rondes. Échoue si les sorties greedy divergent (le module doit
 * rester déterministe) ou si le blocage 4 lignes ne gagne pas.
 */
import { join } from 'node:path';
import { buildWasm, compare, findCompiler, loadWasm, root } from './wasm_harness.mjs';

const compiler = findCompiler();
if (!compiler) {
  console.log('· aucun compilateur C→wasm trouvé (zig / emcc / clang+wasm-ld) — bench sauté.');
  console.log('  installer : `pip install ziglang`');
  process.exit(0);
}

const variants = [
  { name: 'scalaire 1 ligne', defines: ['LM_NO_WASM_SIMD'] },
  { name: 'v128 1 ligne', defines: ['LM_WASM_1ROW'] },
  { name: 'v128 4 lignes', defines: [] },
];

const mods = [];
for (const v of variants) {
  const b = buildWasm({ defines: v.defines, compiler });
  const m = loadWasm(b.path);
  console.log(`· ${v.name.padEnd(16)} ${String(b.bytes).padStart(6)} o`);
  mods.push(m);
}

const base = mods[0];
const rows = [];
let fail = 0;
for (let i = 0; i < mods.length; i++) {
  const c = i === 0 ? { b: 0, ratio: 1, sigma: 0, identical: true } : compare(base, mods[i], { tokens: 120, rounds: 5 });
  const tps = i === 0 ? base.gen('Once upon a', 120, 0.0).tps : c.b;
  rows.push({ name: variants[i].name, tps, ratio: c.ratio, sigma: c.sigma, identical: c.identical });
  if (!c.identical) { console.error(`✗ ${variants[i].name} : sortie greedy différente du scalaire`); fail = 1; }
}
const c1 = compare(mods[1], mods[2], { tokens: 120, rounds: 5 });

console.log('\n| variante | tok/s | ×/scalaire | σ | sortie |');
console.log('|---|---|---|---|---|');
for (const r of rows) {
  console.log(`| ${r.name} | ${r.tps.toFixed(1)} | ×${r.ratio.toFixed(2)} | ${r.sigma.toFixed(1)} | ${r.identical ? 'identique' : 'divergente'} |`);
}
console.log(`\nblocage 4 lignes vs SIMD 1 ligne : ×${c1.ratio.toFixed(3)} (${c1.sigma.toFixed(1)}σ)`);

if (rows[2].ratio < 1.1) { console.error('✗ le blocage 4 lignes ne gagne pas sur le SIMD 1 ligne'); fail = 1; }
if (rows[2].tps < rows[1].tps) { console.error('✗ le blocage 4 lignes est plus lent que le SIMD 1 ligne'); fail = 1; }

if (fail) process.exit(1);
console.log(`\n✅ wasm : ×${rows[2].ratio.toFixed(1)} vs scalaire, sorties déterministes (${rows[2].tps.toFixed(0)} tok/s, module ${mods[2].size} o)`);
void join; void root;
