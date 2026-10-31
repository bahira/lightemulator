/*
 * build_wasm.mjs — reconstruit public/lm_infer.wasm depuis lm_c/lm_wasm.c.
 *
 *   node tools/build_wasm.mjs            # build à blanc : mesure et compare
 *   node tools/build_wasm.mjs --write    # écrit public/lm_infer.wasm
 *   node tools/build_wasm.mjs --no-simd  # témoin scalaire (A/B)
 *   node tools/build_wasm.mjs --1row     # témoin SIMD sans blocage 4 lignes
 *
 * Aucun compilateur wasm installé → message clair, code de sortie 0 : le dépôt
 * n'exige pas la toolchain pour se tester (le .wasm publié est versionné).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildWasm, compare, findCompiler, loadWasm, root, writeIfChanged } from './wasm_harness.mjs';

const argv = process.argv.slice(2);
const write = argv.includes('--write');
const defines = [];
if (argv.includes('--no-simd')) defines.push('LM_NO_WASM_SIMD');
if (argv.includes('--1row')) defines.push('LM_WASM_1ROW');

const compiler = findCompiler();
if (!compiler) {
  console.log('· aucun compilateur C→wasm trouvé (zig / emcc / clang+wasm-ld) — build sauté.');
  console.log('  installer : `pip install ziglang` (paquet pip, ~50 Mo) puis relancer.');
  process.exit(0);
}

const built = buildWasm({ defines, compiler });
console.log(`· compilateur : ${compiler.cmd}`);
console.log(`· ${built.bytes} octets  ${defines.length ? '(variante ' + defines.join(',') + ')' : '(SIMD v128 + blocage 4 lignes)'}`);

const pub = join(root, 'public/lm_infer.wasm');
let next, prev = null;
try {
  next = loadWasm(built.path);
} catch (e) {
  console.error(`✗ module produit inutilisable : ${e.message}`);
  process.exit(1);
}
try {
  prev = loadWasm(pub);
} catch { /* pas d'ancien module : on mesure quand même */ }

if (prev) {
  const c = compare(prev, next, { tokens: 120, rounds: 5 });
  console.log(`· publié ${prev.size} o · ${c.a.toFixed(1)} tok/s   →   nouveau ${next.size} o · ${c.b.toFixed(1)} tok/s`);
  console.log(`· nouveau / publié = ×${c.ratio.toFixed(3)} (${c.sigma.toFixed(1)}σ) — sorties greedy ${c.identical ? 'identiques' : 'DIFFÉRENTES'}`);
  if (!c.identical) {
    console.error('✗ le nouveau module change la génération greedy : refus d’écrire.');
    process.exit(1);
  }
} else {
  const o = next.gen('Once upon a', 120, 0.0);
  console.log(`· ${o.tps.toFixed(1)} tok/s (${o.n} tokens)`);
}

if (write) {
  const changed = writeIfChanged(pub, readFileSync(built.path));
  console.log(changed ? `✓ public/lm_infer.wasm mis à jour (${next.size} o)` : '· public/lm_infer.wasm déjà à jour');
  console.log('  → relancer `npm run docs:sync` pour publier la même version dans docs/');
}
