#!/usr/bin/env node
/*
 * Vérifie le chemin d'inférence était réellement utilisable, hors navigateur —
 * exactement ce que fait src/labs/InferenceLab.tsx, avec le même shim WASI.
 *
 *   npm run verify:inference
 *
 * Contrôles (échec = exit 1) :
 *   1. les 3 assets existent et sont servables (wasm / checkpoint / vocab)
 *   2. le wasm s'instancie et wasm_init() alloue nParams > 0
 *   3. le checkpoint est cohérent : V == |vocab| et taille == 4 + 4·nParams
 *   4. wasm_gen() produit des tokens, et les 4 modes de quantification donnent
 *      des poids différents (la quantification n'est pas un no-op)
 *
 * Le checkpoint est le vrai modèle entraîné par lm_c/lm_main.c (V=104, 270k
 * params) ; régénérable avec `npm run assets:inference`.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => {
  if (!existsSync(p)) {
    console.error(`✗ asset manquant : ${p.replace(root + '/', '')} — lancer \`npm run assets:inference\``);
    process.exit(1);
  }
  return readFileSync(p);
};

let fail = 0;
const check = (ok, label, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) fail++;
};

// 1. assets
const wasmBuf = read(join(root, 'public', 'lm_infer.wasm'));
const ck = read(join(root, 'public', 'lm_checkpoint.bin'));
const vocab = read(join(root, 'public', 'lm_vocab.bin'));
check(WebAssembly.validate(wasmBuf), 'wasm valide', `${(wasmBuf.length / 1024).toFixed(1)} kB`);
check(vocab.length > 0 && vocab.length <= 256, 'vocabulaire plausible', 'V=' + vocab.length);

// 2. instanciation avec le même shim WASI que le lab
let ex = null;
const imports = {
  wasi_snapshot_preview1: {
    fd_write: (_fd, iovsPtr, iovsLen, nWrittenPtr) => {
      const dv = new DataView(ex.memory.buffer);
      let total = 0;
      for (let i = 0; i < iovsLen; i++) total += dv.getUint32(iovsPtr + i * 8 + 4, true);
      dv.setUint32(nWrittenPtr, total, true);
      return 0;
    },
  },
};
for (const n of ['args_get','args_sizes_get','environ_get','environ_sizes_get','clock_time_get','random_get','fd_seek','fd_close','fd_fdstat_get','fd_read','proc_exit','fd_prestat_get','fd_prestat_dir_name'])
  imports.wasi_snapshot_preview1[n] = () => 0;

const { instance } = await WebAssembly.instantiate(wasmBuf, imports);
ex = instance.exports;

const scratch = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
scratch.set(new Uint8Array(vocab), 0);
ex.wasm_vocab(ex.wasm_scratch(), vocab.length);

const nParams = ex.wasm_init();
const arena = ex.wasm_arena();
check(nParams > 0 && arena > 0, 'wasm_init', `${nParams} params · arena @${arena}`);

// 3. cohérence checkpoint ↔ vocabulaire ↔ dimensions du modèle
const dv = new DataView(ck.buffer, ck.byteOffset, ck.byteLength);
const V = dv.getInt32(0, true);
check(V === vocab.length, 'V du checkpoint == |vocab|', `V=${V}`);
check(4 + nParams * 4 === ck.length, 'taille du checkpoint', `${ck.length} octets`);

const mem = new Float32Array(ex.memory.buffer, arena, nParams);
for (let i = 0; i < nParams; i++) mem[i] = dv.getFloat32(4 + i * 4, true);
const pristine = new Float32Array(mem);
ex.wasm_refresh();

// 4. génération + quantification réelle
const l2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };
const base = l2(pristine, new Float32Array(pristine.length)) || 1;

const outs = [];
for (const [mode, label] of [[0, 'FP32'], [1, 'E8'], [2, 'INT8'], [3, 'INT4']]) {
  mem.set(pristine);
  if (mode) ex.wasm_quant(mode);
  else ex.wasm_refresh();
  const rel = mode ? l2(mem, pristine) / base : 0;
  const scr = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
  const prompt = 'Once upon a';
  for (let i = 0; i < prompt.length; i++) scr[128 + i] = prompt.charCodeAt(i) & 0xff;
  ex.wasm_seed(42n);
  const t0 = performance.now();
  const n = ex.wasm_gen(ex.wasm_scratch() + 128, prompt.length, 120, 0.8);
  const ms = performance.now() - t0;
  const txt = new TextDecoder('latin1').decode(new Uint8Array(ex.memory.buffer, ex.wasm_out(), n));
  outs.push(txt);
  check(n > 0, `génération ${label}`, `${n} tokens · ${(n / ms * 1000).toFixed(0)} tok/s · ‖Δw‖ ${(rel * 100).toFixed(2)} %`);
  if (mode) check(rel > 1e-4, `  quantification ${label} effective`, `‖Δw‖/‖w‖ = ${(rel * 100).toFixed(2)} %`);
}
check(outs[0] !== outs[3], 'INT4 change la sortie vs FP32', outs[0] !== outs[3] ? '(divergence attendue)' : '(identique — suspect)');

mem.set(pristine);
ex.wasm_refresh();

if (fail) {
  console.error(`\n✗ ${fail} vérification(s) en échec`);
  process.exit(1);
}
console.log('\n✅ chaîne d\'inférence wasm opérationnelle (assets, dimensions, quantification)');
