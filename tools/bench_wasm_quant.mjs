/*
 * bench_wasm_quant.mjs — débit du module wasm par mode de quantification.
 *
 * FP32 / INT8 packé / INT4 packé, même prompt, décodage glouton, instances
 * réutilisées, rondes alternées : chaque mode est mesuré 5 fois et on garde la
 * médiane (le JIT wasm n'est pas stable sur une seule passe).
 */
import { join } from 'node:path';
import { loadWasm, median, root } from './wasm_harness.mjs';

const mod = loadWasm(join(root, 'public/lm_infer.wasm'));
const ex = mod.ex;
const prompt = 'Once upon a';
const TOK = 160, ROUNDS = 5;

const gen = (mode) => {
  if (mode === 0) ex.wasm_refresh();
  else ex.wasm_quant(mode);
  ex.wasm_seed(42n);
  const scr = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
  for (let i = 0; i < prompt.length; i++) scr[128 + i] = prompt.charCodeAt(i) & 0xff;
  const t0 = performance.now();
  const n = ex.wasm_gen(ex.wasm_scratch() + 128, prompt.length, TOK, 0.0);
  const ms = performance.now() - t0;
  const txt = new TextDecoder('latin1').decode(new Uint8Array(ex.memory.buffer, ex.wasm_out(), n));
  const bytes = ex.wasm_mem_weights();
  return { tps: n / ms * 1000, txt, n, bytes };
};

const modes = [0, 2, 3];
const names = { 0: 'FP32', 2: 'INT8 packé', 3: 'INT4 packé' };
const samples = { 0: [], 2: [], 3: [] };
const info = {};

for (let r = 0; r < ROUNDS; r++) {
  for (const mode of modes) {
    const o = gen(mode);
    samples[mode].push(o.tps);
    info[mode] = { bytes: o.bytes, txt: o.txt };
  }
}

const base = median(samples[0]);
console.log('| mode | tok/s | ×/FP32 | poids | sortie gloutonne |');
console.log('|---|---|---|---|---|');
for (const mode of modes) {
  const t = median(samples[mode]);
  const kib = (info[mode].bytes / 1024).toFixed(1);
  const same = info[mode].txt === info[0].txt ? 'identique' : 'différente';
  console.log(`| ${names[mode]} | ${t.toFixed(0)} | ×${(t / base).toFixed(2)} | ${kib} KiB | ${same} |`);
}
const slow = median(samples[2]) < 0.4 * base || median(samples[3]) < 0.4 * base;
if (slow) {
  console.error('✗ un mode quantifié est >2,5× plus lent que FP32');
  process.exit(1);
}
console.log(`\n✅ quantification wasm : INT8 ×${(median(samples[2]) / base).toFixed(2)} et INT4 ×${(median(samples[3]) / base).toFixed(2)} vs FP32, mémoire ÷${(info[0].bytes / info[2].bytes).toFixed(1)}`);
