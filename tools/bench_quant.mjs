/*
 * bench_quant.mjs — compile et lance bench/bench_quant.c (int8/int4 packés).
 * Échoue si l'erreur de quantification dépasse les bornes annoncées
 * (int8 < 2 %, int4 < 15 %) : le code de sortie du binaire porte le verdict.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const gcc = spawnSync('gcc', ['--version'], { encoding: 'utf8' });
if (gcc.status !== 0) {
  console.log('· gcc absent — bench quant sauté');
  process.exit(0);
}
const bin = join(mkdtempSync(join(tmpdir(), 'lm-q-')), 'bench_quant');
execFileSync('gcc', ['-O2', '-march=native', '-o', bin, join(root, 'bench/bench_quant.c'), '-lm'], { stdio: 'inherit' });
const r = spawnSync(bin, [], { stdio: 'inherit', env: process.env });
process.exit(r.status ?? 1);
