#!/usr/bin/env node
/*
 * Prépare les assets du lab « Inference edge » (le module wasm lit un vrai
 * checkpoint). Copie le checkpoint entraîné par lm_c/lm_main.c et le vocabulaire
 * dans public/ (dev + build) et docs/ (GitHub Pages).
 *
 *   node tools/prepare_inference_assets.mjs
 *
 * Le checkpoint est produit par :
 *   gcc -O2 -march=native -ffast-math -fopenmp -o lm_c/lm_train.exe lm_c/lm_main.c -lm
 *   lm_c/lm_train.exe --data ci-data/tinystories_ci.txt --steps 500 --nogen
 *
 * Contrat vérifié ici (c'est exactement ce que teste InferenceLab.tsx) :
 *   checkpoint = [int32 V][V-dim model weights as float32]
 *   vocab.bin  = V octets, dans l'ordre de première occurrence du corpus
 * donc V doit égaler la taille du fichier de vocabulaire, sinon le module
 * fut exponentiellement alloué et le lab affiche « checkpoint incompatible ».
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const ckPath = join(root, 'lm_c', 'checkpoint.bin');
if (!existsSync(ckPath)) {
  console.error('✗ lm_c/checkpoint.bin absent — entraîner d\'abord :');
  console.error('    gcc -O2 -march=native -ffast-math -fopenmp -o lm_c/lm_train.exe lm_c/lm_main.c -lm');
  console.error('    lm_c/lm_train.exe --data ci-data/tinystories_ci.txt --steps 500 --nogen');
  process.exit(1);
}

const ck = readFileSync(ckPath);
const V = ck.readInt32LE(0);
const nParams = (ck.length - 4) / 4;
if (!Number.isInteger(nParams) || nParams <= 0) {
  console.error(`✗ checkpoint illisible : ${ck.length} octets`);
  process.exit(1);
}

const vocabCandidates = [join(root, 'lm_c', 'vocab.bin'), join(root, 'public', 'vocab.bin')];
const vocabPath = vocabCandidates.find((p) => existsSync(p));
if (!vocabPath) {
  console.error('✗ vocabulaire introuvable (lm_c/vocab.bin ou public/vocab.bin)');
  process.exit(1);
}
const vocab = readFileSync(vocabPath);
if (vocab.length !== V) {
  console.error(`✗ V=${V} dans le checkpoint ≠ ${vocab.length} octets de vocabulaire (${vocabPath}).`);
  console.error('  Le checkpoint et le vocab doivent venir du MÊME corpus.');
  process.exit(1);
}

for (const dir of ['public', 'docs']) {
  writeFileSync(join(root, dir, 'lm_checkpoint.bin'), ck);
  writeFileSync(join(root, dir, 'lm_vocab.bin'), vocab);
}
console.log(`✓ assets inference synchronisés — V=${V}, ${nParams} params, ${(ck.length / 1024).toFixed(0)} kB (public/ + docs/)`);
