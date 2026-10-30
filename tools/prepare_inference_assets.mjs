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
const pubCkPath = join(root, 'public', 'lm_checkpoint.bin');
const pubVocabPath = join(root, 'public', 'lm_vocab.bin');

// Deux cas : un checkpoint vient d'être entraîné (lm_c/checkpoint.bin, ignoré par
// git) — on le publie ; sinon on republie les assets déjà versionnés dans public/
// (utile en CI, où gcc n'est pas forcément là pour réentraîner).
let ck;
let vocab;
let origin;

const vocabCandidates = [join(root, 'lm_c', 'vocab.bin'), pubVocabPath, join(root, 'public', 'vocab.bin')];
const vocabPath = vocabCandidates.find((p) => existsSync(p));
const localV = existsSync(ckPath) ? readFileSync(ckPath).readInt32LE(0) : -1;
const localOk = existsSync(ckPath) && vocabPath && localV === readFileSync(vocabPath).length;

if (localOk) {
  ck = readFileSync(ckPath);
  vocab = readFileSync(vocabPath);
  origin = 'lm_c/checkpoint.bin (fraîchement entraîné)';
} else if (existsSync(ckPath)) {
  // un checkpoint local existe mais a été entraîné sur un autre corpus (V différent) :
  // on le signale et on republie les assets versionnés plutôt que de casser le lab.
  console.warn(`· lm_c/checkpoint.bin ignoré (V=${localV} ≠ vocabulaire ${vocabPath ? readFileSync(vocabPath).length : '?'} octets)`);
  if (!existsSync(pubCkPath) || !existsSync(pubVocabPath)) {
    console.error('✗ aucun checkpoint compatible à publier');
    process.exit(1);
  }
  ck = readFileSync(pubCkPath);
  vocab = readFileSync(pubVocabPath);
  origin = 'public/ (assets versionnés)';
} else if (existsSync(pubCkPath) && existsSync(pubVocabPath)) {
  ck = readFileSync(pubCkPath);
  vocab = readFileSync(pubVocabPath);
  origin = 'public/ (assets déjà versionnés)';
} else {
  console.error('✗ aucun checkpoint : entraîner d\'abord');
  console.error('    gcc -O2 -march=native -ffast-math -fopenmp -o lm_c/lm_train.exe lm_c/lm_main.c -lm');
  console.error('    lm_c/lm_train.exe --data ci-data/tinystories_ci.txt --steps 500 --nogen');
  process.exit(1);
}

const V = ck.readInt32LE(0);
const nParams = (ck.length - 4) / 4;
if (!Number.isInteger(nParams) || nParams <= 0) {
  console.error(`✗ checkpoint illisible : ${ck.length} octets`);
  process.exit(1);
}
if (vocab.length !== V) {
  console.error(`✗ V=${V} dans le checkpoint ≠ ${vocab.length} octets de vocabulaire.`);
  console.error('  Le checkpoint et le vocab doivent venir du MÊME corpus.');
  process.exit(1);
}

for (const dir of ['public', 'docs']) {
  writeFileSync(join(root, dir, 'lm_checkpoint.bin'), ck);
  writeFileSync(join(root, dir, 'lm_vocab.bin'), vocab);
}
console.log(`✓ assets inference synchronisés depuis ${origin} — V=${V}, ${nParams} params, ${(ck.length / 1024).toFixed(0)} kB (public/ + docs/)`);
