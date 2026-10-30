#!/usr/bin/env node
/*
 * Reconstruit la démo publiée sur GitHub Pages depuis les sources.
 * GitHub Pages sert le dossier docs/ → docs/demo.html est le build single-file.
 *
 *   npm run docs:sync          # écrit docs/
 *   npm run docs:check         # vérifie, n'écrit rien (CI de PR)
 */
import { copyFileSync, existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const check = process.argv.includes('--check');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist', 'index.html');
if (!existsSync(dist)) {
  console.error('✗ dist/index.html absent — lancer `vite build` d\'abord (npm run docs:sync le fait).');
  process.exit(1);
}

const files = ['index.html', 'lm_infer.wasm', 'lm_checkpoint.bin', 'lm_vocab.bin', 'vocab.bin'];
const stale = [];
for (const f of files) {
  const src = join(root, 'dist', f);
  if (!existsSync(src)) {
    console.warn(`· ${f} absent de dist/ — ignoré`);
    continue;
  }
  const dst = join(root, 'docs', f === 'index.html' ? 'demo.html' : f);
  const rel = `docs/${f === 'index.html' ? 'demo.html' : f}`;
  if (check) {
    const same = existsSync(dst) && readFileSync(dst).equals(readFileSync(src));
    if (!same) stale.push(rel);
    console.log(`  ${same ? '✓' : '✗'} ${rel}`);
    continue;
  }
  copyFileSync(src, dst);
  console.log(`  ${f} → ${rel} (${(statSync(dst).size / 1024).toFixed(0)} kB)`);
}
if (check) {
  if (stale.length) {
    console.error(`\n✗ ${stale.length} fichier(s) publié(s) périmé(s) : ${stale.join(', ')}`);
    console.error('  lancer `npm run docs:sync` et committer docs/');
    process.exit(1);
  }
  console.log('\n✅ docs/ est à jour avec les sources');
} else {
  console.log('✓ docs/ à jour (docs/index.html, la page de vérification, n\'est pas touchée)');
}
