#!/usr/bin/env node
/*
 * Reconstruit la démo publiée sur GitHub Pages depuis les sources.
 * GitHub Pages sert le dossier docs/ → docs/demo.html est le build single-file.
 *
 *   npm run docs:sync
 */
import { copyFileSync, existsSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist', 'index.html');
if (!existsSync(dist)) {
  console.error('✗ dist/index.html absent — lancer `vite build` d\'abord (npm run docs:sync le fait).');
  process.exit(1);
}

const files = ['index.html', 'lm_infer.wasm', 'lm_checkpoint.bin', 'lm_vocab.bin', 'vocab.bin'];
for (const f of files) {
  const src = join(root, 'dist', f);
  if (!existsSync(src)) {
    console.warn(`· ${f} absent de dist/ — ignoré`);
    continue;
  }
  const dst = join(root, 'docs', f === 'index.html' ? 'demo.html' : f);
  copyFileSync(src, dst);
  console.log(`  ${f} → docs/${f === 'index.html' ? 'demo.html' : f} (${(statSync(dst).size / 1024).toFixed(0)} kB)`);
}
console.log('✓ docs/ à jour (docs/index.html, la page de vérification, n\'est pas touchée)');
