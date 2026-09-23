#!/usr/bin/env node
/* ============================================================================
 * ti_corpus.mjs — Assemble le corpus d'entraînement SPEAR-T1 (déterministe).
 *
 * Objectif : un corpus reproductible, vérifié par empreinte SHA-256, jamais
 * committé (data/ est gitignoré — même politique que le reste du repo).
 *
 *   node tools/ti_corpus.mjs             # assemble data/gutenberg.txt + data/shakespeare.txt
 *   node tools/ti_corpus.mjs --check     # vérifie les empreintes sans réseau
 *
 * Sources (domaine public, archives publiques) :
 *   - Gutenberg (NLTK nltk_data / packages / corpora / gutenberg.zip, 18 livres)
 *   - TinyShakespeare (karpathy/char-rnn data/tinyshakespeare/input.txt)
 *
 * Encode : latin-1 → utf-8 non, on garde l'octet brut (LM byte-level : le
 * corpus EST une suite d'octets, aucune normalisation cachée).
 * ========================================================================== */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { join, basename } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const DATA = join(ROOT, 'data');

/* Empreintes de référence — mesurées une fois, puis vérifiées à chaque build. */
const EXPECT = {
  'shakespeare.txt': '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed',
};

/* Gutenberg : l'ordre de concaténation est trié (déterministe). */
const GUT_ORDER = [
  'austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
  'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt',
  'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt',
  'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt',
  'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt',
  'whitman-leaves.txt',
];

function sha256(buf) { return createHash('sha256').update(buf).digest('hex'); }

function tryCmd(cmd, args) {
  try { return execFileSync(cmd, args, { maxBuffer: 1 << 30 }); } catch { return null; }
}

/* Téléchargement : curl d'abord (portable), puis `gh api` (sandbox Arena). */
function fetch(url, ghArgs) {
  const direct = tryCmd('curl', ['-sSL', '--max-time', '180', '-o', '-', url]);
  if (direct && direct.length > 0) return direct;
  const viaGh = tryCmd('gh', ghArgs);
  if (viaGh && viaGh.length > 0) return viaGh;
  return null;
}

function assembleGutenberg() {
  let zip = null;
  if (existsSync('/tmp/gutenberg.zip')) zip = readFileSync('/tmp/gutenberg.zip');
  else zip = fetch('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip',
                   ['api', '-H', 'Accept: application/vnd.github.raw',
                    'repos/nltk/nltk_data/contents/packages/corpora/gutenberg.zip']);
  if (!zip) { console.error('  gutenberg.zip introuvable (réseau)'); return null; }
  writeFileSync('/tmp/.ti_gutenberg.zip', zip);
  /* Décompression via node:zlib — zip classique, entrées store/deflate. */
  const files = readZip(zip);
  const parts = [];
  for (const name of GUT_ORDER) {
    const e = files.find((f) => f.name === `gutenberg/${name}`);
    if (!e) { console.error(`  manquant dans le zip : ${name}`); return null; }
    parts.push(e.buf);
  }
  return Buffer.concat(parts);
}

/* Décodeur ZIP minimal (stored + deflate) — zéro dépendance. */
import zlib from 'node:zlib';
function readZip(buf) {
  const out = [];
  for (let i = 0; i < buf.length - 4; i++) {
    if (buf.readUInt32LE(i) !== 0x04034b50) continue;
    const method = buf.readUInt16LE(i + 8);
    const csize = buf.readUInt32LE(i + 18);
    const usize = buf.readUInt32LE(i + 22);
    const nlen = buf.readUInt16LE(i + 26);
    const elen = buf.readUInt16LE(i + 28);
    const name = buf.toString('latin1', i + 30, i + 30 + nlen);
    const start = i + 30 + nlen + elen;
    const raw = buf.subarray(start, start + csize);
    if (name.endsWith('.txt')) {
      const data = method === 0 ? Buffer.from(raw) : zlib.inflateRawSync(raw, { maxOutputLength: usize + 1024 });
      out.push({ name, buf: data });
    }
    i = start + csize - 1;
  }
  return out;
}

function main() {
  const check = process.argv.includes('--check');
  mkdirSync(DATA, { recursive: true });
  let ok = true;

  /* --- Shakespeare ------------------------------------------------------- */
  const shPath = join(DATA, 'shakespeare.txt');
  const sh = fetch('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                   ['api', '-H', 'Accept: application/vnd.github.raw',
                    'repos/karpathy/char-rnn/contents/data/tinyshakespeare/input.txt']);
  if (sh) writeFileSync(shPath, sh);
  if (!existsSync(shPath)) { console.error('shakespeare.txt absent'); ok = false; }
  else {
    const h = sha256(readFileSync(shPath));
    const good = EXPECT['shakespeare.txt'] === h;
    console.log(`shakespeare.txt  ${readFileSync(shPath).length} o  sha256=${h.slice(0, 16)}…  ${good ? 'OK' : 'MISMATCH'}`);
    if (!good) ok = false;
  }

  /* --- Gutenberg --------------------------------------------------------- */
  const guPath = join(DATA, 'gutenberg.txt');
  if (!check) {
    const buf = assembleGutenberg();
    if (buf) writeFileSync(guPath, buf);
  }
  if (!existsSync(guPath)) { console.error('gutenberg.txt absent'); ok = false; }
  else {
    const b = readFileSync(guPath);
    console.log(`gutenberg.txt    ${b.length} o  sha256=${sha256(b).slice(0, 16)}…`);
    const bytes = new Set(b); 
    console.log(`  octets distincts : ${bytes.size}  |  1er octet : 0x${b[0].toString(16)}`);
  }
  process.exit(ok ? 0 : 1);
}
main();
