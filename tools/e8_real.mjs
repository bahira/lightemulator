// ============================================================================
//  e8_real.mjs — VRAI E8 (D8 ∪ D8+½, recherche coset + parité) vs Z^8 scalaire
//  à bits quasi-égaux, sur TOUS les poids réels du checkpoint.
//  Objectif : mesurer le gain réel et trancher le claim README (« +16% »).
// ============================================================================
import { readFileSync, statSync } from 'node:fs';

const EXPECTED = 143712;
const bytes = statSync('lm_c/checkpoint.bin').size;
let hdr = null, N = EXPECTED;
if (bytes === EXPECTED * 4) hdr = 0;
else if (bytes === EXPECTED * 4 + 4) hdr = 4;
else { N = readFileSync('lm_c/checkpoint.bin').readInt32LE(0); hdr = 4; }

const buf = readFileSync('lm_c/checkpoint.bin');
const W = new Float32Array(buf.buffer, buf.byteOffset + hdr, N);
console.log(`${bytes} octets · header=${hdr} · poids lus=${W.length} (attendu ${EXPECTED})${W.length !== EXPECTED ? '  ⚠ MISMATCH' : ''}`);

// Plus proche point de D8 pour y (décalé de -off) : arrondi + fix parité
function nearestCoset(blk, s, off) {
  const y = new Float64Array(8);
  for (let i = 0; i < 8; i++) y[i] = blk[i] / s - off;
  const z = Float64Array.from(y, Math.round);
  let sum = 0, d = 0;
  for (let i = 0; i < 8; i++) { sum += z[i]; const e = y[i] - z[i]; d += e * e; }
  if (((sum % 2) + 2) % 2 === 1) {
    let bi = 0;
    for (let i = 1; i < 8; i++) if (Math.abs(y[i] - z[i]) > Math.abs(y[bi] - z[bi])) bi = i;
    const e = y[bi] - z[bi];
    d -= e * e;
    const dz = e > 0 ? 1 : -1;
    z[bi] += dz;
    d += (e - dz) ** 2;
  }
  return { z, d };
}

let sseZ = 0, sseE = 0, n = 0, blocks = 0;
const blk = new Float64Array(8);
for (let o = 0; o + 8 <= W.length; o += 8) {
  let mx = 0;
  for (let i = 0; i < 8; i++) { blk[i] = W[o + i]; const a = Math.abs(blk[i]); if (a > mx) mx = a; }
  if (mx === 0) continue;                       // bloc nul : exact partout
  const s = mx / 15; blocks++;
  for (let i = 0; i < 8; i++) {                 // baseline Z^8 scalaire (31 niveaux)
    const q = Math.max(-15, Math.min(15, Math.round(blk[i] / s))) * s;
    sseZ += (blk[i] - q) ** 2; n++;
  }
  const A = nearestCoset(blk, s, 0), B = nearestCoset(blk, s, 0.5);
  const win = A.d <= B.d ? A : B;
  for (let i = 0; i < 8; i++)                   // reconstruction E8
    sseE += (blk[i] - (win.z[i] + (win === B ? 0.5 : 0)) * s) ** 2;
}

const rmsZ = Math.sqrt(sseZ / n), rmsE = Math.sqrt(sseE / n);
console.log(`blocs non nuls=${blocks} · poids=${n}`);
console.log(`RMS Z^8 = ${rmsZ.toExponential(4)} · RMS E8 = ${rmsE.toExponential(4)}`);
console.log(`GAIN MSE Z/E8 = ${(sseZ / sseE).toFixed(3)}×   (théorie moments G : 1.087)`);
console.log(`GAIN RMS Z/E8 = ${(rmsZ / rmsE).toFixed(3)}×   (théorie : 1.043)`);
console.log(`bits/poids : Z=5.000 · E8=5.125 (1 bit de coset par bloc de 8)`);

// --- Auto-vérification structurelle : la sortie doit être dans E8 scalé ---
{
  const t = [3.2, -1.7, 0.4, 2.9, -0.1, -2.2, 1.1, 0.9];
  const s = Math.max(...t.map(Math.abs)) / 15;
  const A = nearestCoset(t, s, 0), B = nearestCoset(t, s, 0.5);
  const win = A.d <= B.d ? A : B;
  const rec = [...win.z].map((v, i) => (v + (win === B ? 0.5 : 0)) * s);
  const units = rec.map(v => v / s);
  const allHalf = units.every(u => Math.abs(u - Math.round(u)) < 1e-9 || Math.abs(2 * u - Math.round(2 * u)) < 1e-9);
  const allInt = units.every(u => Math.abs(u - Math.round(u)) < 1e-9);
  const sumU = units.reduce((a, b) => a + b, 0);
  const okStruct = (allInt && Math.abs(sumU % 2) < 1e-9) || (!allInt && Math.abs(((sumU % 2) + 2) % 2 - 1) < 1e-9);
  console.log(`self-check E8 : cohérence entiers/demi-entiers=${allHalf} · parité somme=${okStruct} ${okStruct && allHalf ? 'OK' : '** ÉCHEC **'}`);
}
