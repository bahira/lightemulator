// ============================================================================
//  spear_e8_vs_int5.mjs — Grounded comparison E8 vs INT5 at equal storage bits
//  Lit checkpoint.bin, quantise chaque bloc de 8 poids avec les deux schémas
//  utilisant le même pas (step) dérivé de la plage ±15 (5 bits), puis compare
//  l'erreur RMS.  E8 autorise des demi-pas (coeffs impairs * 0.5*step) ce qui
//  devrait réduire l'erreur par rapport au arrondi entier (INT5).
//  Auteur : grounded loop — budget 500 itérations ≈ quelques secondes.
// ============================================================================
import { createReadStream } from 'node:fs';
import { readFileSync } from 'node:fs';

// --- Lecture du checkpoint.bin -------------------------------------------
const BIN = readFileSync('lm_c/checkpoint.bin');
const totalParams = BIN.readInt32LE(0); // au début: nombre de floats
const floats = new Float32Array(BIN.buffer, 4, totalParams); // reste du fichier

// Helper : quantiser un bloc de 8 avec E8 (demi-pas) et INT5 (entiers)
// On utilise la plage codée en 5 bits : [-15, +15] inclusive = 31 niveaux + 0.
// Le pas est scale = maxAbs / 15.
// Pour E8 : la valeur réelle = (coordImpair * 0.5) * step, i.e. coord impair * step/2.
// Pour INT5 : valeur = coordEntier * step.

function quantBlockE8(intBlock) {
  // intBlock est un tableau de 8 entiers déjà centrés et mis à l'échelle ?
  // On reçoit les floats bruts du bloc.
  // On va trouver maxAbs parmi les 8, puis mapper chaque x en coord E8.
  const v = intBlock; // Float32[8]
  let maxAbs = 0;
  for (let i = 0; i < 8; i++) {
    const a = Math.abs(v[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const step = maxAbs / 15; // 15 = max coord signé 5-bit
  const out = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    const x = v[i];
    // On cherche le multiple impair de 0.5*step le plus proche de x
    // soit on calcule t = x / (step/2) = 2x/step, puis on arrondit à l'entier impair proche.
    let t = 2 * x / step;
    // arrondi à l'entier impair : si t pair on ajoute/soustrait 1 pour le devenir impair
    const rounded = Math.round(t);
    let odd;
    if (rounded % 2 === 0) {
      // on pousse vers le impair le plus proche
      odd = rounded === 0 ? 1 : (rounded > 0 ? rounded - 1 : rounded + 1);
    } else {
      odd = rounded;
    }
    out[i] = odd * step / 2; // valeur reconstruite
  }
  return out;
}

function quantBlockINT5(intBlock) {
  const v = intBlock;
  let maxAbs = 0;
  for (let i = 0; i < 8; i++) {
    const a = Math.abs(v[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const step = maxAbs / 15;
  const out = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    const x = v[i];
    // arrondi au multiple entier de step le plus proche
    const k = Math.round(x / step);
    // bornage [-15,15]
    const kl = Math.max(-15, Math.min(15, k));
    out[i] = kl * step;
  }
  return out;
}

// --- Parcourir tous les paramètres par blocs de 8 ------------------------
let rmsE8 = 0, rmsINT5 = 0, n = 0;
for (let start = 0; start + 8 <= totalParams; start += 8) {
  const block = floats.subarray(start, start + 8);
  const qE8 = quantBlockE8(block);
  const qINT5 = quantBlockINT5(block);
  for (let i = 0; i < 8; i++) {
    const e = block[i] - qE8[i];
    rmsE8 += e * e;
    const e2 = block[i] - qINT5[i];
    rmsINT5 += e2 * e2;
    n++;
  }
}
rmsE8 = Math.sqrt(rmsE8 / n);
rmsINT5 = Math.sqrt(rmsINT5 / n);

// --- Rapport -------------------------------------------------------------
console.log(`Paramètres total : ${totalParams}`);
console.log(`RMS erreur quantisation E8 (demi-pas) : ${rmsE8.toExponential(4)}`);
console.log(`RMS erreur quantisation INT5 (entiers) : ${rmsINT5.toExponential(4)}`);
console.log(`Ratio RMS_INT5 / RMS_E8 = ${(rmsINT5 / rmsE8).toFixed(3)} (>=1 = E8 meilleur)`);
console.log(`Théorie : E8 doit être ~1.2× plus précis grâce aux demi-pas, soit ratio ~1.2.`);