// ============================================================================
//  holography.ts — Récupération de phase & holographie de Fourier.
//
//  Tout est calculé ici, rien n'est décoratif :
//   · FFT 2D construite sur la FFT radix-2 existante (lignes + colonnes)
//   · champ lointain de Fraunhofer = DFT de l'ouverture — vérifié contre le
//     noyau de Dirichlet, qui est la forme EXACTE de la DFT d'un rect discret
//   · Gerchberg–Saxton avec liberté d'amplitude hors fenêtre signal :
//     hologramme de phase pur, efficacité de diffraction et RMSE mesurés
//     à chaque itération (courbe de convergence réelle, pas un habit)
//
//  TypeScript pur, zéro dépendance. Copiez le fichier, il marche.
// ============================================================================

import { fft } from './fft';

// ---------------------------------------------------------------------------
//  FFT 2D — n×n, n puissance de 2. Buffers plats de longueur n*n.
// ---------------------------------------------------------------------------

// scratch en cache par taille — zéro allocation sur la boucle chaude
const scratch2d = new Map<number, { rowR: Float64Array; rowI: Float64Array; colR: Float64Array; colI: Float64Array }>();
function scratch2(n: number) {
  let s = scratch2d.get(n);
  if (!s) {
    s = { rowR: new Float64Array(n), rowI: new Float64Array(n), colR: new Float64Array(n), colI: new Float64Array(n) };
    scratch2d.set(n, s);
  }
  return s;
}

/** FFT 2D in-place (sens = e^{-i...}, comme fft.ts ; inverse avec échelle 1/n²). */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse = false): void {
  const s = scratch2(n);
  // lignes
  for (let y = 0; y < n; y++) {
    const off = y * n;
    s.rowR.set(re.subarray(off, off + n));
    s.rowI.set(im.subarray(off, off + n));
    fft(s.rowR, s.rowI, inverse);
    re.set(s.rowR, off);
    im.set(s.rowI, off);
  }
  // colonnes (copie → FFT → réécriture)
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { s.colR[y] = re[y * n + x]; s.colI[y] = im[y * n + x]; }
    fft(s.colR, s.colI, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = s.colR[y]; im[y * n + x] = s.colI[y]; }
  }
}

/** Cyclic shift (quadrant swap) — centre le spectre pour l'affichage. */
export function fftShift2d(src: Float64Array, n: number, dst?: Float64Array): Float64Array {
  const out = dst ?? new Float64Array(src.length);
  const h = n >> 1;
  for (let y = 0; y < n; y++) {
    const yy = (y + h) % n;
    for (let x = 0; x < n; x++) {
      out[yy * n + ((x + h) % n)] = src[y * n + x];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Fraunhofer — champ lointain = DFT de l'ouverture.
//  Référence indépendante : la DFT d'un rect discret w×h vaut EXACTEMENT le
//  noyau de Dirichlet séparable  |U[kx,ky]| = |D_w(kx)·D_h(ky)|,
//  D_w(k) = sin(πkw/n) / sin(πk/n).  (somme géométrique fermée)
// ---------------------------------------------------------------------------

/** Module de la DFT 2D d'une ouverture rectangulaire w×h (coin en (0,0)). */
export function farFieldMagnitude(n: number, w: number, h: number): Float64Array {
  const re = new Float64Array(n * n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) re[y * n + x] = 1;
  const im = new Float64Array(n * n);
  fft2d(re, im, n, false);
  const mag = new Float64Array(n * n);
  for (let i = 0; i < re.length; i++) mag[i] = Math.hypot(re[i], im[i]);
  return mag;
}

/** Noyau de Dirichlet : |D_w(k)|, forme fermée de la somme géométrique. */
export function dirichlet(k: number, w: number, n: number): number {
  const s = Math.sin(Math.PI * k / n);
  const p = Math.sin(Math.PI * k * w / n);
  return (Math.abs(s) < 1e-300) ? w : Math.abs(p / s);
}

/** Écart relatif max entre le champ lointain FFT et le Dirichlet fermé. */
export function fraunhoferDirichletError(n = 128, w = 12, h = 20): number {
  const mag = farFieldMagnitude(n, w, h);
  const peak = mag[0];
  let err = 0;
  for (let ky = 0; ky < n; ky++) {
    const dy = dirichlet(ky, h, n);
    for (let kx = 0; kx < n; kx++) {
      const ref = dirichlet(kx, w, n) * dy;
      err = Math.max(err, Math.abs(mag[ky * n + kx] - ref));
    }
  }
  return err / peak;
}

/**
 * Coupe 1D du motif de Fraunhofer d'une fente 1D, comparée au sinc² continu.
 * La DFT d'un rect discret est le noyau de Dirichlet ; il ne coïncide avec le
 * sinc² continu que dans la limite w ≪ n. On compare donc sur kw/n ≤ 1 où
 * l'écart discret/continu mesuré reste faible (facteur (π/w)/sin(π/w) au bord).
 */
export function fraunhoferSincCut(n = 128, w = 12, samples = 96): {
  k: Float64Array; measured: Float64Array; analytic: Float64Array; maxDiff: number;
} {
  const mag = farFieldMagnitude(n, w, 1);
  const kMax = Math.min(0.45 * n, n / w);
  const k = new Float64Array(samples);
  const measured = new Float64Array(samples);
  const analytic = new Float64Array(samples);
  let maxDiff = 0;
  for (let i = 0; i < samples; i++) {
    const kk = (i / (samples - 1)) * kMax;
    const kb = Math.min(Math.round(kk), n - 1);      // le spectre vit aux bins
    k[i] = kb;
    const m = mag[kb] / mag[0];                      // amplitude normalisée
    const u = (Math.PI * kb * w) / n;
    const a = u === 0 ? 1 : (Math.sin(u) / u) ** 2;  // sinc² continu
    measured[i] = m * m;                             // même quantité : intensité
    analytic[i] = a;
    maxDiff = Math.max(maxDiff, Math.abs(m * m - a));
  }
  return { k, measured, analytic, maxDiff };
}

// ---------------------------------------------------------------------------
//  Gerchberg–Saxton — hologramme de phase pur par liberté d'amplitude.
// ---------------------------------------------------------------------------

export interface GsTarget {
  /** Amplitude cible (0..1), longueur n*n. */
  amp: Float64Array;
  /** Masque signal (1 = pixel contraint), longueur n*n. */
  mask: Uint8Array;
  n: number;
}

export interface GsResult {
  /** Phase de l'hologramme (rad), longueur n*n — le livrable physique. */
  hologramPhase: Float64Array;
  /** Amplitude reconstruite dans le plan image (depuis l'hologramme à amplitude plate). */
  reconstruction: Float64Array;
  /** RMSE normalisé par itération (fenêtre signal). */
  rmseHistory: Float64Array;
  /** Efficacité de diffraction par itération (0..1). */
  effHistory: Float64Array;
  rmse: number;
  efficiency: number;
  iterations: number;
  /** Défaut de conservation : |Σ|image|² − Σ| Recon depuis hologramme |²|. */
  energyDefect: number;
}

/**
 * Normalise une cible : Σ_masque|amp|² = 1. Obligatoire — l'hologramme à
 * amplitude plate porte exactement n² de puissance, donc Parseval impose
 * Σ|image|² = 1. Une cible non normalisée rendrait l'efficacité mensongère.
 */
export function normalizeTarget(t: GsTarget): GsTarget {
  let p = 0;
  for (let i = 0; i < t.amp.length; i++) if (t.mask[i]) p += t.amp[i] * t.amp[i];
  const s = 1 / Math.sqrt(p || 1);
  const amp = new Float64Array(t.amp.length);
  for (let i = 0; i < t.amp.length; i++) amp[i] = t.amp[i] * s;
  return { amp, mask: t.mask, n: t.n };
}

/** Générateur déterministe (LCG) — phase initiale reproductible. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Boucle Gerchberg–Saxton standard :
 *   image ⇄ hologramme, amplitude imposée d'un côté (cible / plate),
 *   phase conservée. Hors fenêtre signal, l'amplitude est LIBRE —
 *   c'est ce qui rend l'efficacité mesurée (et non déclarée) intéressante.
 * La cible est normalisée en interne (Σ_masque amp² = 1) pour matcher la
 * puissance unitaire de l'hologramme : l'efficacité est alors directement
 * la fraction de puissance dans la fenêtre signal, bornée [0, 1].
 */
export function gerchbergSaxton(target: GsTarget, iterations = 40, seed = 42): GsResult {
  const norm = normalizeTarget(target);
  const { amp, mask, n } = norm;
  const N = n * n;

  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const rnd = lcg(seed);

  // init : cible × phase aléatoire déterministe
  for (let i = 0; i < N; i++) {
    const ph = rnd() * 2 * Math.PI;
    re[i] = amp[i] * Math.cos(ph);
    im[i] = amp[i] * Math.sin(ph);
  }

  const rmseHistory = new Float64Array(iterations);
  const effHistory = new Float64Array(iterations);
  const hologramPhase = new Float64Array(N);
  const reconstruction = new Float64Array(N);
  let energyDefect = 0;

  // cible normalisée → puissance signal = 1 ; le RMSE est déjà relatif
  const invSig = 1;

  const fR = new Float64Array(N);
  const fI = new Float64Array(N);
  const uR = new Float64Array(N);
  const uI = new Float64Array(N);

  for (let it = 0; it < iterations; it++) {
    // --- image → hologramme : amplitude plate, phase conservée
    fR.set(re); fI.set(im);
    fft2d(fR, fI, n, false);
    for (let i = 0; i < N; i++) {
      hologramPhase[i] = Math.atan2(fI[i], fR[i]);
      const m = Math.hypot(fR[i], fI[i]) || 1;
      fR[i] /= m; fI[i] /= m;                      // hologramme à amplitude 1
    }

    // --- hologramme → image : la reconstruction HONNÊTE (avant contrainte)
    uR.set(fR); uI.set(fI);
    fft2d(uR, uI, n, true);

    let se = 0, inPow = 0, totPow = 0;
    for (let i = 0; i < N; i++) {
      const a = Math.hypot(uR[i], uI[i]);
      reconstruction[i] = a;
      totPow += a * a;
      if (mask[i]) { const d = a - amp[i]; se += d * d; inPow += a * a; }
    }
    rmseHistory[it] = Math.sqrt(se) * invSig;
    effHistory[it] = inPow;

    // conservation de l'énergie (Parseval à travers les deux plans) :
    // Σ|hologramme|² = n² doit donner Σ|image|² = 1 — mesuré à chaque fin.
    if (it === iterations - 1) {
      let holo = 0;
      for (let i = 0; i < N; i++) holo += fR[i] * fR[i] + fI[i] * fI[i];
      energyDefect = Math.abs(totPow - holo / (n * n));
    }

    // --- contrainte image : cible imposée dans le masque, liberté ailleurs
    for (let i = 0; i < N; i++) {
      if (mask[i]) {
        const ph = Math.atan2(uI[i], uR[i]);
        re[i] = amp[i] * Math.cos(ph);
        im[i] = amp[i] * Math.sin(ph);
      } else {
        re[i] = uR[i]; im[i] = uI[i];
      }
    }
  }

  return {
    hologramPhase, reconstruction, rmseHistory, effHistory,
    rmse: rmseHistory[iterations - 1],
    efficiency: effHistory[iterations - 1],
    iterations, energyDefect,
  };
}

/**
 * Quantification de phase type SLM : L niveaux uniformes sur [−π, π).
 * Renvoie l'efficacité et le RMSE APRÈS quantification — la perte est
 * mesurée, pas supposée. Référence analytique classique : pour un réseau
 * blazé idéal, η(L)/η(∞) = (sin(π/L)/(π/L))² — un hologramme GS n'est pas
 * un réseau blazé, l'écart avec cette référence est affiché dans le lab.
 */
export function quantizePhase(
  target: GsTarget, hologramPhase: Float64Array, levels: number,
): { reconstruction: Float64Array; efficiency: number; rmse: number } {
  const { amp, mask, n } = normalizeTarget(target);
  const N = n * n;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const q = (2 * Math.PI) / levels;
  for (let i = 0; i < N; i++) {
    const p = Math.round(hologramPhase[i] / q) * q;
    re[i] = Math.cos(p); im[i] = Math.sin(p);
  }
  fft2d(re, im, n, true);
  let se = 0, inPow = 0;
  const rec = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const a = Math.hypot(re[i], im[i]);
    rec[i] = a;
    if (mask[i]) { const d = a - amp[i]; se += d * d; inPow += a * a; }
  }
  return { reconstruction: rec, efficiency: inPow, rmse: Math.sqrt(se) };
}

// ---------------------------------------------------------------------------
//  Cibles prédéfinies — déterministes, pour les tests et le lab.
// ---------------------------------------------------------------------------

/** Anneaux concentriques + croix centrale. */
export function ringsTarget(n: number): GsTarget {
  const amp = new Float64Array(n * n);
  const mask = new Uint8Array(n * n);
  const c = (n - 1) / 2;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const r = Math.hypot(x - c, y - c);
      const ring = Math.abs((r % 12) - 6) < 2.2 || Math.abs(r - 4) < 2.2;
      const cross = (Math.abs(x - c) < 1.6 || Math.abs(y - c) < 1.6) && r < 30;
      if (ring || cross) { amp[y * n + x] = 1; mask[y * n + x] = 1; }
    }
  }
  return { amp, mask, n };
}

/** Grille de points — stress-test classique des hologrammes de Fourier. */
export function dotsTarget(n: number): GsTarget {
  const amp = new Float64Array(n * n);
  const mask = new Uint8Array(n * n);
  const p = Math.max(8, Math.round(n / 8));
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (x % p === Math.floor(p / 2) && y % p === Math.floor(p / 2)) {
        amp[y * n + x] = 1; mask[y * n + x] = 1;
      }
    }
  }
  return { amp, mask, n };
}

/** Écart GS : amélioration du RMSE entre l'itération 1 et la fin. */
export function gsImprovementFactor(target: GsTarget, iterations = 40): number {
  const r = gerchbergSaxton(target, iterations);
  return r.rmseHistory[0] / Math.max(r.rmse, 1e-12);
}
