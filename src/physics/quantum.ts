// ============================================================================
//  quantum.ts — Optique quantique discrète : Fock, HOM, MZI, CHSH.
//
//  Simulation EXACTE par algèbre linéaire sur les états à 1 et 2 photons
//  (pas de Monte-Carlo caché pour la mécanique quantique — les probabilités
//  viennent d'amplitudes calculées en fermé). Les références indépendantes :
//   · unitarité du séparateur de faisceau (U†U = I, conservation exacte)
//   · dip HOM fermé  P(1,1) = T² + R² − 2TR·x²  — à x=1 : (T−R)², nul à 50/50
//   · μ(τ) = ⟨f|f_τ⟩ par quadrature vs transformée de Gaussienne fermée
//   · CHSH quantique E(a,b) = cos 2(a−b) → S = 2√2, comparé à une simulation
//     de théorie à variables cachées locale (bornée par 2, mesuré)
//
//  TypeScript pur, zéro dépendance.
// ============================================================================

/** LCG déterministe — les simulations reproductibles ne tirent pas à la main. */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ---------------------------------------------------------------------------
//  Séparateur de faisceau — modes d'entrée (a1,a2) → sortie (b1,b2)
//  b1 = t·a1 + r·a2,  b2 = r·a1 + t·a2,  t = √T,  r = i·√R.
// ---------------------------------------------------------------------------

export function bsT(R: number): number { return Math.sqrt(1 - R); }
export function bsR(R: number): number { return Math.sqrt(R); }

/** Défaut d'unitarité max de la matrice 2×2 du BS : max |(U†U − I)_jk|. */
export function bsUnitarityDefect(R: number): number {
  const t = bsT(R), r = bsR(R);
  // U = [[t, i r],[i r, t]] → (U†U)00 = t² + r², (U†U)01 = −i·t·r + i·r·t = 0
  const e00 = Math.abs(t * t + r * r - 1);
  const e01 = Math.abs(-t * r + r * t);
  return Math.max(e00, e01);
}

// ---------------------------------------------------------------------------
//  Deux photons indiscernables |1,1⟩ → probabilités de sortie en fermé.
//   P(1,1) = T² + R² − 2TR·x²    (x = recouvrement des modes spectraux)
//   P(2,0) = P(0,2) = TR(1 + x²)
// ---------------------------------------------------------------------------

export function homP11(R: number, x2 = 1): number {
  const T = 1 - R;
  return T * T + R * R - 2 * T * R * x2;
}
export function homP20(R: number, x2 = 1): number {
  return (1 - R) * R * (1 + x2);
}

/** Somme des probabilités — doit valoir 1 (unitarité du BS sur 2 photons). */
export function homConservationDefect(R: number, x2 = 1): number {
  return Math.abs(homP11(R, x2) + 2 * homP20(R, x2) - 1);
}

/**
 * Chemin de calcul CROISé (x = 1) : on développe explicitement l'état de
 * sortie  (t c† + r d†)(r c† + t d†)|0⟩  avec t = √(1−R), r = i√R, puis on
 * normalise les Fock :  A(2,0) = √2·t·r,  A(1,1) = t² + r².
 * Aucune formule de variance réutilisée — uniquement l'arithmétique complexe.
 */
export function homFromAmplitudes(R: number): { p11: number; p20: number } {
  const t = bsT(R), r = bsR(R);
  const a11r = t * t - r * r, a11i = 0;                // t² + r², r² = (i√R)² = −R
  const a20 = Math.SQRT2 * t * r;                      // √2 t r (réel pur)
  return { p11: a11r * a11r + a11i * a11i, p20: a20 * a20 };
}

// --- spectra : recouvrement par quadrature vs forme fermée -----------------

const SIGMA = 1;            // rad/ns — largeur spectrale (unités internes)
const OMEGA_MAX = 7;        // troncature ±7σ
const OMEGA_N = 512;        // points de quadrature

/**
 * μ(τ) = ∫ f*(ω)·f(ω)·e^{−iωτ} dω / ∫|f|² dω par rectangles — doit
 * reconstruire la transformée de Fourier de la Gaussienne : |μ(τ)|² = e^{−(στ)²}.
 */
export function spectralOverlapMu(tau: number): number {
  const dw = (2 * OMEGA_MAX) / OMEGA_N;
  let re = 0, im = 0, norm = 0;
  for (let i = 0; i < OMEGA_N; i++) {
    const w = -OMEGA_MAX + (i + 0.5) * dw;
    const I = Math.exp(-(w * w) / (2 * SIGMA * SIGMA)); // |f|²
    norm += I * dw;
    re += I * Math.cos(w * tau) * dw;
    im += I * -Math.sin(w * tau) * dw;
  }
  return Math.hypot(re, im) / norm;
}

/** Dip HOM analytique : x² = |μ(τ)|² = e^{−(τ/τc)²}, P = ½(1 − x²). */
export function homGaussianP11(tauOverTc: number): number {
  return 0.5 * (1 - Math.exp(-(tauOverTc * tauOverTc)));
}

/** Écart quadrature ↔ forme fermée, max sur une grille de retards. */
export function homQuadratureError(): number {
  let err = 0;
  for (let i = 0; i <= 24; i++) {
    const tau = (i / 24) * 2.5;                       // τ/τc ∈ [0, 2.5], τc = 1/σ
    const mu = spectralOverlapMu(tau * SIGMA);
    const analytic = Math.exp(-(tau * tau));
    err = Math.max(err, Math.abs(mu * mu - analytic));
  }
  return err;
}

// ---------------------------------------------------------------------------
//  MZI photon unique — deux BS en séquence, phase φ sur un bras.
//  Fermé : P(port sombre) = sin²(φ/2). Opératoire : produit des matrices.
// ---------------------------------------------------------------------------

/** Probabilité du port « sombre » par produit matriciel 2×2 explicite. */
export function mziDarkFromMatrix(phi: number): number {
  const h = 1 / Math.SQRT2;                            // BS 50/50 : t = r = 1/√2
  // B·(x,y) avec B = [[t, i·r],[i·r, t]] — arithmétique complexe explicite
  const bs = (xr: number, xi: number, yr: number, yi: number) => ({
    pr: h * xr - h * yi, pi: h * xi + h * yr,
    qr: h * yr - h * xi, qi: h * xr + h * yi,
  });
  const s1 = bs(1, 0, 0, 0);                           // BS·(1,0) = (t, i·r)
  const cp = Math.cos(phi), sp = Math.sin(phi);
  const pr = s1.pr * cp - s1.pi * sp, pi = s1.pr * sp + s1.pi * cp; // bras phasé
  const s2 = bs(pr, pi, s1.qr, s1.qi);                 // second BS
  return s2.pr * s2.pr + s2.pi * s2.pi;                // port sombre
}

/** Écart max matrice ↔ forme fermée sin²(φ/2) sur [0, 2π]. */
export function mziClosedFormError(): number {
  let err = 0;
  for (let i = 0; i <= 64; i++) {
    const phi = (i / 64) * 2 * Math.PI;
    err = Math.max(err, Math.abs(mziDarkFromMatrix(phi) - Math.sin(phi / 2) ** 2));
  }
  return err;
}

// ---------------------------------------------------------------------------
//  CHSH — polarisation. |Φ+⟩ : E(a,b) = cos 2(a−b).
// ---------------------------------------------------------------------------

export function chshE(a: number, b: number): number {
  return Math.cos(2 * (a - b));
}

/** S = |E(a,b) − E(a,b') + E(a',b) + E(a',b')|. */
export function chshS(a: number, ap: number, b: number, bp: number): number {
  return Math.abs(chshE(a, b) - chshE(a, bp) + chshE(ap, b) + chshE(ap, bp));
}

/** Réglages canoniques a=0, a'=45°, b=22.5°, b'=67.5° → S = 2√2. */
export function chshStandardAngles(): { a: number; ap: number; b: number; bp: number } {
  return { a: 0, ap: Math.PI / 4, b: Math.PI / 8, bp: (3 * Math.PI) / 8 };
}

/**
 * Recherche exhaustive du max de S sur une grille fine de (a', b, b') avec
 * a = 0 (symétrie de rotation : E ne dépend que des différences d'angles).
 * Développement exact : E(a,b) = cos2a·cos2b + sin2a·sin2b — les tables
 * trigonométriques sont précalculées, la boucle interne est purement
 * algébrique (aucune approximation). Référence : le max doit retomber sur
 * la borne de Tsirelson 2√2 (atteinte à b−a=22.5°, b'−a=67.5°).
 */
export function chshGridMax(stepDeg = 0.5): { sMax: number; a: number; ap: number; b: number; bp: number } {
  const d2r = Math.PI / 180;
  const steps = Math.round(180 / stepDeg);
  // tables cos(2θ), sin(2θ) pour chaque angle de la grille
  const C = new Float64Array(steps), S = new Float64Array(steps);
  for (let i = 0; i < steps; i++) {
    const th = 2 * i * stepDeg * d2r;
    C[i] = Math.cos(th); S[i] = Math.sin(th);
  }
  let sMax = -1, best = { a: 0, ap: 0, b: 0, bp: 0 };
  for (let ia = 0; ia < steps; ia++) {
    const ca = C[ia], sa = S[ia];
    for (let ib = 0; ib < steps; ib++) {
      const cb = C[ib], sb = S[ib];
      const e3 = ca * cb + sa * sb;                  // E(a', b)
      for (let ic = 0; ic < steps; ic++) {
        // E(0,b) = cos2b = C[ib] ; E(0,b') = C[ic]
        const e4 = ca * C[ic] + sa * S[ic];          // E(a', b')
        const s = Math.abs(C[ib] - C[ic] + e3 + e4);
        if (s > sMax) {
          sMax = s;
          best = { a: 0, ap: ia * stepDeg * d2r, b: ib * stepDeg * d2r, bp: ic * stepDeg * d2r };
        }
      }
    }
  }
  return { sMax, ...best };
}

/**
 * Simulation de théorie à variables cachées LOCALE : chaque paire porte un
 * λ ∈ [0, π) commun ; chaque analyseur rend ±1 par Malus prédéterminé
 * A = sign(cos 2(θ − λ)). Aucune influence à distance → |S| ≤ 2 (Bell, 1964),
 * mesuré ici sur les mêmes réglages que la mécanique quantique.
 */
export function chshHiddenVariables(
  a: number, ap: number, b: number, bp: number, nPairs = 200_000, seed = 7,
): { S: number; E: number[] } {
  const rnd = lcg(seed);
  const corr = (theta1: number, theta2: number) => {
    let sum = 0;
    for (let i = 0; i < nPairs; i++) {
      const lambda = rnd() * Math.PI;
      const A = Math.cos(2 * (theta1 - lambda)) >= 0 ? 1 : -1;
      const B = Math.cos(2 * (theta2 - lambda)) >= 0 ? 1 : -1;
      sum += A * B;
    }
    return sum / nPairs;
  };
  const E1 = corr(a, b), E2 = corr(a, bp), E3 = corr(ap, b), E4 = corr(ap, bp);
  return { S: Math.abs(E1 - E2 + E3 + E4), E: [E1, E2, E3, E4] };
}
