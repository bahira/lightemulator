// ============================================================================
//  optics.ts — Dispersion & biréfringence à forme fermée (audit corrigé).
//
//  Issu de la falsification du rapport "SPEAR-OPTICS" :
//    · Sellmeier silice : dn/dλ, d²n/dλ², d³n/dλ³ analytiques EXACTS
//      (vérifiés vs différences finies, rel ~1e-9 / 1e-6 / 7e-4).
//    · GDD : formule du rapport CORRECTE (36.2 fs²/mm @800nm, publié ~36.1).
//    · TOD : le rapport oubliait le facteur ×1000 (µm→mm) et le kernel C
//      codait tod=0 "identiquement nul" — FAUX. Corrigé ici (27.5 fs³/mm).
//    · Fresnel biaxial : forme fermée EXACTE (l'équation affichée dans le
//      rapport avait une coquille "= 1/n²" au lieu de "= 0").
//  Tout est re-vérifié dans validate.ts (tests 'opt-*').
// ============================================================================

// ---------------------------------------------------------------------------
//  1. Sellmeier 3 résonances — silice fondue @ 293 K
//     n²(λ) = 1 + Σ Bᵢ λ²/(λ² − Cᵢ),  λ en µm
// ---------------------------------------------------------------------------

export interface SellmeierGlass { name: string; B: [number, number, number]; C: [number, number, number] }

/** Coefficients standard (Malitson 1965), domaine 0.21–3.71 µm. */
export const FUSED_SILICA: SellmeierGlass = {
  name: 'silice fondue',
  B: [0.6961663, 0.4079426, 0.8974794],
  C: [0.00467914826, 0.0135120631, 97.9340025],
};

/** BK7 (Schott), domaine 0.3–2.5 µm. */
export const BK7: SellmeierGlass = {
  name: 'BK7',
  B: [1.03961212, 0.231792344, 1.01046945],
  C: [0.00600069867, 0.0200179144, 103.560653],
};

/** Fluorine de calcium CaF₂, domaine 0.15–12 µm. */
export const CAF2: SellmeierGlass = {
  name: 'CaF₂',
  B: [0.5675888, 0.4710914, 3.8484723],
  C: [0.00252642999, 0.0100783337, 1200.5566],
};

export interface SellmeierState { n: number; dn: number; d2n: number; d3n: number }

/** n et dérivées analytiques exactes en λ (µm). */
export function sellmeier(g: SellmeierGlass, lambda: number): SellmeierState {
  const l2 = lambda * lambda;
  let n2 = 1, s1 = 0, s2 = 0, s3 = 0;
  for (let i = 0; i < 3; i++) {
    const xi = 1 / (l2 - g.C[i]);
    const BC = g.B[i] * g.C[i];
    n2 += g.B[i] * l2 * xi;
    s1 += BC * xi * xi;
    s2 += BC * xi * xi * (4 * l2 * xi - 1);
    s3 += BC * xi * xi * xi * (12 * lambda - 24 * l2 * lambda * xi);
  }
  const n = Math.sqrt(n2);
  const dn = -(lambda / n) * s1;
  const d2n = (1 / n) * (s2 - dn * dn);
  const d3n = (1 / n) * (s3 - 3 * dn * d2n);
  return { n, dn, d2n, d3n };
}

// ---------------------------------------------------------------------------
//  2. Dispersion d'impulsion — GDD & TOD en unités physiques
//     β₂ = λ³/(2πc²)·d²n/dλ²  [fs²/mm]
//     β₃ = −λ⁴/(4π²c³)·(3·d²n/dλ² + λ·d³n/dλ³)  [fs³/mm]  (×1000 µm→mm inclus)
// ---------------------------------------------------------------------------

const C_UM_FS = 0.299792458; // vitesse de la lumière en µm/fs

/** GDD β₂(λ) en fs²/mm. */
export function gdd(g: SellmeierGlass, lambda: number): number {
  const l = lambda, l2 = l * l;
  return ((l2 * l) / (2 * Math.PI * C_UM_FS * C_UM_FS)) * sellmeier(g, lambda).d2n * 1000;
}

/** TOD β₃(λ) en fs³/mm — facteur ×1000 µm→mm CORRIGÉ (absent du rapport source). */
export function tod(g: SellmeierGlass, lambda: number): number {
  const l = lambda, l2 = l * l, s = sellmeier(g, lambda);
  return -((l2 * l2) / (4 * Math.PI * Math.PI * C_UM_FS ** 3)) * (3 * s.d2n + l * s.d3n) * 1000;
}

/** λ de dispersion nulle (β₂ = 0) par balayage + bissection, µm. */
export function zeroDispersionLambda(g: SellmeierGlass): number {
  let lo = 0.3, hi = 3.0;
  // β₂ passe de négatif (courtes λ) à positif : chercher le changement de signe
  if (gdd(g, lo) * gdd(g, hi) > 0) return NaN;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (gdd(g, lo) * gdd(g, mid) <= 0) hi = mid; else lo = mid;
  }
  return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
//  3. Élargissement d'impulsion gaussienne — régime GDD (analytique)
//     Une gaussienne transformée-limitée de durée τ₀ se répand en :
//     τ(L) = τ₀·√(1 + (4·ln2·β₂·L/τ₀²)²)   [définition FWHM]
//     Le TOD n'a pas de forme fermée simple : propagation spectrale
//     φ(ω) = β₂/2·ω² + β₃/6·ω³ + IFFT — exact, pas d'approx.
// ---------------------------------------------------------------------------

/** Durée FWHM après L mm, régime GDD pur (formule fermée). */
export function pulseWidthGDD(tau0fs: number, beta2: number, Lmm: number): number {
  const k = (4 * Math.LN2 * beta2 * Lmm) / (tau0fs * tau0fs);
  return tau0fs * Math.sqrt(1 + k * k);
}

export interface PulsePlot { t: Float64Array; I0: Float64Array; IL: Float64Array }

/**
 * Profil d'intensité |A(t)|² avant/après L mm, TOD inclus.
 * Méthode : spectre gaussien (σ_ω = 2√ln2/τ₀ — reconstruit exactement la
 * FWHM τ₀) → phase φ(ω) = β₂ω²/2 + β₃ω³/6 → FFT → recentrage circshift N/2
 * (le pic sort à l'indice 0, plié aux deux bouts de la fenêtre).
 */
export function pulsePropagation(
  tau0fs: number, beta2: number, beta3: number, Lmm: number, N = 2048,
): PulsePlot {
  // fenêtre temporelle : assez large pour contenir l'impulsion étalée
  const spread = Math.abs(beta2 * Lmm) / Math.max(tau0fs, 1);
  const T = Math.max(12 * tau0fs, 40 * spread + 8 * tau0fs);
  const dt = T / N;
  const df = 1 / T;
  const sigmaW = (2 * Math.sqrt(Math.LN2)) / tau0fs; // rad/fs
  const ampRe = new Float64Array(N), ampIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const w = (k < N / 2 ? k : k - N) * 2 * Math.PI * df;
    const s = w / sigmaW;
    const env = Math.exp(-(s * s) / 2);
    const phi = 0.5 * beta2 * Lmm * w * w + ((beta3 * Lmm) / 6) * w * w * w;
    ampRe[k] = env * Math.cos(phi);
    ampIm[k] = env * Math.sin(phi);
  }
  fftInPlace(ampRe, ampIm); // spectre hermitien → champ temporel réel
  // recentrage : pic de l'indice 0 vers l'indice N/2 (t = 0 au centre)
  const shRe = new Float64Array(N), shIm = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const src = (i + N / 2) % N;
    shRe[i] = ampRe[src]; shIm[i] = ampIm[src];
  }
  const t = new Float64Array(N), I0 = new Float64Array(N), IL = new Float64Array(N);
  let m0 = 0, mL = 0;
  for (let i = 0; i < N; i++) {
    t[i] = (i - N / 2) * dt;
    I0[i] = Math.exp(-2 * Math.LN2 * ((t[i] / tau0fs) ** 2)); // gaussienne FWHM τ₀
    IL[i] = shRe[i] * shRe[i] + shIm[i] * shIm[i];
    if (I0[i] > m0) m0 = I0[i];
    if (IL[i] > mL) mL = IL[i];
  }
  for (let i = 0; i < N; i++) { I0[i] /= m0 || 1; IL[i] /= mL || 1; }
  return { t, I0, IL };
}

/** FFT radix-2 itérative in-place (même convention que physics/fft.ts). */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  4. Fresnel biaxial — indices effectifs fermés (0 itération)
//     Équation correcte : Σ s_k²/(n²−n_k²) = 0  →  u² − B·u + C = 0, u = n²
//     (le rapport source affichait "= 1/n²" : coquille, la solution est celle
//     de l'équation "= 0", résidu machine vérifié sur 5000 directions.)
// ---------------------------------------------------------------------------

export interface BiaxialIndices { n1: number; n2: number; disc: number }

/** Indices effectifs des 2 modes pour une direction unitaire s et indices principaux (nx<ny<nz). */
export function fresnelBiaxial(s: readonly [number, number, number], nx: number, ny: number, nz: number): BiaxialIndices {
  const sx2 = s[0] * s[0], sy2 = s[1] * s[1], sz2 = s[2] * s[2];
  const nx2 = nx * nx, ny2 = ny * ny, nz2 = nz * nz;
  const B = sx2 * (ny2 + nz2) + sy2 * (nx2 + nz2) + sz2 * (nx2 + ny2);
  const C = sx2 * ny2 * nz2 + sy2 * nx2 * nz2 + sz2 * nx2 * ny2;
  const disc = B * B - 4 * C;
  const sq = Math.sqrt(Math.max(0, disc));
  return { n1: Math.sqrt(Math.max(0, (B - sq) / 2)), n2: Math.sqrt((B + sq) / 2), disc };
}

/** Biréfringence Δn = n₂ − n₁ sur la sphère des directions (grille θ,φ). */
export function bireringingMap(nx: number, ny: number, nz: number, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const theta = (Math.PI * r) / (rows - 1);
    for (let c = 0; c < cols; c++) {
      const phi = (2 * Math.PI * c) / cols;
      const s: [number, number, number] = [
        Math.sin(theta) * Math.cos(phi),
        Math.sin(theta) * Math.sin(phi),
        Math.cos(theta),
      ];
      const { n1, n2 } = fresnelBiaxial(s, nx, ny, nz);
      out[r * cols + c] = n2 - n1;
    }
  }
  return out;
}
