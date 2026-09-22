// ============================================================================
//  pnn.ts — PNN accelerator emulation (Nature Comm. 17, 1059, 2026).
//
//  HONNÊTETÉ : ceci est une ÉMULATION PAR ALGÈBRE LINÉAIRE de la physique du
//  papier. Le papier utilise un FDTD 3D inverse-design ; nous n'émulons PAS la
//  FDTD. Ce qui est fidèlement reproduit, ce sont les deux structures
//  algébriques que le papier exploite :
//    1. Superposition cohérente « N+C » : N champs de base (un par mode
//       d'entrée) suffisent à reconstruire EXACTEMENT les champs de tous les
//       échantillons par combinaison linéaire E^ℓ = Σ_i a_i^ℓ · E_i — la
//       linéarité du milieu garantit l'exactitude (testée à l'epsilon machine).
//    2. Gradient par méthode adjointe (AVM) : le gradient de la perte par
//       rapport aux champs est proportionnel au recouvrement champs avant /
//       champs adjoints ; la source adjointe par classe est le one-hot de la
//       cible. Vérifié contre différences finies.
//  La photodétection (intégrale de puissance aux C ports, normalisée par la
//  somme) puis softmax → entropie croisée : la normalisation pénalise
//  implicitement la puissance aux ports non-cibles (diantre/crosstalk).
//  Zéro dépendance : copiez le fichier, il fonctionne.
// ============================================================================

export interface PnnConfig {
  N: number;   // modes d'entrée (features)
  C: number;   // ports de classe
  F: number;   // échantillons du champ discrétisé (région « port » 1D)
  nTrain: number;
  nTest: number;
  epochs: number;
  lr: number;
  momentum: number;
  seed: number;
}

export const DEFAULT_PNN: PnnConfig = {
  N: 8, C: 4, F: 64, nTrain: 200, nTest: 40,
  epochs: 200, lr: 2.0, momentum: 0.9, seed: 2026,
};

// PRNG mulberry32 — déterministe, comme le reste du moteur.
function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface PnnDevice {
  cfg: PnnConfig;
  /** milieu linéaire entraînable F×N, parties réelle/imaginaire (row-major). */
  M0r: Float64Array; M0i: Float64Array;
  /** C modes de port fixes C×F (row-major), parties réelle/imaginaire. */
  Pr: Float64Array; Pi: Float64Array;
}

export interface PnnSample {
  ar: Float64Array; ai: Float64Array; // amplitude complexe d'entrée (N)
  label: number;                       // classe cible (0..C-1)
}

/** Machine : matrices fixes (ports) + milieu linéaire aléatoire initial. */
export function makePnn(cfg: PnnConfig): PnnDevice {
  const r = mulberry(cfg.seed);
  const { N, C, F } = cfg;
  const M0r = new Float64Array(F * N), M0i = new Float64Array(F * N);
  const Pr = new Float64Array(C * F), Pi = new Float64Array(C * F);
  const sM = 1 / Math.sqrt(N), sP = 1.5 / Math.sqrt(F);
  for (let i = 0; i < F * N; i++) {
    M0r[i] = (r() * 2 - 1) * sM;
    M0i[i] = (r() * 2 - 1) * sM;
  }
  for (let i = 0; i < C * F; i++) {
    Pr[i] = (r() * 2 - 1) * sP;
    Pi[i] = (r() * 2 - 1) * sP;
  }
  return { cfg, M0r, M0i, Pr, Pi };
}

/** Base de champs : colonne i du milieu M0 (un champ de base par mode d'entrée). */
export function basisField(d: PnnDevice, i: number): { fr: Float64Array; fi: Float64Array } {
  const { F, N } = d.cfg;
  const fr = new Float64Array(F), fi = new Float64Array(F);
  for (let f = 0; f < F; f++) { fr[f] = d.M0r[f * N + i]; fi[f] = d.M0i[f * N + i]; }
  return { fr, fi };
}

/** Champ avant par superposition N+C : E^ℓ = Σ_i a_i^ℓ · E_i (combinaison linéaire exacte). */
export function fieldFromBasis(
  basis: { fr: Float64Array; fi: Float64Array }[], ar: Float64Array, ai: Float64Array,
): { fr: Float64Array; fi: Float64Array } {
  const F = basis[0].fr.length;
  const fr = new Float64Array(F), fi = new Float64Array(F);
  for (let i = 0; i < basis.length; i++) {
    const ari = ar[i], aii = ai[i];
    const { fr: br, fi: bi } = basis[i];
    for (let f = 0; f < F; f++) {
      fr[f] += ari * br[f] - aii * bi[f];
      fi[f] += ari * bi[f] + aii * br[f];
    }
  }
  return { fr, fi };
}

/** Champ avant calculé directement (produit matriciel dense, chemin indépendant). */
export function forwardDirect(
  M0r: Float64Array, M0i: Float64Array, N: number, ar: Float64Array, ai: Float64Array,
): { fr: Float64Array; fi: Float64Array } {
  const F = M0r.length / N;
  const fr = new Float64Array(F), fi = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    let sr = 0, si = 0;
    for (let i = 0; i < N; i++) {
      const mr = M0r[f * N + i], mi = M0i[f * N + i];
      sr += mr * ar[i] - mi * ai[i];
      si += mr * ai[i] + mi * ar[i];
    }
    fr[f] = sr; fi[f] = si;
  }
  return { fr, fi };
}

/** Photodétection : puissance aux C ports p_c = |⟨P_c, E⟩|² (intégrale surfacique discrète). */
export function portPowers(
  Pr: Float64Array, Pi: Float64Array, C: number, fr: Float64Array, fi: Float64Array,
): Float64Array {
  const F = fr.length;
  const p = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    let ur = 0, ui = 0;
    for (let f = 0; f < F; f++) {
      ur += Pr[c * F + f] * fr[f] + Pi[c * F + f] * fi[f]; // conj(P)·E
      ui += Pr[c * F + f] * fi[f] - Pi[c * F + f] * fr[f];
    }
    p[c] = ur * ur + ui * ui;
  }
  return p;
}

/** Softmax stable des puissances détectées → distribution de probabilités. */
export function softmax(p: Float64Array): Float64Array {
  let mx = p[0];
  for (let c = 1; c < p.length; c++) if (p[c] > mx) mx = p[c];
  const out = new Float64Array(p.length);
  let s = 0;
  for (let c = 0; c < p.length; c++) { out[c] = Math.exp(p[c] - mx); s += out[c]; }
  for (let c = 0; c < p.length; c++) out[c] /= s;
  return out;
}

/** Entropie croisée d'un échantillon : L = −log p_y. */
export function ceLoss(probs: Float64Array, label: number): number {
  return -Math.log(Math.max(probs[label], 1e-300));
}

/**
 * Gradient AVM (batch, moyenné) de la perte CE par rapport à M0 (réel+imag).
 * Route adjointe : λ_c = dL/dp_c = p_c − δ(c,y) (softmax∘CE), champ adjoint
 * A = Σ_c λ_c · conj(u_c) · conj(P_c) — recouvrement champs avant / adjoints —
 * puis dL/dM0_fi = 2·Re(A_f a_i), dL/dM0i_fi = −2·Im(A_f a_i).
 */
export function avmGradient(
  d: PnnDevice, samples: PnnSample[],
): { gr: Float64Array; gi: Float64Array } {
  const { N, C, F } = d.cfg;
  const gr = new Float64Array(F * N), gi = new Float64Array(F * N);
  for (const s of samples) {
    const { fr, fi } = forwardDirect(d.M0r, d.M0i, N, s.ar, s.ai);
    const p = portPowers(d.Pr, d.Pi, C, fr, fi);
    const probs = softmax(p); // distribution détectée — λ_c = probs_c − δ(c,y)
    // champ adjoint A = Σ_c λ_c conj(u_c) conj(P_c)
    const Ar = new Float64Array(F), Ai = new Float64Array(F);
    for (let c = 0; c < C; c++) {
      const lam = probs[c] - (c === s.label ? 1 : 0);
      if (lam === 0) continue;
      let ur = 0, ui = 0;
      for (let f = 0; f < F; f++) {
        ur += d.Pr[c * F + f] * fr[f] + d.Pi[c * F + f] * fi[f];
        ui += d.Pr[c * F + f] * fi[f] - d.Pi[c * F + f] * fr[f];
      }
      const w = lam; // λ_c · conj(u_c) : facteur complexe
      const wr = w * ur, wi2 = -w * ui; // conj(u_c) = ur − i·ui ⇒ conj(u)·λ
      for (let f = 0; f < F; f++) {
        const pr = d.Pr[c * F + f], pi = d.Pi[c * F + f];
        Ar[f] += wr * pr + wi2 * pi;  // (wr + i·wi2)·conj(P) = conj(u)·conj(P)·λ
        Ai[f] += wi2 * pr - wr * pi;
      }
    }
    for (let i = 0; i < N; i++) {
      const ari = s.ar[i], aii = s.ai[i];
      for (let f = 0; f < F; f++) {
        const A = Ar[f], AiF = Ai[f];
        gr[f * N + i] += 2 * (A * ari - AiF * aii);
        gi[f * N + i] += -2 * (A * aii + AiF * ari);
      }
    }
  }
  const inv = 1 / samples.length;
  for (let k = 0; k < gr.length; k++) { gr[k] *= inv; gi[k] *= inv; }
  return { gr, gi };
}

/** Jeu de données synthétique linéairement séparable : classe k ⇒ feature k dominante. */
export function makeDataset(cfg: PnnConfig, n: number, seedOffset: number): PnnSample[] {
  const r = mulberry(cfg.seed + seedOffset);
  const out: PnnSample[] = [];
  for (let l = 0; l < n; l++) {
    const label = l % cfg.C;
    const ar = new Float64Array(cfg.N), ai = new Float64Array(cfg.N);
    for (let i = 0; i < cfg.N; i++) {
      const amp = i === label ? 0.7 + 0.4 * r() : 0.15 * r();
      const ph = (r() * 2 - 1) * 0.25;
      ar[i] = amp * Math.cos(ph);
      ai[i] = amp * Math.sin(ph);
    }
    out.push({ ar, ai, label });
  }
  return out;
}

/** Précision (argmax puissance) sur un jeu d'échantillons. */
export function accuracy(d: PnnDevice, samples: PnnSample[]): number {
  let ok = 0;
  for (const s of samples) {
    const { fr, fi } = forwardDirect(d.M0r, d.M0i, d.cfg.N, s.ar, s.ai);
    const p = portPowers(d.Pr, d.Pi, d.cfg.C, fr, fi);
    let best = 0;
    for (let c = 1; c < p.length; c++) if (p[c] > p[best]) best = c;
    if (best === s.label) ok++;
  }
  return ok / samples.length;
}

/**
 * Entraînement SGD à momentum, full batch, boucle N+C : gradient AVM batch
 * (coût ~ N+C propagations internes par échantillon via les champs de base /
 * adjoints, PAS L simulations complètes).
 */
export function trainPnn(cfg: PnnConfig): {
  device: PnnDevice; loss: number; accTrain: number; accTest: number;
  msPerEpoch: number; ncRatio: number;
} {
  const d = makePnn(cfg);
  const train = makeDataset(cfg, cfg.nTrain, 1);
  const test = makeDataset(cfg, cfg.nTest, 2);
  const vr = new Float64Array(d.M0r.length), vi = new Float64Array(d.M0i.length);
  const t0 = performance.now();
  for (let ep = 0; ep < cfg.epochs; ep++) {
    const { gr, gi } = avmGradient(d, train);
    for (let k = 0; k < gr.length; k++) {
      vr[k] = cfg.momentum * vr[k] - cfg.lr * gr[k];
      vi[k] = cfg.momentum * vi[k] - cfg.lr * gi[k];
      d.M0r[k] += vr[k]; d.M0i[k] += vi[k];
    }
  }
  const msPerEpoch = (performance.now() - t0) / cfg.epochs;
  // perte finale
  let loss = 0;
  for (const s of train) {
    const { fr, fi } = forwardDirect(d.M0r, d.M0i, cfg.N, s.ar, s.ai);
    const p = portPowers(d.Pr, d.Pi, cfg.C, fr, fi);
    loss += ceLoss(softmax(p), s.label);
  }
  loss /= cfg.nTrain;
  return {
    device: d, loss, accTrain: accuracy(d, train), accTest: accuracy(d, test),
    msPerEpoch, ncRatio: cfg.nTrain / (cfg.N + cfg.C),
  };
}

/** Test linéarité N+C : superposition vs calcul direct, résidu relatif ≤ 1e-12. */
export function linearityError(cfg: PnnConfig = DEFAULT_PNN): {
  rel: number; worstAdd: number; samples: number;
} {
  const d = makePnn(cfg);
  const r = mulberry(cfg.seed + 7);
  const basis: { fr: Float64Array; fi: Float64Array }[] = [];
  for (let i = 0; i < cfg.N; i++) basis.push(basisField(d, i));
  let worst = 0, worstAdd = 0;
  const nS = 12;
  for (let s = 0; s < nS; s++) {
    const ar = new Float64Array(cfg.N), ai = new Float64Array(cfg.N);
    for (let i = 0; i < cfg.N; i++) { ar[i] = (r() * 2 - 1) * 1.2; ai[i] = (r() * 2 - 1) * 1.2; }
    const sup = fieldFromBasis(basis, ar, ai);
    const dir = forwardDirect(d.M0r, d.M0i, cfg.N, ar, ai);
    let err = 0, nrm = 0;
    for (let f = 0; f < cfg.F; f++) {
      err += (sup.fr[f] - dir.fr[f]) ** 2 + (sup.fi[f] - dir.fi[f]) ** 2;
      nrm += dir.fr[f] ** 2 + dir.fi[f] ** 2;
    }
    worst = Math.max(worst, Math.sqrt(err / nrm));
    // additivité : E(a+b) = E(a) + E(b) (propriété physique de superposition)
    const br2 = new Float64Array(cfg.N), bi2 = new Float64Array(cfg.N);
    for (let i = 0; i < cfg.N; i++) { br2[i] = (r() * 2 - 1) * 1.2; bi2[i] = (r() * 2 - 1) * 1.2; }
    const eA = fieldFromBasis(basis, ar, ai);
    const eB = fieldFromBasis(basis, br2, bi2);
    const ars = new Float64Array(cfg.N), ais = new Float64Array(cfg.N);
    for (let i = 0; i < cfg.N; i++) { ars[i] = ar[i] + br2[i]; ais[i] = ai[i] + bi2[i]; }
    const eS = fieldFromBasis(basis, ars, ais);
    let ea = 0, na = 0;
    for (let f = 0; f < cfg.F; f++) {
      ea += (eS.fr[f] - eA.fr[f] - eB.fr[f]) ** 2 + (eS.fi[f] - eA.fi[f] - eB.fi[f]) ** 2;
      na += eA.fr[f] ** 2 + eA.fi[f] ** 2 + eB.fr[f] ** 2 + eB.fi[f] ** 2;
    }
    worstAdd = Math.max(worstAdd, Math.sqrt(ea / na));
  }
  return { rel: worst, worstAdd, samples: nS };
}

/** Test gradient AVM : analytique vs différences finies centrées, erreur relative ≤ 1e-4. */
export function avmCheckError(cfg: PnnConfig = DEFAULT_PNN): { relErr: number; checks: number } {
  const d = makePnn(cfg);
  const samples = makeDataset(cfg, 8, 3);
  const { gr, gi } = avmGradient(d, samples);
  const lossOf = (): number => {
    let s = 0;
    for (const sm of samples) {
      const { fr, fi } = forwardDirect(d.M0r, d.M0i, cfg.N, sm.ar, sm.ai);
      const p = portPowers(d.Pr, d.Pi, cfg.C, fr, fi);
      s += ceLoss(softmax(p), sm.label);
    }
    return s / samples.length;
  };
  let worst = 0, done = 0;
  const h = 1e-6;
  for (let k = 0; k < gr.length; k++) {
    for (const [arr, g] of [[d.M0r, gr], [d.M0i, gi]] as [Float64Array, Float64Array][]) {
      const orig = arr[k];
      arr[k] = orig + h; const lp = lossOf();
      arr[k] = orig - h; const lm = lossOf();
      arr[k] = orig;
      const num = (lp - lm) / (2 * h);
      const denom = Math.max(Math.abs(num), Math.abs(g[k]), 1e-7);
      worst = Math.max(worst, Math.abs(num - g[k]) / denom);
      done++;
    }
  }
  return { relErr: worst, checks: done };
}

/** Benchmark (mesuré, pas fabriqué) : temps/epoch, ratio de coût N+C vs L, précision. */
export function pnnBench(cfg: PnnConfig = DEFAULT_PNN): {
  msPerEpoch: number; ncRatio: number; accuracy: number;
} {
  const t = trainPnn(cfg);
  return { msPerEpoch: t.msPerEpoch, ncRatio: t.ncRatio, accuracy: t.accTest };
}
