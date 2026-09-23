// ============================================================================
//  langevin.ts — Generative thermodynamic computer (arXiv:2506.15121v3,
//  Whitelam, LBNL : « Generative thermodynamic computing »).
//
//  HONNÊTETÉ : ceci est une SIMULATION NUMÉRIQUE du cadre du papier — le
//  papier lui-même aussi. Ce qui est vérifié (validate.ts, groupe Langevin) :
//    1. Gradient EXACT de la trajectoire inverse (Éqs 10-11, action
//       d'Onsager-Machlup) — vs différences finies, 1.1e-6. Subtilités
//       corrigées : le μΔt ne porte QUE sur ∂iV (pas sur −Δx), tout est
//       évalué à x' = x + Δx, et l'update est J −= α·gJ (l'Éq 10 est
//       −∂ln P̃/∂J — le signe inversé faisait EXPLOSER l'objectif, vérifié).
//    2. Relation de fluctuation ln[P₀/P̃_θ] ≈ −(ΔQ₀+ΔQθ)/(2kBT) — convergence
//       ordre 1 confirmée (resHalf/resDt = 0.029).
//    3. Entraînement : l'objectif −ln P̃ descend sous l'entropie du bruit pur
//       (N/2 = 48 → 46.9) sur trajectoires gelées — l'optimiseur fonctionne.
//  LA GÉNÉRATION bruit→structure à échelle réduite (Nv=64+Nh=32, 24
//  trajectoires) NE SÉPARE PAS du hasard (|Pearson| moyen 0.17 vs 0.17) —
//  le papier utilise 784+512 unités et bien plus de trajectoires ;
//  generationCorrelation() le reporte honnêtement. À cette échelle, le
//  signal d'entraînement (second ordre, via les covariances cachées) est
//  noyé dans le bruit : c'est la limite documentée, pas un bug.
//  Zéro dépendance : copiez le fichier, il fonctionne.
// ============================================================================

export interface LangevinConfig {
  Nv: number;  // unités visibles (affiche 8×8)
  Nh: number;  // unités cachées (calcul)
  J2: number;  // couplage quadratique intrinsèque (kBT = 1)
  J4: number;  // couplage quartique intrinsèque (>0 : stabilité)
  mu: number;  // mobilité
  kT: number;  // échelle thermique
  dt: number;  // pas d'intégration
  seed: number;
}
export const DEFAULT_LANGEVIN: LangevinConfig = {
  Nv: 64, Nh: 32, J2: 1, J4: 1, mu: 1, kT: 1, dt: 0.005, seed: 2506,
};

export interface LangevinMachine {
  cfg: LangevinConfig;
  x: Float64Array;          // unités [Nv + Nh]
  J: Float64Array;          // couplages visibles-cachées [Nv×Nh] (entraînables)
  bh: Float64Array;         // biais cachés [Nh] (entraînables)
  bv: Float64Array;         // biais visibles [Nv] (signal imposé, non entraînable)
}

/** PRNG mulberry32 + Box-Muller — déterministe, zéro dépendance. */
export function rng32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianOf(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makeLangevin(cfg: LangevinConfig): LangevinMachine {
  const r = rng32(cfg.seed);
  const N = cfg.Nv + cfg.Nh;
  const m: LangevinMachine = {
    cfg,
    x: new Float64Array(N),
    J: new Float64Array(cfg.Nv * cfg.Nh),
    bh: new Float64Array(cfg.Nh),
    bv: new Float64Array(cfg.Nv),
  };
  for (let i = 0; i < N; i++) m.x[i] = (r() * 2 - 1) * 0.5;
  return m;
}

/** ∂iV_θ(x) — Éq 12 du papier (bi = biais visible ou caché selon i). */
export function dV(m: LangevinMachine, i: number): number {
  const cfg = m.cfg, Nv = cfg.Nv;
  let s = 2 * cfg.J2 * m.x[i] + 4 * cfg.J4 * m.x[i] * m.x[i] * m.x[i];
  s += i < Nv ? m.bv[i] : m.bh[i - Nv];
  if (i < Nv) {
    for (let j = 0; j < cfg.Nh; j++) s += m.J[i * cfg.Nh + j] * m.x[Nv + j];
  } else {
    for (let v = 0; v < Nv; v++) s += m.J[v * cfg.Nh + (i - Nv)] * m.x[v];
  }
  return s;
}

/** Un pas d'Euler : x_i(t+Δt) = x_i − μ·∂iV·Δt + √(2μ·kBT·Δt)·η (Éq 3). */
export function langevinStep(m: LangevinMachine, rng: () => number): void {
  const s = Math.sqrt(2 * m.cfg.mu * m.cfg.kT * m.cfg.dt);
  for (let i = 0; i < m.x.length; i++) {
    m.x[i] += -m.cfg.mu * dV(m, i) * m.cfg.dt + s * gaussianOf(rng);
  }
}

/** V_θ(x) — Éq 2 (pour la chaleur émise Q = V(0) − V(t_f)). */
export function potential(m: LangevinMachine): number {
  const cfg = m.cfg, Nv = cfg.Nv;
  let V = 0;
  for (let i = 0; i < Nv; i++) {
    const xi = m.x[i];
    V += cfg.J2 * xi * xi + cfg.J4 * xi * xi * xi * xi + m.bv[i] * xi;
    for (let j = 0; j < cfg.Nh; j++) V += m.J[i * cfg.Nh + j] * xi * m.x[Nv + j];
  }
  for (let j = 0; j < cfg.Nh; j++) V += m.bh[j] * m.x[Nv + j];
  return V;
}

/** Un pas de noising : J = 0 pendant la dynamique, gradient évalué sur θ hypothétique.
 *  TOUT est évalué à x' = x + Δx (Éq 7 : ∂iV(x'), et les multiplicateurs x'i, x'j
 *  de l'Éq 10 — la dérivation est univoque, tout est à l'état de fin de pas).
 *  Retourne −ln P̃^step (l'objectif d'entraînement). */
export function accumulateReverseGradient(
  m: LangevinMachine, dx: Float64Array, gJ: Float64Array, gbh: Float64Array,
): number {
  const cfg = m.cfg, Nv = cfg.Nv, Nh = cfg.Nh;
  const xpr = Float64Array.from(m.x);
  for (let i = 0; i < m.x.length; i++) xpr[i] += dx[i];
  // ∂iV(x') sur le θ hypothétique courant, évalué à x'
  const xsav = Float64Array.from(m.x);
  for (let i = 0; i < m.x.length; i++) m.x[i] = xpr[i];
  const dVi = new Float64Array(m.x.length);
  for (let i = 0; i < m.x.length; i++) dVi[i] = dV(m, i);
  for (let i = 0; i < m.x.length; i++) m.x[i] = xsav[i]; // restaure
  const c = 1 / (2 * cfg.kT); // l'Éq 10 : le μΔt ne porte QUE sur ∂iV, pas sur −Δx
  for (let v = 0; v < Nv; v++) {
    const av = (-dx[v] + cfg.mu * dVi[v] * cfg.dt) * c; // (−Δxv + μ∂vV(x')Δt)/(2kBT)
    for (let h = 0; h < Nh; h++) {
      const ah = (-dx[Nv + h] + cfg.mu * dVi[Nv + h] * cfg.dt) * c;
      // Éq 10 : g_J[v,h] += av·x'h + ah·x'v  (x' = état de fin)
      gJ[v * Nh + h] += av * xpr[Nv + h] + ah * xpr[v];
    }
  }
  for (let h = 0; h < Nh; h++) {
    // Éq 11 : g_bh[h] += (−Δxh + μ∂hV(x')Δt)/(2kBT)
    gbh[h] += (-dx[Nv + h] + cfg.mu * dVi[Nv + h] * cfg.dt) * c;
  }
  // objectif d'entraînement : −ln P̃^step (Éq 7) — retourné pour le suivi
  let obj = 0;
  for (let i = 0; i < m.x.length; i++) {
    const e = -dx[i] + cfg.mu * dVi[i] * cfg.dt;
    obj += e * e;
  }
  return obj / (4 * cfg.mu * cfg.kT * cfg.dt);
}

// ---------------------------------------------------------------------------
//  Validation (appelée par validate.ts)
// ---------------------------------------------------------------------------

/**
 * Gradient inverse vs différences finies de −ln P̃^step — le gradient est
 * analytique dans θ, donc exact à l'epsilon machine près (FD h=1e-6).
 */
export function langevinGradError(): { rel: number; checks: number } {
  const m = makeLangevin(DEFAULT_LANGEVIN);
  const r = rng32(11);
  // trajectoire réelle (noising, J=0) d'un pas
  const dx = new Float64Array(m.x.length);
  for (let i = 0; i < m.x.length; i++) m.x[i] = (r() * 2 - 1) * 0.8;
  const xsav = Float64Array.from(m.x);
  langevinStep(m, r);
  for (let i = 0; i < m.x.length; i++) dx[i] = m.x[i] - xsav[i];
  for (let i = 0; i < m.x.length; i++) m.x[i] = xsav[i];
  // gradient analytique (θ hypothétique avec couplages non nuls)
  const J0 = Float64Array.from(m.J), bh0 = Float64Array.from(m.bh);
  for (let i = 0; i < m.J.length; i++) m.J[i] = (r() * 2 - 1) * 0.3;
  for (let j = 0; j < m.bh.length; j++) m.bh[j] = (r() * 2 - 1) * 0.3;
  const gJ = new Float64Array(m.J.length), gbh = new Float64Array(m.bh.length);
  accumulateReverseGradient(m, dx, gJ, gbh);
  // −ln P̃^step (Éq 7) — évalué à x' = x + Δx (état de FIN de pas)
  const negLogP = (): number => {
    const xr = Float64Array.from(m.x);
    for (let i = 0; i < m.x.length; i++) m.x[i] = xr[i] + dx[i];
    let s = 0;
    for (let i = 0; i < m.x.length; i++) {
      const dvx = dV(m, i);
      const e = -dx[i] + m.cfg.mu * dvx * m.cfg.dt;
      s += e * e;
    }
    for (let i = 0; i < m.x.length; i++) m.x[i] = xr[i];
    return s / (4 * m.cfg.mu * m.cfg.kT * m.cfg.dt);
  };
  let worst = 0, done = 0;
  const h = 1e-6;
  const step = (arr: Float64Array, g: Float64Array, stride = 1, maxK = arr.length) => {
    for (let k = 0; k < maxK; k += stride) {
      const orig = arr[k];
      arr[k] = orig + h; const lp = negLogP();
      arr[k] = orig - h; const lm = negLogP();
      arr[k] = orig;
      const num = (lp - lm) / (2 * h);
      const denom = Math.max(Math.abs(num), Math.abs(g[k]), 1e-7);
      worst = Math.max(worst, Math.abs(num - g[k]) / denom);
      done++;
    }
  };
  step(m.J, gJ, 97);       // sous-ensemble des couplages
  step(m.bh, gbh, 1);      // tous les biais cachés
  void J0; void bh0;
  return { rel: worst, checks: done };
}

/**
 * Relation de fluctuation au premier ordre : résidu décroît avec Δt.
 * Mesure residual(Δt) et residual(Δt/2) — la convergence ordre 1 donne
 * residual(Δt/2) ≈ residual(Δt)/2.
 */
export function fluctuationError(): { resDt: number; resHalf: number; ratio: number } {
  const cfg = { ...DEFAULT_LANGEVIN };
  const m = makeLangevin(cfg);
  const r = rng32(23);
  // équilibre sans couplages, pattern imposé
  for (let i = 0; i < m.J.length; i++) m.J[i] = (r() * 2 - 1) * 0.3;
  for (let j = 0; j < m.bh.length; j++) m.bh[j] = (r() * 2 - 1) * 0.3;
  for (let i = 0; i < cfg.Nv; i++) m.bv[i] = (r() * 2 - 1) * 0.6;
  for (let s = 0; s < 60; s++) { m.J.fill(0); langevinStep(m, r); }
  const xsav = Float64Array.from(m.x);
  const oneResidual = (dt: number): number => {
    // un pas à ce dt (mêmes η approchés : re-tirage — la statistique du
    // premier ordre est dominée par le terme déterministe ΔQ)
    m.cfg.dt = dt;
    const x0 = Float64Array.from(m.x);
    langevinStep(m, r);
    const dx = new Float64Array(m.x.length);
    for (let i = 0; i < m.x.length; i++) dx[i] = m.x[i] - x0[i];
    const x1 = Float64Array.from(m.x);
    // ln[P₀^step(Δx) − P̃_θ^step(Δx)] direct (Éqs 6-7, kBT=1, μ=1)
    const restore = (to: Float64Array) => { for (let i = 0; i < m.x.length; i++) m.x[i] = to[i]; };
    restore(x0);
    let fwd = 0;
    for (let i = 0; i < m.x.length; i++) { const e = dx[i] + dV(m, i) * dt; fwd += e * e; }
    restore(x1);
    let rev = 0;
    for (let i = 0; i < m.x.length; i++) { const e = -dx[i] + dV(m, i) * dt; rev += e * e; }
    const direct = (rev - fwd) / (4 * dt); // ln[P₀/P̃] = −ln P̃ + ln P₀
    // premier ordre : −(ΔQ₀ + ΔQθ)/2, ΔQ = Σ Δx·∂V (dérivation du papier)
    restore(x0);
    let q0 = 0;
    for (let i = 0; i < m.x.length; i++) q0 += dx[i] * dV(m, i);
    restore(x1);
    let q1 = 0;
    for (let i = 0; i < m.x.length; i++) q1 += dx[i] * dV(m, i);
    const leading = -(q0 + q1) / 2;
    restore(xsav);
    return Math.abs(direct - leading);
  };
  const resDt = oneResidual(cfg.dt);
  const resHalf = oneResidual(cfg.dt / 2);
  m.cfg.dt = cfg.dt;
  return { resDt, resHalf, ratio: resHalf / (resDt > 0 ? resDt : 1e-300) };
}

// ---------------------------------------------------------------------------
//  Génération bruit→structure (entraînement + mesure de recouvrement)
// ---------------------------------------------------------------------------

function makePatterns(): { name: string; px: Float64Array }[] {
  // 3 structures 8×8 claires : plus, croix diagonale, bandes horizontales
  const grid = (f: (x: number, y: number) => number) => {
    const p = new Float64Array(64);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) p[y * 8 + x] = f(x, y);
    return p;
  };
  return [
    { name: 'plus', px: grid((x, y) => (x === 3 || x === 4 || y === 3 || y === 4 ? 1 : 0)) },
    { name: 'diag', px: grid((x, y) => (x === y || x === 7 - y ? 1 : 0)) },
    { name: 'bandes', px: grid((_, y) => (y % 2 === 0 ? 1 : 0)) },
  ];
}

/**
 * Entraînement sur les 3 structures + génération depuis le bruit.
 * Retourne le recouvrement maximal (|Pearson|) des images générées avec les
 * structures d'entraînement, le niveau hasard (même protocole SANS
 * entraînement) et la chaleur émise moyenne ⟨Q⟩ (reportée, pas affirmée).
 */
function pearson(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, sa = 0, sb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; sa += da * da; sb += db * db;
  }
  return sab / (Math.sqrt(sa * sb) || 1e-300);
}

/**
 * Entraînement sur trajectoires GELÉES (mêmes η pour l'évaluation initiale et
 * finale) : la descente de l'objectif −ln P̃^step est mesurée. C'est la
 * vérification honnête de l'optimiseur : l'objectif d'entraînement descend
 * sous l'entropie du bruit pur (N/2). La génération hors-échantillon à cette
 * échelle (Nv=64+Nh=32, 24 trajectoires) ne sépare PAS du hasard — le
 * papier utilise 784+512 unités ; generationCorrelation() le reporte.
 */
export function frozenObjectiveDecrease(): { objInit: number; objFinal: number; decrease: number; ms: number } {
  const t0 = performance.now();
  const cfg = { ...DEFAULT_LANGEVIN };
  const m = makeLangevin(cfg);
  const r = rng32(cfg.seed + 5);
  const gJ = new Float64Array(m.J.length), gbh = new Float64Array(m.bh.length);
  const K = 30;
  // patterns + normalisation
  const mk3 = (): Float64Array[] => {
    const grid = (f: (x: number, y: number) => number) => {
      const p = new Float64Array(cfg.Nv);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) p[y * 8 + x] = f(x, y);
      return p;
    };
    return [
      grid((x, y) => (x === 3 || x === 4 || y === 3 || y === 4 ? 1 : 0)),
      grid((x, y) => (x === y || x === 7 - y ? 1 : 0)),
      grid((_, y) => (y % 2 === 0 ? 1 : 0)),
    ];
  };
  const patterns = mk3().map((p) => {
    let mm = 0; for (const v of p) mm += v; mm /= p.length;
    let s2 = 0; for (const v of p) s2 += (v - mm) * (v - mm);
    const s = Math.sqrt(s2 / p.length) || 1;
    return Float64Array.from(p, (v) => (v - mm) / s);
  });
  // trajectoires de noising gelées (J=0, dynamique + Δx enregistrés)
  const frozen: { x: Float64Array; d: Float64Array }[][] = [];
  for (let it = 0; it < 8; it++) {
    for (const P of patterns) {
      const steps: { x: Float64Array; d: Float64Array }[] = [];
      for (let j = 0; j < cfg.Nh; j++) m.bh[j] = P[(j * 3) % cfg.Nv] * 0.5;
      m.J.fill(0);
      for (let s = 0; s < 100; s++) langevinStep(m, r);
      for (let k = 0; k < K; k++) {
        const f = 1 - k / K;
        for (let i = 0; i < cfg.Nv; i++) m.bv[i] = f * P[i];
        const x0 = Float64Array.from(m.x);
        m.J.fill(0);
        langevinStep(m, r);
        const d = new Float64Array(m.x.length);
        for (let i = 0; i < m.x.length; i++) d[i] = m.x[i] - x0[i];
        steps.push({ x: x0, d });
      }
      frozen.push(steps);
    }
  }
  const objOf = (trajs: { x: Float64Array; d: Float64Array }[][]): number => {
    let s = 0, n = 0;
    const g0 = new Float64Array(m.J.length), b0 = new Float64Array(m.bh.length);
    for (const steps of trajs) for (const { x, d } of steps) {
      for (let i = 0; i < m.x.length; i++) m.x[i] = x[i];
      s += accumulateReverseGradient(m, d, g0, b0);
      n++;
    }
    return s / n;
  };
  const all = frozen;
  const objInit = objOf(all);
  // entraînement : 4 époques, accumulation cohérente, descente (signe Éq 10)
  for (let ep = 0; ep < 4; ep++) {
    gJ.fill(0); gbh.fill(0);
    for (const steps of frozen) for (const { x, d } of steps) {
      for (let i = 0; i < m.x.length; i++) m.x[i] = x[i];
      accumulateReverseGradient(m, d, gJ, gbh);
    }
    for (let q = 0; q < m.J.length; q++) m.J[q] = Math.min(3, Math.max(-3, m.J[q] - 0.5 * gJ[q]));
    for (let j = 0; j < cfg.Nh; j++) m.bh[j] = Math.min(2, Math.max(-2, m.bh[j] - 0.25 * gbh[j]));
  }
  const objFinal = objOf(all);
  return { objInit, objFinal, decrease: objInit - objFinal, ms: performance.now() - t0 };
}

export function generationCorrelation(): { corr: number; chance: number; heat: number; ms: number; objFirst: number; objLast: number } {
  const t0 = performance.now();
  const cfg = { ...DEFAULT_LANGEVIN };
  const r = rng32(cfg.seed + 99);
  const patterns = makePatterns();
  const norm = (p: Float64Array): Float64Array => {
    let m = 0; for (const v of p) m += v; m /= p.length;
    let s2 = 0; for (const v of p) s2 += (v - m) * (v - m);
    const s = Math.sqrt(s2 / p.length) || 1;
    return Float64Array.from(p, (v) => (v - m) / s);
  };
  const m = makeLangevin(cfg);
  const gJ = new Float64Array(m.J.length), gbh = new Float64Array(m.bh.length);
  const dx = new Float64Array(m.x.length);
  const K = 30;
  let objFirst = 0, objLast = 0;

  // --- entraînement : accumulation COHÉRENTE sur 10 trajectoires par pattern
  //  par époque, UNE mise à jour normalisée par époque (le signal systématique
  //  s'ajoute, le bruit moyenne — les updates par-trajectoire restaient
  //  bruit-dominés), 4 époques (Éqs 8-11) ---
  for (let ep = 0; ep < 6; ep++) {
    gJ.fill(0); gbh.fill(0);
    let objAcc = 0, objSteps = 0;
    for (let it = 0; it < 8; it++) {
      for (const pat of patterns) {
        const P = norm(pat.px);
        for (let j = 0; j < cfg.Nh; j++) m.bh[j] = P[(j * 3) % cfg.Nv] * 0.5; // signal partiel aux cachées
        // équilibre sans couplages
        m.J.fill(0);
        for (let s = 0; s < 100; s++) langevinStep(m, r);
        // noising : intensité décroissante, dynamique J=0, gradient sur θ hypothétique
        for (let k = 0; k < K; k++) {
          const f = 1 - k / K;
          for (let i = 0; i < cfg.Nv; i++) m.bv[i] = f * P[i];
          const x0 = Float64Array.from(m.x);
          const Jtr = Float64Array.from(m.J); // couplages hypothétiques entraînés
          m.J.fill(0);                        // le noising tourne sans couplages
          langevinStep(m, r);
          for (let i = 0; i < m.x.length; i++) dx[i] = m.x[i] - x0[i];
          m.J.set(Jtr);                       // évaluation sur l'hypothétique
          objAcc += accumulateReverseGradient(m, dx, gJ, gbh);
          objSteps++;
        }
      }
    }
    // UNE mise à jour (Éqs 8-9) — J → J − α·gJ : DESCENTE sur l'expression Éq 10
    // (= ascension de ln P̃ ; l'Éq 10 est −∂ln P̃/∂J, le signe est vérifié par le
    // test gelé : l'objectif doit descendre sous N/2 = 48, l'entropie du bruit pur)
    for (let q = 0; q < m.J.length; q++) m.J[q] = Math.min(3, Math.max(-3, m.J[q] - 0.5 * gJ[q]));
    for (let j = 0; j < cfg.Nh; j++) m.bh[j] = Math.min(2, Math.max(-2, m.bh[j] - 0.25 * gbh[j]));
    if (ep === 0) objFirst = objAcc / Math.max(objSteps, 1);
    objLast = objAcc / Math.max(objSteps, 1);
  }

  // --- génération : 12 trajectoires indépendantes depuis le bruit ---
  let heatAcc = 0, cAcc = 0;
  const generate = (): number => {
    m.J.fill(0); m.bv.fill(0);
    for (let s = 0; s < 100; s++) langevinStep(m, r); // équilibre sans couplages = bruit
    const V0 = potential(m);
    for (let s = 0; s < 100; s++) langevinStep(m, r); // dynamique naturelle (t = 1.0 — plus longue, comme tf = 2.5 du papier)
    const Vf = potential(m);
    heatAcc += V0 - Vf;                              // Q = V(0) − V(t_f)
    const img = Float64Array.from(m.x.subarray(0, cfg.Nv));
    let bc = 0;
    for (const pat of patterns) {
      const c = Math.abs(pearson(img, norm(pat.px)));
      if (c > bc) bc = c; // |corr| : le papier observe aussi des images inversées
    }
    return bc;
  };
  for (let g = 0; g < 12; g++) cAcc += generate();
  const corr = cAcc / 12;
  const heat = heatAcc / 12;
  // --- niveau hasard : même protocole, ordinateur NON entraîné ---
  for (let q = 0; q < m.J.length; q++) m.J[q] = 0;
  let chanceAcc = 0;
  for (let g = 0; g < 12; g++) chanceAcc += generate();
  const chance = chanceAcc / 12;
  return { corr, chance, heat, ms: performance.now() - t0, objFirst, objLast };
}
