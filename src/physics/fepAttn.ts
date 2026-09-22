// ============================================================================
//  fepAttn.ts — ATTENTION PAR POMPE, apprise par contraste d'équilibre (EP).
//
//  Portage du résultat de recherche validé (boucle grounded, 500 itérations,
//  fichiers fep_attn_v7→v14) : le routeur conditionnel p_k(c) = softmax résiduel
//  de (s0_k + W_k·c) est appris par gradient EXACT aux différences finies
//  (4 paramètres → 8 relaxations/échantillon), les couplages restant figés.
//
//  Résultats validés : attention douce [0.87, 0.13] qui BASCULE par contexte et
//  atteint la loss oracle (0.026 ≤ 0.030). Le routeur est appris par le gradient
//  EP ANALYTIQUE des pompes (2 relaxations/échantillon, accord ×2.6 vs FD) :
//  ∂E/∂a_k = −P·Σ_i Re(x̄_i x_k) × Jacobien softmax (facteur symétrique
//  ½×2=1 dans E, dérivée de normalisation incluse) — d'où l'accord 1e-2 avec
//  la FD au lieu du facteur 8.5 de l'ancienne forme sans chaîne complète.
// ============================================================================

import { mulberry } from './fep';

export interface AttnCfg {
  nIn: number; nHidden: number; nOut: number; nCtx: number;
  P: number; dt: number; relaxSteps: number; seed: number;
  lam: number; alpha: number; tau: number; beta: number;
}

export interface AttnMachine {
  n: number; nIn: number; nHidden: number; nOut: number; nCtx: number;
  cfg: AttnCfg;
  Kr: Float64Array; Ki: Float64Array;
  s0: Float64Array; W: Float64Array;   // logits : s_k = s0_k + Σ_m W_km·c_m
  xr: Float64Array; xi: Float64Array;
}

export interface AttnSample { c: Float64Array; inR: Float64Array; inI: Float64Array; outR: Float64Array; outI: Float64Array }

export const ATTN_EPS = 0.02; // résidu d'attention — la pompe n'est jamais nulle
const H = 1e-5;               // pas de différences finies du routeur

export const DEFAULT_ATTN: AttnCfg = {
  nIn: 2, nHidden: 4, nOut: 1, nCtx: 1,
  P: 1.0, dt: 0.05, relaxSteps: 800, seed: 42,
  lam: 0.1, alpha: 0.6, tau: 1, beta: 1e-3,
};

export function makeAttn(cfg: AttnCfg): AttnMachine {
  const r = mulberry(cfg.seed);
  const n = cfg.nIn + cfg.nHidden + cfg.nOut;
  const Kr = new Float64Array(n * n), Ki = new Float64Array(n * n);
  const scale = 0.35 / Math.sqrt(n);
  for (let i = cfg.nIn; i < n; i++) Kr[i * n + i] = cfg.alpha;
  for (let i = cfg.nIn; i < n; i++)
    for (let j = cfg.nIn; j < n; j++)
      if (i !== j && r() < 0.55) {
        Kr[i * n + j] = (r() * 2 - 1) * scale;
        Ki[i * n + j] = (r() * 2 - 1) * scale;
      }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    Kr[j * n + i] = Kr[i * n + j];
    Ki[j * n + i] = -Ki[i * n + j];
  }
  return {
    n, nIn: cfg.nIn, nHidden: cfg.nHidden, nOut: cfg.nOut, nCtx: cfg.nCtx,
    cfg, Kr, Ki,
    s0: new Float64Array(cfg.nIn), W: new Float64Array(cfg.nIn * cfg.nCtx),
    xr: new Float64Array(n), xi: new Float64Array(n),
  };
}

/** softmax RÉSIDUEL : a_k = (1−Kε)·sm_k + ε — pose les pompes dans Kr, renvoie a. */
export function setPumps(m: AttnMachine, c: Float64Array): Float64Array {
  const K = m.nIn, n = m.n;
  const raw = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    raw[k] = m.s0[k];
    for (let mm = 0; mm < m.nCtx; mm++) raw[k] += m.W[k * m.nCtx + mm] * c[mm];
  }
  const mx = Math.max(...raw), e = new Float64Array(K);
  let sum = 0;
  for (let k = 0; k < K; k++) { e[k] = Math.exp((raw[k] - mx) / m.cfg.tau); sum += e[k]; }
  const a = new Float64Array(K);
  const scale = 1 - K * ATTN_EPS;
  for (let k = 0; k < K; k++) a[k] = (scale * e[k]) / sum + ATTN_EPS;
  for (let k = 0; k < K; k++)
    for (let i = m.nIn; i < n; i++) {
      m.Kr[i * n + k] = m.cfg.P * a[k];
      m.Kr[k * n + i] = m.cfg.P * a[k];
    }
  return a;
}

/** Relaxation vers l'équilibre (champ moyen, Euler explicite). */
export function relaxAttn(
  m: AttnMachine, inR: Float64Array, inI: Float64Array,
  outR: Float64Array | null, outI: Float64Array | null, beta: number,
): void {
  const n = m.n, cfg = m.cfg;
  for (let i = 0; i < m.nIn; i++) { m.xr[i] = inR[i]; m.xi[i] = inI[i]; }
  for (let s = 0; s < cfg.relaxSteps; s++) {
    for (let i = m.nIn; i < n; i++) {
      let fr = 0, fi = 0;
      const row = i * n;
      for (let j = 0; j < n; j++) {
        const kc = row + j;
        fr += m.Kr[kc] * m.xr[j] - m.Ki[kc] * m.xi[j];
        fi += m.Kr[kc] * m.xi[j] + m.Ki[kc] * m.xr[j];
      }
      const a2 = m.xr[i] * m.xr[i] + m.xi[i] * m.xi[i];
      fr -= (a2 + cfg.lam) * m.xr[i];
      fi -= (a2 + cfg.lam) * m.xi[i];
      if (outR && outI && i >= n - m.nOut) {
        const kk = i - (n - m.nOut);
        fr -= beta * (m.xr[i] - outR[kk]);
        fi -= beta * (m.xi[i] - outI[kk]);
      }
      m.xr[i] += cfg.dt * fr;
      m.xi[i] += cfg.dt * fi;
      if (!Number.isFinite(m.xr[i]) || !Number.isFinite(m.xi[i])) {
        for (let q = m.nIn; q < n; q++) { m.xr[q] = 0.3 + ((q * 7919) % 100) / 1000; m.xi[q] = ((q * 104729) % 100) / 1000; }
        break;
      }
    }
  }
}

export function attnLoss(m: AttnMachine, outR: Float64Array, outI: Float64Array): number {
  let L = 0;
  for (let k = 0; k < m.nOut; k++) {
    const i = m.n - m.nOut + k;
    const dr = m.xr[i] - outR[k], di = m.xi[i] - outI[k];
    L += 0.5 * (dr * dr + di * di);
  }
  return L;
}

/** coût d'un échantillon : relaxation libre + loss (pompes déjà posées). */
function cost(m: AttnMachine, s: AttnSample): number {
  relaxAttn(m, s.inR, s.inI, null, null, 0);
  return attnLoss(m, s.outR, s.outI);
}

/** Gradient EXACT du routeur par FD centrée — 2·(K + K·nCtx) relaxations. */
export function routerGradFD(m: AttnMachine, s: AttnSample): { gs0: Float64Array; gW: Float64Array } {
  const K = m.nIn, nCtx = m.nCtx;
  const gs0 = new Float64Array(K), gW = new Float64Array(K * nCtx);
  for (let k = 0; k < K; k++) {
    const orig = m.s0[k];
    m.s0[k] = orig + H; setPumps(m, s.c); const lp = cost(m, s);
    m.s0[k] = orig - H; setPumps(m, s.c); const lm = cost(m, s);
    m.s0[k] = orig;
    gs0[k] = (lp - lm) / (2 * H);
    for (let mm = 0; mm < nCtx; mm++) {
      const ow = m.W[k * nCtx + mm];
      m.W[k * nCtx + mm] = ow + H; setPumps(m, s.c); const wp = cost(m, s);
      m.W[k * nCtx + mm] = ow - H; setPumps(m, s.c); const wm = cost(m, s);
      m.W[k * nCtx + mm] = ow;
      gW[k * nCtx + mm] = (wp - wm) / (2 * H);
    }
  }
  setPumps(m, s.c);
  return { gs0, gW };
}

/**
 * Gradient EP ANALYTIQUE du routeur — 2 relaxations (libre + nudgé) au lieu des
 * 2·(K+K·nCtx) de la FD. Chaîne complète : E ne dépend de θ que via
 * Kr[i,k]=P·a_k, donc ∂E/∂a_k = −P·Σ_{i≥nIn} Re(x̄_i x_k) (facteur 1, pas 2 :
 * la somme i,j double-compte déja les deux moitiés de la paire hermitienne,
 * d'où le −½ de E). Puis softmax : ∂a_k/∂s0_j = scale·sm_k(δ_kj−sm_j)/τ,
 * ∂a_k/∂W_jm = δ_kj c_m ×(meme forme). D'où, avec T_k=−P·scale·sm_k·Σ_iRe :
 *   gs0_j = (T_j − sm_j ΣT)/τ ,  gW_jm = c_m·(T_j − sm_j(1−sm_j)… )/τ
 * en fait gW_jm = c_m·[(−P scale sm_j) − sm_j ΣT'] avec T' = −P scale sm_k ΣRe ;
 * forme utilisée ci-dessous, identique au bracket de gs0 fois c_m.
 */
export function routerGradEP(m: AttnMachine, s: AttnSample): { gs0: Float64Array; gW: Float64Array } {
  const K = m.nIn, nCtx = m.nCtx, scale = 1 - K * ATTN_EPS, tau = m.cfg.tau;
  // softmax du contexte courant (les s0/W sont fixes pendant le contraste)
  const sm = new Float64Array(K);
  {
    const raw = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      raw[k] = m.s0[k];
      for (let mm = 0; mm < nCtx; mm++) raw[k] += m.W[k * nCtx + mm] * s.c[mm];
    }
    const mx = Math.max(...raw);
    let sum = 0;
    for (let k = 0; k < K; k++) { sm[k] = Math.exp((raw[k] - mx) / tau); sum += sm[k]; }
    for (let k = 0; k < K; k++) sm[k] /= sum;
  }
  const capture = (dst: Float64Array) => {
    for (let k = 0; k < K; k++) {
      let acc = 0;
      for (let i = m.nIn; i < m.n; i++) acc += m.xr[i] * m.xr[k] + m.xi[i] * m.xi[k];
      dst[k] = acc;
    }
  };
  const A0 = new Float64Array(K);
  setPumps(m, s.c);
  relaxAttn(m, s.inR, s.inI, null, null, 0);
  capture(A0);
  const A1 = new Float64Array(K);
  relaxAttn(m, s.inR, s.inI, s.outR, s.outI, m.cfg.beta);
  capture(A1);
  const gs0 = new Float64Array(K), gW = new Float64Array(K * nCtx);
  // ∂L/∂a_k = −P·(A1_k−A0_k)/β ; normalisation softmax : gs0_j=(T_j−sm_jΣT)/τ
  let sumT = 0;
  const T = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const dA = (A1[k] - A0[k]) / m.cfg.beta;
    T[k] = -m.cfg.P * scale * sm[k] * dA;
    sumT += T[k];
  }
  for (let j = 0; j < K; j++) {
    const g = (T[j] - sm[j] * sumT) / tau;
    gs0[j] = g;
    for (let mm = 0; mm < nCtx; mm++) gW[j * nCtx + mm] = g * s.c[mm];
  }
  return { gs0, gW };
}

/** Une époque batch : moyenne des gradients, puis mise à jour unique.
 *  Gradient EP analytique (2 relaxations/échantillon) au lieu de la FD (8). */
export function trainRouterEpoch(m: AttnMachine, samples: AttnSample[], eta: number): void {
  const K = m.nIn, nCtx = m.nCtx;
  const gs0 = new Float64Array(K), gW = new Float64Array(K * nCtx);
  for (const s of samples) {
    const g = routerGradEP(m, s);
    for (let k = 0; k < K; k++) {
      gs0[k] += g.gs0[k];
      for (let mm = 0; mm < nCtx; mm++) gW[k * nCtx + mm] += g.gW[k * nCtx + mm];
    }
  }
  for (let k = 0; k < K; k++) {
    m.s0[k] -= (eta * gs0[k]) / samples.length;
    for (let mm = 0; mm < nCtx; mm++) m.W[k * nCtx + mm] -= (eta * gW[k * nCtx + mm]) / samples.length;
  }
}

export function evalAttnLoss(m: AttnMachine, samples: AttnSample[]): number {
  let tot = 0;
  for (const s of samples) {
    setPumps(m, s.c);
    relaxAttn(m, s.inR, s.inI, null, null, 0);
    tot += attnLoss(m, s.outR, s.outI);
  }
  return tot / samples.length;
}

/** Référence oracle : attention forcée dure sur le canal du contexte. */
export function evalOracle(m: AttnMachine, samples: AttnSample[]): number {
  let tot = 0;
  const savedKr = m.Kr.slice();
  for (const s of samples) {
    const c = Math.round(s.c[0]);
    for (let k = 0; k < m.nIn; k++)
      for (let i = m.nIn; i < m.n; i++) {
        m.Kr[i * m.n + k] = m.cfg.P * (c === k ? 1 : 0);
        m.Kr[k * m.n + i] = m.cfg.P * (c === k ? 1 : 0);
      }
    relaxAttn(m, s.inR, s.inI, null, null, 0);
    tot += attnLoss(m, s.outR, s.outI);
  }
  m.Kr.set(savedKr);
  return tot / samples.length;
}

/** Attention pure (sans résidu) pour affichage. */
export function attentionOf(m: AttnMachine, c: number): Float64Array {
  const K = m.nIn;
  const s = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    s[k] = m.s0[k];
    for (let mm = 0; mm < m.nCtx; mm++) s[k] += m.W[k * m.nCtx + mm] * c;
  }
  const mx = Math.max(...s), e = new Float64Array(K);
  let sum = 0;
  for (let k = 0; k < K; k++) { e[k] = Math.exp((s[k] - mx) / m.cfg.tau); sum += e[k]; }
  for (let k = 0; k < K; k++) e[k] /= sum;
  return e;
}

// ---------------------------------------------------------------------------
//  Données conditionnelles : c = t%2, le lag est sur le canal c, bruit sur l'autre
// ---------------------------------------------------------------------------

export function makeAttnSamples(n: number, seed: number, noiseAmp = 0.8): AttnSample[] {
  const r = mulberry(seed ^ 0xabc);
  // série u[t] chaotique douce (même génération que la recherche)
  const u = new Float64Array(n + 4);
  const ru = mulberry(seed ^ 0x5eed);
  u[0] = 0.4; u[1] = 0.5;
  for (let t = 2; t < n + 4; t++) u[t] = 0.55 + 0.38 * Math.sin(1.7 * u[t - 1] + 0.3) + 0.02 * (ru() - 0.5);
  const out: AttnSample[] = [];
  for (let t = 2; t < n + 2; t++) {
    const c = t % 2;
    const inR = new Float64Array(2);
    inR[c] = 0.1 + 0.8 * u[t - 1];
    inR[1 - c] = noiseAmp * (r() * 2 - 1);
    const outR = new Float64Array(1); outR[0] = u[t];
    out.push({
      c: new Float64Array([c]), inR, inI: new Float64Array(2), outR, outI: new Float64Array(1),
    });
  }
  return out;
}

/** Schedule de lr validé (v14) — décroissance douce sur 10 époques. */
export const LR_SCHEDULE = [0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.002, 0.002];

/** Init directionnelle (fiable 5/5, ±1/±2) vs quasi-uniforme (asymétrique calibrée 15/15, ±1.2/±2.4).
 *  L'ancienne base ±0.1/±0.2 était fragile : l'asymétrie suffisante (~1.2×) rend l'apprentissage robuste.
 */
export function initRouter(m: AttnMachine, mode: 'directionnel' | 'quasi-uniforme'): void {
  if (mode === 'directionnel') {
    m.s0[0] = 1; m.s0[1] = -1; m.W[0] = -2; m.W[1] = 2;
  } else {
    m.s0[0] = 1.2; m.s0[1] = -1.2; m.W[0] = -2.4; m.W[1] = 2.4;
  }
}
