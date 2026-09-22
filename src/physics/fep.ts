// ============================================================================
//  fep.ts — Coherent Free-Energy Machine (Equilibrium Propagation, complex).
//
//  A network of complex fields {x_i} minimizing a shared free energy
//
//      E(x) = -1/2 Re[x† K x] + 1/4 Σ |x_i|⁴ + 1/2 λ Σ |x_i|²
//
//  learns a temporal prediction task by CONTRAST between two equilibria
//  (Scellier-Bengio): a free equilibrium x⁰ and a nudged equilibrium x^β
//  reached under the nudge  -β·∇L  on the output units.  The phase-gradient
//  identity
//
//      dL/dθ = lim_{β→0} (1/β) [ ∂E/∂θ(x^β) − ∂E/∂θ(x⁰) ]
//
//  is checked numerically against centred finite differences, for BOTH the
//  real and the imaginary parts of every coupling.
//
//  Design constraints that make the identity EXACT (each one is required):
//    1. K is HERMITIAN  (K_ji = conj(K_ij))  → the flow  ẋ = Kx − (|x|²+λ)x
//       is the real gradient of E (a non-Hermitian K is not a gradient flow
//       and the identity breaks by an asymmetric-Hessian factor).
//    2. Diagonal self-coupling  Kr[ii] = α  → healthy amplitudes √(α−λ),
//       the trivial x=0 fixed point is repelled.
//    3. Real input pump  Kr[i, in] = pump  → phases pinned, well-conditioned
//       phase modes (no near-zero Hessian eigenvalue).
//    4. Symmetric-pair parameterisation: for θ = Kr_ij = Kr_ji (shared),
//         dE/dθ = −Re[x̄_i x_j]  ⇒  gEP = 2(dEdKr(x^β) − dEdKr(x⁰))/β.
//       For the antisymmetric imaginary part θ = Ki_ij, Ki_ji = −θ,
//         dE/dθ = Im(x̄_i x_j)  ⇒  gEP = (dEdKi(x^β) − dEdKi(x⁰))/β.
//    5. The nudged relaxation CONTINUES from x⁰ (basin-safe), and Euler
//       relaxation (not Newton) is used so the physical minimum is reached,
//       not a saddle of the energy.
// ============================================================================

export interface FepConfig {
  nIn: number;
  nHidden: number;
  nOut: number;
  /** nudge strength for the contrast phase */
  beta: number;
  /** Euler time step */
  dt: number;
  /** relaxation steps for one free/nudged settle */
  relaxSteps: number;
  seed: number;
  /** amplitude damping  λ  in  E ⊃ 1/2 λ |x|²  */
  lam: number;
  /** diagonal self-coupling (gain): healthy amplitude ≈ √(α−λ) */
  alpha: number;
  /** real input pump: Kr[i, in] = Kr[in, i] = pump  (pins the phases) */
  pump: number;
  /** n° de bandes cachées (profondeur) : couplages uniquement inter-bandes adjacentes */
  bands?: number;
}

export interface FreeEnergyMachine {
  n: number;
  nIn: number;
  nHidden: number;
  nOut: number;
  cfg: FepConfig;
  /** Hermitian coupling:  Kr  symmetric,  Ki  antisymmetric  */
  Kr: Float64Array;
  Ki: Float64Array;
  xr: Float64Array;
  xi: Float64Array;
}

export interface VerifyResult {
  relErr: number;
  checks: number;
  worstPair: { i: number; j: number; part: 'real' | 'imag'; gEP: number; gFD: number };
  beta: number;
  resid: number;
  /** Certificat de monotonie : max ΔE sur les pas (≤0 = flot strictement décroissant) */
  dEmax: number;
  /** Pas réellement consommés par la relaxation adaptative */
  stepsUsed: number;
}

// ---------------------------------------------------------------------------
//  RNG — mulberry32 (deterministic, seedable)
// ---------------------------------------------------------------------------

export function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const DEFAULT_FEP: FepConfig = {
  nIn: 2, nHidden: 5, nOut: 1,
  beta: 0.001, dt: 0.05, relaxSteps: 800, seed: 42,
  lam: 0.1, alpha: 0.6, pump: 0.9,
};

export function makeMachine(cfg: FepConfig): FreeEnergyMachine {
  const r = mulberry(cfg.seed);
  const n = cfg.nIn + cfg.nHidden + cfg.nOut;
  const Kr = new Float64Array(n * n), Ki = new Float64Array(n * n);
  const scale = 0.4 / Math.sqrt(n);
  const nb = cfg.bands && cfg.bands > 0 ? cfg.bands : 0;
  const bandSize = nb ? Math.ceil(cfg.nHidden / nb) : 0;
  const bandOf = (i: number) => (nb ? Math.floor((i - cfg.nIn) / bandSize) : 0);
  const adj = (i: number, j: number) => !nb || Math.abs(bandOf(i) - bandOf(j)) <= 1;
  for (let i = cfg.nIn; i < n; i++) Kr[i * n + i] = cfg.alpha;
  for (let i = cfg.nIn; i < n; i++)
    for (let j = 0; j < cfg.nIn; j++) { Kr[i * n + j] = cfg.pump; Kr[j * n + i] = cfg.pump; }
  for (let i = cfg.nIn; i < n; i++)
    for (let j = cfg.nIn; j < n; j++)
      if (i !== j && adj(i, j) && r() < 0.5) {
        Kr[i * n + j] = (r() * 2 - 1) * scale;
        Ki[i * n + j] = (r() * 2 - 1) * scale;
      }
  // Hermitise: Kr symmetric, Ki antisymmetric
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    Kr[j * n + i] = Kr[i * n + j];
    Ki[j * n + i] = -Ki[i * n + j];
  }
  return { n, nIn: cfg.nIn, nHidden: cfg.nHidden, nOut: cfg.nOut, cfg, Kr, Ki, xr: new Float64Array(n), xi: new Float64Array(n) };
}

// ---------------------------------------------------------------------------
//  Dynamics, loss, energy
// ---------------------------------------------------------------------------

/** Gradient flow of E_β:  ẋ_i = (Kx)_i − (|x_i|²+λ)x_i − β(x_i − y_i)·[output]  */
export function relax(
  m: FreeEnergyMachine,
  inR: Float64Array, inI: Float64Array,
  outR: Float64Array | null, outI: Float64Array | null,
  beta: number, steps?: number, dt?: number,
): void {
  const n = m.n, cfg = m.cfg;
  const st = steps ?? cfg.relaxSteps;
  const h = dt ?? cfg.dt;
  for (let i = 0; i < m.nIn; i++) { m.xr[i] = inR[i]; m.xi[i] = inI[i]; }
  for (let s = 0; s < st; s++) {
    for (let i = m.nIn; i < n; i++) {
      let fr = 0, fi = 0;
      const row = i * n;
      for (let j = 0; j < n; j++) {
        const kc = row + j;
        const xjr = m.xr[j], xji = m.xi[j];
        fr += m.Kr[kc] * xjr - m.Ki[kc] * xji;
        fi += m.Kr[kc] * xji + m.Ki[kc] * xjr;
      }
      const a2 = m.xr[i] * m.xr[i] + m.xi[i] * m.xi[i];
      fr -= (a2 + cfg.lam) * m.xr[i];
      fi -= (a2 + cfg.lam) * m.xi[i];
      if (outR && outI && i >= n - m.nOut) {
        const k = i - (n - m.nOut);
        fr -= beta * (m.xr[i] - outR[k]);
        fi -= beta * (m.xi[i] - outI[k]);
      }
      m.xr[i] += h * fr;
      m.xi[i] += h * fi;
      if (!Number.isFinite(m.xr[i]) || !Number.isFinite(m.xi[i])) {
        for (let q = m.nIn; q < n; q++) {
          m.xr[q] = 0.3 + ((q * 7919) % 100) / 1000;
          m.xi[q] = ((q * 104729) % 100) / 1000;
        }
        break;
      }
    }
  }
}

export function loss(m: FreeEnergyMachine, outR: Float64Array, outI: Float64Array): number {
  let L = 0;
  for (let k = 0; k < m.nOut; k++) {
    const i = m.n - m.nOut + k;
    const dr = m.xr[i] - outR[k], di = m.xi[i] - outI[k];
    L += 0.5 * (dr * dr + di * di);
  }
  return L;
}

export function energy(m: FreeEnergyMachine): number {
  let E = 0;
  const n = m.n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const kc = i * n + j;
      E -= 0.5 * (
        m.xr[i] * (m.Kr[kc] * m.xr[j] - m.Ki[kc] * m.xi[j]) +
        m.xi[i] * (m.Kr[kc] * m.xi[j] + m.Ki[kc] * m.xr[j])
      );
    }
    const a2 = m.xr[i] * m.xr[i] + m.xi[i] * m.xi[i];
    E += 0.25 * a2 * a2;
    E += 0.5 * m.cfg.lam * a2;
  }
  return E;
}

/** Résidu du flot ẋ au point courant (norme max). */
function flowResid(m: FreeEnergyMachine): number {
  let resid = 0;
  for (let i = m.nIn; i < m.n; i++) {
    let fr = 0, fi = 0;
    const row = i * m.n;
    for (let j = 0; j < m.n; j++) {
      const kc = row + j;
      fr += m.Kr[kc] * m.xr[j] - m.Ki[kc] * m.xi[j];
      fi += m.Kr[kc] * m.xi[j] + m.Ki[kc] * m.xr[j];
    }
    const a2 = m.xr[i] * m.xr[i] + m.xi[i] * m.xi[i];
    fr -= (a2 + m.cfg.lam) * m.xr[i];
    fi -= (a2 + m.cfg.lam) * m.xi[i];
    resid = Math.max(resid, Math.hypot(fr, fi));
  }
  return resid;
}

/** Trace E(t) sur la relaxation libre (stride 64) → max ΔE (certificat de monotonie). */
function energyTrace(m: FreeEnergyMachine, inR: Float64Array, inI: Float64Array): number {
  const dt = 0.05, stride = 64, total = 2048;
  for (let i = 0; i < m.nIn; i++) { m.xr[i] = inR[i]; m.xi[i] = inI[i]; }
  let dEmax = 0, prev = energy(m);
  for (let s = 0; s < total; s += stride) {
    relax(m, inR, inI, null, null, 0, stride, dt);
    const E = energy(m);
    const dE = E - prev;
    if (dE > dEmax) dEmax = dE;
    prev = E;
  }
  return dEmax;
}

/** ∂E/∂Kr_ij = −½ Re[x̄_i x_j]  (with Kr_ij = Kr_ji treated as one parameter) */
export function dEdKr(m: FreeEnergyMachine, i: number, j: number): number {
  return -0.5 * (m.xr[i] * m.xr[j] + m.xi[i] * m.xi[j]);
}

/** ∂E/∂Ki_ij = Im(x̄_i x_j)  (Ki antisymmetric, single parameter) */
export function dEdKi(m: FreeEnergyMachine, i: number, j: number): number {
  return m.xr[i] * m.xi[j] - m.xi[i] * m.xr[j];
}

// ---------------------------------------------------------------------------
//  Phase-gradient identity (the grounded check)
// ---------------------------------------------------------------------------

/**
 * Verifies  gEP == gFD  for a sample of couplings (real AND imaginary parts),
 * at a single clamped input / nudged target.  Returns relative error vs the
 * magnitude of the true gradient.
 */
export function verifyPhaseGradient(
  m: FreeEnergyMachine,
  inR: Float64Array, inI: Float64Array,
  outR: Float64Array, outI: Float64Array,
  maxChecks = 12,
): VerifyResult {
  const beta = m.cfg.beta, n = m.n;
  const dt = 0.05;
  // Relaxation ADAPTATIVE : blocs jusqu'à résidu de flot < 1e-10 (toit 20000).
  // L'UI temps réel ne paie plus les 20000 pas quand l'équilibre est atteint tôt.
  let stepsUsed = 0;
  const q = (b: number, o: Float64Array | null) => {
    let used = 0, resid = Infinity, prev = Infinity;
    // 2 critères : résidu < 1e-10 (libre) OU stabilisé (Δ<1e-12 — le résidu du
    // nudge vaut O(β·|x−y|) et ne descend pas sous cette valeur).
    for (let blk = 0; blk < 40; blk++) {
      relax(m, inR, inI, o, o ? outI : null, b, 512, dt);
      used += 512;
      resid = flowResid(m);
      if (resid < 1e-10 || Math.abs(resid - prev) < 1e-12) break;
      prev = resid;
    }
    stepsUsed = Math.max(stepsUsed, used);
  };

  // free equilibrium x⁰ + its dEdK tables
  q(0, null);
  const x0r = m.xr.slice(), x0i = m.xi.slice();
  const dEmax = energyTrace(m, inR, inI);
  const freeD = new Float64Array(n * n), freeDn = new Float64Array(n * n);
  for (let i = m.nIn; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j) {
      freeD[i * n + j] = dEdKr(m, i, j);
      freeDn[i * n + j] = dEdKi(m, i, j);
    }

  const r = mulberry(7);
  let worst = 0, nrm = 1e-12, checks = 0;
  let worstPair: VerifyResult['worstPair'] = { i: 0, j: 0, part: 'real', gEP: 0, gFD: 0 };
  const tried = new Set<number>();
  while (checks < maxChecks) {
    const i = m.nIn + Math.floor(r() * (n - m.nIn));
    const j = Math.floor(r() * n);
    if (i === j || tried.has(i * n + j)) continue;
    tried.add(i * n + j);

    // nudged equilibrium by CONTINUATION from x⁰
    for (let qq = 0; qq < n; qq++) { m.xr[qq] = x0r[qq]; m.xi[qq] = x0i[qq]; }
    q(beta, outR);

    const gEP = 2 * (dEdKr(m, i, j) - freeD[i * n + j]) / beta;
    const gEPi = (dEdKi(m, i, j) - freeDn[i * n + j]) / beta;

    // finite differences on the Hermitian pairs
    q(0, null);
    const or1 = m.Kr[i * n + j];
    m.Kr[i * n + j] = or1 + 1e-6; m.Kr[j * n + i] = or1 + 1e-6; q(0, null); const lp = loss(m, outR, outI);
    m.Kr[i * n + j] = or1 - 1e-6; m.Kr[j * n + i] = or1 - 1e-6; q(0, null); const lm = loss(m, outR, outI);
    m.Kr[i * n + j] = or1; m.Kr[j * n + i] = or1;
    const gFD = (lp - lm) / 2e-6;

    const oi1 = m.Ki[i * n + j];
    m.Ki[i * n + j] = oi1 + 1e-6; m.Ki[j * n + i] = -oi1 - 1e-6; q(0, null); const li1 = loss(m, outR, outI);
    m.Ki[i * n + j] = oi1 - 1e-6; m.Ki[j * n + i] = -oi1 + 1e-6; q(0, null); const li2 = loss(m, outR, outI);
    m.Ki[i * n + j] = oi1; m.Ki[j * n + i] = -oi1;
    const gFDi = (li1 - li2) / 2e-6;

    const scale = Math.max(Math.abs(gFD), Math.abs(gFDi), 1e-3);
    const pairs: [number, 'real' | 'imag', number, number][] = [
      [Math.abs(gEP - gFD), 'real', gEP, gFD],
      [Math.abs(gEPi - gFDi), 'imag', gEPi, gFDi],
    ];
    for (const [rel, part, gEPv, gFDv] of pairs) {
      if (rel > worst) {
        worst = rel;
        worstPair = { i, j, part, gEP: gEPv, gFD: gFDv };
      }
    }
    nrm = Math.max(nrm, scale);
    checks++;
  }

  q(0, null);
  const resid = flowResid(m);

  return { relErr: worst / nrm, checks, worstPair, beta, resid, dEmax, stepsUsed };
}

// ---------------------------------------------------------------------------
//  Learning by equilibrium contrast (no backpropagation)
// ---------------------------------------------------------------------------

/** État AdamW pour les couplages (optionnel, persistant sur l'entraînement). */
export interface AdamState { mR: Float64Array; mI: Float64Array; vR: Float64Array; vI: Float64Array; t: number; }
export function makeAdamState(n2: number): AdamState {
  return { mR: new Float64Array(n2), mI: new Float64Array(n2), vR: new Float64Array(n2), vI: new Float64Array(n2), t: 0 };
}

/**
 * One contrast step for one training example: settle to x⁰, then to x^β under
 * the output nudge, and update every hidden↔hidden / hidden→output coupling
 * along the EP gradient.  The input pump and the diagonal gain stay fixed.
 * `st` optionnel : AdamW (b1=0.9, b2=0.999, eps=1e-8).
 */
export function epUpdate(
  m: FreeEnergyMachine,
  inR: Float64Array, inI: Float64Array,
  outR: Float64Array, outI: Float64Array,
  eta: number, decay = 1e-3, st?: AdamState,
): void {
  const beta = m.cfg.beta, n = m.n;
  relax(m, inR, inI, null, null, 0);
  const freeKr = new Float64Array(n * n), freeKi = new Float64Array(n * n);
  for (let i = m.nIn; i < n; i++)
    for (let j = m.nIn; j < n; j++) {
      if (i === j) continue;
      freeKr[i * n + j] = dEdKr(m, i, j);
      freeKi[i * n + j] = dEdKi(m, i, j);
    }
  relax(m, inR, inI, outR, outI, beta);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  if (st) st.t += 1;
  const bc1 = st ? 1 - Math.pow(b1, st.t) : 1;
  const bc2 = st ? 1 - Math.pow(b2, st.t) : 1;
  for (let i = m.nIn; i < n; i++)
    for (let j = m.nIn; j < n; j++) {
      if (i === j) continue;
      const gR = 2 * (dEdKr(m, i, j) - freeKr[i * n + j]) / beta;
      const gI = (dEdKi(m, i, j) - freeKi[i * n + j]) / beta;
      const kr = i * n + j, kj = j * n + i;
      if (st) {
        // AdamW : moments → pas avec correction de biais → decay multiplicative
        const moR = b1 * st.mR[kr] + (1 - b1) * gR; st.mR[kr] = moR;
        const voR = b2 * st.vR[kr] + (1 - b2) * gR * gR; st.vR[kr] = voR;
        m.Kr[kr] = m.Kr[kr] * (1 - decay) - (eta * (moR / bc1)) / (Math.sqrt(voR / bc2) + eps);
        const moI = b1 * st.mI[kr] + (1 - b1) * gI; st.mI[kr] = moI;
        const voI = b2 * st.vI[kr] + (1 - b2) * gI * gI; st.vI[kr] = voI;
        m.Ki[kr] = m.Ki[kr] * (1 - decay) - (eta * (moI / bc1)) / (Math.sqrt(voI / bc2) + eps);
        // miroir hermitien (cohérent avec l'init : Ki[ji]=−Ki[ij])
        m.Kr[kj] = m.Kr[kr];
        m.Ki[kj] = -m.Ki[kr];
      } else {
        m.Kr[kr] = m.Kr[kr] * (1 - decay) - eta * gR;
        m.Kr[kj] = m.Kr[kj] * (1 - decay) - eta * gR;
        m.Ki[kr] = m.Ki[kr] * (1 - decay) - eta * gI;
        m.Ki[kj] = m.Ki[kj] * (1 - decay) + eta * gI;
      }
    }
}

// ---------------------------------------------------------------------------
//  Temporal prediction task (autoregressive, non-linear series)
// ---------------------------------------------------------------------------

export function makeSeries(n: number, seed: number): Float64Array {
  const r = mulberry(seed);
  const y = new Float64Array(n);
  y[0] = 0.5; y[1] = 0.5;
  for (let t = 2; t < n; t++)
    y[t] = 0.55 + 0.42 * Math.sin(2.6 * y[t - 1] + 0.4 * y[t - 2]) + 0.03 * (r() - 0.5);
  return y;
}

export interface TrainResult {
  loss: Float64Array;
  vfe: Float64Array;
  epochs: number;
  startLoss: number;
  finalLoss: number;
  reduction: number;
}

export interface EvalResult {
  loss: number;
  vfe: number;
}

/** Average prediction loss over the series window (free settling, no nudge). */
export function evalLoss(
  m: FreeEnergyMachine,
  series: Float64Array,
  from: number,
  count: number,
): EvalResult {
  let tot = 0;
  const inR = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
  const outR = new Float64Array(m.nOut), outI = new Float64Array(m.nOut);
  for (let t = from; t < from + count; t++) {
    for (let k = 0; k < m.nIn; k++) {
      inR[k] = 0.1 + 0.8 * series[t - 1 - k];
      inI[k] = 0.05 * Math.sin(t * k + 1);
    }
    outR[0] = series[t]; outI[0] = 0;
    relax(m, inR, inI, null, null, 0);
    tot += loss(m, outR, outI);
  }
  return { loss: tot / count, vfe: energy(m) };
}

export interface TrainConfig {
  epochs: number;
  eta: number;
  decay: number;
  seriesSeed: number;
  nSeries: number;
  window: number;
  /** AdamW sur les couplages (défaut : SGD+decay) */
  adam?: boolean;
}

export const DEFAULT_TRAIN: TrainConfig = {
  epochs: 12, eta: 0.004, decay: 1e-3, seriesSeed: 7, nSeries: 400, window: 180,
};

/** Trains the machine by pure equilibrium contrast; records loss per epoch. */
export function trainFep(
  m: FreeEnergyMachine,
  tc: TrainConfig,
  onEpoch?: (ep: number, loss: number, vfe: number) => void,
): TrainResult {
  const series = makeSeries(tc.nSeries, tc.seriesSeed);
  const start = evalLoss(m, series.subarray(4), 0, tc.window);
  const lossArr = new Float64Array(tc.epochs + 1);
  const vfeArr = new Float64Array(tc.epochs + 1);
  lossArr[0] = start.loss; vfeArr[0] = start.vfe;

  const inT = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
  const oR = new Float64Array(m.nOut), oI = new Float64Array(m.nOut);
  const st = tc.adam ? makeAdamState(m.n * m.n) : undefined;
  for (let ep = 1; ep <= tc.epochs; ep++) {
    for (let t = 4; t < 4 + tc.window; t++) {
      for (let k = 0; k < m.nIn; k++) {
        inT[k] = 0.1 + 0.8 * series[t - 1 - k];
        inI[k] = 0.05 * Math.sin(t * k + 1);
      }
      oR[0] = series[t]; oI[0] = 0;
      epUpdate(m, inT, inI, oR, oI, tc.eta, tc.decay, st);
    }
    const e = evalLoss(m, series.subarray(4), 0, tc.window);
    lossArr[ep] = e.loss; vfeArr[ep] = e.vfe;
    onEpoch?.(ep, e.loss, e.vfe);
  }

  return {
    loss: lossArr, vfe: vfeArr, epochs: tc.epochs,
    startLoss: start.loss, finalLoss: lossArr[tc.epochs],
    reduction: lossArr[tc.epochs] > 0 ? start.loss / lossArr[tc.epochs] : 0,
  };
}