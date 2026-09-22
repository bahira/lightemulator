// ============================================================================
//  control.ts — Contrôle non-linéaire à forme fermée, audité.
//
//  Issu de l'audit des rapports SPEAR v2.2 / PMP (falsification numérique
//  indépendante). Ce qui est ici est exact ou honnêtement borné :
//    · IK SCARA 2 liaisons  : forme fermée EXACTE (loi des cosinus).
//    · Trajectoire jerk-borné : profil 7 segments FERMÉ, 3 régimes exacts —
//      le cas "T2<0" du rapport source était FAUX et est corrigé ici.
//    · Pendule inversé       : loi π* conservée, mais la preuve « V̇ = −6.4 θ̇² ≤ 0 »
//      du papier est FAUSSE (résidu −0.4·θ̇·sinθ) ; on expose V̇ exacte.
//
//  Chaque affirmation est re-vérifiée dans validate.ts (tests 'ctrl-*').
// ============================================================================

// ---------------------------------------------------------------------------
//  1. Cinématique inverse SCARA 2 liaisons — 0 itération
//     cos θ₂ = (r² − L1² − L2²)/(2·L1·L2), exactement affine en r².
// ---------------------------------------------------------------------------

export interface ScaraPose { th1: number; th2: number; ok: boolean }

/** Loi des cosinus : cos θ₂ en fonction de r² = x²+z² — affine exacte. */
export const cosTheta2 = (r2: number, L1: number, L2: number): number =>
  (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);

/** Géométrie directe. */
export function fkScara(th1: number, th2: number, L1: number, L2: number): { x: number; z: number } {
  return {
    x: L1 * Math.cos(th1) + L2 * Math.cos(th1 + th2),
    z: L1 * Math.sin(th1) + L2 * Math.sin(th1 + th2),
  };
}

/** IK fermée, solution coudée haute (th2 ∈ [0, π]). ok=false si hors atteinte. */
export function ikScara(x: number, z: number, L1: number, L2: number): ScaraPose {
  const c2 = cosTheta2(x * x + z * z, L1, L2);
  if (!(c2 >= -1 && c2 <= 1)) return { th1: NaN, th2: NaN, ok: false };
  const th2 = Math.acos(c2);
  const th1 = Math.atan2(z, x) - Math.atan2(L2 * Math.sin(th2), L1 + L2 * Math.cos(th2));
  return { th1, th2, ok: true };
}

// ---------------------------------------------------------------------------
//  2. Trajectoire temps-minimale à jerk borné (triple intégrateur, PMP)
//     j* = −j_max·sgn(λ₃), λ₃ parabole → ≤ 2 commutations → 7 segments.
//
//     Correction vs rapport source : le régime où NI v_max NI a_max ne sont
//     atteints a pour solution exacte a_p = (D·j_max²/2)^{1/3} (le rapport
//     extrapolait la formule à palier, résidu mesuré jusqu'à 34 %).
// ---------------------------------------------------------------------------

export interface ProfileParams { D: number; vmax: number; amax: number; jmax: number }

export type ProfileRegime = 'reduit' | 'triangulaire' | 'palier';

export interface ProfilePlan {
  regime: ProfileRegime;
  /** durées des 7 segments [T1..T7] */
  seg: [number, number, number, number, number, number, number];
  /** jerk de chaque segment (signé selon D) */
  jerk: [number, number, number, number, number, number, number];
  vPeak: number;
  aPeak: number;
  duration: number;
}

/** Planificateur fermé : durées exactes, zéro itération, zéro solveur NLP.
 *
 *  Clé de la correction : l'accélération réellement exploitable est
 *  a_eff = min(a_max, √(v_max·j)) — au-delà de √(v_max·j), un profil pur
 *  jerk dépasse v_max avant même d'avoir un plateau d'accélération.
 *  Le régime « triangulaire » du rapport source n'existe que si a_max ≤ a_eff,
 *  sinon c'est un PALIER DE VITESSE À ACCÉLÉRATION RÉDUITE (cas absent du
 *  rapport, dont la formule v_peak violait v_max dans ~15 % des tirages). */
export function planProfile(p: ProfileParams): ProfilePlan {
  const sgn = p.D < 0 ? -1 : 1;
  const D = Math.abs(p.D);
  const { vmax, amax, jmax } = p;

  const aEff = Math.min(amax, Math.sqrt(vmax * jmax));   // accélération exploitable
  const Dsat = (2 * aEff ** 3) / jmax ** 2;              // mini pour saturer a_eff
  const Dcrz = (vmax * vmax) / amax + (amax * vmax) / jmax; // mini pour saturer v_max via a_max

  let seg: [number, number, number, number, number, number, number];
  let vPeak: number, aPeak: number, regime: ProfileRegime;

  if (D <= Dsat) {
    // rien ne sature : accélération crête réduite, profil symétrique pur
    regime = 'reduit';
    aPeak = Math.cbrt((D * jmax * jmax) / 2);
    vPeak = (aPeak * aPeak) / jmax;
    const T = aPeak / jmax;
    seg = [T, 0, T, 0, T, 0, T];
  } else if (aEff === amax && D <= Dcrz) {
    // a_max saturé, vitesse triangulaire (v_peak < v_max) — formule corrigée
    regime = 'triangulaire';
    aPeak = amax;
    vPeak = amax * Math.sqrt(D / amax + (amax * amax) / (4 * jmax * jmax)) - (amax * amax) / (2 * jmax);
    const T1 = amax / jmax;
    // plateau d'accélération requis pour atteindre v_peak > a_max²/j
    const T2 = Math.max(0, (vPeak - (amax * amax) / jmax) / amax);
    seg = [T1, T2, T1, 0, T1, T2, T1];
  } else {
    // palier de vitesse, accélération crête a_eff (≤ a_max), durées exactes :
    // D_acc = 2a³/j² + 3a²T₂/j + a·T₂²  avec  T₂ = (v_max − a²/j)/a  (≥ 0)
    regime = 'palier';
    aPeak = aEff; vPeak = vmax;
    const a = aEff;
    const T1 = a / jmax;
    const T2 = Math.max(0, (vmax - (a * a) / jmax) / a);
    const Dacc = (2 * a ** 3) / jmax ** 2 + (3 * a * a * T2) / jmax + a * T2 * T2;
    seg = [T1, T2, T1, Math.max(0, (D - Dacc) / vmax), T1, T2, T1];
  }

  const jerk: ProfilePlan['jerk'] = [
    sgn * jmax, 0, -sgn * jmax, 0, -sgn * jmax, 0, sgn * jmax,
  ];
  const duration = seg.reduce((a, b) => a + b, 0);
  return { regime, seg, jerk, vPeak, aPeak, duration };
}

/** État exact [x, v, a] au temps t — intégration fermée segment par segment. */
export function profileState(plan: ProfilePlan, t: number): { x: number; v: number; a: number } {
  let x = 0, v = 0, a = 0, rem = t;
  for (let i = 0; i < 7; i++) {
    if (rem <= 0) break;
    const dt = Math.min(rem, plan.seg[i]);
    const j = plan.jerk[i];
    x += v * dt + (a * dt * dt) / 2 + (j * dt * dt * dt) / 6;
    v += a * dt + (j * dt * dt) / 2;
    a += j * dt;
    rem -= dt;
  }
  return { x, v, a };
}

// ---------------------------------------------------------------------------
//  3. Pendule inversé  θ̈ = α·sinθ + β·u, |u| ≤ u_max
//     Loi découverte par évolution : u = sat(−(k1·sinθ + k2·θ̇)).
//
//     Audit : la preuve du papier (« V̇ = −β·k2·θ̇² ≤ 0 sur tout le plan »)
//     est fausse — 2α − β·k1 = −0.4 ≠ 0 pour les paramètres annoncés.
//     V̇ exacte ci-dessous ; la stabilité locale reste vraie (linéarisé),
//     elle est vérifiée par rollout dans validate.ts.
// ---------------------------------------------------------------------------

export interface PendulumLaw { alpha: number; beta: number; umax: number; k1: number; k2: number }

/** Paramètres du papier SPEAR v2.2 (α=6, β=4, |u|≤2, k1=3.1, k2=1.6). */
export const DEFAULT_PENDULUM: PendulumLaw = { alpha: 6, beta: 4, umax: 2, k1: 3.1, k2: 1.6 };

export function pendulumU(th: number, thd: number, law: PendulumLaw): number {
  const raw = -(law.k1 * Math.sin(th) + law.k2 * thd);
  return raw > law.umax ? law.umax : raw < -law.umax ? -law.umax : raw;
}

/** Énergie candidat V = ½θ̇² + α(1−cosθ) (min au sommet θ=0). */
export function pendulumV(th: number, thd: number, law: PendulumLaw): number {
  return 0.5 * thd * thd + law.alpha * (1 - Math.cos(th));
}

/** Dérivée EXACTE le long des trajectoires : V̇ = θ̇·(2α·sinθ + β·u). */
export function pendulumVdot(th: number, thd: number, law: PendulumLaw): number {
  return thd * (2 * law.alpha * Math.sin(th) + law.beta * pendulumU(th, thd, law));
}

/** Valeurs propres du linéarisé au sommet : λ² + βk₂λ + (βk₁−α) = 0. */
export function pendulumEigen(law: PendulumLaw): { re: number; im: number }[] {
  const b = law.beta * law.k2;
  const c = law.beta * law.k1 - law.alpha; // raideur effective (stabilité ssi > 0)
  const disc = b * b - 4 * c;
  if (disc >= 0) {
    const r = Math.sqrt(disc);
    return [{ re: (-b + r) / 2, im: 0 }, { re: (-b - r) / 2, im: 0 }];
  }
  const im = Math.sqrt(-disc) / 2;
  return [{ re: -b / 2, im }, { re: -b / 2, im: -im }];
}

/** Rollout boucle fermée (Euler semi-implicite, pas fixe). */
export function simulatePendulum(
    law: PendulumLaw, th0: number, thd0: number, steps: number, dt = 1e-4,
): { th: Float64Array; thd: Float64Array } {
    const thArr = new Float64Array(steps);
    const thdArr = new Float64Array(steps);
    let th = th0, thd = thd0;
    for (let i = 0; i < steps; i++) {
        thArr[i] = th; thdArr[i] = thd;
        const acc = law.alpha * Math.sin(th) + law.beta * pendulumU(th, thd, law);
        thd += acc * dt;
        th += thd * dt;
    }
    return { th: thArr, thd: thdArr };
}

// ---------------------------------------------------------------------------
//  Swing-up global — falsification du contrôleur hybride « V1200 §4 ».
//
//  Résultat mesuré (tools/hybrid_swingup_verify.mjs, 15 seeds × 3 autorités) :
//    π* simple gagne PARTOUT (15/15) dès que β·u_max > α : le couple saturé
//    bat la gravité en chaque point, le pendule « roule » vers le haut.
//    L'hybride energy-shaping (terme E_err·cosθ qui s'annule à π/2) fait
//    pire (3–11/15). La dichotomie « pompage requis » est fause ici.
// ---------------------------------------------------------------------------

export interface HybridLaw {
    alpha: number; beta: number; umax: number;
    kSwing: number; kp: number; kd: number; wcGain: number; wcOff: number;
}

export const DEFAULT_HYBRID: HybridLaw = {
    alpha: 6, beta: 4, umax: 2,
    kSwing: 4.3278, kp: 1.7222, kd: 8.0402, wcGain: 10.1786, wcOff: 0.70,
};

export function pendulumUHybrid(th: number, thd: number, L: HybridLaw): number {
    const s = Math.sin(th), c = Math.cos(th);
    const eErr = 0.5 * thd * thd + L.alpha * (1 - c) - 2 * L.alpha; // cible E_top = 2α
    const uSwing = -L.kSwing * thd * eErr * c;
    const uCatch = -(L.kp * s + L.kd * thd);
    const w = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, L.wcGain * (c - L.wcOff)))));
    return clampSym((1 - w) * uSwing + w * uCatch, L.umax);
}
function clampSym(v: number, lim: number) { return v < -lim ? -lim : v > lim ? lim : v; }

type UFn = (th: number, thd: number) => number;

/** Rollout générique : dynamique θ̈ = α·sinθ + β·u(th,θ̇), Euler semi-implicite. */
export function simulatePendulumLaw(
    alpha: number, beta: number, uFn: UFn,
    th0: number, thd0: number, steps: number, dt = 1e-4,
): { th: Float64Array; thd: Float64Array } {
    const thArr = new Float64Array(steps);
    const thdArr = new Float64Array(steps);
    let th = th0, thd = thd0;
    for (let i = 0; i < steps; i++) {
        thArr[i] = th; thdArr[i] = thd;
        const acc = alpha * Math.sin(th) + beta * uFn(wrapPi(th), thd);
        thd += acc * dt;
        th += thd * dt;
    }
    return { th: thArr, thd: thdArr };
}
function wrapPi(th: number): number {
    let d = th % (2 * Math.PI);
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    return d;
}

/** Succès de swing-up : verrouillé au sommet ≥ holdSec avant T. */
export function swingUpSuccess(
    alpha: number, beta: number, uFn: UFn,
    th0: number, T = 12, dt = 2e-4, holdSec = 0.5,
): boolean {
    let th = th0, thd = 0, okRun = 0;
    const need = Math.round(holdSec / dt);
    for (let i = 0; i < Math.round(T / dt); i++) {
        const acc = alpha * Math.sin(th) + beta * uFn(wrapPi(th), thd);
        thd += acc * dt;
        th += thd * dt;
        const dev = Math.abs(wrapPi(th));
        if (dev < 0.15 && Math.abs(thd) < 0.5) { okRun++; if (okRun >= need) return true; }
        else okRun = 0;
    }
    return false;
}
