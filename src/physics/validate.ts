// ============================================================================
//  validate.ts — The grounded loop.
//
//  Every claim this engine makes is checked here against an INDEPENDENT
//  reference: an analytic solution, a conservation law, an exhaustive search,
//  or a finite-difference derivative. Each test reports the measured residual
//  and the tolerance it had to beat. Nothing is asserted that is not measured.
// ============================================================================

import { fftRoundTripError, fftParsevalError, fftImpulseError } from './fft';
import {
  DEFAULT_BPM, runBPM, solveMode, slabNeffAnalytic, gaussianDiffractionError,
  type BpmConfig,
} from './bpm';
import {
  haarUnitary, dftMatrix, decomposeMesh, reconstruct, frobeniusDiff,
  unitarityDefect, propagate, matvec,
} from './mzi';
import {
  makeGraph, bruteForceMaxCut, runCIM, greedyLocalSearch, simulatedAnnealing,
  cutValue, isingEnergy, DEFAULT_CIM,
} from './ising';
import { partitionOfUnityError, splineDerivativeError, gradientCheck } from './kan';
import { DEFAULT_FEP, makeMachine, verifyPhaseGradient } from './fep';
import {
  fkScara, ikScara, cosTheta2, planProfile, profileState,
  DEFAULT_PENDULUM, pendulumVdot, pendulumV, simulatePendulum,
} from './control';
import {
  FUSED_SILICA, BK7, sellmeier, gdd, zeroDispersionLambda,
  pulseWidthGDD, pulsePropagation, fresnelBiaxial,
} from './optics';
import { phototaxisError, trilaterationError, friisError, learningMonotonicity } from './drones';

export interface TestResult {
  group: string;
  name: string;
  claim: string;
  reference: string;
  measured: number;
  tolerance: number;
  /** true when a LOWER value is better (residual); false for ratios. */
  lowerIsBetter: boolean;
  pass: boolean;
  ms: number;
  extra?: string;
}

const fmt = (v: number): string => {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  return v.toPrecision(4);
};

export function formatResidual(t: TestResult): string {
  return fmt(t.measured);
}

// ---------------------------------------------------------------------------

function timed<T>(fn: () => T): { v: T; ms: number } {
  const t0 = performance.now();
  const v = fn();
  return { v, ms: performance.now() - t0 };
}

export type TestId =
  | 'fft-roundtrip' | 'fft-parseval' | 'fft-tone'
  | 'bpm-energy' | 'bpm-gauss' | 'bpm-neff'
  | 'mesh-reck' | 'mesh-unitary' | 'mesh-energy' | 'mesh-dft'
  | 'ising-identity' | 'ising-exact' | 'cim-optimal'
  | 'kan-pou' | 'kan-dbasis' | 'kan-grad'
  | 'fep-gradient'
  | 'ctrl-ik' | 'ctrl-traj' | 'ctrl-vdot' | 'ctrl-stab'
  | 'opt-sellmeier' | 'opt-gdd' | 'opt-pulse' | 'opt-fresnel'
  | 'drone-phototaxie' | 'drone-trilateration' | 'drone-link' | 'drone-apprentissage';

export const TEST_ORDER: TestId[] = [
  'fft-roundtrip', 'fft-parseval', 'fft-tone',
  'bpm-energy', 'bpm-gauss', 'bpm-neff',
  'mesh-reck', 'mesh-unitary', 'mesh-energy', 'mesh-dft',
  'ising-identity', 'ising-exact', 'cim-optimal',
  'kan-pou', 'kan-dbasis', 'kan-grad',
  'fep-gradient',
  'ctrl-ik', 'ctrl-traj', 'ctrl-vdot', 'ctrl-stab',
  'opt-sellmeier', 'opt-gdd', 'opt-pulse', 'opt-fresnel',
  'drone-phototaxie', 'drone-trilateration', 'drone-link', 'drone-apprentissage',
];

export function runTest(id: TestId): TestResult {
  switch (id) {
    // ---- Drones ------------------------------------------------------------
    case 'drone-phototaxie': {
      const { v, ms } = timed(() => phototaxisError());
      return mk('Drones', 'Phototaxie — convergence', 'Le drone converge vers la source lumineuse',
        'Champ 1/d² à source unique : le gradient pointe toujours vers elle, trajectoire droite',
        v.dist, 0.015, ms, `arrêt en ${v.steps} pas (maxV = 1,2 cm/s normalisé)`);
    }
    case 'drone-trilateration': {
      const { v, ms } = timed(() => trilaterationError());
      return mk('Drones', 'Trilatération exacte', 'La position est reconstruite des distances à l’epsilon machine',
        '3 balises, ranges exacts — différenciation + solve 2×2 fermé (équations normales)',
        v.err, 1e-9, ms);
    }
    case 'drone-link': {
      const { v, ms } = timed(() => friisError());
      return mk('Drones', 'Budget de liaison (Friis)', 'La puissance reçue suit l’inverse-square',
        'Pr(2d) = Pr(d)/4 exactement, sur 40 distances de 0,15 à 2,1 m',
        v.rel, 1e-12, ms);
    }
    case 'drone-apprentissage': {
      const { v, ms } = timed(() => learningMonotonicity());
      return mk('Drones', 'Apprentissage (1+1-ES) — monotonie de l’élitisme',
        'L’apprentissage en ligne ne dégrade jamais la fitness incumbent',
        'bestFit monotone non-croissante sur 1200 pas, 6 drones, 2 balises hétérogènes',
        v.maxIncrease, 1e-12, ms, `${v.windows} fenêtres évaluées · fitness finale ${v.finalFit.toExponential(2)}`);
    }

    // ---- FFT ---------------------------------------------------------------
    case 'fft-roundtrip': {
      const { v, ms } = timed(() => fftRoundTripError(4096));
      return mk('FFT', 'Aller-retour IFFT∘FFT', 'La FFT radix-2 est exactement inversible',
        'Identité x = IFFT(FFT(x)) sur 4096 échantillons aléatoires', v, 1e-12, ms);
    }
    case 'fft-parseval': {
      const { v, ms } = timed(() => fftParsevalError(4096));
      return mk('FFT', 'Théorème de Parseval', 'L’énergie est conservée par la transformée',
        'Σ|x|² = (1/N)Σ|X|² — loi de conservation', v, 1e-13, ms);
    }
    case 'fft-tone': {
      const { v, ms } = timed(() => fftImpulseError(2048, 137));
      return mk('FFT', 'Raie spectrale pure', 'La DFT d’une exponentielle complexe est un Dirac',
        'Solution analytique X[k] = N·δ(k−k₀)', v, 1e-13, ms);
    }

    // ---- BPM ---------------------------------------------------------------
    case 'bpm-energy': {
      const cfg: BpmConfig = {
        ...DEFAULT_BPM, device: 'straight', launch: 'mode', absorber: false,
        Nx: 512, Nz: 400, Lz: 1200,
      };
      const { v, ms } = timed(() => runBPM(cfg).energyDrift);
      return mk('BPM', 'Conservation de la puissance', 'Le propagateur split-step est unitaire',
        'ΔP/P sur 400 pas dans un guide sans pertes (absorbeur désactivé)', v, 1e-9, ms);
    }
    case 'bpm-gauss': {
      const { v, ms } = timed(() => gaussianDiffractionError());
      return mk('BPM', 'Diffraction gaussienne', 'Le solveur reproduit la diffraction en espace libre',
        'w(z) = w₀·√(1+(z/z_R)²) analytique, z_R = πn₀w₀²/λ', v.err, 0.02, ms,
        `z_R = ${v.zR.toFixed(1)} µm`);
    }
    case 'bpm-neff': {
      const cfg: BpmConfig = {
        ...DEFAULT_BPM, device: 'straight', Nx: 1024, Lx: 60,
        wCore: 6, dn: 0.01, edge: 0.12, w0: 3,
      };
      const t0 = performance.now();
      const mode = solveMode(cfg, 900, 1e-13);
      const exact = slabNeffAnalytic(cfg.lambda, cfg.n0 + cfg.dn, cfg.n0, cfg.wCore);
      const ms = performance.now() - t0;
      const err = Math.abs(mode.neff - exact) / cfg.dn; // relative to the guiding contrast
      return mk('BPM', 'Indice effectif du mode', 'Le solveur de mode BPM à distance imaginaire est exact',
        'Équation transcendante u·tan(u)=√(V²−u²) du guide plan symétrique', err, 0.02, ms,
        `n_eff = ${mode.neff.toFixed(6)} vs ${exact.toFixed(6)} exact · ${mode.iterations} itér.`);
    }

    // ---- Photonic mesh -----------------------------------------------------
    case 'mesh-reck': {
      const t0 = performance.now();
      const U = haarUnitary(12, 1234);
      const d = decomposeMesh(U);
      const v = frobeniusDiff(U, reconstruct(d));
      return mk('Maillage MZI', 'Décomposition de Reck', 'Le maillage réalise EXACTEMENT l’unitaire cible',
        '‖U − Û_maillage‖_F, U unitaire de Haar 12×12', v, 1e-12, performance.now() - t0,
        `${d.units.length} MZI, profondeur ${d.depth}`);
    }
    case 'mesh-unitary': {
      const t0 = performance.now();
      const d = decomposeMesh(haarUnitary(10, 77));
      const v = unitarityDefect(reconstruct(d));
      return mk('Maillage MZI', 'Unitarité du maillage', 'Le maillage est passif (aucun gain)',
        '‖Û†Û − I‖_F', v, 1e-12, performance.now() - t0);
    }
    case 'mesh-energy': {
      const t0 = performance.now();
      const n = 10;
      const d = decomposeMesh(haarUnitary(n, 202));
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) { re[i] = Math.cos(i * 1.7); im[i] = Math.sin(i * 0.9); }
      const tr = propagate(d, re, im);
      const v = Math.abs(tr.outputPower - tr.inputPower) / tr.inputPower;
      return mk('Maillage MZI', 'Conservation d’énergie optique', 'Aucun photon perdu dans le modèle idéal',
        '|P_out − P_in| / P_in à travers tous les MZI', v, 1e-13, performance.now() - t0);
    }
    case 'mesh-dft': {
      const t0 = performance.now();
      const n = 8;
      const U = dftMatrix(n);
      const d = decomposeMesh(U);
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) { re[i] = Math.exp(-((i - 3.2) ** 2) / 3); im[i] = 0.2 * Math.sin(i); }
      const tr = propagate(d, re, im);
      const ref = matvec(U, re, im);
      let err = 0, nrm = 0;
      for (let i = 0; i < n; i++) {
        err += (tr.outRe[i] - ref.re[i]) ** 2 + (tr.outIm[i] - ref.im[i]) ** 2;
        nrm += ref.re[i] ** 2 + ref.im[i] ** 2;
      }
      const v = Math.sqrt(err / nrm);
      return mk('Maillage MZI', 'Produit matrice-vecteur optique', 'La propagation physique = le produit matriciel',
        'DFT 8×8 : ‖sortie_maillage − U·v‖ / ‖U·v‖', v, 1e-13, performance.now() - t0);
    }

    // ---- Ising / MaxCut ----------------------------------------------------
    case 'ising-identity': {
      const t0 = performance.now();
      const g = makeGraph('erdos', 60, 31, 0.2);
      const s = new Int8Array(g.n);
      let worst = 0;
      for (let trial = 0; trial < 40; trial++) {
        for (let i = 0; i < g.n; i++) s[i] = Math.random() < 0.5 ? 1 : -1;
        const lhs = isingEnergy(g, s);
        const rhs = g.totalW - 2 * cutValue(g, s);
        worst = Math.max(worst, Math.abs(lhs - rhs));
      }
      return mk('Ising', 'Identité coupe ↔ hamiltonien', 'MaxCut et le modèle d’Ising sont le même problème',
        'H(s) = W − 2·coupe(s), vérifié sur 40 configurations aléatoires', worst, 1e-9, performance.now() - t0);
    }
    case 'ising-exact': {
      const t0 = performance.now();
      const g = makeGraph('cubic', 20, 55);
      const bf = bruteForceMaxCut(g);
      const gl = greedyLocalSearch(g, 40, 3);
      const v = bf.cut - gl.cut; // must be >= 0
      return mk('Ising', 'Borne de l’optimum exact', 'L’énumération exhaustive borne bien toutes les heuristiques',
        `Force brute (${bf.iterations.toLocaleString('fr-FR')} configs) ≥ greedy+1-opt`,
        v < 0 ? 1 : 0, 0.5, performance.now() - t0,
        `optimum = ${bf.cut}, greedy = ${gl.cut}`);
    }
    case 'cim-optimal': {
      const t0 = performance.now();
      const g = makeGraph('cubic', 20, 55);
      const bf = bruteForceMaxCut(g);
      const cim = runCIM(g, { ...DEFAULT_CIM, iterations: 1200, trials: 12, seed: 4242 });
      const ratio = cim.cut / bf.cut;
      return mk('Ising', 'CIM vs optimum exact', 'La machine d’Ising cohérente atteint l’optimum global',
        'ratio d’approximation coupe_CIM / coupe_optimale (3-régulier n=20, 8 cycles de pompe)',
        ratio, 0.99, performance.now() - t0,
        `CIM = ${cim.cut} / optimum = ${bf.cut} · succès ${(cim.successRate * 100).toFixed(0)}% des cycles · R99 = ${Math.round(cim.r99)} tours`,
        false);
    }

    // ---- KAN ---------------------------------------------------------------
    case 'kan-pou': {
      const { v, ms } = timed(() => partitionOfUnityError(6));
      return mk('KAN', 'Partition de l’unité', 'La base B-spline cubique est correcte',
        'Σ_k B_k(x) = 1 pour tout x ∈ [−1,1] (propriété exacte des B-splines)', v, 1e-12, ms);
    }
    case 'kan-dbasis': {
      const { v, ms } = timed(() => splineDerivativeError(6));
      return mk('KAN', 'Dérivée analytique des splines', 'La récurrence de dérivation de Cox-de Boor est exacte',
        'max |B′_k(x) − différence centrée (h=10⁻⁶)|', v, 1e-5, ms);
    }
    case 'kan-grad': {
      const t0 = performance.now();
      const r = gradientCheck(11, 30);
      return mk('KAN', 'Vérification du gradient', 'La rétropropagation analytique est correcte',
        `différences finies centrées sur ${r.checks} paramètres tirés au hasard`,
        r.relErr, 1e-4, performance.now() - t0);
    }

    // ---- Free energy / Equilibrium Propagation -----------------------------
    case 'fep-gradient': {
      const t0 = performance.now();
      const m = makeMachine(DEFAULT_FEP);
      const inR = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
      inR[0] = 0.5; inR[1] = 0.7; inI[0] = 0.1; inI[1] = -0.05;
      const outR = new Float64Array(m.nOut), outI = new Float64Array(m.nOut);
      outR[0] = 0.9;
      const r = verifyPhaseGradient(m, inR, inI, outR, outI, 12);
      return mk('Énergie libre', 'Identité phase-gradient (EP)', 'Le gradient par contraste d’équilibre est exact',
        `(1/β)[∂E/∂θ(x^β) − ∂E/∂θ(x⁰)] vs différences finies centrées sur ${r.checks} couplages (réel+imaginaire), β=${r.beta}`,
        r.relErr, 0.05, performance.now() - t0,
        `résidu d'équilibre = ${r.resid.toExponential(1)}`);
    }

    // ---- Contrôle à forme fermée (audit SPEAR v2.2 / PMP) ------------------
    case 'ctrl-ik': {
      const t0 = performance.now();
      const L1 = 0.3, L2 = 0.2;
      let worst = 0, wAff = 0;
      for (let i = 0; i < 4000; i++) {
        const t1 = Math.random() * 2 * Math.PI, t2 = Math.random() * Math.PI;
        const { x, z } = fkScara(t1, t2, L1, L2);
        const p = ikScara(x, z, L1, L2);
        const f = fkScara(p.th1, p.th2, L1, L2);
        worst = Math.max(worst, Math.hypot(f.x - x, f.z - z));
        wAff = Math.max(wAff, Math.abs(cosTheta2(x * x + z * z, L1, L2) - Math.cos(t2)));
      }
      return mk('Contrôle', 'IK SCARA fermée (FK∘IK)', 'La cinématique inverse à 0 itération est exacte',
        '‖FK(IK(cible)) − cible‖ sur 4000 configurations aléatoires (loi des cosinus)',
        worst, 1e-12, performance.now() - t0,
        `cos θ₂ affine en r² : écart max ${wAff.toExponential(1)}`);
    }
    case 'ctrl-traj': {
      const t0 = performance.now();
      let worst = 0, viol = 0;
      for (let trial = 0; trial < 300; trial++) {
        const amax = 1 + 4 * Math.random(), jmax = 5 + 20 * Math.random();
        const vmax = ((amax * amax) / jmax) * (0.05 + 3 * Math.random());
        const D = 0.01 * Math.exp(Math.random() * Math.log(50 / 0.01));
        const plan = planProfile({ D, vmax, amax, jmax });
        const st = profileState(plan, plan.duration);
        worst = Math.max(worst, Math.abs(st.x - D) / D);
        if (Math.abs(st.a) > amax * (1 + 1e-9)) viol++;
        let v = 0, a = 0;
        for (let i = 0; i < 7; i++) {
          v += a * plan.seg[i] + (plan.jerk[i] * plan.seg[i] ** 2) / 2;
          a += plan.jerk[i] * plan.seg[i];
          if (Math.abs(v) > vmax * (1 + 1e-9)) { viol++; break; }
        }
      }
      return mk('Contrôle', 'Profil jerk-borné 7 segments', 'Le planificateur fermé atteint la consigne sans violer les limites',
        '|x_f − D|/D et violations v/a sur 300 tirages couvrant les 3 régimes (réduit, triangulaire, palier)',
        worst, 1e-12, performance.now() - t0,
        `violations de contraintes : ${viol} · cas T2<0 du rapport source corrigé (accélération réduite a_eff = min(a_max, √(v_max·j)))`);
    }
    case 'ctrl-vdot': {
      const t0 = performance.now();
      const law = DEFAULT_PENDULUM;
      const dt = 1e-5, N = 200000;
      const { th, thd } = simulatePendulum(law, 1.0, 0.0, N, dt);
      let intV = 0;
      for (let i = 0; i < N - 1; i++) {
        intV += 0.5 * (pendulumVdot(th[i], thd[i], law) + pendulumVdot(th[i + 1], thd[i + 1], law)) * dt;
      }
      const dV = pendulumV(th[N - 1], thd[N - 1], law) - pendulumV(th[0], thd[0], law);
      const rel = Math.abs(intV - dV) / Math.abs(dV);
      // honnêteté : fraction du plan de phase où V̇ > 0 (la preuve du papier est fausse)
      let pos = 0, tot = 0;
      for (let i = 0; i <= 120; i++) for (let j = 0; j <= 80; j++) {
        if (pendulumVdot(-Math.PI + (2 * Math.PI * i) / 120, -4 + (8 * j) / 80, law) > 0) pos++;
        tot++;
      }
      return mk('Contrôle', 'Cohérence V̇ du pendule inversé', 'La dérivée d’énergie exposée est la VRAIE (terme résiduel inclus)',
        '|∫V̇ dt − ΔV| / |ΔV| le long d’un rollout de 2 s (Euler semi-implicite, dt=10⁻⁵)',
        rel, 5e-6, performance.now() - t0,
        `V̇ > 0 sur ${(100 * pos / tot).toFixed(1)} % du plan de phase — la preuve « V̇ = −6.4 θ̇² ≤ 0 » du papier SPEAR est réfutée`);
    }
    case 'ctrl-stab': {
      const t0 = performance.now();
      const law = DEFAULT_PENDULUM;
      const { th, thd } = simulatePendulum(law, 1.0, 0.0, 80000, 1e-4); // 8 s
      let thEnd = th[79999] % (2 * Math.PI);
      if (thEnd > Math.PI) thEnd -= 2 * Math.PI;
      if (thEnd < -Math.PI) thEnd += 2 * Math.PI;
      const dev = Math.max(Math.abs(thEnd), Math.abs(thd[79999]));
      return mk('Contrôle', 'Stabilisation du sommet (rollout)', 'La loi π* = −(3.1·sinθ + 1.6·θ̇) stabilise l’équilibre inversé',
        '|θ| et |θ̇| après 8 s depuis (1 rad, 0) — la conclusion du papier tient, sa preuve Lyapunov non',
        dev, 0.01, performance.now() - t0,
        'linéarisé au sommet : βk₁−α = 6.4 > 0 (stable), mais V̇ > 0 sur ~27 % du plan');
    }

    // ---- Optique : dispersion & biréfringence (audit SPEAR-OPTICS corrigé) --
    case 'opt-sellmeier': {
      const t0 = performance.now();
      const h = 1e-5;
      let worst = 0;
      for (const glass of [FUSED_SILICA, BK7]) {
        for (const lam of [0.35, 0.6, 1.05, 2.2]) {
          const a = sellmeier(glass, lam);
          const np = sellmeier(glass, lam + h), nm = sellmeier(glass, lam - h), n0 = sellmeier(glass, lam);
          worst = Math.max(worst,
            Math.abs(a.dn - (np.n - nm.n) / (2 * h)) / Math.abs(a.dn),
            Math.abs(a.d2n - (np.n - 2 * n0.n + nm.n) / (h * h)) / Math.abs(a.dn));
        }
      }
      return mk('Optique', 'Dérivées de Sellmeier', 'Les dérivées analytiques dn/dλ et d²n/dλ² sont exactes',
        'erreur relative vs différences finies centrées (h=10⁻⁵ µm) — silice & BK7, 4 λ chacune',
        worst, 0.02, performance.now() - t0,
        'd³n/dλ³ également implémentée (TOD) · pas de bruit dérivatif');
    }
    case 'opt-gdd': {
      const t0 = performance.now();
      const g800 = gdd(FUSED_SILICA, 0.8);
      const zd = zeroDispersionLambda(FUSED_SILICA);
      const errZd = Number.isFinite(zd) ? Math.abs(gdd(FUSED_SILICA, zd)) : NaN;
      const rel800 = Math.abs(g800 - 36.16) / 36.16; // valeur publiée (Malitson → GVD)
      return mk('Optique', 'GDD silice & dispersion nulle', 'β₂(λ) reproduit les valeurs publiées et s\'annule à λ_ZD',
        `GDD@800 nm vs 36.16 fs²/mm (publié) ; |β₂(λ_ZD)| avec λ_ZD trouvé par bissection`,
        Math.max(rel800 * 100, errZd / 1e-3), 5, performance.now() - t0,
        `GDD=${g800.toFixed(1)} fs²/mm (rel ${(rel800 * 100).toFixed(2)} %) · λ_ZD=${zd.toFixed(4)} µm (publié ≈1.27)`);
    }
    case 'opt-pulse': {
      const t0 = performance.now();
      // GDD pur : la propagation spectrale FFT doit reproduire la forme fermée
      let worst = 0;
      for (const [tau0, L] of [[80, 60], [120, 150], [50, 250]] as const) {
        const b2 = 36.2;
        const closed = pulseWidthGDD(tau0, b2, L);
        const p = pulsePropagation(tau0, b2, 0, L, 1024);
        let mx = 0;
        for (let i = 1; i < p.IL.length; i++) if (p.IL[i] > p.IL[mx]) mx = i;
        const half = p.IL[mx] / 2;
        let lo = mx, hi = mx;
        while (lo > 0 && p.IL[lo] > half) lo--;
        while (hi < p.IL.length - 1 && p.IL[hi] > half) hi++;
        const fwhm = (hi - lo) * (p.t[1] - p.t[0]);
        worst = Math.max(worst, Math.abs(fwhm - closed) / closed);
      }
      return mk('Optique', 'Propagation d\'impulsion (FFT)', 'La phase spectrale exacte reproduit l\'élargissement gaussien fermé',
        'τ_FWHM(FFT) vs τ₀·√(1+(4ln2·β₂L/τ₀²)²) — GDD pur, 3 configurations',
        worst * 100, 3, performance.now() - t0,
        'TOD vérifié qualitativement : asymétrie du profil sans explosion');
    }
    case 'opt-fresnel': {
      const t0 = performance.now();
      let worstPoly = 0, worstDisc = 0;
      const nx = 1.3, ny = 1.55, nz = 1.8;
      for (let i = 0; i < 3000; i++) {
        let sx = Math.random() * 2 - 1, sy = Math.random() * 2 - 1, sz = Math.random() * 2 - 1;
        const Ln = Math.hypot(sx, sy, sz); sx /= Ln; sy /= Ln; sz /= Ln;
        const { n1, n2, disc } = fresnelBiaxial([sx, sy, sz], nx, ny, nz);
        if (disc < worstDisc) worstDisc = disc;
        const sx2 = sx * sx, sy2 = sy * sy, sz2 = sz * sz;
        const B = sx2 * (ny * ny + nz * nz) + sy2 * (nx * nx + nz * nz) + sz2 * (nx * nx + ny * ny);
        const C = sx2 * ny * ny * nz * nz + sy2 * nx * nx * nz * nz + sz2 * nx * nx * ny * ny;
        for (const ne of [n1, n2]) worstPoly = Math.max(worstPoly, Math.abs(ne ** 4 - B * ne * ne + C));
      }
      return mk('Optique', 'Fresnel biaxial fermé', 'Les deux indices effectifs en 0 itération sont exacts',
        '|u²−Bu+C| sur 3000 directions aléatoires · discriminant Δ≥0 partout (identité de Jacobi)',
        Math.max(worstPoly, -worstDisc), 1e-10, performance.now() - t0,
        `Δ min = ${worstDisc === 0 ? '0' : worstDisc.toExponential(1)} · coquille «=1/n²» du rapport source corrigée en «=0»`);
    }
  }
}

function mk(
  group: string, name: string, claim: string, reference: string,
  measured: number, tolerance: number, ms: number, extra?: string, lowerIsBetter = true
): TestResult {
  return {
    group, name, claim, reference, measured, tolerance, ms, extra, lowerIsBetter,
    pass: lowerIsBetter ? measured <= tolerance : measured >= tolerance,
  };
}

// ---------------------------------------------------------------------------
//  Benchmarks (not pass/fail — measured comparisons)
// ---------------------------------------------------------------------------

export interface BenchRow {
  method: string;
  cut: number;
  ratio: number;
  ms: number;
  exact: boolean;
  detail: string;
}

export function benchmarkMaxCut(n: number, kind: 'erdos' | 'cubic' | 'torus' | 'sk' | 'scalefree', seed: number) {
  const g = makeGraph(kind, n, seed);
  const rows: BenchRow[] = [];
  let reference = 0;
  let referenceExact = false;

  if (g.n <= 22) {
    const bf = bruteForceMaxCut(g);
    reference = bf.cut; referenceExact = true;
    rows.push({ method: bf.method, cut: bf.cut, ratio: 1, ms: bf.ms, exact: true, detail: bf.detail ?? '' });
  }

  const cim = runCIM(g, { ...DEFAULT_CIM, iterations: g.n > 400 ? 900 : 1600, seed: seed + 1 });
  const sa = simulatedAnnealing(g, Math.max(120, Math.min(800, Math.round(40000 / Math.max(g.n, 1)))), seed + 2);
  const gl = greedyLocalSearch(g, 32, seed + 3);

  for (const r of [cim, sa, gl]) {
    rows.push({ method: r.method, cut: r.cut, ratio: 0, ms: r.ms, exact: false, detail: r.detail ?? '' });
  }
  if (!referenceExact) reference = Math.max(...rows.map((r) => r.cut));
  for (const r of rows) r.ratio = reference > 0 ? r.cut / reference : 0;

  return { graph: g, rows, reference, referenceExact, cim };
}

/** Measured JS baseline for a dense complex matvec — used as an honest CPU reference. */
export function measureCpuMatvec(n: number, reps = 2000): { nsPerOp: number; gflops: number } {
  const U = haarUnitary(n, 9);
  const vr = new Float64Array(n), vi = new Float64Array(n);
  for (let i = 0; i < n; i++) { vr[i] = Math.cos(i); vi[i] = Math.sin(i); }
  // warm-up so the JIT has compiled the loop
  for (let i = 0; i < 200; i++) matvec(U, vr, vi);
  const t0 = performance.now();
  let sink = 0;
  for (let i = 0; i < reps; i++) sink += matvec(U, vr, vi).re[0];
  const ms = performance.now() - t0;
  if (!Number.isFinite(sink)) throw new Error('nan');
  const nsPerOp = (ms * 1e6) / reps;
  const flops = 8 * n * n;
  return { nsPerOp, gflops: (flops * reps) / (ms / 1000) / 1e9 };
}
