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
  | 'kan-pou' | 'kan-dbasis' | 'kan-grad';

export const TEST_ORDER: TestId[] = [
  'fft-roundtrip', 'fft-parseval', 'fft-tone',
  'bpm-energy', 'bpm-gauss', 'bpm-neff',
  'mesh-reck', 'mesh-unitary', 'mesh-energy', 'mesh-dft',
  'ising-identity', 'ising-exact', 'cim-optimal',
  'kan-pou', 'kan-dbasis', 'kan-grad',
];

export function runTest(id: TestId): TestResult {
  switch (id) {
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
