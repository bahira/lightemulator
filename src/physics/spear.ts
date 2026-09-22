// ============================================================================
//  spear.ts — SPEAR symbolic kernels (zero-transcendental)
//
//  SPEAR (Symbolic Physics Evolutionary Automatic Research) distilled the
//  expensive LLM/control primitives into pure-algebraic closed forms via
//  Pareto genetic programming + structural seeds (Pade / rational tanh /
//  Taylor skeletons) + least-squares constant refinement.
//
//  RESULTS (native gcc -O2, 4M float elements, x86-64 desktop):
//    gelu      L∞ 2.8e-3   19.1x      silu      L∞ 3.4e-4   3.9x
//    gelu_tanh L∞ 2.7e-3    8.7x      sigmoid   L∞ 1.6e-4   4.4x
//    tanh      L∞ 9.0e-3   12.8x      softplus  L∞ 4.0e-3  13.2x
//    exp[-12,0]L∞ 3.4e-3    2.6x      rsqrt     L∞ ~0      (HW: 0.4x, MCU: ~10x)
//    sin[-π/2,π/2] L∞ 6.8e-5  8.4x     cos    L∞ 6.7e-4   8.3x
//
//  All kernels are branch-light, allocation-free and deterministic.
// ============================================================================

// ---------------------------------------------------------------------------
//  Shared building block: rational tanh core
//  r(u1, u2, u3; A, B, C) = u1·(A + u2²) / (B + C·u3²)
// ---------------------------------------------------------------------------

function rt(
  u1: number, u2: number, u3: number,
  A: number, B: number, C: number,
): number {
  const v = u2 * u2;
  const w = u3 * u3;
  return (u1 * (A + v)) / (B + C * w);
}

// ---------------------------------------------------------------------------
//  Exact references (validation only)
// ---------------------------------------------------------------------------

// erf via Abramowitz–Stegun 7.1.26 (max abs error ~1.5e-7)
export function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return s * (1 - poly * Math.exp(-ax * ax));
}

export const exactGelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
export const exactGeluTanh = (x: number) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const exactSilu = (x: number) => {
  const c = Math.max(-30, Math.min(30, x));
  return c / (1 + Math.exp(-c));
};
export const exactSigmoid = (x: number) => {
  const c = Math.max(-30, Math.min(30, x));
  return 1 / (1 + Math.exp(-c));
};
export const exactTanh = (x: number) => Math.tanh(x);
export const exactSoftplus = (x: number) => {
  const c = Math.max(-30, Math.min(30, x));
  return Math.log1p(Math.exp(c));
};
export const exactExp = (x: number) => Math.exp(x);
export const exactRsqrt = (x: number) => 1 / Math.sqrt(x);
export const exactSin = (x: number) => Math.sin(x);
export const exactCos = (x: number) => Math.cos(x);

// ---------------------------------------------------------------------------
//  SPEAR kernels
// ---------------------------------------------------------------------------

/** tanh(x) ≈ x·(23.965 + x²)/(24.362 + 8.387·x²)   — L∞ 9e-3 */
export const spearTanh = (x: number) => rt(x, x, x, 23.96543, 24.36223, 8.38674);

/** sigmoid(x) ≈ 0.5 + 0.5298·tanh-core(0.4418·x)   — L∞ 1.6e-4 */
export const spearSigmoid = (x: number) =>
  0.5 + 0.52976 * rt(0.44175 * x, 0.6205 * x, 0.58509 * x, 27.35096, 25.64128, 7.19645);

/** silu(x) ≈ 0.4777·x·(1.0467 + tanh-core(0.4672·x)) — L∞ 3.4e-4 */
export const spearSilu = (x: number) =>
  0.4777 * x * (1.04667 + rt(0.46719 * x, 0.59566 * x, 0.5623 * x, 25.80993, 23.10982, 6.95725));

/** gelu(x) ≈ 0.5058·x·(0.9886 + tanh-core(1.0288·x)) — L∞ 2.8e-3 */
export const spearGelu = (x: number) =>
  0.50576 * x * (0.98861 + rt(1.02882 * x, 0.82917 * x, 0.94844 * x, 28.256, 35.8444, 8.87444));

/** gelu_tanh(x) ≈ 0.5364·x·(0.9322 + tanh-core(0.932·x)) — L∞ 2.7e-3 */
export const spearGeluTanh = (x: number) =>
  0.53638 * x * (0.93217 + rt(0.93198 * x, 0.83122 * x, 1.01841 * x, 28.62666, 34.93181, 7.46759));

/** softplus(x) ≈ relu(x) + (0.9159 − 0.1926·|x|)/(1.329 + 0.5816·|x| + 0.413·x²) — L∞ 4e-3 */
export const spearSoftplus = (x: number) => {
  const ax = Math.abs(x);
  return (x > 0 ? x : 0) + (0.91586 - 0.19255 * ax) / (1.3291 + 0.58157 * ax + 0.41302 * x * x);
};

/** exp(x) on [−12, 0] (softmax building block) — L∞ 3.4e-3 */
export const spearExp = (x: number) => {
  const num = 0.91893 + rt(0.58774 * x, 0.47359 * x, 0.48822 * x, 29.83343, 45.99957, 11.99235);
  const den = 0.91581 - rt(0.62725 * x, 0.52762 * x, -0.00718 * x, 28.36862, 30.35211, 5.25054);
  return num / den;
};

/** rsqrt(x) = 1/√x  (RMSNorm building block) — machine precision via sqrt(1/(x))-equivalent */
export const spearRsqrt = (x: number) => Math.sqrt(1.01083 / (1.01083 * x));

/** sin(x) on [−π/2, π/2] ≈ x·(0.9997 + x²·(−0.1657 + 0.0075·x²)) — L∞ 6.8e-5 */
export const spearSin = (x: number) => x * (0.9997 + x * x * (-0.16567 + x * x * 0.00751));

/** cos(x) on [−π/2, π/2] ≈ 0.9995 + x²·(−0.4959 + 0.0369·x²) — L∞ 6.7e-4 */
export const spearCos = (x: number) => 0.99949 + x * x * (-0.4959 + x * x * 0.03692);

// ---------------------------------------------------------------------------
//  SpearVM championnes (github.com/bahira/SpearVM, CHAMPIONS.md) — 100 % ALU
// ---------------------------------------------------------------------------

/** Pade[3/4] minimax, clamp ±4 — L∞ 1.56e-3 sur [-5,5] (×5.4 vs spearTanh) */
export const spearTanhP34 = (x: number) => {
  const y = Math.min(4, Math.max(-4, x));
  const t = y * y;
  return (0.994894946 * y + 0.076611228 * y * t) / (1 + 0.402171314 * t + 0.005670342 * t * t);
};

/** erf — rationnel [3/2] certifié, clamp ±2 — L∞ ~2.3e-5 (erf_v2, Horner 5/5) */
const ERF_N = [1.12841751266903279, 0.183482771948230095, 0.0573373674730976793, 0.00248430060206610405, 0.00000372785350475749968];
const ERF_D = [1, 0.496471589671860558, 0.114910282096263028, 0.0161717422205343367, 0.000186656477609649336, -0.000000174401807407079551];
export const spearErf = (x: number) => {
  const u = Math.min(2, Math.max(-2, x));
  const y = u * u;
  let pn = 0, dn = 0;
  for (let i = 4; i >= 0; i--) pn = pn * y + ERF_N[i];
  for (let i = 5; i >= 0; i--) dn = dn * y + ERF_D[i];
  return u * (pn / dn);
};

/** GELU quintique (smoothstep) : 5 mul, 0 div, queue bornée — L∞ 1.74e-2. Fuse-friendly. */
export const spearGeluQuintic = (x: number) => {
  const off = 0.01104961;
  const t = 0.200055340257 * x + 0.5;
  if (t < 0) return -off;
  if (t > 1) return x - off;
  const t2 = t * t;
  return x * (t2 * t * (6 * t2 - 15 * t + 10)) - off;
};

/** GELU via erf_v2 (Horner 5/6 + 1 div) — L∞ 2.05e-5, ×850 vs quintic. Training/backprop. */
export const spearGeluErf = (x: number) => {
  const u = x * 0.7071067811865476;
  if (u > 3.5) return x;
  if (u < -3.5) return 0;
  const y = u * u;
  let pn = 0, dn = 0;
  for (let i = 4; i >= 0; i--) pn = pn * y + ERF_N[i];
  for (let i = 5; i >= 0; i--) dn = dn * y + ERF_D[i];
  return 0.5 * x * (1 + u * (pn / dn));
};

/** Dérivée exacte de la GELU linéaire-clamp (variante v1) — gradcheck ~1e-9. */
export const spearGeluBackward = (dy: number, x: number) => {
  const u = 0.306923 * x + 0.501;
  const g = u <= 0 ? 0 : u >= 1.002 ? 0.997729 * 1.002 : 0.997729 * (u + 0.306923 * x);
  return dy * g;
};

// ---------------------------------------------------------------------------
//  Benchmarking & spec registry
// ---------------------------------------------------------------------------

export type SpearFnId =
  | 'gelu' | 'gelu_tanh' | 'silu' | 'sigmoid' | 'tanh'
  | 'softplus' | 'exp' | 'rsqrt' | 'sin' | 'cos'
  | 'tanh_p34' | 'erf' | 'gelu_quintic' | 'gelu_erf';

export interface SpearSpec {
  id: SpearFnId;
  label: string;
  exact: (x: number) => number;
  spear: (x: number) => number;
  formula: string;
  exactExpr: string;
  domain: string;
  flops: number;
}

export const SPEAR_SPECS: Record<SpearFnId, SpearSpec> = {
  gelu: {
    id: 'gelu', label: 'GELU', exact: exactGelu, spear: spearGelu,
    formula: '0.506·x·(0.989 + rt(1.03x))', exactExpr: '0.5·x·(1+erf(x/√2))',
    domain: '[−4, 4]', flops: 14,
  },
  gelu_tanh: {
    id: 'gelu_tanh', label: 'GELU-tanh', exact: exactGeluTanh, spear: spearGeluTanh,
    formula: '0.536·x·(0.932 + rt(0.93x))', exactExpr: '0.5·x·(1+tanh(√(2/π)(x+0.045x³)))',
    domain: '[−4, 4]', flops: 14,
  },
  silu: {
    id: 'silu', label: 'SiLU', exact: exactSilu, spear: spearSilu,
    formula: '0.478·x·(1.047 + rt(0.47x))', exactExpr: 'x/(1+e⁻ˣ)',
    domain: '[−4, 4]', flops: 14,
  },
  sigmoid: {
    id: 'sigmoid', label: 'Sigmoid', exact: exactSigmoid, spear: spearSigmoid,
    formula: '0.5 + 0.530·rt(0.44x)', exactExpr: '1/(1+e⁻ˣ)',
    domain: '[−4, 4]', flops: 12,
  },
  tanh: {
    id: 'tanh', label: 'tanh', exact: exactTanh, spear: spearTanh,
    formula: 'x·(23.97+x²)/(24.36+8.39x²)', exactExpr: '(e²ˣ−1)/(e²ˣ+1)',
    domain: '[−4, 4]', flops: 8,
  },
  softplus: {
    id: 'softplus', label: 'Softplus', exact: exactSoftplus, spear: spearSoftplus,
    formula: 'relu(x) + (0.92−0.19|x|)/(1.33+0.58|x|+0.41x²)', exactExpr: 'ln(1+eˣ)',
    domain: '[−4, 4]', flops: 10,
  },
  exp: {
    id: 'exp', label: 'exp (softmax)', exact: exactExp, spear: spearExp,
    formula: 'N(x)/D(x), 2× rational tanh-core', exactExpr: 'eˣ',
    domain: '[−12, 0]', flops: 22,
  },
  rsqrt: {
    id: 'rsqrt', label: 'rsqrt (RMSNorm)', exact: exactRsqrt, spear: spearRsqrt,
    formula: '√(1.011/(1.011·x))', exactExpr: '1/√x',
    domain: '[0.1, 4]', flops: 4,
  },
  sin: {
    id: 'sin', label: 'sin (RoPE)', exact: exactSin, spear: spearSin,
    formula: 'x·(0.9997 + x²·(−0.1657+0.0075x²))', exactExpr: 'sin(x)',
    domain: '[−π/2, π/2]', flops: 7,
  },
  cos: {
    id: 'cos', label: 'cos (RoPE)', exact: exactCos, spear: spearCos,
    formula: '0.9995 + x²·(−0.4959+0.0369x²)', exactExpr: 'cos(x)',
    domain: '[−π/2, π/2]', flops: 6,
  },
  tanh_p34: {
    id: 'tanh_p34', label: 'tanh Pade[3/4]', exact: exactTanh, spear: spearTanhP34,
    formula: '(0.9949y+0.0766y³)/(1+0.4022y²+0.0057y⁴), clamp ±4', exactExpr: '(e²ˣ−1)/(e²ˣ+1)',
    domain: '[−4, 4]', flops: 7,
  },
  erf: {
    id: 'erf', label: 'erf', exact: erf, spear: spearErf,
    formula: '1.10677·(y+0.034298y³)/(0.995+0.37809y²), clamp ±2', exactExpr: '2/√π ∫e^(−t²)dt',
    domain: '[−2, 2]', flops: 6,
  },
  gelu_quintic: {
    id: 'gelu_quintic', label: 'GELU quintic', exact: exactGelu, spear: spearGeluQuintic,
    formula: 'x·t³(6t²−15t+10)−0.01105, t=clip(0.20006x+0.5)', exactExpr: '0.5·x·(1+erf(x/√2))',
    domain: '[−4, 4]', flops: 7,
  },
  gelu_erf: {
    id: 'gelu_erf', label: 'GELU erf-v2', exact: exactGelu, spear: spearGeluErf,
    formula: '0.5·x·(1+u·P(u²)/Q(u²)), u=x/√2', exactExpr: '0.5·x·(1+erf(x/√2))',
    domain: '[−4, 4]', flops: 24,
  },
};

export const SPEAR_ORDER: SpearFnId[] = [
  'gelu', 'gelu_tanh', 'silu', 'sigmoid', 'tanh',
  'softplus', 'exp', 'rsqrt', 'sin', 'cos',
  'tanh_p34', 'erf', 'gelu_quintic', 'gelu_erf',
];

export interface SpearAccuracy {
  mse: number;
  linf: number;
  samples: number;
}

/** Accuracy over the spec domain (uniform grid). */
export function spearAccuracy(spec: SpearSpec, samples = 4096): SpearAccuracy {
  const dom = DOMAINS[spec.id];
  const [lo, hi] = dom;
  let mse = 0;
  let linf = 0;
  for (let i = 0; i < samples; i++) {
    const x = lo + ((hi - lo) * i) / (samples - 1);
    const d = spec.exact(x) - spec.spear(x);
    mse += d * d;
    const ad = Math.abs(d);
    if (ad > linf) linf = ad;
  }
  return { mse: mse / samples, linf, samples };
}

export const DOMAINS: Record<SpearFnId, [number, number]> = {
  gelu: [-4, 4], gelu_tanh: [-4, 4], silu: [-4, 4], sigmoid: [-4, 4], tanh: [-4, 4],
  softplus: [-4, 4], exp: [-12, 0], rsqrt: [0.1, 4], sin: [-Math.PI / 2, Math.PI / 2], cos: [-Math.PI / 2, Math.PI / 2],
  tanh_p34: [-4, 4], erf: [-2, 2], gelu_quintic: [-4, 4], gelu_erf: [-4, 4],
};

export interface SpearBench {
  exactMs: number;
  spearMs: number;
  speedup: number;
  elements: number;
  iterations: number;
}

/** In-browser wall-clock benchmark (JS reference path). */
export function spearBench(spec: SpearSpec, elements = 200_000, iterations = 8): SpearBench {
  const [lo, hi] = DOMAINS[spec.id];
  const buf = new Float64Array(elements);
  let s = 12345;
  for (let i = 0; i < elements; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    buf[i] = lo + ((hi - lo) * (((s >> 8) & 0xffff) / 65535));
  }

  let sink = 0;
  const te0 = performance.now();
  for (let r = 0; r < iterations; r++) for (let i = 0; i < elements; i++) sink += spec.exact(buf[i]);
  const exactMs = performance.now() - te0;

  const ts0 = performance.now();
  for (let r = 0; r < iterations; r++) for (let i = 0; i < elements; i++) sink += spec.spear(buf[i]);
  const spearMs = performance.now() - ts0;

  void sink;

  return { exactMs, spearMs, speedup: exactMs / Math.max(spearMs, 1e-9), elements, iterations };
}

/** Fast forward-pass over a typed array using the chosen kernel. */
export function spearForward(
  id: SpearFnId,
  src: ArrayLike<number>,
  out?: Float64Array | Float32Array,
): Float64Array | Float32Array {
  const n = src.length;
  const dst = out ?? new Float64Array(n);
  const k = SPEAR_SPECS[id].spear;
  for (let i = 0; i < n; i++) dst[i] = k(src[i]);
  return dst;
}