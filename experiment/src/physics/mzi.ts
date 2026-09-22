// ============================================================================
//  mzi.ts — Programmable linear-optical processor
//
//  Any N x N unitary can be realised exactly by a mesh of N(N-1)/2 Mach-Zehnder
//  interferometers plus a final layer of phase shifters (Reck '94, Clements '16).
//
//  We build the mesh by complex Givens elimination. Define the two-mode block
//  acting on adjacent waveguides (m, m+1):
//
//      T(m, theta, phi) = [ e^{i phi} cos(theta)   -sin(theta) ]
//                         [ e^{i phi} sin(theta)    cos(theta) ]
//
//  Left-multiplying a matrix by T mixes ROWS m and m+1, so we can annihilate
//  the sub-diagonal of U column by column (exactly like a QR factorisation):
//
//      T_K ... T_2 T_1 U = D      (D diagonal & unitary)
//   => U = T_1^H T_2^H ... T_K^H D
//
//  Physically: light enters, meets the diagonal phase screen D, then traverses
//  the MZIs in the order K, K-1, ..., 1. Every operation is energy conserving
//  by construction, so ||out||^2 == ||in||^2 to machine precision.
// ============================================================================

export interface CMat {
  n: number;
  re: Float64Array; // row-major n*n
  im: Float64Array;
}

export interface MZIUnit {
  m: number;      // acts on modes (m, m+1)
  theta: number;  // internal phase -> splitting ratio  cos^2(theta)
  phi: number;    // external phase
  layer: number;  // physical column in the mesh (for layout + latency)
  order: number;  // application order during forward propagation
}

export interface MeshDecomposition {
  n: number;
  units: MZIUnit[];      // in APPLICATION order (after the diagonal screen)
  diagRe: Float64Array;  // final diagonal phase screen
  diagIm: Float64Array;
  depth: number;         // number of physical layers
  ms: number;
}

export function cmat(n: number): CMat {
  return { n, re: new Float64Array(n * n), im: new Float64Array(n * n) };
}

export function identity(n: number): CMat {
  const M = cmat(n);
  for (let i = 0; i < n; i++) M.re[i * n + i] = 1;
  return M;
}

export function cloneMat(A: CMat): CMat {
  return { n: A.n, re: Float64Array.from(A.re), im: Float64Array.from(A.im) };
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Box-Muller standard normal. */
function makeGauss(seed: number) {
  const r = rng(seed);
  return () => {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Haar-distributed random unitary via modified Gram-Schmidt on a complex
 * Ginibre matrix. Exact to machine precision for the sizes we use (N <= 24).
 */
export function haarUnitary(n: number, seed = 42): CMat {
  const g = makeGauss(seed);
  const A = cmat(n);
  for (let i = 0; i < n * n; i++) { A.re[i] = g(); A.im[i] = g(); }
  // orthonormalise columns
  for (let j = 0; j < n; j++) {
    for (let p = 0; p < j; p++) {
      // <col_p, col_j>
      let dr = 0, di = 0;
      for (let i = 0; i < n; i++) {
        const a = i * n + p, b = i * n + j;
        dr += A.re[a] * A.re[b] + A.im[a] * A.im[b];
        di += A.re[a] * A.im[b] - A.im[a] * A.re[b];
      }
      for (let i = 0; i < n; i++) {
        const a = i * n + p, b = i * n + j;
        A.re[b] -= dr * A.re[a] - di * A.im[a];
        A.im[b] -= dr * A.im[a] + di * A.re[a];
      }
    }
    let nrm = 0;
    for (let i = 0; i < n; i++) {
      const b = i * n + j;
      nrm += A.re[b] * A.re[b] + A.im[b] * A.im[b];
    }
    nrm = 1 / Math.sqrt(Math.max(nrm, 1e-300));
    for (let i = 0; i < n; i++) {
      const b = i * n + j;
      A.re[b] *= nrm; A.im[b] *= nrm;
    }
  }
  return A;
}

/** Discrete Fourier transform matrix (a canonical, exactly-known unitary). */
export function dftMatrix(n: number): CMat {
  const M = cmat(n);
  const s = 1 / Math.sqrt(n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const ang = (-2 * Math.PI * a * b) / n;
      M.re[a * n + b] = s * Math.cos(ang);
      M.im[a * n + b] = s * Math.sin(ang);
    }
  }
  return M;
}

/** Hadamard-like real unitary (only for n = power of two). */
export function hadamard(n: number): CMat {
  const M = cmat(n);
  const s = 1 / Math.sqrt(n);
  for (let a = 0; a < n; a++)
    for (let b = 0; b < n; b++) {
      let bits = a & b, p = 0;
      while (bits) { p ^= bits & 1; bits >>= 1; }
      M.re[a * n + b] = p ? -s : s;
    }
  return M;
}

// ---------------------------------------------------------------------------
//  Decomposition
// ---------------------------------------------------------------------------

/**
 * Complex Givens elimination of the sub-diagonal.
 * Nulls M[i][j] using the block on rows (i-1, i):
 *   (T M)[i][j] = e^{i phi} sin(theta) * a + cos(theta) * b,  a=M[i-1][j], b=M[i][j]
 * Choosing phi = arg(b) - arg(a) + pi makes e^{i phi} a = -|a| e^{i arg b},
 * so the bracket vanishes for tan(theta) = |b| / |a|  ->  real theta.
 */
export function decomposeMesh(U: CMat): MeshDecomposition {
  const t0 = performance.now();
  const n = U.n;
  const M = cloneMat(U);
  const raw: { m: number; theta: number; phi: number }[] = [];

  for (let j = 0; j < n - 1; j++) {
    for (let i = n - 1; i > j; i--) {
      const ar = M.re[(i - 1) * n + j], ai = M.im[(i - 1) * n + j];
      const br = M.re[i * n + j], bi = M.im[i * n + j];
      const aMag = Math.hypot(ar, ai);
      const bMag = Math.hypot(br, bi);

      let theta: number, phi: number;
      if (bMag < 1e-300) {
        theta = 0; phi = 0; // already zero, T = identity-ish
      } else if (aMag < 1e-300) {
        theta = Math.PI / 2; phi = 0;
      } else {
        theta = Math.atan2(bMag, aMag);
        phi = Math.atan2(bi, br) - Math.atan2(ai, ar) + Math.PI;
      }
      raw.push({ m: i - 1, theta, phi });
      applyTLeft(M, i - 1, theta, phi);
    }
  }

  const diagRe = new Float64Array(n), diagIm = new Float64Array(n);
  for (let i = 0; i < n; i++) { diagRe[i] = M.re[i * n + i]; diagIm[i] = M.im[i * n + i]; }

  // Application order is the reverse of the elimination order:
  //   U = T_1^H T_2^H ... T_K^H D   =>  apply D, then T_K^H, ..., T_1^H
  const units: MZIUnit[] = [];
  const lastLayer = new Int32Array(n);
  for (let idx = raw.length - 1, o = 0; idx >= 0; idx--, o++) {
    const { m, theta, phi } = raw[idx];
    const layer = Math.max(lastLayer[m], lastLayer[m + 1]) + 1;
    lastLayer[m] = layer; lastLayer[m + 1] = layer;
    units.push({ m, theta, phi, layer, order: o });
  }
  let depth = 0;
  for (let i = 0; i < n; i++) depth = Math.max(depth, lastLayer[i]);

  // The greedy schedule guarantees that two cells sharing a waveguide always
  // have layer order == application order. Cells in different layers that do
  // NOT share a waveguide commute, so we may stably reorder by layer without
  // changing the realised unitary. This makes the physical left-to-right
  // ordering monotone, which the renderer and the stage probes rely on.
  units.sort((a, b) => (a.layer - b.layer) || (a.order - b.order));

  return { n, units, diagRe, diagIm, depth, ms: performance.now() - t0 };
}

/** In-place left multiplication  M <- T(m, theta, phi) M  (mixes rows m, m+1). */
function applyTLeft(M: CMat, m: number, theta: number, phi: number): void {
  const n = M.n;
  const ct = Math.cos(theta), st = Math.sin(theta);
  const pr = Math.cos(phi), pi = Math.sin(phi);
  const r0 = m * n, r1 = (m + 1) * n;
  for (let b = 0; b < n; b++) {
    const xr = M.re[r0 + b], xi = M.im[r0 + b];
    const yr = M.re[r1 + b], yi = M.im[r1 + b];
    // row m   :  e^{i phi} cos * x  - sin * y
    const exr = xr * pr - xi * pi, exi = xr * pi + xi * pr;
    M.re[r0 + b] = ct * exr - st * yr;
    M.im[r0 + b] = ct * exi - st * yi;
    // row m+1 :  e^{i phi} sin * x  + cos * y
    M.re[r1 + b] = st * exr + ct * yr;
    M.im[r1 + b] = st * exi + ct * yi;
  }
}

/** In-place left multiplication  M <- T(m,theta,phi)^H M. */
function applyTHLeft(M: CMat, m: number, theta: number, phi: number): void {
  const n = M.n;
  const ct = Math.cos(theta), st = Math.sin(theta);
  const pr = Math.cos(phi), pi = -Math.sin(phi); // e^{-i phi}
  const r0 = m * n, r1 = (m + 1) * n;
  for (let b = 0; b < n; b++) {
    const xr = M.re[r0 + b], xi = M.im[r0 + b];
    const yr = M.re[r1 + b], yi = M.im[r1 + b];
    // row m   :  e^{-i phi} (cos * x + sin * y)
    const sr = ct * xr + st * yr, si = ct * xi + st * yi;
    M.re[r0 + b] = sr * pr - si * pi;
    M.im[r0 + b] = sr * pi + si * pr;
    // row m+1 :  -sin * x + cos * y
    M.re[r1 + b] = -st * xr + ct * yr;
    M.im[r1 + b] = -st * xi + ct * yi;
  }
}

/** Rebuild the unitary implemented by the physical mesh. */
export function reconstruct(d: MeshDecomposition): CMat {
  const n = d.n;
  const M = cmat(n);
  for (let i = 0; i < n; i++) { M.re[i * n + i] = d.diagRe[i]; M.im[i * n + i] = d.diagIm[i]; }
  // units are stored in application order (T_K^H first); left-multiply in that order
  for (const u of d.units) applyTHLeft(M, u.m, u.theta, u.phi);
  return M;
}

// ---------------------------------------------------------------------------
//  Forward propagation through the physical mesh
// ---------------------------------------------------------------------------

export interface PropagationTrace {
  /** (depth+2) x n intensities: stage 0 = input, stage 1 = after phase screen. */
  stages: Float64Array;
  nStages: number;
  n: number;
  outRe: Float64Array;
  outIm: Float64Array;
  inputPower: number;
  outputPower: number;
}

export function propagate(d: MeshDecomposition, inRe: Float64Array, inIm: Float64Array): PropagationTrace {
  const n = d.n;
  const re = Float64Array.from(inRe);
  const im = Float64Array.from(inIm);

  let inputPower = 0;
  for (let i = 0; i < n; i++) inputPower += re[i] * re[i] + im[i] * im[i];

  const nStages = d.depth + 2;
  const stages = new Float64Array(nStages * n);
  for (let i = 0; i < n; i++) stages[i] = re[i] * re[i] + im[i] * im[i];

  // diagonal phase screen
  for (let i = 0; i < n; i++) {
    const r = re[i], m = im[i];
    re[i] = r * d.diagRe[i] - m * d.diagIm[i];
    im[i] = r * d.diagIm[i] + m * d.diagRe[i];
  }
  for (let i = 0; i < n; i++) stages[n + i] = re[i] * re[i] + im[i] * im[i];

  let currentLayer = 0;
  for (const u of d.units) {
    if (u.layer !== currentLayer) {
      // snapshot the layer we just finished
      if (currentLayer > 0) {
        const base = (currentLayer + 1) * n;
        for (let i = 0; i < n; i++) stages[base + i] = re[i] * re[i] + im[i] * im[i];
      }
      currentLayer = u.layer;
    }
    const ct = Math.cos(u.theta), st = Math.sin(u.theta);
    const pr = Math.cos(u.phi), pi = -Math.sin(u.phi);
    const a = u.m, b = u.m + 1;
    const xr = re[a], xi = im[a], yr = re[b], yi = im[b];
    const sr = ct * xr + st * yr, si = ct * xi + st * yi;
    re[a] = sr * pr - si * pi;
    im[a] = sr * pi + si * pr;
    re[b] = -st * xr + ct * yr;
    im[b] = -st * xi + ct * yi;
  }
  for (let s = currentLayer + 1; s < nStages; s++) {
    const base = s * n;
    for (let i = 0; i < n; i++) stages[base + i] = re[i] * re[i] + im[i] * im[i];
  }

  let outputPower = 0;
  for (let i = 0; i < n; i++) outputPower += re[i] * re[i] + im[i] * im[i];

  return { stages, nStages, n, outRe: re, outIm: im, inputPower, outputPower };
}

/** Direct dense matrix-vector product (reference implementation). */
export function matvec(U: CMat, vr: Float64Array, vi: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = U.n;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    let sr = 0, si = 0;
    const row = a * n;
    for (let b = 0; b < n; b++) {
      const ur = U.re[row + b], ui = U.im[row + b];
      sr += ur * vr[b] - ui * vi[b];
      si += ur * vi[b] + ui * vr[b];
    }
    re[a] = sr; im[a] = si;
  }
  return { re, im };
}

// ---------------------------------------------------------------------------
//  Norms / verification
// ---------------------------------------------------------------------------

export function frobeniusDiff(A: CMat, B: CMat): number {
  let s = 0;
  for (let i = 0; i < A.re.length; i++) {
    const dr = A.re[i] - B.re[i], di = A.im[i] - B.im[i];
    s += dr * dr + di * di;
  }
  return Math.sqrt(s);
}

/** || U^H U - I ||_F */
export function unitarityDefect(U: CMat): number {
  const n = U.n;
  let s = 0;
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      let dr = 0, di = 0;
      for (let k = 0; k < n; k++) {
        const p = k * n + a, q = k * n + b;
        dr += U.re[p] * U.re[q] + U.im[p] * U.im[q];
        di += U.re[p] * U.im[q] - U.im[p] * U.re[q];
      }
      if (a === b) dr -= 1;
      s += dr * dr + di * di;
    }
  }
  return Math.sqrt(s);
}
