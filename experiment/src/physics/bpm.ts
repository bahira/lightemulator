// ============================================================================
//  bpm.ts — Beam Propagation Method (split-step Fourier)
//
//  Solves the scalar paraxial Helmholtz equation for the slowly-varying
//  envelope A(x, z) of E = A(x,z) * exp(i * k * z), k = k0 * n0:
//
//      dA/dz = (i / 2k) * d2A/dx2  +  i * k0 * dn(x,z) * A
//
//  Split-step (Strang, 2nd order): half diffraction -> full index -> half
//  diffraction. Diffraction is diagonal in Fourier space:
//
//      A_hat <- A_hat * exp(-i * kx^2 * dz / (4k))     (each half step)
//
//  Also provides an imaginary-distance BPM eigenmode solver: substituting
//  z -> -i*tau turns the propagator into a contraction that converges onto
//  the fundamental guided mode, whose effective index is then extracted with
//  the exact (non-paraxial) variational Rayleigh quotient:
//
//      beta^2 = [ int k0^2 n(x)^2 |A|^2 dx - int |dA/dx|^2 dx ] / int |A|^2 dx
//
//  All lengths are in micrometres (um).
// ============================================================================

import { fft, fftAngularFreq } from './fft';

export type DeviceId =
  | 'straight'
  | 'coupler'
  | 'ybranch'
  | 'mmi'
  | 'mzi'
  | 'grin'
  | 'freespace'
  | 'disorder'
  | 'grating';

export interface BpmConfig {
  device: DeviceId;
  lambda: number;     // vacuum wavelength (um)
  n0: number;         // background / cladding index
  dn: number;         // core-cladding index contrast
  wCore: number;      // core width (um)
  gap: number;        // edge-to-edge gap for coupler (um)
  sep: number;        // arm separation for y-branch / MZI (um)
  Lx: number;         // transverse window (um)
  Lz: number;         // propagation length (um)
  Nx: number;         // transverse samples (power of two)
  Nz: number;         // z steps
  armPhase: number;   // extra phase on MZI upper arm (rad)
  w0: number;         // launch waist (um)
  tilt: number;       // launch tilt (deg)
  launch: 'mode' | 'gauss' | 'twin' | 'plane';
  absorber: boolean;  // super-gaussian absorbing boundary
  disorderAmp: number;
  seed: number;
  /** tanh smoothing half-width of the core walls (um). Keep >= ~1.5*dx. */
  edge: number;
}

export const DEFAULT_BPM: BpmConfig = {
  device: 'coupler',
  lambda: 1.55,
  n0: 1.444,
  dn: 0.012,
  wCore: 4.0,
  gap: 3.0,
  sep: 12.0,
  Lx: 80,
  Lz: 3000,
  Nx: 512,
  Nz: 900,
  armPhase: 0,
  w0: 3.0,
  tilt: 0,
  launch: 'mode',
  absorber: true,
  disorderAmp: 0.004,
  seed: 12345,
  edge: 0.25,
};

export const DEVICE_INFO: Record<DeviceId, { name: string; blurb: string }> = {
  straight:  { name: 'Guide droit',        blurb: 'Mode guidé unique — vérifie la conservation de puissance et l’absence de diffraction.' },
  coupler:   { name: 'Coupleur directif',  blurb: 'Deux guides évanescemment couplés — battement périodique, longueur de couplage Lc = pi/(2*kappa).' },
  ybranch:   { name: 'Jonction Y',         blurb: 'Séparateur adiabatique 1→2, 50/50 par symétrie.' },
  mmi:       { name: 'MMI 1×2',            blurb: 'Interférence multimode — auto-imagerie de Talbot dans une section large.' },
  mzi:       { name: 'Mach-Zehnder',       blurb: 'Y-split → déphaseur → Y-combine. Le mode antisymétrique rayonne: extinction réelle.' },
  grin:      { name: 'Lentille GRIN',      blurb: 'Indice parabolique — refocalisation périodique du faisceau (pitch de Wood).' },
  freespace: { name: 'Espace libre',       blurb: 'Diffraction gaussienne pure — comparée à w(z)=w0*sqrt(1+(z/zR)^2).' },
  disorder:  { name: 'Milieu désordonné',  blurb: 'Fluctuations d’indice aléatoires — speckle et localisation transverse.' },
  grating:   { name: 'Réseau périodique',  blurb: 'Modulation longitudinale — couplage de modes / diffraction de Bragg.' },
};

// --------------------------------------------------------------------------
//  Geometry helpers
// --------------------------------------------------------------------------

function slab(x: number, center: number, width: number, edge: number): number {
  const h = width * 0.5;
  return 0.5 * (Math.tanh((x - center + h) / edge) - Math.tanh((x - center - h) / edge));
}

function smoothstep(t: number): number {
  const u = t < 0 ? 0 : t > 1 ? 1 : t;
  return u * u * (3 - 2 * u);
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Writes the index perturbation dn(x, z) into `out` for a single z slice.
 * Pure function of (x, z) — no hidden state, so it is trivially verifiable.
 */
export function indexSlice(out: Float64Array, x: Float64Array, z: number, c: BpmConfig, noise?: Float64Array): void {
  const N = x.length;
  const d = c.dn;
  const e = c.edge;
  switch (c.device) {
    case 'freespace': {
      out.fill(0);
      break;
    }
    case 'straight': {
      for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, c.wCore, e);
      break;
    }
    case 'coupler': {
      const off = (c.gap + c.wCore) * 0.5;
      for (let i = 0; i < N; i++) out[i] = d * (slab(x[i], -off, c.wCore, e) + slab(x[i], off, c.wCore, e));
      break;
    }
    case 'ybranch': {
      const zStart = c.Lz * 0.15, zEnd = c.Lz * 0.85;
      const t = smoothstep((z - zStart) / Math.max(zEnd - zStart, 1e-9));
      const s = 0.5 * c.sep * t;
      if (s < c.wCore * 0.55) {
        // still merged: widen the single core continuously
        const w = c.wCore + 2 * s;
        for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, w, e);
      } else {
        for (let i = 0; i < N; i++) out[i] = d * (slab(x[i], -s, c.wCore, e) + slab(x[i], s, c.wCore, e));
      }
      break;
    }
    case 'mmi': {
      const z1 = c.Lz * 0.12, z2 = c.Lz * 0.72;
      const wMmi = c.sep + c.wCore * 1.6;
      if (z < z1) {
        for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, c.wCore, e);
      } else if (z < z2) {
        for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, wMmi, e);
      } else {
        const s = c.sep * 0.5;
        for (let i = 0; i < N; i++) out[i] = d * (slab(x[i], -s, c.wCore, e) + slab(x[i], s, c.wCore, e));
      }
      break;
    }
    case 'mzi': {
      const zA = c.Lz * 0.08, zB = c.Lz * 0.28, zC = c.Lz * 0.72, zD = c.Lz * 0.92;
      let s: number;
      if (z < zA) s = 0;
      else if (z < zB) s = 0.5 * c.sep * smoothstep((z - zA) / (zB - zA));
      else if (z < zC) s = 0.5 * c.sep;
      else if (z < zD) s = 0.5 * c.sep * (1 - smoothstep((z - zC) / (zD - zC)));
      else s = 0;
      // phase shifter: extra index on the +x arm inside the straight section
      const inPhaseRegion = z >= zB && z < zC;
      const phaseLen = zC - zB;
      // dn_extra * k0 * phaseLen = armPhase   ->  dn_extra = armPhase*lambda/(2*pi*phaseLen)
      const dnExtra = inPhaseRegion ? (c.armPhase * c.lambda) / (2 * Math.PI * phaseLen) : 0;
      if (s < c.wCore * 0.55) {
        const w = c.wCore + 2 * s;
        for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, w, e);
      } else {
        for (let i = 0; i < N; i++) {
          const up = slab(x[i], s, c.wCore, e);
          const lo = slab(x[i], -s, c.wCore, e);
          out[i] = d * (up + lo) + dnExtra * up;
        }
      }
      break;
    }
    case 'grin': {
      const R = c.Lx * 0.28;
      for (let i = 0; i < N; i++) {
        const u = x[i] / R;
        out[i] = u * u < 1 ? d * (1 - u * u) : 0;
      }
      break;
    }
    case 'grating': {
      const period = 20.0; // um
      const m = 0.5 * (1 + Math.cos((2 * Math.PI * z) / period));
      for (let i = 0; i < N; i++) out[i] = d * slab(x[i], 0, c.wCore, e) * (0.55 + 0.45 * m);
      break;
    }
    case 'disorder': {
      for (let i = 0; i < N; i++) {
        out[i] = d * slab(x[i], 0, c.wCore * 6, e) + (noise ? noise[i] * c.disorderAmp : 0);
      }
      break;
    }
  }
}

// --------------------------------------------------------------------------
//  Fields
// --------------------------------------------------------------------------

export function xAxis(c: BpmConfig): Float64Array {
  const x = new Float64Array(c.Nx);
  const dx = c.Lx / c.Nx;
  for (let i = 0; i < c.Nx; i++) x[i] = (i - c.Nx / 2) * dx;
  return x;
}

function normalize(re: Float64Array, im: Float64Array, dx: number): void {
  let p = 0;
  for (let i = 0; i < re.length; i++) p += re[i] * re[i] + im[i] * im[i];
  p *= dx;
  const s = 1 / Math.sqrt(Math.max(p, 1e-300));
  for (let i = 0; i < re.length; i++) { re[i] *= s; im[i] *= s; }
}

// --------------------------------------------------------------------------
//  Imaginary-distance BPM eigenmode solver
// --------------------------------------------------------------------------

export interface ModeResult {
  re: Float64Array;
  im: Float64Array;
  neff: number;
  iterations: number;
  residual: number;
  ms: number;
}

/**
 * Fundamental guided mode by imaginary-distance BPM.
 * Substituting z -> -i*tau makes the split-step operator a contraction whose
 * dominant eigenvector is the fundamental mode.
 */
export function solveMode(c: BpmConfig, maxIter = 400, tol = 1e-11): ModeResult {
  const t0 = performance.now();
  const N = c.Nx;
  const dx = c.Lx / N;
  const x = xAxis(c);
  const k0 = (2 * Math.PI) / c.lambda;
  const k = k0 * c.n0;

  const dn = new Float64Array(N);
  indexSlice(dn, x, 0, c);

  // Fourier-domain decay factor for half a tau-step.
  //
  // Step choice: the transverse diffusion length over one tau-step is
  // l = sqrt(step / (2k)). We set l ~ wCore/6 so the operator resolves the
  // mode profile (Strang splitting error stays O(step^2 * [T,V])), while the
  // modal growth-rate separation ~ k0*dn/2 still filters higher-order modes
  // within a few hundred iterations.
  const kx = fftAngularFreq(N, dx);
  const lTarget = Math.max(c.wCore, 1) / 6;
  const step = Math.min(2 * k * lTarget * lTarget, 8.0);
  const diff = new Float64Array(N);
  for (let i = 0; i < N; i++) diff[i] = Math.exp((-kx[i] * kx[i] * step) / (4 * k));
  const gain = new Float64Array(N);
  for (let i = 0; i < N; i++) gain[i] = Math.exp(k0 * dn[i] * step);

  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const u = x[i] / Math.max(c.w0, 0.5);
    re[i] = Math.exp(-u * u);
  }
  normalize(re, im, dx);

  const prev = new Float64Array(N);
  let iterations = 0, residual = 1;

  for (let it = 0; it < maxIter; it++) {
    prev.set(re);
    fft(re, im, false);
    for (let i = 0; i < N; i++) { re[i] *= diff[i]; im[i] *= diff[i]; }
    fft(re, im, true);
    for (let i = 0; i < N; i++) { re[i] *= gain[i]; im[i] *= gain[i]; }
    fft(re, im, false);
    for (let i = 0; i < N; i++) { re[i] *= diff[i]; im[i] *= diff[i]; }
    fft(re, im, true);
    normalize(re, im, dx);
    // fix global sign for a stable convergence metric
    let dot = 0;
    for (let i = 0; i < N; i++) dot += re[i] * prev[i];
    if (dot < 0) for (let i = 0; i < N; i++) { re[i] = -re[i]; im[i] = -im[i]; }
    let d = 0;
    for (let i = 0; i < N; i++) { const e = re[i] - prev[i]; d += e * e; }
    residual = Math.sqrt(d * dx);
    iterations = it + 1;
    if (residual < tol) break;
  }

  const neff = effectiveIndex(re, im, dn, c);
  return { re, im, neff, iterations, residual, ms: performance.now() - t0 };
}

/**
 * Exact (non-paraxial) variational effective index from a field profile:
 *   beta^2 = [ sum k0^2 n^2 |A|^2 - (1/N) sum kx^2 |A_hat|^2 ] / sum |A|^2
 */
export function effectiveIndex(re: Float64Array, im: Float64Array, dn: Float64Array, c: BpmConfig): number {
  const N = c.Nx;
  const dx = c.Lx / N;
  const k0 = (2 * Math.PI) / c.lambda;
  const kx = fftAngularFreq(N, dx);

  let num = 0, den = 0;
  for (let i = 0; i < N; i++) {
    const p = re[i] * re[i] + im[i] * im[i];
    const n = c.n0 + dn[i];
    num += k0 * k0 * n * n * p;
    den += p;
  }
  const fr = Float64Array.from(re), fi = Float64Array.from(im);
  fft(fr, fi, false);
  let grad = 0;
  for (let i = 0; i < N; i++) grad += kx[i] * kx[i] * (fr[i] * fr[i] + fi[i] * fi[i]);
  grad /= N;

  const beta2 = (num - grad) / den;
  return Math.sqrt(Math.max(beta2, 0)) / k0;
}

/**
 * Exact analytic effective index of the fundamental TE mode of a symmetric
 * step-index slab, from the transcendental equation  u*tan(u) = sqrt(V^2-u^2).
 * Used as ground truth for the BPM mode solver.
 */
export function slabNeffAnalytic(lambda: number, ncore: number, nclad: number, width: number): number {
  const k0 = (2 * Math.PI) / lambda;
  const a = width / 2;
  const V = k0 * a * Math.sqrt(ncore * ncore - nclad * nclad);
  if (V <= 0) return nclad;
  const g = (u: number) => u * Math.tan(u) - Math.sqrt(Math.max(V * V - u * u, 0));
  let lo = 1e-9;
  let hi = Math.min(Math.PI / 2 - 1e-9, V - 1e-12);
  if (hi <= lo) return nclad;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (g(mid) > 0) hi = mid; else lo = mid;
  }
  const u = 0.5 * (lo + hi);
  const kappa = u / a;
  const beta = Math.sqrt(Math.max(k0 * k0 * ncore * ncore - kappa * kappa, 0));
  return beta / k0;
}

// --------------------------------------------------------------------------
//  Propagation
// --------------------------------------------------------------------------

export interface BpmResult {
  Nx: number;
  NzOut: number;
  intensity: Float32Array; // NzOut * Nx, normalised to peak = 1
  phase: Float32Array;     // NzOut * Nx, wrapped to [-pi, pi]
  index: Float32Array;     // NzOut * Nx, dn map normalised to [0,1]
  power: Float64Array;     // NzOut, total transverse power
  width: Float64Array;     // NzOut, 2nd-moment beam radius (um)
  zAxis: Float64Array;
  x: Float64Array;
  portLeft: number;        // fractional output power, x < 0
  portRight: number;       // fractional output power, x > 0
  peakIntensity: number;
  energyDrift: number;     // |P_end - P_0| / P_0 (absorber off => conservation test)
  ms: number;
  steps: number;
  megaCellSteps: number;   // Nx * Nz / 1e6
  throughputMcs: number;   // measured Mcell-steps / s
}

export function runBPM(c: BpmConfig): BpmResult {
  const t0 = performance.now();
  const N = c.Nx;
  const dx = c.Lx / N;
  const dz = c.Lz / c.Nz;
  const x = xAxis(c);
  const k0 = (2 * Math.PI) / c.lambda;
  const k = k0 * c.n0;

  // half-step diffraction operator (constant): exp(-i kx^2 dz / (4k))
  const kx = fftAngularFreq(N, dx);
  const dHr = new Float64Array(N), dHi = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const a = (-kx[i] * kx[i] * dz) / (4 * k);
    dHr[i] = Math.cos(a); dHi[i] = Math.sin(a);
  }

  // absorbing boundary (super-gaussian), applied once per step
  const absorb = new Float64Array(N).fill(1);
  if (c.absorber) {
    const xa = c.Lx * 0.38, wa = c.Lx * 0.09;
    for (let i = 0; i < N; i++) {
      const d = Math.abs(x[i]) - xa;
      if (d > 0) { const u = d / wa; absorb[i] = Math.exp(-(u * u * u * u)); }
    }
  }

  // launch field
  const re = new Float64Array(N), im = new Float64Array(N);
  if (c.launch === 'mode') {
    const m = solveMode({ ...c, device: c.device === 'coupler' ? 'straight' : c.device }, 260, 1e-10);
    if (c.device === 'coupler') {
      const off = (c.gap + c.wCore) * 0.5;
      const shift = Math.round(-off / dx);
      for (let i = 0; i < N; i++) {
        const j = i - shift;
        if (j >= 0 && j < N) re[i] = m.re[j];
      }
    } else {
      re.set(m.re);
    }
  } else if (c.launch === 'gauss') {
    for (let i = 0; i < N; i++) {
      const u = x[i] / c.w0;
      re[i] = Math.exp(-u * u);
    }
  } else if (c.launch === 'twin') {
    const s = c.sep * 0.5;
    for (let i = 0; i < N; i++) {
      const a = (x[i] - s) / c.w0, b = (x[i] + s) / c.w0;
      re[i] = Math.exp(-a * a) + Math.exp(-b * b);
    }
  } else {
    for (let i = 0; i < N; i++) {
      const u = x[i] / (c.Lx * 0.3);
      re[i] = Math.exp(-(u ** 8));
    }
  }
  if (c.tilt !== 0) {
    const kt = k * Math.sin((c.tilt * Math.PI) / 180);
    for (let i = 0; i < N; i++) {
      const a = kt * x[i];
      const cr = Math.cos(a), ci = Math.sin(a);
      const r = re[i], m2 = im[i];
      re[i] = r * cr - m2 * ci; im[i] = r * ci + m2 * cr;
    }
  }
  normalize(re, im, dx);

  const NzOut = Math.min(c.Nz, 512);
  const stride = Math.max(1, Math.floor(c.Nz / NzOut));
  const rows = Math.floor(c.Nz / stride);

  const intensity = new Float32Array(rows * N);
  const phase = new Float32Array(rows * N);
  const indexMap = new Float32Array(rows * N);
  const power = new Float64Array(rows);
  const width = new Float64Array(rows);
  const zAxis = new Float64Array(rows);

  const dn = new Float64Array(N);
  const noise = new Float64Array(N);
  if (c.device === 'disorder') {
    const rnd = mulberry(c.seed);
    for (let i = 0; i < N; i++) noise[i] = rnd() * 2 - 1;
  }

  let p0 = 0;
  for (let i = 0; i < N; i++) p0 += re[i] * re[i] + im[i] * im[i];
  p0 *= dx;

  let peak = 0;
  let row = 0;

  for (let s = 0; s < c.Nz; s++) {
    const z = s * dz;

    // ---- half diffraction
    fft(re, im, false);
    for (let i = 0; i < N; i++) {
      const r = re[i], m = im[i];
      re[i] = r * dHr[i] - m * dHi[i];
      im[i] = r * dHi[i] + m * dHr[i];
    }
    fft(re, im, true);

    // ---- full index step
    indexSlice(dn, x, z + dz * 0.5, c, noise);
    for (let i = 0; i < N; i++) {
      const a = k0 * dn[i] * dz;
      const cr = Math.cos(a), ci = Math.sin(a);
      const r = re[i], m = im[i];
      re[i] = r * cr - m * ci;
      im[i] = r * ci + m * cr;
    }

    // ---- half diffraction
    fft(re, im, false);
    for (let i = 0; i < N; i++) {
      const r = re[i], m = im[i];
      re[i] = r * dHr[i] - m * dHi[i];
      im[i] = r * dHi[i] + m * dHr[i];
    }
    fft(re, im, true);

    if (c.absorber) for (let i = 0; i < N; i++) { re[i] *= absorb[i]; im[i] *= absorb[i]; }

    if (s % stride === 0 && row < rows) {
      const base = row * N;
      let p = 0, m1 = 0, m2 = 0;
      for (let i = 0; i < N; i++) {
        const ii = re[i] * re[i] + im[i] * im[i];
        intensity[base + i] = ii;
        phase[base + i] = Math.atan2(im[i], re[i]);
        indexMap[base + i] = dn[i];
        p += ii; m1 += ii * x[i]; m2 += ii * x[i] * x[i];
        if (ii > peak) peak = ii;
      }
      const mean = p > 0 ? m1 / p : 0;
      const varx = p > 0 ? m2 / p - mean * mean : 0;
      power[row] = p * dx;
      width[row] = 2 * Math.sqrt(Math.max(varx, 0));
      // the recorded field is the state AFTER this step, i.e. at z + dz
      zAxis[row] = z + dz;
      row++;
    }
  }

  // normalise maps
  const invPeak = peak > 0 ? 1 / peak : 1;
  for (let i = 0; i < intensity.length; i++) intensity[i] *= invPeak;
  let dnMax = 1e-12;
  for (let i = 0; i < indexMap.length; i++) dnMax = Math.max(dnMax, Math.abs(indexMap[i]));
  for (let i = 0; i < indexMap.length; i++) indexMap[i] /= dnMax;

  // output ports
  let pl = 0, pr = 0;
  for (let i = 0; i < N; i++) {
    const ii = re[i] * re[i] + im[i] * im[i];
    if (x[i] < 0) pl += ii; else pr += ii;
  }
  const ptot = pl + pr || 1;

  let pEnd = 0;
  for (let i = 0; i < N; i++) pEnd += re[i] * re[i] + im[i] * im[i];
  pEnd *= dx;

  const ms = performance.now() - t0;
  const megaCellSteps = (N * c.Nz) / 1e6;
  return {
    Nx: N,
    NzOut: rows,
    intensity, phase, index: indexMap,
    power, width, zAxis, x,
    portLeft: pl / ptot,
    portRight: pr / ptot,
    peakIntensity: peak,
    energyDrift: Math.abs(pEnd - p0) / p0,
    ms,
    steps: c.Nz,
    megaCellSteps,
    throughputMcs: megaCellSteps / (ms / 1000),
  };
}

/**
 * Free-space Gaussian diffraction check: compares the simulated 2nd-moment
 * radius against the analytic  w(z) = w0*sqrt(1 + (z/zR)^2),  zR = pi*n0*w0^2/lambda.
 * Returns the max relative deviation over the propagation length.
 */
export function gaussianDiffractionError(): { err: number; zR: number; ms: number } {
  const c: BpmConfig = {
    ...DEFAULT_BPM,
    device: 'freespace',
    launch: 'gauss',
    absorber: false,
    w0: 6,
    Lx: 400,
    Lz: 300,
    Nx: 1024,
    Nz: 300,
    tilt: 0,
  };
  const r = runBPM(c);
  const zR = (Math.PI * c.n0 * c.w0 * c.w0) / c.lambda;
  let err = 0;
  // 2nd-moment radius of exp(-x^2/w^2) intensity exp(-2x^2/w^2) is w/2 * ... ->
  // sigma = w/2, and our `width` = 2*sigma = w. Compare directly.
  for (let i = 1; i < r.NzOut; i++) {
    const z = r.zAxis[i];
    const analytic = c.w0 * Math.sqrt(1 + (z / zR) ** 2);
    err = Math.max(err, Math.abs(r.width[i] - analytic) / analytic);
  }
  return { err, zR, ms: r.ms };
}
