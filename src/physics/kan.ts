// ============================================================================
//  kan.ts — Photonic Kolmogorov-Arnold Network
//
//  A KAN puts the learnable nonlinearity on the EDGES:
//
//      y_j = sum_i  phi_ij(x_i),
//      phi_ij(x) = wb_ij * silu(x) + ws_ij * sum_k c_ijk B_k(x)
//
//  where B_k are cubic B-splines on a uniform knot vector (Cox-de Boor).
//  This is the exact formulation of Liu et al. 2024, trained here with real
//  analytic backpropagation + Adam. A finite-difference gradient check is
//  exposed so the derivation can be verified numerically at runtime.
//
//  Photonic mapping: each B-spline basis function is one micro-ring resonator
//  detuned to its own wavelength channel; the sum over k is a passive WDM
//  combiner; c_ijk are the ring drop-port transmissions. The whole edge bank
//  therefore evaluates in a single optical pass.
// ============================================================================

export const SPLINE_DEGREE = 3;

export interface SplineBasis {
  degree: number;
  nBasis: number;
  knots: Float64Array;
  /** scratch, length nBasis + degree + 1 */
  work: Float64Array;
  workLow: Float64Array;
}

export function makeBasis(gridIntervals: number, degree = SPLINE_DEGREE): SplineBasis {
  const k = degree;
  const G = gridIntervals;
  const nBasis = G + k;
  const h = 2 / G;
  const knots = new Float64Array(G + 2 * k + 1);
  for (let i = 0; i < knots.length; i++) knots[i] = -1 + (i - k) * h;
  return {
    degree: k,
    nBasis,
    knots,
    work: new Float64Array(nBasis + k + 1),
    workLow: new Float64Array(nBasis + k + 1),
  };
}

/**
 * Evaluates all B-spline basis functions and their derivatives at x.
 * `outB` and `outD` must have length >= nBasis.
 */
export function evalBasis(b: SplineBasis, xRaw: number, outB: Float64Array, outD: Float64Array | null): void {
  const k = b.degree;
  const nb = b.nBasis;
  const t = b.knots;
  const top = nb + k;
  const x = xRaw <= -1 ? -1 + 1e-12 : xRaw >= 1 ? 1 - 1e-12 : xRaw;

  const w = b.work;
  for (let i = 0; i < top; i++) w[i] = x >= t[i] && x < t[i + 1] ? 1 : 0;
  w[top] = 0;

  const low = b.workLow;
  for (let p = 1; p <= k; p++) {
    if (p === k) low.set(w.subarray(0, top + 1));
    for (let i = 0; i + p <= top; i++) {
      const d1 = t[i + p] - t[i];
      const d2 = t[i + p + 1] - t[i + 1];
      const a = d1 > 0 ? ((x - t[i]) / d1) * w[i] : 0;
      const c = d2 > 0 ? ((t[i + p + 1] - x) / d2) * w[i + 1] : 0;
      w[i] = a + c;
    }
  }
  for (let i = 0; i < nb; i++) outB[i] = w[i];

  if (outD) {
    for (let i = 0; i < nb; i++) {
      const d1 = t[i + k] - t[i];
      const d2 = t[i + k + 1] - t[i + 1];
      const a = d1 > 0 ? low[i] / d1 : 0;
      const c = d2 > 0 ? low[i + 1] / d2 : 0;
      outD[i] = k * (a - c);
    }
  }
}

/** Partition of unity residual: max | sum_k B_k(x) - 1 | over the domain. */
export function partitionOfUnityError(gridIntervals = 6, samples = 501): number {
  const b = makeBasis(gridIntervals);
  const out = new Float64Array(b.nBasis);
  let err = 0;
  for (let s = 0; s < samples; s++) {
    const x = -1 + (2 * s) / (samples - 1);
    evalBasis(b, x, out, null);
    let sum = 0;
    for (let i = 0; i < b.nBasis; i++) sum += out[i];
    err = Math.max(err, Math.abs(sum - 1));
  }
  return err;
}

/** Max |analytic B'_k(x) - central difference| over the domain. */
export function splineDerivativeError(gridIntervals = 6, samples = 401): number {
  const b = makeBasis(gridIntervals);
  const nb = b.nBasis;
  const B = new Float64Array(nb), D = new Float64Array(nb);
  const Bp = new Float64Array(nb), Bm = new Float64Array(nb);
  const h = 1e-6;
  let err = 0;
  for (let s = 1; s < samples - 1; s++) {
    const x = -0.98 + (1.96 * s) / (samples - 1);
    evalBasis(b, x, B, D);
    evalBasis(b, x + h, Bp, null);
    evalBasis(b, x - h, Bm, null);
    for (let i = 0; i < nb; i++) err = Math.max(err, Math.abs(D[i] - (Bp[i] - Bm[i]) / (2 * h)));
  }
  return err;
}

// ---------------------------------------------------------------------------
//  Layer
// ---------------------------------------------------------------------------

function sigmoid(x: number) { return 1 / (1 + Math.exp(-x)); }

export class KANLayer {
  nIn: number; nOut: number; nb: number;
  basis: SplineBasis;
  coef: Float64Array; ws: Float64Array; wb: Float64Array;
  gCoef: Float64Array; gWs: Float64Array; gWb: Float64Array;
  mCoef: Float64Array; vCoef: Float64Array;
  mWs: Float64Array; vWs: Float64Array;
  mWb: Float64Array; vWb: Float64Array;

  // caches
  private B: Float64Array; private D: Float64Array;
  private xin: Float64Array;
  private splineVal: Float64Array;
  private siluVal: Float64Array;
  private siluDer: Float64Array;
  private tmpB: Float64Array; private tmpD: Float64Array;

  constructor(nIn: number, nOut: number, gridIntervals: number, rand: () => number) {
    this.nIn = nIn; this.nOut = nOut;
    this.basis = makeBasis(gridIntervals);
    this.nb = this.basis.nBasis;
    const e = nIn * nOut, c = e * this.nb;
    this.coef = new Float64Array(c);
    this.ws = new Float64Array(e).fill(1);
    this.wb = new Float64Array(e);
    const scale = 1 / Math.sqrt(nIn);
    for (let i = 0; i < c; i++) this.coef[i] = (rand() * 2 - 1) * 0.1 * scale;
    for (let i = 0; i < e; i++) this.wb[i] = (rand() * 2 - 1) * scale;
    this.gCoef = new Float64Array(c); this.gWs = new Float64Array(e); this.gWb = new Float64Array(e);
    this.mCoef = new Float64Array(c); this.vCoef = new Float64Array(c);
    this.mWs = new Float64Array(e); this.vWs = new Float64Array(e);
    this.mWb = new Float64Array(e); this.vWb = new Float64Array(e);

    this.B = new Float64Array(nIn * this.nb);
    this.D = new Float64Array(nIn * this.nb);
    this.xin = new Float64Array(nIn);
    this.splineVal = new Float64Array(e);
    this.siluVal = new Float64Array(nIn);
    this.siluDer = new Float64Array(nIn);
    this.tmpB = new Float64Array(this.nb);
    this.tmpD = new Float64Array(this.nb);
  }

  zeroGrad() { this.gCoef.fill(0); this.gWs.fill(0); this.gWb.fill(0); }

  forward(x: Float64Array, y: Float64Array): void {
    const { nIn, nOut, nb } = this;
    this.xin.set(x);
    y.fill(0);
    for (let i = 0; i < nIn; i++) {
      evalBasis(this.basis, x[i], this.tmpB, this.tmpD);
      this.B.set(this.tmpB, i * nb);
      this.D.set(this.tmpD, i * nb);
      const s = sigmoid(x[i]);
      this.siluVal[i] = x[i] * s;
      this.siluDer[i] = s * (1 + x[i] * (1 - s));
    }
    for (let i = 0; i < nIn; i++) {
      const bBase = i * nb;
      for (let j = 0; j < nOut; j++) {
        const e = i * nOut + j;
        const cBase = e * nb;
        let sp = 0;
        for (let k = 0; k < nb; k++) sp += this.coef[cBase + k] * this.B[bBase + k];
        this.splineVal[e] = sp;
        y[j] += this.wb[e] * this.siluVal[i] + this.ws[e] * sp;
      }
    }
  }

  /** dy -> dx, accumulating parameter gradients. */
  backward(dy: Float64Array, dx: Float64Array): void {
    const { nIn, nOut, nb } = this;
    dx.fill(0);
    for (let i = 0; i < nIn; i++) {
      const bBase = i * nb;
      let acc = 0;
      for (let j = 0; j < nOut; j++) {
        const e = i * nOut + j;
        const g = dy[j];
        if (g === 0) continue;
        const cBase = e * nb;
        this.gWb[e] += g * this.siluVal[i];
        this.gWs[e] += g * this.splineVal[e];
        const gws = g * this.ws[e];
        let dSpline = 0;
        for (let k = 0; k < nb; k++) {
          this.gCoef[cBase + k] += gws * this.B[bBase + k];
          dSpline += this.coef[cBase + k] * this.D[bBase + k];
        }
        acc += g * this.wb[e] * this.siluDer[i] + gws * dSpline;
      }
      dx[i] = acc;
    }
  }

  /** Samples phi_ij(x) on a uniform grid — used for the edge-function plots. */
  sampleEdge(i: number, j: number, samples: number, out: Float64Array): void {
    const e = i * this.nOut + j;
    const cBase = e * this.nb;
    const B = new Float64Array(this.nb);
    for (let s = 0; s < samples; s++) {
      const x = -1 + (2 * s) / (samples - 1);
      evalBasis(this.basis, x, B, null);
      let sp = 0;
      for (let k = 0; k < this.nb; k++) sp += this.coef[cBase + k] * B[k];
      out[s] = this.wb[e] * (x * sigmoid(x)) + this.ws[e] * sp;
    }
  }

  edgeMagnitude(i: number, j: number): number {
    const e = i * this.nOut + j;
    const cBase = e * this.nb;
    let s = Math.abs(this.wb[e]);
    for (let k = 0; k < this.nb; k++) s += Math.abs(this.ws[e] * this.coef[cBase + k]) / this.nb;
    return s;
  }
}

// ---------------------------------------------------------------------------
//  Network
// ---------------------------------------------------------------------------

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

export type TargetId = 'kanClassic' | 'ring' | 'saddle' | 'bessel';

export const TARGETS: Record<TargetId, { name: string; tex: string; f: (x: number, y: number) => number }> = {
  kanClassic: {
    name: 'KAN canonique',
    tex: 'f = exp( sin(pi x) + y² )',
    f: (x, y) => (Math.exp(Math.sin(Math.PI * x) + y * y) - 3.0) / 3.0,
  },
  ring: {
    name: 'Anneau oscillant',
    tex: 'f = sin( 6 sqrt(x² + y²) )',
    f: (x, y) => Math.sin(6 * Math.sqrt(x * x + y * y)),
  },
  saddle: {
    name: 'Selle modulée',
    tex: 'f = (x² − y²) · cos(3xy)',
    f: (x, y) => (x * x - y * y) * Math.cos(3 * x * y),
  },
  bessel: {
    name: 'Interférence 2D',
    tex: 'f = cos(4x) · sin(4y) · e^(−(x²+y²))',
    f: (x, y) => Math.cos(4 * x) * Math.sin(4 * y) * Math.exp(-(x * x + y * y)),
  },
};

export class KAN {
  layers: KANLayer[];
  sizes: number[];
  private buf: Float64Array[];
  private gbuf: Float64Array[];
  private hiddenPre: Float64Array[];
  step = 0;
  lr = 0.02;
  beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
  weightDecay = 1e-5;

  constructor(sizes: number[], gridIntervals = 6, seed = 3) {
    const rand = mulberry(seed);
    this.sizes = sizes;
    this.layers = [];
    for (let l = 0; l + 1 < sizes.length; l++) {
      this.layers.push(new KANLayer(sizes[l], sizes[l + 1], gridIntervals, rand));
    }
    this.buf = sizes.map((s) => new Float64Array(s));
    this.gbuf = sizes.map((s) => new Float64Array(s));
    this.hiddenPre = sizes.map((s) => new Float64Array(s));
  }

  /** Hidden activations are confined to the spline domain by tanh. */
  forward(x: Float64Array): Float64Array {
    this.buf[0].set(x);
    for (let l = 0; l < this.layers.length; l++) {
      this.layers[l].forward(this.buf[l], this.buf[l + 1]);
      if (l + 1 < this.layers.length) {
        this.hiddenPre[l + 1].set(this.buf[l + 1]);
        const b = this.buf[l + 1];
        for (let i = 0; i < b.length; i++) b[i] = Math.tanh(b[i]);
      }
    }
    return this.buf[this.layers.length];
  }

  backward(dOut: Float64Array): void {
    const L = this.layers.length;
    this.gbuf[L].set(dOut);
    for (let l = L - 1; l >= 0; l--) {
      if (l + 1 < L) {
        const g = this.gbuf[l + 1], pre = this.hiddenPre[l + 1];
        for (let i = 0; i < g.length; i++) {
          const t = Math.tanh(pre[i]);
          g[i] *= 1 - t * t;
        }
      }
      this.layers[l].backward(this.gbuf[l + 1], this.gbuf[l]);
    }
  }

  zeroGrad() { for (const l of this.layers) l.zeroGrad(); }

  adamStep(scale: number): void {
    this.step++;
    const b1t = 1 - Math.pow(this.beta1, this.step);
    const b2t = 1 - Math.pow(this.beta2, this.step);
    const upd = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * scale + this.weightDecay * p[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * gi;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * gi * gi;
        p[i] -= (this.lr * (m[i] / b1t)) / (Math.sqrt(v[i] / b2t) + this.eps);
      }
    };
    for (const l of this.layers) {
      upd(l.coef, l.gCoef, l.mCoef, l.vCoef);
      upd(l.ws, l.gWs, l.mWs, l.vWs);
      upd(l.wb, l.gWb, l.mWb, l.vWb);
    }
  }

  paramCount(): number {
    let c = 0;
    for (const l of this.layers) c += l.coef.length + l.ws.length + l.wb.length;
    return c;
  }
}

// ---------------------------------------------------------------------------
//  Dataset + training
// ---------------------------------------------------------------------------

export interface Dataset {
  X: Float64Array; // n * 2
  Y: Float64Array; // n
  n: number;
}

export function makeDataset(target: TargetId, n: number, seed: number): Dataset {
  const r = mulberry(seed);
  const f = TARGETS[target].f;
  const X = new Float64Array(n * 2);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = r() * 2 - 1, y = r() * 2 - 1;
    X[2 * i] = x; X[2 * i + 1] = y;
    Y[i] = f(x, y);
  }
  return { X, Y, n };
}

export interface TrainStats {
  trainMse: number;
  testMse: number;
  r2: number;
  ms: number;
  samplesPerSecond: number;
}

const xin = new Float64Array(2);
const dOut = new Float64Array(1);

export function trainEpochs(
  net: KAN, train: Dataset, test: Dataset, epochs: number, batch: number, seed: number
): TrainStats {
  const t0 = performance.now();
  const r = mulberry(seed);
  let processed = 0;
  let lastLoss = 0;

  for (let ep = 0; ep < epochs; ep++) {
    net.zeroGrad();
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      const i = Math.floor(r() * train.n);
      xin[0] = train.X[2 * i]; xin[1] = train.X[2 * i + 1];
      const out = net.forward(xin);
      const err = out[0] - train.Y[i];
      loss += err * err;
      dOut[0] = 2 * err;
      net.backward(dOut);
      processed++;
    }
    lastLoss = loss / batch;
    net.adamStep(1 / batch);
  }

  // evaluation
  let se = 0, mean = 0;
  for (let i = 0; i < test.n; i++) mean += test.Y[i];
  mean /= test.n;
  let sst = 0;
  for (let i = 0; i < test.n; i++) {
    xin[0] = test.X[2 * i]; xin[1] = test.X[2 * i + 1];
    const out = net.forward(xin);
    const e = out[0] - test.Y[i];
    se += e * e;
    const d = test.Y[i] - mean;
    sst += d * d;
  }
  const ms = performance.now() - t0;
  return {
    trainMse: lastLoss,
    testMse: se / test.n,
    r2: 1 - se / Math.max(sst, 1e-12),
    ms,
    samplesPerSecond: processed / (ms / 1000),
  };
}

/** Renders prediction / target / error maps on a res x res grid. */
export function renderMaps(net: KAN, target: TargetId, res: number) {
  const f = TARGETS[target].f;
  const pred = new Float32Array(res * res);
  const truth = new Float32Array(res * res);
  const err = new Float32Array(res * res);
  const v = new Float64Array(2);
  let maxAbs = 1e-9, maxErr = 1e-9;
  for (let iy = 0; iy < res; iy++) {
    const y = -1 + (2 * iy) / (res - 1);
    for (let ix = 0; ix < res; ix++) {
      const x = -1 + (2 * ix) / (res - 1);
      v[0] = x; v[1] = y;
      const p = net.forward(v)[0];
      const t = f(x, y);
      const k = iy * res + ix;
      pred[k] = p; truth[k] = t; err[k] = Math.abs(p - t);
      maxAbs = Math.max(maxAbs, Math.abs(p), Math.abs(t));
      maxErr = Math.max(maxErr, err[k]);
    }
  }
  return { pred, truth, err, maxAbs, maxErr, res };
}

// ---------------------------------------------------------------------------
//  Analytic gradient verification
// ---------------------------------------------------------------------------

/**
 * Central-difference check of the analytic backprop.
 * Returns the max relative error over `nChecks` randomly chosen parameters.
 */
export function gradientCheck(seed = 11, nChecks = 24): { relErr: number; checks: number } {
  const net = new KAN([2, 4, 1], 5, seed);
  const data = makeDataset('kanClassic', 32, seed + 1);
  const r = mulberry(seed + 2);

  const lossOf = (): number => {
    let s = 0;
    const v = new Float64Array(2);
    for (let i = 0; i < data.n; i++) {
      v[0] = data.X[2 * i]; v[1] = data.X[2 * i + 1];
      const e = net.forward(v)[0] - data.Y[i];
      s += e * e;
    }
    return s / data.n;
  };

  net.zeroGrad();
  const v = new Float64Array(2);
  const g = new Float64Array(1);
  for (let i = 0; i < data.n; i++) {
    v[0] = data.X[2 * i]; v[1] = data.X[2 * i + 1];
    const out = net.forward(v);
    g[0] = (2 * (out[0] - data.Y[i])) / data.n;
    net.backward(g);
  }

  const pools: { p: Float64Array; gr: Float64Array }[] = [];
  for (const l of net.layers) {
    pools.push({ p: l.coef, gr: l.gCoef });
    pools.push({ p: l.ws, gr: l.gWs });
    pools.push({ p: l.wb, gr: l.gWb });
  }

  let worst = 0, done = 0;
  const h = 1e-6;
  for (let c = 0; c < nChecks; c++) {
    const pool = pools[Math.floor(r() * pools.length)];
    const idx = Math.floor(r() * pool.p.length);
    const orig = pool.p[idx];
    pool.p[idx] = orig + h; const lp = lossOf();
    pool.p[idx] = orig - h; const lm = lossOf();
    pool.p[idx] = orig;
    const num = (lp - lm) / (2 * h);
    const ana = pool.gr[idx];
    const denom = Math.max(Math.abs(num), Math.abs(ana), 1e-7);
    worst = Math.max(worst, Math.abs(num - ana) / denom);
    done++;
  }
  return { relErr: worst, checks: done };
}
