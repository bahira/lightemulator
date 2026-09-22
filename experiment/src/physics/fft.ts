// ============================================================================
//  fft.ts — Radix-2 iterative in-place FFT (decimation-in-time)
//  Zero allocation on the hot path. Cached bit-reversal + twiddle tables.
//  Operates on split Float64Array (re, im) for cache-friendly access.
// ============================================================================

const twCache = new Map<number, Float64Array>();
const revCache = new Map<number, Uint32Array>();

/** Interleaved [cos, sin] twiddles for k = 0 .. n/2-1, angle = 2*pi*k/n. */
function twiddles(n: number): Float64Array {
  let t = twCache.get(n);
  if (t) return t;
  t = new Float64Array(n);
  for (let k = 0; k < n >> 1; k++) {
    const a = (2 * Math.PI * k) / n;
    t[2 * k] = Math.cos(a);
    t[2 * k + 1] = Math.sin(a);
  }
  twCache.set(n, t);
  return t;
}

function revTable(n: number): Uint32Array {
  let r = revCache.get(n);
  if (r) return r;
  let bits = 0;
  while (1 << bits < n) bits++;
  r = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i, y = 0;
    for (let b = 0; b < bits; b++) {
      y = (y << 1) | (x & 1);
      x >>>= 1;
    }
    r[i] = y >>> 0;
  }
  revCache.set(n, r);
  return r;
}

export function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * In-place complex FFT. `re`/`im` must have identical power-of-two length.
 * Forward uses exp(-2*pi*i*k/n); inverse uses exp(+2*pi*i*k/n) and scales 1/n.
 */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (n <= 1) return;
  if (!isPow2(n)) throw new Error(`fft: length ${n} is not a power of two`);

  const rev = revTable(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  const tw = twiddles(n);
  const sgn = inverse ? 1 : -1;

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const stride = n / len;
    for (let base = 0; base < n; base += len) {
      let ti = 0;
      for (let k = 0; k < half; k++, ti += stride << 1) {
        const wr = tw[ti];
        const wi = sgn * tw[ti + 1];
        const a = base + k;
        const b = a + half;
        const xr = re[b], xi = im[b];
        const vr = xr * wr - xi * wi;
        const vi = xr * wi + xi * wr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] = re[a] + vr; im[a] = im[a] + vi;
      }
    }
  }

  if (inverse) {
    const s = 1 / n;
    for (let i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
  }
}

/**
 * Angular spatial frequencies kx (rad / length-unit) in FFT storage order,
 * matching numpy.fft.fftfreq * 2*pi.
 */
export function fftAngularFreq(n: number, dx: number): Float64Array {
  const k = new Float64Array(n);
  const f = (2 * Math.PI) / (n * dx);
  const halfN = n >> 1;
  for (let i = 0; i < n; i++) k[i] = (i < halfN ? i : i - n) * f;
  return k;
}

// ---------------------------------------------------------------------------
//  Self-verification helpers (used by the grounded validation suite)
// ---------------------------------------------------------------------------

/** Max abs deviation of IFFT(FFT(x)) from x. Should be ~1e-15 * ||x||. */
export function fftRoundTripError(n: number, seed = 1): number {
  const re = new Float64Array(n), im = new Float64Array(n);
  const r0 = new Float64Array(n), i0 = new Float64Array(n);
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
  for (let i = 0; i < n; i++) { re[i] = r0[i] = rnd(); im[i] = i0[i] = rnd(); }
  fft(re, im, false);
  fft(re, im, true);
  let err = 0;
  for (let i = 0; i < n; i++) {
    err = Math.max(err, Math.abs(re[i] - r0[i]), Math.abs(im[i] - i0[i]));
  }
  return err;
}

/** Relative Parseval defect: | sum|x|^2 - (1/n) sum|X|^2 | / sum|x|^2. */
export function fftParsevalError(n: number, seed = 7): number {
  const re = new Float64Array(n), im = new Float64Array(n);
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
  let e0 = 0;
  for (let i = 0; i < n; i++) {
    re[i] = rnd(); im[i] = rnd();
    e0 += re[i] * re[i] + im[i] * im[i];
  }
  fft(re, im, false);
  let e1 = 0;
  for (let i = 0; i < n; i++) e1 += re[i] * re[i] + im[i] * im[i];
  e1 /= n;
  return Math.abs(e0 - e1) / e0;
}

/** Compares against the analytic DFT of a pure tone: X[k0] = n, else 0. */
export function fftImpulseError(n: number, k0 = 5): number {
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * k0 * i) / n;
    re[i] = Math.cos(a); im[i] = Math.sin(a);
  }
  fft(re, im, false);
  let err = 0;
  for (let k = 0; k < n; k++) {
    const expectRe = k === k0 ? n : 0;
    err = Math.max(err, Math.abs(re[k] - expectRe), Math.abs(im[k]));
  }
  return err / n;
}
