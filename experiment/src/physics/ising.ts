// ============================================================================
//  ising.ts — Coherent Ising Machine + verifiable MaxCut benchmarks
//
//  MaxCut on a weighted graph:
//        cut(s) = sum_{(i,j) in E} w_ij (1 - s_i s_j) / 2
//               = (W - sum_ij w_ij s_i s_j) / 2
//  so maximising the cut is exactly minimising the Ising Hamiltonian
//        H(s) = sum_{(i,j) in E} w_ij s_i s_j       (antiferromagnetic J = w)
//
//  The CIM is simulated with the standard measurement-feedback DOPO
//  mean-field map (Wang/Marandi/Yamamoto), including amplitude-heterogeneity
//  correction (AHC, Leleu et al.):
//
//      dx_i/dt = (p(t) - 1 - x_i^2) x_i  -  xi * e_i * sum_j J_ij x_j  +  noise
//      de_i/dt = -beta * e_i * (x_i^2 - tau)
//
//  p(t) is the pump ramped through threshold. Spins are read out as sign(x_i).
//  Every reported cut value is recomputed from scratch on the real graph, and
//  for n <= 22 it is compared against exhaustive enumeration.
// ============================================================================

export interface Graph {
  n: number;
  m: number;
  eu: Int32Array;
  ev: Int32Array;
  ew: Float32Array;
  off: Int32Array;   // CSR row offsets, length n+1
  adj: Int32Array;   // 2m
  aw: Float32Array;  // 2m
  totalW: number;
  name: string;
  weighted: boolean;
}

export type GraphKind = 'erdos' | 'cubic' | 'torus' | 'sk' | 'scalefree';

export const GRAPH_INFO: Record<GraphKind, string> = {
  erdos: 'Erdős–Rényi G(n,p) — graphe aléatoire non pondéré',
  cubic: '3-régulier aléatoire — instance MaxCut classique difficile',
  torus: 'Grille torique 2D — couplage local, frustration géométrique',
  sk: 'Sherrington–Kirkpatrick — complet, poids ±1, verre de spin',
  scalefree: 'Barabási–Albert — attachement préférentiel, hubs',
};

function rng(seed: number) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function buildCSR(n: number, eu: number[], ev: number[], ew: number[], name: string, weighted: boolean): Graph {
  const m = eu.length;
  const deg = new Int32Array(n);
  for (let k = 0; k < m; k++) { deg[eu[k]]++; deg[ev[k]]++; }
  const off = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
  const cursor = Int32Array.from(off.subarray(0, n));
  const adj = new Int32Array(2 * m);
  const aw = new Float32Array(2 * m);
  for (let k = 0; k < m; k++) {
    const a = eu[k], b = ev[k], w = ew[k];
    adj[cursor[a]] = b; aw[cursor[a]++] = w;
    adj[cursor[b]] = a; aw[cursor[b]++] = w;
  }
  let totalW = 0;
  for (let k = 0; k < m; k++) totalW += ew[k];
  return {
    n, m,
    eu: Int32Array.from(eu), ev: Int32Array.from(ev), ew: Float32Array.from(ew),
    off, adj, aw, totalW, name, weighted,
  };
}

export function makeGraph(kind: GraphKind, n: number, seed = 1234, density = 0.15): Graph {
  const r = rng(seed);
  const eu: number[] = [], ev: number[] = [], ew: number[] = [];
  const seen = new Set<number>();
  const push = (a: number, b: number, w: number) => {
    if (a === b) return;
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const key = lo * 100000 + hi;
    if (seen.has(key)) return;
    seen.add(key);
    eu.push(lo); ev.push(hi); ew.push(w);
  };

  switch (kind) {
    case 'erdos': {
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (r() < density) push(i, j, 1);
      // guarantee connectivity
      for (let i = 1; i < n; i++) if (r() < 0.6) push(i, Math.floor(r() * i), 1);
      return buildCSR(n, eu, ev, ew, `ER(${n}, p=${density.toFixed(2)})`, false);
    }
    case 'cubic': {
      // union of 3 random perfect matchings (approximate 3-regular)
      for (let round = 0; round < 3; round++) {
        const perm = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
          const j = Math.floor(r() * (i + 1));
          [perm[i], perm[j]] = [perm[j], perm[i]];
        }
        for (let i = 0; i + 1 < n; i += 2) push(perm[i], perm[i + 1], 1);
      }
      return buildCSR(n, eu, ev, ew, `3-régulier(${n})`, false);
    }
    case 'torus': {
      const side = Math.max(2, Math.round(Math.sqrt(n)));
      const N = side * side;
      for (let y = 0; y < side; y++)
        for (let x = 0; x < side; x++) {
          const a = y * side + x;
          push(a, y * side + ((x + 1) % side), 1);
          push(a, ((y + 1) % side) * side + x, 1);
        }
      return buildCSR(N, eu, ev, ew, `Tore ${side}x${side}`, false);
    }
    case 'sk': {
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          push(i, j, r() < 0.5 ? -1 : 1);
      return buildCSR(n, eu, ev, ew, `SK(${n}) ±1`, true);
    }
    case 'scalefree': {
      const m0 = 3;
      const targets: number[] = [];
      for (let i = 0; i < m0; i++)
        for (let j = i + 1; j < m0; j++) { push(i, j, 1); targets.push(i, j); }
      for (let v = m0; v < n; v++) {
        const chosen = new Set<number>();
        while (chosen.size < Math.min(m0, targets.length)) {
          chosen.add(targets[Math.floor(r() * targets.length)]);
        }
        for (const t of chosen) { push(v, t, 1); targets.push(v, t); }
      }
      return buildCSR(n, eu, ev, ew, `Barabási–Albert(${n})`, false);
    }
  }
}

// ---------------------------------------------------------------------------
//  Objective
// ---------------------------------------------------------------------------

export function cutValue(g: Graph, spins: Int8Array): number {
  let c = 0;
  for (let k = 0; k < g.m; k++) if (spins[g.eu[k]] !== spins[g.ev[k]]) c += g.ew[k];
  return c;
}

/** H(s) = sum_{(i,j) in E} w_ij s_i s_j  (equivalently  W - 2*cut). */
export function isingEnergy(g: Graph, spins: Int8Array): number {
  return g.totalW - 2 * cutValue(g, spins);
}

// ---------------------------------------------------------------------------
//  Exhaustive ground truth (Gray-code enumeration, O(2^(n-1) * deg))
// ---------------------------------------------------------------------------

export const BRUTE_FORCE_MAX_N = 22;

export interface SolverResult {
  method: string;
  cut: number;
  spins: Int8Array;
  ms: number;
  iterations: number;
  exact: boolean;
  detail?: string;
}

export function bruteForceMaxCut(g: Graph): SolverResult {
  const t0 = performance.now();
  const n = g.n;
  if (n > BRUTE_FORCE_MAX_N) throw new Error(`brute force limited to n <= ${BRUTE_FORCE_MAX_N}`);
  const spins = new Int8Array(n).fill(1);
  let cut = 0;
  let best = 0;
  let bestGray = 0;
  const total = 1 << (n - 1); // vertex n-1 pinned to +1 (Z2 symmetry)

  for (let step = 1; step < total; step++) {
    const low = step & -step;
    const i = 31 - Math.clz32(low);
    let delta = 0;
    const si = spins[i];
    for (let k = g.off[i]; k < g.off[i + 1]; k++) delta += g.aw[k] * si * spins[g.adj[k]];
    cut += delta;
    spins[i] = si === 1 ? -1 : 1;
    if (cut > best) { best = cut; bestGray = step ^ (step >> 1); }
  }

  const out = new Int8Array(n).fill(1);
  for (let i = 0; i < n - 1; i++) if ((bestGray >> i) & 1) out[i] = -1;

  return {
    method: 'Force brute (exact)',
    cut: best,
    spins: out,
    ms: performance.now() - t0,
    iterations: total,
    exact: true,
    detail: `${total.toLocaleString('fr-FR')} configurations énumérées`,
  };
}

// ---------------------------------------------------------------------------
//  Classical baselines
// ---------------------------------------------------------------------------

/** Greedy construction + 1-opt local search to a local optimum, multi-restart. */
export function greedyLocalSearch(g: Graph, restarts = 24, seed = 7): SolverResult {
  const t0 = performance.now();
  const r = rng(seed);
  const n = g.n;
  const spins = new Int8Array(n);
  let best = -Infinity;
  const bestSpins = new Int8Array(n);
  let sweeps = 0;

  for (let rep = 0; rep < restarts; rep++) {
    for (let i = 0; i < n; i++) spins[i] = r() < 0.5 ? 1 : -1;
    let improved = true;
    while (improved) {
      improved = false;
      sweeps++;
      for (let i = 0; i < n; i++) {
        let local = 0;
        const si = spins[i];
        for (let k = g.off[i]; k < g.off[i + 1]; k++) local += g.aw[k] * si * spins[g.adj[k]];
        if (local > 0) { spins[i] = si === 1 ? -1 : 1; improved = true; }
      }
    }
    const c = cutValue(g, spins);
    if (c > best) { best = c; bestSpins.set(spins); }
  }

  return {
    method: 'Greedy + 1-opt',
    cut: best,
    spins: bestSpins,
    ms: performance.now() - t0,
    iterations: sweeps,
    exact: false,
    detail: `${restarts} redémarrages`,
  };
}

/** Metropolis simulated annealing with geometric cooling. */
export function simulatedAnnealing(g: Graph, sweeps = 600, seed = 99): SolverResult {
  const t0 = performance.now();
  const r = rng(seed);
  const n = g.n;
  const spins = new Int8Array(n);
  for (let i = 0; i < n; i++) spins[i] = r() < 0.5 ? 1 : -1;

  let scale = 0;
  for (let k = 0; k < g.m; k++) scale = Math.max(scale, Math.abs(g.ew[k]));
  const degAvg = (2 * g.m) / n;
  let T = 2 * scale * Math.sqrt(Math.max(degAvg, 1));
  const Tmin = 0.01 * scale;
  const alpha = Math.pow(Tmin / T, 1 / Math.max(sweeps, 1));

  let cut = cutValue(g, spins);
  let best = cut;
  const bestSpins = Int8Array.from(spins);

  for (let s = 0; s < sweeps; s++) {
    for (let t = 0; t < n; t++) {
      const i = Math.floor(r() * n);
      let delta = 0;
      const si = spins[i];
      for (let k = g.off[i]; k < g.off[i + 1]; k++) delta += g.aw[k] * si * spins[g.adj[k]];
      if (delta >= 0 || r() < Math.exp(delta / T)) {
        spins[i] = si === 1 ? -1 : 1;
        cut += delta;
        if (cut > best) { best = cut; bestSpins.set(spins); }
      }
    }
    T *= alpha;
  }

  return {
    method: 'Recuit simulé',
    cut: best,
    spins: bestSpins,
    ms: performance.now() - t0,
    iterations: sweeps * n,
    exact: false,
    detail: `${sweeps} balayages, T: ${(2 * scale * Math.sqrt(Math.max(degAvg, 1))).toFixed(2)} → ${Tmin.toFixed(3)}`,
  };
}

// ---------------------------------------------------------------------------
//  Coherent Ising Machine
// ---------------------------------------------------------------------------

export interface CimParams {
  iterations: number;
  dt: number;
  pStart: number;
  pEnd: number;
  xi: number;
  noise: number;
  ahc: boolean;
  beta: number;
  tau: number;
  seed: number;
  sampleEvery: number;
  nTrace: number;
  /** independent pump cycles; real CIMs are characterised over many trials */
  trials: number;
}

export const DEFAULT_CIM: CimParams = {
  iterations: 1600,
  dt: 0.02,
  pStart: -0.6,
  pEnd: 1.1,
  xi: 0.55,
  noise: 0.035,
  ahc: true,
  beta: 0.35,
  tau: 0.35,
  seed: 2026,
  sampleEvery: 8,
  nTrace: 36,
  trials: 4,
};

export interface CimResult extends SolverResult {
  /** nSamples x nTrace amplitude trajectories. */
  trace: Float32Array;
  nSamples: number;
  nTrace: number;
  traceIdx: Int32Array;
  cutHistory: Float32Array;
  bestHistory: Float32Array;
  pHistory: Float32Array;
  ampHistory: Float32Array;   // rms amplitude
  finalX: Float32Array;
  timeToBest: number;         // iterations until the best cut appeared
  mvmPerSecond: number;
  trials: number;
  trialCuts: Float64Array;
  /** fraction of pump cycles that reached the best cut found */
  successRate: number;
  /** iterations needed for 99% confidence: R99 = ln(0.01)/ln(1-p) */
  r99: number;
  spectralScale: number;
}

/**
 * Runs `p.trials` independent pump cycles and keeps the best readout.
 * The success probability over trials is the standard way CIM performance is
 * reported, because a single cycle is inherently stochastic.
 */
export function runCIM(g: Graph, p: CimParams = DEFAULT_CIM): CimResult {
  const t0 = performance.now();
  const trials = Math.max(1, Math.round(p.trials));
  const trialCuts = new Float64Array(trials);
  let best: CimTrial | null = null;
  let first: CimTrial | null = null;

  for (let t = 0; t < trials; t++) {
    const run = cimTrial(g, p, p.seed + t * 7919, t === 0);
    trialCuts[t] = run.cut;
    if (t === 0) first = run;
    if (!best || run.cut > best.cut) best = run;
  }

  const b = best!;
  const f = first!;
  let hits = 0;
  for (let t = 0; t < trials; t++) if (trialCuts[t] >= b.cut) hits++;
  const successRate = hits / trials;
  const r99 = successRate >= 1
    ? p.iterations
    : (p.iterations * Math.log(0.01)) / Math.log(1 - Math.min(successRate, 0.999999));

  const ms = performance.now() - t0;
  return {
    method: 'CIM (DOPO mean-field)',
    cut: b.cut,
    spins: b.spins,
    ms,
    iterations: p.iterations * trials,
    exact: false,
    detail: `${trials} cycles × ${p.iterations} tours · AHC ${p.ahc ? 'on' : 'off'}`,
    trace: f.trace, nSamples: f.nSamples, nTrace: f.nTrace, traceIdx: f.traceIdx,
    cutHistory: f.cutHistory, bestHistory: f.bestHistory,
    pHistory: f.pHistory, ampHistory: f.ampHistory,
    finalX: f.finalX,
    timeToBest: b.timeToBest,
    mvmPerSecond: (p.iterations * trials) / (ms / 1000),
    trials, trialCuts, successRate, r99,
    spectralScale: f.spectralScale,
  };
}

interface CimTrial {
  cut: number;
  spins: Int8Array;
  timeToBest: number;
  trace: Float32Array;
  nSamples: number;
  nTrace: number;
  traceIdx: Int32Array;
  cutHistory: Float32Array;
  bestHistory: Float32Array;
  pHistory: Float32Array;
  ampHistory: Float32Array;
  finalX: Float32Array;
  spectralScale: number;
}

/**
 * Power iteration on |J| to estimate the spectral radius. Normalising the
 * feedback by this makes the coupling strength xi meaningful across graph
 * families instead of being an arbitrary knob.
 */
function spectralRadius(g: Graph, iters = 24): number {
  const n = g.n;
  let v = new Float32Array(n).fill(1 / Math.sqrt(n));
  let w = new Float32Array(n);
  let lambda = 1;
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = g.off[i]; k < g.off[i + 1]; k++) s += Math.abs(g.aw[k]) * v[g.adj[k]];
      w[i] = s;
    }
    let nrm = 0;
    for (let i = 0; i < n; i++) nrm += w[i] * w[i];
    nrm = Math.sqrt(nrm);
    if (nrm < 1e-12) return 1;
    lambda = nrm;
    for (let i = 0; i < n; i++) w[i] /= nrm;
    const tmp = v; v = w; w = tmp;
  }
  return Math.max(lambda, 1e-6);
}

function cimTrial(g: Graph, p: CimParams, seed: number, record: boolean): CimTrial {
  const n = g.n;
  const r = rng(seed);
  const gauss = () => {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const x = new Float32Array(n);
  const e = new Float32Array(n).fill(1);
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = 0.001 * gauss();

  // Normalise the feedback by the spectral radius of |J| so that xi has the
  // same meaning on a sparse 3-regular graph and on a dense SK instance.
  const lambdaMax = spectralRadius(g);
  const xiEff = p.xi / lambdaMax;

  const spins = new Int8Array(n);
  const bestSpins = new Int8Array(n);
  let best = -Infinity;
  let timeToBest = 0;

  const nSamples = record ? Math.floor(p.iterations / p.sampleEvery) + 1 : 1;
  const nTrace = Math.min(p.nTrace, n);
  const traceIdx = new Int32Array(nTrace);
  for (let i = 0; i < nTrace; i++) traceIdx[i] = Math.floor((i * n) / nTrace);

  const trace = new Float32Array(nSamples * nTrace);
  const cutHistory = new Float32Array(nSamples);
  const bestHistory = new Float32Array(nSamples);
  const pHistory = new Float32Array(nSamples);
  const ampHistory = new Float32Array(nSamples);
  let sample = 0;

  const sqrtDt = Math.sqrt(p.dt);

  // Reading out the cut is O(m). On dense instances that would dominate the
  // O(m) dynamics itself, so we sample the readout instead — exactly like a
  // real machine whose measurement rate is slower than its round-trip rate.
  const evalEvery = Math.max(1, Math.floor(g.m / 3000));

  for (let it = 0; it < p.iterations; it++) {
    const frac = it / Math.max(p.iterations - 1, 1);
    const pump = p.pStart + (p.pEnd - p.pStart) * frac;

    // ---- optical matrix-vector multiply: f = J x   (CSR, cache friendly)
    for (let i = 0; i < n; i++) {
      let s = 0;
      const end = g.off[i + 1];
      for (let k = g.off[i]; k < end; k++) s += g.aw[k] * x[g.adj[k]];
      f[i] = s;
    }

    // ---- DOPO mean-field update + AHC error feedback
    for (let i = 0; i < n; i++) {
      const xi_ = x[i];
      let v = xi_ + p.dt * ((pump - 1 - xi_ * xi_) * xi_ - xiEff * e[i] * f[i]) + sqrtDt * p.noise * gauss();
      if (v > 1.6) v = 1.6; else if (v < -1.6) v = -1.6;
      x[i] = v;
      if (p.ahc) {
        let ei = e[i] - p.dt * p.beta * e[i] * (v * v - p.tau);
        if (ei < 0.02) ei = 0.02; else if (ei > 6) ei = 6;
        e[i] = ei;
      }
      spins[i] = v >= 0 ? 1 : -1;
    }

    let c = best;
    if (it % evalEvery === 0 || it === p.iterations - 1) {
      c = cutValue(g, spins);
      if (c > best) { best = c; bestSpins.set(spins); timeToBest = it; }
    }

    if (record && it % p.sampleEvery === 0 && sample < nSamples) {
      const base = sample * nTrace;
      for (let i = 0; i < nTrace; i++) trace[base + i] = x[traceIdx[i]];
      let a = 0;
      for (let i = 0; i < n; i++) a += x[i] * x[i];
      cutHistory[sample] = c;
      bestHistory[sample] = best;
      pHistory[sample] = pump;
      ampHistory[sample] = Math.sqrt(a / n);
      sample++;
    }
  }

  return {
    cut: best,
    spins: bestSpins,
    timeToBest,
    trace, nSamples: sample, nTrace, traceIdx,
    cutHistory: cutHistory.subarray(0, sample),
    bestHistory: bestHistory.subarray(0, sample),
    pHistory: pHistory.subarray(0, sample),
    ampHistory: ampHistory.subarray(0, sample),
    finalX: x,
    spectralScale: lambdaMax,
  };
}

// ---------------------------------------------------------------------------
//  Layout for rendering (deterministic Fruchterman–Reingold)
// ---------------------------------------------------------------------------

export function layoutGraph(g: Graph, iters = 220, seed = 5): { x: Float32Array; y: Float32Array } {
  const n = g.n;
  const r = rng(seed);
  const x = new Float32Array(n), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    x[i] = 0.42 * Math.cos(a) + (r() - 0.5) * 0.05;
    y[i] = 0.42 * Math.sin(a) + (r() - 0.5) * 0.05;
  }
  const dx = new Float32Array(n), dy = new Float32Array(n);
  const k = Math.sqrt(1 / n) * 0.9;
  let temp = 0.12;

  for (let it = 0; it < iters; it++) {
    dx.fill(0); dy.fill(0);
    // repulsion (O(n^2) — fine for the sizes we visualise)
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let ex = x[i] - x[j], ey = y[i] - y[j];
        let d2 = ex * ex + ey * ey;
        if (d2 < 1e-8) { ex = (r() - 0.5) * 1e-3; ey = (r() - 0.5) * 1e-3; d2 = 1e-8; }
        const f = (k * k) / d2;
        dx[i] += ex * f; dy[i] += ey * f;
        dx[j] -= ex * f; dy[j] -= ey * f;
      }
    }
    // attraction along edges
    for (let m = 0; m < g.m; m++) {
      const a = g.eu[m], b = g.ev[m];
      const ex = x[a] - x[b], ey = y[a] - y[b];
      const d = Math.sqrt(ex * ex + ey * ey) + 1e-9;
      const f = (d * d) / k;
      const ux = (ex / d) * f, uy = (ey / d) * f;
      dx[a] -= ux; dy[a] -= uy;
      dx[b] += ux; dy[b] += uy;
    }
    for (let i = 0; i < n; i++) {
      const d = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]) + 1e-9;
      const lim = Math.min(d, temp);
      x[i] += (dx[i] / d) * lim;
      y[i] += (dy[i] / d) * lim;
      x[i] = Math.max(-0.48, Math.min(0.48, x[i]));
      y[i] = Math.max(-0.48, Math.min(0.48, y[i]));
    }
    temp *= 0.985;
  }
  for (let i = 0; i < n; i++) { x[i] += 0.5; y[i] += 0.5; }
  return { x, y };
}
