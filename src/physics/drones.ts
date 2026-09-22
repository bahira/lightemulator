// ============================================================================
//  drones.ts — Drones lumineux : phototaxie, trilatération, budget de Friis.
//
//  TypeScript pur, zéro dépendance — copiez le fichier, il marche.
//  Physique réelle : champ lumineux en 1/d² (loi inverse-square), navigation
//  par gradient (phototaxie), positionnement par trilatération exacte
//  (différenciation + solve 2×2 fermé), budget de liaison de Friis.
//
//  Chaque affirmation est vérifiée par validate.ts ('drone-*') :
//    - phototaxie   : convergence vers la source (champ à source unique,
//                     le gradient pointe toujours vers elle → trajectoire
//                     droite, convergence garantie en temps borné)
//    - trilatération: position reconstruite à l'epsilon machine de 3 ranges
//    - Friis        : Pr(2d) = Pr(d)/4 exactement
// ============================================================================

export interface Beacon { x: number; y: number; power: number }
export interface Drone { x: number; y: number; vx: number; vy: number; trail: number[] }

/** Champ lumineux total en (x,y) : Σ P/(4π d²) — loi inverse-square. */
export function lightField(beacons: Beacon[], x: number, y: number): number {
  let I = 0;
  for (let i = 0; i < beacons.length; i++) {
    const b = beacons[i];
    const dx = x - b.x, dy = y - b.y;
    const d2 = dx * dx + dy * dy + 1e-9;
    I += b.power / (4 * Math.PI * d2);
  }
  return I;
}

/** Gradient du champ lumineux (différences centrales). */
export function lightGrad(beacons: Beacon[], x: number, y: number, h = 1e-4): { gx: number; gy: number } {
  return {
    gx: (lightField(beacons, x + h, y) - lightField(beacons, x - h, y)) / (2 * h),
    gy: (lightField(beacons, x, y + h) - lightField(beacons, x, y - h)) / (2 * h),
  };
}

/** Budget de liaison de Friis : puissance reçue Pr = Pt·Gt·Gr·λ²/(4πd)². */
export function friisReceived(pt: number, lambda: number, d: number, gt = 1, gr = 1): number {
  return (pt * gt * gr * lambda * lambda) / (16 * Math.PI * Math.PI * d * d);
}

/**
 * Trilatération exacte 2D : en différenciant l'équation i de l'équation 0,
 * le problème devient linéaire A·[x,y] = c, résolu en forme fermée 2×2
 * (équations normales). 3 balises à ranges exacts → position exacte.
 */
export function trilaterate(beacons: Beacon[], ranges: number[]): { x: number; y: number; ok: boolean } {
  const n = Math.min(beacons.length, ranges.length);
  if (n < 3) return { x: 0, y: 0, ok: false };
  const x0 = beacons[0].x, y0 = beacons[0].y, r0 = ranges[0];
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  for (let i = 1; i < n; i++) {
    const Ai = 2 * (x0 - beacons[i].x);
    const Bi = 2 * (y0 - beacons[i].y);
    const Ci = ranges[i] * ranges[i] - r0 * r0
      + (x0 * x0 - beacons[i].x * beacons[i].x)
      + (y0 * y0 - beacons[i].y * beacons[i].y);
    a11 += Ai * Ai; a12 += Ai * Bi; a22 += Bi * Bi;
    b1 += Ai * Ci; b2 += Bi * Ci;
  }
  const det = a11 * a22 - a12 * a12;
  if (Math.abs(det) < 1e-18) return { x: 0, y: 0, ok: false }; // balises alignées
  return { x: (a22 * b1 - a12 * b2) / det, y: (a11 * b2 - a12 * b1) / det, ok: true };
}

export interface SwarmParams {
  kPh: number;   // gain de phototaxie (gradient normalisé → vitesse de croisière)
  kSep: number;  // gain de séparation mutuelle
  rSep: number;  // rayon de séparation (unités normalisées)
  maxV: number;  // vitesse max
  dt: number;    // pas d'intégration
}

/**
 * Un pas d'essaim : phototaxie (grimpe le gradient de lumière) + séparation
 * + murs mous + amortissement + clamp de vitesse. Mute les drones en place.
 */
export function swarmStep(drones: Drone[], beacons: Beacon[], p: SwarmParams): void {
  const n = drones.length;
  for (let i = 0; i < n; i++) {
    const d = drones[i];
    // phototaxie : direction = gradient normalisé (source unique → droite)
    const { gx, gy } = lightGrad(beacons, d.x, d.y);
    const gn = Math.hypot(gx, gy) + 1e-12;
    let ax = (gx / gn) * p.kPh;
    let ay = (gy / gn) * p.kPh;
    // séparation : répulsion 1/r dans le rayon rSep
    if (p.kSep > 0) {
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const o = drones[j];
        const dx = d.x - o.x, dy = d.y - o.y;
        const r2 = dx * dx + dy * dy;
        if (r2 < p.rSep * p.rSep && r2 > 1e-12) {
          const r = Math.sqrt(r2);
          const f = p.kSep * (1 - r / p.rSep) / r;
          ax += dx * f; ay += dy * f;
        }
      }
      // murs mous (bord de l'arène unité)
      const W = 0.07;
      if (d.x < W) ax += p.kSep * (W - d.x) / W;
      if (d.x > 1 - W) ax -= p.kSep * (d.x - (1 - W)) / W;
      if (d.y < W) ay += p.kSep * (W - d.y) / W;
      if (d.y > 1 - W) ay -= p.kSep * (d.y - (1 - W)) / W;
    }
    // intégration + amortissement + clamp
    d.vx = (d.vx + ax * p.dt) * 0.985;
    d.vy = (d.vy + ay * p.dt) * 0.985;
    const v = Math.hypot(d.vx, d.vy);
    if (v > p.maxV) { d.vx *= p.maxV / v; d.vy *= p.maxV / v; }
    d.x += d.vx * p.dt;
    d.y += d.vy * p.dt;
    d.trail.push(d.x, d.y);
    if (d.trail.length > 80) d.trail.splice(0, d.trail.length - 80);
  }
}

/** Distance moyenne de chaque drone à la balise la plus proche. */
export function meanNearestDist(drones: Drone[], beacons: Beacon[]): number {
  let s = 0;
  for (const d of drones) {
    let m = Infinity;
    for (const b of beacons) {
      const r = Math.hypot(d.x - b.x, d.y - b.y);
      if (r < m) m = r;
    }
    s += m;
  }
  return drones.length ? s / drones.length : 0;
}

// ---------------------------------------------------------------------------
//  Validation (appelée par validate.ts)
// ---------------------------------------------------------------------------

/** Phototaxie : convergence du drone vers la source. */
export function phototaxisError(): { dist: number; steps: number; ms: number } {
  const t0 = performance.now();
  const beacons: Beacon[] = [{ x: 0.85, y: 0.15, power: 1 }];
  const d: Drone = { x: 0.08, y: 0.92, vx: 0, vy: 0, trail: [] };
  const p: SwarmParams = { kPh: 2.2, kSep: 0, rSep: 0.08, maxV: 0.012, dt: 1 };
  let steps = 0;
  let dist = Math.hypot(d.x - beacons[0].x, d.y - beacons[0].y);
  for (; steps < 4000; steps++) {
    swarmStep([d], beacons, p);
    dist = Math.hypot(d.x - beacons[0].x, d.y - beacons[0].y);
    if (dist < 0.015) break;
  }
  return { dist, steps, ms: performance.now() - t0 };
}

/** Trilatération : position reconstruite de 3 ranges exacts. */
export function trilaterationError(): { err: number; ms: number } {
  const t0 = performance.now();
  const beacons: Beacon[] = [
    { x: 0.10, y: 0.10, power: 1 }, { x: 0.90, y: 0.20, power: 1 }, { x: 0.40, y: 0.90, power: 1 },
  ];
  const truth = { x: 0.62, y: 0.47 };
  const ranges = beacons.map((b) => Math.hypot(truth.x - b.x, truth.y - b.y));
  const est = trilaterate(beacons, ranges);
  return { err: Math.hypot(est.x - truth.x, est.y - truth.y), ms: performance.now() - t0 };
}

/** Friis : Pr(2d) = Pr(d)/4 exactement, sur 40 distances. */
export function friisError(): { rel: number; ms: number } {
  const t0 = performance.now();
  const pt = 0.5, lambda = 0.05;
  let worst = 0;
  for (let i = 1; i <= 40; i++) {
    const d = 0.1 + i * 0.05;
    const p1 = friisReceived(pt, lambda, d);
    const p2 = friisReceived(pt, lambda, 2 * d);
    worst = Math.max(worst, Math.abs(p2 - p1 / 4) / p1);
  }
  return { rel: worst, ms: performance.now() - t0 };
}
