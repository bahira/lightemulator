// ============================================================================
//  hybrid_swingup_verify.mjs — FALSIFICATION du contrôleur hybride V1200 §4.
//  Loi : u = clamp((1−wc)·u_swing + wc·u_catch, ±2)
//    u_swing = −4.3278·θ̇·[½θ̇²+6(1−cosθ)−12]·cosθ     (pompage d'énergie, E0=12)
//    u_catch = −(1.7222·sinθ + 8.0402·θ̇)
//    wc = sigmoid(10.1786·(cosθ−0.70))
//  Dynamique : θ̈ = 6 sinθ + 4u   (autorité β·umax = 8 > α = 6 ✓)
//  TEST : départ pendule BAS (θ=π), atteinte+stabilisation du SOMMET.
//  Comparaison : π* = −(3.1 sinθ + 1.6 θ̇) seule (ne peut PAS swinguer).
// ============================================================================
const A = 6, B = 4, UMAX = 2;
const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

function uHybrid(th, thd) {
  const s = Math.sin(th), c = Math.cos(th);
  const Eerr = 0.5 * thd * thd + A * (1 - c) - 12;
  const uSwing = -4.3278 * thd * Eerr * c;
  const uCatch = -(1.7222 * s + 8.0402 * thd);
  const w = 1 / (1 + Math.exp(-clamp(10.1786 * (c - 0.70), -20, 20)));
  return clamp((1 - w) * uSwing + w * uCatch, -UMAX, UMAX);
}
function uPiStar(th, thd) {
  return clamp(-(3.1 * Math.sin(th) + 1.6 * thd), -UMAX, UMAX);
}

function simulate(uLaw, th0, thd0, T, dt) {
  let th = th0, thd = thd0;
  const N = Math.round(T / dt);
  let okLast = 0;
  for (let i = 0; i < N; i++) {
    const acc = A * Math.sin(th) + B * uLaw(th % (2 * Math.PI), thd);
    thd += acc * dt;
    th += thd * dt;
    // déviation à l'équilibre haut le plus proche
    let dev = ((th % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
    if (Math.abs(dev) < 0.15 && Math.abs(thd) < 0.5) okLast++;
    else okLast = 0;
  }
  return { th, thd, locked: okLast > Math.round(0.5 / dt) }; // verrouillé ≥ 0.5 s
}

console.log('=== SWING-UP GLOBAL depuis la position BASSE (θ₀=π±bruit, θ̇₀=0) ===');
let hybOK = 0, piOK = 0;
const Nseeds = 20, T = 12;
for (let s = 0; s < Nseeds; s++) {
  const jitter = (s / Nseeds - 0.5) * 0.4; // ±0.2 rad autour du bas
  const rH = simulate(uHybrid, Math.PI + jitter, 0, T, 1e-4);
  const rP = simulate(uPiStar, Math.PI + jitter, 0, T, 1e-4);
  hybOK += rH.locked ? 1 : 0;
  piOK += rP.locked ? 1 : 0;
}
console.log(`  HYBRIDE V1200 : ${hybOK}/${Nseeds} swings-up réussis`);
console.log(`  π* simple     : ${piOK}/${Nseeds} (attendu 0 — pas d'autorité de pompage)`);

console.log('\n=== Robustesse hybride : conditions initiales variées ===');
let ok2 = 0, tot2 = 0;
for (let th0 of [Math.PI - 0.5, Math.PI, Math.PI + 0.5]) for (let thd0 of [-1, 0, 1]) {
  tot2++;
  if (simulate(uHybrid, th0, thd0, 12, 1e-4).locked) ok2++;
}
console.log(`  9 CI mixtes : ${ok2}/${tot2} verrouillés au sommet`);

// temps moyen de verrouillage quand succès
{
  const times = [];
  for (let s = 0; s < 10; s++) {
    const jit = (s / 10 - 0.5) * 0.4;
    let th = Math.PI + jit, thd = 0, lockT = -1, okRun = 0;
    const dt = 1e-4;
    for (let i = 0; i < 12 / dt; i++) {
      const acc = A * Math.sin(th) + B * uHybrid(th % (2 * Math.PI), thd);
      thd += acc * dt; th += thd * dt;
      let dev = ((th % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      if (Math.abs(dev) < 0.15 && Math.abs(thd) < 0.5) { okRun++; if (okRun * dt >= 0.5 && lockT < 0) lockT = i * dt; }
      else okRun = 0;
    }
    if (lockT > 0) times.push(lockT);
  }
  if (times.length) console.log(`  temps de verrouillage : min=${Math.min(...times).toFixed(2)}s moy=${(times.reduce((a,b)=>a+b,0)/times.length).toFixed(2)}s max=${Math.max(...times).toFixed(2)}s (${times.length}/10 succès)`);
}
