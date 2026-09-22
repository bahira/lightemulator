// ============================================================================
//  spear_lattice_verify.mjs ??? FALSIFICATION du papier Voronoi-Fisher/E8.
//  C1 : loi de distorsion E||R||?? = d??G(??)??V^{2/d}  (Z??, A2, rect 2:1)
//  C2 : gain de forme E8 vs Z8 = G_Z8/G_E8 ??? 1.1622 + correction algorithme E8
//  C3 : Root-E8 "gradient exact variance nulle" ??? vrai pour quadratique,
//       FAUX en g??n??ral (quadrature d??terministe, pas miracle)
// ============================================================================
const rng = (() => { let s = 987654321; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
function gauss() { const u = rng() + 1e-12, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

console.log('=== C1. Loi de distorsion E||R||?? = d??G(??)??V ===');
{
  const N = 2_000_000, V = 36.0;
  // Z??
  {
    const d = Math.sqrt(V);
    let mse = 0;
    for (let i = 0; i < N; i++) {
      const x = 128 * gauss(), y = 128 * gauss();
      const rx = x - Math.round(x / d) * d, ry = y - Math.round(y / d) * d;
      mse += rx * rx + ry * ry;
    }
    mse /= N;
    const theo = 2 * (1 / 12) * V;
    console.log(`  Z2    : MSE=${mse.toFixed(4)} th??o=${theo.toFixed(4)} ??cart=${(Math.abs(mse - theo) / theo * 100).toFixed(2)}%`);
  }
  // A2 hexagonal (V = ???? = 36 ??? ?? = 6)
  {
    const D = Math.sqrt(72 / Math.sqrt(3)); // V = sqrt(3)/2 * D^2 = 36
    const u1 = [D, 0], u2 = [D / 2, D * Math.sqrt(3) / 2];
    let mse = 0;
    for (let i = 0; i < N; i++) {
      const x = 128 * gauss(), y = 128 * gauss();
      // coordonn??es duales : q,r tels que point = q??u1 + r??u2
      // duales exactes : M=[[D,D/2],[0,sqrt3*D/2]] -> q = x/D - y/(sqrt3*D), r = 2y/(sqrt3*D)
      const q = x / D - y / (Math.sqrt(3) * D);
      const r = 2 * y / (Math.sqrt(3) * D);
      // arrondi hexagonal standard via cube
      let rx = Math.round(q), rz = Math.round(r), ry = Math.round(-q - r);
      const dx = Math.abs(rx - q), dy = Math.abs(ry - (-q - r)), dz = Math.abs(rz - r);
      if (dx > dy && dx > dz) rx = -ry - rz; else if (dy > dz) ry = -rx - rz; else rz = -rx - ry;
      const px = rx * u1[0] + rz * u2[0], py = rx * u1[1] + rz * u2[1];
      mse += (x - px) ** 2 + (y - py) ** 2;
    }
    mse /= N;
    const theo = 2 * (5 / (36 * Math.sqrt(3))) * V;
    console.log(`  A2    : MSE=${mse.toFixed(4)} th??o=${theo.toFixed(4)} ??cart=${(Math.abs(mse - theo) / theo * 100).toFixed(2)}%`);
  }
  // rect 2:1 (w??h=36, w=2h)
  {
    const h = Math.sqrt(V / 2), w = 2 * h;
    let mse = 0;
    for (let i = 0; i < N; i++) {
      const x = 128 * gauss(), y = 128 * gauss();
      const rx = x - Math.round(x / w) * w, ry = y - Math.round(y / h) * h;
      mse += rx * rx + ry * ry;
    }
    mse /= N;
    const theo = 2 * (5 / 48) * V;
    console.log(`  rect2:1: MSE=${mse.toFixed(4)} th??o=${theo.toFixed(4)} ??cart=${(Math.abs(mse - theo) / theo * 100).toFixed(2)}%`);
  }
}

console.log('\n=== C2. E8 vs Z8 : gain de forme + algorithme de projection ===');
// projection D8 : arrondi, parit?? corrig??e sur la coordonn??e la plus loin
function projD8(v) {
  const o = new Array(8); let sum = 0, maxd = -1, mi = 0;
  for (let i = 0; i < 8; i++) { o[i] = Math.round(v[i]); sum += o[i]; const dd = Math.abs(v[i] - o[i]); if (dd > maxd) { maxd = dd; mi = i; } }
  if ((sum & 1) !== 0) o[mi] += (v[mi] > o[mi]) ? 1 : -1;
  return o;
}
function projE8(v) {
  const z0 = projD8(v);
  const z1 = projD8(v.map(x => x - 0.5)).map(x => x + 0.5);
  let d0 = 0, d1 = 0;
  for (let i = 0; i < 8; i++) { d0 += (v[i] - z0[i]) ** 2; d1 += (v[i] - z1[i]) ** 2; }
  return d0 <= d1 ? z0 : z1;
}
{
  const N = 500_000;
  // gain de forme ?? volume ??gal : E8 V=1 vs Z8 V=1
  let mseZ = 0, mseE = 0;
  for (let i = 0; i < N; i++) {
    const v = []; for (let j = 0; j < 8; j++) v.push(gauss() * 2);
    let z = 0, e = 0; const qe = projE8(v);
    for (let j = 0; j < 8; j++) { const rz = v[j] - Math.round(v[j]); z += rz * rz; const re = v[j] - qe[j]; e += re * re; }
    mseZ += z; mseE += e;
  }
  const gZ = mseZ / (N * 8), gE = mseE / (N * 8);
  const theo = (1 / 12) / 0.0716821;
  console.log(`  G mesur?? Z8=${gZ.toFixed(5)} (th??o 0.083333) | G mesur?? E8=${gE.toFixed(5)} (th??o 0.071682)`);
  console.log(`  gain de forme E8/Z8 = ${(gZ / gE).toFixed(4)} (th??orie ${(theo).toFixed(4)}, soit ${(10 * Math.log10(theo)).toFixed(2)} dB)`);
  // correction de l'algorithme : distance au projet?? ??? distance ?? un point E8 al??atoire perturb??
  let bad = 0;
  for (let i = 0; i < 20000; i++) {
    const v = []; for (let j = 0; j < 8; j++) v.push(gauss() * 3);
    const q = projE8(v);
    const dQ = v.reduce((a, x, j) => a + (x - q[j]) ** 2, 0);
    // perturber le projet?? d'un vecteur du r??seau (reste dans E8)
    const lat = [1, -1, 0, 0, 0, 0, 0, 0];
    const cand = q.map((x, j) => x + lat[j]);
    const dC = v.reduce((a, x, j) => a + (x - cand[j]) ** 2, 0);
    if (dC < dQ - 1e-9) bad++;
  }
  console.log(`  violations de minimalit?? sur 20k essais : ${bad} ${bad === 0 ? '(algorithme correct)' : '** ALGO FAUX **'}`);
}

console.log('\n=== C3. Root-E8 ZO : "gradient exact variance nulle" ===');
{
  const roots = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++)
    for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) { const v = new Array(8).fill(0); v[i] = s1; v[j] = s2; roots.push(v); }
  for (let m = 0; m < 256; m++) {
    const v = []; let par = 0;
    for (let k = 0; k < 8; k++) { const b = (m >> k) & 1; v.push(b ? -0.5 : 0.5); par ^= b; }
    if (par % 2 === 0) roots.push(v);
  }
  function estGrad(L, x, mu) {
    const g = new Array(8).fill(0);
    for (const r of roots) {
      const u = r.map(v => v / Math.SQRT2);
      const xp = x.map((v, i) => v + mu * u[i]), xm = x.map((v, i) => v - mu * u[i]);
      const d = (L(xp) - L(xm)) / (2 * mu);
      for (let i = 0; i < 8; i++) g[i] += d * u[i];
    }
    return g.map(v => v / 30);
  }
  // quadratique : la revendication "0.0000% d'erreur"
  {
    const H = [], x0 = [];
    for (let i = 0; i < 8; i++) { H.push(new Array(8).fill(0)); H[i][i] = Math.pow(10, i / 7); x0.push(1); }
    const L = (x) => { let s = 0; for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) s += 0.5 * x[i] * H[i][j] * x[j]; return s; };
    const g = estGrad(L, x0, 1e-4);
    const gt = H.map((row, i) => row.reduce((a, v, j) => a + v * x0[j], 0));
    const err = Math.hypot(...g.map((v, i) => v - gt[i])) / Math.hypot(...gt);
    console.log(`  quadratique : erreur relative = ${(err * 100).toFixed(6)}% ${err < 1e-10 ? '(exact, revendication OK sur ce cas)' : ''}`);
  }
  // non-quadratique : la "variance nulle" doit fuir
  {
    const L = (x) => Math.sin(x[0] * 3) + x[1] ** 3 + Math.exp(-x[2]) + Math.tanh(x[3] * 2) + x.slice(4).reduce((a, v) => a + v * v * v * 0.1, 0);
    const x = new Array(8).fill(0.7);
    const h = 1e-6;
    const gt = [];
    for (let i = 0; i < 8; i++) {
      const xp = x.slice(), xm = x.slice(); xp[i] += h; xm[i] -= h;
      gt.push((L(xp) - L(xm)) / (2 * h));
    }
    const g = estGrad(L, x, 1e-3);
    const err = Math.hypot(...g.map((v, i) => v - gt[i])) / (Math.hypot(...gt) + 1e-12);
    console.log(`  non-quadratique : erreur relative = ${(err * 100).toFixed(3)}% ${err > 0.01 ? '(variance NULLE = FAUX en g??n??ral ??? quadrature d??terministe seulement)' : ''}`);
  }
}



