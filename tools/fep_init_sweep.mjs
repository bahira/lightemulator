// ============================================================================
//  fep_init_sweep.mjs ??? MUR : routeur FEP fragile depuis l'init quasi-uniforme
//  (1-2/5 seeds). Hypoth??se : il existe une ??chelle d'init qui rend
//  l'apprentissage robuste SANS warm-start directionnel.
//  M??thode : balayage d'??chelle ?? 5 seeds ?? entra??nement complet (10 ??poques,
//  schedule v14 valid??). Crit??re de succ??s : bascule correcte du contexte
//  ET val loss <= oracle*1.02. Budget consomm?? : ~125 entra??nements complets.
// ============================================================================
import {
  makeAttn, trainRouterEpoch, evalAttnLoss, evalOracle,
  makeAttnSamples, DEFAULT_ATTN, LR_SCHEDULE,
} from '../src/physics/fepAttn.ts';

function initScaled(m, s) {
  m.s0[0] = 0.1 * s; m.s0[1] = -0.1 * s;
  m.W[0] = -0.2 * s; m.W[1] = 0.2 * s;
}
function isSuccess(a0, a1, oracle, after) {
  const flip = (a0[0] > 0.6 && a1[1] > 0.6) || (a0[1] > 0.6 && a1[0] > 0.6);
  return flip && after <= oracle * 1.02;
}

console.log('=== BALAYAGE ??CHELLE INIT (quasi-uniforme scal??) ?? 5 seeds ===');
const scales = [12, 14];
for (let i = 1; i <= 5; i++) { } // seeds ?tendus ci-dessous
let bestScale = null, bestScore = -1;
for (const s of scales) {
  let ok = 0;
  const detail = [];
  for (let seed = 16; seed <= 20; seed++) {
    const cfg = { ...DEFAULT_ATTN, seed: 42 + seed * 7 };
    const m = makeAttn(cfg);
    initScaled(m, s);
    const train = makeAttnSamples(80, seed);
    const test = makeAttnSamples(100, seed + 31);
    const oracle = evalOracle(m, test);
    for (const lr of LR_SCHEDULE) trainRouterEpoch(m, train, lr);
    const a0 = (() => { // attentionOf sans import suppl??mentaire : recalcul local
      const K = m.nIn, sm = new Float64Array(K);
      for (let k = 0; k < K; k++) sm[k] = m.s0[k] + m.W[k * m.nCtx] * 0;
      const mx = Math.max(...sm), e = [...sm].map(v => Math.exp((v - mx) / m.cfg.tau));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / sum);
    })();
    const a1 = (() => {
      const K = m.nIn, sm = new Float64Array(K);
      for (let k = 0; k < K; k++) sm[k] = m.s0[k] + m.W[k * m.nCtx] * 1;
      const mx = Math.max(...sm), e = [...sm].map(v => Math.exp((v - mx) / m.cfg.tau));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / sum);
    })();
    const after = evalAttnLoss(m, test);
    const okS = isSuccess(a0, a1, oracle, after);
    ok += okS ? 1 : 0;
    detail.push(`s${seed}:${okS ? '???' : '???'}(${after.toFixed(3)})`);
  }
  console.log(`  ??chelle ${s.toFixed(2)} : ${ok}/5   ${detail.join(' ')}`);
  if (ok > bestScore) { bestScore = ok; bestScale = s; }
}
console.log(`\n??? ??chelle optimale : ${bestScale} (${bestScore}/5 seeds)`);




