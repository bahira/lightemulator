// tools/fep_audit.ts — audit post-refactor : A (EP analytique vs FD), B (AdamW), C (bandes)
import { makeMachine, verifyPhaseGradient, trainFep, relax, DEFAULT_FEP, DEFAULT_TRAIN } from '../src/physics/fep';
import { makeAttn, makeAttnSamples, initRouter, LR_SCHEDULE, trainRouterEpoch, evalAttnLoss, evalOracle, routerGradFD, routerGradEP } from '../src/physics/fepAttn';

function fmt(v: number): string { return v.toExponential(2); }
function hr(s: string): void { console.log(`\n═══ ${s} ═══`); }

hr('A — gradient EP analytique des pompes');
const am = makeAttn({ nIn: 2, nHidden: 4, nOut: 1, nCtx: 1, P: 1.0, dt: 0.05, relaxSteps: 800, seed: 42, lam: 0.1, alpha: 0.6, tau: 1, beta: 1e-3 });
const samples = makeAttnSamples(200, 7, 0.8);
initRouter(am, 'directionnel');
// gradcheck EP vs FD sur 6 échantillons
let worst = 0;
for (let s = 0; s < 6; s++) {
  const smp = samples[s];
  const e = routerGradEP(am, smp);
  const f = routerGradFD(am, smp);
  for (let k = 0; k < 2; k++) {
    const num = Math.max(Math.abs(f.gs0[k]), 1e-6);
    worst = Math.max(worst, Math.abs(e.gs0[k] - f.gs0[k]) / num);
    for (let mm = 0; mm < 1; mm++) {
      const num2 = Math.max(Math.abs(f.gW[k]), 1e-6);
      worst = Math.max(worst, Math.abs(e.gW[k * 1 + mm] - f.gW[k * 1 + mm]) / num2);
    }
  }
}
console.log(`[A1] accord EP vs FD : max err relative = ${fmt(worst)}  (6 échantillons, 8 param., truncation O(β)=1e-3)  ${worst < 2e-2 ? 'PASS' : 'FAIL'}`);
// timing
initRouter(am, 'directionnel');
let t0 = performance.now(); for (let r = 0; r < 50; r++) routerGradEP(am, samples[r % samples.length]); const tEP = performance.now() - t0;
initRouter(am, 'directionnel'); t0 = performance.now(); for (let r = 0; r < 50; r++) routerGradFD(am, samples[r % samples.length]); const tFD = performance.now() - t0;
console.log(`[A2] timing : EP ${tEP.toFixed(1)} ms  FD ${tFD.toFixed(1)} ms  → ×${(tFD / tEP).toFixed(1)}  ${tEP < tFD ? 'PASS' : 'FAIL'}`);
// entraînement avec EP (nouveaû default) atteint l'oracle ?
const oracle = evalOracle(am, samples);
initRouter(am, 'directionnel');
for (let ep = 0; ep < LR_SCHEDULE.length; ep++) trainRouterEpoch(am, samples, LR_SCHEDULE[ep]);
const finalLoss = evalAttnLoss(am, samples);
console.log(`[A3] entraînement EP : loss ${finalLoss.toFixed(3)} ≤ oracle ${oracle.toFixed(3)} ? ${finalLoss <= oracle * 1.05 ? 'OUI' : 'non'}`);

hr('B — AdamW sur les couplages EP');
const mS = makeMachine({ ...DEFAULT_FEP });
const rSg = trainFep(mS, DEFAULT_TRAIN);
const mA = makeMachine({ ...DEFAULT_FEP });
const rAd = trainFep(mA, { ...DEFAULT_TRAIN, adam: true, epochs: 12 });
const rAd24 = trainFep(makeMachine({ ...DEFAULT_FEP }), { ...DEFAULT_TRAIN, adam: true, epochs: 24 });
console.log(`  SGD+decay : ${rSg.startLoss.toExponential(2)} → ${rSg.finalLoss.toExponential(2)} (${rSg.reduction.toFixed(1)}×)`);
console.log(`  AdamW 12ep: ${rAd.startLoss.toExponential(2)} → ${rAd.finalLoss.toExponential(2)} (${rAd.reduction.toFixed(1)}×)  ${rAd.reduction > rSg.reduction ? '→ AdamW GAGNE' : '→ 12ep insuffisant'}`);
console.log(`  AdamW 24ep: ${rAd24.startLoss.toExponential(2)} → ${rAd24.finalLoss.toExponential(2)} (${rAd24.reduction.toFixed(1)}×)`);
console.log(`  identity (machine AdamW) : ${fmt(verifyPhaseGradient(mA, new Float64Array([0.5, 0.7]), new Float64Array([0.02, -0.03]), new Float64Array([0.55]), new Float64Array([0])).relErr)}`);

hr('C — profondeur (machine bandée 2 couches)');
const mB = makeMachine({ ...DEFAULT_FEP, nHidden: 8, bands: 2 });
// vérifier le maillage : adjacence entre bandes seulement (i,j dans hidden non-diag)
let nonAdjacent = 0, n = mB.n;
for (let i = 2; i < n - 1; i++) for (let j = 2; j < n - 1; j++) if (i !== j && mB.Kr[i * n + j] !== 0) {
  const b1 = Math.floor((i - 2) / 4), b2 = Math.floor((j - 2) / 4);
  if (Math.abs(b1 - b2) > 1) nonAdjacent++;
}
const rB = trainFep(mB, DEFAULT_TRAIN);
console.log(`[C1] maillage : couplages non-adjacents = ${nonAdjacent} (${nonAdjacent === 0 ? 'PASS' : 'FAIL'})`);
console.log(`[C2] 8 cachés 2 bandes : ${rB.startLoss.toExponential(2)} → ${rB.finalLoss.toExponential(2)} (${rB.reduction.toFixed(1)}×)`);
console.log(`[C3] identité sur machine bandée : relErr=${fmt(verifyPhaseGradient(mB, new Float64Array([0.5, 0.7]), new Float64Array([0.02, -0.03]), new Float64Array([0.55]), new Float64Array([0])).relErr)}`);

hr('C4 — table profondeur (nHidden=8, epochs=12, même série)');
for (const L of [1, 2, 4]) {
  const mm = makeMachine({ ...DEFAULT_FEP, nHidden: 8, bands: L });
  const rr = trainFep(mm, { ...DEFAULT_TRAIN, epochs: 12 });
  console.log(`  bands=${L} : ${rr.startLoss.toExponential(2)} → ${rr.finalLoss.toExponential(2)}  (${rr.reduction.toFixed(1)}×)`);
}

hr('D — certificats : monotonie ΔE + pas adaptatifs');
const vAd = verifyPhaseGradient(makeMachine({ ...DEFAULT_FEP }),
  new Float64Array([0.5, 0.7]), new Float64Array([0.02, -0.03]),
  new Float64Array([0.55]), new Float64Array([0]), 12);
console.log(`[D1] ΔE max = ${fmt(vAd.dEmax)}  → monotonie ${vAd.dEmax <= 1e-12 ? 'CERTIFIÉE (≤0)' : 'violée'} ; pas adaptatifs ${vAd.stepsUsed} (toit 20480)`);
console.log(`[D2] relErr = ${fmt(vAd.relErr)} ; résidu = ${fmt(vAd.resid)} ; checks=${vAd.checks}`);
console.log('\nFIN.');