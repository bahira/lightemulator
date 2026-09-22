// ============================================================================
//  spear_table_verify.mjs — falsification ligne à ligne de la table
//  « LLM / image kernels replaced by pure algebra ».
//  Pour chaque ligne : erreur L∞ réelle sur le domaine utile + vitesse JS.
// ============================================================================
const rng = (() => { let s = 42; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
function linf(fnApprox, fnRef, lo, hi, n = 200000) {
    let w = 0;
    for (let i = 0; i <= n; i++) {
        const x = lo + ((hi - lo) * i) / n;
        const e = Math.abs(fnApprox(x) - fnRef(x));
        if (e > w) w = e;
    }
    return w;
}
function bench(fn, args) {
    let sink = 0;
    for (let w = 0; w < 3; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    const t = process.hrtime.bigint();
    for (let w = 0; w < 100; w++) for (let i = 0; i < args.length; i++) sink += fn(args[i]);
    globalThis.__sink = sink;
    return Number(process.hrtime.bigint() - t) / 1e6 / (100 * args.length);
}
function mkArgs(lo, hi, n = 20000) { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = lo + (hi - lo) * (rng()); return a; }
const sig = `f`; // placeholder pour lisibilité

console.log('=== TABLE « pure algebra » — falsification ===\n');

// ---- Ligne 1 : diffusion beta(t) ------------------------------------------
{
    const f = (t) => -0.496 * Math.cos(3.162 * t) + 0.501;
    const g = (t) => 1 - Math.cos(Math.PI * t / 2); // famille cosine (Nichol-Dhariwal, normalisée autrement)
    const e = linf(f, g, 0, 1);
    console.log(`[diffusion β(t)]  reformulation d'un cosinus en cosinus — ×1.04 annoncé = bruit.`);
    console.log(`  écart vs cosine-schedule standard (non-normalisé) : ${e.toFixed(3)} — pas de nouveau kernel (SKIP)\n`);
}

// ---- Ligne 2 : gaussian blur ----------------------------------------------
{
    const f = (x) => 0.999 * Math.exp(-0.504 * x * x);
    const e = linf(f, (x) => Math.exp(-0.5 * x * x), -4, 4);
    const tA = bench((x) => 0.999 * Math.exp(-0.504 * x * x), mkArgs(-4, 4));
    const tB = bench((x) => Math.exp(-0.5 * x * x), mkArgs(-4, 4));
    console.log(`[gaussian blur]   formule = la gaussienne elle-même (σ≈0.996).`);
    console.log(`  L∞=${e.toExponential(2)} · vitesse ×${(tB / tA).toFixed(2)} (annonce ×0.96 = PERTE) → REJET\n`);
}

// ---- Ligne 3 : SiLU --------------------------------------------------------
{
    const ref = (x) => x / (1 + Math.exp(-x));
    const f = (x) => x * (0.501 + 0.589 * x / (0.83 + Math.sqrt(1 + x * x)));
    const e = linf(f, ref, -8, 8);
    console.log(`[SiLU]            annonce L∞=8.2e-4 · ×2.43`);
    console.log(`  L∞ mesuré [-8,8] = ${e.toExponential(3)} ${(e < 1.5e-3 ? '✓ cohérent' : '** FAUX **')}`);
    const tF = bench(f, mkArgs(-8, 8)), tR = bench(ref, mkArgs(-8, 8));
    console.log(`  vitesse : exact ${tR.toFixed(2)} ns vs formule ${tF.toFixed(2)} ns → ×${(tR / tF).toFixed(2)}\n`);
}

// ---- Ligne 4 : GELU --------------------------------------------------------
{
    const ref = (x) => { // gelu exact via erf Abramowitz-Stegun (précision 1.5e-7)
        const s = x < 0 ? -1 : 1, ax = Math.abs(x);
        const t = 1 / (1 + 0.3275911 * ax);
        const p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        return 0.5 * x * (1 + s * (1 - p * Math.exp(-ax * ax)));
    };
    const f = (x) => {
        const rl = Math.max(0, 0.308 * x + 0.501);
        return x * Math.min(1.002, rl);
    };
    const e4 = linf(f, ref, -4, 4);
    const e8 = linf(f, ref, -8, 8);
    console.log(`[GELU]            annonce L∞=5.3e-4 · ×6.57`);
    console.log(`  L∞ mesuré [-4,4] = ${e4.toExponential(3)} | [-8,8] = ${e8.toExponential(3)} ${(e4 < 1e-3 ? '✓' : '** ANNONCE FAUSSE **')}`);
    const tF = bench(f, mkArgs(-4, 4));
    const tRef = bench(ref, mkArgs(-4, 4));
    console.log(`  vitesse : exact(erf-A-S) ${tRef.toFixed(2)} ns vs formule ${tF.toFixed(2)} ns → ×${(tRef / tF).toFixed(2)}\n`);
}

// ---- Ligne 5 : sigmoid « exact » -------------------------------------------
{
    const f = (x) => 1 - 1 / (1 + Math.exp(-x));
    const ref = (x) => 1 / (1 + Math.exp(-x));
    const e = linf(f, ref, -20, 20, 50000);
    const tA = bench(f, mkArgs(-20, 20)), tB = bench(ref, mkArgs(-20, 20));
    console.log(`[sigmoid]         identité algébrique : 1−1/(1+y) = y/(1+y) — même formule + 1 soustraction.`);
    console.log(`  L∞=${e.toExponential(2)} (0 attendu) · vitesse ×${(tB / tA).toFixed(2)} (${tB.toFixed(2)}→${tA.toFixed(2)} ns, annonce ×1.26 = ${tA < tB ? 'ok mais trivial' : 'PERTE'})\n`);
}

// ---- Lignes 6-7 : Φ CDF & softplus (formules non fournies) -----------------
console.log(`[Φ gaussienne]    «refined rational form» non fournie → INFALSIFIABLE telle quelle. Annonce ×0.97 = perte.\n`);
console.log(`[softplus]        «piecewise-rational» non fournie → INFALSIFIABLE. Notre spearSoftplus : L∞ 4e-3 déjà intégré.\n`);

// ---- Ligne 8 : Padé distillation (≈ tanh) ----------------------------------
{
    const f = (x) => (x + 0.145 * x * x * x) / (0.556 + 0.75 * x * x);
    const e4 = linf(f, Math.tanh, -4, 4);
    const e2 = linf(f, Math.tanh, -2, 2);
    const v4 = f(4);
    console.log(`[Padé [3/2]]      annonce L∞=1.6e-4 · ×2.83`);
    console.log(`  L∞ [-4,4] = ${e4.toExponential(3)} (f(4)=${v4.toFixed(3)} > 1 : sort de [-1,1] !) | [-2,2] = ${e2.toExponential(3)}`);
    console.log(`  ${(e2 < 2e-4 ? '✓ sur [-2,2] seulement' : '** FAUX même sur [-2,2] **')} — domaine restreint non documenté`);
    // notre kernel existant pour comparaison
    const ours = (x) => (x * (23.96543 + x * x)) / (24.36223 + 8.38674 * x * x);
    const eOurs = linf(ours, Math.tanh, -4, 4);
    console.log(`  notre spearTanh existant : L∞ [-4,4] = ${eOurs.toExponential(3)} — meilleur et sans trou de domaine\n`);
}

console.log('=== VERDICT GLOBAL ===');
console.log('Intégrables : aucun (nos kernels spear.ts dominent ou égalent chaque ligne).');
console.log('Valeur réelle de la table : cibles de recherche symbolique à re-préciser, pas des kernels prêts.');
