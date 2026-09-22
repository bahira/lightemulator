// ============================================================================
//  selftest.ts — boucle grounded en headless (CI).
//
//  Bundle via esbuild (dépendance de vite) puis exécution node :
//    npm run test
//  Code de sortie = nombre d'échecs. Aucune assertion cachée : chaque test
//  affiche son résidu mesuré et sa tolérance.
// ============================================================================

import { runTest, TEST_ORDER, type TestResult } from './validate';

function fmt(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  return v.toPrecision(6);
}

export function main(): number {
  console.log(`╔══════════════════════════════════════════════════════════════╗`);
  console.log(`║  BOUCLE GROUNDED — ${TEST_ORDER.length} tests · chaque chiffre est mesuré      ║`);
  console.log(`╚══════════════════════════════════════════════════════════════╝`);
  let fails = 0;
  let totalMs = 0;
  const rows: string[] = [];
  for (const id of TEST_ORDER) {
    let r: TestResult;
    try {
      r = runTest(id);
    } catch (e) {
      console.error(`CRASH  ${id}: ${e instanceof Error ? e.message : String(e)}`);
      fails++;
      continue;
    }
    totalMs += r.ms;
    const ok = r.pass;
    if (!ok) fails++;
    rows.push(
      `${ok ? 'PASS' : 'FAIL'}  [${r.group}] ${r.name}  ` +
      `résidu=${fmt(r.measured)} ${r.lowerIsBetter ? '≤' : '≥'} tol=${fmt(r.tolerance)}  (${r.ms.toFixed(1)} ms)`,
    );
    console.log(rows[rows.length - 1]);
    if (r.extra) console.log(`      · ${r.extra}`);
  }
  console.log('─'.repeat(64));
  console.log(`${fails === 0 ? '✅ VALIDÉ' : `❌ ${fails} ÉCHEC(S)`} — ${TEST_ORDER.length} tests en ${totalMs.toFixed(0)} ms`);
  return fails;
}

// exécution directe (node)
if (typeof process !== 'undefined' && process.argv?.[1]?.includes('selftest')) {
  process.exitCode = main();
}
