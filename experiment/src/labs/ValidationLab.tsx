import { useState, useCallback, useEffect, useRef } from 'react';
import { runTest, TEST_ORDER, formatResidual, type TestResult, type TestId } from '../physics/validate';
import { Panel, Stat, Btn, Tag } from '../ui/kit';

export default function ValidationLab() {
  const [results, setResults] = useState<Partial<Record<TestId, TestResult>>>({});
  const [current, setCurrent] = useState<TestId | null>(null);
  const [queue, setQueue] = useState<TestId[]>([]);
  const [autoStarted, setAutoStarted] = useState(false);
  const startedAt = useRef(0);

  const runAll = useCallback(() => {
    setResults({});
    startedAt.current = performance.now();
    setQueue([...TEST_ORDER]);
  }, []);

  useEffect(() => {
    if (!autoStarted) { setAutoStarted(true); runAll(); }
  }, [autoStarted, runAll]);

  // Runs one test per animation frame so the UI stays responsive and the
  // progression is visible — the suite is genuinely executing, not replaying.
  useEffect(() => {
    if (queue.length === 0) { setCurrent(null); return; }
    const id = requestAnimationFrame(() => {
      const [head, ...rest] = queue;
      setCurrent(head);
      const r = runTest(head);
      setResults((p) => ({ ...p, [head]: r }));
      setQueue(rest);
    });
    return () => cancelAnimationFrame(id);
  }, [queue]);

  const all = TEST_ORDER.map((id) => results[id]).filter(Boolean) as TestResult[];
  const passed = all.filter((r) => r.pass).length;
  const failed = all.filter((r) => !r.pass).length;
  const totalMs = all.reduce((s, r) => s + r.ms, 0);
  const done = all.length === TEST_ORDER.length;
  const progress = all.length / TEST_ORDER.length;

  const groups = Array.from(new Set(TEST_ORDER.map((id) => results[id]?.group).filter(Boolean))) as string[];

  return (
    <div className="space-y-3">
      <Panel
        tag="LOOP"
        title="Boucle grounded — chaque affirmation est vérifiée numériquement"
        subtitle="Aucun chiffre affiché dans ce moteur n'est arbitraire. Chaque test confronte le solveur à une référence indépendante : solution analytique, loi de conservation, énumération exhaustive ou dérivée par différences finies."
        right={<Btn tone="primary" onClick={runAll} disabled={queue.length > 0}>{queue.length > 0 ? '⋯ exécution' : '↻ Tout relancer'}</Btn>}
      >
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Tests réussis" value={`${passed}/${TEST_ORDER.length}`} tone={failed === 0 && done ? 'green' : 'cyan'} />
          <Stat label="Échecs" value={failed} tone={failed > 0 ? 'rose' : 'slate'} />
          <Stat label="Temps total" value={totalMs.toFixed(0)} unit="ms" tone="violet" hint="calcul réel, dans ce navigateur" />
          <Stat label="Statut" value={done ? (failed === 0 ? 'VALIDÉ' : 'ANOMALIE') : 'EN COURS'} tone={done ? (failed === 0 ? 'green' : 'rose') : 'amber'} />
        </div>

        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-white/6">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-400 transition-all duration-200"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </Panel>

      {groups.map((g) => {
        const rows = TEST_ORDER.map((id) => results[id]).filter((r): r is TestResult => !!r && r.group === g);
        if (rows.length === 0) return null;
        const gPass = rows.every((r) => r.pass);
        return (
          <Panel
            key={g}
            tag={g.slice(0, 4).toUpperCase()}
            title={g}
            right={<Tag tone={gPass ? 'green' : 'rose'}>{rows.filter((r) => r.pass).length}/{rows.length}</Tag>}
            flush
          >
            <div className="divide-y divide-white/5">
              {rows.map((r) => (
                <div key={r.name} className="px-3.5 py-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className={`inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold ${
                          r.pass ? 'bg-emerald-400/15 text-emerald-300 ring-1 ring-emerald-400/30'
                                 : 'bg-rose-500/15 text-rose-300 ring-1 ring-rose-500/30'}`}>
                          {r.pass ? '✓' : '✕'}
                        </span>
                        <span className="text-[11px] font-semibold text-slate-100">{r.name}</span>
                        <span className="font-mono text-[9px] text-slate-600">{r.ms.toFixed(1)} ms</span>
                      </div>
                      <p className="mt-1 pl-6 text-[10px] leading-relaxed text-slate-400">{r.claim}</p>
                      <p className="mt-0.5 pl-6 text-[10px] leading-relaxed text-slate-600">
                        <span className="text-slate-500">Référence : </span>{r.reference}
                      </p>
                      {r.extra && <p className="mt-0.5 pl-6 font-mono text-[10px] text-cyan-300/70">{r.extra}</p>}
                    </div>
                    <div className="shrink-0 text-right font-mono">
                      <div className={`text-[13px] font-semibold tabular-nums ${r.pass ? 'text-emerald-300' : 'text-rose-300'}`}>
                        {formatResidual(r)}
                      </div>
                      <div className="text-[9px] text-slate-600">
                        {r.lowerIsBetter ? 'seuil ≤ ' : 'seuil ≥ '}{r.tolerance.toExponential(0)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        );
      })}

      {current && queue.length > 0 && (
        <div className="rounded-lg border border-cyan-400/20 bg-cyan-400/5 px-3.5 py-2 font-mono text-[10px] text-cyan-300">
          ▸ exécution : {current} …
        </div>
      )}

      <Panel tag="MÉTHODE" title="Ce que « grounded » veut dire ici">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {[
            ['Solutions analytiques', 'La diffraction gaussienne, l’indice effectif du guide plan et la DFT d’une exponentielle complexe ont des expressions fermées. Le solveur est confronté directement à elles.'],
            ['Lois de conservation', 'Parseval pour la FFT, la puissance optique pour le BPM, l’unitarité pour le maillage MZI. Ces quantités doivent être invariantes — toute dérive est une erreur numérique mesurable.'],
            ['Énumération exhaustive', 'Pour n ≤ 22 sommets, l’optimum MaxCut est calculé en énumérant les 2ⁿ⁻¹ configurations par code de Gray. Le ratio d’approximation de la CIM n’est donc pas estimé, il est exact.'],
            ['Différences finies', 'Les dérivées de B-splines et le gradient complet du KAN sont comparés à des différences centrées. Une rétropropagation fausse serait immédiatement visible.'],
          ].map(([t, d]) => (
            <div key={t} className="rounded-lg border border-white/6 bg-white/[0.015] p-2.5">
              <div className="text-[10px] font-semibold text-cyan-300">{t}</div>
              <p className="mt-1 text-[10px] leading-relaxed text-slate-500">{d}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[10px] leading-relaxed text-slate-500">
          Les seuils sont volontairement serrés : la plupart des tests exigent la précision machine (10⁻¹²).
          Là où la physique impose une erreur de discrétisation — diffraction gaussienne, indice effectif —
          le seuil reflète l'ordre de convergence attendu du schéma, pas une marge de confort.
        </p>
      </Panel>
    </div>
  );
}
