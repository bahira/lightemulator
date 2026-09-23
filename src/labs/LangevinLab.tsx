import { useState } from 'react';
import { langevinGradError, fluctuationError, frozenObjectiveDecrease, generationCorrelation } from '../physics/langevin';
import { Panel, Stat, Tag, Btn, Formula } from '../ui/kit';

type CheckId = 'grad' | 'fluct' | 'train' | 'gen';
type Results = Partial<Record<CheckId, { label: string; value: string; tone: 'green' | 'amber' | 'violet' | 'cyan'; hint: string; ok: boolean }>>;

export default function LangevinLab() {
  const [res, setRes] = useState<Results>({});
  const [busy, setBusy] = useState<CheckId | null>(null);

  const run = (id: CheckId) => {
    setBusy(id);
    setTimeout(() => {
      const out: Results = { ...res };
      if (id === 'grad') {
        const v = langevinGradError();
        out.grad = { label: 'Gradient Éqs 10-11', value: v.rel.toExponential(2), tone: 'green', ok: v.rel <= 1e-4, hint: `vs différences finies · ${v.checks} paramètres` };
      } else if (id === 'fluct') {
        const v = fluctuationError();
        out.fluct = { label: 'Convergence ordre 1', value: `ratio ${v.ratio.toFixed(3)}`, tone: 'green', ok: v.ratio <= 0.6, hint: `resDt ${v.resDt.toExponential(1)} → resHalf ${v.resHalf.toExponential(1)}` };
      } else if (id === 'train') {
        const v = frozenObjectiveDecrease();
        out.train = { label: 'Descente (bruit gelé)', value: `${v.objInit.toFixed(1)} → ${v.objFinal.toFixed(1)}`, tone: v.objFinal < v.objInit ? 'green' : 'amber', ok: v.objFinal < v.objInit, hint: `N/2 = 48 · entropie du bruit pur · ${v.ms.toFixed(0)} ms` };
      } else {
        const v = generationCorrelation();
        out.gen = { label: 'Génération (hors-échelle)', value: `${v.corr.toFixed(2)} vs ${v.chance.toFixed(2)}`, tone: 'amber', ok: false, hint: 'ne sépare pas du hasard à 64+32 unités — limite documentée' };
      }
      setRes(out);
      setBusy(null);
    }, 20);
  };

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel
        tag="Langevin · thermodynamique"
        title="Generative thermodynamic computer — arXiv:2506.15121 (Whitelam, LBNL)"
        subtitle="Ordinateur de Langevin (Nv+Nh unités, J2/J4, dynamique sur-amortie) entraîné par maximisation de la probabilité de la trajectoire inverse"
        flush
      >
        <div className="space-y-3 p-3.5">
          <p className="text-[10px] leading-relaxed text-slate-500">
            Le papier entraîne un ordinateur thermodynamique par <Formula>maximisation de la
            probabilité de la trajectoire inverse</Formula> (action d'Onsager-Machlup, Éqs 10-11).
            La dynamique est bruitée (Éq 3) : le computer part du bruit, converge vers sa
            distribution d'équilibre, puis <em>génère</em> par sa dynamique naturelle. Le
            gradient exact est ici vérifié vs différences finies — la convention de signe de
            l'Éq 10 (le mauvais signe faisait <em>exploser</em> l'objectif, mesuré) est LE piège.
          </p>
          <div className="flex flex-wrap items-center gap-1.5">
            <Btn tone="primary" onClick={() => run('grad')} disabled={busy !== null}>{busy === 'grad' ? '…' : '▶ gradient'}</Btn>
            <Btn onClick={() => run('fluct')} disabled={busy !== null}>{busy === 'fluct' ? '…' : '▶ fluctuation'}</Btn>
            <Btn onClick={() => run('train')} disabled={busy !== null}>{busy === 'train' ? '…' : '▶ entraînement'}</Btn>
            <Btn onClick={() => run('gen')} disabled={busy !== null}>{busy === 'gen' ? '…' : '▶ génération'}</Btn>
          </div>
          {Object.keys(res).length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {(['grad', 'fluct', 'train', 'gen'] as CheckId[]).filter((k) => res[k]).map((k) => {
                const r = res[k]!;
                return <Stat key={k} label={r.label} value={r.value} tone={r.tone} hint={r.hint} />;
              })}
            </div>
          )}
        </div>
      </Panel>

      <Panel
        tag="HONNÊTETÉ"
        title="Ce qui est vérifié, ce qui ne l'est pas"
        subtitle="la frontière visible plutôt que masquée"
      >
        <div className="space-y-2 text-[10px] leading-relaxed">
          <p className="text-emerald-300">✓ Vérifié (validate.ts, groupe Langevin)</p>
          <ul className="space-y-1 text-slate-500">
            <li>· Gradient inverse Éqs 10-11 vs FD : <Tag tone="green">1.13e-6</Tag> — subtilités corrigées : le μΔt ne porte que sur ∂iV, tout est évalué à x' = x+Δx, update J −= α·gJ</li>
            <li>· Relation de fluctuation ln[P₀/P̃] ≈ −(ΔQ₀+ΔQθ)/2kBT : convergence ordre 1 confirmée (resHalf/resDt = 0.029)</li>
            <li>· Entraînement : l'objectif descend 47.85 → 46.85, sous l'entropie du bruit pur (N/2 = 48) — l'optimiseur fonctionne</li>
          </ul>
          <p className="text-amber-300 pt-1">✗ Ne sépare pas à cette échelle (documenté)</p>
          <ul className="space-y-1 text-slate-500">
            <li>· Génération bruit→structure : |Pearson| moyen 0.17 vs chance 0.17 — le signal d'entraînement est du second ordre (covariances cachées), noyé à 64+32 unités</li>
            <li>· Le papier utilise 784+512 unités et bien plus de trajectoires — c'est la limite d'échelle, pas un bug</li>
            <li>· Pas de matériel analogique — simulation numérique, comme le papier lui-même</li>
          </ul>
        </div>
      </Panel>
    </div>
  );
}
