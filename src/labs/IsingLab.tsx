import { useMemo, useState, useEffect, useCallback } from 'react';
import {
  makeGraph, layoutGraph, runCIM, bruteForceMaxCut, simulatedAnnealing, greedyLocalSearch,
  cutValue, parseWeightMatrix, DEFAULT_CIM, GRAPH_INFO, BRUTE_FORCE_MAX_N,
  type GraphKind, type Graph, type CimResult, type SolverResult,
} from '../physics/ising';
import { cimPhysicalTime } from '../physics/latency';
import {
  Panel, Stat, Slider, Btn, Tag, Formula,
  useCanvas, drawGrid, drawSeries, axisLabel, downloadFile, useHashParam,
} from '../ui/kit';

const KINDS: GraphKind[] = ['cubic', 'erdos', 'torus', 'sk', 'scalefree'];
type KindSel = GraphKind | 'custom';

interface Bench {
  graph: Graph;
  cim: CimResult;
  rows: { r: SolverResult; ratio: number }[];
  reference: number;
  exact: boolean;
}

export default function IsingLab() {
  // instance partageable par URL : #gkind=sk&gn=60
  const [kindStr, setKindStr] = useHashParam('gkind', 'cubic');
  const KIND_SEL: KindSel[] = ['cubic', 'erdos', 'torus', 'sk', 'scalefree', 'custom'];
  const kind = (KIND_SEL.includes(kindStr as KindSel) ? kindStr : 'cubic') as KindSel;
  const setKind = (k: KindSel) => setKindStr(k);
  const [nStr, setNStr] = useHashParam('gn', '20');
  const n = Math.max(8, Math.min(400, Number(nStr) || 20));
  const setN = (v: number) => setNStr(String(v));
  const [seed, setSeed] = useState(55);
  const [iterations, setIterations] = useState(1600);
  const [pump, setPump] = useState(1.1);
  const [xi, setXi] = useState(0.55);
  const [noise, setNoise] = useState(0.035);
  const [ahc, setAhc] = useState(true);
  const [trials, setTrials] = useState(4);
  const [busy, setBusy] = useState(false);
  const [bench, setBench] = useState<Bench | null>(null);
  const [playhead, setPlayhead] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [customText, setCustomText] = useState('');

  // graphe actif : généré, ou matrice collée par l'utilisateur
  const parsed = useMemo(() => (kind === 'custom' ? parseWeightMatrix(customText) : null), [kind, customText]);
  const graph = useMemo(
    () => (kind === 'custom' ? parsed ?? makeGraph('cubic', 20, seed) : makeGraph(kind, n, seed)),
    [kind, parsed, n, seed],
  );
  const layout = useMemo(() => {
    if (graph.n > 320) {
      const x = new Float32Array(graph.n), y = new Float32Array(graph.n);
      for (let i = 0; i < graph.n; i++) {
        const a = (i / graph.n) * Math.PI * 2;
        x[i] = 0.5 + 0.44 * Math.cos(a); y[i] = 0.5 + 0.44 * Math.sin(a);
      }
      return { x, y };
    }
    return layoutGraph(graph, graph.n > 160 ? 130 : 220, seed);
  }, [graph, seed]);

  const solve = useCallback(() => {
    setBusy(true);
    setTimeout(() => {
      // Keep the main thread responsive: the CIM costs O(m) per round trip, so
      // a dense SK instance needs fewer trials than a sparse cubic one for the
      // same wall-clock budget. Both numbers are reported in the results.
      const BUDGET = 4.5e7; // edge-updates
      const perTrial = Math.max(graph.m, 1) * iterations;
      const effTrials = Math.max(1, Math.min(trials, Math.floor(BUDGET / perTrial) || 1));
      const effIter = perTrial > BUDGET
        ? Math.max(300, Math.floor(BUDGET / Math.max(graph.m, 1)))
        : iterations;

      const cim = runCIM(graph, {
        ...DEFAULT_CIM, iterations: effIter, pEnd: pump, xi, noise, ahc,
        trials: effTrials, seed: seed + 7,
        sampleEvery: Math.max(1, Math.round(effIter / 260)),
      });
      const sa = simulatedAnnealing(graph, Math.max(120, Math.min(900, Math.round(50000 / Math.max(graph.n, 1)))), seed + 2);
      const gl = greedyLocalSearch(graph, 32, seed + 3);
      const all: SolverResult[] = [cim, sa, gl];
      let reference = Math.max(...all.map((r) => r.cut));
      let exact = false;
      if (graph.n <= BRUTE_FORCE_MAX_N) {
        const bf = bruteForceMaxCut(graph);
        all.unshift(bf);
        reference = bf.cut; exact = true;
      }
      setBench({
        graph, cim, reference, exact,
        rows: all.map((r) => ({ r, ratio: reference > 0 ? r.cut / reference : 0 })),
      });
      setPlayhead(1);
      setBusy(false);
    }, 20);
  }, [graph, iterations, pump, xi, noise, ahc, trials, seed]);

  useEffect(() => { setBench(null); }, [graph]);

  // export de la meilleure solution trouvée (CSV)
  const downloadSolution = useCallback(() => {
    if (!bench) return;
    const best = bench.rows.reduce((a, b) => (b.r.cut > a.r.cut ? b : a));
    const lines = [
      `# MaxCut — PHOTONIC ENGINE / ${bench.graph.name}`,
      `# methode,${best.r.method}`,
      `# coupe,${best.r.cut}`,
      `# reference,${bench.reference}${bench.exact ? ' (optimum exact)' : ' (meilleur trouve)'}`,
      'noeud,spin',
      ...Array.from(best.r.spins).map((s, i) => `${i},${s}`),
    ];
    downloadFile(`maxcut_${bench.graph.n}n_${best.r.cut}.csv`, lines.join('\n'), 'text/csv');
  }, [bench]);

  useEffect(() => {
    if (!playing || !bench) return;
    let id = 0;
    const loop = () => {
      setPlayhead((p) => (p >= 1 ? 0 : Math.min(1, p + 0.006)));
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing, bench]);

  // spins at the current playhead (reconstructed from the recorded trajectory)
  const liveSpins = useMemo(() => {
    const s = new Int8Array(graph.n);
    if (!bench) { s.fill(1); return s; }
    const c = bench.cim;
    const si = Math.min(c.nSamples - 1, Math.max(0, Math.round(playhead * (c.nSamples - 1))));
    if (playhead >= 0.999) return c.spins;
    // trajectory only stores a subset; use final x sign for the rest
    s.set(c.spins);
    for (let i = 0; i < c.nTrace; i++) s[c.traceIdx[i]] = c.trace[si * c.nTrace + i] >= 0 ? 1 : -1;
    return s;
  }, [bench, playhead, graph.n]);

  const liveCut = useMemo(() => cutValue(graph, liveSpins), [graph, liveSpins]);

  // ------------------------------------------------------------- graph view
  const gview = useCanvas((ctx, w, h) => {
    const pad = 14;
    const S = Math.min(w, h) - pad * 2;
    const ox = (w - S) / 2, oy = (h - S) / 2;
    const X = (i: number) => ox + layout.x[i] * S;
    const Y = (i: number) => oy + layout.y[i] * S;

    const showEdges = graph.m <= 4000;
    if (showEdges) {
      for (let k = 0; k < graph.m; k++) {
        const a = graph.eu[k], b = graph.ev[k];
        const cut = liveSpins[a] !== liveSpins[b];
        ctx.beginPath();
        ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b));
        if (cut) {
          ctx.strokeStyle = graph.ew[k] >= 0 ? 'rgba(103,232,249,0.55)' : 'rgba(251,113,133,0.5)';
          ctx.lineWidth = 1.1;
        } else {
          ctx.strokeStyle = 'rgba(148,163,184,0.10)';
          ctx.lineWidth = 0.6;
        }
        ctx.stroke();
      }
    }
    const r = Math.max(1.8, Math.min(6, 90 / Math.sqrt(graph.n)));
    for (let i = 0; i < graph.n; i++) {
      const up = liveSpins[i] === 1;
      ctx.beginPath();
      ctx.arc(X(i), Y(i), r + 2.5, 0, Math.PI * 2);
      ctx.fillStyle = up ? 'rgba(56,189,248,0.14)' : 'rgba(251,146,60,0.14)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(X(i), Y(i), r, 0, Math.PI * 2);
      ctx.fillStyle = up ? 'rgba(125,211,252,0.95)' : 'rgba(253,186,116,0.95)';
      ctx.fill();
    }
    axisLabel(ctx, `${graph.name} · ${graph.n} sommets, ${graph.m} arêtes`, 6, 12, 'left');
    axisLabel(ctx, `coupe = ${liveCut}`, w - 6, 12, 'right', 'rgba(103,232,249,0.95)');
  }, [graph, layout, liveSpins, liveCut], 0.82);

  // ------------------------------------------------------- CIM trajectories
  const traj = useCanvas((ctx, w, h) => {
    const pad = { l: 32, r: 10, t: 12, b: 18 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4 || !bench) {
      axisLabel(ctx, 'lancez une résolution', w / 2, h / 2, 'center');
      return;
    }
    drawGrid(ctx, pad.l, pad.t, iw, ih, 6, 4);
    const c = bench.cim;
    const ns = c.nSamples;
    const yOf = (v: number) => pad.t + ih / 2 - (v / 1.7) * (ih / 2);

    // zero axis
    ctx.beginPath();
    ctx.moveTo(pad.l, yOf(0)); ctx.lineTo(pad.l + iw, yOf(0));
    ctx.strokeStyle = 'rgba(255,255,255,0.14)'; ctx.lineWidth = 1; ctx.stroke();

    for (let t = 0; t < c.nTrace; t++) {
      ctx.beginPath();
      for (let s = 0; s < ns; s++) {
        const x = pad.l + (s / Math.max(ns - 1, 1)) * iw;
        const y = yOf(c.trace[s * c.nTrace + t]);
        if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      const final = c.trace[(ns - 1) * c.nTrace + t];
      ctx.strokeStyle = final >= 0 ? 'rgba(125,211,252,0.55)' : 'rgba(253,186,116,0.55)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // pump ramp
    ctx.beginPath();
    for (let s = 0; s < ns; s++) {
      const x = pad.l + (s / Math.max(ns - 1, 1)) * iw;
      const y = yOf(c.pHistory[s]);
      if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(167,139,250,0.9)';
    ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4; ctx.stroke(); ctx.setLineDash([]);

    // playhead
    const px = pad.l + playhead * iw;
    ctx.beginPath();
    ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ih);
    ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1; ctx.stroke();

    axisLabel(ctx, 'amplitudes x_i des DOPO', pad.l + 3, pad.t + 9, 'left', 'rgba(148,163,184,0.85)');
    axisLabel(ctx, 'pompe p(t)', pad.l + iw - 3, pad.t + 9, 'right', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, 'tours de cavité →', pad.l + iw, h - 4, 'right');
    axisLabel(ctx, '+1', pad.l - 5, yOf(1) + 3, 'right');
    axisLabel(ctx, '−1', pad.l - 5, yOf(-1) + 3, 'right');
  }, [bench, playhead], 0.42);

  // ----------------------------------------------------------- cut history
  const cutChart = useCanvas((ctx, w, h) => {
    const pad = { l: 34, r: 10, t: 12, b: 16 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4 || !bench) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 6, 3);
    const c = bench.cim;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < c.nSamples; i++) { mn = Math.min(mn, c.cutHistory[i]); mx = Math.max(mx, c.bestHistory[i]); }
    const lo = mn - (mx - mn) * 0.1, hi = bench.reference * 1.02;
    drawSeries(ctx, c.cutHistory, pad.l, pad.t, iw, ih, lo, hi, 'rgba(148,163,184,0.5)', undefined, 1);
    drawSeries(ctx, c.bestHistory, pad.l, pad.t, iw, ih, lo, hi, 'rgba(103,232,249,0.95)', 'rgba(103,232,249,0.10)', 1.6);

    // optimum line
    const yOpt = pad.t + ih - ((bench.reference - lo) / (hi - lo)) * ih;
    ctx.beginPath();
    ctx.moveTo(pad.l, yOpt); ctx.lineTo(pad.l + iw, yOpt);
    ctx.strokeStyle = bench.exact ? 'rgba(52,211,153,0.9)' : 'rgba(251,191,36,0.7)';
    ctx.setLineDash([5, 3]); ctx.lineWidth = 1.2; ctx.stroke(); ctx.setLineDash([]);
    axisLabel(ctx, bench.exact ? `optimum exact = ${bench.reference}` : `meilleur connu = ${bench.reference}`,
      pad.l + iw - 3, yOpt - 4, 'right', bench.exact ? 'rgba(52,211,153,0.95)' : 'rgba(251,191,36,0.9)');
    axisLabel(ctx, 'coupe instantanée / meilleure', pad.l + 3, pad.t + 9, 'left', 'rgba(103,232,249,0.9)');
  }, [bench], 0.32);

  const phys = useMemo(() => cimPhysicalTime(iterations, 1.0), [iterations]);

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_310px]">
      <div className="space-y-3">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
          <Panel tag="G" title="Graphe & configuration de spins" flush>
            <div className="w-full"><canvas ref={gview.ref} style={{ width: '100%', height: gview.size.h }} className="block" /></div>
          </Panel>

          <div className="space-y-3">
            <Panel
              tag="CIM"
              title="Bifurcation des oscillateurs paramétriques"
              subtitle="dx/dt = (p−1−x²)x − ξ·e·Σ J x + bruit — chaque courbe est un DOPO."
              right={
                <div className="flex gap-1.5">
                  <Btn active={playing} onClick={() => setPlaying((p) => !p)} disabled={!bench}>{playing ? '⏸' : '▶'}</Btn>
                </div>
              }
              flush
            >
              <div className="w-full"><canvas ref={traj.ref} style={{ width: '100%', height: traj.size.h }} className="block" /></div>
              {bench && (
                <div className="px-3 pb-2.5">
                  <input
                    type="range" min={0} max={1} step={0.002} value={playhead}
                    onChange={(e) => setPlayhead(Number(e.target.value))}
                    className="h-1 w-full cursor-pointer appearance-none rounded-full bg-white/10
                      [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none
                      [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
                  />
                </div>
              )}
            </Panel>

            <Panel tag="OBJ" title="Convergence vers la coupe maximale" flush>
              <div className="w-full"><canvas ref={cutChart.ref} style={{ width: '100%', height: cutChart.size.h }} className="block" /></div>
            </Panel>
          </div>
        </div>

        <Panel
          tag="BENCH"
          title="Comparaison de solveurs — mesurée, pas estimée"
          subtitle={graph.n <= BRUTE_FORCE_MAX_N
            ? `n ≤ ${BRUTE_FORCE_MAX_N} : l'optimum global est calculé par énumération exhaustive (code de Gray) et sert de vérité terrain.`
            : `n > ${BRUTE_FORCE_MAX_N} : la force brute est hors de portée, la référence est le meilleur résultat trouvé.`}
          right={
            <div className="flex gap-1.5">
              {bench && <Btn onClick={downloadSolution} title="télécharger la meilleure solution (CSV)">↓ spins</Btn>}
              <Btn tone="primary" onClick={solve} disabled={busy}>{busy ? '⋯ calcul' : '▶ Résoudre'}</Btn>
            </div>
          }
          flush
        >
          {!bench ? (
            <div className="px-4 py-8 text-center text-[11px] text-slate-500">
              Lancez une résolution pour comparer CIM, recuit simulé, greedy et l'optimum exact.
            </div>
          ) : (
            <table className="w-full text-left text-[11px]">
              <thead>
                <tr className="border-b border-white/6 text-[9px] uppercase tracking-wider text-slate-500">
                  <th className="px-3.5 py-2 font-medium">Méthode</th>
                  <th className="px-3 py-2 text-right font-medium">Coupe</th>
                  <th className="px-3 py-2 text-right font-medium">Ratio</th>
                  <th className="px-3 py-2 text-right font-medium">Temps</th>
                  <th className="hidden px-3 py-2 font-medium sm:table-cell">Détail</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {bench.rows.map(({ r, ratio }, i) => (
                  <tr key={r.method} className={i % 2 ? 'bg-white/[0.015]' : ''}>
                    <td className="px-3.5 py-2 text-slate-200">
                      <span className="flex items-center gap-1.5">
                        {r.method}
                        {r.exact && <Tag tone="green">exact</Tag>}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-cyan-300">{r.cut}</td>
                    <td className={`px-3 py-2 text-right tabular-nums ${ratio >= 0.999 ? 'text-emerald-300' : ratio >= 0.98 ? 'text-amber-300' : 'text-rose-300'}`}>
                      {(ratio * 100).toFixed(2)}%
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-400">{r.ms.toFixed(1)} ms</td>
                    <td className="hidden px-3 py-2 text-[10px] text-slate-500 sm:table-cell">{r.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        {bench && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <Stat label="Ratio CIM" value={`${(bench.rows.find((x) => x.r.method.startsWith('CIM'))!.ratio * 100).toFixed(2)}%`}
                tone={bench.rows.find((x) => x.r.method.startsWith('CIM'))!.ratio >= 0.999 ? 'green' : 'amber'}
                hint={bench.exact ? 'vs optimum exact' : 'vs meilleur trouvé'} />
              <Stat label="Taux de succès" value={`${(bench.cim.successRate * 100).toFixed(0)}%`}
                tone={bench.cim.successRate >= 0.5 ? 'green' : 'amber'}
                hint={`${bench.cim.trials} cycles de pompe`} />
              <Stat label="R99" value={Math.round(bench.cim.r99).toLocaleString('fr-FR')} unit="tours" tone="violet"
                hint="ln(0,01)/ln(1−p)" />
              <Stat label="Tours jusqu'à l'optimum" value={bench.cim.timeToBest} tone="cyan" hint="dans le meilleur cycle" />
              <Stat label="λ_max(|J|)" value={bench.cim.spectralScale.toFixed(2)} tone="slate"
                hint="normalisation du couplage" />
              <Stat label="Temps physique R99" value={((phys.perTripNs * bench.cim.r99) / 1000).toFixed(1)} unit="µs" tone="amber"
                hint="cavité fibrée de 1 m" />
            </div>

            <Panel tag="TTS" title="Distribution des cycles de pompe"
              subtitle="Un cycle de CIM est stochastique. La performance se lit sur la distribution, pas sur un tir unique — d'où la métrique R99 = ln(0,01)/ln(1−p_succès).">
              <div className="flex flex-wrap gap-1.5">
                {Array.from(bench.cim.trialCuts).map((c, i) => (
                  <div key={i}
                    className={`rounded px-2 py-1 font-mono text-[10px] tabular-nums ring-1 ${
                      c >= bench.reference
                        ? 'bg-emerald-400/12 text-emerald-300 ring-emerald-400/25'
                        : 'bg-white/[0.03] text-slate-400 ring-white/8'}`}>
                    #{i + 1} · {c}
                  </div>
                ))}
              </div>
            </Panel>
          </>
        )}
      </div>

        {/* controls */}
      <div className="space-y-3">
        <Panel tag="GRAPH" title="Instance" subtitle={kind === 'custom' ? 'votre matrice — symétrisée, diagonale ignorée' : GRAPH_INFO[kind]}>
          <div className="mb-3 grid grid-cols-3 gap-1">
            {KINDS.map((k) => (
              <Btn key={k} active={kind === k} onClick={() => setKind(k)} className="!text-[9px]">
                {k === 'erdos' ? 'Erdős' : k === 'cubic' ? '3-rég' : k === 'torus' ? 'Tore' : k === 'sk' ? 'SK' : 'Scale'}
              </Btn>
            ))}
            <Btn active={kind === 'custom'} onClick={() => setKind('custom')} className="!text-[9px]">★ Perso</Btn>
          </div>
          {kind === 'custom' && (
            <div className="mb-3">
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                spellCheck={false}
                placeholder={'Matrice de couplage n×n (CSV, espaces ou ;)\nex.\n0 1 -1\n1 0 1\n-1 1 0'}
                className="h-28 w-full resize-y rounded-lg border border-white/8 bg-black/40 p-2.5 font-mono text-[10px] leading-relaxed text-cyan-100/90 outline-none focus:border-cyan-400/30"
              />
              <div className="mt-1 flex items-center gap-2 text-[9px] text-slate-500">
                {parsed ? (
                  <Tag tone="green">{`valide · ${parsed.n} sommets, ${parsed.m} arêtes`}</Tag>
                ) : customText.trim() ? (
                  <Tag tone="rose">matrice invalide</Tag>
                ) : (
                  <span>collez une matrice carrée ci-dessus</span>
                )}
              </div>
            </div>
          )}
          <Slider label="Sommets n" value={n} min={8} max={400} step={2} onChange={setN} format={(v) => `${v}`} />
          <div className="mt-3 flex gap-1.5">
            <Btn onClick={() => setSeed((s) => s + 1)} className="flex-1">↻ Nouvelle instance</Btn>
          </div>
          {graph.n <= BRUTE_FORCE_MAX_N ? (
            <div className="mt-2 rounded-md bg-emerald-400/8 px-2 py-1.5 text-[9px] leading-relaxed text-emerald-300/90 ring-1 ring-emerald-400/20">
              Optimum exact calculable : {(2 ** (graph.n - 1)).toLocaleString('fr-FR')} configurations à énumérer
              pour {graph.n} sommets.
            </div>
          ) : (
            <div className="mt-2 rounded-md bg-amber-400/8 px-2 py-1.5 text-[9px] leading-relaxed text-amber-300/90 ring-1 ring-amber-400/20">
              2^{graph.n - 1} configurations — énumération impossible. Référence = meilleur résultat obtenu.
            </div>
          )}
        </Panel>

        <Panel tag="DOPO" title="Paramètres de la cavité">
          <div className="space-y-3">
            <Slider label="Tours de cavité" value={iterations} min={200} max={4000} step={100} onChange={setIterations} format={(v) => `${v}`} />
            <Slider label="Cycles de pompe (essais)" value={trials} min={1} max={16} step={1} onChange={setTrials} format={(v) => `${v}`} />
            <Slider label="Pompe finale p₁" value={pump} min={0.4} max={2} step={0.02} onChange={setPump} format={(v) => v.toFixed(2)} />
            <Slider label="Couplage ξ" value={xi} min={0.05} max={2} step={0.01} onChange={setXi} format={(v) => v.toFixed(2)} />
            <Slider label="Bruit quantique" value={noise} min={0} max={0.15} step={0.002} onChange={setNoise} format={(v) => v.toFixed(3)} />
          </div>
          <div className="mt-3">
            <Btn active={ahc} onClick={() => setAhc((a) => !a)} className="w-full">
              AHC (correction d'amplitude) : {ahc ? 'ON' : 'OFF'}
            </Btn>
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            L'AHC ajoute une variable d'erreur <Formula>de/dt = −β e (x² − τ)</Formula> qui égalise
            les amplitudes et supprime les minima parasites induits par l'hétérogénéité.
          </p>
        </Panel>

        <Panel tag="THÉORIE" title="Pourquoi ça marche">
          <p className="text-[10px] leading-relaxed text-slate-400">
            Sous le seuil, tous les DOPO sont au vide. En montant la pompe, le réseau bifurque
            <span className="text-slate-200"> collectivement</span> vers la configuration de phases qui
            minimise les pertes — c'est-à-dire le hamiltonien d'Ising
            <Formula>H = Σ w_ij s_i s_j</Formula>.
          </p>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-400">
            Comme <Formula>coupe = (W − H)/2</Formula>, minimiser H revient exactement à maximiser la coupe.
            Cette identité est vérifiée numériquement dans l'onglet Validation.
          </p>
        </Panel>
      </div>
    </div>
  );
}
