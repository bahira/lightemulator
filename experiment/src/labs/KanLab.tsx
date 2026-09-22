import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import {
  KAN, makeDataset, trainEpochs, renderMaps, TARGETS, gradientCheck,
  type TargetId, type TrainStats,
} from '../physics/kan';
import {
  Panel, Stat, Slider, Segmented, Btn, Tag, Formula,
  useCanvas, LUT, paintField, drawImageBuffer, drawGrid, axisLabel,
} from '../ui/kit';

export default function KanLab() {
  const [target, setTarget] = useState<TargetId>('kanClassic');
  const [hidden, setHidden] = useState(5);
  const [grid, setGrid] = useState(6);
  const [lr, setLr] = useState(0.02);
  const [running, setRunning] = useState(false);
  const [version, setVersion] = useState(0);
  const [stats, setStats] = useState<TrainStats | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [epochs, setEpochs] = useState(0);
  const [gcheck, setGcheck] = useState<{ relErr: number; checks: number } | null>(null);

  const train = useMemo(() => makeDataset(target, 3000, 17), [target]);
  const test = useMemo(() => makeDataset(target, 900, 991), [target]);

  const netRef = useRef<KAN | null>(null);
  const epochRef = useRef(0);
  const lrRef = useRef(lr);
  lrRef.current = lr;

  const reset = useCallback(() => {
    // architecture or target changed -> the problem changed, restart cleanly.
    // The learning rate is applied live instead, so tuning it never wipes progress.
    netRef.current = new KAN([2, hidden, 1], grid, 3);
    netRef.current.lr = lrRef.current;
    epochRef.current = 0;
    setHistory([]); setEpochs(0); setStats(null); setVersion((v) => v + 1);
  }, [hidden, grid, target]);

  useEffect(() => { reset(); }, [reset]);
  useEffect(() => { if (netRef.current) netRef.current.lr = lr; }, [lr]);

  useEffect(() => {
    if (!running) return;
    let id = 0;
    let frame = 0;
    const loop = () => {
      const net = netRef.current;
      if (net) {
        const s = trainEpochs(net, train, test, 25, 48, 1000 + epochRef.current);
        epochRef.current += 25;
        setStats(s);
        setEpochs(epochRef.current);
        setHistory((h) => [...h.slice(-400), s.testMse]);
        // the prediction maps are 72x72 forward passes — refresh them at a
        // lower rate than the training loop so training stays the bottleneck
        if (frame % 3 === 0) setVersion((v) => v + 1);
        frame++;
      }
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [running, train, test]);

  const maps = useMemo(() => {
    const net = netRef.current;
    if (!net) return null;
    return renderMaps(net, target, 72);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, target]);

  // ---------------------------------------------------------------- heatmaps
  const heat = useCanvas((ctx, w, h) => {
    if (!maps) return;
    const gapx = 10;
    const cw = (w - gapx * 2) / 3;
    const ch = Math.min(cw, h - 16);
    const y0 = 14;
    const panels: [string, Float32Array, Uint8Array, number, number][] = [
      ['prédiction KAN', maps.pred, LUT.diverging, -maps.maxAbs, maps.maxAbs],
      ['cible analytique', maps.truth, LUT.diverging, -maps.maxAbs, maps.maxAbs],
      ['|erreur|', maps.err, LUT.inferno, 0, maps.maxErr],
    ];
    panels.forEach(([label, data, lut, lo, hi], i) => {
      const img = ctx.createImageData(maps.res, maps.res);
      paintField(img, data, maps.res, maps.res, lut, lo, hi, 1, false);
      const x = i * (cw + gapx);
      drawImageBuffer(ctx, img, x, y0, cw, ch, true);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.strokeRect(x + 0.5, y0 + 0.5, cw - 1, ch - 1);
      axisLabel(ctx, label, x, 10, 'left');
    });
    axisLabel(ctx, `max|err| = ${maps.maxErr.toFixed(4)}`, w, 10, 'right', 'rgba(251,146,60,0.9)');
  }, [maps], 0.36);

  // ------------------------------------------------------------- loss curve
  const loss = useCanvas((ctx, w, h) => {
    const pad = { l: 42, r: 10, t: 12, b: 18 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 6, 4);
    if (history.length < 2) {
      axisLabel(ctx, 'lancez l’entraînement', pad.l + iw / 2, pad.t + ih / 2, 'center');
      return;
    }
    let mn = Infinity, mx = -Infinity;
    for (const v of history) { const l = Math.log10(Math.max(v, 1e-10)); mn = Math.min(mn, l); mx = Math.max(mx, l); }
    const span = Math.max(mx - mn, 0.5);
    const lo = mn - span * 0.08, hi = mx + span * 0.08;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = pad.l + (i / (history.length - 1)) * iw;
      const y = pad.t + ih - ((Math.log10(Math.max(history[i], 1e-10)) - lo) / (hi - lo)) * ih;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(167,139,250,0.95)';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    for (let t = Math.ceil(lo); t <= Math.floor(hi); t++) {
      const y = pad.t + ih - ((t - lo) / (hi - lo)) * ih;
      axisLabel(ctx, `1e${t}`, pad.l - 5, y + 3, 'right');
    }
    axisLabel(ctx, 'MSE test (échelle log)', pad.l + 3, pad.t + 9, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, `${epochs} epochs`, pad.l + iw, h - 4, 'right');
  }, [history, epochs], 0.34);

  // -------------------------------------------------------- edge functions
  const edges = useCanvas((ctx, w, h) => {
    const net = netRef.current;
    if (!net) return;
    const layer = net.layers[0];
    const cols = layer.nOut;
    const rows = layer.nIn;
    const gx = 6, gy = 14;
    const cw = (w - gx * (cols - 1)) / cols;
    const chh = (h - gy * rows - 8) / rows;
    if (cw <= 4 || chh <= 4) return;
    const S = 64;
    const buf = new Float64Array(S);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        layer.sampleEdge(i, j, S, buf);
        let m = 1e-6;
        for (let s = 0; s < S; s++) m = Math.max(m, Math.abs(buf[s]));
        const x0 = j * (cw + gx);
        const y0 = 8 + i * (chh + gy);
        ctx.fillStyle = 'rgba(255,255,255,0.02)';
        ctx.fillRect(x0, y0, cw, chh);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, cw - 1, chh - 1);
        ctx.beginPath();
        ctx.moveTo(x0, y0 + chh / 2); ctx.lineTo(x0 + cw, y0 + chh / 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.stroke();
        ctx.beginPath();
        for (let s = 0; s < S; s++) {
          const x = x0 + (s / (S - 1)) * cw;
          const y = y0 + chh / 2 - (buf[s] / m) * (chh / 2 - 3);
          if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        const mag = Math.min(1, m);
        ctx.strokeStyle = `hsla(${185 + mag * 80}, 90%, ${58 + mag * 12}%, 0.95)`;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        axisLabel(ctx, `φ${i}${j}`, x0 + 2, y0 + chh + 10, 'left', 'rgba(100,116,139,0.8)');
      }
    }
  }, [version, hidden, grid], 0.55);

  const net = netRef.current;

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_310px]">
      <div className="space-y-3">
        <Panel
          tag="KAN"
          title="Régression sur fonction analytique"
          subtitle={`Cible : ${TARGETS[target].tex} — les cartes sont recalculées à chaque rafraîchissement.`}
          right={
            <div className="flex gap-1.5">
              <Btn tone="primary" active={running} onClick={() => setRunning((r) => !r)}>
                {running ? '⏸ Pause' : '▶ Entraîner'}
              </Btn>
              <Btn onClick={reset}>↻ Reset</Btn>
            </div>
          }
          flush
        >
          <div className="w-full px-3.5 pb-2"><canvas ref={heat.ref} style={{ width: '100%', height: heat.size.h }} className="block" /></div>
        </Panel>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Stat label="MSE test" value={stats ? stats.testMse.toExponential(2) : '—'} tone="violet" />
          <Stat label="R²" value={stats ? stats.r2.toFixed(5) : '—'} tone={stats && stats.r2 > 0.99 ? 'green' : 'amber'} />
          <Stat label="Epochs" value={epochs} tone="cyan" />
          <Stat label="Paramètres" value={net ? net.paramCount() : 0} tone="slate" hint={`[2, ${hidden}, 1] · grille ${grid}`} />
          <Stat label="Débit" value={stats ? (stats.samplesPerSecond / 1000).toFixed(1) : '—'} unit="k éch/s" tone="amber" />
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <Panel tag="LOSS" title="Courbe d'apprentissage" flush>
            <div className="w-full"><canvas ref={loss.ref} style={{ width: '100%', height: loss.size.h }} className="block" /></div>
          </Panel>
          <Panel
            tag="φ" title="Fonctions d'arête apprises"
            subtitle="La signature du KAN : la nonlinéarité vit sur les arêtes, pas sur les nœuds."
            flush
          >
            <div className="w-full px-3 pb-2"><canvas ref={edges.ref} style={{ width: '100%', height: edges.size.h }} className="block" /></div>
          </Panel>
        </div>

        <Panel tag="OPTIQUE" title="Réalisation photonique de la couche KAN">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {[
              ['Base B-spline → bancs de micro-anneaux', 'Chaque fonction de base B_k(x) est un résonateur en anneau désaccordé sur son propre canal WDM. Les nb fonctions sont évaluées en parallèle sur nb longueurs d’onde.'],
              ['Coefficients c_ijk → transmissions', 'Les coefficients appris deviennent les transmissions du port drop de chaque anneau, ajustables par effet thermo-optique.'],
              ['Somme Σ_k → combineur passif', 'La sommation sur k est réalisée par un simple multiplexeur passif : zéro énergie, zéro latence supplémentaire.'],
            ].map(([t, d]) => (
              <div key={t} className="rounded-lg border border-white/6 bg-white/[0.015] p-2.5">
                <div className="text-[10px] font-semibold text-cyan-300">{t}</div>
                <p className="mt-1 text-[10px] leading-relaxed text-slate-500">{d}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[10px] leading-relaxed text-slate-500">
            Conséquence : toute la couche s'évalue en <span className="text-slate-300">un seul passage optique</span>.
            Le KAN est particulièrement adapté au photonique car sa largeur est faible mais sa richesse
            fonctionnelle est portée par les splines — exactement ce qu'un banc d'anneaux sait faire nativement.
          </p>
        </Panel>
      </div>

      {/* controls */}
      <div className="space-y-3">
        <Panel tag="TARGET" title="Fonction à approcher">
          <Segmented
            className="mb-2"
            value={target}
            onChange={setTarget}
            options={(Object.keys(TARGETS) as TargetId[]).map((k) => ({ id: k, label: TARGETS[k].name.split(' ')[0] }))}
          />
          <div className="rounded-md bg-black/40 px-2 py-1.5 text-center font-mono text-[10px] text-cyan-200/85 ring-1 ring-white/6">
            {TARGETS[target].tex}
          </div>
        </Panel>

        <Panel tag="ARCH" title="Architecture & optimisation">
          <div className="space-y-3">
            <Slider label="Neurones cachés" value={hidden} min={2} max={12} step={1} onChange={setHidden} format={(v) => `${v}`} />
            <Slider label="Intervalles de grille" value={grid} min={3} max={14} step={1} onChange={setGrid} format={(v) => `${v}`} />
            <Slider label="Taux d'apprentissage" value={lr} min={0.002} max={0.08} step={0.002} onChange={setLr} format={(v) => v.toFixed(3)} />
          </div>
          <div className="mt-3 text-[10px] leading-relaxed text-slate-500">
            <div><Formula>φ(x) = w_b·silu(x) + w_s·Σ_k c_k B_k(x)</Formula></div>
            <div className="mt-1.5">B-splines cubiques par récurrence de Cox-de Boor, dérivées analytiques,
              optimiseur Adam. Base de {grid + 3} fonctions par arête.</div>
          </div>
        </Panel>

        <Panel tag="VÉRIF" title="Contrôle du gradient"
          subtitle="La rétropropagation est-elle vraiment correcte ?">
          <Btn tone="primary" className="w-full" onClick={() => setGcheck(gradientCheck(Date.now() & 0xffff, 30))}>
            ▶ Lancer les différences finies
          </Btn>
          {gcheck && (
            <div className="mt-2.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-400">Erreur relative max</span>
                <Tag tone={gcheck.relErr < 1e-4 ? 'green' : 'rose'}>{gcheck.relErr.toExponential(2)}</Tag>
              </div>
              <p className="mt-1.5 text-[10px] leading-relaxed text-slate-500">
                {gcheck.checks} paramètres testés contre <Formula>(L(θ+h) − L(θ−h)) / 2h</Formula> avec h = 10⁻⁶.
                {gcheck.relErr < 1e-4 ? ' Le gradient analytique est confirmé.' : ' Écart anormal détecté.'}
              </p>
            </div>
          )}
        </Panel>

        <Panel tag="LIMITE" title="Ce que ça ne fait pas">
          <p className="text-[10px] leading-relaxed text-slate-400">
            Un KAN photonique de cette taille excelle sur des fonctions lisses de faible dimension —
            contrôle, égalisation de canal, capteurs. Il ne remplace pas un transformer :
            la mémoire, l'attention et la profondeur restent électroniques aujourd'hui.
          </p>
          <div className="mt-2 flex flex-wrap gap-1">
            <Tag tone="green">edge temps-réel</Tag>
            <Tag tone="green">faible dimension</Tag>
            <Tag tone="rose">pas de transformer</Tag>
          </div>
        </Panel>
      </div>
    </div>
  );
}
