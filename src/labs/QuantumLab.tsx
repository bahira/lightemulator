import { useMemo, useState } from 'react';

import {
  bsT, bsR, homP11, mziDarkFromMatrix,
  chshE, chshHiddenVariables, lcg,
} from '../physics/quantum';
import {
  Panel, Stat, Tag, Slider, Formula, Btn,
  useCanvas, drawGrid, drawSeries, axisLabel,
} from '../ui/kit';

// ---------------------------------------------------------------------------
//  Dip HOM — courbe fermée + simulation de comptage honnête (Bernoulli semée)
// ---------------------------------------------------------------------------

function HomPanel() {
  const [R, setR] = useState(0.5);
  const [x2, setX2] = useState(1);

  const curve = useMemo(() => {
    const N = 240;
    const P: number[] = [];
    for (let i = 0; i < N; i++) {
      const tau = (i / (N - 1)) * 3;
      // P = T²+R² − 2TR·x²·e^{−τ²}  (l'enveloppe Gaussienne du recouvrement)
      P.push((1 - R) ** 2 + R * R - 2 * (1 - R) * R * x2 * Math.exp(-tau * tau));
    }
    return P;
  }, [R, x2]);

  // comptage : 27 retards × 600 paires, Bernoulli semée (LCG) — du bruit réel,
  // l'erreur d'échantillonnage est visible, comme sur une vraie Table optique.
  const counts = useMemo(() => {
    const rnd = lcg(1234);
    const nPairs = 600;
    const pts: { tau: number; p: number }[] = [];
    for (let j = 0; j < 27; j++) {
      const tau = (j / 26) * 3;
      const p = (1 - R) ** 2 + R * R - 2 * (1 - R) * R * x2 * Math.exp(-tau * tau);
      let c = 0;
      for (let i = 0; i < nPairs; i++) if (rnd() < p) c++;
      pts.push({ tau, p: c / nPairs });
    }
    return pts;
  }, [R, x2]);

  const pDip = homP11(R, x2);
  const pFar = (1 - R) ** 2 + R * R;
  const vis = pFar > 0 ? (pFar - pDip) / pFar : 0;

  const { ref } = useCanvas((ctx, w, h) => {
    const pad = { l: 30, r: 10, t: 10, b: 16 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    drawGrid(ctx, pad.l, pad.t, pw, ph, 6, 4);
    drawSeries(ctx, curve, pad.l, pad.t, pw, ph, 0, 1.02, 'rgba(34,211,238,0.95)', 'rgba(34,211,238,0.06)', 1.6);
    // points de comptage simulés
    ctx.fillStyle = 'rgba(251,191,36,0.95)';
    for (const c of counts) {
      const X = pad.l + (c.tau / 3) * pw;
      const Y = pad.t + ph - (c.p / 1.02) * ph;
      ctx.beginPath(); ctx.arc(X, Y, 2.2, 0, 7); ctx.fill();
    }
    // ligne zéro du dip
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
    axisLabel(ctx, `P(1,1) min = ${pDip.toFixed(3)}`, pad.l + 6, pad.t + 12, 'left', 'rgba(52,211,153,0.95)');
    axisLabel(ctx, 'τ = 0', pad.l, h - 4);
    axisLabel(ctx, 'retard τ/τc → 3', pad.l + pw - 96, h - 4);
  }, [curve, counts, R, x2]);

  return (
    <Panel
      tag="HOM"
      title="Dip de Hong–Ou–Mandel — deux photons indiscernables ne se séparent jamais"
      subtitle="Entrée |1,1⟩ sur un séparateur : P(1,1) = T² + R² − 2TR·x². À 50/50 et x²=1, le dip touche zéro exactement (coalescence)."
      right={<Tag tone={vis > 0.95 ? 'green' : 'amber'}>visibilité {(vis * 100).toFixed(0)}%</Tag>}
    >
      <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div>
          <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2.2 / 1' }} />
          <p className="mt-1.5 text-[10px] leading-relaxed text-slate-500">
            Cyan : probabilité fermée (algèbre de Fock exacte) · Ambre : comptage simulé, 600 paires/point,
            bruit de Poisson visible — <em>volontairement</em>, parce qu'un comptage sans bruit est un mensonge.
          </p>
        </div>
        <div className="space-y-3">
          <Slider label="Réflectivité R" min={0.1} max={0.9} step={0.01} value={R} onChange={setR} format={(v) => (v * 100).toFixed(0)} unit=" %" />
          <Slider label="Indiscernabilité x²" min={0} max={1} step={0.01} value={x2} onChange={setX2} format={(v) => (v * 100).toFixed(0)} unit=" %" />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="P dip (τ=0)" value={pDip.toFixed(3)} tone={(R === 0.5 && x2 === 1) ? 'green' : 'cyan'} hint="= (T−R)² si x²=1" />
            <Stat label="P loin" value={pFar.toFixed(3)} tone="slate" hint="photons distinguables" />
          </div>
          <p className="text-[10px] leading-relaxed text-slate-500">
            T = 1 − R = {(1 - R).toFixed(2)} · t = √T = {bsT(R).toFixed(3)} · r = i√R = {bsR(R).toFixed(3)}i.
            Hors 50/50, le dip ne ferme pas —{' '}
            <Formula>P_min = (T−R)²</Formula> — et la visibilité plafonne à <Formula>2TR/(T²+R²)</Formula> ={' '}
            <span className="text-cyan-300">{((2 * (1 - R) * R) / ((1 - R) ** 2 + R * R)).toFixed(3)}</span>.
          </p>
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  MZI photon unique
// ---------------------------------------------------------------------------

function MziPanel() {
  const [phiDeg, setPhiDeg] = useState(90);
  const phi = (phiDeg * Math.PI) / 180;

  const curve = useMemo(() => {
    const N = 240;
    const P: number[] = [];
    for (let i = 0; i < N; i++) P.push(mziDarkFromMatrix((i / (N - 1)) * 2 * Math.PI));
    return P;
  }, []);

  const pMeas = mziDarkFromMatrix(phi);
  const pClosed = Math.sin(phi / 2) ** 2;

  const { ref } = useCanvas((ctx, w, h) => {
    const pad = { l: 30, r: 10, t: 10, b: 16 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    drawGrid(ctx, pad.l, pad.t, pw, ph, 6, 4);
    drawSeries(ctx, curve, pad.l, pad.t, pw, ph, 0, 1.05, 'rgba(167,139,250,0.95)', 'rgba(167,139,250,0.06)', 1.6);
    const X = pad.l + (phiDeg / 360) * pw;
    const Y = pad.t + ph - (pMeas / 1.05) * ph;
    ctx.fillStyle = '#22d3ee';
    ctx.beginPath(); ctx.arc(X, Y, 3, 0, 7); ctx.fill();
    ctx.strokeStyle = 'rgba(34,211,238,0.4)';
    ctx.beginPath(); ctx.moveTo(X, pad.t); ctx.lineTo(X, pad.t + ph); ctx.stroke();
    axisLabel(ctx, 'P(port sombre)', pad.l + 6, pad.t + 12, 'left', 'rgba(167,139,250,0.95)');
    axisLabel(ctx, 'φ = 0', pad.l, h - 4);
    axisLabel(ctx, 'φ = 2π', pad.l + pw - 40, h - 4);
  }, [curve, phiDeg]);

  return (
    <Panel
      tag="MZI"
      title="Un photon unique dans l'interféromètre"
      subtitle="Le photon interfère avec lui-même : produit matriciel BS·phase·BS calculé explicitement, comparé à sin²(φ/2)."
      right={<Tag tone="cyan">P = sin²(φ/2)</Tag>}
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '3 / 1' }} />
      <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <Slider label="Phase φ" min={0} max={360} step={1} value={phiDeg} onChange={setPhiDeg} unit=" °" />
        <div className="grid grid-cols-3 gap-2">
          <Stat label="P matrice" value={pMeas.toFixed(4)} tone="violet" hint="BS·e^{iφ}·BS explicite" />
          <Stat label="P fermé" value={pClosed.toFixed(4)} tone="cyan" hint="sin²(φ/2)" />
          <Stat label="Écart" value={(Math.abs(pMeas - pClosed) * 1e15).toFixed(0)} unit="e-15" tone="green" hint="epsilon machine" />
        </div>
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
        Honnêteté : pour UN photon, cette courbe est identique à l'intensité d'une onde classique —
        la mécanique quantique ne se révèle ici qu'avec DEUX photons (dip HOM ci-dessus) ou des états intriqués (CHSH ci-dessous).
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  CHSH — violation de Bell mesurée en direct
// ---------------------------------------------------------------------------

function ChshPanel() {
  const [a, setA] = useState(0);
  const [ap, setAp] = useState(45);
  const [b, setB] = useState(22.5);
  const [bp, setBp] = useState(67.5);
  const d2r = Math.PI / 180;

  const qm = useMemo(() => {
    const E = [chshE(a * d2r, b * d2r), chshE(a * d2r, bp * d2r), chshE(ap * d2r, b * d2r), chshE(ap * d2r, bp * d2r)];
    return { E, S: Math.abs(E[0] - E[1] + E[2] + E[3]) };
  }, [a, ap, b, bp]);

  const hv = useMemo(() => chshHiddenVariables(a * d2r, ap * d2r, b * d2r, bp * d2r, 100_000, 7), [a, ap, b, bp]);

  const SQ2 = 2 * Math.SQRT2;
  const violating = qm.S > 2.02; // marge > bruit HV (~1 %)

  const { ref } = useCanvas((ctx, w, h) => {
    const pad = { l: 30, r: 10, t: 10, b: 16 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    drawGrid(ctx, pad.l, pad.t, pw, ph, 6, 4);
    // E(Δ) = cos 2Δ — la courbe que la physique quantique prédit
    const N = 240;
    const c: number[] = [];
    for (let i = 0; i < N; i++) c.push(chshE((i / (N - 1)) * Math.PI, 0));
    drawSeries(ctx, c, pad.l, pad.t, pw, ph, -1, 1, 'rgba(167,139,250,0.85)', undefined, 1.4);
    // les 4 points mesurés (en fonction de Δ = |θ1−θ2|)
    const pts = [[a, b, 0], [a, bp, 1], [ap, b, 2], [ap, bp, 3]] as const;
    for (const [t1, t2, idx] of pts) {
      const X = pad.l + (Math.abs(t1 - t2) / 180) * pw;
      const Y = pad.t + ph - ((qm.E[idx] + 1) / 2) * ph;
      ctx.fillStyle = 'rgba(34,211,238,0.95)';
      ctx.beginPath(); ctx.arc(X, Y, 3, 0, 7); ctx.fill();
    }
    // zéro
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    const y0 = pad.t + ph / 2;
    ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + pw, y0); ctx.stroke();
    axisLabel(ctx, 'E(Δ) = cos 2Δ', pad.l + 6, pad.t + 12, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, 'Δ = 0', pad.l, h - 4);
    axisLabel(ctx, 'Δ = 180°', pad.l + pw - 52, h - 4);
  }, [qm, a, ap, b, bp]);

  // jauge S
  const { ref: gaugeRef } = useCanvas((ctx, w, h) => {
    const x0 = 8, pw = w - 16, cy = h * 0.42, barH = 10;
    const toX = (s: number) => x0 + (s / 3) * pw;
    // zone locale [0,2]
    ctx.fillStyle = 'rgba(148,163,184,0.10)';
    ctx.fillRect(x0, cy - barH / 2, toX(2) - x0, barH);
    ctx.fillStyle = 'rgba(52,211,153,0.10)';
    ctx.fillRect(toX(2), cy - barH / 2, toX(3) - toX(2), barH);
    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
    ctx.strokeRect(x0, cy - barH / 2, pw, barH);
    // bornes
    const tick = (s: number, color: string, label: string, sub: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(toX(s), cy - barH / 2 - 5); ctx.lineTo(toX(s), cy + barH / 2 + 5); ctx.stroke();
      axisLabel(ctx, label, toX(s), cy - barH / 2 - 8, 'center', color);
      axisLabel(ctx, sub, toX(s), cy + barH / 2 + 13, 'center', color);
    };
    tick(2, 'rgba(251,191,36,0.9)', 'Bell · 2', 'local');
    tick(SQ2, 'rgba(52,211,153,0.9)', 'Tsirelson · 2√2', 'max quantique');
    // curseurs
    const needle = (s: number, color: string) => {
      const X = toX(Math.max(0, Math.min(3, s)));
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(X, cy - barH / 2 - 12);
      ctx.lineTo(X - 4, cy - barH / 2 - 19);
      ctx.lineTo(X + 4, cy - barH / 2 - 19);
      ctx.closePath(); ctx.fill();
      ctx.fillRect(X - 1, cy - barH / 2 - 12, 2, barH + 12 + 2);
    };
    needle(qm.S, '#a78bfa');
    needle(hv.S, '#fbbf24');
    axisLabel(ctx, `QM ${qm.S.toFixed(4)}`, toX(Math.max(0.15, Math.min(2.9, qm.S))), cy + barH / 2 + 26, 'center', '#a78bfa');
    axisLabel(ctx, `HV ${hv.S.toFixed(3)}`, toX(Math.max(0.15, Math.min(2.9, hv.S))), h - 4, 'center', '#fbbf24');
  }, [qm, hv]);

  return (
    <Panel
      tag="CHSH"
      title="Violation de Bell — la nature refuse les variables cachées locales"
      subtitle="S = |E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′)|. Toute théorie locale plafonne à 2 ; la mécanique quantique atteint 2√2."
      right={<Tag tone={violating ? 'green' : 'rose'}>{violating ? `VIOLATION · +${((qm.S / 2 - 1) * 100).toFixed(1)}%` : 'pas de violation'}</Tag>}
    >
      <canvas ref={gaugeRef} className="block w-full" style={{ aspectRatio: '3.4 / 1' }} />
      <div className="mt-2 grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div>
          <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2.4 / 1' }} />
          <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
            Courbe : corrélation quantique E(Δ) = cos 2Δ pour |Φ+⟩. Points cyan : vos 4 réglages.
            Le max de S est atteint à 22,5° / 67,5° — bougez les curseurs et regardez l'aiguille violette descendre sous 2.
          </p>
        </div>
        <div className="space-y-2.5">
          <div className="grid grid-cols-2 gap-2.5">
            <Slider label="a" min={0} max={180} step={0.5} value={a} onChange={setA} unit=" °" />
            <Slider label="a′" min={0} max={180} step={0.5} value={ap} onChange={setAp} unit=" °" />
            <Slider label="b" min={0} max={180} step={0.5} value={b} onChange={setB} unit=" °" />
            <Slider label="b′" min={0} max={180} step={0.5} value={bp} onChange={setBp} unit=" °" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="S quantique" value={qm.S.toFixed(4)} tone="violet" hint="fermé, exact" />
            <Stat label="S variables cachées" value={hv.S.toFixed(4)} tone="amber" hint="100k paires simulées" />
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Btn onClick={() => { setA(0); setAp(45); setB(22.5); setBp(67.5); }}>réglages optimaux</Btn>
            <Btn onClick={() => { setA(0); setAp(90); setB(0); setBp(0); }}>contre-exemple local</Btn>
          </div>
        </div>
      </div>
    </Panel>
  );
}

export default function QuantumLab() {
  return (
    <div className="space-y-3">
      <HomPanel />
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <MziPanel />
        <ChshPanel />
      </div>
      <Panel flush={false} className="border-white/5">
        <p className="text-[10px] leading-relaxed text-slate-500">
          <span className="font-semibold text-slate-400">Ce qui est simulé, exactement.</span> Les probabilités
          quantiques viennent d'algèbre linéaire fermée sur les états de Fock (amplitudes complexes, normalisation
          vérifiée Σ P = 1 à 1e-12) — aucun Monte-Carlo caché ne décide de la physique. Les seules simulations
          stochastiques sont déclarées comme telles : le comptage HOM (Bernoulli semée, LCG) et le modèle à variables
          cachées locales de Bell (polarisations prédéterminées λ, Malus, 100k paires). Les tests correspondants
          tournent dans l'onglet <span className="text-cyan-300">Validation</span> — groupe « Optique quantique ».
        </p>
      </Panel>
    </div>
  );
}
