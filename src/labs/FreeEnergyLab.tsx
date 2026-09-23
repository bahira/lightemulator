import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
  DEFAULT_FEP, makeMachine, relax, verifyPhaseGradient, trainFep,
  type FreeEnergyMachine, type FepConfig, type VerifyResult, type TrainResult,
} from '../physics/fep';
import {
  makeAttn, trainRouterEpoch, evalAttnLoss, evalOracle,
  attentionOf, makeAttnSamples, DEFAULT_ATTN, LR_SCHEDULE, initRouter,
  type AttnSample,
} from '../physics/fepAttn';
import {
  Panel, Stat, Slider, Btn, Tag, Formula, Segmented,
  useCanvas, drawGrid, drawSeries, axisLabel,
} from '../ui/kit';

interface LabConfig extends FepConfig {
  eta: number;
  epochs: number;
}

export default function FreeEnergyLab() {
  const [cfg, setCfg] = useState<LabConfig>({
    ...DEFAULT_FEP, eta: 0.004, epochs: 12,
  });
  const [running, setRunning] = useState(false);
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [train, setTrain] = useState<TrainResult | null>(null);
  const [lossHist, setLossHist] = useState<number[]>([]);
  const [epochsDone, setEpochsDone] = useState(0);
  const [live, setLive] = useState(true);
  const [version, setVersion] = useState(0);

  const machineRef = useRef<FreeEnergyMachine | null>(null);
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;

  const make = useCallback(() => {
    machineRef.current = makeMachine(cfgRef.current);
    setVersion((v) => v + 1);
  }, []);

  useEffect(() => { make(); }, [make]);
  // relaxation continue : rafraîchissement périodique du settle — PAS un
  // effet sur `version` lui-même (boucle de mise à jour infinie, cf. React).
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => setVersion((v) => v + 1), 450);
    return () => clearInterval(id);
  }, [live]);

  // background settle loop — keeps a live phasor picture
  useEffect(() => {
    if (!running) return;
    let id = 0;
    const loop = () => {
      const m = machineRef.current;
      if (m) {
        const inR = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
        for (let k = 0; k < m.nIn; k++) { inR[k] = 0.55 + 0.2 * Math.sin(Date.now() / 900 + k * 1.7); inI[k] = 0.1 * Math.sin(Date.now() / 500 + k); }
        relax(m, inR, inI, null, null, 0, 900, m.cfg.dt);
        setVersion((v) => v + 1);
      }
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [running]);

  const runVerify = useCallback(() => {
    const m = machineRef.current;
    if (!m) return;
    const inR = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
    inR[0] = 0.5; inR[1] = 0.7; inI[0] = 0.1; inI[1] = -0.05;
    const outR = new Float64Array(m.nOut), outI = new Float64Array(m.nOut);
    outR[0] = 0.9;
    setTimeout(() => {
      const v = verifyPhaseGradient(m, inR, inI, outR, outI, 12);
      setVerify(v);
      setVersion((x) => x + 1);
    }, 30);
  }, []);

  const runTrain = useCallback(() => {
    const m = machineRef.current;
    if (!m) return;
    setTrain(null); setLossHist([]); setEpochsDone(0);
    setRunning(true);
    setTimeout(() => {
      const res = trainFep(
        m,
        { epochs: cfgRef.current.epochs, eta: cfgRef.current.eta, decay: 1e-3, seriesSeed: 7, nSeries: 400, window: 180 },
        (ep, loss) => {
          setEpochsDone(ep);
          setLossHist((h) => [...h, loss]);
        },
      );
      setTrain(res);
      setRunning(false);
      setVersion((x) => x + 1);
    }, 30);
  }, []);

  const freeState = useMemo(() => {
    const m = machineRef.current;
    if (!m) return null;
    const inR = new Float64Array(m.nIn), inI = new Float64Array(m.nIn);
    for (let k = 0; k < m.nIn; k++) { inR[k] = 0.55 + 0.2 * Math.sin(k * 1.7); inI[k] = 0.1 * Math.sin(k); }
    relax(m, inR, inI, null, null, 0, 3000, m.cfg.dt);
    const amps: number[] = [];
    for (let i = m.nIn; i < m.n; i++) amps.push(Math.hypot(m.xr[i], m.xi[i]));
    return { amps, vfe: (() => {
      let E = 0;
      for (let i = 0; i < m.n; i++) {
        for (let j = 0; j < m.n; j++) {
          const kc = i * m.n + j;
          E -= 0.5 * (m.xr[i] * (m.Kr[kc] * m.xr[j] - m.Ki[kc] * m.xi[j]) + m.xi[i] * (m.Kr[kc] * m.xi[j] + m.Ki[kc] * m.xr[j]));
        }
        const a2 = m.xr[i] * m.xr[i] + m.xi[i] * m.xi[i];
        E += 0.25 * a2 * a2 + 0.5 * m.cfg.lam * a2;
      }
      return E;
    })() };
  }, [version]);

  // ------------------------------------------------------- phasor network
  const net = useCanvas((ctx, w, h) => {
    const m = machineRef.current;
    if (!m) return;
    ctx.clearRect(0, 0, w, h);
    const pad = 26;
    const xIn = pad, xOut = w - pad;
    const xHid = (xIn + xOut) / 2;
    const col = (n: number, x: number) => {
      const span = Math.max(n - 1, 1);
      return (i: number) => {
        const y = h * 0.18 + ((h * 0.64) * i) / span;
        return { x, y };
      };
    };
    const pos: { x: number; y: number }[] = [];
    for (let i = 0; i < m.nIn; i++) pos.push(col(m.nIn, xIn)(i));
    for (let i = 0; i < m.nHidden; i++) pos.push(col(m.nHidden, xHid)(i));
    for (let i = 0; i < m.nOut; i++) pos.push(col(m.nOut, xOut)(i));

    // edges
    let maxK = 1e-6;
    for (let i = 0; i < m.n; i++) for (let j = i + 1; j < m.n; j++) {
      const k = Math.hypot(m.Kr[i * m.n + j], m.Ki[i * m.n + j]);
      if (k > maxK) maxK = k;
    }
    for (let i = 0; i < m.n; i++) for (let j = i + 1; j < m.n; j++) {
      const k = Math.hypot(m.Kr[i * m.n + j], m.Ki[i * m.n + j]) / maxK;
      if (k < 0.02) continue;
      ctx.beginPath();
      ctx.moveTo(pos[i].x, pos[i].y);
      ctx.lineTo(pos[j].x, pos[j].y);
      ctx.strokeStyle = `rgba(${m.Kr[i * m.n + j] >= 0 ? '34,211,238' : '167,139,250'},${(0.08 + k * 0.5).toFixed(3)})`;
      ctx.lineWidth = k * 2.4;
      ctx.stroke();
    }

    // nodes + phasors
    for (let i = 0; i < m.n; i++) {
      const { x, y } = pos[i];
      const amp = Math.hypot(m.xr[i], m.xi[i]);
      const ph = Math.atan2(m.xi[i], m.xr[i]);
      const r = 5 + Math.min(6, amp * 6);
      const kind = i < m.nIn ? 'in' : i >= m.n - m.nOut ? 'out' : 'hid';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = kind === 'in'
        ? 'rgba(34,211,238,0.14)' : kind === 'out'
          ? 'rgba(52,211,153,0.14)' : 'rgba(255,255,255,0.05)';
      ctx.fill();
      ctx.strokeStyle = kind === 'in'
        ? 'rgba(34,211,238,0.7)' : kind === 'out'
          ? 'rgba(52,211,153,0.7)' : 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
      // phasor
      const len = Math.min(16, amp * 14);
      if (len > 0.5) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + len * Math.cos(ph), y + len * Math.sin(ph));
        ctx.strokeStyle = `hsla(${Math.round(((ph + Math.PI) / (2 * Math.PI)) * 360)}, 90%, 62%, 0.95)`;
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
      axisLabel(ctx, `${i}`, x + r + 2, y + 3, 'left', 'rgba(148,163,184,0.6)');
    }
    axisLabel(ctx, 'entrées clampées', xIn, 12, 'center');
    axisLabel(ctx, 'unités cohérentes (phasors = champs complexes)', w / 2, 12, 'center');
    axisLabel(ctx, 'sortie', xOut, 12, 'center');
  }, [version], 0.5);

  // ------------------------------------------------------------ loss curve
  const loss = useCanvas((ctx, w, h) => {
    const pad = { l: 44, r: 10, t: 14, b: 20 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 6, 4);
    if (lossHist.length < 2) {
      axisLabel(ctx, 'lancez l’entraînement par contraste', pad.l + iw / 2, pad.t + ih / 2, 'center');
      return;
    }
    let mn = Infinity, mx = -Infinity;
    for (const v of lossHist) { const l = Math.log10(Math.max(v, 1e-10)); mn = Math.min(mn, l); mx = Math.max(mx, l); }
    const span = Math.max(mx - mn, 0.5);
    const lo = mn - span * 0.08, hi = mx + span * 0.08;
    ctx.beginPath();
    for (let i = 0; i < lossHist.length; i++) {
      const x = pad.l + (i / (lossHist.length - 1)) * iw;
      const y = pad.t + ih - ((Math.log10(Math.max(lossHist[i], 1e-10)) - lo) / (hi - lo)) * ih;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(34,211,238,0.9)';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    for (let t = Math.ceil(lo); t <= Math.floor(hi); t++) {
      const y = pad.t + ih - ((t - lo) / (hi - lo)) * ih;
      axisLabel(ctx, `1e${t}`, pad.l - 5, y + 3, 'right');
    }
    axisLabel(ctx, 'perte de prédiction (log)', pad.l + 3, pad.t + 10, 'left', 'rgba(34,211,238,0.9)');
    axisLabel(ctx, `epoch ${epochsDone}`, pad.l + iw, h - 4, 'right');
  }, [lossHist, epochsDone], 0.34);

  const m = machineRef.current;

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_310px]">
      <div className="space-y-3">
        <Panel
          tag="FEP"
          title="Machine à énergie libre cohérente"
          subtitle="Champs complexes minimisant E = −½Re[x†Kx] + ¼Σ|x|⁴ + ½λΣ|x|² — apprentissage par contraste d'équilibre, sans rétropropagation."
          right={
            <div className="flex gap-1.5">
              <Btn tone="primary" active={live} onClick={() => setLive((v) => !v)} title="relaxation continue">
                {live ? '◉ Live' : '○ Gelé'}
              </Btn>
              <Btn onClick={() => setRunning((r) => !r)}>{running ? '⏸ Pause' : '▶ Souffler'}</Btn>
            </div>
          }
          flush
        >
          <div className="w-full px-3.5 pb-2"><canvas ref={net.ref} style={{ width: '100%', height: net.size.h }} className="block" /></div>
        </Panel>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Stat
            label="Identité phase-gradient"
            value={verify ? verify.relErr.toExponential(2) : '—'}
            tone={verify && verify.relErr < 0.05 ? 'green' : verify ? 'amber' : 'slate'}
            hint={verify ? `${verify.checks} couplages (réel+imaginaire)` : 'lancer la vérification'}
          />
          <Stat label="Réduction de perte" value={train ? `${train.reduction.toFixed(1)}×` : '—'} tone={train && train.reduction > 3 ? 'green' : 'amber'} hint={train ? `${train.startLoss.toExponential(2)} → ${train.finalLoss.toExponential(2)}` : 'après entraînement'} />
          <Stat label="Perte" value={freeState ? '—' : '—'} tone="slate" hint="voir la courbe" />
          <Stat label="Énergie libre" value={freeState ? freeState.vfe.toFixed(1) : '—'} tone="violet" hint="E de l'équilibre libre" />
          <Stat label="Amplitudes" value={freeState ? `${Math.min(...freeState.amps).toFixed(2)}–${Math.max(...freeState.amps).toFixed(2)}` : '—'} tone="cyan" hint={`[${m ? m.nIn : '?'}→${m ? m.nHidden : '?'}→${m ? m.nOut : '?'}]`} />
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <Panel tag="CONTRASTE" title="Apprentissage par contraste d'équilibre" flush>
            <div className="w-full"><canvas ref={loss.ref} style={{ width: '100%', height: loss.size.h }} className="block" /></div>
            <div className="px-3.5 pb-3.5">
              <div className="flex flex-wrap gap-1.5">
                <Btn tone="primary" onClick={runVerify} disabled={running}>⛁ Vérifier l'identité</Btn>
                <Btn onClick={runTrain} disabled={running}>{running ? 'Entraînement…' : '▶ Entraîner par contraste'}</Btn>
              </div>
            </div>
          </Panel>
          <Panel tag="Δθ" title="Gradient par contraste de phases"
            subtitle="La nudge β déplace l'équilibre ; la différence d'énergie entre les deux équilibres EST le gradient.">
            <div className="space-y-2.5 text-[10px] leading-relaxed text-slate-500">
              <div className="flex items-center gap-2">
                <span className="w-4 text-center text-slate-600">1</span>
                <span>Équilibre libre <Formula>x⁰</Formula> : les unités minimisent E sous entrées clampées.</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 text-center text-slate-600">2</span>
                <span>Équilibre nudgé <Formula>x^β</Formula> : on ajoute <Formula>β·L</Formula> sur la sortie (petit tirage vers la cible).</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 text-center text-slate-600">3</span>
                <span>
                  <Formula>∂L/∂θ = (1/β)·[∂E/∂θ(x^β) − ∂E/∂θ(x⁰)]</Formula>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 text-center text-slate-600">4</span>
                <span>Aucune chaîne de dérivées : la physique calcule le gradient, on ne fait que lire <Formula>∂E/∂θ</Formula> aux deux équilibres.</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                <Tag tone="cyan">pas de backprop</Tag>
                <Tag tone="cyan">2 équilibres par pas</Tag>
                <Tag tone="violet">phase = variable</Tag>
              </div>
            </div>
          </Panel>
        </div>

        <Panel tag="IDENTITÉ" title="Ce qui est réellement vérifié">
          <p className="text-[10px] leading-relaxed text-slate-500">
            Le bouton <span className="text-slate-300">Vérifier l'identité</span> compare, pour des couplages tirés au hasard
            (partie <span className="text-slate-300">réelle</span> — paire hermitienne symétrique — et partie{' '}
            <span className="text-slate-300">imaginaire</span> — paire antisymétrique), le gradient EP
            <Formula>(1/β)[∂E/∂θ(x^β) − ∂E/∂θ(x⁰)]</Formula> contre des différences finies centrées
            <Formula>(L(θ+h) − L(θ−h)) / 2h</Formula>, h = 10⁻⁶. L'équilibre libre est un minimum local strict
            (le test s'exécute en régime cohérent stable), la relaxation est une continuation à partir de x⁰,
            et le résidu d'équilibre est mesuré (~10⁻¹⁵).
          </p>
          {verify && (
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-white/6 bg-white/[0.015] px-2.5 py-2 text-[10px]">
              <span className="text-slate-400">erreur relative max :</span>
              <Tag tone={verify.relErr < 0.05 ? 'green' : 'rose'}>{verify.relErr.toExponential(2)}</Tag>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">pire couple ({verify.worstPair.i},{verify.worstPair.j}) {verify.worstPair.part} : gEP={verify.worstPair.gEP.toExponential(2)} vs gFD={verify.worstPair.gFD.toExponential(2)}</span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">résidu équilibre {verify.resid.toExponential(1)}</span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">ΔE max {verify.dEmax.toExponential(1)} {verify.dEmax <= 1e-12 ? '(monotone ✓)' : ''}</span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">pas adaptatifs {verify.stepsUsed}</span>
            </div>
          )}
        </Panel>

        <Panel tag="OPTIQUE" title="Réalisation photonique de la machine cohérente">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {[
              ['Champs complexes → phases de cavités', 'Chaque unité x_i est l’amplitude complexe d’un microrésonateur. L’énergie libre E émerge du couplage évanscent entre cavités (matrice K, ajustable thermo-optiquement).'],
              ['La nudge → injection de signal pilote', 'Le terme β·L sur la sortie est un pilote faible injecté dans la cavité de sortie. Deux régimes de relaxation (libre puis nudgé) = deux acquisitions rapides de phase.'],
              ['Le gradient → un contraste, pas un calcul', '∂E/∂θ est lu comme un produit de champs (produit homodyne x̄_ix_j) aux deux équilibres. C’est le même dispositif qui calcule et qui apprend.'],
            ].map(([t, d]) => (
              <div key={t} className="rounded-lg border border-white/6 bg-white/[0.015] p-2.5">
                <div className="text-[10px] font-semibold text-cyan-300">{t}</div>
                <p className="mt-1 text-[10px] leading-relaxed text-slate-500">{d}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[10px] leading-relaxed text-slate-500">
            Le coût par pas d'apprentissage est <span className="text-slate-300">deux relaxations d'équilibre</span> — pas de
            passage arrière, pas de chaîne de dérivées. La machine corrige ses couplages d'un contraste de phases.
          </p>
        </Panel>
      </div>

      {/* controls */}
      <div className="space-y-3">
        <Panel tag="GÉOMÉTRIE" title="Réseau & pompe">
          <div className="space-y-3">
            <Slider label="Unités cachées" value={cfg.nHidden} min={2} max={10} step={1} onChange={(v) => setCfg((c) => ({ ...c, nHidden: v }))} format={(v) => `${v}`} />
            <Slider label="Pompe d'entrée (épingle les phases)" value={cfg.pump} min={0.3} max={2} step={0.05} onChange={(v) => setCfg((c) => ({ ...c, pump: v }))} format={(v) => v.toFixed(2)} />
            <Slider label="Auto-couplage α (amplitude √(α−λ))" value={cfg.alpha} min={0.2} max={1.5} step={0.05} onChange={(v) => setCfg((c) => ({ ...c, alpha: v }))} format={(v) => v.toFixed(2)} />
            <Slider label="Amortissement λ" value={cfg.lam} min={0.02} max={0.5} step={0.02} onChange={(v) => setCfg((c) => ({ ...c, lam: v }))} format={(v) => v.toFixed(2)} />
          </div>
          <div className="mt-3 text-[10px] leading-relaxed text-slate-500">
            Changer un paramètre <span className="text-slate-300">réinitialise</span> la matrice K (nouveau seed). L'identité
            phase-gradient est vérifiée dans le régime cohérent stable — condition physique de fonctionnement.
          </div>
        </Panel>

        <Panel tag="CONTRASTE" title="Paramètres du contraste">
          <div className="space-y-3">
            <Slider label="Nudge β" value={cfg.beta} min={0.0002} max={0.02} step={0.0002} onChange={(v) => setCfg((c) => ({ ...c, beta: v }))} format={(v) => v.toFixed(4)} />
            <Slider label="Pas d'apprentissage η" value={cfg.eta} min={0.001} max={0.02} step={0.0005} onChange={(v) => setCfg((c) => ({ ...c, eta: v }))} format={(v) => v.toFixed(4)} />
            <Slider label="Époques" value={cfg.epochs} min={2} max={40} step={1} onChange={(v) => setCfg((c) => ({ ...c, epochs: v }))} format={(v) => `${v}`} />
          </div>
          <p className="mt-3 text-[10px] leading-relaxed text-slate-500">
            β→0 donne le gradient exact ; le protocole en deux temps (libre puis nudgé, continuation) évite les sauts de bassin.
          </p>
        </Panel>

        <Panel tag="VÉRIF" title="Régime de validité">
          <p className="text-[10px] leading-relaxed text-slate-400">
            L'identité est exacte quand : K est hermitien (flot = gradient de E), l'équilibre libre est un minimum
            strict (amplitudes saines, phases épinglées par la pompe), et la relaxation rejoint l'équilibre physique.
            Sans cela, les modes de phase restent marginaux et le contraste se trompe.
          </p>
          <div className="mt-2 flex flex-wrap gap-1">
            <Tag tone="green">K hermitien</Tag>
            <Tag tone="green">min local strict</Tag>
            <Tag tone="amber">phase-mode conditionné</Tag>
            <Tag tone="rose">pas de col de l'énergie</Tag>
          </div>
        </Panel>

        <Panel tag="LIMITE" title="Ce que ça ne fait pas">
          <p className="text-[10px] leading-relaxed text-slate-400">
            C'est une preuve de principe du contraste d'équilibre sur champs complexes — une machine de
            dimension <Formula>{m ? `${m.n} champs` : '—'}</Formula> sur une tâche prédictive scalaire. Pas de
            profondeur, pas de mémoire longue, pas de gros modèles : la frontière avec le transformer reste électronique.
          </p>
          <div className="mt-2 flex flex-wrap gap-1">
            <Tag tone="green">contraste d'équilibre</Tag>
            <Tag tone="green">prédiction temporelle</Tag>
            <Tag tone="rose">pas de LLM</Tag>
          </div>
        </Panel>
      </div>

      <AttentionPanel />
    </div>
  );
}

// ---------------------------------------------------------------------------
//  ATTENTION CONDITIONNELLE — le résultat de recherche porté en démo live.
//  Routeur s_k = s0_k + W_k·c appris par FD exacte (couplages EP figés).
// ---------------------------------------------------------------------------

function AttentionPanel() {
  const [initMode, setInitMode] = useState<'directionnel' | 'quasi-uniforme'>('directionnel');
  const [seed, setSeed] = useState(7);
  const [busy, setBusy] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [result, setResult] = useState<null | {
    a0: Float64Array; a1: Float64Array; before: number; after: number; oracle: number;
  }>(null);
  const [history, setHistory] = useState<number[]>([]);
  const machineRef = useRef<ReturnType<typeof makeAttn> | null>(null);
  const trainRef = useRef<AttnSample[]>([]);
  const testRef = useRef<AttnSample[]>([]);

  const train = useCallback(() => {
    if (busy) return;
    const cfg = { ...DEFAULT_ATTN, seed: 42 + seed * 7 };
    const m = makeAttn(cfg);
    initRouter(m, initMode);
    trainRef.current = makeAttnSamples(80, seed);
    testRef.current = makeAttnSamples(100, seed + 31);
    machineRef.current = m;
    setHistory([evalAttnLoss(m, testRef.current)]);
    setResult(null);
    setEpoch(0);
    setBusy(true);

    // époques asynchrones : l'UI respire entre chaque
    let ep = 0;
    const step = () => {
      if (ep >= LR_SCHEDULE.length || !machineRef.current) {
        const mm = machineRef.current!;
        setResult({
          a0: attentionOf(mm, 0), a1: attentionOf(mm, 1),
          before: history[0] ?? evalAttnLoss(mm, testRef.current),
          after: evalAttnLoss(mm, testRef.current),
          oracle: evalOracle(mm, testRef.current),
        });
        setBusy(false);
        return;
      }
      trainRouterEpoch(machineRef.current, trainRef.current, LR_SCHEDULE[ep]);
      ep++;
      setEpoch(ep);
      setHistory((h) => [...h, evalAttnLoss(machineRef.current!, testRef.current)]);
      setTimeout(step, 0);
    };
    setTimeout(step, 0);
  }, [busy, seed, initMode, history]);

  const { ref } = useCanvas((ctx, w, h) => {
    drawGrid(ctx, 0, 0, w, h, 6, 3);
    if (history.length < 2) return;
    const lo = 0, hi = Math.max(...history) * 1.15 || 1;
    drawSeries(ctx, history, 0, 0, w, h, lo, hi, 'rgba(34,211,238,0.95)', 'rgba(34,211,238,0.07)', 1.6);
    if (result) {
      // ligne oracle
      const py = h - (result.oracle / hi) * h;
      ctx.strokeStyle = 'rgba(52,211,153,0.7)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
      ctx.setLineDash([]);
      axisLabel(ctx, `oracle ${result.oracle.toFixed(4)}`, w - 110, py - 5, 'left', 'rgba(52,211,153,0.95)');
    }
    axisLabel(ctx, `${history.length - 1} époques`, 6, h - 5);
  }, [history, result]);

  const ok = result && ((result.a0[0] > 0.6 && result.a1[1] > 0.6) || (result.a0[1] > 0.6 && result.a1[0] > 0.6));
  const beatsOracle = result && result.after < result.oracle;

  return (
    <Panel
      tag="ATTENTION ⚡"
      title="Attention conditionnelle par pompe — le routeur apprend à BASCULER selon le contexte"
      subtitle="c = t%2 : le lag est sur le canal c, du bruit sur l'autre. Le routeur (4 paramètres) est appris par différences finies exactes — le gradient EP des pompes seul est faux d'un facteur 8.5, cf. boucle grounded."
      right={<Btn tone="primary" onClick={train} disabled={busy}>{busy ? `⋯ époque ${epoch}/10` : '▶ Entraîner le routeur'}</Btn>}
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '6 / 1.4' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            options={[
              { id: 'directionnel', label: 'init directionnelle (5/5 fiable)', title: 's0=[1,−1], W=[−2,2] — warm-start légitime' },
              { id: 'quasi-uniforme', label: 'init quasi-uniforme (fragile)', title: 's0=[0.1,−0.1] — montre la fragilité réelle du minimum symétrique' },
            ]}
            value={initMode}
            onChange={(v) => setInitMode(v as 'directionnel' | 'quasi-uniforme')}
          />
          <div className="w-40">
            <Slider label="seed données" value={seed} min={1} max={40} step={1} onChange={setSeed} format={(v) => `#${v}`} />
          </div>
        </div>
        {result && (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="attn(c=0)" value={Array.from(result.a0).map((x) => x.toFixed(2)).join(' / ')} tone="cyan"
              hint="canal attendu : 0" />
            <Stat label="attn(c=1)" value={Array.from(result.a1).map((x) => x.toFixed(2)).join(' / ')} tone="violet"
              hint="canal attendu : 1" />
            <Stat label="Loss test" value={result.after.toFixed(4)} tone={result.after <= result.oracle * 1.02 ? 'green' : 'amber'}
              hint={`avant ${result.before.toFixed(4)}`} />
            <Stat label="vs oracle dur" value={beatsOracle ? 'DÉPASSÉE' : `${((result.after / result.oracle - 1) * 100).toFixed(1)}%`}
              tone={beatsOracle ? 'green' : 'slate'}
              hint={`oracle ${result.oracle.toFixed(4)} — attention douce = beamforming`} />
          </div>
        )}
        {result && (
          <div className="flex items-center gap-2">
            <Tag tone={ok ? 'green' : 'rose'}>{ok ? 'BASCULE PAR CONTEXTE ✓' : 'pas de bascule — minimum symétrique'}</Tag>
            <Tag tone={beatsOracle ? 'green' : 'slate'}>{beatsOracle ? 'bat l\'attention forcée' : '≈ oracle'}</Tag>
            {initMode === 'quasi-uniforme' && !ok && <Tag tone="amber">fragilité réelle : l'init uniforme tombe dans l'inversé</Tag>}
          </div>
        )}
      </div>
    </Panel>
  );
}