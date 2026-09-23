import { useMemo, useRef, useState, useCallback } from 'react';

import {
  ringsTarget, dotsTarget, gerchbergSaxton, quantizePhase,
  fraunhoferSincCut, normalizeTarget,
  type GsTarget, type GsResult,
} from '../physics/holography';
import {
  Panel, Stat, Tag, Slider, Formula, Segmented, Btn,
  useCanvas, drawGrid, drawSeries, axisLabel, paintField, drawImageBuffer, LUT,
} from '../ui/kit';

const GRID = 128;

// ---------------------------------------------------------------------------
//  Cibles
// ---------------------------------------------------------------------------

function targetFromAmp(amp: Float64Array): GsTarget {
  const mask = new Uint8Array(amp.length);
  for (let i = 0; i < amp.length; i++) mask[i] = amp[i] > 0 ? 1 : 0;
  return { amp, mask, n: GRID };
}

function textPreset(): Float64Array {
  const off = document.createElement('canvas');
  off.width = GRID; off.height = GRID;
  const c = off.getContext('2d');
  if (!c) return new Float64Array(GRID * GRID);
  c.fillStyle = '#000'; c.fillRect(0, 0, GRID, GRID);
  c.fillStyle = '#fff';
  c.textAlign = 'center'; c.textBaseline = 'middle';
  c.font = 'bold 23px ui-monospace, monospace';
  c.fillText('LUMIÈRE', GRID / 2, 48);
  c.font = 'bold 30px ui-monospace, monospace';
  c.fillText('2026', GRID / 2, 88);
  const img = c.getImageData(0, 0, GRID, GRID);
  const amp = new Float64Array(GRID * GRID);
  for (let i = 0; i < amp.length; i++) amp[i] = img.data[i * 4] > 90 ? 1 : 0;
  return amp;
}

const PRESETS = [
  { id: 'anneaux', label: 'Anneaux' },
  { id: 'points', label: 'Points' },
  { id: 'texte', label: 'Texte' },
] as const;
type PresetId = (typeof PRESETS)[number]['id'];

function presetAmp(id: PresetId): Float64Array {
  if (id === 'anneaux') return ringsTarget(GRID).amp;
  if (id === 'points') return dotsTarget(GRID).amp;
  return textPreset();
}

// ---------------------------------------------------------------------------
//  Lab
// ---------------------------------------------------------------------------

export default function HoloLab() {
  const [preset, setPreset] = useState<PresetId>('anneaux');
  const [iters, setIters] = useState(40);
  const [levels, setLevels] = useState<'2' | '4' | '8' | '16' | '256'>('256');
  const [version, setVersion] = useState(0);       // cible affichée
  const [solveVersion, setSolveVersion] = useState(0); // solveur relancé

  const ampRef = useRef<Float64Array>(presetAmp('anneaux'));
  const paintRef = useRef<{ painting: boolean; erase: boolean }>({ painting: false, erase: false });

  const bumpBoth = useCallback(() => { setVersion((v) => v + 1); setSolveVersion((v) => v + 1); }, []);

  const loadPreset = useCallback((id: PresetId) => {
    ampRef.current = presetAmp(id);
    setPreset(id);
    bumpBoth();
  }, [bumpBoth]);

  const clearTarget = useCallback(() => {
    ampRef.current = new Float64Array(GRID * GRID);
    bumpBoth();
  }, [bumpBoth]);

  // ---- solveur : cible → GS → hologramme + reconstruction + métriques
  const solve = useMemo(() => {
    const t0 = performance.now();
    const target = targetFromAmp(ampRef.current);
    const filled = target.mask.reduce((s, m) => s + m, 0);
    const empty = filled === 0;
    const r: GsResult = empty
      ? { ...gerchbergSaxton(ringsTarget(GRID), 1), } // placeholder stable, écran vide géré à l'affichage
      : gerchbergSaxton(target, iters);
    const lvl = Number(levels);
    const q = empty || lvl >= 256 ? null : quantizePhase(target, r.hologramPhase, lvl);
    return { r, q, ms: performance.now() - t0, filled, empty, lvl };
  }, [iters, levels, solveVersion]);

  const nrm = useMemo(() => normalizeTarget(targetFromAmp(ampRef.current)), [version]);

  // ---- peinture sur la cible
  const paintAt = useCallback((e: React.PointerEvent<Element>) => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * GRID);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * GRID);
    const amp = ampRef.current;
    const erase = paintRef.current.erase;
    const rad = 2.4;
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= GRID || y >= GRID) continue;
        if (dx * dx + dy * dy <= rad * rad) {
          amp[y * GRID + x] = erase ? 0 : 1;
        }
      }
    }
    setVersion((v) => v + 1);
  }, []);

  // ---- canvases ------------------------------------------------------------
  const { ref: targetRef } = useCanvas((ctx, w, h) => {
    const img = new ImageData(GRID, GRID);
    let tMax = 1e-9;
    for (let i = 0; i < nrm.amp.length; i++) if (nrm.amp[i] > tMax) tMax = nrm.amp[i];
    paintField(img, nrm.amp, GRID, GRID, LUT.ice, 0, tMax, 1);
    drawImageBuffer(ctx, img, 0, 0, w, h, false);
  }, [version, nrm]);

  const { ref: reconRef } = useCanvas((ctx, w, h) => {
    if (solve.empty) {
      drawGrid(ctx, 0, 0, w, h, 8, 8);
      axisLabel(ctx, 'dessinez la cible ←', w / 2 - 52, h / 2, 'left', 'rgba(148,163,184,0.8)');
      return;
    }
    const img = new ImageData(GRID, GRID);
    let mx = 1e-12;
    for (let i = 0; i < solve.r.reconstruction.length; i++) mx = Math.max(mx, solve.r.reconstruction[i]);
    paintField(img, solve.r.reconstruction, GRID, GRID, LUT.inferno, 0, mx, 0.45);
    drawImageBuffer(ctx, img, 0, 0, w, h, true);
  }, [solve, version]);

  const { ref: phaseRef } = useCanvas((ctx, w, h) => {
    if (solve.empty) return;
    // phase décalée (fftshift) pour centrer — affichage tel quel, sans embellie
    const n = GRID;
    const ph = solve.r.hologramPhase;
    const shifted = new Float64Array(n * n);
    const half = n >> 1;
    for (let y = 0; y < n; y++) {
      const yy = (y + half) % n;
      for (let x = 0; x < n; x++) shifted[yy * n + ((x + half) % n)] = ph[y * n + x];
    }
    const img = new ImageData(n, n);
    paintField(img, shifted, n, n, LUT.twilight, -Math.PI, Math.PI, 1);
    drawImageBuffer(ctx, img, 0, 0, w, h, false);
  }, [solve, version]);

  const { ref: convRef } = useCanvas((ctx, w, h) => {
    drawGrid(ctx, 26, 6, w - 32, h - 24, 6, 4);
    if (solve.empty) return;
    const rh = solve.r.rmseHistory;
    const eh = solve.r.effHistory;
    let rMax = 1e-9;
    for (let i = 0; i < rh.length; i++) if (rh[i] > rMax) rMax = rh[i];
    const x0 = 26, pw = w - 32, phh = h - 24;
    const halfH = phh * 0.46;
    // RMSE (haut, violet) — normalisé à son max
    drawSeries(ctx, Array.from(rh, (v) => v / rMax), x0, 6, pw, halfH, 0, 1, 'rgba(167,139,250,0.95)', undefined, 1.6);
    // η (bas, cyan) — échelle absolue [0,1] : comparable d'un essai à l'autre
    drawSeries(ctx, Array.from(eh), x0, 6 + halfH + 8, pw, halfH, 0, 1, 'rgba(34,211,238,0.95)', undefined, 1.6);
    axisLabel(ctx, `RMSE ${rh[0].toFixed(3)} → ${rh[rh.length - 1].toFixed(3)}`, x0 + 4, 16, 'left', 'rgba(167,139,250,0.95)');
    axisLabel(ctx, `η → ${(eh[eh.length - 1] * 100).toFixed(1)}%`, x0 + 4, 6 + halfH + 20, 'left', 'rgba(34,211,238,0.95)');
    axisLabel(ctx, 'itération 1', x0, h - 4);
    axisLabel(ctx, String(iters), x0 + pw - 8, h - 4, 'right');
  }, [solve, iters, version]);

  // ---- Fraunhofer
  const [slitW, setSlitW] = useState(12);
  const cut = useMemo(() => fraunhoferSincCut(128, slitW, 110), [slitW]);
  const { ref: frRef } = useCanvas((ctx, w, h) => {
    const pad = { l: 8, r: 8, t: 12, b: 14 };
    const pw = w - pad.l - pad.r, phh = h - pad.t - pad.b;
    drawGrid(ctx, pad.l, pad.t, pw, phh, 6, 4);
    // sinc² continu (référence du manuel, évaluée en continu)
    const cont: number[] = [];
    for (let i = 0; i < 240; i++) {
      const kk = (i / 239) * cut.k[cut.k.length - 1];
      const u = (Math.PI * kk * slitW) / 128;
      cont.push(u === 0 ? 1 : (Math.sin(u) / u) ** 2);
    }
    drawSeries(ctx, cont, pad.l, pad.t, pw, phh, 0, 1.05, 'rgba(167,139,250,0.8)', undefined, 1.2);
    // mesure FFT (escalier des bins — c'est la vérité discrète)
    drawSeries(ctx, Array.from(cut.measured), pad.l, pad.t, pw, phh, 0, 1.05, 'rgba(34,211,238,0.95)', undefined, 1);
    axisLabel(ctx, `FFT mesurée`, pad.l + 6, pad.t + 12, 'left', 'rgba(34,211,238,0.95)');
    axisLabel(ctx, 'sinc² continu', pad.l + 108, pad.t + 12, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, `écart max ${(cut.maxDiff * 100).toFixed(2)}%`, pad.l + 210, pad.t + 12, 'left', 'rgba(52,211,153,0.95)');
    axisLabel(ctx, `k·w/n = 0 → ${((cut.k[cut.k.length - 1] * slitW) / 128).toFixed(2)} (première extinction)`, pad.l + 6, h - 3);
  }, [cut, slitW]);

  // ---- rendu ---------------------------------------------------------------
  const blazedRef = solve.lvl < 256 ? (Math.sin(Math.PI / solve.lvl) / (Math.PI / solve.lvl)) ** 2 : 1;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        <Panel
          tag="CIBLE"
          title="Plan image — dessinez l'intensité à reconstruire"
          subtitle="Chaque pixel peint est une contrainte d'amplitude. Le masque est l'ensemble des pixels contraints."
          right={
            <div className="flex flex-wrap items-center gap-1.5">
              {PRESETS.map((p) => (
                <Btn key={p.id} active={preset === p.id} onClick={() => loadPreset(p.id)}>{p.label}</Btn>
              ))}
              <Btn onClick={clearTarget} title="vider la cible">✕ Vider</Btn>
            </div>
          }
        >
          <div
            onPointerDown={(e) => {
              paintRef.current = { painting: true, erase: e.shiftKey || e.button === 2 };
              e.currentTarget.setPointerCapture(e.pointerId);
              paintAt(e);
            }}
            onPointerMove={(e) => { if (paintRef.current.painting) paintAt(e); }}
            onPointerUp={() => {
              // le solveur ne relance qu'au relâchement — le tracé reste à 60 fps
              if (paintRef.current.painting) { paintRef.current.painting = false; bumpBoth(); }
            }}
            onContextMenu={(e) => e.preventDefault()}
            className="cursor-crosshair select-none overflow-hidden rounded-lg ring-1 ring-white/10"
            title="clic = peindre · Maj+clic (ou clic droit) = effacer"
          >
            <canvas ref={targetRef} className="block w-full" style={{ aspectRatio: '1 / 1' }} />
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            {solve.filled} pixels contraints sur {GRID * GRID} · amplitude cible normalisée à{' '}
            <Formula>Σ_masque|amp|² = 1</Formula> — la puissance unitaire de l'hologramme l'exige (Parseval).
          </p>
        </Panel>

        <Panel
          tag="RECONSTRUCTION"
          title="Ce que l'hologramme éclaire vraiment"
          subtitle="IFFT de la phase seule — amplitude de l'hologramme ignorée, lumière perdue affichée (zéro ordre au centre)."
          right={solve.empty ? <Tag>en attente</Tag> : <Tag tone={solve.r.efficiency > 0.8 ? 'green' : 'amber'}>η = {(solve.r.efficiency * 100).toFixed(1)}%</Tag>}
        >
          <div className="overflow-hidden rounded-lg ring-1 ring-white/10">
            <canvas ref={reconRef} className="block w-full" style={{ aspectRatio: '1 / 1' }} />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Efficacité" value={solve.empty ? '—' : (solve.r.efficiency * 100).toFixed(1)} unit={solve.empty ? undefined : '%'} tone="green" hint="puissance dans le masque" />
            <Stat label="RMSE" value={solve.empty ? '—' : solve.r.rmse.toFixed(3)} tone="violet" hint="vs cible, fenêtre signal" />
            <Stat label="Résolution" value="2 FFT" unit="/ itération" tone="cyan" hint={`${GRID}×${GRID} · ${solve.ms.toFixed(0)} ms`} />
            <Stat label="η quantifiée" value={solve.q ? (solve.q.efficiency * 100).toFixed(1) : solve.empty ? '—' : (solve.r.efficiency * 100).toFixed(1)} unit={solve.q ? '%' : undefined} tone="amber" hint={solve.q ? `${solve.lvl} niveaux · perte ${((1 - solve.q.efficiency / solve.r.efficiency) * 100).toFixed(1)}%` : 'phase continue'} />
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Panel
          tag="HOLOGRAMME"
          title="Phase pure — le livrable physique"
          subtitle="Ce masque de phase, gravé sur un SLM, produit la reconstruction à gauche. Rien d'autre n'est nécessaire : amplitude uniforme."
          right={<Tag tone="cyan">[−π, π]</Tag>}
        >
          <div className="overflow-hidden rounded-lg ring-1 ring-white/10">
            <canvas ref={phaseRef} className="block w-full" style={{ aspectRatio: '1 / 1' }} />
          </div>
          <div className="mt-2.5">
            <div className="mb-1 text-[10px] text-slate-400">Quantification SLM (mesurée après coup)</div>
            <Segmented
              options={[
                { id: '2', label: '2 niv.' }, { id: '4', label: '4' }, { id: '8', label: '8' },
                { id: '16', label: '16' }, { id: '256', label: 'continue' },
              ]}
              value={levels}
              onChange={setLevels}
            />
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
              {solve.q ? (
                <>
                  η mesurée à {solve.lvl} niveaux : <span className="text-amber-300">{(solve.q.efficiency * 100).toFixed(1)}%</span>{' '}
                  (perte {((1 - solve.q.efficiency / solve.r.efficiency) * 100).toFixed(1)} pts). Référence réseau blazé idéal :{' '}
                  <Formula>η/η∞ = (sin(π/L)/(π/L))²</Formula> = {(blazedRef * 100).toFixed(1)}% — un hologramme GS n'est pas un réseau blazé, l'écart est réel.
                </>
              ) : (
                'Phase continue — la référence à laquelle les niveaux quantifiés sont comparés.'
              )}
            </p>
          </div>
        </Panel>

        <Panel
          tag="CONVERGENCE"
          title="Boucle Gerchberg–Saxton, mesurée à chaque itération"
          subtitle="Image ⇄ hologramme : amplitude imposée d'un côté, phase conservée. Hors masque, l'amplitude est libre."
        >
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <Slider label="Itérations" min={4} max={80} step={2} value={iters} onChange={setIters} />
            </div>
            <Tag tone="violet">{iters} it.</Tag>
          </div>
          <div className="mt-2">
            <canvas ref={convRef} className="block w-full" style={{ aspectRatio: '1.7 / 1' }} />
          </div>
          <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
            Violet : RMSE fenêtre signal (échelle propre) · Cyan : efficacité de diffraction (échelle absolue 0–100%).
            La phase initiale est un LCG semé — le même essai redonne les mêmes chiffres.
          </p>
        </Panel>

        <Panel
          tag="FRAUNHOFER"
          title="Le moteur FFT est-il un vrai champ lointain ?"
          subtitle="Coupe d'intensité de la DFT d'une fente vs le sinc² du manuel — mesuré, pas supposé."
        >
          <Slider label="Largeur de fente" min={4} max={24} step={1} value={slitW} onChange={setSlitW} unit=" px" />
          <div className="mt-2">
            <canvas ref={frRef} className="block w-full" style={{ aspectRatio: '2.6 / 1' }} />
          </div>
          <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
            Cyan : intensité FFT (escalier des bins — vérité discrète) · Violet : sinc² continu.
            La forme exacte de la DFT d'un rect discret est le noyau de Dirichlet ; il coïncide avec le sinc²
            d'autant mieux que la fente est fine — l'écart affiché <em>est</em> la physique discrète.
          </p>
        </Panel>
      </div>
    </div>
  );
}
