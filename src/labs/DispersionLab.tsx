import { useMemo, useState } from 'react';

import {
  FUSED_SILICA, BK7, CAF2,
  sellmeier, gdd, tod, zeroDispersionLambda,
  pulseWidthGDD, pulsePropagation,
  bireringingMap,
  type SellmeierGlass,
} from '../physics/optics';
import {
  Panel, Stat, Tag, Slider, Formula, Segmented,
  useCanvas, drawGrid, drawSeries, axisLabel, paintField, drawImageBuffer, LUT,
  useHashParam,
} from '../ui/kit';

const GLASSES: { id: string; label: string; g: SellmeierGlass }[] = [
  { id: 'silice', label: 'Silice', g: FUSED_SILICA },
  { id: 'bk7', label: 'BK7', g: BK7 },
  { id: 'caf2', label: 'CaF₂', g: CAF2 },
];

const L_MIN = 0.25, L_MAX = 3.5;

// ---------------------------------------------------------------------------
//  Panneau dispersion n(λ) · GDD(λ) · TOD(λ)
// ---------------------------------------------------------------------------

function DispersionCurves({ glass, lambda0 }: { glass: SellmeierGlass; lambda0: number }) {
  const N = 400;
  const data = useMemo(() => {
    const ns: number[] = [], b2s: number[] = [], b3s: number[] = [];
    for (let i = 0; i < N; i++) {
      const l = L_MIN + ((L_MAX - L_MIN) * i) / (N - 1);
      const s = sellmeier(glass, l);
      ns.push(s.n); b2s.push(gdd(glass, l)); b3s.push(tod(glass, l));
    }
    return { ns, b2s, b3s };
  }, [glass]);
  const zd = useMemo(() => zeroDispersionLambda(glass), [glass]);

  const { ref } = useCanvas((ctx, w, h) => {
    const pad = { l: 34, r: 12, t: 10, b: 16 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    drawGrid(ctx, pad.l, pad.t, pw, ph, 8, 4);

    // --- n(λ) moitié haute, échelle propre ---
    const nMin = Math.min(...data.ns), nMax = Math.max(...data.ns);
    const halfH = ph * 0.44;
    drawSeries(ctx, data.ns, pad.l, pad.t, pw, halfH, nMin, nMax, 'rgba(34,211,238,0.95)', undefined, 1.6);
    axisLabel(ctx, `n ∈ [${nMin.toFixed(3)}, ${nMax.toFixed(3)}]`, pad.l + 6, pad.t + 11, 'left', 'rgba(34,211,238,0.9)');

    // --- GDD & TOD moitié basse, zéro centré ---
    const y0 = pad.t + ph * 0.56, half2 = ph * 0.40;
    const m2 = Math.max(...data.b2s.map(Math.abs), ...data.b3s.map(Math.abs), 1e-9);
    // ligne zéro
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath(); ctx.moveTo(pad.l, y0 + half2 / 1.08); ctx.lineTo(pad.l + pw, y0 + half2 / 1.08); ctx.stroke();
    drawSeries(ctx, data.b2s.map((v) => v / (1.08 * m2)), pad.l, y0, pw, half2, -1, 1, 'rgba(167,139,250,0.95)', undefined, 1.6);
    drawSeries(ctx, data.b3s.map((v) => v / (1.08 * m2)), pad.l, y0, pw, half2, -1, 1, 'rgba(251,191,36,0.75)', undefined, 1);
    axisLabel(ctx, `GDD ±${m2.toFixed(0)} fs²/mm`, pad.l + 6, y0 + 11, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, `TOD ×10`, pad.l + 130, y0 + 11, 'left', 'rgba(251,191,36,0.8)');

    // --- repères λ ---
    const px = (l: number) => pad.l + ((l - L_MIN) / (L_MAX - L_MIN)) * pw;
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(px(lambda0), pad.t); ctx.lineTo(px(lambda0), pad.t + ph); ctx.stroke();
    if (Number.isFinite(zd)) {
      ctx.strokeStyle = 'rgba(52,211,153,0.55)';
      ctx.beginPath(); ctx.moveTo(px(zd), pad.t); ctx.lineTo(px(zd), pad.t + ph); ctx.stroke();
      axisLabel(ctx, `λ_ZD=${zd.toFixed(2)}µm`, px(zd) + 4, pad.t + ph - 4, 'left', 'rgba(52,211,153,0.95)');
    }
    ctx.setLineDash([]);
    axisLabel(ctx, `${L_MIN}`, pad.l, pad.t + ph + 12);
    axisLabel(ctx, `λ₀=${lambda0.toFixed(2)}µm`, px(lambda0) + 4, pad.t + 24, 'left');
    axisLabel(ctx, `${L_MAX} µm`, pad.l + pw - 46, pad.t + ph + 12);
  }, [data, lambda0, zd]);

  return (
    <>
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2.4 / 1' }} />
      <div className="border-t border-white/5 p-3">
        <p className="text-[10px] leading-relaxed text-slate-500">
          Cyan : indice n(λ) · Violet : GDD β₂ (change de signe à λ_ZD, vert) · Ambre : TOD β₃.
          Dérivées de Sellmeier <Formula>exactes</Formula> — pas de différences finies, pas de bruit dérivatif.
        </p>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
//  Panneau propagation d'impulsion
// ---------------------------------------------------------------------------

function PulsePanel({ glass, lambda0 }: { glass: SellmeierGlass; lambda0: number }) {
  const [tau0, setTau0] = useState(100);
  const [Lmm, setLmm] = useState(150);

  const b2 = gdd(glass, lambda0), b3 = tod(glass, lambda0);
  const closed = pulseWidthGDD(tau0, b2, Lmm);
  const plot = useMemo(() => pulsePropagation(tau0, b2, b3, Lmm), [tau0, b2, b3, Lmm]);

  // FWHM mesurée du profil propagé (même mesure que le test 'opt-pulse')
  const measured = useMemo(() => {
    let mx = 0;
    for (let i = 1; i < plot.IL.length; i++) if (plot.IL[i] > plot.IL[mx]) mx = i;
    const half = plot.IL[mx] / 2;
    let lo = mx, hi = mx;
    while (lo > 0 && plot.IL[lo] > half) lo--;
    while (hi < plot.IL.length - 1 && plot.IL[hi] > half) hi++;
    return (hi - lo) * (plot.t[1] - plot.t[0]);
  }, [plot]);

  const { ref } = useCanvas((ctx, w, h) => {
    drawGrid(ctx, 0, 0, w, h, 8, 4);
    // fenêtre temporelle commune (centrée)
    const tMin = plot.t[0], tMax = plot.t[plot.t.length - 1];
    const span = tMax - tMin || 1;
    const sample = (arr: Float64Array) => {
      const out: number[] = [];
      const step = Math.max(1, Math.floor(arr.length / 600));
      for (let i = 0; i < arr.length; i += step) {
        const tt = tMin + (span * i) / (arr.length - 1);
        void tt;
        out.push(arr[i]);
      }
      return out;
    };
    drawSeries(ctx, sample(plot.I0), 0, 0, w, h, 0, 1.05, 'rgba(34,211,238,0.45)', undefined, 1);
    drawSeries(ctx, sample(plot.IL), 0, 0, w, h, 0, 1.05, 'rgba(167,139,250,0.95)', 'rgba(167,139,250,0.07)', 1.6);
    axisLabel(ctx, `entrée ${tau0} fs`, 8, 14, 'left', 'rgba(34,211,238,0.8)');
    axisLabel(ctx, `sortie ${measured.toFixed(0)} fs`, 110, 14, 'left', 'rgba(167,139,250,0.95)');
    axisLabel(ctx, `${tMin.toFixed(0)}`, 4, h - 5);
    axisLabel(ctx, `temps (fs) · ${tMax.toFixed(0)}`, w - 92, h - 5);
  }, [plot]);

  const regimeTag = Math.abs(b2 * Lmm) < tau0 / 2
    ? { txt: 'régime Fourier', tone: 'green' as const }
    : b3 !== 0 && Math.abs(b3 * Lmm) > Math.abs(b2 * Lmm * 20 / Math.max(tau0, 1))
      ? { txt: 'TOD dominant', tone: 'amber' as const }
      : { txt: 'GDD dominant', tone: 'violet' as const };

  return (
    <Panel
      tag="IMPULSION"
      title="Propagation femtoseconde — phase spectrale exacte, TOD inclus"
      subtitle={`φ(ω) = β₂ω²/2 + β₃ω³/6 sur ${Lmm} mm de ${glass.name} à λ₀ — spectre gaussien transformé-limité, FFT radix-2 locale.`}
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2 / 1' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <Slider label="durée τ₀ (FWHM)" value={tau0} min={20} max={500} step={5} onChange={setTau0} format={(v) => v.toFixed(0)} unit=" fs" />
          <Slider label="longueur L" value={Lmm} min={0} max={500} step={5} onChange={setLmm} format={(v) => v.toFixed(0)} unit=" mm" />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="β₂ / β₃ à λ₀" value={`${b2.toFixed(1)} / ${b3.toFixed(0)}`} hint="fs²/mm · fs³/mm" tone="cyan" />
          <Stat label="Régime" value={<Tag tone={regimeTag.tone}>{regimeTag.txt}</Tag>} mono={false} />
          <Stat label="τ fermée (GDD)" value={closed.toFixed(0)} unit="fs" tone="violet" hint="formule analytique" />
          <Stat label="τ mesurée (FFT)" value={measured.toFixed(0)} unit="fs" tone={Math.abs(measured - closed) / closed < 0.03 ? 'green' : 'amber'} hint={`écart ${(Math.abs(measured - closed) / closed * 100).toFixed(1)} %`} />
        </div>
        <p className="text-[10px] leading-relaxed text-slate-500">
          Le TOD n'a pas de forme fermée simple : ici il est calculé exactement par phase
          spectrale cubique — l'impulsion s'<em>asymétrise</em> (glissement du centroïde) sans se
          borner à s'élargir.
        </p>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  Panneau Fresnel biaxial — carte de biréfringence
// ---------------------------------------------------------------------------

function BiaxialPanel() {
  const [nx, setNx] = useState(1.35);
  const [ny, setNy] = useState(1.55);
  const [nz, setNz] = useState(1.85);

  const map = useMemo(() => bireringingMap(nx, ny, nz, 90, 180), [nx, ny, nz]);
  let mn = Infinity, mx = -Infinity;
  for (const v of map) { if (v < mn) mn = v; if (v > mx) mx = v; }

  const ordered = nx <= ny && ny <= nz;
  const { ref } = useCanvas((ctx, w, h) => {
    const img = new ImageData(180, 90);
    paintField(img, map, 180, 90, LUT.ice, mn, mx);
    drawImageBuffer(ctx, img, 0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    axisLabel(ctx, 'φ ∈ [−π, π]', 6, h - 6);
    axisLabel(ctx, 'θ 0→π', 6, 12);
    axisLabel(ctx, `Δn max = ${mx.toFixed(4)}`, w - 100, h - 6, 'left', 'rgba(103,232,249,0.95)');
  }, [map]);

  // deux axes optiques : points où Δn = 0 hors plans principaux
  const uniaxial = Math.abs(ny * ny - nx * nz) < 0.004 * nx * nz;

  return (
    <Panel
      tag="FRESNEL BIAXIAL"
      title="Ellipsoïde des indices — Δn sur la sphère des directions, forme fermée"
      subtitle="u² − B·u + C résolu exactement (le rapport source affichait « Σ s_k²/(n²−n_k²) = 1/n² » : coquille corrigée en « = 0 », même factorisation)."
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '4 / 1.4' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="grid grid-cols-3 gap-x-4 gap-y-2">
          <Slider label="n_x" value={nx} min={1.2} max={2.2} step={0.01} onChange={setNx} format={(v) => v.toFixed(2)} />
          <Slider label="n_y" value={ny} min={1.2} max={2.2} step={0.01} onChange={setNy} format={(v) => v.toFixed(2)} />
          <Slider label="n_z" value={nz} min={1.2} max={2.2} step={0.01} onChange={setNz} format={(v) => v.toFixed(2)} />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Ordre principal" value={<Tag tone={ordered ? 'green' : 'rose'}>{ordered ? 'n_x ≤ n_y ≤ n_z ✓' : 'désordonné'}</Tag>} mono={false}
            hint={ordered ? undefined : 'réordonnez : la théorie suppose nx<ny<nz'} />
          <Stat label="Δn max sphère" value={mx.toFixed(4)} tone="cyan" />
          <Stat label="Δn moyen" value={(map.reduce((a, b) => a + b, 0) / map.length).toFixed(4)} tone="slate" />
          <Stat label="Type" value={<Tag tone={uniaxial ? 'amber' : 'violet'}>{uniaxial ? '~uniaxe' : 'biaxe'}</Tag>} mono={false}
            hint="biaxe ⇔ n_y² ≠ n_x·n_z" />
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------

export default function DispersionLab() {
  // état partagé par URL : #verre=bk7&l0=1.55
  const [gid, setGid] = useHashParam('verre', 'silice');
  const [l0str, setL0str] = useHashParam('l0', '0.8');
  const lambda0 = Math.max(0.3, Math.min(3, Number(l0str) || 0.8));
  const glass = GLASSES.find((g) => g.id === gid)?.g ?? FUSED_SILICA;

  return (
    <div className="space-y-3">
      <Panel
        tag="DISPERSION"
        title="Dispersion chromatique & biréfringence — kernels audités, corrections incluses"
        subtitle="Sellmeier analytique (dérivées 1–3 exactes) · GDD validé (36.2 fs²/mm @800 nm, publié ≈36) · TOD réparé (le rapport source codait 0 « identiquement nul » et oubliait le facteur µm→mm)"
        right={
          <Segmented
            options={GLASSES.map((g) => ({ id: g.id, label: g.label }))}
            value={GLASSES.some((g) => g.id === gid) ? gid : 'silice'}
            onChange={setGid}
          />
        }
      >
        <Slider label="λ₀ analyse" value={lambda0} min={0.3} max={3} step={0.01} onChange={(v) => setL0str(String(v))} format={(v) => v.toFixed(2)} unit=" µm" />
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel tag="COURBES" title="n(λ) · GDD(λ) · TOD(λ)" flush>
          <DispersionCurves glass={glass} lambda0={lambda0} />
        </Panel>
        <PulsePanel glass={glass} lambda0={lambda0} />
      </div>

      <BiaxialPanel />
    </div>
  );
}
