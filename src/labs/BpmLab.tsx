import { useMemo, useState, useRef, useEffect } from 'react';
import {
  DEFAULT_BPM, DEVICE_INFO, runBPM, solveMode, slabNeffAnalytic,
  type BpmConfig, type DeviceId,
} from '../physics/bpm';
import {
  Panel, Stat, Slider, Segmented, Btn, Tag, Formula,
  useCanvas, LUT, paintField, drawImageBuffer, drawGrid, drawSeries, axisLabel,
} from '../ui/kit';

type ViewMode = 'intensity' | 'wave' | 'phase' | 'index';

const DEVICES: DeviceId[] = ['coupler', 'mzi', 'ybranch', 'mmi', 'grin', 'straight', 'grating', 'disorder', 'freespace'];

export default function BpmLab() {
  const [device, setDevice] = useState<DeviceId>('coupler');
  const [view, setView] = useState<ViewMode>('intensity');
  const [animate, setAnimate] = useState(true);
  const [lambda, setLambda] = useState(1.55);
  const [dn, setDn] = useState(0.012);
  const [wCore, setWCore] = useState(4);
  const [gap, setGap] = useState(3);
  const [Lz, setLz] = useState(3000);
  const [armPhase, setArmPhase] = useState(0);
  const [Nx, setNx] = useState(512);
  const [quality, setQuality] = useState(900);
  const [tick, setTick] = useState(0);

  const cfg: BpmConfig = useMemo(() => ({
    ...DEFAULT_BPM,
    device, lambda, dn, wCore, gap, Lz, armPhase, Nx, Nz: quality,
    sep: device === 'mmi' ? 10 : 12,
    launch: device === 'freespace' || device === 'grin' || device === 'disorder' ? 'gauss' : 'mode',
    Lx: device === 'freespace' ? 200 : 80,
    w0: device === 'freespace' ? 8 : 3,
  }), [device, lambda, dn, wCore, gap, Lz, armPhase, Nx, quality]);

  const result = useMemo(() => runBPM(cfg), [cfg]);

  // The mode solver gets its own well-resolved grid: the propagation window is
  // sized for the device, which would under-resolve the core walls (we need
  // edge >= ~2*dx for the tanh profile to be band-limited).
  const modeCfg: BpmConfig = useMemo(() => {
    const Lxm = Math.max(40, wCore * 10);
    const Nxm = 1024;
    const dx = Lxm / Nxm;
    return { ...DEFAULT_BPM, device: 'straight', lambda, dn, wCore, Lx: Lxm, Nx: Nxm, edge: Math.max(0.12, 2 * dx), w0: wCore / 2 };
  }, [lambda, dn, wCore]);

  const mode = useMemo(() => solveMode(modeCfg, 600, 1e-12), [modeCfg]);
  const neffExact = useMemo(
    () => slabNeffAnalytic(modeCfg.lambda, modeCfg.n0 + modeCfg.dn, modeCfg.n0, modeCfg.wCore),
    [modeCfg],
  );

  // animation clock
  useEffect(() => {
    if (!animate || view !== 'wave') return;
    let id = 0;
    const loop = () => { setTick((t) => t + 1); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [animate, view]);

  const waveBuf = useRef<Float32Array>(new Float32Array(0));

  // sqrt(I) is needed every animation frame — compute it once per solve
  const amplitude = useMemo(() => {
    const a = new Float32Array(result.intensity.length);
    for (let i = 0; i < a.length; i++) a[i] = Math.sqrt(result.intensity[i]);
    return a;
  }, [result]);

  // core boundary polyline (Δn = 0.5 crossing), extracted once per solve
  const contour = useMemo(() => {
    const { NzOut, Nx: nx, index } = result;
    const pts: number[] = [];
    const step = Math.max(1, Math.floor(NzOut / 240));
    for (let r = 0; r < NzOut; r += step) {
      for (let c = 1; c < nx; c++) {
        if ((index[r * nx + c - 1] < 0.5) !== (index[r * nx + c] < 0.5)) {
          pts.push(r / Math.max(NzOut - 1, 1), c / Math.max(nx - 1, 1));
        }
      }
    }
    return new Float32Array(pts);
  }, [result]);

  const field = useCanvas((ctx, w, h) => {
    const { NzOut, Nx: nx } = result;
    const pad = { l: 34, r: 12, t: 10, b: 22 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    if (iw <= 4 || ih <= 4) return;

    const img = ctx.createImageData(NzOut, nx);

    if (view === 'wave') {
      if (waveBuf.current.length !== NzOut * nx) waveBuf.current = new Float32Array(NzOut * nx);
      const buf = waveBuf.current;
      const phi = animate ? (tick * 0.18) % (Math.PI * 2) : 0;
      for (let i = 0; i < buf.length; i++) {
        buf[i] = amplitude[i] * Math.cos(result.phase[i] + phi);
      }
      paintField(img, buf, nx, NzOut, LUT.diverging, -0.85, 0.85, 1, true);
    } else if (view === 'phase') {
      paintField(img, result.phase, nx, NzOut, LUT.twilight, -Math.PI, Math.PI, 1, true);
    } else if (view === 'index') {
      paintField(img, result.index, nx, NzOut, LUT.viridis, 0, 1, 1, true);
    } else {
      paintField(img, result.intensity, nx, NzOut, LUT.inferno, 0, 1, 0.45, true);
    }

    drawImageBuffer(ctx, img, pad.l, pad.t, iw, ih, true);

    // core boundary overlay (precomputed crossings, drawn as a point cloud)
    if (view !== 'index' && contour.length > 0) {
      ctx.save();
      ctx.fillStyle = 'rgba(125,211,252,0.28)';
      for (let i = 0; i < contour.length; i += 2) {
        ctx.fillRect(pad.l + contour[i] * iw, pad.t + contour[i + 1] * ih, 1.1, 1.1);
      }
      ctx.restore();
    }

    // frame + axes
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, iw - 1, ih - 1);
    axisLabel(ctx, `z → ${cfg.Lz.toFixed(0)} µm`, pad.l + iw, h - 6, 'right');
    axisLabel(ctx, `0`, pad.l, h - 6, 'left');
    ctx.save();
    ctx.translate(11, pad.t + ih / 2);
    ctx.rotate(-Math.PI / 2);
    axisLabel(ctx, `x  ±${(cfg.Lx / 2).toFixed(0)} µm`, 0, 0, 'center');
    ctx.restore();
  }, [result, view, tick, animate, cfg.Lz, cfg.Lx], 0.42);

  const profiles = useCanvas((ctx, w, h) => {
    const pad = { l: 30, r: 8, t: 12, b: 16 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 6, 3);

    let wMax = 0;
    for (let i = 0; i < result.width.length; i++) wMax = Math.max(wMax, result.width[i]);
    drawSeries(ctx, result.width, pad.l, pad.t, iw, ih, 0, wMax * 1.1, 'rgba(167,139,250,0.95)', 'rgba(167,139,250,0.10)');

    let pMax = 0;
    for (let i = 0; i < result.power.length; i++) pMax = Math.max(pMax, result.power[i]);
    drawSeries(ctx, result.power, pad.l, pad.t, iw, ih, 0, pMax * 1.1, 'rgba(103,232,249,0.95)');

    axisLabel(ctx, 'largeur RMS', pad.l + 3, pad.t + 10, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, 'puissance', pad.l + 3, pad.t + 21, 'left', 'rgba(103,232,249,0.9)');
    axisLabel(ctx, `${wMax.toFixed(1)} µm`, pad.l + iw, pad.t + 10, 'right', 'rgba(148,163,184,0.6)');
    axisLabel(ctx, 'z', pad.l + iw, h - 4, 'right');
  }, [result], 0.34);

  const outProfile = useCanvas((ctx, w, h) => {
    const pad = { l: 8, r: 8, t: 12, b: 16 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 4, 3);
    const nx = result.Nx;
    const last = (result.NzOut - 1) * nx;
    const slice = result.intensity.subarray(last, last + nx);
    let m = 1e-9;
    for (let i = 0; i < nx; i++) m = Math.max(m, slice[i]);
    drawSeries(ctx, slice, pad.l, pad.t, iw, ih, 0, m * 1.15, 'rgba(251,191,36,0.95)', 'rgba(251,191,36,0.12)');

    // index overlay
    const idxSlice = result.index.subarray(last, last + nx);
    drawSeries(ctx, idxSlice, pad.l, pad.t, iw, ih, 0, 1.6, 'rgba(56,189,248,0.35)', undefined, 1);
    ctx.beginPath();
    ctx.moveTo(pad.l + iw / 2, pad.t); ctx.lineTo(pad.l + iw / 2, pad.t + ih);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
    axisLabel(ctx, 'I(x) en sortie', pad.l + 2, pad.t + 9, 'left', 'rgba(251,191,36,0.9)');
  }, [result], 0.34);

  const modeCanvas = useCanvas((ctx, w, h) => {
    const pad = { l: 8, r: 8, t: 12, b: 12 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    drawGrid(ctx, pad.l, pad.t, iw, ih, 4, 2);
    let m = 1e-9;
    for (let i = 0; i < mode.re.length; i++) m = Math.max(m, Math.abs(mode.re[i]));
    drawSeries(ctx, mode.re, pad.l, pad.t, iw, ih, -m * 0.2, m * 1.1, 'rgba(52,211,153,0.95)', 'rgba(52,211,153,0.12)');
    axisLabel(ctx, 'mode fondamental (distance imaginaire)', pad.l + 2, pad.t + 9, 'left', 'rgba(52,211,153,0.9)');
  }, [mode], 0.3);

  const neffErr = Math.abs(mode.neff - neffExact);
  const info = DEVICE_INFO[device];

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_310px]">
      <div className="space-y-3">
        <Panel
          tag="BPM"
          title={`Propagation — ${info.name}`}
          subtitle={info.blurb}
          right={
            <div className="flex items-center gap-1.5">
              <Segmented
                value={view}
                onChange={setView}
                options={[
                  { id: 'intensity', label: '|A|²' },
                  { id: 'wave', label: 'Re(E)' },
                  { id: 'phase', label: 'arg A' },
                  { id: 'index', label: 'Δn' },
                ]}
              />
              {view === 'wave' && (
                <Btn active={animate} onClick={() => setAnimate((a) => !a)}>
                  {animate ? '⏸' : '▶'}
                </Btn>
              )}
            </div>
          }
          flush
        >
          <div className="relative w-full">
            <canvas ref={field.ref} style={{ width: '100%', height: field.size.h }} className="block" />
          </div>
        </Panel>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Temps de calcul" value={result.ms.toFixed(1)} unit="ms" tone="cyan" />
          <Stat label="Débit" value={result.throughputMcs.toFixed(0)} unit="Mcell·pas/s" tone="violet"
            hint={`${result.megaCellSteps.toFixed(2)} M cellules-pas`} />
          <Stat label="Dérive d'énergie" value={result.energyDrift.toExponential(1)} tone={result.energyDrift < 1e-3 ? 'green' : 'amber'}
            hint={cfg.absorber ? 'absorbeur actif' : 'sans absorbeur'} />
          <Stat label="Port gauche" value={(result.portLeft * 100).toFixed(1)} unit="%" tone="amber" />
          <Stat label="Port droit" value={(result.portRight * 100).toFixed(1)} unit="%" tone="amber" />
          <Stat label="Pas z" value={(cfg.Lz / cfg.Nz).toFixed(2)} unit="µm" tone="slate" hint={`${cfg.Nz} pas`} />
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <Panel tag="Z" title="Puissance & largeur du faisceau" flush>
            <div className="w-full"><canvas ref={profiles.ref} style={{ width: '100%', height: profiles.size.h }} className="block" /></div>
          </Panel>
          <Panel tag="OUT" title="Profil transverse en sortie" flush>
            <div className="w-full"><canvas ref={outProfile.ref} style={{ width: '100%', height: outProfile.size.h }} className="block" /></div>
          </Panel>
        </div>

        <Panel
          tag="MODE"
          title="Solveur de mode par BPM à distance imaginaire"
          subtitle="z → −iτ transforme le propagateur en contraction : le vecteur propre dominant est le mode fondamental."
          right={<Tag tone={neffErr / dn < 0.02 ? 'green' : 'amber'}>écart {((neffErr / dn) * 100).toFixed(2)}% du contraste</Tag>}
        >
          <div className="w-full"><canvas ref={modeCanvas.ref} style={{ width: '100%', height: modeCanvas.size.h }} className="block" /></div>
          <div className="mt-2.5 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="n_eff (BPM)" value={mode.neff.toFixed(6)} tone="green" />
            <Stat label="n_eff (exact)" value={neffExact.toFixed(6)} tone="cyan" hint="u·tan u = √(V²−u²)" />
            <Stat label="Écart absolu" value={neffErr.toExponential(2)} tone="violet" />
            <Stat label="Convergence" value={mode.iterations} unit="itér." tone="slate" hint={`résidu ${mode.residual.toExponential(1)} · ${mode.ms.toFixed(0)} ms`} />
          </div>
        </Panel>
      </div>

      {/* ------------------------------- controls ------------------------------- */}
      <div className="space-y-3">
        <Panel tag="DUT" title="Composant sous test">
          <div className="grid grid-cols-3 gap-1">
            {DEVICES.map((d) => (
              <Btn key={d} active={device === d} onClick={() => setDevice(d)} className="!text-[9px]">
                {DEVICE_INFO[d].name.split(' ')[0]}
              </Btn>
            ))}
          </div>
        </Panel>

        <Panel tag="PARAM" title="Paramètres physiques">
          <div className="space-y-3">
            <Slider label="Longueur d'onde λ" value={lambda} min={1.2} max={1.7} step={0.01} onChange={setLambda} format={(v) => v.toFixed(2)} unit=" µm" />
            <Slider label="Contraste d'indice Δn" value={dn} min={0.002} max={0.08} step={0.001} onChange={setDn} format={(v) => v.toFixed(3)} />
            <Slider label="Largeur de cœur" value={wCore} min={1} max={10} step={0.1} onChange={setWCore} format={(v) => v.toFixed(1)} unit=" µm" />
            <Slider label="Espacement (coupleur)" value={gap} min={0.5} max={10} step={0.1} onChange={setGap} format={(v) => v.toFixed(1)} unit=" µm" />
            <Slider label="Longueur de propagation" value={Lz} min={200} max={8000} step={100} onChange={setLz} format={(v) => v.toFixed(0)} unit=" µm" />
            {device === 'mzi' && (
              <Slider label="Déphasage du bras Δφ" value={armPhase} min={0} max={6.2832} step={0.02} onChange={setArmPhase}
                format={(v) => `${(v / Math.PI).toFixed(2)}π`} />
            )}
          </div>
        </Panel>

        <Panel tag="GRID" title="Discrétisation" subtitle="Compromis vitesse / précision — recalculé en direct.">
          <div className="space-y-3">
            <Slider label="Points transverses Nx" value={Nx} min={128} max={2048} step={128}
              onChange={(v) => setNx(1 << Math.round(Math.log2(v)))} format={(v) => `${v}`} />
            <Slider label="Pas longitudinaux Nz" value={quality} min={200} max={2400} step={100} onChange={setQuality} format={(v) => `${v}`} />
          </div>
          <div className="mt-3 space-y-1.5 text-[10px] leading-relaxed text-slate-500">
            <div>Schéma : <Formula>e^(iΔz∇²/4k) · e^(ik₀Δn Δz) · e^(iΔz∇²/4k)</Formula></div>
            <div>Découpage de Strang, précision <span className="text-slate-300">O(Δz²)</span>. Diffraction diagonale en Fourier, indice diagonal en réel.</div>
          </div>
        </Panel>

        {device === 'mzi' && (
          <Panel tag="MZI" title="Fonction de transfert">
            <div className="text-[10px] leading-relaxed text-slate-400">
              Le combineur en Y ne reguide que le mode symétrique : le mode antisymétrique rayonne.
              L'extinction observée est donc un <span className="text-cyan-300">résultat de simulation</span>, pas une formule cos².
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <Stat label="Transmission" value={((result.portLeft + result.portRight) * 100).toFixed(1)} unit="%" tone="cyan" />
              <Stat label="Δφ appliqué" value={`${(armPhase / Math.PI).toFixed(2)}π`} tone="violet" />
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}
