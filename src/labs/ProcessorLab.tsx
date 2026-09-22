import { useMemo, useState, useEffect } from 'react';
import {
  haarUnitary, dftMatrix, hadamard, decomposeMesh, reconstruct,
  frobeniusDiff, unitarityDefect, propagate, matvec, parseMatrixText, type CMat,
} from '../physics/mzi';
import { HW_PRESETS, meshPerformance } from '../physics/latency';
import { measureCpuMatvec } from '../physics/validate';
import {
  Panel, Stat, Slider, Segmented, Btn, Tag, Formula,
  useCanvas, LUT, paintField, drawImageBuffer, axisLabel,
} from '../ui/kit';

type UKind = 'haar' | 'dft' | 'hadamard' | 'custom';

export default function ProcessorLab() {
  const [n, setN] = useState(10);
  const [kind, setKind] = useState<UKind>('haar');
  const [seed, setSeed] = useState(1234);
  const [inPort, setInPort] = useState(0);
  const [spread, setSpread] = useState(false);
  const [hw, setHw] = useState<keyof typeof HW_PRESETS>('mems');
  const [tick, setTick] = useState(0);
  const [flow, setFlow] = useState(true);
  const [customText, setCustomText] = useState('');

  // matrice collée : parsée une fois par texte ; fallback Haar si invalide
  const custom = useMemo(() => parseMatrixText(customText), [customText]);

  useEffect(() => {
    if (!flow) return;
    let id = 0;
    const loop = () => { setTick((t) => t + 1); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [flow]);

  const size = kind === 'hadamard' ? 1 << Math.round(Math.log2(n)) : n;

  const U: CMat = useMemo(() => {
    if (kind === 'custom' && custom) return custom;
    if (kind === 'dft') return dftMatrix(size);
    if (kind === 'hadamard') return hadamard(size);
    return haarUnitary(size, seed);
  }, [kind, custom, size, seed]);

  // honnêteté : le défaut d'unitarité de la matrice BRUTE collée
  const rawDefect = useMemo(
    () => (kind === 'custom' && custom ? unitarityDefect(custom) : 0),
    [kind, custom],
  );

  const decomp = useMemo(() => decomposeMesh(U), [U]);
  const rebuilt = useMemo(() => reconstruct(decomp), [decomp]);
  const reckErr = useMemo(() => frobeniusDiff(U, rebuilt), [U, rebuilt]);
  const unitErr = useMemo(() => unitarityDefect(rebuilt), [rebuilt]);

  const input = useMemo(() => {
    const re = new Float64Array(size), im = new Float64Array(size);
    if (spread) {
      for (let i = 0; i < size; i++) {
        const u = (i - (size - 1) / 2) / (size / 3);
        re[i] = Math.exp(-u * u);
      }
    } else {
      re[Math.min(inPort, size - 1)] = 1;
    }
    let p = 0;
    for (let i = 0; i < size; i++) p += re[i] * re[i] + im[i] * im[i];
    const s = 1 / Math.sqrt(p || 1);
    for (let i = 0; i < size; i++) { re[i] *= s; im[i] *= s; }
    return { re, im };
  }, [size, inPort, spread]);

  const trace = useMemo(() => propagate(decomp, input.re, input.im), [decomp, input]);
  const ref = useMemo(() => matvec(U, input.re, input.im), [U, input]);

  const mvErr = useMemo(() => {
    let e = 0, nrm = 0;
    for (let i = 0; i < size; i++) {
      e += (trace.outRe[i] - ref.re[i]) ** 2 + (trace.outIm[i] - ref.im[i]) ** 2;
      nrm += ref.re[i] ** 2 + ref.im[i] ** 2;
    }
    return Math.sqrt(e / Math.max(nrm, 1e-300));
  }, [trace, ref, size]);

  const perf = useMemo(
    () => meshPerformance(size, decomp.depth, decomp.units.length, HW_PRESETS[hw].p),
    [size, decomp, hw],
  );
  const cpu = useMemo(() => measureCpuMatvec(size, 1500), [size]);

  // ------------------------------------------------------------------ mesh
  const mesh = useCanvas((ctx, w, h) => {
    const pad = { l: 46, r: 58, t: 18, b: 18 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 10 || ih <= 10) return;

    const segs = decomp.depth + 1;
    const xOf = (s: number) => pad.l + (s / segs) * iw;
    const yOf = (i: number) => pad.t + ((i + 0.5) / size) * ih;
    const dash = (tick * 0.9) % 14;

    // waveguide segments, brightness = optical power
    for (let s = 0; s <= decomp.depth; s++) {
      for (let i = 0; i < size; i++) {
        const p = trace.stages[Math.min(s + 1, trace.nStages - 1) * size + i];
        const a = Math.min(1, Math.sqrt(Math.max(p, 0)) * 1.25);
        const x0 = xOf(s), x1 = xOf(s + 1);
        const y = yOf(i);
        ctx.beginPath();
        ctx.moveTo(x0, y); ctx.lineTo(x1, y);
        ctx.strokeStyle = `rgba(56,189,248,${0.06 + a * 0.10})`;
        ctx.lineWidth = 7;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x0, y); ctx.lineTo(x1, y);
        ctx.strokeStyle = `rgba(125,211,252,${0.10 + a * 0.85})`;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        if (flow && a > 0.06) {
          ctx.beginPath();
          ctx.moveTo(x0, y); ctx.lineTo(x1, y);
          ctx.setLineDash([2.5, 11]);
          ctx.lineDashOffset = -dash;
          ctx.strokeStyle = `rgba(255,255,255,${a * 0.85})`;
          ctx.lineWidth = 2.2;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // MZI cells
    for (const u of decomp.units) {
      const x = xOf(u.layer) + (iw / segs) * 0.5;
      const y0 = yOf(u.m), y1 = yOf(u.m + 1);
      const split = Math.cos(u.theta) ** 2;
      const cw = Math.min(18, (iw / segs) * 0.62);
      const chn = Math.abs(y1 - y0);
      ctx.save();
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') ctx.roundRect(x - cw / 2, y0 - 3, cw, chn + 6, 4);
      else ctx.rect(x - cw / 2, y0 - 3, cw, chn + 6);
      ctx.fillStyle = `hsla(${190 + split * 90}, 85%, ${22 + split * 26}%, 0.95)`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.16)';
      ctx.lineWidth = 0.8;
      ctx.stroke();
      // phase shifter tick
      ctx.beginPath();
      ctx.arc(x, y0, 1.6, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${45 + (u.phi / (2 * Math.PI)) * 260}, 95%, 62%, 0.95)`;
      ctx.fill();
      ctx.restore();
    }

    // port labels
    ctx.font = '9px ui-monospace, monospace';
    for (let i = 0; i < size; i++) {
      const y = yOf(i);
      axisLabel(ctx, `in ${i}`, pad.l - 6, y + 3, 'right');
      const p = trace.stages[(trace.nStages - 1) * size + i];
      const pct = (p * 100).toFixed(0);
      ctx.fillStyle = p > 0.02 ? 'rgba(103,232,249,0.95)' : 'rgba(100,116,139,0.6)';
      ctx.textAlign = 'left';
      ctx.fillText(`${pct}%`, pad.l + iw + 6, y + 3);
      // output bar
      ctx.fillStyle = 'rgba(103,232,249,0.25)';
      ctx.fillRect(pad.l + iw + 34, y - 3, Math.max(1, p * 20), 6);
    }
    axisLabel(ctx, `${decomp.depth} couches · ${decomp.units.length} MZI`, pad.l, 11, 'left', 'rgba(148,163,184,0.7)');
  }, [decomp, trace, tick, flow, size], 0.5);

  // ------------------------------------------------------------- matrix map
  const matrix = useCanvas((ctx, w, h) => {
    const s = Math.min(w - 20, h - 20);
    if (s <= 4) return;
    const mag = new Float32Array(size * size);
    let mx = 1e-9;
    for (let i = 0; i < size * size; i++) {
      mag[i] = Math.hypot(U.re[i], U.im[i]);
      mx = Math.max(mx, mag[i]);
    }
    const img = ctx.createImageData(size, size);
    paintField(img, mag, size, size, LUT.ice, 0, mx, 0.7, false);
    const x0 = (w - s) / 2, y0 = (h - s) / 2;
    drawImageBuffer(ctx, img, x0, y0, s, s, false);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);
    axisLabel(ctx, `|U| ${size}×${size}`, x0, y0 - 4, 'left');
  }, [U, size], 0.85);

  // ------------------------------------------------------------ output bars
  const outputs = useCanvas((ctx, w, h) => {
    const pad = { l: 26, r: 8, t: 14, b: 18 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    if (iw <= 4) return;
    const bw = iw / size;
    for (let i = 0; i < size; i++) {
      const pMesh = trace.outRe[i] ** 2 + trace.outIm[i] ** 2;
      const pRef = ref.re[i] ** 2 + ref.im[i] ** 2;
      const x = pad.l + i * bw;
      ctx.fillStyle = 'rgba(103,232,249,0.75)';
      ctx.fillRect(x + bw * 0.12, pad.t + ih - pMesh * ih, bw * 0.5, pMesh * ih);
      ctx.strokeStyle = 'rgba(251,191,36,0.9)';
      ctx.lineWidth = 1.5;
      const yr = pad.t + ih - pRef * ih;
      ctx.beginPath();
      ctx.moveTo(x + bw * 0.1, yr); ctx.lineTo(x + bw * 0.78, yr);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ih); ctx.lineTo(pad.l + iw, pad.t + ih);
    ctx.stroke();
    axisLabel(ctx, 'maillage optique', pad.l + 2, pad.t + 8, 'left', 'rgba(103,232,249,0.9)');
    axisLabel(ctx, '— référence U·v', pad.l + iw, pad.t + 8, 'right', 'rgba(251,191,36,0.9)');
    axisLabel(ctx, 'port de sortie', pad.l + iw / 2, h - 4, 'center');
  }, [trace, ref, size], 0.4);

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_310px]">
      <div className="space-y-3">
        <Panel
          tag="MESH"
          title="Maillage MZI programmable"
          subtitle="Décomposition de Givens complexe : T_K···T_1·U = D. La lumière traverse D puis les MZI dans l'ordre inverse."
          right={
            <div className="flex gap-1.5">
              <Btn active={flow} onClick={() => setFlow((f) => !f)}>{flow ? '⏸ flux' : '▶ flux'}</Btn>
            </div>
          }
          flush
        >
          <div className="w-full"><canvas ref={mesh.ref} style={{ width: '100%', height: mesh.size.h }} className="block" /></div>
        </Panel>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="‖U − Û‖_F" value={reckErr.toExponential(2)} tone={reckErr < 1e-12 ? 'green' : 'rose'}
            hint="reconstruction du maillage" />
          <Stat label="‖Û†Û − I‖_F" value={unitErr.toExponential(2)} tone={unitErr < 1e-12 ? 'green' : 'rose'} hint="unitarité / passivité" />
          <Stat label="Erreur MV optique" value={mvErr.toExponential(2)} tone={mvErr < 1e-12 ? 'green' : 'rose'} hint="vs produit matriciel dense" />
          <Stat label="Énergie" value={(trace.outputPower / trace.inputPower).toFixed(12)} tone="cyan" hint="P_out / P_in" />
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[240px_minmax(0,1fr)]">
          <Panel tag="U" title="Matrice cible" flush>
            <div className="w-full"><canvas ref={matrix.ref} style={{ width: '100%', height: matrix.size.h }} className="block" /></div>
          </Panel>
          <Panel tag="OUT" title="Puissance de sortie : maillage physique vs algèbre linéaire" flush>
            <div className="w-full"><canvas ref={outputs.ref} style={{ width: '100%', height: outputs.size.h }} className="block" /></div>
          </Panel>
        </div>

        <Panel
          tag="HW"
          title="Modèle de performance dérivé de la physique"
          subtitle={HW_PRESETS[hw].note}
          right={
            <Segmented
              value={hw}
              onChange={(v) => setHw(v as keyof typeof HW_PRESETS)}
              options={Object.entries(HW_PRESETS).map(([k, v]) => ({ id: k, label: v.label.split(' ')[0] }))}
            />
          }
        >
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Temps de vol" value={perf.timeOfFlightPs.toFixed(1)} unit="ps" tone="cyan" />
            <Stat label="Latence bout-en-bout" value={(perf.totalLatencyPs / 1000).toFixed(2)} unit="ns" tone="violet"
              hint="dominée par la conversion E/O" />
            <Stat label="Débit crête" value={perf.peakTflops.toFixed(2)} unit="TFLOP/s" tone="amber" />
            <Stat label="Efficacité" value={perf.tflopsPerWatt.toFixed(1)} unit="TFLOP/s/W" tone="green" />
            <Stat label="Longueur optique" value={perf.opticalLengthMm.toFixed(2)} unit="mm" tone="slate" />
            <Stat label="Pertes d'insertion" value={perf.insertionLossDb.toFixed(2)} unit="dB" tone={perf.insertionLossDb > 6 ? 'rose' : 'slate'}
              hint={`T = ${(perf.transmission * 100).toFixed(1)}%`} />
            <Stat label="Puissance statique" value={perf.staticPowerMw.toFixed(1)} unit="mW" tone="slate" />
            <Stat label="Baseline CPU (mesurée)" value={cpu.nsPerOp.toFixed(0)} unit="ns/op" tone="rose"
              hint={`${cpu.gflops.toFixed(2)} GFLOP/s en JS`} />
          </div>

          <div className="mt-3 overflow-hidden rounded-lg ring-1 ring-white/6">
            <table className="w-full text-left font-mono text-[10px]">
              <tbody>
                {perf.formulas.map((f, i) => (
                  <tr key={f.label} className={i % 2 ? 'bg-white/[0.015]' : ''}>
                    <td className="px-2.5 py-1.5 text-slate-400">{f.label}</td>
                    <td className="px-2.5 py-1.5 text-cyan-300/70">{f.expr}</td>
                    <td className="px-2.5 py-1.5 text-right text-slate-200">{f.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            La partie optique est limitée par le temps de vol (<span className="text-cyan-300">{perf.timeOfFlightPs.toFixed(0)} ps</span>),
            mais la latence système reste dominée par les convertisseurs. C'est précisément pourquoi la revendication
            tenable est <span className="text-slate-300">sub-microseconde</span>, pas sub-nanoseconde.
          </p>
        </Panel>
      </div>

      {/* controls */}
      <div className="space-y-3">
        <Panel tag="U" title="Unitaire cible">
          <Segmented
            className="mb-3"
            value={kind}
            onChange={setKind}
            options={[
              { id: 'haar', label: 'Haar' },
              { id: 'dft', label: 'DFT' },
              { id: 'hadamard', label: 'Hadamard' },
              { id: 'custom', label: '★ Perso', title: 'collez votre matrice (réelle ou complexe a+bi)' },
            ]}
          />
          {kind === 'custom' && (
            <div className="mb-3">
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                spellCheck={false}
                placeholder={'Matrice carrée n×n — ex. Hadamard 4×4 :\n1 1 1 1\n1 -1 1 -1\n1 1 -1 -1\n1 -1 -1 1\n(ou complexe : 0.7i, 1+2i…)'}
                className="h-32 w-full resize-y rounded-lg border border-white/8 bg-black/40 p-2.5 font-mono text-[10px] leading-relaxed text-cyan-100/90 outline-none focus:border-cyan-400/30"
              />
              <div className="mt-1 flex flex-wrap items-center gap-2 text-[9px]">
                {custom ? (
                  <>
                    <Tag tone="green">{`valide · ${custom.n}×${custom.n}`}</Tag>
                    {rawDefect > 1e-6
                      ? <Tag tone="amber">{`non unitaire · ‖U†U−I‖=${rawDefect.toFixed(3)} — le maillage réalisera l'unitaire la plus proche`}</Tag>
                      : <Tag tone="cyan">unitaire ✓</Tag>}
                  </>
                ) : customText.trim() ? (
                  <Tag tone="rose">matrice invalide (carrée, 2–40, tokens réels/complexes)</Tag>
                ) : (
                  <span className="text-slate-500">collez une matrice ci-dessus</span>
                )}
              </div>
            </div>
          )}
          {kind !== 'custom' && (
            <Slider label="Nombre de modes N" value={n} min={3} max={18} step={1} onChange={setN} format={(v) => `${v}`} />
          )}
          {kind === 'haar' && (
            <div className="mt-3">
              <Btn onClick={() => setSeed((s) => s + 1)} tone="primary" className="w-full">↻ Nouvelle unitaire aléatoire</Btn>
            </div>
          )}
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Stat label="MZI requis" value={decomp.units.length} tone="cyan" hint="N(N−1)/2" />
            <Stat label="Profondeur" value={decomp.depth} tone="violet" hint="couches physiques" />
          </div>
        </Panel>

        <Panel tag="IN" title="Excitation d'entrée">
          <Segmented
            className="mb-3"
            value={spread ? 'gauss' : 'single'}
            onChange={(v) => setSpread(v === 'gauss')}
            options={[{ id: 'single', label: 'Port unique' }, { id: 'gauss', label: 'Gaussienne' }]}
          />
          {!spread && (
            <Slider label="Port injecté" value={inPort} min={0} max={size - 1} step={1} onChange={setInPort} format={(v) => `#${v}`} />
          )}
          <div className="mt-3 text-[10px] leading-relaxed text-slate-500">
            Chaque cellule mélange deux guides adjacents :
            <div className="mt-1"><Formula>T = [[e^iφ cosθ, −sinθ], [e^iφ sinθ, cosθ]]</Formula></div>
            <div className="mt-1.5">La couleur d'une cellule code le taux de division <span className="text-slate-300">cos²θ</span>,
              le point coloré code la phase <span className="text-slate-300">φ</span>.</div>
          </div>
        </Panel>

        <Panel tag="THÉORÈME" title="Universalité linéaire">
          <p className="text-[10px] leading-relaxed text-slate-400">
            Reck (1994) puis Clements (2016) : <span className="text-slate-200">toute</span> transformation unitaire N×N
            est réalisable exactement par N(N−1)/2 interféromètres Mach-Zehnder et une couche de déphaseurs.
          </p>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-400">
            Ici la décomposition est calculée par élimination de Givens complexe et <span className="text-emerald-300">vérifiée
            numériquement</span> : la reconstruction retombe sur la cible à {reckErr.toExponential(1)} près en norme de Frobenius.
          </p>
          <div className="mt-2 flex flex-wrap gap-1">
            <Tag tone={reckErr < 1e-12 ? 'green' : 'rose'}>exactitude machine</Tag>
            <Tag tone="cyan">passif</Tag>
            <Tag tone="violet">O(N²) cellules</Tag>
          </div>
        </Panel>
      </div>
    </div>
  );
}
