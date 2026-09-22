import { useState } from 'react';
import {
  SPEAR_SPECS, SPEAR_ORDER, spearAccuracy, spearBench,
  type SpearFnId,
} from '../physics/spear';
import { Panel, Stat, Btn, Tag, Segmented, Formula, useCanvas, drawGrid, drawSeries } from '../ui/kit';

const N_PLOT = 256;

export default function SpearLab() {
  const [id, setId] = useState<SpearFnId>('gelu');
  const [bench, setBench] = useState<ReturnType<typeof spearBench> | null>(null);

  const spec = SPEAR_SPECS[id];
  const acc = spearAccuracy(spec);

  const run = () => setBench(spearBench(spec));

  const xs: number[] = [];
  const ye: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < N_PLOT; i++) {
    const x = -4 + (8 * i) / (N_PLOT - 1);
    xs.push(x);
    ye.push(spec.exact(x));
    ys.push(spec.spear(x));
  }

  const { ref } = useCanvas((ctx, w, h) => {
    const pad = { l: 30, r: 10, t: 12, b: 18 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const ymin = Math.min(...ye, ...ys, 0);
    const ymax = Math.max(...ye, ...ys, 1);
    drawGrid(ctx, pad.l, pad.t, pw, ph, 6, 4);
    drawSeries(ctx, ye, pad.l, pad.t, pw, ph, ymin, ymax, 'rgba(34,211,238,0.85)', 'rgba(34,211,238,0.06)');
    drawSeries(ctx, ys, pad.l, pad.t, pw, ph, ymin, ymax, 'rgba(167,139,250,0.9)', undefined, 1.2);
    ctx.strokeStyle = 'rgba(251,146,60,0.5)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let i = 0; i < N_PLOT; i++) {
      const px = pad.l + (i / (N_PLOT - 1)) * pw;
      const py = pad.t + ph - ((ye[i] - ys[i] - ymin) / (ymax - ymin)) * ph;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }, [id]);

  return (
    <div className="space-y-3">
      <Panel
        tag="SPEAR"
        title="Kernels symboliques v3 — 10 primitives LLM distillées en algèbre pure"
        subtitle="Pareto-GP + seeds structurés (rationnels / Pade / Taylor) + raffinement de constantes. Mesuré dans votre navigateur."
        right={
          <Btn tone="primary" onClick={run}>
            {bench ? '↻ Re-benchmark' : '▶ Benchmark'}
          </Btn>
        }
      >
        <Segmented
          options={SPEAR_ORDER.map((k) => ({ id: k, label: SPEAR_SPECS[k].label }))}
          value={id}
          onChange={setId}
        />

        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Fonction exacte" value={<Formula>{spec.exactExpr}</Formula>} tone="cyan" />
          <Stat label="Formule SPEAR" value={<Formula>{spec.formula}</Formula>} tone="violet" />
          <Stat label="Domaine" value={spec.domain} tone="slate" hint={`${spec.flops} FLOPs contre ~40 pour exp/erf`} />
          <Stat
            label="Accélération"
            value={bench ? `${bench.speedup.toFixed(1)}×` : '—'}
            tone={bench && bench.speedup > 3 ? 'green' : 'amber'}
            hint={bench ? `${bench.elements.toLocaleString('fr-FR')} éch. × ${bench.iterations} it.` : 'lancez le benchmark'}
          />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
          <Stat label="MSE (domaine)" value={acc.mse.toExponential(1)} tone="slate" />
          <Stat label="Erreur L∞" value={acc.linf.toFixed(4)} tone={acc.linf < 1e-3 ? 'green' : 'amber'} />
          <Stat
            label="Exact vs SPEAR"
            value={bench ? `${bench.exactMs.toFixed(1)} / ${bench.spearMs.toFixed(1)}` : '—'}
            unit="ms"
            tone="violet"
            hint="exacte / kernel SPEAR"
          />
        </div>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[1fr_1fr]">
        <Panel tag="CURVES" title="Courbes — cyan exacte, violette SPEAR, orange erreur (sur [−4,4])" flush>
          <canvas ref={ref} className="block h-full w-full" style={{ aspectRatio: '2 / 1' }} />
        </Panel>

        <Panel
          tag="C99"
          title={`Kernel C99 embarqué — spear_${id}`}
          subtitle="src/kernels/spear_kernels.h — zéro allocation, zéro transcendante, utilisable en ISR sur MCU (Cortex-M0+, STM32C0, FPGA softcore)."
          flush
        >
          <pre className="overflow-x-auto p-3.5 font-mono text-[10px] leading-relaxed text-cyan-100/80">{`static inline float spear_${id}(float x)
{
    return ${spec.formula.replace(/·/g, '*')};
}`}</pre>
          <div className="border-t border-white/5 px-3.5 py-2">
            <Tag tone="green">gcc -O2</Tag>{' '}
            <Tag tone="cyan">{spec.flops} FLOPs</Tag>{' '}
            <Tag tone="amber">0 octet heap</Tag>{' '}
            <Tag tone="violet">{spec.domain}</Tag>
          </div>
        </Panel>
      </div>
    </div>
  );
}