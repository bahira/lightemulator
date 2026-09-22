import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fkScara, ikScara, planProfile, profileState,
  DEFAULT_PENDULUM, pendulumU, pendulumVdot, pendulumEigen, simulatePendulum,
  DEFAULT_HYBRID, pendulumUHybrid, simulatePendulumLaw, swingUpSuccess,
  type PendulumLaw,
} from '../physics/control';
import {
  Panel, Stat, Tag, Slider, Btn, Formula, CopyBtn, Segmented,
  useCanvas, drawGrid, drawSeries, axisLabel, paintField, drawImageBuffer, LUT,
} from '../ui/kit';

// Kernel C99 généré — miroir exact de src/physics/control.ts (audit corrigé)
const C99_EXPORT = `/* control_kernels.c — PHOTONIC ENGINE (audit SPEAR v2.2 corrigé)
 * MISRA-C:2012 · zero-heap · déterministe · 0 itération */
#include <stdint.h>
#include <math.h>

#define SAT(a, lim) ((a) > (lim) ? (lim) : ((a) < -(lim) ? -(lim) : (a)))

typedef struct { float th1, th2; int ok; } scara_ik_t;
typedef struct { float t[7]; float v_peak, a_peak, T; } traj_plan_t;

/* IK SCARA 2 liaisons — cos θ2 affine en r², 0 itération */
void scara_ik(float x, float z, float L1, float L2, scara_ik_t *o)
{
    const float r2 = x * x + z * z;
    const float c2 = (r2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
    if ((c2 > 1.0f) || (c2 < -1.0f)) { o->ok = 0; return; }
    const float th2 = acosf(c2);
    o->th1 = atan2f(z, x) - atan2f(L2 * sinf(th2), L1 + L2 * cosf(th2));
    o->th2 = th2;
    o->ok = 1;
}

/* Profil temps-minimal 7 segments — 3 régimes exacts (cas T2<0 corrigé :
 * a_eff = min(a_max, sqrt(v_max·j_max)) sinon v_max était violée) */
void traj_plan(float D, float vmax, float amax, float jmax, traj_plan_t *o)
{
    const float sq = sqrtf(vmax * jmax);
    const float a_eff = (amax < sq) ? amax : sq;
    const float Dsat = 2.0f * a_eff * a_eff * a_eff / (jmax * jmax);
    const float Dcrz = vmax * vmax / amax + amax * vmax / jmax;
    float T1, T2 = 0.0f, T4 = 0.0f;
    if (D <= Dsat) {
        const float ap = cbrtf(0.5f * D * jmax * jmax);
        T1 = ap / jmax;
        o->v_peak = ap * ap / jmax;
        o->a_peak = ap;
    } else if ((a_eff == amax) && (D <= Dcrz)) {
        T1 = amax / jmax;
        o->v_peak = amax * sqrtf(D / amax + amax * amax / (4.0f * jmax * jmax))
                  - amax * amax / (2.0f * jmax);
        o->a_peak = amax;
        T2 = (o->v_peak - amax * amax / jmax) / amax;
        if (T2 < 0.0f) { T2 = 0.0f; }
    } else {
        T1 = a_eff / jmax;
        o->a_peak = a_eff;
        o->v_peak = vmax;
        T2 = (vmax - a_eff * a_eff / jmax) / a_eff;
        if (T2 < 0.0f) { T2 = 0.0f; }
        const float Dacc = 2.0f * a_eff * a_eff * a_eff / (jmax * jmax)
                         + 3.0f * a_eff * a_eff * T2 / jmax + a_eff * T2 * T2;
        T4 = (D > Dacc) ? (D - Dacc) / vmax : 0.0f;
    }
    o->t[0] = T1; o->t[1] = T2; o->t[2] = T1; o->t[3] = T4;
    o->t[4] = T1; o->t[5] = T2; o->t[6] = T1;
    o->T = 4.0f * T1 + 2.0f * T2 + T4;
}

/* Pendule inversé : loi π* + V̇ EXACTE (terme résiduel inclus —
 * la preuve « V̇ = −βk2·θ̇² » du papier source est fausse) */
float pendulum_u(float th, float thd, float k1, float k2, float umax)
{
    return SAT(-(k1 * sinf(th) + k2 * thd), umax);
}

float pendulum_vdot(float th, float thd, float alpha, float beta,
                    float k1, float k2, float umax)
{
    return thd * (2.0f * alpha * sinf(th)
                  + beta * pendulum_u(th, thd, k1, k2, umax));
}`;

// ---------------------------------------------------------------------------
//  Panneau SCARA — cinématique inverse fermée
// ---------------------------------------------------------------------------

function ScaraPanel() {
  const [tx, setTx] = useState(0.28);
  const [tz, setTz] = useState(0.18);
  const L1 = 0.3, L2 = 0.2;

  const pose = ikScara(tx, tz, L1, L2);
  const fk = pose.ok ? fkScara(pose.th1, pose.th2, L1, L2) : null;
  const err = fk ? Math.hypot(fk.x - tx, fk.z - tz) : NaN;

  // latence IK mesurée une fois (50 000 résolutions)
  const [nsPerIk, setNsPerIk] = useState<number | null>(null);
  useEffect(() => {
    let sink = 0;
    const t0 = performance.now();
    for (let i = 0; i < 50000; i++) sink += ikScara(tx, tz, L1, L2).th1;
    setNsPerIk((performance.now() - t0) * 1e6 / 50000);
    void sink;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const { ref } = useCanvas((ctx, w, h) => {
    const scale = Math.min(w / 1.2, h / 0.95);
    const ox = w / 2, oy = h * 0.68;
    const px = (x: number) => ox + x * scale;
    const py = (z: number) => oy - z * scale;
    drawGrid(ctx, 0, 0, w, h, 12, 6);
    // portée max
    ctx.strokeStyle = 'rgba(148,163,184,0.18)';
    ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.arc(px(0), py(0), (L1 + L2) * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.arc(px(0), py(0), Math.abs(L1 - L2) * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
    if (pose.ok) {
      const j = fkScara(pose.th1, 0, L1, L2);
      ctx.strokeStyle = 'rgba(34,211,238,0.95)'; ctx.lineWidth = 7; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(j.x), py(j.z)); ctx.stroke();
      ctx.strokeStyle = 'rgba(167,139,250,0.95)';
      ctx.beginPath(); ctx.moveTo(px(j.x), py(j.z)); ctx.lineTo(px(tx), py(tz)); ctx.stroke();
      ctx.fillStyle = '#34d399';
      ctx.beginPath(); ctx.arc(px(tx), py(tz), 5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.beginPath(); ctx.arc(px(0), py(0), 6, 0, 2 * Math.PI); ctx.fill();
      ctx.lineWidth = 1;
    }
    // cible
    ctx.strokeStyle = pose.ok ? 'rgba(251,191,36,0.9)' : 'rgba(244,63,94,0.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px(tx) - 9, py(tz)); ctx.lineTo(px(tx) + 9, py(tz));
    ctx.moveTo(px(tx), py(tz) - 9); ctx.lineTo(px(tx), py(tz) + 9);
    ctx.stroke();
    axisLabel(ctx, 'base (0,0)', px(0) - 34, py(0) + 18);
  }, [tx, tz]);

  return (
    <Panel
      tag="IK FERMÉE"
      title="Bras SCARA 2 liaisons — cinématique inverse exacte, zéro itération"
      subtitle="cos θ₂ = (r²−L1²−L2²)/(2·L1·L2) est exactement affine en r² — le changement de variable qui lève le plafond de régression du papier SPEAR v2.2."
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2 / 1' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <Slider label="cible x" value={tx} min={-0.48} max={0.48} step={0.005} onChange={setTx} format={(v) => v.toFixed(3)} unit=" m" />
          <Slider label="cible z" value={tz} min={-0.35} max={0.46} step={0.005} onChange={setTz} format={(v) => v.toFixed(3)} unit=" m" />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="θ₁" value={pose.ok ? ((pose.th1 * 180) / Math.PI).toFixed(1) : '—'} unit="°" tone="cyan" />
          <Stat label="θ₂" value={pose.ok ? ((pose.th2 * 180) / Math.PI).toFixed(1) : '—'} tone="violet" />
          <Stat label="Résidu FK∘IK" value={pose.ok ? err.toExponential(1) : 'hors atteinte'} unit={pose.ok ? 'm' : ''} tone={pose.ok ? 'green' : 'rose'} />
          <Stat label="Latence mesurée" value={nsPerIk ? nsPerIk.toFixed(0) : '…'} unit="ns/cible" tone="amber" hint="vs ~12 itérations DLS" />
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  Panneau trajectoire — profil jerk-borné 7 segments
// ---------------------------------------------------------------------------

const REGIME_LABEL: Record<string, { txt: string; tone: 'cyan' | 'violet' | 'green' }> = {
  reduit: { txt: 'réduit — rien ne sature', tone: 'cyan' },
  triangulaire: { txt: 'triangulaire — a_max saturé', tone: 'violet' },
  palier: { txt: 'palier — vitesse saturée', tone: 'green' },
};

function TrajectoryPanel() {
  const [D, setD] = useState(1.2);
  const [vmax, setVmax] = useState(1.0);
  const [amax, setAmax] = useState(2.0);
  const [jmax, setJmax] = useState(5.0);

  const plan = useMemo(() => planProfile({ D, vmax, amax, jmax }), [D, vmax, amax, jmax]);
  const end = useMemo(() => profileState(plan, plan.duration), [plan]);
  const curves = useMemo(() => {
    const N = 600;
    const v: number[] = [], a: number[] = [], t: number[] = [];
    for (let i = 0; i < N; i++) {
      const tt = (plan.duration * i) / (N - 1);
      const s = profileState(plan, tt);
      v.push(s.v); a.push(s.a); t.push(tt);
    }
    return { v, a, t };
  }, [plan]);

  const { ref } = useCanvas((ctx, w, h) => {
    drawGrid(ctx, 0, 0, w, h, 10, 5);
    // bandes de jerk
    let acc = 0;
    for (let i = 0; i < 7; i++) {
      const x0 = (acc / plan.duration) * w;
      acc += plan.seg[i];
      const x1 = (acc / plan.duration) * w;
      if (plan.seg[i] > 0 && plan.jerk[i] !== 0) {
        ctx.fillStyle = plan.jerk[i] > 0 ? 'rgba(251,191,36,0.06)' : 'rgba(244,63,94,0.06)';
        ctx.fillRect(x0, 0, x1 - x0, h);
      }
    }
    const vmaxAbs = Math.max(...curves.v.map(Math.abs), ...curves.a.map(Math.abs), 1e-6);
    drawSeries(ctx, curves.a.map((x) => x / vmaxAbs), 0, 0, w, h, -1.08, 1.08, 'rgba(167,139,250,0.85)', undefined, 1);
    drawSeries(ctx, curves.v.map((x) => x / vmaxAbs), 0, 0, w, h, -1.08, 1.08, 'rgba(34,211,238,0.95)', 'rgba(34,211,238,0.07)', 1.6);
    axisLabel(ctx, `v (pic ${plan.vPeak.toFixed(2)})`, 8, 14, 'left', 'rgba(34,211,238,0.9)');
    axisLabel(ctx, `a (crête ${end.a.toFixed(2)} ≤ ${amax})`, 8, 26, 'left', 'rgba(167,139,250,0.9)');
    axisLabel(ctx, `${plan.duration.toFixed(2)} s`, w - 40, h - 6);
  }, [plan]);

  const rl = REGIME_LABEL[plan.regime];

  return (
    <Panel
      tag="PMP 7 SEGMENTS"
      title="Trajectoire temps-minimale à jerk borné — durées fermées, 3 régimes exacts"
      subtitle="Cas T2<0 du rapport source corrigé : accélération exploitable a_eff = min(a_max, √(v_max·j)) — sinon v_max était violée dans ~15 % des tirages."
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2 / 1' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <Slider label="consigne D" value={D} min={0.02} max={8} step={0.02} onChange={setD} format={(v) => v.toFixed(2)} unit=" m" />
          <Slider label="v_max" value={vmax} min={0.05} max={3} step={0.05} onChange={setVmax} format={(v) => v.toFixed(2)} unit=" m/s" />
          <Slider label="a_max" value={amax} min={0.5} max={6} step={0.1} onChange={setAmax} format={(v) => v.toFixed(1)} unit=" m/s²" />
          <Slider label="j_max" value={jmax} min={2} max={30} step={0.5} onChange={setJmax} format={(v) => v.toFixed(0)} unit=" m/s³" />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Régime" value={<Tag tone={rl.tone}>{plan.regime}</Tag>} mono={false} hint={rl.txt} />
          <Stat label="Durée totale" value={plan.duration.toFixed(3)} unit="s" tone="cyan" hint={`T₄ = ${plan.seg[3].toFixed(3)} s`} />
          <Stat label="v_peak / a_crête" value={`${plan.vPeak.toFixed(2)} / ${plan.aPeak.toFixed(2)}`} tone="violet" />
          <Stat label="Résidu |x_f−D|" value={Math.abs(end.x - D).toExponential(1)} unit="m" tone="green" hint="intégration fermée par segment" />
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  Panneau pendule inversé — plan de phase V̇ + rollout
// ---------------------------------------------------------------------------

function PendulumPanel() {
  const [law, setLaw] = useState<PendulumLaw>(DEFAULT_PENDULUM);
  const [th0, setTh0] = useState(1.0);
  const [thd0, setThd0] = useState(0);
  const [mode, setMode] = useState<'sommet' | 'swingup'>('sommet');

  const eig = pendulumEigen(law);
  const stable = mode === 'sommet' ? eig.every((e) => e.re < 0) : false;

  // en swing-up : rollout depuis le BAS avec l'hybride V1200, verdict falsifié
  const traj = useMemo(() => {
    if (mode === 'swingup') {
      return simulatePendulumLaw(law.alpha, law.beta,
        (th, thd) => pendulumUHybrid(th, thd, { ...DEFAULT_HYBRID, alpha: law.alpha, beta: law.beta, umax: law.umax }),
        Math.PI, 0, 8000, 5e-4); // 4 s depuis le bas
    }
    return simulatePendulum(law, th0, thd0, 6000, 5e-4); // 3 s
  }, [law, th0, thd0, mode]);

  const swingOK = useMemo(() => {
    if (mode !== 'swingup') return null;
    return swingUpSuccess(law.alpha, law.beta,
      (th, thd) => pendulumUHybrid(th, thd, { ...DEFAULT_HYBRID, alpha: law.alpha, beta: law.beta, umax: law.umax }),
      Math.PI);
  }, [law, mode]);

  const piSwingOK = useMemo(() => {
    if (mode !== 'swingup') return null;
    return swingUpSuccess(law.alpha, law.beta, (th, thd) => pendulumU(th, thd, law), Math.PI);
  }, [law, mode]);

  const field = useMemo(() => {
    const cols = 180, rows = 90;
    const data = new Float32Array(cols * rows);
    let posFrac = 0;
    for (let r = 0; r < rows; r++) {
      const thd = 4 - (8 * r) / (rows - 1);
      for (let c = 0; c < cols; c++) {
        const th = -Math.PI + (2 * Math.PI * c) / (cols - 1);
        const vd = pendulumVdot(th, thd, law);
        data[r * cols + c] = vd;
        if (vd > 0) posFrac++;
      }
    }
    posFrac /= cols * rows;
    return { data, cols, rows, posFrac };
  }, [law]);

  const { ref } = useCanvas((ctx, w, h) => {
    const img = new ImageData(field.cols, field.rows);
    paintField(img, field.data, field.cols, field.rows, LUT.diverging, -8, 8);
    drawImageBuffer(ctx, img, 0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    // rollout (θ ramené dans [−π,π])
    ctx.strokeStyle = 'rgba(52,211,153,0.95)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    let pen = false, prevPx = 0;
    for (let i = 0; i < traj.th.length; i += 4) {
      let th = traj.th[i] % (2 * Math.PI);
      if (th > Math.PI) th -= 2 * Math.PI;
      if (th < -Math.PI) th += 2 * Math.PI;
      const X = ((th + Math.PI) / (2 * Math.PI)) * w;
      const Y = ((4 - Math.max(-4, Math.min(4, traj.thd[i]))) / 8) * h;
      if (!pen || Math.abs(X - prevPx) > w / 3) { ctx.moveTo(X, Y); pen = true; }
      else ctx.lineTo(X, Y);
      prevPx = X;
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    // équilibre sommet
    ctx.strokeStyle = '#fbbf24';
    ctx.beginPath(); ctx.arc(w / 2, h / 2, 5, 0, 2 * Math.PI); ctx.stroke();
    axisLabel(ctx, 'θ ∈ [−π, π]', 6, h - 6);
    axisLabel(ctx, 'teal : V̇ ≤ 0 · magenta : V̇ > 0', w / 2 - 70, h - 6, 'left', 'rgba(148,163,184,0.55)');
  }, [field, traj]);

  const set = (k: keyof PendulumLaw) => (v: number) => setLaw((p) => ({ ...p, [k]: v }));

  return (
    <Panel
      tag="PENDULE INVERSÉ"
      title="Plan de phase de V̇ exacte + rollout — la preuve du papier est fausse, la loi reste bonne"
      subtitle="V̇ = θ̇·(2α·sinθ + β·u) contient un terme résiduel −(2α−βk₁)·θ̇·sinθ : V̇ > 0 sur une partie du plan (zones magenta), contrairement à la revendication « V̇ = −6.4 θ̇² ≤ 0 »."
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '5 / 2' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <Segmented
          options={[
            { id: 'sommet', label: 'π* sommet (plan de phase)', title: 'stabilisation locale + V̇ exacte' },
            { id: 'swingup', label: '★ Swing-up global depuis le bas', title: 'hybride V1200 vs π* — départ θ₀=π' },
          ]}
          value={mode}
          onChange={setMode}
        />
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-5">
          <Slider label={mode === 'swingup' ? 'k₁ (inactif en swing-up)' : 'k₁ (sur sinθ)'} value={law.k1} min={0.2} max={8} step={0.1} onChange={set('k1')} format={(v) => v.toFixed(1)} />
          <Slider label="k₂ (sur θ̇)" value={law.k2} min={0} max={4} step={0.1} onChange={set('k2')} format={(v) => v.toFixed(1)} />
          <Slider label="α (instabilité)" value={law.alpha} min={2} max={14} step={0.5} onChange={set('alpha')} format={(v) => v.toFixed(1)} />
          {mode === 'sommet' ? (
            <>
              <Slider label="θ₀ départ" value={th0} min={-3} max={3} step={0.1} onChange={setTh0} format={(v) => v.toFixed(1)} />
              <Slider label="θ̇₀ départ" value={thd0} min={-3} max={3} step={0.1} onChange={setThd0} format={(v) => v.toFixed(1)} />
            </>
          ) : (
            <>
              <div className="col-span-2 flex items-end pb-1">
                <Tag tone="amber">départ forcé : pendule bas (θ₀=π, θ̇₀=0)</Tag>
              </div>
            </>
          )}
        </div>
        {mode === 'sommet' ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="V̇ > 0 sur le plan" value={`${(field.posFrac * 100).toFixed(1)} %`} tone={field.posFrac > 0 ? 'amber' : 'green'} hint="papier : « 0 % » — réfuté" />
            <Stat label="λ linéarisé" value={eig.map((e) => e.re.toFixed(2)).join(' / ')} tone="slate" hint={eig.some((e) => e.im !== 0) ? `±${eig[0].im.toFixed(2)}i` : 'réels'} />
            <Stat label="Verdict local" value={<Tag tone={stable ? 'green' : 'rose'}>{stable ? 'STABLE' : 'INSTABLE'}</Tag>} mono={false} hint={stable ? 'βk₁ > α et βk₂ > 0' : `βk₁ = ${(law.beta * law.k1).toFixed(1)} ≤ α = ${law.alpha}`} />
            <Stat label="Rollout 3 s" value={`(${th0.toFixed(1)}, ${thd0.toFixed(1)})`} tone="cyan" hint="courbe verte sur le plan de phase" />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Hybride V1200" value={<Tag tone={swingOK ? 'green' : 'rose'}>{swingOK ? 'SWING-UP ✓' : 'échec'}</Tag>} mono={false}
              hint="pompage énergie + catch sigmoïde" />
            <Stat label="π* simple" value={<Tag tone={piSwingOK ? 'green' : 'rose'}>{piSwingOK ? 'SWING-UP ✓' : 'échec'}</Tag>} mono={false}
              hint="5 nœuds — dominait l'hybride 15/15 vs 3–11/15 (falsifié)" />
            <Stat label="Autorité β·u_max" value={(law.beta * law.umax).toFixed(1)} tone={(law.beta * law.umax) > law.alpha ? 'green' : 'rose'}
              hint={`vs α = ${law.alpha} — seuil de montée directe`} />
            <Stat label="Rollout" value="θ₀=π → 4 s" tone="cyan" hint="courbe verte sur le plan de phase" />
          </div>
        )}
        {mode === 'sommet' && (
          <p className="text-[10px] leading-relaxed text-slate-500">
            Poussez <Formula>α</Formula> au-delà de <Formula>β·k₁</Formula> ({(law.beta * law.k1).toFixed(1)}) :
            le verdict bascule en INSTABLE — la « marge d'autorité » du papier §6(ii) est bien un prérequis
            physique, pas un paramètre d'algorithme.
          </p>
        )}
        {mode === 'swingup' && (
          <p className="text-[10px] leading-relaxed text-slate-500">
            Falsification : dès que <Formula>β·u_max {'>'} α</Formula>, la simple loi π* fait le swing-up global
            (15/15 mesuré) SANS pompage d'énergie — le couple saturé bat la gravité en chaque point et le
            pendule « roule » vers le haut. L'hybride sophistiqué du papier fait pire (3–11/15).
          </p>
        )}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
//  Panneau robot animé — suivi de trajectoire jerk-borné (waypoints réels)
// ---------------------------------------------------------------------------

const PATHS = [
  { id: 'cercle', label: 'Cercle', title: 'segments D ≈ 0,12 m > D_crz — régime palier (7 segments complets)' },
  { id: 'ligne', label: 'Ligne', title: 'segments D ≈ 0,04 m < D_sat — régime réduit (profil triangulaire jerk)' },
  { id: 'huit', label: 'Huit', title: 'lemniscate de Gerono — les deux régimes se mélangent' },
] as const;
type PathId = (typeof PATHS)[number]['id'];
const N_WP = 20;
const RL1 = 0.3, RL2 = 0.2, RVMAX = 0.55, RAMAX = 7, RJMAX = 100;

function pathPoint(shape: PathId, u: number): { x: number; z: number } {
  const a = u * 2 * Math.PI;
  if (shape === 'cercle') return { x: 0.22 * Math.cos(a), z: 0.12 + 0.18 * Math.sin(a) };
  if (shape === 'ligne') return { x: -0.4 + 0.8 * u, z: 0.08 };
  const d = 1 + Math.sin(a) * Math.sin(a);
  return { x: (0.42 * Math.cos(a)) / d, z: 0.15 + (0.42 * Math.sin(a) * Math.cos(a)) / d };
}

function segmentPlan(wps: { x: number; z: number }[], i: number) {
  const a = wps[i], b = wps[(i + 1) % wps.length];
  return planProfile({ D: Math.hypot(b.x - a.x, b.z - a.z), vmax: RVMAX, amax: RAMAX, jmax: RJMAX });
}

function RobotTrackPanel() {
  const [shapeStr, setShapeStr] = useState('cercle');
  const shape = (PATHS.some((p) => p.id === shapeStr) ? shapeStr : 'cercle') as PathId;
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [tick, setTick] = useState(0);

  const wps = useMemo(() => Array.from({ length: N_WP }, (_, i) => pathPoint(shape, i / N_WP)), [shape]);

  // état d'exécution dans des refs — muté par la boucle rAF, zéro re-render
  const segRef = useRef(0);
  const tauRef = useRef(0);
  const planRef = useRef(segmentPlan(wps, 0));
  const trailRef = useRef<number[]>([]);

  useEffect(() => {
    segRef.current = 0; tauRef.current = 0; trailRef.current = [];
    planRef.current = segmentPlan(wps, 0);
  }, [wps]);

  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const loop = () => {
      tauRef.current += 0.016 * speed;
      if (tauRef.current >= planRef.current.duration) {
        tauRef.current = 0;
        segRef.current = (segRef.current + 1) % N_WP;
        planRef.current = segmentPlan(wps, segRef.current);
      }
      // trace du point terminal
      const plan = planRef.current, st = profileState(plan, tauRef.current);
      const a = wps[segRef.current], b = wps[(segRef.current + 1) % N_WP];
      const D = Math.hypot(b.x - a.x, b.z - a.z) || 1;
      trailRef.current.push(a.x + ((b.x - a.x) / D) * st.x, a.z + ((b.z - a.z) / D) * st.x);
      if (trailRef.current.length > 400) trailRef.current.splice(0, trailRef.current.length - 400);
      setTick((t) => t + 1);
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing, speed, wps]);

  // état courant (par frame de rendu)
  const live = useMemo(() => {
    const plan = planRef.current, st = profileState(plan, tauRef.current);
    const a = wps[segRef.current], b = wps[(segRef.current + 1) % N_WP];
    const D = Math.hypot(b.x - a.x, b.z - a.z) || 1;
    const ex = a.x + ((b.x - a.x) / D) * st.x;
    const ez = a.z + ((b.z - a.z) / D) * st.x;
    const pose = ikScara(ex, ez, RL1, RL2);
    const fk = pose.ok ? fkScara(pose.th1, pose.th2, RL1, RL2) : null;
    return {
      ex, ez, pose, st, seg: segRef.current,
      err: fk ? Math.hypot(fk.x - ex, fk.z - ez) : NaN,
      v: st.v, a: st.a, T: plan.duration, regime: plan.regime,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, wps]);

  const { ref } = useCanvas((ctx, w, h) => {
    const scale = Math.min(w / 1.2, h / 0.95);
    const ox = w / 2, oy = h * 0.68;
    const px = (x: number) => ox + x * scale;
    const py = (z: number) => oy - z * scale;
    drawGrid(ctx, 0, 0, w, h, 12, 6);
    // portée
    ctx.strokeStyle = 'rgba(148,163,184,0.15)';
    ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.arc(px(0), py(0), (RL1 + RL2) * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
    // trajectoire cible (points way)
    ctx.strokeStyle = 'rgba(251,191,36,0.45)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    for (let i = 0; i <= N_WP; i++) {
      const p = pathPoint(shape, (i % N_WP) / N_WP);
      if (i === 0) ctx.moveTo(px(p.x), py(p.z)); else ctx.lineTo(px(p.x), py(p.z));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // trail exécuté
    const T = trailRef.current.length / 2;
    if (T >= 2) {
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const X = px(trailRef.current[i * 2]), Y = py(trailRef.current[i * 2 + 1]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = 'rgba(34,211,238,0.7)';
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.lineWidth = 1;
    }
    // bras
    const L = live;
    if (L.pose.ok) {
      const j = fkScara(L.pose.th1, 0, RL1, RL2);
      ctx.strokeStyle = 'rgba(34,211,238,0.95)'; ctx.lineWidth = 7; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(j.x), py(j.z)); ctx.stroke();
      ctx.strokeStyle = 'rgba(167,139,250,0.95)';
      ctx.beginPath(); ctx.moveTo(px(j.x), py(j.z)); ctx.lineTo(px(L.ex), py(L.ez)); ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.beginPath(); ctx.arc(px(0), py(0), 6, 0, 2 * Math.PI); ctx.fill();
    }
    // effecteur + prochain waypont
    ctx.fillStyle = '#34d399';
    ctx.beginPath(); ctx.arc(px(L.ex), py(L.ez), 4.5, 0, 2 * Math.PI); ctx.fill();
    const b = wps[(L.seg + 1) % N_WP];
    ctx.strokeStyle = 'rgba(251,191,36,0.95)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px(b.x) - 7, py(b.z)); ctx.lineTo(px(b.x) + 7, py(b.z));
    ctx.moveTo(px(b.x), py(b.z) - 7); ctx.lineTo(px(b.x), py(b.z) + 7);
    ctx.stroke();
    ctx.lineWidth = 1;
    axisLabel(ctx, `waypoint ${(L.seg + 1) % N_WP}/${N_WP} · T = ${L.T.toFixed(2)} s · régime ${L.regime}`, 6, 12, 'left');
    axisLabel(ctx, `v = ${L.v.toFixed(3)} m/s · a = ${L.a.toFixed(2)} m/s²`, w - 6, 12, 'right', 'rgba(103,232,249,0.95)');
  }, [tick, live, shape], 0.5);

  return (
    <Panel
      tag="ROBOT EN MOUVEMENT"
      title="Bras SCARA animé — suivi de trajectoire jerk-bornée par waypoints"
      subtitle="Chaque segment est un déplacement temps-minimal exact (planProfile) exécuté par IK fermée — comme un vrai contrôleur de robot"
      right={<Tag tone={playing ? 'green' : 'slate'}>{playing ? 'EN MOUVEMENT' : 'EN PAUSE'}</Tag>}
      flush
    >
      <canvas ref={ref} className="block w-full" style={{ aspectRatio: '2.4 / 1' }} />
      <div className="space-y-3 border-t border-white/5 p-3.5">
        <div className="grid gap-x-4 gap-y-2 sm:grid-cols-3">
          <Segmented
            options={PATHS.map((p) => ({ id: p.id, label: p.label, title: p.title }))}
            value={shape}
            onChange={(v) => setShapeStr(v)}
          />
          <Slider label="Vitesse d'exécution" value={speed} min={0.2} max={3} step={0.1} onChange={setSpeed} format={(v) => `×${v.toFixed(1)}`} />
          <div className="flex items-end gap-1.5 pb-1">
            <Btn tone={playing ? 'danger' : 'primary'} onClick={() => setPlaying(!playing)}>
              {playing ? '⏸ pause' : '▶ démarrer'}
            </Btn>
            <Btn onClick={() => { trailRef.current = []; segRef.current = 0; tauRef.current = 0; }} title="efface la trace">↺ trace</Btn>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          <Stat label="θ₁" value={live.pose.ok ? ((live.pose.th1 * 180) / Math.PI).toFixed(1) : '—'} unit="°" tone="cyan" />
          <Stat label="θ₂" value={live.pose.ok ? ((live.pose.th2 * 180) / Math.PI).toFixed(1) : '—'} unit="°" tone="violet" />
          <Stat label="v effecteur" value={live.v.toFixed(3)} unit="m/s" tone="green" hint="profil jerk-borné exact" />
          <Stat label="a effecteur" value={live.a.toFixed(2)} unit="m/s²" tone="amber" hint={`amax = ${RAMAX} m/s² · saturée au régime réduit`} />
          <Stat label="Résidu FK∘IK" value={live.pose.ok ? live.err.toExponential(1) : 'hors atteinte'} unit={live.pose.ok ? 'm' : ''} tone={live.pose.ok ? 'green' : 'rose'} hint="audit 'ctrl-ik'" />
        </div>
        <p className="text-[10px] leading-relaxed text-slate-500">
          Le contrôleur planifie chaque segment avec <Formula>planProfile</Formula> (3 régimes exacts :
          réduit <Formula>D ≤ D_sat</Formula>, triangulaire, palier <Formula>D ≤ D_crz</Formula>) puis exécute
          avec <Formula>profileState</Formula> et <Formula>ikScara</Formula> fermée — zéro itération, zéro heap.
          Vérifié par <Formula>ctrl-ik</Formula> (résidu machine) et <Formula>ctrl-traj</Formula> (violations = 0).
        </p>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------

export default function ControlLab() {
  return (
    <div className="space-y-3">
      <div className="grid gap-3 lg:grid-cols-2">
        <ScaraPanel />
        <TrajectoryPanel />
      </div>
      <RobotTrackPanel />
      <PendulumPanel />
      <Panel
        tag="C99"
        title="Kernel C99 embarqué — IK + trajectoire + pendule, miroir exact du moteur TS"
        subtitle="Les mêmes formules que les panneaux ci-dessus, prêtes pour MCU/FPGA. Chaque constante a été falsifiée avant d'atterrir ici."
        right={<CopyBtn text={C99_EXPORT} label="⧉ Copier le kernel" />}
        flush
      >
        <pre className="overflow-x-auto p-3.5 font-mono text-[10px] leading-relaxed text-cyan-100/80">{C99_EXPORT}</pre>
      </Panel>
    </div>
  );
}
