import { useRef, useState, useEffect, useMemo } from 'react';
import {
  lightField, swarmStep, friisReceived, meanNearestDist,
  type Beacon, type Drone, type SwarmParams,
} from '../physics/drones';
import { Panel, Stat, Slider, Btn, Tag, Formula, Segmented, useCanvas, axisLabel } from '../ui/kit';

const MODES = [
  { id: 'phototaxie', label: 'Phototaxie', title: 'les drones grimpent le gradient de lumière, trajectoire droite (champ à source unique)' },
  { id: 'essaim', label: 'Essaim', title: 'phototaxie + séparation mutuelle + murs mous' },
] as const;
type ModeId = (typeof MODES)[number]['id'];

const ROOM = 3; // l'arène unité = une pièce de 3 m × 3 m
const IR_LAMBDA = 940e-9; // LED IR 940 nm
const PT_W = 1.0; // 1 W optique

function mulberry(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeBeacons(seed: number): Beacon[] {
  const rng = mulberry(seed);
  return [0, 1, 2].map((i) => ({
    x: 0.2 + rng() * 0.6,
    y: 0.2 + rng() * 0.6,
    power: 0.7 + rng() * 0.6 + i * 0.15,
  }));
}

function makeDrones(n: number, seed: number): Drone[] {
  const rng = mulberry(seed + 31);
  return Array.from({ length: n }, () => {
    const x = 0.08 + rng() * 0.84, y = 0.08 + rng() * 0.84;
    return { x, y, vx: 0, vy: 0, trail: [x, y] };
  });
}

export default function DroneLab() {
  const [nStr, setNStr] = useState('12');
  const n = Math.max(2, Math.min(60, Number(nStr) || 12));
  const [maxV, setMaxV] = useState(0.012);
  const [kSep, setKSep] = useState(0.35);
  const [mode, setMode] = useState<ModeId>('essaim');
  const [playing, setPlaying] = useState(true);
  const [seed, setSeed] = useState(42);
  const [tick, setTick] = useState(0);
  const [steps, setSteps] = useState(0);

  const beaconsRef = useRef<Beacon[]>(makeBeacons(seed));
  const dronesRef = useRef<Drone[]>(makeDrones(n, seed));

  const params = useMemo<SwarmParams>(
    () => ({ kPh: 2.2, kSep: mode === 'essaim' ? kSep : 0, rSep: 0.09, maxV, dt: 1 }),
    [mode, kSep, maxV],
  );

  // reseed / resize → rebuild (garde les balises si seulement n change)
  useEffect(() => { dronesRef.current = makeDrones(n, seed); setSteps(0); }, [n, seed]);
  useEffect(() => { beaconsRef.current = makeBeacons(seed); setSteps(0); }, [seed]);

  // boucle d'animation : physique + horloge
  useEffect(() => {
    if (!playing) return;
    let id = 0, frame = 0;
    const loop = () => {
      swarmStep(dronesRef.current, beaconsRef.current, params);
      if (++frame % 4 === 0) setSteps((s) => s + 4);
      setTick((t) => t + 1);
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing, params]);

  // télémétrie dérivée (par frame de rendu)
  const telemetry = useMemo(() => {
    const drones = dronesRef.current, beacons = beaconsRef.current;
    const mean = meanNearestDist(drones, beacons);
    let converged = 0;
    for (const d of drones) {
      let m = Infinity;
      for (const b of beacons) m = Math.min(m, Math.hypot(d.x - b.x, d.y - b.y));
      if (m < 0.1) converged++;
    }
    // drone 0 : puissance reçue de la balise la plus proche (Friis, physique réelle)
    const d0 = drones[0];
    let nearest = beacons[0], nd = Infinity;
    for (const b of beacons) {
      const r = Math.hypot(d0.x - b.x, d0.y - b.y);
      if (r < nd) { nd = r; nearest = b; }
    }
    const pr = friisReceived(PT_W * nearest.power, IR_LAMBDA, Math.max(nd, 1e-6) * ROOM);
    return { mean, converged, pr, nd: nd * ROOM };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  // capteurs du drone 0 : 4 photodiodes à ±90° autour de sa direction
  const sensors = useMemo(() => {
    const d0 = dronesRef.current[0], beacons = beaconsRef.current;
    const v = Math.hypot(d0.vx, d0.vy) + 1e-9;
    const ux = d0.vx / v, uy = d0.vy / v;
    const rS = 0.05;
    return [0, Math.PI / 2, Math.PI, -Math.PI / 2].map((a) => {
      const c = Math.cos(a), s = Math.sin(a);
      // rotation du vecteur direction
      const dx = ux * c - uy * s, dy = ux * s + uy * c;
      return lightField(beacons, d0.x + dx * rS, d0.y + dy * rS);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  // ------------------------------------------------------------ canvas
  const view = useCanvas((ctx, w, h) => {
    const S = Math.min(w, h) - 24;
    const ox = (w - S) / 2, oy = (h - S) / 2;
    const X = (u: number) => ox + u * S;
    const Y = (u: number) => oy + (1 - u) * S; // y normalisé vers le haut
    const drones = dronesRef.current, beacons = beaconsRef.current;

    // fond
    ctx.fillStyle = '#04070d';
    ctx.fillRect(0, 0, w, h);

    // champ lumineux : halo radial par balise (I ∝ 1/d² visuellement)
    for (const b of beacons) {
      const g = ctx.createRadialGradient(X(b.x), Y(b.y), 2, X(b.x), Y(b.y), S * 0.42);
      g.addColorStop(0, `rgba(251,191,36,${0.22 * b.power})`);
      g.addColorStop(0.4, 'rgba(251,191,36,0.05)');
      g.addColorStop(1, 'rgba(251,191,36,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    }

    // arène
    ctx.strokeStyle = 'rgba(148,163,184,0.18)';
    ctx.lineWidth = 1;
    ctx.strokeRect(X(0), Y(1), S, S);

    // traînées
    for (const d of drones) {
      const T = d.trail.length / 2;
      if (T < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const px = X(d.trail[i * 2]), py = Y(d.trail[i * 2 + 1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = 'rgba(34,211,238,0.16)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // balises : points ambrés pulsants
    const pulse = 0.5 + 0.5 * Math.sin(tick * 0.08);
    for (const b of beacons) {
      ctx.beginPath();
      ctx.arc(X(b.x), Y(b.y), 5 + 2 * pulse, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(251,191,36,0.95)';
      ctx.shadowColor = 'rgba(251,191,36,0.9)';
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // drones : triangles orientés par vitesse
    for (let i = 0; i < drones.length; i++) {
      const d = drones[i];
      const v = Math.hypot(d.vx, d.vy) + 1e-9;
      const ux = d.vx / v, uy = d.vy / v;
      const px = X(d.x), py = Y(d.y), R = 6.5;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(Math.atan2(-uy, ux)); // y canvas inversé
      ctx.beginPath();
      ctx.moveTo(R, 0); ctx.lineTo(-R * 0.7, R * 0.6); ctx.lineTo(-R * 0.7, -R * 0.6);
      ctx.closePath();
      ctx.fillStyle = i === 0 ? 'rgba(165,243,252,0.98)' : 'rgba(34,211,238,0.88)';
      ctx.shadowColor = 'rgba(34,211,238,0.8)';
      ctx.shadowBlur = i === 0 ? 16 : 8;
      ctx.fill();
      ctx.restore();
      ctx.shadowBlur = 0;
    }

    // liaisons optiques du drone 0 (lignes vers les balises vues)
    const d0 = drones[0];
    for (const b of beacons) {
      const dist = Math.hypot(d0.x - b.x, d0.y - b.y);
      const alpha = Math.min(0.5, 0.05 / (dist * dist + 0.02));
      ctx.beginPath();
      ctx.moveTo(X(d0.x), Y(d0.y)); ctx.lineTo(X(b.x), Y(b.y));
      ctx.strokeStyle = `rgba(251,191,36,${alpha})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    axisLabel(ctx, `arène ${ROOM}×${ROOM} m · ${drones.length} drones · ${beacons.length} balises IR 940 nm`, 6, 12, 'left');
    axisLabel(ctx, `d̄ = ${(telemetry.mean * ROOM).toFixed(3)} m`, w - 6, 12, 'right', 'rgba(103,232,249,0.95)');
  }, [tick, telemetry, n], 0.62);

  const maxSensor = Math.max(...sensors, 1e-12);

  return (
    <div className="grid gap-3 xl:grid-cols-[1fr_320px]">
      {/* ------------------------------------------------------------- vue */}
      <Panel
        tag="PLAYGROUND"
        title="Drones lumineux — navigation par la lumière"
        subtitle="Phototaxie réelle (gradient du champ 1/d²), positionnement par trilatération, liaison de Friis — chaque chiffre est physique"
        right={<Tag tone={playing ? 'green' : 'slate'}>{playing ? 'EN VOL' : 'EN PAUSE'}</Tag>}
        flush
      >
        <div className="p-3.5">
          <div className="w-full"><canvas ref={view.ref} style={{ width: '100%', height: view.size.h }} className="block" /></div>
        </div>
      </Panel>

      {/* ------------------------------------------------------- contrôles */}
      <div className="flex flex-col gap-3">
        <Panel title="Commandes de vol" subtitle="l'arène unité = 3 m ; vitesses en unités normalisées/s">
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-1.5">
              <Btn tone={playing ? 'danger' : 'primary'} onClick={() => setPlaying(!playing)}>
                {playing ? '⏸ pause' : '▶ envol'}
              </Btn>
              <Btn onClick={() => setSeed((s) => s + 1)} title="repositionne balises + drones">↺ balises</Btn>
            </div>
            <Slider label="Drones" value={n} min={2} max={60} step={1} onChange={(v) => setNStr(String(v))} />
            <Slider label="Vitesse max" value={maxV} min={0.004} max={0.03} step={0.001} onChange={setMaxV} format={(v) => (v * ROOM * 100).toFixed(1)} unit=" cm/s" />
            <Slider label="Séparation" value={kSep} min={0} max={1} step={0.05} onChange={setKSep} />
            <Segmented options={MODES.map((m) => ({ id: m.id, label: m.label, title: m.title }))} value={mode} onChange={setMode} />
          </div>
        </Panel>

        <Panel title="Télémétrie" subtitle={`t+${steps} pas · mesure directe de l'état`}>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="d̄ balise" value={(telemetry.mean * ROOM).toFixed(3)} unit="m" hint="distance moyenne à la plus proche" />
            <Stat label="convergence" value={`${Math.round((100 * telemetry.converged) / n)}%`} tone="green" hint={`${telemetry.converged}/${n} drones à < 0,3 m`} />
            <Stat label="Pr drone 0" value={(telemetry.pr * 1e12).toFixed(3)} unit="pW" tone="amber" hint="Friis : Pt·G·λ²/(4πd)²" />
            <Stat label="d drone 0" value={telemetry.nd.toFixed(3)} unit="m" hint="vers la balise la plus proche" />
          </div>
        </Panel>

        <Panel title="Cerveau photonique — drone 0" subtitle="4 photodiodes à ±90° autour de sa direction : le gradient est visible">
          <div className="flex items-end justify-between gap-2 px-1">
            {sensors.map((s, i) => (
              <div key={i} className="flex flex-1 flex-col items-center gap-1">
                <div className="flex h-16 w-full items-end overflow-hidden rounded bg-white/[0.03] ring-1 ring-white/6">
                  <div
                    className="w-full bg-gradient-to-t from-amber-500/70 to-amber-300/90"
                    style={{ height: `${Math.max(4, (100 * s) / maxSensor)}%` }}
                  />
                </div>
                <span className="font-mono text-[9px] text-slate-500">{['avant', 'g⁺90°', 'arrière', 'd⁺90°'][i]}</span>
                <span className="font-mono text-[9px] tabular-nums text-amber-300">{s.toExponential(1)}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[9px] leading-relaxed text-slate-600">
            La phototaxie suit <Formula>I = Σ P/(4πd²)</Formula> : la diode qui voit le plus fort
            tire le drone vers la source. Vérifié par <Formula>validate.ts</Formula> (groupe Drones) :
            convergence garantie, trilatération à l'epsilon machine, <Formula>Pr(2d) = Pr(d)/4</Formula>.
          </p>
        </Panel>
      </div>
    </div>
  );
}
