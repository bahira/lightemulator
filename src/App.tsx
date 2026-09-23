import { useState, lazy, Suspense, useEffect } from 'react';
import { cn } from './utils/cn';
import BpmLab from './labs/BpmLab';

const ProcessorLab = lazy(() => import('./labs/ProcessorLab'));
const IsingLab = lazy(() => import('./labs/IsingLab'));
const KanLab = lazy(() => import('./labs/KanLab'));
const FreeEnergyLab = lazy(() => import('./labs/FreeEnergyLab'));
const ValidationLab = lazy(() => import('./labs/ValidationLab'));
const SpearLab = lazy(() => import('./labs/SpearLab'));
const ControlLab = lazy(() => import('./labs/ControlLab'));
const DispersionLab = lazy(() => import('./labs/DispersionLab'));
const DroneLab = lazy(() => import('./labs/DroneLab'));
const PnnLab = lazy(() => import('./labs/PnnLab'));
const HoloLab = lazy(() => import('./labs/HoloLab'));
const QuantumLab = lazy(() => import('./labs/QuantumLab'));
const LangevinLab = lazy(() => import('./labs/LangevinLab'));
const OptoTransformerLab = lazy(() => import('./labs/OptoTransformerLab'));
const InferenceLab = lazy(() => import('./labs/InferenceLab'));

type TabId = 'bpm' | 'processor' | 'ising' | 'kan' | 'fep' | 'validate' | 'spear' | 'control' | 'dispersion' | 'drones' | 'pnn' | 'holo' | 'quantum' | 'langevin' | 'opto' | 'inference';

const TABS: { id: TabId; label: string; glyph: string; blurb: string }[] = [
  { id: 'bpm', label: 'Moteur de lumière', glyph: '≈', blurb: 'BPM split-step Fourier — propagation réelle dans les guides' },
  { id: 'processor', label: 'Processeur MZI', glyph: '⧉', blurb: 'Maillage universel — toute unitaire N×N, exactement' },
  { id: 'ising', label: 'Machine d’Ising', glyph: '⬡', blurb: 'CIM sur MaxCut, comparée à l’optimum exact' },
  { id: 'kan', label: 'KAN photonique', glyph: '◈', blurb: 'B-splines apprises, backprop vérifiée' },
  { id: 'fep', label: 'Énergie libre', glyph: '◉', blurb: 'Champs complexes apprenant par contraste d’équilibre' },
  { id: 'validate', label: 'Validation', glyph: '✓', blurb: 'La boucle grounded — 44 tests numériques' },
  { id: 'spear', label: 'Kernels SPEAR', glyph: 'ƒ', blurb: 'Activations LLM distillées en algèbre pure, benchmarkée ici' },
  { id: 'control', label: 'Contrôle', glyph: '⌖', blurb: 'IK fermée, trajectoire jerk-bornée, pendule inversé — audités & corrigés' },
  { id: 'dispersion', label: 'Dispersion', glyph: '∿', blurb: 'Sellmeier exact, GDD/TOD réparés, impulsion femtoseconde & biréfringence' },
  { id: 'drones', label: 'Drones lumineux', glyph: '➤', blurb: 'Playground — phototaxie réelle, trilatération exacte, budget de Friis' },
  { id: 'pnn', label: 'PNN', glyph: '⬢', blurb: 'Accélérateur photonique émulé — superposition N+C + gradient AVM (Nat. Com. 2026)' },
  { id: 'holo', label: 'Holographie', glyph: '⊛', blurb: 'Gerchberg–Saxton réel — dessinez la cible, l’hologramme de phase est calculé, quantifié, mesuré' },
  { id: 'quantum', label: 'Optique quantique', glyph: 'Ψ', blurb: 'États de Fock exacts — dip HOM, MZI photon unique, violation de Bell CHSH en direct' },
  { id: 'langevin', label: 'Langevin', glyph: '⋯', blurb: 'Ordinateur thermodynamique génératif — gradient inverse exact, relation de fluctuation (arXiv:2506.15121)' },
  { id: 'opto', label: 'Opto-Transformer', glyph: '⚡', blurb: 'Co-processeur hybride photonique-SPEAR : attention optique MZI + non-linéarités rationnelles' },
  { id: 'inference', label: 'Inference edge', glyph: '▲', blurb: 'Le vrai module C en wasm32 — 270k params, INT4, ~120 tok/s offline dans le navigateur' },
];

export default function App() {
  const [tab, setTab] = useState<TabId>('bpm');
  const [clock, setClock] = useState(0);
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const id = setInterval(() => setClock((c) => c + 1), 1000);
    return () => clearInterval(id);
  }, []);

  // ---- raccourcis clavier : ⌘K palette · chiffres 1-9,0 → 10 premiers labs
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((o) => !o);
        return;
      }
      if (paletteOpen) return;
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
      const di = '1234567890'.indexOf(e.key);
      if (di >= 0 && di < TABS.length) setTab(TABS[di].id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [paletteOpen]);

  return (
    <div className="min-h-screen bg-[#04060d] text-slate-200 antialiased">
      {/* ambient background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-[15%] -top-[25%] h-[55vh] w-[55vw] rounded-full bg-cyan-500/8 blur-[130px]" />
        <div className="absolute -right-[10%] top-[25%] h-[50vh] w-[45vw] rounded-full bg-violet-600/8 blur-[130px]" />
        <div className="absolute bottom-[-20%] left-[25%] h-[45vh] w-[50vw] rounded-full bg-emerald-500/5 blur-[140px]" />
        <div
          className="absolute inset-0 opacity-[0.35]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(148,163,184,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.045) 1px, transparent 1px)',
            backgroundSize: '52px 52px',
            maskImage: 'radial-gradient(ellipse 90% 60% at 50% 0%, black 30%, transparent 75%)',
          }}
        />
      </div>

      <div className="relative">
        {/* -------------------------------------------------------- header */}
        <header className="sticky top-0 z-40 border-b border-white/6 bg-[#04060d]/85 backdrop-blur-xl">
          <div className="mx-auto max-w-[1720px] px-4">
            <div className="flex flex-wrap items-center justify-between gap-3 py-2.5">
              <div className="flex items-center gap-3">
                <div className="relative grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-cyan-400/15 to-violet-500/15 ring-1 ring-cyan-400/25">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <defs>
                      <radialGradient id="core">
                        <stop offset="0%" stopColor="#fff" />
                        <stop offset="45%" stopColor="#22d3ee" />
                        <stop offset="100%" stopColor="#a855f7" />
                      </radialGradient>
                    </defs>
                    <circle cx="12" cy="12" r="2.6" fill="url(#core)" />
                    {[6.5, 9.5].map((r, i) => (
                      <circle key={r} cx="12" cy="12" r={r} fill="none" stroke={i ? 'rgba(168,85,247,0.35)' : 'rgba(34,211,238,0.5)'} strokeWidth="0.7" strokeDasharray={i ? '2 4' : '3 3'}>
                        <animateTransform attributeName="transform" type="rotate" values={i ? '360 12 12;0 12 12' : '0 12 12;360 12 12'} dur={i ? '14s' : '9s'} repeatCount="indefinite" />
                      </circle>
                    ))}
                  </svg>
                  <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300 shadow-[0_0_6px_#22d3ee]" />
                </div>
                <div>
                  <h1 className="flex items-center gap-2 font-mono text-[13px] font-bold tracking-tight text-white">
                    PHOTONIC ENGINE
                    <span className="rounded bg-white/6 px-1.5 py-px text-[9px] font-medium tracking-widest text-slate-400 ring-1 ring-white/10">
                      2026
                    </span>
                  </h1>
                  <p className="text-[10px] leading-tight text-slate-500">
                    Calcul direct avec la lumière · substrat photonique &amp; analogique
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2 font-mono text-[9px] text-slate-600">
                <span className="hidden items-center gap-1.5 sm:flex">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                  SOLVEURS ACTIFS
                </span>
                <span className="hidden md:inline">·</span>
                <span className="hidden tabular-nums md:inline">t+{clock}s</span>
                <button
                  onClick={() => setPaletteOpen(true)}
                  title="palette de commandes (⌘K)"
                  className="ml-1 hidden items-center gap-1 rounded bg-white/[0.04] px-1.5 py-0.5 font-mono text-[9px] text-slate-400 ring-1 ring-white/10 transition-all hover:bg-white/8 hover:text-slate-200 sm:flex"
                >
                  <span className="text-[10px]">⌘K</span>
                </button>
              </div>
            </div>

            {/* tabs */}
            <nav className="-mb-px flex gap-1 overflow-x-auto pb-0">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  title={t.blurb}
                  className={cn(
                    'group relative flex shrink-0 items-center gap-2 border-b-2 px-3 py-2.5 font-mono text-[10px] tracking-wide transition-all',
                    tab === t.id
                      ? 'border-cyan-400 text-cyan-200'
                      : 'border-transparent text-slate-500 hover:text-slate-300',
                  )}
                >
                  <span className={cn('text-[13px] leading-none', tab === t.id ? 'text-cyan-300' : 'text-slate-600')}>{t.glyph}</span>
                  <span className="font-semibold uppercase">{t.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </header>

        {/* ------------------------------------------------------ thesis bar */}
        <div className="border-b border-white/5 bg-white/[0.012]">
          <div className="mx-auto flex max-w-[1720px] flex-wrap items-center gap-x-5 gap-y-1.5 px-4 py-2 text-[10px]">
            <span className="text-slate-500">
              <span className="text-cyan-300">Pas de qubits</span> — un substrat physique différent
            </span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-500">
              Latence optique <span className="text-violet-300">limitée par le temps de vol</span>
            </span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-500">
              Utile pour <span className="text-emerald-300">l’optimisation combinatoire</span> et l’edge ultra-rapide
            </span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-500">
              <span className="text-amber-300">Pas de forward-pass transformer</span> en production aujourd’hui
            </span>
          </div>
        </div>

        {/* ------------------------------------------------------------ body */}
        <main className="mx-auto max-w-[1720px] px-4 py-4">
          <div className="mb-3 flex items-baseline gap-2">
            <h2 className="text-[13px] font-semibold text-slate-200">{TABS.find((t) => t.id === tab)!.label}</h2>
            <p className="text-[10px] text-slate-500">{TABS.find((t) => t.id === tab)!.blurb}</p>
          </div>

          <Suspense fallback={<Loading />}>
            {tab === 'bpm' && <BpmLab />}
            {tab === 'processor' && <ProcessorLab />}
            {tab === 'ising' && <IsingLab />}
            {tab === 'kan' && <KanLab />}
            {tab === 'fep' && <FreeEnergyLab />}
            {tab === 'validate' && <ValidationLab />}
            {tab === 'spear' && <SpearLab />}
            {tab === 'control' && <ControlLab />}
            {tab === 'dispersion' && <DispersionLab />}
            {tab === 'drones' && <DroneLab />}
            {tab === 'pnn' && <PnnLab />}
            {tab === 'holo' && <HoloLab />}
            {tab === 'quantum' && <QuantumLab />}
            {tab === 'langevin' && <LangevinLab />}
            {tab === 'opto' && <OptoTransformerLab />}
            {tab === 'inference' && <InferenceLab />}
          </Suspense>
        </main>

        {/* ------------------------------------------------ palette de commandes */}
        {paletteOpen && (
          <CommandPalette
            onClose={() => setPaletteOpen(false)}
            onSelect={(id) => { setTab(id); setPaletteOpen(false); }}
          />
        )}

        {/* ---------------------------------------------------------- footer */}
        <footer className="mt-8 border-t border-white/6 bg-black/25">
          <div className="mx-auto max-w-[1720px] px-4 py-5">
            <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
              <div>
                <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400">Ce qui tourne réellement</div>
                <ul className="mt-2 space-y-1 text-[10px] leading-relaxed text-slate-500">
                  <li>· FFT radix-2 itérative in-place, tables de twiddles en cache</li>
                  <li>· BPM split-step Fourier (Strang, O(Δz²)) + solveur de mode à distance imaginaire</li>
                  <li>· Élimination de Givens complexe → maillage MZI exact</li>
                  <li>· CIM DOPO mean-field avec correction d’hétérogénéité d’amplitude</li>
                  <li>· B-splines de Cox-de Boor + rétropropagation analytique + Adam</li>
                  <li>· Énergie libre cohérente : contraste d'équilibre (EP) sur champs complexes</li>
                  <li>· Contrôle audité : IK fermée, profil jerk-borné 3 régimes exacts, pendule π*</li>
                  <li>· Dispersion : dérivées de Sellmeier exactes, phase spectrale β₂ω²/2+β₃ω³/6</li>
                  <li>· Holographie : Gerchberg–Saxton sur FFT 2D, efficacité &amp; conservation mesurées</li>
                  <li>· Optique quantique : Fock 1–2 photons en algèbre fermée, CHSH vs variables cachées simulées</li>
                  <li>· Opto-Transformer : co-processeur hybride MZI SVD + Attention &amp; FFN SPEAR rationnel</li>
                </ul>
              </div>
              <div>
                <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400">Honnêteté des chiffres</div>
                <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
                  Les temps affichés sont mesurés avec <code className="text-cyan-300/80">performance.now()</code> dans
                  votre navigateur. Les latences photoniques sont dérivées de <code className="text-cyan-300/80">t = n_g·L/c</code> avec
                  des paramètres de dispositifs publiés, et la formule est affichée à côté de chaque résultat.
                  Aucune métrique n’est générée aléatoirement.
                </p>
              </div>
              <div>
                <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400">Périmètre de la revendication</div>
                <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
                  Le photonique gagne aujourd’hui sur les produits matriciels linéaires à très faible latence
                  et sur l’optimisation combinatoire de type Ising. Il ne gagne pas sur la mémoire, le contrôle
                  de flux ni les grands modèles de langage. Cet émulateur est construit pour rendre cette
                  frontière visible plutôt que pour la masquer.
                </p>
              </div>
            </div>
            <div className="mt-5 border-t border-white/5 pt-3 text-center font-mono text-[9px] text-slate-700">
              PHOTONIC ENGINE · émulateur de calcul photonique &amp; analogique · tous les solveurs s’exécutent côté client
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div className="grid h-64 place-items-center">
      <div className="flex flex-col items-center gap-3">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-cyan-400/25 border-t-cyan-300" />
        <span className="font-mono text-[10px] text-slate-500">initialisation des solveurs…</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
//  Palette de commandes — ⌘K : navigation fuzzy sur les labs
// ---------------------------------------------------------------------------

function CommandPalette({ onClose, onSelect }: { onClose: () => void; onSelect: (id: TabId) => void }) {
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState(0);

  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const matches = TABS.filter((t) => {
    const hay = `${t.label} ${t.blurb} ${t.id}`.toLowerCase();
    return tokens.every((tok) => hay.includes(tok));
  });

  const commit = (i: number) => {
    const m = matches[i];
    if (m) onSelect(m.id);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 px-4 pt-[14vh] backdrop-blur-sm"
      onMouseDown={onClose}
    >
      <div
        className="w-full max-w-md overflow-hidden rounded-xl border border-white/10 bg-[#0a0f1e] shadow-[0_30px_80px_-20px_rgba(0,0,0,1)]"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <input
          autoFocus
          value={query}
          placeholder="aller au lab… (holographie, bell, ising…)"
          onChange={(e) => { setQuery(e.target.value); setIndex(0); }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); setIndex((i) => Math.min(matches.length - 1, i + 1)); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setIndex((i) => Math.max(0, i - 1)); }
            else if (e.key === 'Enter') commit(index);
            else if (e.key === 'Escape') onClose();
          }}
          className="w-full border-b border-white/8 bg-transparent px-4 py-3 font-mono text-[12px] text-slate-100 placeholder:text-slate-600 focus:outline-none"
        />
        <ul className="max-h-[46vh] overflow-y-auto p-1.5">
          {matches.length === 0 && (
            <li className="px-3 py-4 text-center font-mono text-[10px] text-slate-600">aucun lab ne correspond</li>
          )}
          {matches.map((t, i) => (
            <li key={t.id}>
              <button
                onMouseEnter={() => setIndex(i)}
                onClick={() => commit(i)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-all',
                  i === index ? 'bg-cyan-400/10 ring-1 ring-cyan-400/25' : 'hover:bg-white/4',
                )}
              >
                <span className={cn('w-4 text-center text-[14px]', i === index ? 'text-cyan-300' : 'text-slate-600')}>{t.glyph}</span>
                <span className="min-w-0">
                  <span className={cn('block font-mono text-[11px] font-semibold', i === index ? 'text-slate-100' : 'text-slate-300')}>{t.label}</span>
                  <span className="block truncate text-[9px] text-slate-600">{t.blurb}</span>
                </span>
                {i === index && <span className="ml-auto shrink-0 font-mono text-[9px] text-slate-600">↵</span>}
              </button>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-3 border-t border-white/6 px-4 py-2 font-mono text-[9px] text-slate-600">
          <span>↑↓ naviguer</span><span>↵ ouvrir</span><span>esc fermer</span>
          <span className="ml-auto">1-9,0 → accès direct</span>
        </div>
      </div>
    </div>
  );
}
