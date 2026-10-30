import { useState } from 'react';
import { cn } from '../utils/cn';
import { PRESETS, presetUrl, type Preset } from '../gallery';
import { copyText } from './kit';

const LAB_LABEL: Record<string, string> = {
  bpm: 'Moteur de lumière',
  processor: 'Processeur MZI',
  ising: 'Machine d’Ising',
  kan: 'KAN photonique',
  fep: 'Énergie libre',
  validate: 'Validation',
  spear: 'Kernels SPEAR',
  control: 'Contrôle',
  dispersion: 'Dispersion',
  drones: 'Drones lumineux',
  pnn: 'PNN',
  holo: 'Holographie',
  quantum: 'Optique quantique',
  langevin: 'Langevin',
  opto: 'Opto-Transformer',
  inference: 'Inference edge',
};

/**
 * Galerie d'exemples : chaque carte est un lien. Sélectionner une carte réécrit
 * le hash, remonte le lab (les états lisent le hash au montage) et laisse
 * l'URL partageable telle quelle.
 */
export function Gallery({
  open,
  onClose,
  onPick,
}: {
  open: boolean;
  onClose: () => void;
  onPick: (p: Preset) => void;
}) {
  const [copied, setCopied] = useState<string | null>(null);
  if (!open) return null;

  const copy = (p: Preset) => {
    void copyText(presetUrl(p));
    setCopied(p.id);
    setTimeout(() => setCopied(null), 1400);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/65 px-4 py-10 backdrop-blur-sm"
      onMouseDown={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Galerie d'exemples"
    >
      <div
        className="w-full max-w-3xl overflow-hidden rounded-xl border border-white/10 bg-[#0a0f1e] shadow-[0_30px_80px_-20px_rgba(0,0,0,1)]"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/8 px-4 py-3">
          <div>
            <h2 className="font-mono text-[12px] font-semibold tracking-wide text-slate-100">
              ◇ Galerie d’exemples — chaque carte est un lien
            </h2>
            <p className="mt-0.5 text-[10px] text-slate-500">
              {PRESETS.length} scènes réglées, reproductibles par URL. Le hash contient l’état exact du lab.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="fermer"
            className="rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-slate-400 ring-1 ring-white/10 hover:bg-white/10 hover:text-slate-200"
          >
            esc
          </button>
        </div>

        <ul className="grid gap-2 p-3 sm:grid-cols-2">
          {PRESETS.map((p) => (
            <li key={p.id}>
              <div className="group flex h-full flex-col gap-1.5 rounded-lg border border-white/8 bg-white/[0.015] p-3 transition-colors hover:border-cyan-400/30 hover:bg-cyan-400/[0.04]">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] leading-none text-cyan-300/80">{p.glyph}</span>
                  <span className="font-mono text-[8.5px] uppercase tracking-widest text-slate-500">
                    {LAB_LABEL[p.lab] ?? p.lab}
                  </span>
                </div>
                <button
                  onClick={() => onPick(p)}
                  className="text-left text-[11px] font-semibold leading-snug text-slate-200 hover:text-cyan-200"
                >
                  {p.title}
                </button>
                <p className="text-[9.5px] leading-relaxed text-slate-500">{p.blurb}</p>
                <div className="mt-auto flex items-center gap-2 pt-1">
                  <code className="truncate rounded bg-black/40 px-1.5 py-0.5 font-mono text-[9px] text-cyan-200/70 ring-1 ring-white/6">
                    #{p.hash}
                  </code>
                  <button
                    onClick={() => copy(p)}
                    className={cn(
                      'ml-auto shrink-0 rounded px-1.5 py-0.5 font-mono text-[9px] ring-1 transition-colors',
                      copied === p.id
                        ? 'bg-cyan-400/15 text-cyan-200 ring-cyan-400/40'
                        : 'bg-white/[0.03] text-slate-400 ring-white/10 hover:text-slate-200',
                    )}
                  >
                    {copied === p.id ? '✓ copié' : '⧉ lien'}
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>

        <p className="border-t border-white/8 px-4 py-2 text-[9.5px] leading-relaxed text-slate-600">
          Les paramètres exportés sont ceux déclarés dans <code className="text-slate-400">src/gallery.ts</code>{' '}
          (<code className="text-slate-400">HASH_KEYS_BY_LAB</code>) — un lab sans clé exportée s’ouvre avec ses
          valeurs par défaut, jamais avec un état inventé.
        </p>
      </div>
    </div>
  );
}
