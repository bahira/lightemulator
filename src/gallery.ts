/**
 * Galerie d'exemples partageables.
 *
 * Chaque preset n'est **que** l'URL du lab : les paramètres listés dans
 * `HASH_KEYS_BY_LAB` sont lus par les labs via `useHashParam` / `useHashNumber`,
 * donc un lien suffit à restaurer la scène à l'identique.
 *
 * Ajouter un preset = ajouter un objet ci-dessous (aucun code de rendu).
 */

export type LabId =
  | 'bpm' | 'processor' | 'ising' | 'kan' | 'fep' | 'validate' | 'spear' | 'control'
  | 'dispersion' | 'drones' | 'pnn' | 'holo' | 'quantum' | 'langevin' | 'opto' | 'inference';

/** Clés d'URL reconnues par chaque lab — sert aussi à nettoyer le hash au changement d'onglet. */
export const HASH_KEYS_BY_LAB: Partial<Record<LabId, string[]>> = {
  bpm: ['blam', 'bdn', 'bgap'],
  processor: ['pn', 'ps'],
  ising: ['gkind', 'gn', 'gp', 'gnz'],
  dispersion: ['verre', 'l0'],
  quantum: ['qr', 'qa', 'qb'],
  holo: ['hit'],
};

export interface Preset {
  id: string;
  lab: LabId;
  title: string;
  blurb: string;
  /** query string sans `#` */
  hash: string;
  glyph: string;
}

export const PRESETS: Preset[] = [
  {
    id: 'coupler',
    lab: 'bpm',
    glyph: '≈',
    title: 'Coupleur directionnel — un battement complet',
    blurb: 'λ = 1,55 µm, Δn = 0,012, écart 3 µm : la lumière passe d’un guide à l’autre et revient.',
    hash: 'blam=1.55&bdn=0.012&bgap=3',
  },
  {
    id: 'oband',
    lab: 'bpm',
    glyph: '≈',
    title: 'O-band — 1,31 µm, couplage plus court',
    blurb: 'Longueur d’onde télécom et contraste plus fort : la période de battement se raccourcit.',
    hash: 'blam=1.31&bdn=0.02&bgap=2',
  },
  {
    id: 'mesh6',
    lab: 'processor',
    glyph: '⧉',
    title: 'Maillage MZI 6×6 — 15 cellules',
    blurb: 'Décomposition de Reck d’une unitaire 6×6 : 15 MZI, profondeur 5, unitarité à 1e-15.',
    hash: 'pn=6&ps=7',
  },
  {
    id: 'mesh24',
    lab: 'processor',
    glyph: '⧉',
    title: 'Maillage 18×18 — le coût de la profondeur',
    blurb: '153 MZI : le produit matrice-vecteur reste exact, c’est la profondeur du circuit qui explose.',
    hash: 'pn=18&ps=3',
  },
  {
    id: 'ising-sf',
    lab: 'ising',
    glyph: '⬡',
    title: 'Ising scale-free, au-dessus du seuil',
    blurb: '60 nœuds, pompe 1,1 : le CIM atteint l’optimum exact (comparé par force brute).',
    hash: 'gkind=scalefree&gn=60&gp=1.1',
  },
  {
    id: 'ising-sub',
    lab: 'ising',
    glyph: '⬡',
    title: 'Ising sous le seuil — le CIM décroche',
    blurb: 'Erdős–Rényi 100 nœuds, pompe 0,8 : le ratio au meilleur connu chute. C’est la limite honnête du dispositif.',
    hash: 'gkind=erdos&gn=100&gp=0.8',
  },
  {
    id: 'zdw',
    lab: 'dispersion',
    glyph: '∿',
    title: 'Silice — zéro de dispersion à 1,27 µm',
    blurb: 'Le GDD s’annule près de 1,27 µm : c’est le point où l’impulsion ne s’élargit plus au 1er ordre.',
    hash: 'verre=silice&l0=1.27',
  },
  {
    id: 'chsh',
    lab: 'quantum',
    glyph: 'Ψ',
    title: 'HOM parfait + CHSH à 2√2',
    blurb: 'R = 50 %, angles 0/22,5/45/67,5° : dip P(1,1) = 0 exact et S = 2,828 mesuré.',
    hash: 'qr=0.5&qa=0&qb=22.5',
  },
  {
    id: 'hom-bias',
    lab: 'quantum',
    glyph: 'Ψ',
    title: 'Séparateur 70/30 — le dip remonte',
    blurb: 'Même photons indiscernables, mais R ≠ ½ : la probabilité de coïncidence n’est plus nulle.',
    hash: 'qr=0.7',
  },
  {
    id: 'gs80',
    lab: 'holo',
    glyph: '⊛',
    title: 'Gerchberg–Saxton — 80 itérations',
    blurb: 'Deux fois plus d’itérations que le défaut : regardez la RMSE et l’efficacité η converger.',
    hash: 'hit=80',
  },
];

/** Hash complet d'un preset : le lab d'abord, puis ses paramètres. */
export function presetHash(p: Preset): string {
  return `lab=${p.lab}&${p.hash}`;
}

export function presetUrl(p: Preset): string {
  const base = typeof window === 'undefined' ? '' : window.location.href.split('#')[0];
  return `${base}#${presetHash(p)}`;
}
