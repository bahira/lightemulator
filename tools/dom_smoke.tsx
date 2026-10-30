/*
 * Smoke test DOM réel (jsdom) — va plus loin que tools/smoke_labs.tsx (SSR) :
 * les useEffect tournent, les canvas sont dessinés (context 2D factice), on
 * clique vraiment dans l'UI.
 *
 *   npm run smoke:dom
 *
 * Ce qui est vérifié :
 *   1. les 16 labs montent ET dessinent au moins une frame sans exception
 *   2. le shell (App) se monte, on traverse les 16 onglets par clic
 *   3. la galerie s'ouvre, un preset réécrit le hash, change d'onglet et
 *      remonte le lab (le lab relit donc bien l'état depuis l'URL)
 *   4. le bouton « partager » copie l'URL courante (hash inclus)
 */
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'http://localhost:5173/',
  pretendToBeVisual: true,
});
const { window } = dom;

// ---------------------------------------------------------------- DOM stubs
function makeCtx(canvas: any) {
  const grad = { addColorStop() {} };
  const base: any = {
    canvas,
    measureText: () => ({ width: 10 }),
    createLinearGradient: () => grad,
    createRadialGradient: () => grad,
    createPattern: () => null,
    getImageData: (_x: number, _y: number, w: number, h: number) => ({
      data: new Uint8ClampedArray(Math.max(1, w * h * 4)),
      width: w,
      height: h,
    }),
    createImageData: (w: any, h?: any) => {
      const width = typeof w === 'number' ? w : (w?.width ?? 1);
      const height = typeof h === 'number' ? h : (w?.height ?? 1);
      return { data: new Uint8ClampedArray(Math.max(1, width * height * 4)), width, height };
    },
  };
  return new Proxy(base, {
    get(t, k) {
      if (k in t) return (t as any)[k];
      return () => {}; // toute méthode 2D non listée
    },
    set(t, k, v) {
      (t as any)[k] = v;
      return true;
    },
  });
}

class FakeImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  constructor(a: any, b: any, c?: any) {
    if (typeof a === 'number') {
      this.width = a;
      this.height = b;
      this.data = new Uint8ClampedArray(a * b * 4);
    } else {
      this.data = a;
      this.width = b;
      this.height = c ?? a.length / 4 / b;
    }
  }
}

(window.HTMLCanvasElement.prototype as any).getContext = function () {
  return makeCtx(this);
};
(window as any).ImageData = FakeImageData;
(window as any).ResizeObserver = class {
  constructor(private cb: any) {}
  observe(el: any) {
    // une seule notification, comme un vrai ResizeObserver au premier layout
    this.cb([{ target: el, contentRect: { width: 800, height: 400 } }]);
  }
  unobserve() {}
  disconnect() {}
};
(window as any).matchMedia = () => ({
  matches: false,
  addEventListener() {},
  removeEventListener() {},
  addListener() {},
  removeListener() {},
});
Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', { get: () => 800, configurable: true });
Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', { get: () => 400, configurable: true });
Object.defineProperty(window, 'devicePixelRatio', { value: 1, configurable: true });

let rafSeq = 0;
(window as any).requestAnimationFrame = (cb: any) => setTimeout(() => cb(performance.now()), 0) as unknown as number;
(window as any).cancelAnimationFrame = (id: any) => clearTimeout(id);
void rafSeq;

const clipboard: string[] = [];
Object.defineProperty(window.navigator, 'clipboard', {
  value: { writeText: async (s: string) => { clipboard.push(s); } },
  configurable: true,
});

// ------------------------------------------------------------ globals Node
const g = globalThis as any;
for (const k of [
  'window', 'document', 'navigator', 'HTMLElement', 'HTMLCanvasElement', 'Element', 'Node',
  'Event', 'MouseEvent', 'KeyboardEvent', 'ResizeObserver', 'ImageData', 'getComputedStyle',
  'requestAnimationFrame', 'cancelAnimationFrame', 'localStorage', 'devicePixelRatio',
]) {
  if ((window as any)[k] === undefined) continue;
  try {
    g[k] = (window as any)[k];
  } catch {
    Object.defineProperty(g, k, { value: (window as any)[k], configurable: true, writable: true });
  }
}
g.IS_REACT_ACT_ENVIRONMENT = true;

const errors: string[] = [];
const origError = console.error;
console.error = (...args: any[]) => {
  errors.push(args.map(String).join(' '));
  origError(...args);
};

// ---------------------------------------------------------------- React
const { act } = await import('react');
const { createElement } = await import('react');
const { createRoot } = await import('react-dom/client');
const { default: App } = await import('../src/App');

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const root = createRoot(window.document.getElementById('root')!);

type LabLoader = () => Promise<{ default: any }>;
const labs: [string, LabLoader][] = [
  ['BpmLab', () => import('../src/labs/BpmLab')],
  ['ProcessorLab', () => import('../src/labs/ProcessorLab')],
  ['IsingLab', () => import('../src/labs/IsingLab')],
  ['KanLab', () => import('../src/labs/KanLab')],
  ['FreeEnergyLab', () => import('../src/labs/FreeEnergyLab')],
  ['ValidationLab', () => import('../src/labs/ValidationLab')],
  ['SpearLab', () => import('../src/labs/SpearLab')],
  ['ControlLab', () => import('../src/labs/ControlLab')],
  ['DispersionLab', () => import('../src/labs/DispersionLab')],
  ['DroneLab', () => import('../src/labs/DroneLab')],
  ['PnnLab', () => import('../src/labs/PnnLab')],
  ['HoloLab', () => import('../src/labs/HoloLab')],
  ['QuantumLab', () => import('../src/labs/QuantumLab')],
  ['LangevinLab', () => import('../src/labs/LangevinLab')],
  ['OptoTransformerLab', () => import('../src/labs/OptoTransformerLab')],
  ['InferenceLab', () => import('../src/labs/InferenceLab')],
];

let failures = 0;
const report = (ok: boolean, label: string, detail = '') => {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
};

// ------------------------------------------------- 1. chaque lab, dessiné
for (const [name, load] of labs) {
  errors.length = 0;
  const mod = await load();
  await act(async () => {
    root.render(createElement(mod.default));
  });
  await act(async () => { await sleep(24); });
  const html = window.document.getElementById('root')!.innerHTML;
  const drew = html.includes('<canvas') || html.length > 500;
  await act(async () => { root.render(createElement('div')); });
  await sleep(0);
  report(html.length > 200 && drew && errors.length === 0, `${name}`, `${html.length} chars${errors.length ? ' · ' + errors[0].slice(0, 120) : ''}`);
}

// ------------------------------------------------------ 2-4. le shell complet
errors.length = 0;
await act(async () => {
  root.render(createElement(App));
});
await act(async () => { await sleep(40); });

const doc = window.document;
const tabButtons = [...doc.querySelectorAll('nav button')] as HTMLButtonElement[];
report(tabButtons.length === 16, 'shell : 16 onglets rendus', `${tabButtons.length} boutons`);

let tabsOk = 0;
for (const btn of tabButtons) {
  await act(async () => { btn.click(); });
  await act(async () => { await sleep(18); });
  if (doc.querySelector('canvas') || doc.body.innerHTML.length > 2000) tabsOk++;
}
report(tabsOk === tabButtons.length, 'shell : les 16 onglets s’ouvrent par clic', `${tabsOk}/16`);

// galerie : ouverture par le bouton, puis sélection d'un preset
const galleryBtn = [...doc.querySelectorAll('button')].find((b) => (b.textContent ?? '').includes('galerie'))!;
report(!!galleryBtn, 'galerie : bouton présent dans l’en-tête');
await act(async () => { galleryBtn.click(); });
await act(async () => { await sleep(20); });
const dialog = doc.querySelector('[role="dialog"]');
report(!!dialog, 'galerie : modale ouverte');

const presetCards = dialog ? [...dialog.querySelectorAll('button')].filter((b) => (b.textContent ?? '').length > 12) : [];
report(presetCards.length >= 10, 'galerie : presets listés', `${presetCards.length} cartes`);

const chsh = presetCards.find((b) => (b.textContent ?? '').includes('CHSH'));
if (chsh) {
  await act(async () => { chsh.click(); });
  await act(async () => { await sleep(40); });
  const hash = window.location.hash;
  report(hash.includes('lab=quantum'), 'preset CHSH : lab épinglé dans l’URL', hash);
  const activeTab = [...doc.querySelectorAll('nav button')].find((b) => b.className.includes('border-cyan-400'));
  report((activeTab?.textContent ?? '').includes('Optique quantique'), 'preset CHSH : onglet changé', activeTab?.textContent ?? '—');
  report(!doc.querySelector('[role="dialog"]'), 'galerie : refermée après sélection');
} else {
  report(false, 'preset CHSH introuvable dans la galerie');
}

// un preset non-défaut doit laisser son paramètre dans l'URL
const openGallery = () => [...doc.querySelectorAll('button')].find((b) => (b.textContent ?? '').includes('galerie'))!;
await act(async () => { openGallery().click(); });
await act(async () => { await sleep(20); });
const dlg2 = doc.querySelector('[role="dialog"]');
const bias = dlg2 ? [...dlg2.querySelectorAll('button')].find((b) => (b.textContent ?? '').includes('70/30')) : undefined;
if (bias) {
  await act(async () => { bias.click(); });
  await act(async () => { await sleep(40); });
  report(window.location.hash.includes('qr=0.7'), 'preset 70/30 : paramètre conservé dans l’URL', window.location.hash);
  const activeTab2 = [...doc.querySelectorAll('nav button')].find((b) => b.className.includes('border-cyan-400'));
  report((activeTab2?.textContent ?? '').includes('Optique quantique'), 'preset 70/30 : reste sur le bon lab');
} else {
  report(false, 'preset 70/30 introuvable');
}

// bouton partager : l'URL copiée contient le hash courant
const shareBtn = [...doc.querySelectorAll('button')].find((b) => (b.textContent ?? '').includes('partager'))!;
if (shareBtn) {
  clipboard.length = 0;
  await act(async () => { shareBtn.click(); });
  await act(async () => { await sleep(10); });
  report(clipboard.length === 1 && clipboard[0].includes('#lab=quantum'), 'partager : URL (avec hash) copiée', clipboard[0] ?? '—');
} else {
  report(false, 'bouton partager introuvable');
}

await act(async () => { root.render(createElement('div')); });
await sleep(0);

// -------------------------------------------- 5. deep-link : un lien rouvre le lab
{
  window.history.replaceState(null, '', '#lab=holo&hit=80');
  const host = window.document.createElement('div');
  window.document.body.appendChild(host);
  const root2 = createRoot(host);
  errors.length = 0;
  await act(async () => { root2.render(createElement(App)); });
  await act(async () => { await sleep(40); });
  const active = [...host.querySelectorAll('nav button')].find((b) => b.className.includes('border-cyan-400'));
  report((active?.textContent ?? '').includes('Holographie'), 'deep-link : #lab=holo rouvre le bon lab', active?.textContent ?? '—');
  report(window.location.hash.includes('lab=holo'), 'deep-link : le hash est préservé', window.location.hash);
  await act(async () => { root2.render(createElement('div')); });
  host.remove();
  await sleep(0);
}

if (errors.length) {
  console.log('\nErreurs console React :');
  for (const e of errors.slice(0, 5)) console.log('  ' + e.slice(0, 200));
}

if (failures) {
  console.error(`\n✗ ${failures} vérification(s) en échec`);
  process.exit(1);
}
console.log(`\n✅ 16 labs dessinés + shell, galerie et partage vérifiés (jsdom)`);
process.exit(0);
