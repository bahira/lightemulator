/*
 * Smoke test headless des 16 labs : chaque lab doit se rendre au premier cycle
 * (aucune exception dans le corps du composant, aucun accès DOM non stubé).
 *
 *   npm run smoke:labs
 *
 * Le rendu est fait avec react-dom/server, donc AUCUN canvas n'est dessiné :
 * ça ne valide pas le pixels, ça valide que le lab monte sans crash et que les
 * hooks (état dans le hash d'URL, useCanvas, etc.) sont appelables hors navigateur.
 * Les API navigateur minimales utilisées par src/ui/kit.tsx sont stubées ci-dessous.
 */
const g = globalThis as any;

if (typeof g.window === 'undefined') {
  g.window = {
    innerWidth: 1440,
    innerHeight: 900,
    devicePixelRatio: 1,
    location: { hash: '', pathname: '/', search: '', href: 'http://localhost/' },
    history: { replaceState() {} },
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    addEventListener() {},
    removeEventListener() {},
    localStorage: { getItem: () => null, setItem() {} },
  };
  g.document = {
    addEventListener() {},
    removeEventListener() {},
    documentElement: { style: {} },
    body: {},
    createElement: () => ({ style: {}, getContext: () => null, appendChild() {} }),
  };
  g.location = g.window.location;
  g.localStorage = g.window.localStorage;
  g.devicePixelRatio = 1;
  g.requestAnimationFrame = (cb: (t: number) => void) => setTimeout(() => cb(16), 0) as unknown as number;
  g.cancelAnimationFrame = () => {};
}

const { renderToString } = await import('react-dom/server');
const { createElement, Suspense } = await import('react');

const labs: [string, () => Promise<{ default: any }>][] = [
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

let fail = 0;
for (const [name, load] of labs) {
  try {
    const m = await load();
    const html = renderToString(createElement(m.default));
    if (html.length < 200) throw new Error(`rendu suspicieusement vide (${html.length} chars)`);
    console.log(`OK   ${name.padEnd(20)} ${html.length} chars`);
  } catch (e: any) {
    fail++;
    console.log(`FAIL ${name.padEnd(20)} ${e?.message ?? e}`);
  }
}

const { default: App } = await import('../src/App');
try {
  const html = renderToString(createElement(Suspense, { fallback: 'loading' }, createElement(App)));
  console.log(`OK   ${'App (shell)'.padEnd(20)} ${html.length} chars`);
} catch (e: any) {
  fail++;
  console.log(`FAIL ${'App'.padEnd(20)} ${e?.message ?? e}`);
}

if (fail) {
  console.error(`\n✗ ${fail} lab(s) ne montent pas`);
  process.exit(1);
}
console.log(`\n✅ ${labs.length} labs + shell se rendent (headless, sans canvas)`);
