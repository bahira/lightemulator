#!/usr/bin/env node
/*
 * Re-mesure les affirmations du README **sur cette machine** et écrit CLAIMS.md.
 *
 *   npm run claims          # affiche le tableau
 *   npm run claims:write    # écrit/replace CLAIMS.md
 *   npm run claims -- --full  # + entraînement C (60 steps) et gradcheck
 *
 * Pourquoi : les gains des kernels dépendent du CPU, du compilateur et des flags.
 * Le README annonce des valeurs mesurées sur un i7-7660U (2016) ; sur un CPU
 * récent, `exp` scalaire peut être PLUS LENT que libm. Ce script rend l'écart
 * visible au lieu de le laisser pourrir dans un tableau.
 */
import { execFileSync, execSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir, cpus, platform, release } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const full = process.argv.includes('--full');
const write = process.argv.includes('--write');
const rows = [];
const row = (claim, measured, note = '') => rows.push({ claim, measured, note });

const sh = (cmd, opts = {}) => {
  try {
    return execSync(cmd, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts });
  } catch (e) {
    return (e.stdout ?? '') + (e.stderr ?? '');
  }
};

const cpu = cpus()[0]?.model?.trim() ?? 'inconnu';
const gcc = sh('gcc -dumpfullversion -dumpversion').trim().split('\n')[0] || 'absent';
const env = {
  cpu,
  cores: cpus().length,
  os: `${platform()} ${release()}`,
  node: process.version,
  gcc,
  date: new Date().toISOString().slice(0, 10),
};

// ---------------------------------------------------------------- 1. tests
if (existsSync(join(root, 'node_modules'))) {
  const testOut = sh('npm run test --silent');
  const mTests = testOut.match(/VALIDÉ — (\d+) tests en (\d+) ms/);
  row('boucle grounded (tests)', mTests ? `${mTests[1]}/${mTests[1]} PASS en ${mTests[2]} ms` : 'échec', 'npm run test');
} else {
  row('boucle grounded (tests)', 'non mesuré', 'npm ci absent');
}

// ------------------------------------------------------- 2. kernels C (gcc)
let kernelsOk = false;
try {
  const dir = mkdtempSync(join(tmpdir(), 'claims-'));
  const flags = ['-O2', '-march=native', '-ffast-math', '-fopenmp'];
  execFileSync('gcc', [...flags, '-o', join(dir, 'verify'), join(root, 'lm_c/verify_kernels.c'), '-lm'], { stdio: 'ignore' });
  execFileSync('gcc', [...flags, '-o', join(dir, 'bench'), join(root, 'lm_c/lm_bench_exp8.c'), '-lm'], { stdio: 'ignore' });
  const verify = execFileSync(join(dir, 'verify'), { encoding: 'utf8' });
  const bench = execFileSync(join(dir, 'bench'), { encoding: 'utf8' });
  kernelsOk = true;

  for (const name of ['exp', 'rsqrt', 'gelu']) {
    const m = verify.match(new RegExp(`${name}\\s*:\\s*(?:L∞ rel|écart valeur L∞)\\s*=\\s*([0-9.e+-]+)`));
    if (m) row(`exactitude kernel ${name}`, `L∞ ${m[1]}`, 'lm_c/verify_kernels.c');
  }
  const mAvx = bench.match(/spear AVX2 x8[^\n]*×([\d.]+)/);
  const mScal = bench.match(/spear scal[^\n]*×([\d.]+)/);
  const mInf = bench.match(/L∞ rel vs expf = ([0-9.e+-]+)/);
  if (mAvx) row('exp8 AVX2 vs libm expf', `×${mAvx[1]}`, '20M appels, lm_c/lm_bench_exp8.c');
  if (mScal) row('exp scalaire vs libm expf', `×${mScal[1]}`, 'dépend du CPU — voir README');
  if (mInf) row('exp8 AVX2 — erreur', `L∞ ${mInf[1]}`, 'tolérance CI 1e-3');
} catch {
  row('kernels C', 'non mesuré (gcc indisponible)', 'gcc nécessaire');
}

// ---------------------------------------- 2b. GEMM micro-kernel 4 lignes M
try {
  const dir = mkdtempSync(join(tmpdir(), 'claims-mm-'));
  execFileSync('gcc', ['-O2', '-march=native', '-ffast-math', '-o', join(dir, 'mm'), join(root, 'bench/bench_mm_isa.c'), '-lm'], { stdio: 'ignore' });
  const out = execFileSync(join(dir, 'mm'), { encoding: 'utf8' });
  const num = (re) => { const m = out.match(re); return m ? Number(m[1]) : null; };
  const scal = num(/scalaire\s+([\d.]+) ms/);
  const base = num(/AVX2 4x8\s+([\d.]+) ms/);
  const mb4 = num(/AVX2 4Mx16\s+([\d.]+) ms/);
  const w512 = num(/AVX512 4Mx64\s+([\d.]+) ms/);
  const naive = num(/AVX512 4x16\s+([\d.]+) ms/);
  if (base && mb4) row('GEMM 4 lignes M (AVX2 4M×16)', `×${(base / mb4).toFixed(2)} vs noyau 4×8`, 'bench/bench_mm_isa.c, 5 formes réelles');
  if (base && w512) row('GEMM 4 lignes M (AVX-512 4M×64)', `×${(base / w512).toFixed(2)} vs 4×8`, 'max|diff| = 0 vs scalaire');
  if (scal && w512) row('GEMM meilleur vs scalaire', `×${(scal / w512).toFixed(1)}`, '5 formes, 9 rounds × 200 reps');
  if (base && naive) row('AVX-512 sans blocage (4×16)', `×${(base / naive).toFixed(2)} vs 4×8`, 'contre-exemple : la largeur SIMD seule ne paie pas');
} catch {
  row('GEMM micro-kernel', 'non mesuré (gcc indisponible)', 'bench/bench_mm_isa.c');
}

// ------------------------------------------- 3. inférence wasm (tok/s + V)
try {
  const wasm = readFileSync(join(root, 'public/lm_infer.wasm'));
  const ck = readFileSync(join(root, 'public/lm_checkpoint.bin'));
  const vocab = readFileSync(join(root, 'public/lm_vocab.bin'));
  let ex = null;
  const imports = {
    wasi_snapshot_preview1: {
      fd_write: (_fd, iovsPtr, iovsLen, nWrittenPtr) => {
        const dv = new DataView(ex.memory.buffer);
        let total = 0;
        for (let i = 0; i < iovsLen; i++) total += dv.getUint32(iovsPtr + i * 8 + 4, true);
        dv.setUint32(nWrittenPtr, total, true);
        return 0;
      },
    },
  };
  for (const n of ['args_get', 'args_sizes_get', 'environ_get', 'environ_sizes_get', 'clock_time_get', 'random_get', 'fd_seek', 'fd_close', 'fd_fdstat_get', 'fd_read', 'proc_exit', 'fd_prestat_get', 'fd_prestat_dir_name'])
    imports.wasi_snapshot_preview1[n] = () => 0;
  const { instance } = await WebAssembly.instantiate(wasm, imports);
  ex = instance.exports;
  const scr = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
  scr.set(new Uint8Array(vocab), 0);
  ex.wasm_vocab(ex.wasm_scratch(), vocab.length);
  const nParams = ex.wasm_init();
  const arena = ex.wasm_arena();
  const mem = new Float32Array(ex.memory.buffer, arena, nParams);
  const dv = new DataView(ck.buffer, ck.byteOffset, ck.byteLength);
  const V = dv.getInt32(0, true);
  for (let i = 0; i < nParams; i++) mem[i] = dv.getFloat32(4 + i * 4, true);
  ex.wasm_refresh();

  const p = 'Once upon a';
  const s2 = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
  for (let i = 0; i < p.length; i++) s2[128 + i] = p.charCodeAt(i) & 0xff;
  let best = Infinity;
  let n = 0;
  for (let k = 0; k < 3; k++) {
    ex.wasm_seed(42n);
    const t0 = performance.now();
    n = ex.wasm_gen(ex.wasm_scratch() + 128, p.length, 200, 0.8);
    best = Math.min(best, performance.now() - t0);
  }
  row('inférence wasm — débit', `${(n / best * 1000).toFixed(0)} tok/s`, `V=${V}, ${nParams} params, ${(wasm.length / 1024).toFixed(1)} kB`);
} catch (e) {
  row('inférence wasm', 'non mesurée', String(e?.message ?? e).slice(0, 80));
}

// ------------------------------------------------- 4. taille du build web
try {
  const dist = join(root, 'dist/index.html');
  if (existsSync(dist)) {
    const size = readFileSync(dist).length;
    row('build web (single-file)', `${(size / 1024).toFixed(0)} kB`, 'dist/index.html (npm run build)');
  } else {
    row('build web (single-file)', 'non mesuré', 'lancer npm run build');
  }
} catch { /* ignore */ }

// --------------------------------------- 5. entraînement + gradcheck (option)
if (full && kernelsOk) {
  try {
    const dir = mkdtempSync(join(tmpdir(), 'claims-train-'));
    const bin = join(dir, 'lm_train');
    execFileSync('gcc', ['-O2', '-march=native', '-ffast-math', '-fopenmp', '-o', bin, join(root, 'lm_c/lm_main.c'), '-lm'], { stdio: 'ignore' });
    // cwd = dossier temporaire : le trainer écrit `lm_c/checkpoint.bin` en relatif,
    // il ne doit surtout pas écraser le checkpoint versionné de public/.
    const out = execFileSync(bin, ['--data', join(root, 'ci-data/tinystories_ci.txt'), '--steps', '60', '--gradcheck', '--nogen'], { encoding: 'utf8', cwd: dir, env: { ...process.env, GOMP_SPINCOUNT: '0' } });
    const tps = out.match(/([0-9]+) tok\/s/g);
    const gc = out.match(/erreur moyenne norm.*?=\s*([0-9.]+)/);
    const val = out.match(/val loss FP32 : ([0-9.]+)/);
    if (gc) row('gradcheck (FD vs backward analytique)', `erreur moyenne ${gc[1]}`, '40 paramètres');
    if (val) row('training smoke 60 steps (32 kB)', `val loss ${val[1]}`, 'hasard ≈ 4,08');
    if (tps && tps.length) row('débit entraînement (60 steps)', tps[tps.length - 1], 'GEMM 4 lignes M + AVX-512');
  } catch {
    row('training smoke', 'non mesuré', 'échec de compilation/exécution');
  }
}

// ------------------------------------------------------------------ sortie
const md = [
  '# CLAIMS — re-mesuré sur cette machine',
  '',
  `Généré par \`npm run claims${full ? ' -- --full' : ''}\` le ${env.date}.`,
  '',
  `| environnement | valeur |`,
  `|---|---|`,
  `| CPU | ${env.cpu} |`,
  `| cœurs | ${env.cores} |`,
  `| OS | ${env.os} |`,
  `| Node | ${env.node} |`,
  `| gcc | ${env.gcc} |`,
  '',
  '| affirmation | mesuré ici | source |',
  '|---|---|---|',
  ...rows.map((r) => `| ${r.claim} | **${r.measured}** | ${r.note} |`),
  '',
  '> Les gains des kernels dépendent du CPU, du compilateur et des flags. Un chiffre',
  '> du README qui ne se reproduit pas ici n\'est pas faux : il est daté. C\'est la',
  '> raison d\'être de ce fichier.',
  '',
].join('\n');

console.log(md);
if (write) {
  writeFileSync(join(root, 'CLAIMS.md'), md);
  console.log('✓ CLAIMS.md écrit');
}
