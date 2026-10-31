/*
 * wasm_harness.mjs — compiler C→wasm, instanciation du module d'inférence,
 * génération de texte et petit protocole de mesure apparié.
 *
 * Le module `lm_c/lm_wasm.c` n'était auparavant reconstructible qu'avec une
 * toolchain wasm32-wasip1 installée à la main ; on accepte maintenant, dans
 * l'ordre : `$ZIG_CC`, `zig`, `python3 -m ziglang` (pip), `emcc`, `clang`.
 * Si aucune n'est là, les scripts le disent et sortent en code 0 (rien de faux).
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = join(dirname(fileURLToPath(import.meta.url)), '..');

export const EXPORTS = [
  'wasm_mem_weights', 'wasm_qbits',
  'wasm_scratch', 'wasm_init', 'wasm_arena', 'wasm_refresh', 'wasm_vocab',
  'wasm_quant', 'wasm_seed', 'wasm_gen', 'wasm_out',
];

const WASI_STUBS = [
  'args_get', 'args_sizes_get', 'environ_get', 'environ_sizes_get', 'clock_time_get',
  'random_get', 'fd_seek', 'fd_close', 'fd_fdstat_get', 'fd_read', 'proc_exit',
  'fd_prestat_get', 'fd_prestat_dir_name',
];

export function importsFor(exRef) {
  const imports = {
    wasi_snapshot_preview1: {
      fd_write: (_fd, iovsPtr, iovsLen, nWrittenPtr) => {
        const dv = new DataView(exRef.ex.memory.buffer);
        let total = 0;
        for (let i = 0; i < iovsLen; i++) total += dv.getUint32(iovsPtr + i * 8 + 4, true);
        dv.setUint32(nWrittenPtr, total, true);
        return 0;
      },
    },
  };
  for (const n of WASI_STUBS) imports.wasi_snapshot_preview1[n] = () => 0;
  return imports;
}

/* --------------------------------------------------------------- compilateur */
export function findCompiler() {
  const env = process.env.ZIG_CC;
  if (env && existsSync(env.split(' ')[0])) return { cmd: env, kind: 'zig' };

  const tryCmd = (cmd, args) => {
    const r = spawnSync(cmd, args, { encoding: 'utf8' });
    return r.status === 0 ? (r.stdout || '').trim() : null;
  };
  if (tryCmd('zig', ['version'])) return { cmd: 'zig', kind: 'zig' };
  // paquet pip : `python3 -m ziglang cc ...`
  for (const py of ['python3', 'python']) {
    if (tryCmd(py, ['-m', 'ziglang', 'version'])) return { cmd: `${py} -m ziglang`, kind: 'zig' };
  }
  if (tryCmd('emcc', ['--version'])) return { cmd: 'emcc', kind: 'emcc' };
  if (tryCmd('clang', ['--version'])) return { cmd: 'clang', kind: 'clang' };
  return null;
}

/* Compile lm_c/lm_wasm.c en wasm32-wasi. Retourne le chemin du .wasm. */
export function buildWasm({ out, defines = [], compiler = findCompiler() } = {}) {
  if (!compiler) throw new Error('aucun compilateur C→wasm (zig / emcc / clang+wasm-ld)');
  const dst = out || join(mkdtempSync(join(tmpdir(), 'lm-wasm-')), 'lm_infer.wasm');
  const defs = defines.map((d) => `-D${d}`).join(' ').split(' ').filter(Boolean);
  const exportsFlags = EXPORTS.map((e) => `-Wl,--export=${e}`);
  const src = join(root, 'lm_c/lm_wasm.c');
  const small = ['-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-Wl,--strip-all'];

  let cmd, args;
  if (compiler.kind === 'zig') {
    cmd = compiler.cmd.split(' ')[0];
    args = [...compiler.cmd.split(' ').slice(1), 'cc', '-target', 'wasm32-wasi', '-O3', '-msimd128',
      ...defs, ...small, '-Wl,--no-entry', '-o', dst, src, ...exportsFlags];
  } else if (compiler.kind === 'emcc') {
    cmd = 'emcc';
    args = ['-O3', '-msimd128', ...defs, '-o', dst, src,
      '-sSTANDALONE_WASM=1', '-sERROR_ON_UNDEFINED_SYMBOLS=0', ...EXPORTS.map((e) => `-sEXPORTED_FUNCTIONS=_${e}`)];
  } else {
    cmd = 'clang';
    args = ['--target=wasm32-wasi', '-O3', '-msimd128', ...defs, ...small, '-nostartfiles',
      '-Wl,--no-entry', '-o', dst, src, ...exportsFlags];
  }
  // emcc / clang : `-Wl,` n'a pas de sens, on passe les exports autrement
  if (compiler.kind === 'clang') args = args.filter((a) => !a.startsWith('-Wl,--gc-sections') && !a.startsWith('-Wl,--strip-all'));
  execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  return { path: dst, bytes: readFileSync(dst).length, compiler };
}

/* ------------------------------------------------------------- instanciation */
export function loadWasm(path, { ck = join(root, 'public/lm_checkpoint.bin'), vocab = join(root, 'public/lm_vocab.bin') } = {}) {
  const ckBuf = readFileSync(ck);
  const vocabBuf = readFileSync(vocab);
  const module = new WebAssembly.Module(readFileSync(path));
  const names = WebAssembly.Module.exports(module).map((e) => e.name);
  const missing = EXPORTS.filter((e) => !names.includes(e));
  if (missing.length) throw new Error(`exports manquants dans ${path} : ${missing.join(', ')}`);

  const ref = {};
  const instance = new WebAssembly.Instance(module, importsFor(ref));
  const ex = instance.exports;
  ref.ex = ex;

  new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024).set(vocabBuf, 0);
  ex.wasm_vocab(ex.wasm_scratch(), vocabBuf.length);
  const nParams = ex.wasm_init();
  const arena = ex.wasm_arena();
  const dv = new DataView(ckBuf.buffer, ckBuf.byteOffset, ckBuf.byteLength);
  const mem = new Float32Array(ex.memory.buffer, arena, nParams);
  for (let i = 0; i < nParams; i++) mem[i] = dv.getFloat32(4 + i * 4, true);
  ex.wasm_refresh();

  const gen = (prompt, n, temp = 0, seed = 42) => {
    const scr = new Uint8Array(ex.memory.buffer, ex.wasm_scratch(), 1024);
    for (let i = 0; i < prompt.length; i++) scr[128 + i] = prompt.charCodeAt(i) & 0xff;
    ex.wasm_seed(BigInt(seed));
    const t0 = performance.now();
    const len = ex.wasm_gen(ex.wasm_scratch() + 128, prompt.length, n, temp);
    const ms = performance.now() - t0;
    const txt = new TextDecoder('latin1').decode(new Uint8Array(ex.memory.buffer, ex.wasm_out(), len));
    return { txt, n: len, ms, tps: len / ms * 1000 };
  };
  return { ex, gen, nParams, size: readFileSync(path).length };
}

/* ------------------------------------------------- mesure appariée (2 modules) */
export function median(a) { return a.slice().sort((x, y) => x - y)[a.length >> 1]; }

export function compare(A, B, { prompt = 'Once upon a', tokens = 160, rounds = 7 } = {}) {
  const ta = [], tb = [];
  let txtA = '', txtB = '';
  for (let r = 0; r < rounds; r++) {
    const first = r % 2 === 0;
    const o1 = (first ? A : B).gen(prompt, tokens, 0.0);
    const o2 = (first ? B : A).gen(prompt, tokens, 0.0);
    ta.push(first ? o1.tps : o2.tps);
    tb.push(first ? o2.tps : o1.tps);
    txtA = first ? o1.txt : o2.txt;
    txtB = first ? o2.txt : o1.txt;
  }
  // ratio apparié B/A : > 1 ⇒ B plus rapide que A
  const ratio = tb.map((v, i) => v / ta[i]).sort((x, y) => x - y);
  const mean = ratio.reduce((s, v) => s + v, 0) / ratio.length;
  const sd = Math.sqrt(ratio.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, ratio.length - 1));
  return {
    a: median(ta), b: median(tb), ratio: mean, /* B/A */
    sigma: sd > 0 ? Math.abs(mean - 1) / sd : 99,
    identical: txtA === txtB, txtA, txtB,
  };
}

export function writeIfChanged(path, bytes) {
  const old = existsSync(path) ? readFileSync(path) : null;
  if (old && old.equals(bytes)) return false;
  writeFileSync(path, bytes);
  return true;
}
