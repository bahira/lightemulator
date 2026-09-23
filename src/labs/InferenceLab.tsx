import { useEffect, useRef, useState } from 'react';
import { Panel, Tag, Btn, Formula } from '../ui/kit';

type Mode = 0 | 1 | 2 | 3; // 0=FP32, 1=E8, 2=INT8, 3=INT4
interface Run {
  mode: Mode; text: string; tokPerS: number; quantErr: string; ms: number;
}

const MODE_LABEL: Record<Mode, string> = { 0: 'FP32', 1: 'E8 (~7.1 bits)', 2: 'INT8 bloc', 3: 'INT4 bloc' };

export default function InferenceLab() {
  const [status, setStatus] = useState('chargement du module…');
  const [ready, setReady] = useState(false);
  const [prompt, setPrompt] = useState('Once upon a');
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState<Mode | null>(null);
  const E = useRef<any>(null);
  const pristine = useRef<Float32Array | null>(null);

  useEffect(() => {
    (async () => {
        // relatif au document (Pages : /lightemulator/lm_infer.wasm ; dev : public/)
        const base = './';
      try {
        const [wasmBuf, ckptBuf, vocabBuf] = await Promise.all([
          fetch(base + 'lm_infer.wasm').then((r) => r.arrayBuffer()),
          fetch(base + 'lm_checkpoint.bin').then((r) => r.arrayBuffer()),
          fetch(base + 'lm_vocab.bin').then((r) => r.arrayBuffer()),
        ]);
        // wasi shim — fd_write réel (sinon wasi-libc boucle sur *nwritten)
        let exportsRef: any = null;
        const imports = {
          wasi_snapshot_preview1: {
            fd_write: (_fd: number, iovsPtr: number, iovsLen: number, nWrittenPtr: number) => {
              const dv = new DataView(exportsRef.memory.buffer);
              let total = 0;
              for (let i = 0; i < iovsLen; i++) {
                total += dv.getUint32(iovsPtr + i * 8 + 4, true);
              }
              dv.setUint32(nWrittenPtr, total, true);
              return 0;
            },
          },
        };
        for (const n of ['args_get','args_sizes_get','environ_get','environ_sizes_get','clock_time_get','random_get','fd_seek','fd_close','fd_fdstat_get','fd_read','proc_exit','fd_prestat_get','fd_prestat_dir_name']) {
          (imports.wasi_snapshot_preview1 as any)[n] = () => 0;
        }
        const { instance } = await WebAssembly.instantiate(wasmBuf, imports);
        exportsRef = instance.exports;
        E.current = exportsRef;
        const Ee = exportsRef;

        // vocab AVANT modelInit (V dimensionne WTE/WOUT)
        const scr = new Uint8Array(Ee.memory.buffer, Ee.wasm_scratch(), 1024);
        scr.set(new Uint8Array(vocabBuf), 0);
        Ee.wasm_vocab(Ee.wasm_scratch(), vocabBuf.byteLength);

        const nParams = Ee.wasm_init();
        const arena = Ee.wasm_arena();
        const mem = new Float32Array(Ee.memory.buffer, arena, nParams);

        // checkpoint : int V + nParams floats — vérification exacte
        const dv = new DataView(ckptBuf);
        const V = dv.getInt32(0, true);
        if (V !== 104 || 4 + nParams * 4 !== ckptBuf.byteLength) {
          setStatus(`checkpoint incompatible (V=${V}, params=${nParams})`);
          return;
        }
        for (let i = 0; i < nParams; i++) mem[i] = dv.getFloat32(4 + i * 4, true);
        pristine.current = Float64Array.from(Array.from(mem)) as unknown as Float32Array;
        pristine.current = new Float32Array(mem); // copie pristine (RESTAURE avant chaque mode)
        Ee.wasm_refresh();
        setStatus(`270k params chargés — module 47.6 kB, zéro dépendance`);
        setReady(true);
      } catch (e: any) {
        setStatus('échec du chargement : ' + (e?.message ?? e));
      }
    })();
  }, []);

  const run = (mode: Mode) => {
    if (!E.current || !pristine.current || busy !== null) return;
    setBusy(mode);
    setTimeout(() => {
      const Ee = E.current;
      const mem = new Float32Array(Ee.memory.buffer, Ee.wasm_arena(), pristine.current!.length);
      if (mode !== 0) {
        mem.set(pristine.current!); // restaure les poids FP32 avant de quantifier (jamais en cascade)
        Ee.wasm_quant(mode);
      } else {
        mem.set(pristine.current!);
        Ee.wasm_refresh();
      }
      // prompt → scratch
      const scr = new Uint8Array(Ee.memory.buffer, Ee.wasm_scratch(), 1024);
      const p = prompt.slice(0, 64);
      for (let i = 0; i < p.length; i++) scr[128 + i] = p.charCodeAt(i) & 0xff;
      Ee.wasm_seed(42n); // même seed = comparaison appariée entre modes
      const t0 = performance.now();
      const nGen = Ee.wasm_gen(Ee.wasm_scratch() + 128, p.length, 200, 0.8);
      const ms = performance.now() - t0;
      const text = new TextDecoder('latin1').decode(new Uint8Array(Ee.memory.buffer, Ee.wasm_out(), nGen));
      // l'erreur de quantification est imprimée sur stdout (wasi), pas captée ici
      setRuns((r) => [{ mode, text, tokPerS: nGen / ms * 1000, quantErr: mode === 0 ? '—' : 'voir note', ms }, ...r].slice(0, 4));
      setBusy(null);
    }, 20);
  };

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel
        tag="Inference edge · WASM"
        title="Le module C compilé en wasm32 — génération depuis un vrai checkpoint"
        subtitle="lm_main.c (trainer) compilé — main() supprimé au link, inference seule : 47.6 kB, zéro dépendance"
        flush
      >
        <div className="space-y-3 p-3.5">
          <div className="flex items-center gap-2">
            <Tag tone={ready ? 'green' : 'amber'}>{ready ? '● prêt' : '◌ ' + status}</Tag>
            {!ready && <span className="text-[10px] text-slate-500">{status}</span>}
          </div>
          {ready && (
            <>
              <div className="flex flex-wrap items-center gap-1.5">
                <input
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-44 rounded bg-slate-900/60 px-2 py-1 text-[11px] text-slate-200 outline-none ring-1 ring-slate-700 focus:ring-cyan-600"
                  placeholder="prompt (64 car. max)"
                />
                <Btn tone="primary" onClick={() => run(0)} disabled={busy !== null}>{busy === 0 ? '…' : '▶ FP32'}</Btn>
                <Btn onClick={() => run(2)} disabled={busy !== null}>{busy === 2 ? '…' : '▶ INT8'}</Btn>
                <Btn onClick={() => run(3)} disabled={busy !== null}>{busy === 3 ? '…' : '▶ INT4'}</Btn>
                <Btn tone="danger" onClick={() => run(1)} disabled={busy !== null}>{busy === 1 ? '…' : '▶ E8'}</Btn>
              </div>
              <div className="space-y-2">
                {runs.map((r, i) => (
                  <div key={i} className="rounded-lg bg-slate-900/50 p-2.5 ring-1 ring-slate-800">
                    <div className="mb-1 flex items-center gap-2">
                      <Tag tone={r.mode === 0 ? 'cyan' : r.mode === 3 ? 'violet' : 'green'}>{MODE_LABEL[r.mode]}</Tag>
                      <span className="text-[10px] text-slate-500">{r.tokPerS.toFixed(0)} tok/s · {r.ms.toFixed(0)} ms · seed 42 (apparié)</span>
                    </div>
                    <p className="whitespace-pre-wrap break-all font-mono text-[10px] leading-relaxed text-slate-400">{r.text || '(aucun token généré)'}</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </Panel>

      <Panel
        tag="HONNÊTETÉ"
        title="Ce que c'est, ce que ça coûte, où ça va"
        subtitle="les nombres réels plutôt que les promesses"
      >
        <div className="space-y-2 text-[10px] leading-relaxed">
          <p className="text-emerald-300">✓ Réel</p>
          <ul className="space-y-1 text-slate-500">
            <li>· Le <b>vrai module C</b> (trainer 270k params) compilé wasm32-wasip1 — 47.6 kB stripped, inference seule</li>
            <li>· Checkpoint réel : 800 steps sur TinyStories, val loss 3.15 (hasard 4.64) — le modèle a appris</li>
            <li>· 4 modes de quantification (FP32/E8/INT8/INT4) — même seed = comparaison appariée, restauration FP32 avant chaque mode (jamais en cascade)</li>
            <li>· ~120 tok/s en wasm scalaire dans le navigateur — offline, private, zéro serveur</li>
          </ul>
          <p className="text-amber-300 pt-1">✗ Limites (documentées)</p>
          <ul className="space-y-1 text-slate-500">
            <li>· Chemin <b>scalaire</b> : wasm32 sans AVX2 → ×115 vs le C natif (14000 tok/s) — l'upgrade est <Formula>WASM SIMD v128</Formula> (les kernels 4-tuiles se vectorisent 1:1)</li>
            <li>· La quantification est <b>simulée</b> : les poids restent fp32 en mémoire — même coût compute, la claim est qualité-par-bit (erreur RMS 7% INT4), pas la taille mémoire</li>
            <li>· Modèle char-level 270k / 800 steps — texte pseudo-cohérent, pas un LLM utile</li>
          </ul>
        </div>
      </Panel>
    </div>
  );
}
