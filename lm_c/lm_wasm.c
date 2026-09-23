/* lm_wasm.c — inference edge : WASM (wasm32-wasi), generation INT4/FP32
   depuis un checkpoint reel (lm_c/checkpoint.bin).
   Le trainer complet est inclus (statiques accessibles dans le meme TU) ;
   main() n'est pas exporte → supprime au link (--gc-sections).
   Exports : wasm_init, wasm_arena, wasm_refresh, wasm_vocab, wasm_quant,
             wasm_seed, wasm_gen, wasm_out. */
#define main lm_unused_main
#include "lm_main.c"
#undef main

#include <stdint.h>

static char genBuf[4096];
static char scratch[1024]; /* vocab (104) + prompt (TT) — écrit par JS */

__attribute__((used))
char *wasm_scratch(void) { return scratch; }

__attribute__((used))
int wasm_init(void) {
    modelInit();
    actsAlloc(&SA, 1);
    return (int)totalParams;
}

/* offset de l'arena des poids dans la memoire lineaire wasm */
__attribute__((used))
int wasm_arena(void) { return (int)(intptr_t)arenaPar; }

/* JS a copie les floats du checkpoint dans la memoire wasm a arenaPar */
__attribute__((used))
void wasm_refresh(void) { refreshWt(); }

/* JS a ecrit le vocab (octets en ordre de premiere occurrence) a off.
   Reconstruit byte2id exactement comme lm_main.c (ligne 34-37) ET pose V —
   V doit être défini AVANT modelInit (psz[P_WTE] = V*DD). */
__attribute__((used))
void wasm_vocab(int off, int len) {
    memset(byte2id, 0xFF, sizeof(byte2id));
    V = 0;
    const unsigned char *vb = (const unsigned char *)(intptr_t)off;
    for (int i = 0; i < len && i < 256; i++) byte2id[vb[i]] = (unsigned char)(V++);
}

/* quantification des tenseurs matriciels (mode 1=INT8, 2=INT4) puis refresh */
__attribute__((used))
void wasm_quant(int mode) {
    for (int p = 0; p < P_NUM; p++) if (psz[p] > NLv * DD) quantTensor(pname[p], par[p], psz[p], mode);
    refreshWt();
}

/* reproductibilite */
__attribute__((used))
void wasm_seed(unsigned long long s) { rng_seed((uint64_t)s); }

/* generation autoregressive : prompt a off (promptLen octets, modulo si
   plus court que TT), nChars caracteres, temp temperature. Retourne le
   nombre d'octets generes (dans genBuf, via wasm_out()). */
__attribute__((used))
int wasm_gen(int off, int promptLen, int nChars, float temp) {
    int ctx[TT];
    const unsigned char *pbuf = (const unsigned char *)(intptr_t)off;
    for (int t = 0; t < TT; t++)
        ctx[t] = (promptLen > 0) ? byte2id[pbuf[t % promptLen]] : (int)(rng_f() * (float)V) % V;
    int ids[TT], tgtDummy[TT];
    int n = 0;
    for (int g = 0; g < nChars && n < (int)sizeof(genBuf) - 1; g++) {
        for (int t = 0; t < TT; t++) { ids[t] = ctx[t]; tgtDummy[t] = ids[t]; }
        forward(&SA, ids, tgtDummy, 1);
        const float *row = SA.logits + (size_t)(TT - 1) * (size_t)V;
        float mx = -1e30f;
        for (int v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
        float pr[256];
        double tot = 0.0;
        for (int v = 0; v < V; v++) { pr[v] = kexp((row[v] - mx) / temp); tot += pr[v]; }
        const double r = (double)rng_f() * tot;
        int pick = V - 1;
        { double cum = 0.0; for (int v = 0; v < V; v++) { cum += pr[v]; if (cum >= r) { pick = v; break; } } }
        for (int t = 0; t < TT - 1; t++) ctx[t] = ctx[t + 1];
        ctx[TT - 1] = pick;
        for (int b = 0; b < 256; b++) if (byte2id[b] == pick) { genBuf[n++] = (char)b; break; }
    }
    genBuf[n] = 0;
    return n;
}

__attribute__((used))
char *wasm_out(void) { return genBuf; }
