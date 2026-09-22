/* ============================================================================
 * lm_model.h ??? Mini-GPT : registre de param??tres, forward, backward EXACTE.
 * Unit?? de traduction unique incluse par lm_main.c.
 * =========================================================================== */
#ifndef LM_MODEL_H
#define LM_MODEL_H

#include <stdlib.h>
#include <string.h>
#include "lm_kernels.h"

#define TT  32            /* contexte      */
#define DD  72            /* d_model       */
#define NH  3             /* t??tes         */
#define HDv (DD / NH)
#define NLv 4             /* blocs */
#define FFv (4 * DD)

/* ------------------------------------------------------------------ data -- */
static unsigned char *dataBuf;
static long dataLen, trainEnd;
static int V;
static unsigned char byte2id[256];

static int loadData(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    dataLen = ftell(f);
    fseek(f, 0, SEEK_SET);
    dataBuf = (unsigned char *)malloc((size_t)dataLen);
    if (!dataBuf || fread(dataBuf, 1, (size_t)dataLen, f) != (size_t)dataLen) { fclose(f); return -1; }
    fclose(f);
    memset(byte2id, 0xFF, sizeof(byte2id));
    V = 0;
    for (long i = 0; i < dataLen; i++)
        if (byte2id[dataBuf[i]] == 0xFF) byte2id[dataBuf[i]] = (unsigned char)(V++);
    trainEnd = (dataLen * 95) / 100;
    if (trainEnd > dataLen - TT - 2) trainEnd = dataLen - TT - 2;
    return 0;
}

/* ------------------------------------------------------------ param??tres -- */
enum { P_WTE, P_WPE,
       P_LN1G, P_LN1B, P_QKVW, P_QKVB, P_APW, P_APB, P_LN2G, P_LN2B,
       P_F1W, P_F1B, P_F2W, P_F2B, P_LNFG, P_LNFB, P_WOUT, P_NUM };

static float *par[P_NUM], *grad[P_NUM], *mom[P_NUM], *vel[P_NUM];
static long psz[P_NUM];
static long totalParams;
static const char *pname[P_NUM] = {
    "wte", "wpe", "ln1g", "ln1b", "qkvW", "qkvB", "attnW", "attnB",
    "ln2g", "ln2b", "fc1W", "fc1B", "fc2W", "fc2B", "lnfg", "lnfb", "outW"
};
static float *arenaPar;
static float *WtT[P_NUM];   /* poids transpos??s [K][N] pour le GEMM streaming (d??riv??s de par) */

static void transposeOne(float *dst, const float *src, int K, int N) {
    for (int k = 0; k < K; k++) {
        const float *sr = src + (size_t)k;
        float *dr = dst + (size_t)k * N;
        for (int n = 0; n < N; n++) dr[n] = sr[(size_t)n * K];
    }
}
static void refreshWt(void) {
    for (int l = 0; l < NLv; l++) {
        transposeOne(WtT[P_QKVW] + (size_t)l * 3 * DD * DD, par[P_QKVW] + (size_t)l * 3 * DD * DD, DD, 3 * DD);
        transposeOne(WtT[P_APW] + (size_t)l * DD * DD, par[P_APW] + (size_t)l * DD * DD, DD, DD);
        transposeOne(WtT[P_F1W] + (size_t)l * FFv * DD, par[P_F1W] + (size_t)l * FFv * DD, DD, FFv);
        transposeOne(WtT[P_F2W] + (size_t)l * DD * FFv, par[P_F2W] + (size_t)l * DD * FFv, FFv, DD);
    }
    transposeOne(WtT[P_WOUT], par[P_WOUT], DD, V);
}

static float *allocz(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(p, 0, n * sizeof(float));
    return p;
}

static void modelInit(void) {
    psz[P_WTE] = (long)V * DD;     psz[P_WPE] = (long)TT * DD;
    psz[P_LN1G] = NLv * DD;        psz[P_LN1B] = NLv * DD;
    psz[P_QKVW] = NLv * 3 * DD * DD; psz[P_QKVB] = NLv * 3 * DD;
    psz[P_APW] = NLv * DD * DD;    psz[P_APB] = NLv * DD;
    psz[P_LN2G] = NLv * DD;        psz[P_LN2B] = NLv * DD;
    psz[P_F1W] = NLv * FFv * DD;   psz[P_F1B] = NLv * FFv;
    psz[P_F2W] = NLv * DD * FFv;   psz[P_F2B] = NLv * DD;
    psz[P_LNFG] = DD;              psz[P_LNFB] = DD;
    psz[P_WOUT] = (long)V * DD;

    long tot = 0;
    for (int p = 0; p < P_NUM; p++) tot += psz[p];
    totalParams = tot;
    arenaPar = allocz((size_t)tot);
    /* transpos??s pour le GEMM streaming (5 tenseurs matriciels) */
    WtT[P_QKVW] = allocz((size_t)NLv * 3 * DD * DD);
    WtT[P_APW] = allocz((size_t)NLv * DD * DD);
    WtT[P_F1W] = allocz((size_t)NLv * FFv * DD);
    WtT[P_F2W] = allocz((size_t)NLv * DD * FFv);
    WtT[P_WOUT] = allocz((size_t)V * DD);
    long off = 0;
    for (int p = 0; p < P_NUM; p++) {
        par[p] = arenaPar + off; off += psz[p];
        grad[p] = allocz((size_t)psz[p]);
        mom[p] = allocz((size_t)psz[p]);
        vel[p] = allocz((size_t)psz[p]);
    }
    const int mats[6] = { P_WTE, P_QKVW, P_APW, P_F1W, P_F2W, P_WOUT };
    for (int mi = 0; mi < 6; mi++)
        for (long i = 0; i < psz[mats[mi]]; i++) par[mats[mi]][i] = 0.02f * rng_n();
    for (long i = 0; i < psz[P_APW]; i++) par[P_APW][i] *= 0.70710678f;
    refreshWt();
}

/* --------------------------------------------------------------- buffers -- */
typedef struct {
    float *xsav[NLv + 1], *msav[NLv], *f2b[NLv], *xob[NLv];
    float *ln1[NLv], *qkv[NLv], *att[NLv], *xat[NLv], *pro[NLv], *ln2[NLv], *z1[NLv];
    float *lnf, *logits, *probs;
} Acts;

typedef struct {
    float *dxs[NLv + 1];
    float *dm, *dlno, *dqkv, *dxat, *dz1, *f1b;
} Scratch;

static Scratch SC;

static void actsAlloc(Acts *A, int B) {
    const size_t btd = (size_t)B * TT * DD, bt3d = (size_t)B * TT * 3 * DD,
                 batt = (size_t)B * NH * TT * TT, btff = (size_t)B * TT * FFv,
                 btv = (size_t)B * TT * (size_t)V;
    for (int l = 0; l < NLv; l++) {
        A->xsav[l] = allocz(btd);  A->msav[l] = allocz(btd);
        A->f2b[l] = allocz(btff);  A->xob[l] = allocz(btd);
        A->ln1[l] = allocz(btd);   A->qkv[l] = allocz(bt3d);
        A->att[l] = allocz(batt);  A->xat[l] = allocz(btd); A->pro[l] = allocz(btd);
        A->ln2[l] = allocz(btd);   A->z1[l] = allocz(btff);
    }
    A->xsav[NLv] = allocz(btd);
    A->lnf = allocz(btd);
    A->logits = allocz(btv);
    A->probs = allocz(btv);
}

static void scratchAlloc(int B) {
    const size_t btd = (size_t)B * TT * DD, bt3d = (size_t)B * TT * 3 * DD, btff = (size_t)B * TT * FFv;
    for (int l = 0; l <= NLv; l++) SC.dxs[l] = allocz(btd);
    SC.dm = allocz(btd); SC.dlno = allocz(btd); SC.dqkv = allocz(bt3d);
    SC.dxat = allocz(btd); SC.dz1 = allocz(btff); SC.f1b = allocz(btff);
}

#ifdef _OPENMP
#include <omp.h>
#endif
static int cpuAvx2 = -1;
static int g_gemmOld = 0; /* --gemm-old : force la chaîne unique (A/B e2e) */
double g_tGemm = 0, g_tAtt = 0, g_tLn = 0;
static int g_prof = 0;
static inline double profNow(void) { struct timespec ts; timespec_get(&ts, TIME_UTC); return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
#include <immintrin.h>
/* GEMM streaming : poids pr??-transpos??s [K][N] ??? z??ro repack, acc??s contigus.
   Toutes les largeurs N utilis??es (72/216/288/104) sont multiples de 8. */
static void mm_nt(float *Y, const float *X, const float *W, const float *b,
                  int M, int K, int N); /* forward decl ??? chemin scalaire */
__attribute__((target("avx2,fma")))
static void mm_nt_t_avx2(float *Y, const float *X, const float *Wtr, const float *b,
                         int M, int K, int N) {
    /* micro-kernel PR#1 : 4 tuiles de 8 cols partagent le broadcast xm[k]
       (charges contigues k*N+nb..+31) → 4 chaines FMA indépendantes.
       Latence FMA ~4 cy masquée (pic 2 FMA/cy) vs 1 chaîne = 0.25 FMA/cy. */
    const int N8 = N & ~7;
    _Pragma("omp parallel for schedule(static)")
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        int nb = 0;
        if (g_gemmOld) {
            for (; nb + 8 <= N8; nb += 8) {
                __m256 acc = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
                const float *wk = Wtr + nb;
                for (int k = 0; k < K; k++)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(wk + (size_t)k * N), acc);
                _mm256_storeu_ps(ym + nb, acc);
            }
        } else {
        for (; nb + 32 <= N8; nb += 32) {
            __m256 a0 = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
            __m256 a1 = b ? _mm256_loadu_ps(b + nb + 8) : _mm256_setzero_ps();
            __m256 a2 = b ? _mm256_loadu_ps(b + nb + 16) : _mm256_setzero_ps();
            __m256 a3 = b ? _mm256_loadu_ps(b + nb + 24) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m256 xk = _mm256_set1_ps(xm[k]);
                const float *w = Wtr + (size_t)k * N + nb;
                a0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w), a0);
                a1 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 8), a1);
                a2 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 24), a3);
            }
            _mm256_storeu_ps(ym + nb, a0);
            _mm256_storeu_ps(ym + nb + 8, a1);
            _mm256_storeu_ps(ym + nb + 16, a2);
            _mm256_storeu_ps(ym + nb + 24, a3);
        }
        }
        for (; nb + 8 <= N8; nb += 8) {
            __m256 acc = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
            const float *wk = Wtr + nb;
            for (int k = 0; k < K; k++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(wk + (size_t)k * N), acc);
            _mm256_storeu_ps(ym + nb, acc);
        }
        for (int n = N8; n < N; n++) {
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * Wtr[(size_t)k * N + n];
            ym[n] = acc;
        }
    }
}
static void mm_nt_t(float *Y, const float *X, const float *Wrow, const float *Wtr,
                    const float *b, int M, int K, int N) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpuAvx2 < 0) cpuAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (cpuAvx2) { const double tp0 = g_prof ? profNow() : 0; mm_nt_t_avx2(Y, X, Wtr, b, M, K, N); if (g_prof) g_tGemm += profNow() - tp0; return; }
#endif
    {
        const double tp0 = g_prof ? profNow() : 0;
        mm_nt(Y, X, Wrow, b, M, K, N);
        if (g_prof) g_tGemm += profNow() - tp0;
    }
}

/* ------------------------------------------------------------------ GEMM --
 * Deux chemins : AVX2+FMA (dispatch runtime, toutes les dims sont ??8 sauf V)
 * et scalaire de r??f??rence. M??mes r??sultats ?? l'arrondi pr??s.
 */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>


/* profilage grossier ? activ? par --profile */
__attribute__((target("avx2,fma")))
static void mm_nt_avx2(float *Y, const float *X, const float *W, const float *b,
                       int M, int K, int N) {
    /* W est [N,K] row-major : pour vectoriser sur n il faut transposer le
       panneau 8??K en [K][8] (chargements contigus ensuite). C'??tait LE bug
       qui divergeait : la 1re version lisait 8 colonnes d'une m??me ligne. */
    static float panel[FFv * 8]; /* K ??? FFv */
    const int N8 = N & ~7;
    for (int m = 0; m < M; m++) {
        float *ym = Y + (size_t)m * N;
        if (b) memcpy(ym, b, sizeof(float) * (size_t)N);
        else memset(ym, 0, sizeof(float) * (size_t)N);
    }
    for (int nb = 0; nb < N8; nb += 8) {
        for (int k = 0; k < K; k++) {
            const float *wcol = W + (size_t)nb * K + k;
            panel[k * 8 + 0] = wcol[0];
            panel[k * 8 + 1] = wcol[K];
            panel[k * 8 + 2] = wcol[2 * K];
            panel[k * 8 + 3] = wcol[3 * K];
            panel[k * 8 + 4] = wcol[4 * K];
            panel[k * 8 + 5] = wcol[5 * K];
            panel[k * 8 + 6] = wcol[6 * K];
            panel[k * 8 + 7] = wcol[7 * K];
        }
        for (int m = 0; m < M; m++) {
            const float *xm = X + (size_t)m * K;
            float *ym = Y + (size_t)m * N + nb;
            __m256 acc = _mm256_loadu_ps(ym);
            for (int k = 0; k < K; k++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(panel + k * 8), acc);
            _mm256_storeu_ps(ym, acc);
        }
    }
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        for (int n = N8; n < N; n++) {
            float acc = ym[n];
            const float *wn = W + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += xm[k] * wn[k];
            ym[n] = acc;
        }
    }
}
__attribute__((target("avx2,fma")))
static void mm_dxd_avx2(float *dX, const float *dY, const float *W, int M, int K, int N) {
    /* multi-acc PR#1 : 4 tuiles de 8 sur K partagent broadcast dm[n]
       (W[n][k..k+31] contigu) — backward dX, même pattern que mm_nt_t. */
    const int K8 = K & ~7;
    _Pragma("omp parallel for schedule(static)")
    for (int m = 0; m < M; m++) {
        const float *dm = dY + (size_t)m * N;
        float *xm = dX + (size_t)m * K;
        int kb = 0;
        for (; kb + 32 <= K8; kb += 32) {
            __m256 a0 = _mm256_loadu_ps(xm + kb);
            __m256 a1 = _mm256_loadu_ps(xm + kb + 8);
            __m256 a2 = _mm256_loadu_ps(xm + kb + 16);
            __m256 a3 = _mm256_loadu_ps(xm + kb + 24);
            for (int n = 0; n < N; n++) {
                const __m256 xk = _mm256_set1_ps(dm[n]);
                const float *w = W + (size_t)n * K + kb;
                a0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w), a0);
                a1 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 8), a1);
                a2 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(w + 24), a3);
            }
            _mm256_storeu_ps(xm + kb, a0);
            _mm256_storeu_ps(xm + kb + 8, a1);
            _mm256_storeu_ps(xm + kb + 16, a2);
            _mm256_storeu_ps(xm + kb + 24, a3);
        }
        for (; kb + 8 <= K8; kb += 8) {
            __m256 acc = _mm256_loadu_ps(xm + kb);
            for (int n = 0; n < N; n++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(dm[n]), _mm256_loadu_ps(W + (size_t)n * K + kb), acc);
            _mm256_storeu_ps(xm + kb, acc);
        }
        for (int k = K8; k < K; k++) {
            float acc = xm[k];
            for (int n = 0; n < N; n++) acc += dm[n] * W[(size_t)n * K + k];
            xm[k] = acc;
        }
    }
}
__attribute__((target("avx2,fma")))
static void mm_tndw_avx2(float *dW, float *db, const float *dY, const float *X,
                         int M, int K, int N) {
    /* multi-acc : pour chaque n, tuiles K=32 avec accs en registres sur tout M
       (évite load/store dw à chaque m — chaîne mémoire avant). */
    const int K8 = K & ~7;
    _Pragma("omp parallel for schedule(static)")
    for (int n = 0; n < N; n++) {
        float accb = 0.0f;
        float *dw = dW + (size_t)n * K;
        for (int m = 0; m < M; m++) accb += dY[(size_t)m * N + n];
        int kb = 0;
        for (; kb + 32 <= K8; kb += 32) {
            __m256 a0 = _mm256_loadu_ps(dw + kb);
            __m256 a1 = _mm256_loadu_ps(dw + kb + 8);
            __m256 a2 = _mm256_loadu_ps(dw + kb + 16);
            __m256 a3 = _mm256_loadu_ps(dw + kb + 24);
            for (int m = 0; m < M; m++) {
                const __m256 xk = _mm256_set1_ps(dY[(size_t)m * N + n]);
                const float *xm = X + (size_t)m * K + kb;
                a0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(xm), a0);
                a1 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(xm + 8), a1);
                a2 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(xm + 16), a2);
                a3 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(xm + 24), a3);
            }
            _mm256_storeu_ps(dw + kb, a0);
            _mm256_storeu_ps(dw + kb + 8, a1);
            _mm256_storeu_ps(dw + kb + 16, a2);
            _mm256_storeu_ps(dw + kb + 24, a3);
        }
        for (; kb + 8 <= K8; kb += 8) {
            __m256 acc = _mm256_loadu_ps(dw + kb);
            for (int m = 0; m < M; m++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(dY[(size_t)m * N + n]),
                                      _mm256_loadu_ps(X + (size_t)m * K + kb), acc);
            _mm256_storeu_ps(dw + kb, acc);
        }
        for (int k = K8; k < K; k++) {
            float acc = dw[k];
            for (int m = 0; m < M; m++) acc += dY[(size_t)m * N + n] * X[(size_t)m * K + k];
            dw[k] = acc;
        }
        if (db) db[n] += accb;
    }
}
#endif /* x86_64 */

static void mm_nt(float *Y, const float *X, const float *W, const float *b,
                  int M, int K, int N) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpuAvx2 < 0) cpuAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (cpuAvx2) { const double tp0 = g_prof ? profNow() : 0; mm_nt_avx2(Y, X, W, b, M, K, N); if (g_prof) g_tGemm += profNow() - tp0; return; }
#endif
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        if (b) memcpy(ym, b, sizeof(float) * (size_t)N);
        else memset(ym, 0, sizeof(float) * (size_t)N);
        for (int k = 0; k < K; k++) {
            const float xv = xm[k];
            const float *wk = W + (size_t)k;
            for (int n = 0; n < N; n++) ym[n] += xv * wk[(size_t)n * K];
        }
    }
}
static void mm_dxd(float *dX, const float *dY, const float *W, int M, int K, int N) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpuAvx2 < 0) cpuAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (cpuAvx2) { const double tp0 = g_prof ? profNow() : 0; mm_dxd_avx2(dX, dY, W, M, K, N); if (g_prof) g_tGemm += profNow() - tp0; return; }
#endif
    for (int m = 0; m < M; m++) {
        const float *dm = dY + (size_t)m * N;
        float *xm = dX + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const float dv = dm[n];
            const float *wn = W + (size_t)n * K;
            for (int k = 0; k < K; k++) xm[k] += dv * wn[k];
        }
    }
}
static void mm_tndw(float *dW, float *db, const float *dY, const float *X,
                    int M, int K, int N) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpuAvx2 < 0) cpuAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (cpuAvx2) { const double tp0 = g_prof ? profNow() : 0; mm_tndw_avx2(dW, db, dY, X, M, K, N); if (g_prof) g_tGemm += profNow() - tp0; return; }
#endif
    _Pragma("omp parallel for schedule(static)")
    for (int n = 0; n < N; n++) {
        float accb = 0.0f;
        float *dw = dW + (size_t)n * K;
        for (int m = 0; m < M; m++) {
            const float dv = dY[(size_t)m * N + n];
            accb += dv;
            const float *xm = X + (size_t)m * K;
            for (int k = 0; k < K; k++) dw[k] += dv * xm[k];
        }
        if (db) db[n] += accb;
    }
}

/* ------------------------------------------------------------- layernorm -- */
static void ln_fwd(float *y, const float *x, const float *g, const float *b, int n) {
    const double tp0 = g_prof ? profNow() : 0;
    float mu = 0.0f;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { const float d = x[i] - mu; var += d * d; }
    var /= (float)n;
    const float inv = krsqrt(var + 1e-5f);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * g[i] + b[i];
    if (g_prof) g_tLn += profNow() - tp0;
}
static void ln_bwd(float *dx, float *dg, float *db, const float *dy,
                   const float *x, const float *g, int n) {
    const double tp0 = g_prof ? profNow() : 0;
    float mu = 0.0f;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { const float d = x[i] - mu; var += d * d; }
    var /= (float)n;
    const float inv = krsqrt(var + 1e-5f);
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const float xh = (x[i] - mu) * inv;
        db[i] += dy[i];
        dg[i] += dy[i] * xh;
        s1 += dy[i] * g[i];
        s2 += dy[i] * g[i] * xh;
    }
    const float an = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        const float xh = (x[i] - mu) * inv;
        dx[i] += inv * (dy[i] * g[i] - an * s1 - xh * an * s2);
    }
    if (g_prof) g_tLn += profNow() - tp0;
}

/* --------------------------------------------------------------- forward -- */
static float forward(Acts *A, const int *ids, const int *tgt, int B) {
    const size_t btd = (size_t)B * TT * DD;

    for (int b = 0; b < B; b++)
        for (int t = 0; t < TT; t++) {
            float *e = A->xsav[0] + ((size_t)b * TT + t) * DD;
            const float *w = par[P_WTE] + (size_t)ids[(size_t)b * TT + t] * DD;
            const float *pp = par[P_WPE] + (size_t)t * DD;
            for (int d = 0; d < DD; d++) e[d] = w[d] + pp[d];
        }

    const float scale = krsqrt((float)HDv);
    for (int l = 0; l < NLv; l++) {
        const float *xin = A->xsav[l];
        float *ln1 = A->ln1[l];
        for (size_t i = 0; i < btd; i += DD)
            ln_fwd(ln1 + i, xin + i, par[P_LN1G] + l * DD, par[P_LN1B] + l * DD, DD);

        float *qkv = A->qkv[l];
        mm_nt_t(qkv, ln1, par[P_QKVW] + (size_t)l * 3 * DD * DD, WtT[P_QKVW] + (size_t)l * 3 * DD * DD,
              par[P_QKVB] + l * 3 * DD, (int)(B * TT), DD, 3 * DD);

        const double tp0 = g_prof ? profNow() : 0;
        float *att = A->att[l], *xat = A->xat[l];
        memset(xat, 0, btd * sizeof(float));
        for (int b = 0; b < B; b++)
            for (int h = 0; h < NH; h++) {
                const int qo = h * HDv, oo = h * HDv, vo = 2 * DD + h * HDv;
                float *ap = att + (((size_t)b * NH + h) * TT) * TT;
                for (int t = 0; t < TT; t++) {
                    const float *q = qkv + ((size_t)b * TT + t) * 3 * DD + qo;
                    for (int s = 0; s <= t; s++) {
                        const float *kk = qkv + ((size_t)b * TT + s) * 3 * DD + h * HDv;
                        float acc = 0.0f;
                        for (int d = 0; d < HDv; d++) acc += q[d] * kk[d];
                        ap[(size_t)t * TT + s] = acc * scale;
                    }
                    softmaxRow(ap + (size_t)t * TT, t + 1);   /* AVX2 exp8 ou expf */
                    float *o = xat + ((size_t)b * TT + t) * DD + oo;
                    for (int s = 0; s <= t; s++) {
                        const float pv = ap[(size_t)t * TT + s];
                        const float *vv = qkv + ((size_t)b * TT + s) * 3 * DD + vo;
                        for (int d = 0; d < HDv; d++) o[d] += pv * vv[d];
                    }
                }
            }
        if (g_prof) g_tAtt += profNow() - tp0;

        float *pro = A->pro[l];
        mm_nt_t(pro, xat, par[P_APW] + (size_t)l * DD * DD, WtT[P_APW] + (size_t)l * DD * DD, par[P_APB] + l * DD,
              (int)(B * TT), DD, DD);
        float *msav = A->msav[l];
        for (size_t i = 0; i < btd; i++) msav[i] = xin[i] + pro[i];

        float *ln2 = A->ln2[l];
        for (size_t i = 0; i < btd; i += DD)
            ln_fwd(ln2 + i, msav + i, par[P_LN2G] + l * DD, par[P_LN2B] + l * DD, DD);

        float *z1 = A->z1[l], *f2 = A->f2b[l], *xout = A->xob[l];
        mm_nt_t(z1, ln2, par[P_F1W] + (size_t)l * FFv * DD, WtT[P_F1W] + (size_t)l * FFv * DD, par[P_F1B] + l * FFv,
              (int)(B * TT), DD, FFv);
        for (size_t i = 0; i < btd; i++) f2[i] = spear_gelu_vd(z1[i]).v;
        mm_nt_t(xout, f2, par[P_F2W] + (size_t)l * DD * FFv, WtT[P_F2W] + (size_t)l * DD * FFv, par[P_F2B] + l * DD,
              (int)(B * TT), FFv, DD);
        for (size_t i = 0; i < btd; i++) A->xsav[l + 1][i] = msav[i] + xout[i];
    }

    const float *xf = A->xsav[NLv];
    for (size_t i = 0; i < btd; i += DD)
        ln_fwd(A->lnf + i, xf + i, par[P_LNFG], par[P_LNFB], DD);
    mm_nt_t(A->logits, A->lnf, par[P_WOUT], WtT[P_WOUT], NULL, (int)(B * TT), DD, V);

    float loss = 0.0f;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < TT; t++) {
            const size_t ro = ((size_t)b * TT + t) * (size_t)V;
            float *pr = A->probs + ro;
            memcpy(pr, A->logits + ro, sizeof(float) * (size_t)V);
            softmaxRow(pr, V);
            const float pTgt = pr[tgt[(size_t)b * TT + t]];
            loss -= logf(pTgt > 1e-9f ? pTgt : 1e-9f);
        }
    return loss / (float)(B * TT);
}

/* -------------------------------------------------------------- backward -- */
static void backward(Acts *A, const int *ids, const int *tgt, int B) {
    const int M = B * TT;
    const size_t btd = (size_t)B * TT * DD;
    Scratch *S = &SC;
    (void)ids;

    /* --- t??te : CE --- */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < TT; t++) {
            const size_t ro = ((size_t)b * TT + t) * (size_t)V;
            float *row = A->logits + ro;
            for (int v = 0; v < V; v++) row[v] = A->probs[ro + (size_t)v];
            row[tgt[(size_t)b * TT + t]] -= 1.0f;
            for (int v = 0; v < V; v++) row[v] /= (float)M;
        }
    memset(S->dxs[NLv], 0, btd * sizeof(float));
    mm_dxd(S->dxs[NLv], A->logits, par[P_WOUT], M, DD, V);
    mm_tndw(grad[P_WOUT], NULL, A->logits, A->lnf, M, DD, V);
    {
        float *tmp = allocz(btd);
        memcpy(tmp, S->dxs[NLv], sizeof(float) * btd);
        memset(S->dxs[NLv], 0, btd * sizeof(float));
        ln_bwd(S->dxs[NLv], grad[P_LNFG], grad[P_LNFB], tmp,
               A->xsav[NLv], par[P_LNFG], DD);
        free(tmp);
    }

    /* --- blocs, en ordre inverse --- */
    for (int l = NLv - 1; l >= 0; l--) {
        const float *dOut = S->dxs[l + 1];

        /* MLP : xout = m + fc2(gelu(fc1(ln2(m)))) */
        memset(S->dz1, 0, sizeof(float) * btd);
        mm_dxd(S->dz1, dOut, par[P_F2W] + (size_t)l * DD * FFv, M, FFv, DD);
        for (size_t i = 0; i < btd; i++) {
            const sp_gelu_vd gd = spear_gelu_vd(A->z1[l][i]);
            S->f1b[i] = gd.v;
            S->dz1[i] *= gd.d;
        }
        mm_tndw(grad[P_F2W] + (size_t)l * DD * FFv, grad[P_F2B] + l * DD,
                dOut, S->f1b, M, FFv, DD);
        memset(S->dm, 0, btd * sizeof(float));
        mm_dxd(S->dm, S->dz1, par[P_F1W] + (size_t)l * FFv * DD, M, DD, FFv);
        mm_tndw(grad[P_F1W] + (size_t)l * FFv * DD, grad[P_F1B] + l * FFv,
                S->dz1, A->ln2[l], M, DD, FFv);

        /* r??sidu : dm_total = dOut + dm_mlp */
        for (size_t i = 0; i < btd; i++) S->dm[i] += dOut[i];

        /* LN2 : entr??e msav */
        memset(S->dlno, 0, btd * sizeof(float));
        ln_bwd(S->dlno, grad[P_LN2G] + l * DD, grad[P_LN2B] + l * DD,
               S->dm, A->msav[l], par[P_LN2G] + l * DD, DD);

        /* attention : pro = attnW??xat, xat = ?? p??v */
        memset(S->dxat, 0, btd * sizeof(float));
        mm_dxd(S->dxat, S->dm, par[P_APW] + (size_t)l * DD * DD, M, DD, DD);
        mm_tndw(grad[P_APW] + (size_t)l * DD * DD, grad[P_APB] + l * DD,
                S->dm, A->xat[l], M, DD, DD);

        /* coeur causal backward ??? dqkv */
        memset(S->dqkv, 0, sizeof(float) * (size_t)B * TT * 3 * DD);
        const float scale = krsqrt((float)HDv);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < NH; h++) {
                const int qo = h * HDv, ko = h * HDv, oo = h * HDv, vo = 2 * DD + h * HDv;
                const float *ap = A->att[l] + (((size_t)b * NH + h) * TT) * TT;
                float dp[TT];
                for (int t = 0; t < TT; t++) {
                    const float *qt = A->qkv[l] + ((size_t)b * TT + t) * 3 * DD + qo;
                    const float *dxo = S->dxat + ((size_t)b * TT + t) * DD + oo;
                    float dot = 0.0f;
                    for (int s = 0; s <= t; s++) {
                        const float *vs = A->qkv[l] + ((size_t)b * TT + s) * 3 * DD + vo;
                        float acc = 0.0f;
                        for (int d = 0; d < HDv; d++) acc += vs[d] * dxo[d];
                        dp[s] = acc;
                        dot += ap[(size_t)t * TT + s] * acc;
                    }
                    float *dq = S->dqkv + ((size_t)b * TT + t) * 3 * DD + qo;
                    for (int s = 0; s <= t; s++) {
                        const float p = ap[(size_t)t * TT + s];
                        const float ds = p * (dp[s] - dot) * scale;
                        const float *vs = A->qkv[l] + ((size_t)b * TT + s) * 3 * DD + vo;
                        const float *ks = A->qkv[l] + ((size_t)b * TT + s) * 3 * DD + ko;
                        float *dvs = S->dqkv + ((size_t)b * TT + s) * 3 * DD + vo;
                        float *dks = S->dqkv + ((size_t)b * TT + s) * 3 * DD + ko;
                        for (int d = 0; d < HDv; d++) {
                            dvs[d] += p * dxo[d];
                            dks[d] += ds * qt[d];
                            dq[d] += ds * ks[d];
                        }
                    }
                }
            }
        mm_tndw(grad[P_QKVW] + (size_t)l * 3 * DD * DD, grad[P_QKVB] + l * 3 * DD,
                S->dqkv, A->ln1[l], M, DD, 3 * DD);
        memset(S->dlno, 0, btd * sizeof(float));
        mm_dxd(S->dlno, S->dqkv, par[P_QKVW] + (size_t)l * 3 * DD * DD, M, DD, 3 * DD);

        /* r??sidus : dxs[l] = dm_total + LN1^T(dlno) */
        float *dxprev = S->dxs[l];
        memcpy(dxprev, S->dm, sizeof(float) * btd);
        {
            float *tmp = allocz(btd);
            memcpy(tmp, S->dlno, sizeof(float) * btd);
            memset(S->dlno, 0, btd * sizeof(float));
            ln_bwd(S->dlno, grad[P_LN1G] + l * DD, grad[P_LN1B] + l * DD,
                   tmp, A->xsav[l], par[P_LN1G] + l * DD, DD);
            for (size_t i = 0; i < btd; i++) dxprev[i] += S->dlno[i];
            free(tmp);
        }
    }

    /* embeddings */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < TT; t++) {
            const float *g0 = S->dxs[0] + ((size_t)b * TT + t) * DD;
            float *dw = grad[P_WTE] + (size_t)ids[(size_t)b * TT + t] * DD;
            float *dp_ = grad[P_WPE] + (size_t)t * DD;
            for (int d = 0; d < DD; d++) { dw[d] += g0[d]; dp_[d] += g0[d]; }
        }
}
#endif /* LM_MODEL_H */






