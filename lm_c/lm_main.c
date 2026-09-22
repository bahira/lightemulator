/* ============================================================================
 * lm_main.c ??? Entra??nement, gradcheck, bench A/B, ??chantillonnage.
 * =========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
/* ------------------------------------------------------------------ rng -- */
static uint64_t rng_s = 1;
static void rng_seed(uint64_t s) { rng_s = s ? s : 1; }
static uint64_t rng_u64(void) {
    uint64_t z = (rng_s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static float rng_f(void) { return (float)(rng_u64() >> 40) / 16777216.0f; }
static float rng_n(void) {
    const float u1 = rng_f() + 1e-7f, u2 = rng_f();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
#include "lm_model.h"

static Acts TR, SA;
int g_spear = 1;

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ adam -- */
static int adamT = 0;
static void zeroGrads(void) { for (int p = 0; p < P_NUM; p++) memset(grad[p], 0, sizeof(float) * (size_t)psz[p]); }

/* checkpoint minimal : taille du vocab + arena des poids */
static void saveWeights(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "checkpoint: ecriture impossible\n"); return; }
    fwrite(&V, sizeof(int), 1, f);
    fwrite(arenaPar, sizeof(float), (size_t)totalParams, f);
    fclose(f);
    printf("  checkpoint -> %s\n", path);
}
static int loadWeights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int v = 0;
    if (fread(&v, sizeof(int), 1, f) != 1 || v != V) { fclose(f); return -1; }
    const size_t n = fread(arenaPar, sizeof(float), (size_t)totalParams, f);
    fclose(f);
    return n == (size_t)totalParams ? 0 : -1;
}
static float clipGrads(void) {
    double s2 = 0.0;
    for (int p = 0; p < P_NUM; p++)
        for (long i = 0; i < psz[p]; i++) s2 += (double)grad[p][i] * grad[p][i];
    const float nrm = (float)sqrt(s2);
    if (nrm > 1.0f) { const float c = 1.0f / nrm; for (int p = 0; p < P_NUM; p++) for (long i = 0; i < psz[p]; i++) grad[p][i] *= c; }
    return nrm;
}
static void adamStep(float lr) {
    const float b1 = 0.9f, b2 = 0.99f, eps = 1e-8f, wd = 0.01f;
    adamT++;
    const float c1 = 1.0f - powf(b1, (float)adamT);
    const float c2 = 1.0f - powf(b2, (float)adamT);
    for (int p = 0; p < P_NUM; p++) {
        const int isMat = (p == P_WTE || p == P_QKVW || p == P_APW || p == P_F1W || p == P_F2W || p == P_WOUT);
        for (long i = 0; i < psz[p]; i++) {
            const float g = grad[p][i];
            mom[p][i] = b1 * mom[p][i] + (1 - b1) * g;
            vel[p][i] = b2 * vel[p][i] + (1 - b2) * g * g;
            const float mh = mom[p][i] / c1, vh = vel[p][i] / c2;
            par[p][i] -= lr * (mh / (sqrtf(vh) + eps) + (isMat ? wd * par[p][i] : 0.0f));
        }
    }
}

/* ---------------------------------------------------------------- batch -- */
static void fillBatch(int *ids, int *tgt, int B, long maxPos) {
    for (int b = 0; b < B; b++) {
        const long pos = (long)(rng_f() * (float)(maxPos - 1));
        for (int t = 0; t < TT; t++) {
            ids[(size_t)b * TT + t] = byte2id[dataBuf[pos + t]];
            tgt[(size_t)b * TT + t] = byte2id[dataBuf[pos + t + 1]];
        }
    }
}

static float valLoss(Acts *A, int B, int nb) {
    const long maxPos = dataLen - TT - 1;
    int ids[64 * TT], tgt[64 * TT];
    double tot = 0.0;
    for (int i = 0; i < nb; i++) {
        const long pos = trainEnd + (long)(rng_f() * (float)(maxPos - trainEnd));
        for (int b = 0; b < B; b++)
            for (int t = 0; t < TT; t++) {
                ids[(size_t)b * TT + t] = byte2id[dataBuf[pos + (long)b * 0 + t]];
                tgt[(size_t)b * TT + t] = byte2id[dataBuf[pos + t + 1]];
            }
        tot += forward(A, ids, tgt, B);
    }
    return (float)(tot / nb);
}

/* -------------------------------------------------------------- gradcheck -- */
static void gradcheck(int B) {
    const int savedCpu = cpuAvx2; cpuAvx2 = 0; /* FD sur le chemin scalaire, coh?rent */
    printf("== GRADCHECK : backward analytique vs diff??rences finies ==\n");
    int *ids = (int *)malloc(sizeof(int) * (size_t)B * TT);
    int *tgt = (int *)malloc(sizeof(int) * (size_t)B * TT);
    fillBatch(ids, tgt, B, trainEnd);
    zeroGrads();
    forward(&TR, ids, tgt, B);
    backward(&TR, ids, tgt, B);

    double worst = 0.0, sum = 0.0;
    int checked = 0;
    const float h = 1e-2f; /* compromis bruit/troncature en float32 */
    for (int c = 0; c < 120 && checked < 40; c++) {
        const int p = (int)(rng_f() * P_NUM);
        const long i = (long)(rng_f() * (float)psz[p]);
        const float orig = par[p][i];
        par[p][i] = orig + h;
        const float lp = forward(&TR, ids, tgt, B);
        par[p][i] = orig - h;
        const float lm = forward(&TR, ids, tgt, B);
        par[p][i] = orig;
        const float num = (lp - lm) / (2.0f * h);
        const float ana = grad[p][i];
        /* plancher de bruit FD float32 mesur?? : sigma ~ 1.2e-2 avec h=1e-2.
           Crit??re : |num???ana| <= bruit(2e-2) + 10%??|ana| */
        const float tol = 2e-2f + 0.10f * fabsf(ana);
        const float err = fabsf(num - ana);
        checked++;
        sum += err / (fabsf(ana) + 2e-2f);
        if (err > worst) {
            worst = err;
            if (err > tol)
                printf("  gros ??cart : %s[%ld] ana=%.5f num=%.5f err=%.4f\n", pname[p], i, ana, num, err);
        }
    }
    printf("  %d param??tres test??s ?? erreur moyenne normalis??e = %.4f ?? pire abs = %.4f %s\n\n",
           checked, sum / (checked ? checked : 1), worst, worst < 5e-2f ? "PASS" : "FAIL");
    free(ids); free(tgt);
    cpuAvx2 = savedCpu;
}

/* ------------------------------------------------------------- microbench -- */
static float refGelu(float x) {
    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}
static void microbench(void) {
    const int N = 20000000;
    float sink = 0.0f;
    printf("== MICROBENCH kernels (20M appels, x86-64 -O2 -ffast-math) ==\n");
    struct { const char *name; float (*sp)(float); float (*ref)(float); float lo, hi; } ks[] = {
        { "exp   ", spear_expf, expf, -12.0f, 0.0f },
        { "rsqrt ", spear_rsqrtf, NULL, 0.001f, 100.0f },
        { "tanh  ", spear_tanhf, tanhf, -4.0f, 4.0f },
        { "gelu  ", NULL, refGelu, -4.0f, 4.0f },
    };
    for (int k = 0; k < 4; k++) {
        float x = ks[k].lo;
        const float step = (ks[k].hi - ks[k].lo) / 997.0f;
        /* r??f??rence */
        double t0 = now_s();
        for (int i = 0; i < N; i++) {
            x += step; if (x > ks[k].hi) x = ks[k].lo;
            if (ks[k].ref) sink += ks[k].ref(x);
            else sink += 1.0f / sqrtf(x);
        }
        const double tRef = now_s() - t0;
        /* SPEAR */
        t0 = now_s();
        for (int i = 0; i < N; i++) {
            x += step; if (x > ks[k].hi) x = ks[k].lo;
            if (k == 3) sink += spear_gelu_vd(x).v;
            else sink += ks[k].sp(x);
        }
        const double tSp = now_s() - t0;
        printf("  %s libm %6.1f ms | SPEAR %6.1f ms | ??%.2f\n",
               ks[k].name, tRef * 1e3, tSp * 1e3, tRef / tSp);
    }
    if (!isfinite(sink)) printf("  (sink=%g)\n", sink);
}

/* ------------------------------------------------- quantification E8 ------
 * Quantification vectorielle des poids par blocs de 8 sur le r??seau E8
 * (D8 ??? D8+??), scale par bloc, coords ??2 stockables sur 5 bits sign??s.
 * Bits/poids : (8??5 + 1 + 16)/8 = 7.125 vs FP32 32 ??? ??4.5 compression.
 * R??f. falsifi??e : tools/spear_lattice_verify.mjs (gain forme 1.164 mesur??).
 */
static float qe8MaxCoord = 15.0f; /* coord*2 ??? [???15, 15] ??? int5 */

static void projE8f(float *v) {
    /* D8 : arrondi + correction de parit?? sur la coordonn??e la plus ??loign??e */
    float z0[8], z1[8];
    {
        int32_t sum = 0;
        float maxd = -1.0f;
        int mi = 0;
        for (int i = 0; i < 8; i++) {
            z0[i] = roundf(v[i]);
            sum += (int32_t)z0[i];
            const float dd = fabsf(v[i] - z0[i]);
            if (dd > maxd) { maxd = dd; mi = i; }
        }
        if ((sum & 1) != 0) z0[mi] += (v[mi] > z0[mi]) ? 1.0f : -1.0f;
    }
    /* D8 + ?? : m??me algorithme sur v????? puis re-d??calage */
    {
        float zz[8];
        int32_t ss = 0;
        float md = -1.0f;
        int mj = 0;
        for (int i = 0; i < 8; i++) {
            const float sh = v[i] - 0.5f;
            zz[i] = roundf(sh);
            ss += (int32_t)zz[i];
            const float dd = fabsf(sh - zz[i]);
            if (dd > md) { md = dd; mj = i; }
        }
        if ((ss & 1) != 0) zz[mj] += ((v[mj] - 0.5f) > zz[mj]) ? 1.0f : -1.0f;
        for (int i = 0; i < 8; i++) z1[i] = zz[i] + 0.5f;
    }
    float d0 = 0.0f, d1 = 0.0f;
    for (int i = 0; i < 8; i++) {
        d0 += (v[i] - z0[i]) * (v[i] - z0[i]);
        d1 += (v[i] - z1[i]) * (v[i] - z1[i]);
    }
    const int use1 = d1 < d0;
    for (int i = 0; i < 8; i++) v[i] = use1 ? z1[i] : z0[i];
}

static void quantTensor(const char *name, float *w, long n, int mode) {
    /* mode 0 = FP32 (no-op), 1 = E8 ~7.1 bits/poids, 2 = INT8 bloc, 3 = INT4 bloc */
    if (mode == 0 || n < 8) return;
    double mse = 0.0, mag = 0.0;
    const long nblk = n / 8;
    for (long b = 0; b < nblk; b++) {
        float *w8 = w + b * 8;
        float mx = 1e-12f;
        for (int i = 0; i < 8; i++) mx = fmaxf(mx, fabsf(w8[i]));
        if (mode == 1) {
            /* E8 : scale pour que max|y| ??? 7.25 (coords??2 ??? 15) */
            const float s = mx / 7.25f;
            float y[8];
            for (int i = 0; i < 8; i++) y[i] = w8[i] / s;
            projE8f(y);
            for (int i = 0; i < 8; i++) {
                float c2 = y[i] * 2.0f;                       /* ??2 ??? demi-entiers */
                if (c2 > qe8MaxCoord) c2 = qe8MaxCoord;
                if (c2 < -qe8MaxCoord) c2 = -qe8MaxCoord;
                c2 = roundf(c2);                              /* stockable int5    */
                const float dq = (c2 / 2.0f) * s;
                mse += (w8[i] - dq) * (w8[i] - dq); mag += w8[i] * w8[i];
                w8[i] = dq;
            }
        } else {
            const int bits = (mode == 2) ? 8 : 4;
            const float lim = (float)((1 << (bits - 1)) - 1); /* 127 ou 7 */
            const float s = mx / lim;
            for (int i = 0; i < 8; i++) {
                float q = roundf(w8[i] / s);
                if (q > lim) q = lim; if (q < -lim) q = -lim - 1;
                const float dq = q * s;
                mse += (w8[i] - dq) * (w8[i] - dq); mag += w8[i] * w8[i];
                w8[i] = dq;
            }
        }
    }
    printf("  quant %-6s : erreur relative RMS = %.4f%%\n", name, 100.0 * sqrt(mse / (mag + 1e-12)));
}

static void quantModel(int mode) {
    const int mats[] = { P_WTE, P_QKVW, P_APW, P_F1W, P_F2W, P_WOUT };
    for (int p = 0; p < P_NUM; p++) {
        int isMat = 0;
        for (int m = 0; m < 6; m++) isMat |= (p == mats[m]);
        if (isMat) quantTensor(pname[p], par[p], psz[p], mode);
    }
    refreshWt();
}

static void sample(Acts *A, int nChars, float temp) {
    int ctx[TT];
    const long start = (long)(rng_f() * (float)(trainEnd - TT - 2));
    for (int t = 0; t < TT; t++) ctx[t] = byte2id[dataBuf[start + (long)t]];
    printf("== ??CHANTILLON (%d caract??res, T=%.2f) ==\n", nChars, temp);
    for (int t = 0; t < TT; t++) putchar(dataBuf[start + (long)t]);
    int ids[TT], tgtDummy[TT];
    for (int g = 0; g < nChars; g++) {
        for (int t = 0; t < TT; t++) { ids[t] = ctx[t]; tgtDummy[t] = ids[t]; }
        forward(A, ids, tgtDummy, 1);
        const float *row = A->logits + (size_t)(TT - 1) * (size_t)V;
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
        for (int b = 0; b < 256; b++) if (byte2id[b] == pick) { putchar(b); break; }
    }
    printf("\n\n");
}

/* ------------------------------------------------------------------- main -- */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
#if defined(_OPENMP) && defined(_WIN32)
    /* CRITIQUE : sans ça, les threads libgomp busy-waitent (spin SSE) pendant
       les sections scalaires/AVX du thread principal → ×50 de perte mesurée.
       Spin=0 : ils dorment, le main garde tous les cycles. */
    _putenv("GOMP_SPINCOUNT=0");
#endif
    const char *dataPath = "data/tinystories_valid.txt";
    int steps = 300, B = 8, bench = 0, gcheck = 0, gen = 300, resume = 0, quantMode = 0;
    float lr = 1.5e-3f;
    uint64_t seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) dataPath = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bs") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--gradcheck")) gcheck = 1;
        else if (!strcmp(argv[i], "--profile")) g_prof = 1;
        else if (!strcmp(argv[i], "--nogen")) gen = 0;
        else if (!strcmp(argv[i], "--nosimd")) cpuAvx2 = 0;
        else if (!strcmp(argv[i], "--gemm-old")) g_gemmOld = 1;
        else if (!strcmp(argv[i], "--resume")) resume = 1;
        else if (!strcmp(argv[i], "--quant-e8")) quantMode = 1;
        else if (!strcmp(argv[i], "--quant-int8")) quantMode = 2;
        else if (!strcmp(argv[i], "--quant-int4")) quantMode = 3;
        else if (!strcmp(argv[i], "--quant-int8")) quantMode = 2;
        else if (!strcmp(argv[i], "--quant-int4")) quantMode = 3;
    }

    if (loadData(dataPath) != 0) { fprintf(stderr, "donn??es introuvables : %s\n", dataPath); return 1; }
    printf("TinyStories : %.1f Mo ?? vocab %d ?? train %.1f Mo\n",
           (double)dataLen / 1048576.0, V, (double)trainEnd / 1048576.0);
    rng_seed(seed * 7919 + 13);
    modelInit();
    if (resume) {
        if (loadWeights("lm_c/checkpoint.bin") != 0)
            { fprintf(stderr, "aucun checkpoint compatible ??? init fraiche\n"); }
        else { printf("  poids repris depuis lm_c/checkpoint.bin\n"); refreshWt(); }
    }
    actsAlloc(&TR, B);
    actsAlloc(&SA, 1);
    scratchAlloc(B);
    printf("param??tres : %ld (%.1f k)\n\n", totalParams, (double)totalParams / 1000.0);

    /* auto-test GEMM : AVX2 doit matcher le scalaire (N%8 != 0 + nt_t + backward) */
    {
        enum { GM = 5, GK = 11, GN = 44 };
        static float gx[GM * GK], gw[GN * GK], gwt[GK * GN], g1[GM * GN], g2[GM * GN];
        static float gd1[GM * GK], gd2[GM * GK], gdw1[GN * GK], gdw2[GN * GK];
        for (int i = 0; i < GM * GK; i++) { gx[i] = rng_n(); gd1[i] = gd2[i] = rng_n(); }
        for (int i = 0; i < GN * GK; i++) gw[i] = rng_n();
        for (int k = 0; k < GK; k++)
            for (int n = 0; n < GN; n++) gwt[k * GN + n] = gw[n * GK + k];
        float dmax = 0.0f;
        cpuAvx2 = 0; mm_nt(g1, gx, gw, NULL, GM, GK, GN);
        cpuAvx2 = 1; mm_nt(g2, gx, gw, NULL, GM, GK, GN);
        for (int i = 0; i < GM * GN; i++) dmax = fmaxf(dmax, fabsf(g1[i] - g2[i]));
        cpuAvx2 = 0; mm_nt_t(g1, gx, gw, gwt, NULL, GM, GK, GN);
        cpuAvx2 = 1; mm_nt_t(g2, gx, gw, gwt, NULL, GM, GK, GN);
        for (int i = 0; i < GM * GN; i++) dmax = fmaxf(dmax, fabsf(g1[i] - g2[i]));
        /* backward : mm_dxd (dX = dY·W, W est [N][K]) + mm_tndw (dW) */
        for (int i = 0; i < GM * GK; i++) { gd1[i] = rng_n(); gd2[i] = gd1[i]; }
        static float gy[GM * GN];
        for (int i = 0; i < GM * GN; i++) gy[i] = rng_n();
        cpuAvx2 = 0; memset(gdw1, 0, sizeof gdw1); mm_dxd(gd1, gy, gw, GM, GK, GN);
        cpuAvx2 = 1; memset(gdw2, 0, sizeof gdw2); mm_dxd(gd2, gy, gw, GM, GK, GN);
        for (int i = 0; i < GM * GK; i++) dmax = fmaxf(dmax, fabsf(gd1[i] - gd2[i]));
        static float gW0a[GN * GK], gW0b[GN * GK];
        for (int i = 0; i < GN * GK; i++) { gW0a[i] = gW0b[i] = rng_n(); }
        cpuAvx2 = 0; mm_tndw(gW0a, NULL, gy, gx, GM, GK, GN);
        cpuAvx2 = 1; mm_tndw(gW0b, NULL, gy, gx, GM, GK, GN);
        for (int i = 0; i < GN * GK; i++) dmax = fmaxf(dmax, fabsf(gW0a[i] - gW0b[i]));
        printf("auto-test GEMM avx2/scal (nt+nt_t+dxd+tndw) : max|diff| = %.7f %s\n\n",
               dmax, dmax < 1e-4f ? "OK" : "** FAUX ??? AVX2 desactiv? **") ;
        if (dmax >= 1e-4f) { cpuAvx2 = 0; }
        else cpuAvx2 = -1;
    }

    if (gcheck) gradcheck(B);

    if (bench) { /* micro-bench GEMM : shapes réelles, mm_nt_t avx2 vs scalaire */
        struct { int M, K, N; const char *nm; } sh[] = {
            { 256, 72, 216, "qkv" }, { 256, 72, 72, "attn" }, { 256, 72, 288, "fc1" },
            { 256, 288, 72, "fc2" }, { 256, 72, V, "out" },
        };
        const int R = 200;
        printf("GEMM bench mm_nt_t (%d reps, min 3) :\n", R);
        for (int s = 0; s < 5; s++) {
            const int M = sh[s].M, K = sh[s].K, N = sh[s].N;
            float *X = (float *)malloc(sizeof(float) * (size_t)M * K);
            float *W = (float *)malloc(sizeof(float) * (size_t)K * N);
            float *Wt = (float *)malloc(sizeof(float) * (size_t)K * N);
            float *Y = (float *)malloc(sizeof(float) * (size_t)M * N);
            for (int i = 0; i < M * K; i++) X[i] = rng_n();
            for (int i = 0; i < K * N; i++) W[i] = rng_n();
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++) Wt[k * N + n] = W[n * K + k];
            double best[2] = { 1e9, 1e9 };
            for (int round = 0; round < 3; round++) {
                for (int pass = 0; pass < 2; pass++) {
                    cpuAvx2 = pass;
                    for (int it = 0; it < 10; it++) mm_nt_t(Y, X, W, Wt, NULL, M, K, N);
                    const double t0 = now_s();
                    for (int it = 0; it < R; it++) mm_nt_t(Y, X, W, Wt, NULL, M, K, N);
                    const double dt = now_s() - t0;
                    if (dt < best[pass]) best[pass] = dt;
                }
            }
            printf("  %s M=%d K=%d N=%d : scal %.1f ms  avx2 %.1f ms  x%.2f\n",
                   sh[s].nm, M, K, N, best[0] * 1e3, best[1] * 1e3,
                   best[0] / (best[1] > 0 ? best[1] : 1));
            free(X); free(W); free(Wt); free(Y);
        }
        cpuAvx2 = -1;
        printf("\n");
    }

    int *ids = (int *)malloc(sizeof(int) * (size_t)B * TT);
    int *tgt = (int *)malloc(sizeof(int) * (size_t)B * TT);

    /* snapshot pour l'A/B : m??mes poids, m??mes batches, m??me RNG */
    float *snap = (float *)malloc(sizeof(float) * (size_t)totalParams);
    memcpy(snap, arenaPar, sizeof(float) * (size_t)totalParams);

    double tokS[2] = { 0, 0 };
    float lossEnd[2] = { 0, 0 };
    int phases[2] = { 0, 1 };
    if (!bench) phases[0] = 1; /* sans --bench : une seule phase SPEAR, plus longue */
    else if (getenv("LM_REV")) { phases[0] = 1; phases[1] = 0; /* A/B inversé : annule le biais turbo/thermique de l'ordre */ }

    for (int ph = 0; ph < 2; ph++) {
        const int sp = phases[ph];
        if (bench && ph == 1) memcpy(arenaPar, snap, sizeof(float) * (size_t)totalParams);
        g_spear = sp;
        adamT = 0;
        rng_seed(0x5EED);
        for (int p = 0; p < P_NUM; p++) { memset(mom[p], 0, sizeof(float) * (size_t)psz[p]); memset(vel[p], 0, sizeof(float) * (size_t)psz[p]); }
        zeroGrads();

        double ema = 0.0;
        const double t0 = now_s();
        int first = 1;
        for (int st = 1; st <= steps; st++) {
            fillBatch(ids, tgt, B, trainEnd);
            zeroGrads();
            const float loss = forward(&TR, ids, tgt, B);
            backward(&TR, ids, tgt, B);
            clipGrads();
            adamStep(lr * fminf(1.0f, (float)st / 30.0f));
            refreshWt();
            ema = first ? loss : 0.95 * ema + 0.05 * loss;
            first = 0;
            if (st % 50 == 0 || st == 1) {
                const double el = now_s() - t0;
                const double tok = (double)st * B * TT;
                printf("[spear=%d] step %4d ?? loss %.3f (ema %.3f) ?? %.0f tok/s\n", sp, st, loss, ema, tok / el);
            }
        }
        tokS[sp] = (double)steps * B * TT / (now_s() - t0);
        lossEnd[sp] = ema;
        if (phases[0] == 1 && !bench) break;
    }

    printf("\n== R??SULTAT ==\n");
    if (bench)
        printf("  d??bit entra??nement complet : libm %.0f tok/s | SPEAR %.0f tok/s ??? ??%.2f\n",
               tokS[0], tokS[1], tokS[1] / tokS[0]);
    printf("  loss finale (ema) : %.3f ?? vocab %d (hasard = %.3f)\n",
           lossEnd[1], V, log((double)V));

    if (g_prof) {
        const double tot = g_tGemm + g_tAtt + g_tLn;
        if (tot > 0) {
            printf("  PROFIL : GEMMs %.1f%% ?? attention %.1f%% ?? layernorm %.1f%%  (reste = CE, embeddings, adam)\n",
                   100.0 * g_tGemm / tot, 100.0 * g_tAtt / tot, 100.0 * g_tLn / tot);
            printf("  TEMPS  : GEMM %.0f ms ?? attn %.0f ms ?? ln %.0f ms\n",
                   g_tGemm * 1e3, g_tAtt * 1e3, g_tLn * 1e3);
        }
    }
    /* vérité terrain FP32 (mesurée AVANT toute quantification) */
    const float vFP32 = valLoss(&TR, B, 20);
    printf("  val loss FP32 : %.3f\n", vFP32);

    if (quantMode) {
        printf("== QUANTIFICATION (%s) ==\n",
               quantMode == 1 ? "E8 ~7.1 bits/poids" : quantMode == 2 ? "INT8 bloc ~10 bits/poids" : "INT4 bloc ~6 bits/poids");
        quantModel(quantMode);
        const float vQ = valLoss(&TR, B, 20);
        printf("  val loss quantifiée : %.3f (delta %+.3f)\n", vQ, vQ - vFP32);
    }
    if (gen > 0) sample(&SA, gen, 0.8f);

    /* microbench toujours (rapide) */
    microbench();

    free(ids); free(tgt); free(snap);
    saveWeights("lm_c/checkpoint.bin");
    return 0;
}











