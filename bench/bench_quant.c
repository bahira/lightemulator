/*
 * bench_quant.c — quantification int8 / int4 réellement compactée.
 *
 * Référence : GEMM fp32 AVX2 bloqué 4 lignes M (le noyau utilisé par le modèle).
 * Candidats : int8 / int4 packés + déquant dans la tuile, en scalaire, AVX2
 * (maddubs) et AVX-512 VNNI (dpbusd) quand le CPU les expose.
 *
 * Mesure : mêmes 5 formes que le modèle (mm_nt_t), rondes entrelacées, médiane,
 * plus l'écart relatif au fp32 et les octets réellement occupés par les poids.
 *
 *   gcc -O2 -march=native -o bench_quant bench/bench_quant.c -lm && ./bench_quant
 *   LM_NO_VNNI=1 ./bench_quant     # ignore VNNI (A/B sur un CPU sans VNNI)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../lm_c/lm_quant.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* fp32 de référence : 4 lignes M × 16 colonnes, AVX2 */
__attribute__((target("avx2,fma")))
static void mm_fp32_mb4(float *Y, const float *X, const float *Wt, const float *b, int M, int K, int N) {
    const int N16 = N & ~15;
    for (int mb = 0; mb < (M >> 2); mb++) {
        const int m = mb << 2;
        const float *x0 = X + (size_t)(m + 0) * K, *x1 = X + (size_t)(m + 1) * K;
        const float *x2 = X + (size_t)(m + 2) * K, *x3 = X + (size_t)(m + 3) * K;
        for (int nb = 0; nb + 16 <= N16 || (nb == 0 && N16 == 0); nb += 16) {
            __m256 a00, a01, a10, a11, a20, a21, a30, a31;
            a00 = a01 = a10 = a11 = a20 = a21 = a30 = a31 = _mm256_setzero_ps();
            const int end = (nb + 16 <= N16) ? 16 : 0;
            if (!end) break;
            for (int k = 0; k < K; k++) {
                const float *w = Wt + (size_t)k * N + nb;
                const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
                const __m256 v0 = _mm256_set1_ps(x0[k]), v1 = _mm256_set1_ps(x1[k]);
                const __m256 v2 = _mm256_set1_ps(x2[k]), v3 = _mm256_set1_ps(x3[k]);
                a00 = _mm256_fmadd_ps(v0, w0, a00); a01 = _mm256_fmadd_ps(v0, w1, a01);
                a10 = _mm256_fmadd_ps(v1, w0, a10); a11 = _mm256_fmadd_ps(v1, w1, a11);
                a20 = _mm256_fmadd_ps(v2, w0, a20); a21 = _mm256_fmadd_ps(v2, w1, a21);
                a30 = _mm256_fmadd_ps(v3, w0, a30); a31 = _mm256_fmadd_ps(v3, w1, a31);
            }
            if (b) {
                const __m256 b0 = _mm256_loadu_ps(b + nb), b1 = _mm256_loadu_ps(b + nb + 8);
                a00 = _mm256_add_ps(a00, b0); a01 = _mm256_add_ps(a01, b1);
                a10 = _mm256_add_ps(a10, b0); a11 = _mm256_add_ps(a11, b1);
                a20 = _mm256_add_ps(a20, b0); a21 = _mm256_add_ps(a21, b1);
                a30 = _mm256_add_ps(a30, b0); a31 = _mm256_add_ps(a31, b1);
            }
            _mm256_storeu_ps(Y + (size_t)(m + 0) * N + nb, a00);
            _mm256_storeu_ps(Y + (size_t)(m + 0) * N + nb + 8, a01);
            _mm256_storeu_ps(Y + (size_t)(m + 1) * N + nb, a10);
            _mm256_storeu_ps(Y + (size_t)(m + 1) * N + nb + 8, a11);
            _mm256_storeu_ps(Y + (size_t)(m + 2) * N + nb, a20);
            _mm256_storeu_ps(Y + (size_t)(m + 2) * N + nb + 8, a21);
            _mm256_storeu_ps(Y + (size_t)(m + 3) * N + nb, a30);
            _mm256_storeu_ps(Y + (size_t)(m + 3) * N + nb + 8, a31);
        }
        /* queues de colonnes : scalaire */
        for (int r = 0; r < 4; r++) {
            const float *x = X + (size_t)(m + r) * K;
            float *y = Y + (size_t)(m + r) * N;
            for (int nb = N16; nb < N; nb++) {
                float s = b ? b[nb] : 0.f;
                for (int k = 0; k < K; k++) s += x[k] * Wt[(size_t)k * N + nb];
                y[nb] = s;
            }
        }
    }
    for (int m = (M & ~3); m < M; m++) {
        const float *x = X + (size_t)m * K;
        float *y = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            float s = b ? b[n] : 0.f;
            for (int k = 0; k < K; k++) s += x[k] * Wt[(size_t)k * N + n];
            y[n] = s;
        }
    }
}

typedef void (*qfn)(float *, const float *, const QMat *, const float *, int, int, int);
static void q8_scal(float *Y, const float *X, const QMat *Q, const float *b, int M, int K, int N) { mm_nt_t_q_scalar(Y, X, Q, b, M, K, N); }
static void q4_scal(float *Y, const float *X, const QMat *Q, const float *b, int M, int K, int N) { mm_nt_t_q4_scalar(Y, X, Q, b, M, K, N); }
#if defined(__x86_64__) || defined(_M_X64)
static int have_vnni(void) { if (getenv("LM_NO_VNNI")) return 0; return __builtin_cpu_supports("avx512vnni"); }
#endif

int main(void) {
    struct { int M, K, N; const char *nm; } sh[] = {
        { 256, 72, 216, "qkv" }, { 256, 72, 72, "attn" }, { 256, 72, 288, "fc1" },
        { 256, 288, 72, "fc2" }, { 256, 72, 104, "out" },
    };
    const int NS = 5, ROUNDS = 7, REPS = 50;
#define NF 6
    const char *name[NF] = { "fp32 AVX2 4Mx16", "int8 scalaire", "int8 AVX2", "int8 VNNI",
                             "int4 scalaire", "int4 AVX2" };
    double tot[NF] = { 0, 0, 0, 0, 0, 0 };
    size_t bytes8 = 0, bytes4 = 0, bytesf32 = 0;
    double worstr8 = 0, worstr4 = 0;
    int haveV = 0;
#if defined(__x86_64__) || defined(_M_X64)
    haveV = have_vnni();
#endif
    printf("quantification réelle — poids packés + déquant dans la tuile\n");
    if (!haveV) printf("VNNI absent (ou LM_NO_VNNI) — variante VNNI non exécutée\n");
    printf("\n(ms par appel, %d répétitions par mesure)\n", REPS);
    printf("%-4s %-14s %10s %10s %10s %10s %10s %10s %8s\n", "", "forme", "fp32", "int8s", "int8avx2",
           "int8vnni", "int4s", "int4avx2", "octets");

    for (int s = 0; s < NS; s++) {
        const int M = sh[s].M, K = sh[s].K, N = sh[s].N;
        float *X = malloc(sizeof(float) * (size_t)M * K);
        float *W = malloc(sizeof(float) * (size_t)K * N);
        float *Bb = malloc(sizeof(float) * (size_t)N);
        float *Y0 = malloc(sizeof(float) * (size_t)M * N);
        float *Yq = malloc(sizeof(float) * (size_t)M * N);
        srand(1234 + s);
        for (int i = 0; i < M * K; i++) X[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < K * N; i++) W[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int n = 0; n < N; n++) Bb[n] = (float)rand() / RAND_MAX - 0.5f;

        mm_fp32_mb4(Y0, X, W, Bb, M, K, N);
        QMat q8, q4;
        qMatPack(W, K, N, 8, &q8);
        qMatPack(W, K, N, 4, &q4);
        bytes8 += q8.bytes; bytes4 += q4.bytes; bytesf32 += (size_t)K * N * 4;

        /* écart relatif (L∞ / max|Y|) de chaque candidat */
        double refmax = 0;
        for (int i = 0; i < M * N; i++) { const double a = fabs(Y0[i]); if (a > refmax) refmax = a; }
        struct { const char *n; QMat *Q; } cand[4] = {
            { "int8 scalaire", &q8 }, { "int8 AVX2", &q8 }, { "int4 scalaire", &q4 }, { "int4 AVX2", &q4 },
        };
        double relerr[4] = { 0, 0, 0, 0 };
        for (int c = 0; c < 4; c++) {
            qfn f = (c == 0) ? q8_scal : (c == 1) ? (qfn)mm_nt_t_q8_avx2 : (c == 2) ? q4_scal : (qfn)mm_nt_t_q4_avx2;
            f(Yq, X, cand[c].Q, Bb, M, K, N);
            double mx = 0;
            for (int i = 0; i < M * N; i++) { const double d = fabs(Yq[i] - Y0[i]); if (d > mx) mx = d; }
            relerr[c] = mx / (refmax > 1e-9 ? refmax : 1e-9);
        }
        if (relerr[1] > worstr8) worstr8 = relerr[1];
        if (relerr[3] > worstr4) worstr4 = relerr[3];

        double t[NF][ROUNDS], best[NF];
        for (int f = 0; f < NF; f++) best[f] = 1e9;
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NF; j++) {
                const int f = (j + r) % NF;
                if (f == 3 && !haveV) { t[f][r] = 1e9; continue; }
                QMat *Q = (f <= 3) ? &q8 : &q4;
                qfn fn = (f == 1) ? q8_scal : (f == 2) ? (qfn)mm_nt_t_q8_avx2
#ifdef __x86_64__
                         : (f == 3) ? (qfn)mm_nt_t_q8_vnni
#endif
                         : (f == 4) ? q4_scal : (qfn)mm_nt_t_q4_avx2;
                const double t0 = now_s();
                for (int it = 0; it < REPS; it++) {
                    if (f == 0) mm_fp32_mb4(Yq, X, W, Bb, M, K, N);
                    else fn(Yq, X, Q, Bb, M, K, N);
                }
                const double dt = (now_s() - t0) * 1e3 / REPS;
                t[f][r] = dt;
                if (dt < best[f]) best[f] = dt;
            }
        }
        for (int f = 0; f < NF; f++) tot[f] += best[f];
        printf("%-4s %-14s", "", sh[s].nm);
        for (int f = 0; f < NF; f++) {
            if (f == 3 && !haveV) printf(" %10s", "—");
            else printf(" %10.3f", best[f]);
        }
        printf(" %8zu\n", q8.bytes);
        qMatFree(&q8); qMatFree(&q4);
        free(X); free(W); free(Bb); free(Y0); free(Yq);
    }

    printf("\nTOTAL (somme des best, ms) :\n");
    for (int f = 0; f < NF; f++) {
        if (f == 3 && !haveV) continue;
        printf("  %-16s %8.3f ms   x%.2f vs fp32\n", name[f], tot[f], tot[0] / tot[f]);
    }
    printf("\noccupation mémoire des poids (5 formes) : fp32 %.1f KiB → int8 %.1f KiB (x%.1f) → int4 %.1f KiB (x%.1f)\n",
           bytesf32 / 1024.0, bytes8 / 1024.0, (double)bytesf32 / bytes8, bytes4 / 1024.0, (double)bytesf32 / bytes4);
    printf("erreur relative (L∞ / max|Y|) : int8 %.2f %% · int4 %.2f %%\n", worstr8 * 100, worstr4 * 100);
    const int ok8 = worstr8 < 0.02, ok4 = worstr4 < 0.15;
    printf("PARITE QUANT : %s (int8 <2 %%, int4 <15 %%)\n", (ok8 && ok4) ? "OK" : "FAIL");
    return (ok8 && ok4) ? 0 : 1;
}
