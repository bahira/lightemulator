/* bench_mm_t.c — A/B ancien (1 chaîne) vs nouveau micro-kernel (4 tuiles) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

__attribute__((target("avx2,fma")))
static void mm_old(float *Y, const float *X, const float *Wtr, const float *b,
                   int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        for (int nb = 0; nb < N; nb += 8) {
            __m256 acc = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
            const float *wk = Wtr + nb;
            for (int k = 0; k < K; k++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(wk + (size_t)k * N), acc);
            _mm256_storeu_ps(ym + nb, acc);
        }
    }
}

__attribute__((target("avx2,fma")))
static void mm_new(float *Y, const float *X, const float *Wtr, const float *b,
                   int M, int K, int N) {
    const int N8 = N & ~7;
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        int nb = 0;
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

int main(void) {
    struct { int M, K, N; const char *nm; } sh[] = {
        { 256, 72, 216, "qkv" }, { 256, 72, 72, "attn" }, { 256, 72, 288, "fc1" },
        { 256, 288, 72, "fc2" }, { 256, 72, 104, "out" },
    };
    const int R = 200;
    printf("A/B mm_nt_t : old 1-chain vs new 4-tile (%d reps, min 3)\n", R);
    for (int s = 0; s < 5; s++) {
        const int M = sh[s].M, K = sh[s].K, N = sh[s].N;
        float *X = malloc(sizeof(float) * (size_t)M * K);
        float *Wt = malloc(sizeof(float) * (size_t)K * N);
        float *Y = malloc(sizeof(float) * (size_t)M * N);
        for (int i = 0; i < M * K; i++) X[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < K * N; i++) Wt[i] = (float)rand() / RAND_MAX - 0.5f;
        double best[2] = { 1e9, 1e9 };
        for (int round = 0; round < 3; round++) {
            for (int pass = 0; pass < 2; pass++) {
                void (*fn)(float *, const float *, const float *, const float *, int, int, int) =
                    pass ? mm_new : mm_old;
                for (int it = 0; it < 10; it++) fn(Y, X, Wt, NULL, M, K, N);
                const double t0 = now_s();
                for (int it = 0; it < R; it++) fn(Y, X, Wt, NULL, M, K, N);
                const double dt = now_s() - t0;
                if (dt < best[pass]) best[pass] = dt;
            }
        }
        printf("  %s M=%d K=%d N=%d : old %.1f ms  new %.1f ms  x%.3f\n",
               sh[s].nm, M, K, N, best[0] * 1e3, best[1] * 1e3, best[0] / best[1]);
        free(X); free(Wt); free(Y);
    }
    return 0;
}
