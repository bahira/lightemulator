/* bench_mm_isa.c — A/B/C GEMM mm_nt_t : scalaire vs AVX2 (4x8) vs AVX-512F (4x16)
 *
 * Le README affirmait « AVX-512 est hors de portée sur ce matériel ». Sur un CPU
 * qui expose avx512f, la question devient mesurable — et c'est le seul juge.
 *
 *   gcc -O2 -march=native -ffast-math -o bench_mm_isa bench_mm_isa.c -lm && ./bench_mm_isa
 *
 * Sortie : ms par variante, gain vs scalaire, écart-type relatif sur 5 rounds,
 * et max|diff| de chaque chemin vectoriel vs scalaire (parité).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ scalaire */
static void mm_scalar(float *Y, const float *X, const float *Wtr, const float *b,
                      int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * Wtr[(size_t)k * N + n];
            ym[n] = acc;
        }
    }
}

/* ---------------------------------------------------------- AVX2 4 tuiles 8 */
__attribute__((target("avx2,fma")))
static void mm_avx2(float *Y, const float *X, const float *Wtr, const float *b,
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
            for (int k = 0; k < K; k++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(Wtr + (size_t)k * N + nb), acc);
            _mm256_storeu_ps(ym + nb, acc);
        }
        for (int n = N8; n < N; n++) {
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * Wtr[(size_t)k * N + n];
            ym[n] = acc;
        }
    }
}

/* ------------------------------------------------- AVX-512F 4 tuiles de 16 */
__attribute__((target("avx512f,avx2,fma")))
static void mm_avx512(float *Y, const float *X, const float *Wtr, const float *b,
                      int M, int K, int N) {
    const int N64 = N & ~63;
    const int N16 = N & ~15;
    const int N8 = N & ~7;
    for (int m = 0; m < M; m++) {
        const float *xm = X + (size_t)m * K;
        float *ym = Y + (size_t)m * N;
        int nb = 0;
        for (; nb + 64 <= N64; nb += 64) {
            __m512 a0 = b ? _mm512_loadu_ps(b + nb) : _mm512_setzero_ps();
            __m512 a1 = b ? _mm512_loadu_ps(b + nb + 16) : _mm512_setzero_ps();
            __m512 a2 = b ? _mm512_loadu_ps(b + nb + 32) : _mm512_setzero_ps();
            __m512 a3 = b ? _mm512_loadu_ps(b + nb + 48) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m512 xk = _mm512_set1_ps(xm[k]);
                const float *w = Wtr + (size_t)k * N + nb;
                a0 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w), a0);
                a1 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w + 16), a1);
                a2 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w + 32), a2);
                a3 = _mm512_fmadd_ps(xk, _mm512_loadu_ps(w + 48), a3);
            }
            _mm512_storeu_ps(ym + nb, a0);
            _mm512_storeu_ps(ym + nb + 16, a1);
            _mm512_storeu_ps(ym + nb + 32, a2);
            _mm512_storeu_ps(ym + nb + 48, a3);
        }
        for (; nb + 16 <= N16; nb += 16) {
            __m512 acc = b ? _mm512_loadu_ps(b + nb) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(xm[k]), _mm512_loadu_ps(Wtr + (size_t)k * N + nb), acc);
            _mm512_storeu_ps(ym + nb, acc);
        }
        for (; nb + 8 <= N8; nb += 8) {
            __m256 acc = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(xm[k]), _mm256_loadu_ps(Wtr + (size_t)k * N + nb), acc);
            _mm256_storeu_ps(ym + nb, acc);
        }
        for (int n = N8; n < N; n++) {
            float acc = b ? b[n] : 0.0f;
            for (int k = 0; k < K; k++) acc += xm[k] * Wtr[(size_t)k * N + n];
            ym[n] = acc;
        }
    }
}


/* ------------------------------------- AVX2, 4 lignes M x 16 cols (8 acc) --
 * Le noyau 1-ligne relit W une fois par ligne M : sur ces formes (K*N*4 o
 * 62 kB), il est limite par la bande passante. Traiter 4 lignes M par tuile
 * divise le trafic W par 4 => intensite arithmetique x4.
 */
__attribute__((target("avx2,fma")))
static void mm_avx2_mb4(float *Y, const float *X, const float *Wtr, const float *b,
                        int M, int K, int N) {
    const int N16 = N & ~15;
    const int N8 = N & ~7;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float *x0 = X + (size_t)(m + 0) * K, *x1 = X + (size_t)(m + 1) * K;
        const float *x2 = X + (size_t)(m + 2) * K, *x3 = X + (size_t)(m + 3) * K;
        float *y0 = Y + (size_t)(m + 0) * N, *y1 = Y + (size_t)(m + 1) * N;
        float *y2 = Y + (size_t)(m + 2) * N, *y3 = Y + (size_t)(m + 3) * N;
        int nb = 0;
        for (; nb + 16 <= N16; nb += 16) {
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
            __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                const float *w = Wtr + (size_t)k * N + nb;
                const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
                a00 = _mm256_fmadd_ps(_mm256_set1_ps(x0[k]), w0, a00);
                a01 = _mm256_fmadd_ps(_mm256_set1_ps(x0[k]), w1, a01);
                a10 = _mm256_fmadd_ps(_mm256_set1_ps(x1[k]), w0, a10);
                a11 = _mm256_fmadd_ps(_mm256_set1_ps(x1[k]), w1, a11);
                a20 = _mm256_fmadd_ps(_mm256_set1_ps(x2[k]), w0, a20);
                a21 = _mm256_fmadd_ps(_mm256_set1_ps(x2[k]), w1, a21);
                a30 = _mm256_fmadd_ps(_mm256_set1_ps(x3[k]), w0, a30);
                a31 = _mm256_fmadd_ps(_mm256_set1_ps(x3[k]), w1, a31);
            }
            _mm256_storeu_ps(y0 + nb, a00); _mm256_storeu_ps(y0 + nb + 8, a01);
            _mm256_storeu_ps(y1 + nb, a10); _mm256_storeu_ps(y1 + nb + 8, a11);
            _mm256_storeu_ps(y2 + nb, a20); _mm256_storeu_ps(y2 + nb + 8, a21);
            _mm256_storeu_ps(y3 + nb, a30); _mm256_storeu_ps(y3 + nb + 8, a31);
        }
        for (; nb + 8 <= N8; nb += 8) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m256 w = _mm256_loadu_ps(Wtr + (size_t)k * N + nb);
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(x0[k]), w, a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(x1[k]), w, a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(x2[k]), w, a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(x3[k]), w, a3);
            }
            _mm256_storeu_ps(y0 + nb, a0); _mm256_storeu_ps(y1 + nb, a1);
            _mm256_storeu_ps(y2 + nb, a2); _mm256_storeu_ps(y3 + nb, a3);
        }
        for (int n = N8; n < N; n++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < K; k++) {
                const float w = Wtr[(size_t)k * N + n];
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
            }
            y0[n] = s0; y1[n] = s1; y2[n] = s2; y3[n] = s3;
        }
    }
    for (; m < M; m++) mm_avx2(Y + (size_t)m * N, X + (size_t)m * K, Wtr, b, 1, K, N);
    if (b) for (int i = 0; i < M * N; i++) Y[i] += b[i % N];
}

/* ------------------------- AVX512, 4 lignes M x 64 cols (16 acc zmm) ------ */
__attribute__((target("avx512f,avx2,fma")))
static void mm_avx512_mb4(float *Y, const float *X, const float *Wtr, const float *b,
                          int M, int K, int N) {
    const int N64 = N & ~63;
    const int N16 = N & ~15;
    const int N8 = N & ~7;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float *x0 = X + (size_t)(m + 0) * K, *x1 = X + (size_t)(m + 1) * K;
        const float *x2 = X + (size_t)(m + 2) * K, *x3 = X + (size_t)(m + 3) * K;
        float *y0 = Y + (size_t)(m + 0) * N, *y1 = Y + (size_t)(m + 1) * N;
        float *y2 = Y + (size_t)(m + 2) * N, *y3 = Y + (size_t)(m + 3) * N;
        int nb = 0;
        for (; nb + 64 <= N64; nb += 64) {
            __m512 a[4][4];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i][j] = _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                const float *w = Wtr + (size_t)k * N + nb;
                const __m512 w0 = _mm512_loadu_ps(w), w1 = _mm512_loadu_ps(w + 16);
                const __m512 w2 = _mm512_loadu_ps(w + 32), w3 = _mm512_loadu_ps(w + 48);
                const __m512 v0 = _mm512_set1_ps(x0[k]), v1 = _mm512_set1_ps(x1[k]);
                const __m512 v2 = _mm512_set1_ps(x2[k]), v3 = _mm512_set1_ps(x3[k]);
                a[0][0] = _mm512_fmadd_ps(v0, w0, a[0][0]); a[0][1] = _mm512_fmadd_ps(v0, w1, a[0][1]);
                a[0][2] = _mm512_fmadd_ps(v0, w2, a[0][2]); a[0][3] = _mm512_fmadd_ps(v0, w3, a[0][3]);
                a[1][0] = _mm512_fmadd_ps(v1, w0, a[1][0]); a[1][1] = _mm512_fmadd_ps(v1, w1, a[1][1]);
                a[1][2] = _mm512_fmadd_ps(v1, w2, a[1][2]); a[1][3] = _mm512_fmadd_ps(v1, w3, a[1][3]);
                a[2][0] = _mm512_fmadd_ps(v2, w0, a[2][0]); a[2][1] = _mm512_fmadd_ps(v2, w1, a[2][1]);
                a[2][2] = _mm512_fmadd_ps(v2, w2, a[2][2]); a[2][3] = _mm512_fmadd_ps(v2, w3, a[2][3]);
                a[3][0] = _mm512_fmadd_ps(v3, w0, a[3][0]); a[3][1] = _mm512_fmadd_ps(v3, w1, a[3][1]);
                a[3][2] = _mm512_fmadd_ps(v3, w2, a[3][2]); a[3][3] = _mm512_fmadd_ps(v3, w3, a[3][3]);
            }
            _mm512_storeu_ps(y0 + nb, a[0][0]); _mm512_storeu_ps(y0 + nb + 16, a[0][1]);
            _mm512_storeu_ps(y0 + nb + 32, a[0][2]); _mm512_storeu_ps(y0 + nb + 48, a[0][3]);
            _mm512_storeu_ps(y1 + nb, a[1][0]); _mm512_storeu_ps(y1 + nb + 16, a[1][1]);
            _mm512_storeu_ps(y1 + nb + 32, a[1][2]); _mm512_storeu_ps(y1 + nb + 48, a[1][3]);
            _mm512_storeu_ps(y2 + nb, a[2][0]); _mm512_storeu_ps(y2 + nb + 16, a[2][1]);
            _mm512_storeu_ps(y2 + nb + 32, a[2][2]); _mm512_storeu_ps(y2 + nb + 48, a[2][3]);
            _mm512_storeu_ps(y3 + nb, a[3][0]); _mm512_storeu_ps(y3 + nb + 16, a[3][1]);
            _mm512_storeu_ps(y3 + nb + 32, a[3][2]); _mm512_storeu_ps(y3 + nb + 48, a[3][3]);
        }
        for (; nb + 16 <= N16; nb += 16) {
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m512 w = _mm512_loadu_ps(Wtr + (size_t)k * N + nb);
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(x0[k]), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(x1[k]), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(x2[k]), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(x3[k]), w, a3);
            }
            _mm512_storeu_ps(y0 + nb, a0); _mm512_storeu_ps(y1 + nb, a1);
            _mm512_storeu_ps(y2 + nb, a2); _mm512_storeu_ps(y3 + nb, a3);
        }
        for (; nb + 8 <= N8; nb += 8) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m256 w = _mm256_loadu_ps(Wtr + (size_t)k * N + nb);
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(x0[k]), w, a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(x1[k]), w, a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(x2[k]), w, a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(x3[k]), w, a3);
            }
            _mm256_storeu_ps(y0 + nb, a0); _mm256_storeu_ps(y1 + nb, a1);
            _mm256_storeu_ps(y2 + nb, a2); _mm256_storeu_ps(y3 + nb, a3);
        }
        for (int n = N8; n < N; n++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < K; k++) {
                const float w = Wtr[(size_t)k * N + n];
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
            }
            y0[n] = s0; y1[n] = s1; y2[n] = s2; y3[n] = s3;
        }
    }
    if (m < M) mm_avx512(Y + (size_t)m * N, X + (size_t)m * K, Wtr, b, M - m, K, N);
    if (b) for (int i = 0; i < M * N; i++) Y[i] += b[i % N];
}

typedef void (*mmfn)(float *, const float *, const float *, const float *, int, int, int);

static double mean(const double *v, int n) { double s = 0; for (int i = 0; i < n; i++) s += v[i]; return s / n; }
static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    double c[64];
    for (int i = 0; i < n; i++) c[i] = v[i];
    qsort(c, n, sizeof(double), cmp_d);
    return (n & 1) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
}
static double sd(const double *v, int n) {
    const double m = mean(v, n);
    double s = 0;
    for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
    return n > 1 ? sqrt(s / (n - 1)) : 0.0;
}

static int have_avx512(void) {
    if (getenv("LM_NO512")) return 0; /* A/B : simule un CPU sans AVX-512 (CI) */
    return __builtin_cpu_supports("avx512f");
}

int main(void) {
    struct { int M, K, N; const char *nm; } sh[] = {
        { 256, 72, 216, "qkv" }, { 256, 72, 72, "attn" }, { 256, 72, 288, "fc1" },
        { 256, 288, 72, "fc2" }, { 256, 72, 104, "out" },
    };
    const mmfn fn[5] = { mm_scalar, mm_avx2, mm_avx512, mm_avx2_mb4, mm_avx512_mb4 };
    const char *name[5] = { "scalaire", "AVX2 4x8", "AVX512 4x16", "AVX2 4Mx16", "AVX512 4Mx64" };
    const int NF = 5;
    const int NS = 5, ROUNDS = 9, REPS = 200;

    printf("mm_nt_t — %d rounds x %d reps — rondes entrelacées, médiane + ratio apparié\n",
           ROUNDS, REPS);
    if (!have_avx512()) printf("avx512f absent sur ce CPU — variantes 512 marquées non exécutées\n");
    printf("\n");
    double tot[5] = { 0, 0, 0, 0, 0 };

    for (int s = 0; s < NS; s++) {
        const int M = sh[s].M, K = sh[s].K, N = sh[s].N;
        float *X = malloc(sizeof(float) * (size_t)M * K);
        float *Wt = malloc(sizeof(float) * (size_t)K * N);
        float *Y = malloc(sizeof(float) * (size_t)M * N);
        float *Yref = malloc(sizeof(float) * (size_t)M * N);
        srand(12345 + s);
        for (int i = 0; i < M * K; i++) X[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < K * N; i++) Wt[i] = (float)rand() / RAND_MAX - 0.5f;

        double t[5][ROUNDS], best[5] = { 1e9, 1e9, 1e9, 1e9, 1e9 };
        float dif[5] = { 0, 0, 0, 0, 0 };
        mm_scalar(Yref, X, Wt, NULL, M, K, N);

        /* rounds entrelacés : chaque round mesure les 5 variantes dans un ordre
           tournant — la dérive thermique/turbo frappe donc toutes les variantes. */
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NF; j++) {
                const int f = (j + r) % NF;
                if ((f == 2 || f == 4) && !have_avx512()) { t[f][r] = 1e9; continue; }
                const double t0 = now_s();
                for (int it = 0; it < REPS; it++) fn[f](Y, X, Wt, NULL, M, K, N);
                t[f][r] = (now_s() - t0) * 1e3 / REPS;
                if (t[f][r] < best[f]) best[f] = t[f][r];
            }
        }
        for (int f = 1; f < NF; f++) {
            if ((f == 2 || f == 4) && !have_avx512()) continue;
            fn[f](Y, X, Wt, NULL, M, K, N);
            for (int i = 0; i < M * N; i++) {
                const float d = fabsf(Y[i] - Yref[i]);
                if (d > dif[f]) dif[f] = d;
            }
        }

        printf("%-5s M=%-4d K=%-4d N=%-4d\n", sh[s].nm, M, K, N);
        for (int f = 0; f < NF; f++) {
            if ((f == 2 || f == 4) && !have_avx512()) continue; /* non mesuré : ni ligne ni total */
            double ratio[ROUNDS];
            for (int r = 0; r < ROUNDS; r++) ratio[r] = t[1][r] / t[f][r]; /* apparié vs AVX2 4x8 */
            const double mr = mean(ratio, ROUNDS), sr = sd(ratio, ROUNDS);
            printf("  %-13s med %7.3f ms  best %7.3f  x%5.2f/AVX2 (%.1fsigma)  x%5.2f/scal  max|diff| %.1e\n",
                   name[f], median(t[f], ROUNDS), best[f], mr, sr > 0 ? (mr - 1.0) / sr : 99.0,
                   mean(t[0], ROUNDS) / mean(t[f], ROUNDS), dif[f]);
            tot[f] += best[f];
        }
        free(X); free(Wt); free(Y); free(Yref);
    }

    const int a512 = have_avx512();
    int bestf = 1;
    printf("\nTOTAL (somme des best) :\n");
    for (int f = 0; f < NF; f++) {
        if ((f == 2 || f == 4) && !a512) continue;
        printf("  %-13s %6.2f ms  x%5.2f vs scalaire\n", name[f], tot[f], tot[0] / tot[f]);
        if (f > 1 && tot[f] < tot[bestf]) bestf = f;
    }
    printf("meilleur vs AVX2 4x8 : x%.3f (%s)\n", tot[1] / tot[bestf], name[bestf]);
    return 0;
}
