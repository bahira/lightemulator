/* lm_bench_exp8.c — exp SCALAIRE vs exp AVX2 8-lanes (l'objectif ×3). */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) { struct timespec ts; timespec_get(&ts, TIME_UTC); return (double)ts.tv_sec + 1e-9 * ts.tv_nsec; }

static inline float sp_pow2i(int k) { union { float f; uint32_t u; } v; v.f = 1.0f; v.u = (uint32_t)(127 + k) << 23; return v.f; }
static inline float exp_poly_scalar(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float p = ((0.042356002f * r + 0.166933063f) * r + 0.499893348f) * r + 1.000043123f;
    return sp_pow2i(k) * (1.0f + r * p);
}

__attribute__((target("avx2,fma")))
static inline __m256 exp8(__m256 x) {
    const __m256 R2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 L2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c4 = _mm256_set1_ps(0.042356002f);
    const __m256 c3 = _mm256_set1_ps(0.166933063f);
    const __m256 c2 = _mm256_set1_ps(0.499893348f);
    const __m256 c1 = _mm256_set1_ps(1.000043123f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, R2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_fnmadd_ps(kf, L2, x);           /* x − k·ln2 */
    __m256 p = _mm256_fmadd_ps(c4, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    const __m256 poly = _mm256_fmadd_ps(r, p, one);          /* 1 + r·p   */
    const __m256i ki = _mm256_cvtps_epi32(kf);
    const __m256i ebits = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(_mm256_castsi256_ps(ebits), poly);
}

#define N 8000000
int main(void) {
    static float args[N];
    for (int i = 0; i < N; i++) args[i] = -12.0f * ((float)((i * 2654435761u) & 0xFFFFFF) / 16777216.0f);

    /* exactitude AVX2 vs libm */
    double worst = 0;
    for (int i = 0; i < N; i += 997) {
        union { __m256 v; float f[8]; } a, r;
        a.v = exp8(_mm256_set1_ps(args[i]));
        for (int j = 0; j < 8; j++) {
            r.f[j] = expf(args[i]);
            const float e = fabsf(a.f[j] - r.f[j]) / r.f[j];
            if (e > worst) worst = e;
        }
    }
    printf("exp8 AVX2 : L∞ rel vs expf = %.3e\n", worst);

    volatile float sink = 0.0f;
    /* libm */
    double t0 = now_s();
    for (int i = 0; i < N; i++) sink += expf(args[i]);
    const double tLibm = now_s() - t0;
    /* scalaire poly */
    t0 = now_s();
    for (int i = 0; i < N; i++) sink += exp_poly_scalar(args[i]);
    const double tScal = now_s() - t0;
    /* AVX2 8 lanes */
    t0 = now_s();
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 v = _mm256_loadu_ps(args + i);
        acc = _mm256_add_ps(acc, exp8(v));
    }
    const double tAvx = now_s() - t0;
    sink += ((float *)&acc)[0] + ((float *)&acc)[7];

    printf("  libm expf      %7.1f ms (%.2f ns/el)\n", tLibm * 1e3, tLibm * 1e9 / N);
    printf("  spear scal     %7.1f ms (%.2f ns/el) → ×%.2f\n", tScal * 1e3, tScal * 1e9 / N, tLibm / tScal);
    printf("  spear AVX2 x8  %7.1f ms (%.2f ns/el) → ×%.2f%s\n", tAvx * 1e3, tAvx * 1e9 / N, tLibm / tAvx,
           worst < 1e-4 ? "" : "  [PRECISION KO]");
    if (!isfinite(sink)) printf("sink nan\n");
    return 0;
}
