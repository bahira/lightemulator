/* lm_bench_exp16.c — exp: libm vs scalaire vs AVX2 (8 lanes) vs AVX512F (16 lanes).
 * ponytail: pas de cvt/fmadd 512 (cassés dans MinGW gcc 15.2) → poly deg9 pour 2^k. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#define N 8000000
static float args[N];
static double now_ms(void) { struct timespec ts; timespec_get(&ts, TIME_UTC); return 1e3 * (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
static inline float spow2i(int k) { union { float f; uint32_t u; } v; v.f = 1.0f; v.u = (uint32_t)(127 + k) << 23; return v.f; }
static inline float e1(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float p = ((0.042356002f * r + 0.166933063f) * r + 0.499893348f) * r + 1.000043123f;
    return spow2i(k) * (1.0f + r * p);
}
__attribute__((target("avx2,fma")))
static inline __m256 e8(__m256 x) {
    const __m256 R2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 L2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c4 = _mm256_set1_ps(0.042356002f);
    const __m256 c3 = _mm256_set1_ps(0.166933063f);
    const __m256 c2 = _mm256_set1_ps(0.499893348f);
    const __m256 c1 = _mm256_set1_ps(1.000043123f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, R2), 0);
    const __m256 r = _mm256_fnmadd_ps(kf, L2, x);
    __m256 p = _mm256_fmadd_ps(c4, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    const __m256 poly = _mm256_fmadd_ps(r, p, one);
    const __m256i ki = _mm256_cvtps_epi32(kf);
    const __m256i eb = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(_mm256_castsi256_ps(eb), poly);
}
__attribute__((target("avx512f")))
static inline void e16(const __m512 *xi, __m512 *o) {
    const __m512 x = *xi;
    const __m512 R2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 L2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 kf = _mm512_sub_ps(_mm512_add_ps(_mm512_mul_ps(x, R2), _mm512_set1_ps(12582912.0f)), _mm512_set1_ps(12582912.0f));
    const __m512 r = _mm512_sub_ps(x, _mm512_mul_ps(kf, L2));
    __m512 p = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(0.042356002f), r), _mm512_set1_ps(0.166933063f));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(0.499893348f));
    p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(1.000043123f));
    const __m512 poly = _mm512_add_ps(_mm512_mul_ps(r, p), one);
    __m512 pw = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(1.0185e-8f), kf), _mm512_set1_ps(1.3228e-6f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(1.52527e-5f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(1.54033e-4f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(1.33336e-3f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(9.61805e-3f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(5.55041e-2f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(2.4016e-1f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), _mm512_set1_ps(6.93147e-1f));
    pw = _mm512_add_ps(_mm512_mul_ps(pw, kf), one);
    *o = _mm512_mul_ps(pw, poly);
}
static double wL(void) { double a = now_ms(), s = 0.0; int i; for (i = 0; i < N; i++) s += expf(args[i]); if (s < 0.0) printf("x"); return now_ms() - a; }
static double wS(void) { double a = now_ms(), s = 0.0; int i; for (i = 0; i < N; i++) s += e1(args[i]); if (s < 0.0) printf("x"); return now_ms() - a; }
__attribute__((target("avx2,fma")))
static double w8(float *out) { __m256 x = _mm256_setzero_ps(); double a = now_ms(); int i; for (i = 0; i < N; i += 8) x = _mm256_add_ps(x, e8(_mm256_loadu_ps(args + i))); _mm256_storeu_ps(out, x); return now_ms() - a; }
__attribute__((target("avx512f")))
static double w16(float *out) { __m512 x = _mm512_setzero_ps(), t, in; double a = now_ms(); int i; for (i = 0; i < N; i += 16) { in = _mm512_loadu_ps(args + i); e16(&in, &t); x = _mm512_add_ps(x, t); } _mm512_storeu_ps(out, x); return now_ms() - a; }
int main(void) {
    int i; double tl, tc, t8, t16; float v8[8], v16[16];
    for (i = 0; i < N; i++) args[i] = (float)(i & 1023) * 0.001f;
    tl = wL(); tc = wS(); t8 = w8(v8); t16 = w16(v16);
    printf("libm expf   %7.1f ms  (%.2f ns/el)\n", tl, tl * 1e6 / N);
    printf("spear scal  %7.1f ms  (%.2f ns/el)  x%.2f\n", tc, tc * 1e6 / N, tl / tc);
    printf("spear avx2  %7.1f ms  (%.2f ns/el)  x%.2f\n", t8, t8 * 1e6 / N, tl / t8);
    printf("spear 512   %7.1f ms  (%.2f ns/el)  x%.2f  (x%.2f vs avx2)\n", t16, t16 * 1e6 / N, tl / t16, t8 / t16);
    printf("refs: expf=%.7f scal=%.7f  v8/n=%.4f  v16/n=%.4f\n",
           expf(args[0]), e1(args[0]), v8[0] / (1e6 / 8.0), v16[0] / (5e5));
    return 0;
}
