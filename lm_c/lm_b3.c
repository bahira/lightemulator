
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#define N 8000000
static float args[N];
static double now_s(void) { struct timespec ts; timespec_get(&ts, TIME_UTC); return (double)ts.tv_sec + 1e-9 * ts.tv_nsec; }
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
static inline __m512 e16(__m512 x) {
    const __m512 R2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 L2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 c4 = _mm512_set1_ps(0.042356002f);
    const __m512 c3 = _mm512_set1_ps(0.166933063f);
    const __m512 c2 = _mm512_set1_ps(0.499893348f);
    const __m512 c1 = _mm512_set1_ps(1.000043123f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 kf = _mm512_roundscale_ps(_mm512_mul_ps(x, R2), 0);
    const __m512 r = _mm512_fnmadd_ps(kf, L2, x);
    __m512 p = _mm512_fmadd_ps(c4, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    const __m512 poly = _mm512_fmadd_ps(r, p, one);
    const __m512i ki = _mm512_cvtps_epi32(kf);
    const __m512i eb = _mm512_slli_epi32(_mm512_add_epi32(ki, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(_mm512_castsi512_ps(eb), poly);
}
int main(void) { static float buf[1]; printf("x\\n");
    static __m512 a16[1]; static __m256 a8[1];
    int i; double t0, t1, t2; struct timespec ts;
    for (i = 0; i < N; i++) args[i] = (float)(i & 0x3FF) * 0.001f;
    t0 = now_s();
    {
        __m512 acc = _mm512_setzero_ps();
        for (i = 0; i < N; i += 16) acc = _mm512_add_ps(acc, e16(_mm512_loadu_ps(args + i)));
        _mm512_storeu_ps((float*)a16, acc);
    }
    t1 = now_s();
    {
        __m256 acc = _mm256_setzero_ps();
        for (i = 0; i < N; i += 8) acc = _mm256_add_ps(acc, e8(_mm256_loadu_ps(args + i)));
        _mm256_storeu_ps((float*)a8, acc);
    }
    t2 = now_s();
    printf("512 x16 : %6.1f ms (%.2f ns/el) sink=%.4f\n", (t1 - t0) * 1e3, (t1 - t0) * 1e9 / N, ((const float*)a16)[0]);
    printf("256 x8  : %6.1f ms (%.2f ns/el) sink=%.4f\n", (t2 - t1) * 1e3, (t2 - t1) * 1e9 / N, ((const float*)a8)[0]);
    printf("ratio x16/x8 = %.2f\n", (t2 - t1) / (t1 - t0));
    return 0;
}


