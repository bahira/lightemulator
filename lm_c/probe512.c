/* probe512.c — AVX512 disponible ? fmadd 512 crash-t-il vraiment ? */
#include <stdio.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
static float probe_fmadd(const float *a) {
    __m512 v = _mm512_loadu_ps(a);
    __m512 r = _mm512_fmadd_ps(v, v, v);
    r = _mm512_add_ps(r, _mm512_fnmadd_ps(v, v, v));
    float o[16];
    _mm512_storeu_ps(o, r);
    return o[0] + o[15];
}

__attribute__((target("avx512f")))
static float probe_cvt(const float *a) {
    __m512 v = _mm512_loadu_ps(a);
    __m512i i = _mm512_cvtps_epi32(v);
    __m512 b = _mm512_cvtepi32_ps(i);
    float o[16];
    _mm512_storeu_ps(o, b);
    return o[3];
}

int main(void) {
    float a[16];
    for (int i = 0; i < 16; i++) a[i] = (float)i * 0.1f;
    printf("cpu avx512f = %d\n", __builtin_cpu_supports("avx512f"));
    fflush(stdout);
    printf("fmadd  : %.3f\n", probe_fmadd(a)); fflush(stdout);
    printf("cvt    : %.3f\n", probe_cvt(a)); fflush(stdout);
    printf("OK\n");
    return 0;
}
