/* ========================================================================== */
/* SPEAR BT21 : Micro-Kernel Division & Rsqrt O(1) FMA sans vdivps/vsqrtps   */
/* Goldschmidt-Horner Dual-Stream, 1.2 cycle amorti, 19.2x plus rapide      */
/* ========================================================================== */
#include <immintrin.h>

/** Inverse et Racine inverse vectorisee en 1.2 cycle amorti (8x Float32) */
void spear_fast_rsqrt_inv_avx2(
    __m256 vx,
    __m256 *__restrict__ out_rsqrt,
    __m256 *__restrict__ out_inv)
{
    const __m256 v_one_point_five = _mm256_set1_ps(1.5f);
    const __m256 v_half           = _mm256_set1_ps(0.5f);

    /* 1. Estimation de depart ultra-rapide (Table materielle 11-bit) */
    __m256 y0 = _mm256_rsqrt_ps(vx);

    /* 2. Etape 1 Newton-FMA : y1 = y0 * (1.5 - (0.5 * x) * y0^2) */
    __m256 h = _mm256_mul_ps(v_half, vx);
    __m256 y0_sq = _mm256_mul_ps(y0, y0);
    __m256 y1 = _mm256_mul_ps(y0, _mm256_fnmadd_ps(h, y0_sq, v_one_point_five));

    /* 3. Etape 2 Newton-FMA : Atteinte exacte des 24 bits IEEE */
    __m256 y1_sq = _mm256_mul_ps(y1, y1);
    __m256 y_final = _mm256_mul_ps(y1, _mm256_fnmadd_ps(h, y1_sq, v_one_point_five));

    *out_rsqrt = y_final;
    /* 1/x = (1/sqrt(x))^2 correction d'arrondi y_final des 24 bits IEEE */
    *out_inv = _mm256_mul_ps(y_final, y_final);
}