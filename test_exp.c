#include <immintrin.h>
#include <stdio.h>
#include <math.h>

__m256 spear_fast_exp_avx2(__m256 vx) {
    const __m256 v_log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 v_ln2   = _mm256_set1_ps(0.6931471805599453f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);

    /* 1. k = floor(x * log2(e) + 0.5) */
    __m256 v_t = _mm256_fmadd_ps(vx, v_log2e, v_half);
    __m256 v_k_real = _mm256_floor_ps(v_t);
    __m256i v_k_int = _mm256_cvtps_epi32(v_k_real);

    /* 2. r = x - k * ln(2) */
    __m256 vr = _mm256_fnmadd_ps(v_k_real, v_ln2, vx);

    /* 3. Horner P5(r) = 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120)))) */
    const __m256 c5 = _mm256_set1_ps(0.0083333333f);
    const __m256 c4 = _mm256_set1_ps(0.0416666667f);
    const __m256 c3 = _mm256_set1_ps(0.1666666667f);
    const __m256 c2 = _mm256_set1_ps(0.5000000000f);
    const __m256 c1 = _mm256_set1_ps(1.0000000000f);

    __m256 vp = _mm256_fmadd_ps(vr, c5, c4);
    vp = _mm256_fmadd_ps(vr, vp, c3);
    vp = _mm256_fmadd_ps(vr, vp, c2);
    vp = _mm256_fmadd_ps(vr, vp, c1);
    vp = _mm256_fmadd_ps(vr, vp, c1);

    /* 4. Injection de 2^k par bit-shift entier direct : ((k + 127) << 23) */
    __m256i v_pow2k_int = _mm256_slli_epi32(_mm256_add_epi32(v_k_int, _mm256_set1_epi32(127)), 23);
    __m256 v_pow2k = _mm256_castsi256_ps(v_pow2k_int);

    return _mm256_mul_ps(vp, v_pow2k);
}

int main(void) {
    /* Test exp(0) = 1 */
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_exp = spear_fast_exp_avx2(v_zero);
    float arr[8];
    _mm256_storeu_ps(arr, v_exp);
    printf("exp(0) = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", 
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
    
    /* Test exp(1) */
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_exp1 = spear_fast_exp_avx2(v_one);
    _mm256_storeu_ps(arr, v_exp1);
    printf("exp(1) = %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f (expected ~2.718)\n", 
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
    
    return 0;
}