#include <immintrin.h>
#include <stdio.h>
#include <math.h>

__m256 spear_fast_exp_avx2(__m256 vx) {
    const __m256 v_log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 v_ln2   = _mm256_set1_ps(0.6931471805599453f);
    const __m256 v_half  = _mm256_set1_ps(0.5f);

    __m256 v_t = _mm256_fmadd_ps(vx, v_log2e, v_half);
    __m256 v_k_real = _mm256_floor_ps(v_t);
    __m256i v_k_int = _mm256_cvtps_epi32(v_k_real);

    __m256 vr = _mm256_fnmadd_ps(v_k_real, v_ln2, vx);

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

    __m256i v_pow2k_int = _mm256_slli_epi32(_mm256_add_epi32(v_k_int, _mm256_set1_epi32(127)), 23);
    __m256 v_pow2k = _mm256_castsi256_ps(v_pow2k_int);

    __m256 result = _mm256_mul_ps(vp, v_pow2k);
    return result;
}

int main(void) {
    // Test various values
    float test_vals[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 5.0f, -5.0f, 8.0f, -8.0f};
    
    for (int i = 0; i < 9; i++) {
        __m256 v = _mm256_set1_ps(test_vals[i]);
        __m256 res = spear_fast_exp_avx2(v);
        float arr[8];
        _mm256_storeu_ps(arr, res);
        printf("exp_avx2(%+.1f) = [%f, %f, %f, %f, %f, %f, %f, %f]\n", 
            test_vals[i], arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
        printf("  vs libc exp(%+.1f) = %.10f\n", test_vals[i], expf(test_vals[i]));
    }
    
    return 0;
}