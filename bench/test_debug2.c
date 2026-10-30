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

    return _mm256_mul_ps(vp, v_pow2k);
}

float spear_online_logsumexp_avx2_debug(const float *__restrict__ x, size_t n) {
    __m256 v_max = _mm256_set1_ps(-1e30f);
    __m256 v_sum = _mm256_setzero_ps();

    size_t i;
    for (i = 0; i < n; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&x[i]);

        __m256 v_new_max = _mm256_max_ps(v_max, v_x);

        __m256 v_scale = spear_fast_exp_avx2(_mm256_sub_ps(v_max, v_new_max));

        __m256 v_term = spear_fast_exp_avx2(_mm256_sub_ps(v_x, v_new_max));

        __m256 v_old_sum = v_sum;
        v_sum = _mm256_fmadd_ps(v_sum, v_scale, v_term);
        v_max = v_new_max;

        /* Debug print every 8 elements */
        if (i % 64 == 0 || i == n - 8) {
            float max_arr[8], sum_arr[8];
            _mm256_storeu_ps(max_arr, v_max);
            _mm256_storeu_ps(sum_arr, v_sum);
            printf("After chunk starting at %zu: max=%f, sum=%f\n", i/8, max_arr[0], sum_arr[0]);
            for (int j = 0; j < 8; j++) {
                printf("  [%d] max_arr=%f, sum_arr=%f\n", j, max_arr[j], sum_arr[j]);
            }
        }
    }

    /* Reduction finale des 8 voies AVX2 */
    float max_arr[8], sum_arr[8];
    _mm256_storeu_ps(max_arr, v_max);
    _mm256_storeu_ps(sum_arr, v_sum);

    float global_max = -1e30f;
    for (int i = 0; i < 8; ++i) {
        if (max_arr[i] > global_max) global_max = max_arr[i];
    }

    printf("global_max from arr: %f\n", global_max);
    for (int i = 0; i < 8; i++) {
        printf("max_arr[%d] = %f\n", i, max_arr[i]);
    }

    float global_sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float term = sum_arr[i] * expf(max_arr[i] - global_max);
        printf("term[%d] = %f * expf(%f - %f) = %f\n", i, sum_arr[i], max_arr[i], global_max, term);
        global_sum += term;
    }

    printf("global_sum = %f\n", global_sum);
    
    return global_max + logf(global_sum);
}

int main(void) {
    alignas(32) float data[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f, 1.0f, 0.0f
    };
    
    float lse = spear_online_logsumexp_avx2_debug(data, 16);
    printf("Final LSE = %.10f\n", lse);
    return 0;
}