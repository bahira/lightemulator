#include <immintrin.h>
#include <stdio.h>
#include <math.h>

extern __m256 spear_fast_exp_avx2(__m256 vx);

float spear_online_logsumexp_avx2(const float *__restrict__ x, size_t n) {
    __m256 v_max = _mm256_set1_ps(-1e30f);
    __m256 v_sum = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&x[i]);

        __m256 v_new_max = _mm256_max_ps(v_max, v_x);

        __m256 v_scale = spear_fast_exp_avx2(_mm256_sub_ps(v_max, v_new_max));

        __m256 v_term = spear_fast_exp_avx2(_mm256_sub_ps(v_x, v_new_max));

        v_sum = _mm256_fmadd_ps(v_sum, v_scale, v_term);
        v_max = v_new_max;
    }

    float max_arr[8], sum_arr[8];
    _mm256_storeu_ps(max_arr, v_max);
    _mm256_storeu_ps(sum_arr, v_sum);

    float global_max = -1e30f;
    for (int i = 0; i < 8; ++i) {
        if (max_arr[i] > global_max) global_max = max_arr[i];
    }

    float global_sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        global_sum += sum_arr[i] * expf(max_arr[i] - global_max);
    }

    return global_max + logf(global_sum);
}

int main(void) {
    /* Test with just 2 elements: [1, 2] */
    float data[2] = {1.0f, 2.0f};
    float lse = spear_online_logsumexp_avx2(data, 2);
    printf("LSE([1,2]) = %.10f (expected ~2.313)\n", lse);
    
    /* Test with 8 elements: [1,2,3,4,5,6,7,8] */
    alignas(32) float data8[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float lse8 = spear_online_logsumexp_avx2(data8, 8);
    printf("LSE([1..8]) = %.10f (expected ~8.467)\n", lse8);
    
    /* Test with 16 elements as in the harness */
    alignas(32) float data16[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f, 1.0f, 0.0f
    };
    float lse16 = spear_online_logsumexp_avx2(data16, 16);
    printf("LSE([16 data]) = %.10f (expected ~8.467)\n", lse16);
    
    return 0;
}