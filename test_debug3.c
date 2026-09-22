#include <immintrin.h>
#include <stdio.h>
#include <math.h>

extern __m256 spear_fast_exp_avx2(__m256 vx);

float spear_online_logsumexp_avx2(const float *__restrict__ x, size_t n) {
    __m256 v_max = _mm256_set1_ps(-1e30f);
    __m256 v_sum = _mm256_setzero_ps();

    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {  /* Only process full chunks */
        __m256 v_x = _mm256_loadu_ps(&x[i]);

        __m256 v_new_max = _mm256_max_ps(v_max, v_x);

        __m256 v_scale = spear_fast_exp_avx2(_mm256_sub_ps(v_max, v_new_max));

        __m256 v_term = spear_fast_exp_avx2(_mm256_sub_ps(v_x, v_new_max));

        __m256 v_old_sum = v_sum;
        v_sum = _mm256_fmadd_ps(v_sum, v_scale, v_term);
        v_max = v_new_max;

        printf("Chunk %zu: max=",
            (i / 8));
        float max_arr[8], sum_arr[8];
        _mm256_storeu_ps(max_arr, v_max);
        _mm256_storeu_ps(sum_arr, v_sum);
        for (int j = 0; j < 8; j++) printf("%f ", max_arr[j]);
        printf(", sum=");
        for (int j = 0; j < 8; j++) printf("%f ", sum_arr[j]);
        printf("\n");
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        float v = x[i];
        /* Simple scalar processing for remaining */
        /* ... */
    }

    /* Reduction finale des 8 voies AVX2 */
    float max_arr[8], sum_arr[8];
    _mm256_storeu_ps(max_arr, v_max);
    _mm256_storeu_ps(sum_arr, v_sum);

    float global_max = -1e30f;
    for (int j = 0; j < 8; j++) {
        if (max_arr[j] > global_max) global_max = max_arr[j];
    }

    float global_sum = 0.0f;
    for (int j = 0; j < 8; j++) {
        global_sum += sum_arr[j] * expf(max_arr[j] - global_max);
    }

    return global_max + logf(global_sum);
}

int main(void) {
    alignas(32) float data[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f, 1.0f, 0.0f
    };
    
    float lse = spear_online_logsumexp_avx2(data, 16);
    printf("LSE = %.10f\n", lse);
    return 0;
}