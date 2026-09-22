#include <immintrin.h>
#include <stdio.h>
#include <math.h>

__m256 spear_fast_exp_avx2(__m256 vx) {
    /* Clamp input to avoid overflow in integer conversion */
    const __m256 v_min = _mm256_set1_ps(-88.0f);
    const __m256 v_max = _mm256_set1_ps(88.0f);
    vx = _mm256_min_ps(_mm256_max_ps(vx, v_min), v_max);

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

    __m256 result = _mm256_mul_ps(vp, v_pow2k);
    return result;
}

float spear_online_logsumexp_avx2(const float *__restrict__ x, size_t n) {
    __m256 v_max = _mm256_set1_ps(-1e30f);
    __m256 v_sum = _mm256_setzero_ps();

    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&x[i]);

        __m256 v_new_max = _mm256_max_ps(v_max, v_x);

        /* DEBUG: print v_max and v_x */
        float max_buf[8], x_buf[8];
        _mm256_storeu_ps(max_buf, v_max);
        _mm256_storeu_ps(x_buf, v_x);
        printf("Before chunk %zu: v_max=", (i/8));
        for (int j = 0; j < 8; j++) printf("%f ", max_buf[j]);
        printf(", v_x=");
        for (int j = 0; j < 8; j++) printf("%f ", x_buf[j]);
        printf("\n");

        __m256 v_diff = _mm256_sub_ps(v_max, v_new_max);
        printf("v_max - v_new_max=");
        float diff_buf[8];
        _mm256_storeu_ps(diff_buf, v_diff);
        for (int j = 0; j < 8; j++) printf("%f ", diff_buf[j]);
        printf("\n");

        __m256 v_scale = spear_fast_exp_avx2(v_diff);
        float scale_buf[8];
        _mm256_storeu_ps(scale_buf, v_scale);
        printf("v_scale=");
        for (int j = 0; j < 8; j++) printf("%f ", scale_buf[j]);
        printf("\n");

        __m256 v_term = spear_fast_exp_avx2(_mm256_sub_ps(v_x, v_new_max));
        float term_buf[8];
        _mm256_storeu_ps(term_buf, v_term);
        printf("v_term=");
        for (int j = 0; j < 8; j++) printf("%f ", term_buf[j]);
        printf("\n");

        __m256 v_old_sum = v_sum;
        v_sum = _mm256_fmadd_ps(v_sum, v_scale, v_term);
        v_max = v_new_max;

        float sum_buf[8];
        _mm256_storeu_ps(sum_buf, v_sum);
        printf("After chunk %zu: v_sum=", (i/8));
        for (int j = 0; j < 8; j++) printf("%f ", sum_buf[j]);
        printf("\n");
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