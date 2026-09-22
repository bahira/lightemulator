#include <immintrin.h>

/* Somme horizontale ultra-rapide des 8 floats d'un registre AVX2 */
float spear_horizontal_sum_avx2(__m256 v) {
    /* 1. Repliement 256 -> 128 bits : v_low + v_high */
    __m128 v_low  = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    __m128 v128   = _mm_add_ps(v_low, v_high);

    /* 2. Repliement 128 -> 64 bits */
    __m128 v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));

    /* 3. Repliement 64 -> 32 bits scalaire */
    __m128 v32 = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(1, 1, 1, 1)));

    return _mm_cvtss_f32(v32);
}

/* Produit scalaire 8D complet en 3 instructions */
float spear_dot_product_8d_fast(const float *a, const float *b) {
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 vprod = _mm256_mul_ps(va, vb);
    return spear_horizontal_sum_avx2(vprod);
}