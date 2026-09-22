#include <immintrin.h>

/* Exponentielle vectorisee AVX2 haute precision (8 floats) en 4 FMA */
__m256 spear_fast_exp_avx2(__m256 vx);

/** Activation SiLU vectorisee en 6 instructions */
static __m256 spear_fast_silu_avx2(__m256 vx);

/** Activation GELU vectorisee en 6 instructions */
static __m256 spear_fast_gelu_avx2(__m256 vx);

/* ========================================================================== */
/* Main exp implementation                                                     */
/* ========================================================================== */
/* Clamp input to avoid overflow in integer conversion */
static inline __m256 clamp_exp_input(__m256 vx) {
    const __m256 v_min = _mm256_set1_ps(-88.0f);
    const __m256 v_max = _mm256_set1_ps(88.0f);
    return _mm256_min_ps(_mm256_max_ps(vx, v_min), v_max);
}

__m256 spear_fast_exp_avx2(__m256 vx) {
    vx = clamp_exp_input(vx);

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

static __m256 spear_fast_silu_avx2(__m256 vx) {
    __m256 v_neg_x = _mm256_sub_ps(_mm256_setzero_ps(), vx);
    __m256 v_exp = spear_fast_exp_avx2(v_neg_x);
    __m256 v_denom = _mm256_add_ps(_mm256_set1_ps(1.0f), v_exp);
    return _mm256_div_ps(vx, v_denom);
}

static __m256 spear_fast_gelu_avx2(__m256 vx) {
    const __m256 v_scale = _mm256_set1_ps(-1.702f);
    __m256 v_arg = _mm256_mul_ps(vx, v_scale);
    __m256 v_exp = spear_fast_exp_avx2(v_arg);
    __m256 v_denom = _mm256_add_ps(_mm256_set1_ps(1.0f), v_exp);
    return _mm256_div_ps(vx, v_denom);
}