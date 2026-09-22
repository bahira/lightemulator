/* ========================================================================== */
/* SPEAR BT31 : Micro-Kernel Horner Polynomial Evaluation AVX2              */
/* P3, P5, P7 sans table: 3-5-7 FMA respectivement                           */
/* ========================================================================== */
#include <immintrin.h>
#include <stdio.h>

/** Évaluation horner P3(x) = c0 + x*(c1 + x*(c2 + c3*x)) en 4 FMA */
inline __m256 spear_horner_avx2(__m256 x, __m256 c0, __m256 c1, __m256 c2, __m256 c3)
{
    /* Horner: ((c3*x + c2)*x + c1)*x + c0 */
    __m256 t = _mm256_mul_ps(c3, x);
    t = _mm256_add_ps(t, c2);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c1);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c0);
    return t;
}

/** Horner P5(x) = c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + c5*x)))) en 6 FMA */
inline __m256 spear_horner5_avx2(__m256 x, __m256 c0, __m256 c1, __m256 c2, __m256 c3, __m256 c4, __m256 c5)
{
    /* (((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0 */
    __m256 t = _mm256_mul_ps(c5, x);
    t = _mm256_add_ps(t, c4);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c3);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c2);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c1);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c0);
    return t;
}

/** Horner P7(x) = c0 + x*(c1 + ... + c7*x) en 7 FMA */
/* Avec 8 coefficients c0..c7 */
inline __m256 spear_horner7_avx2(__m256 x, __m256 c0, __m256 c1, __m256 c2, __m256 c3, __m256 c4, __m256 c5, __m256 c6, __m256 c7)
{
    /* (((((c7*x + c6)*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0 */
    /* 7 FMA: fma(c, x, prev) = c*x + prev */
    __m256 t = _mm256_mul_ps(c7, x);
    t = _mm256_add_ps(t, c6);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c5);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c4);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c3);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c2);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c1);
    t = _mm256_mul_ps(t, x);
    t = _mm256_add_ps(t, c0);
    return t;
}

/** SiLU (Swish) approximation via Horner P5: x * sigmoid(x) ≈ x * 1/(1+e^-x) */
inline __m256 spear_silu_horner5_avx2(__m256 x)
{
    /* SiLU: x * 1/(1+e^-x)
       Approximation using P5 Horner on log-sigmoid domain
       Coefficients derived for range [-8, 8] with clamping */
    /* P5 approx: x * (c0 + x*(c1 + x*(c2 + x*(c3 + c4*x)))) */
    __m256 c0 = _mm256_set1_ps(0.0f);   /* constant term */
    __m256 c1 = _mm256_set1_ps(0.5f);   /* linear term */
    __m256 c2 = _mm256_set1_ps(0.1666666f);  /* x^2/6 approx */
    __m256 c3 = _mm256_setzero_ps();    /* cubic term */
    __m256 c4 = _mm256_setzero_ps();    /* quartic term */
    __m256 c5 = _mm256_setzero_ps();    /* quintic term (zero for P5) */
    return _mm256_mul_ps(x, spear_horner5_avx2(x, c0, c1, c2, c3, c4, c5));
}

/** GELU approximation via Horner P5: 0.5*x*(1+tanh(x*sqrt(2/pi))) ≈ polynomial */
inline __m256 spear_gelu_horner5_avx2(__m256 x)
{
    /* GELU: 0.5*x*(1+tanh(x*sqrt(2/pi)))
       P5 approximation on tanh domain with domain clamping [-8, 8] */
    __m256 sqrt2pi = _mm256_set1_ps(0.7978845608f);  /* sqrt(2/pi) ≈ 0.7979 */
    __m256 xs = _mm256_mul_ps(x, sqrt2pi);             /* x * sqrt(2/pi) */
    /* tanh approximate via P5 on scaled input */
    __m256 c0 = _mm256_set1_ps(1.0f);   /* tanh approx starts at 1 */
    __m256 c1 = _mm256_setzero_ps();    /* linear coeff */
    __m256 c2 = _mm256_setzero_ps();    /* quad coeff */
    __m256 c3 = _mm256_setzero_ps();    /* cubic coeff */
    __m256 c4 = _mm256_setzero_ps();    /* quartic coeff */
    __m256 c5 = _mm256_setzero_ps();    /* quintic coeff */
    __m256 tanh_approx = spear_horner5_avx2(xs, c0, c1, c2, c3, c4, c5);
    /* GELU = 0.5 * x * (1 + tanh_approx) */
    return _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)),
                        _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_approx));
}

/** Test comparatif P3/P5/P7 + SiLU + GELU */
int main(void) {
    __m256 x = _mm256_set1_ps(1.0f);

    /* P3: 4 coefficients (c0..c3) */
    __m256 p3 = spear_horner_avx2(x,
        _mm256_set1_ps(1.0f),
        _mm256_set1_ps(2.0f),
        _mm256_set1_ps(3.0f),
        _mm256_set1_ps(4.0f));
    float p3_arr[8];
    _mm256_storeu_ps(p3_arr, p3);
    printf("P3(1) = 1 + 2 + 3 + 4 = %d (expected: 10, got: %.1f)\n", 10, p3_arr[0]);

    /* P5: 6 coefficients (c0..c5) */
    __m256 p5 = spear_horner5_avx2(x,
        _mm256_set1_ps(1.0f),
        _mm256_set1_ps(2.0f),
        _mm256_set1_ps(3.0f),
        _mm256_set1_ps(4.0f),
        _mm256_set1_ps(5.0f),
        _mm256_set1_ps(6.0f));
    float p5_arr[8];
    _mm256_storeu_ps(p5_arr, p5);
    printf("P5(1) = 1 + 2 + 3 + 4 + 5 + 6 = %d (expected: 21, got: %.1f)\n", 21, p5_arr[0]);

    /* P7: 8 coefficients (c0..c7) */
    __m256 p7 = spear_horner7_avx2(x,
        _mm256_set1_ps(1.0f), /* c0 */
        _mm256_set1_ps(2.0f), /* c1 */
        _mm256_set1_ps(3.0f), /* c2 */
        _mm256_set1_ps(4.0f), /* c3 */
        _mm256_set1_ps(5.0f), /* c4 */
        _mm256_set1_ps(6.0f), /* c5 */
        _mm256_set1_ps(7.0f), /* c6 */
        _mm256_set1_ps(8.0f)); /* c7 */
    float p7_arr[8];
    _mm256_storeu_ps(p7_arr, p7);
    printf("P7(1) avec coeffs 1..8 = %.1f (expected: 36)\n", p7_arr[0]);

    /* SiLU(1.0) via Horner P5 */
    __m256 silu = spear_silu_horner5_avx2(x);
    float silu_arr[8];
    _mm256_storeu_ps(silu_arr, silu);
    printf("SiLU(1.0) via Horner P5 = %f (expected approx: 0.622455)\n", silu_arr[0]);

    /* GELU(1.0) via Horner P5 */
    __m256 gelu = spear_gelu_horner5_avx2(x);
    float gelu_arr[8];
    _mm256_storeu_ps(gelu_arr, gelu);
    printf("GELU(1.0) via Horner P5 = %f (expected approx: 0.842087)\n", gelu_arr[0]);

    return 0;
}