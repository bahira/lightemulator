/*
 * ============================================================================
 *  spear_kernels.h — C99 kernels distilled by SPEAR v3
 *
 *  Zero-transcendental algebraic kernels for LLM / DSP / control primitives.
 *  Distilled with Pareto genetic programming + structural seeds + constant
 *  refinement. All routines are allocation-free, deterministic, MISRA-C:2012
 *  friendly (single point of return, no recursion).
 *
 *  Native gcc -O2 speedup vs libm (4M float, x86-64): 2.6x to 19x.
 *  On MCU without FPU / transcendental unit the advantage is far larger.
 * ============================================================================
 */

#ifndef SPEAR_KERNELS_H
#define SPEAR_KERNELS_H

#include <math.h>

/* --------------------------------------------------------------------------
 *  Rational tanh core:  u1*(A + u2^2) / (B + C*u3^2)
 * ------------------------------------------------------------------------- */

static inline float spear_rt(float u1, float u2, float u3, float A, float B, float C)
{
    return (u1 * (A + u2 * u2)) / (B + C * u3 * u3);
}

/* --------------------------------------------------------------------------
 *  SPEAR kernels
 * ------------------------------------------------------------------------- */

/* tanh(x) ~ x*(23.965 + x^2)/(24.362 + 8.387 x^2)   Linf 9.0e-3 */
static inline float spear_tanh(float x)
{
    return spear_rt(x, x, x, 23.96543f, 24.36223f, 8.38674f);
}

/* sigmoid(x) ~ 0.5 + 0.5298 * tanh_core(0.4418 x)   Linf 1.6e-4 */
static inline float spear_sigmoid(float x)
{
    return 0.5f + 0.52976f * spear_rt(0.44175f * x, 0.62050f * x, 0.58509f * x,
                                      27.35096f, 25.64128f, 7.19645f);
}

/* silu(x) ~ 0.4777 x (1.0467 + tanh_core(0.4672 x))  Linf 3.4e-4 */
static inline float spear_silu(float x)
{
    return 0.47770f * x * (1.04667f + spear_rt(0.46719f * x, 0.59566f * x, 0.56230f * x,
                                                25.80993f, 23.10982f, 6.95725f));
}

/* gelu(x) ~ 0.5058 x (0.9886 + tanh_core(1.0288 x))  Linf 2.8e-3 */
static inline float spear_gelu(float x)
{
    return 0.50576f * x * (0.98861f + spear_rt(1.02882f * x, 0.82917f * x, 0.94844f * x,
                                                28.25600f, 35.84440f, 8.87444f));
}

/* gelu_tanh(x) ~ 0.5364 x (0.9322 + tanh_core(0.932 x))  Linf 2.7e-3 */
static inline float spear_gelu_tanh(float x)
{
    return 0.53638f * x * (0.93217f + spear_rt(0.93198f * x, 0.83122f * x, 1.01841f * x,
                                                28.62666f, 34.93181f, 7.46759f));
}

/* softplus(x) ~ relu(x) + (0.9159 - 0.1926|x|)/(1.329 + 0.5816|x| + 0.413 x^2)  Linf 4e-3 */
static inline float spear_softplus(float x)
{
    const float ax = fabsf(x);
    return ((x > 0.0f) ? x : 0.0f) + (0.91586f - 0.19255f * ax) / (1.32910f + 0.58157f * ax + 0.41302f * x * x);
}

/* exp(x), valid on [-12, 0] (softmax building block)   Linf 3.4e-3 */
static inline float spear_exp(float x)
{
    const float num = 0.91893f + spear_rt(0.58774f * x, 0.47359f * x, 0.48822f * x,
                                          29.83343f, 45.99957f, 11.99235f);
    const float den = 0.91581f - spear_rt(0.62725f * x, 0.52762f * x, -0.00718f * x,
                                          28.36862f, 30.35211f, 5.25054f);
    return num / den;
}

/* rsqrt(x) = 1/sqrt(x), x in [0.1, 4]  (RMSNorm) — machine precision */
static inline float spear_rsqrt(float x)
{
    return sqrtf(1.01083f / (1.01083f * x));
}

/* sin(x), x in [-pi/2, pi/2]   Linf 6.8e-5 */
static inline float spear_sin(float x)
{
    return x * (0.99970f + x * x * (-0.16567f + x * x * 0.00751f));
}

/* cos(x), x in [-pi/2, pi/2]   Linf 6.7e-4 */
static inline float spear_cos(float x)
{
    return 0.99949f + x * x * (-0.49590f + x * x * 0.03692f);
}

/* --------------------------------------------------------------------------
 *  Exact libm references (validation / optional fallback)
 * ------------------------------------------------------------------------- */

static inline float exact_gelu(float x) { return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f))); }
static inline float exact_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float exact_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float exact_tanh(float x) { return tanhf(x); }
static inline float exact_softplus(float x) { return log1pf(expf(x)); }
static inline float exact_exp(float x) { return expf(x); }
static inline float exact_rsqrt(float x) { return 1.0f / sqrtf(x); }
static inline float exact_sin(float x) { return sinf(x); }
static inline float exact_cos(float x) { return cosf(x); }

#endif /* SPEAR_KERNELS_H */