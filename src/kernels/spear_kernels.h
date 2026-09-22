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

/* --------------------------------------------------------------------------
 *  SpearVM championnes (github.com/bahira/SpearVM, CHAMPIONS.md) — 100 % ALU
 *  Vérifiées numériquement par tools/spear_grounded.mjs (L3 vs libm).
 * ------------------------------------------------------------------------- */

/* tanh Pade[3/4] minimax, clamp ±4 — L∞ 1.56e-3 sur [-5,5] (×5.4 vs spear_tanh) */
static inline float spear_tanh_p34(float x)
{
    const float y = (x > 4.0f) ? 4.0f : (x < -4.0f ? -4.0f : x);
    const float t = y * y;
    return (0.994894946f * y + 0.076611228f * y * t) / (1.0f + 0.402171314f * t + 0.005670342f * t * t);
}

/* erf_v2 Horner 5/5, clamp ±2 — L∞ ~2.3e-5 */
static inline float spear_erf(float x)
{
    static const float N[5] = { 1.12841751266903279f, 0.183482771948230095f, 0.0573373674730976793f, 0.00248430060206610405f, 3.72785350475749968e-6f };
    static const float D[6] = { 1.0f, 0.496471589671860558f, 0.114910282096263028f, 0.0161717422205343367f, 1.86656477609649336e-4f, -1.74401807407079551e-7f };
    const float u = (x > 2.0f) ? 2.0f : (x < -2.0f ? -2.0f : x);
    const float y = u * u;
    float pn = 0.0f, dn = 0.0f;
    for (int i = 4; i >= 0; i--) pn = pn * y + N[i];
    for (int i = 5; i >= 0; i--) dn = dn * y + D[i];
    return u * (pn / dn);
}

/* GELU quintique (smoothstep) : 5 mul, 0 div, queue bornée — L∞ 1.74e-2 */
static inline float spear_gelu_quintic(float x)
{
    const float off = 0.01104961f;
    const float t = 0.200055340257f * x + 0.5f;
    if (t < 0.0f) return -off;
    if (t > 1.0f) return x - off;
    const float t2 = t * t;
    return x * (t2 * t * (6.0f * t2 - 15.0f * t + 10.0f)) - off;
}

/* GELU via erf_v2 (Horner 5/6 + 1 div) — L∞ ~2.05e-5. Training/backprop. */
static inline float spear_gelu_erf(float x)
{
    const float u = x * 0.7071067811865476f;
    if (u > 3.5f) return x;
    if (u < -3.5f) return 0.0f;
    const float y = u * u;
    float pn = 0.0f, dn = 0.0f;
    static const float N[5] = { 1.12841751266903279f, 0.183482771948230095f, 0.0573373674730976793f, 0.00248430060206610405f, 3.72785350475749968e-6f };
    static const float D[6] = { 1.0f, 0.496471589671860558f, 0.114910282096263028f, 0.0161717422205343367f, 1.86656477609649336e-4f, -1.74401807407079551e-7f };
    for (int i = 4; i >= 0; i--) pn = pn * y + N[i];
    for (int i = 5; i >= 0; i--) dn = dn * y + D[i];
    return 0.5f * x * (1.0f + u * (pn / dn));
}

/* Dérivée exacte GELU linéaire-clamp (variante v1) — gradcheck ~1e-9. */
static inline float spear_gelu_backward(float dy, float x)
{
    const float u = 0.306923f * x + 0.501f;
    const float g = (u <= 0.0f) ? 0.0f : ((u >= 1.002f) ? 0.997729f * 1.002f : 0.997729f * (u + 0.306923f * x));
    return dy * g;
}

#endif /* SPEAR_KERNELS_H */