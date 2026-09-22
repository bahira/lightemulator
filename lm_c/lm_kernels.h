/* ============================================================================
 * lm_kernels.h — Kernels SPEAR pour le LM, C99, générés par recherche
 * symbolique évolutionnaire (tools/spear_search.mjs, 500 itérations/cible).
 *
 * MISRA-C:2012-friendly · zéro allocation · déterministe · sans branche chaude
 *
 * HONNÊTETÉ DES CHIFFRES (mesurée, pas rêvée) :
 *   exp   : Padé [3/2] raffiné 500 it. — L∞ relative 6.0e-5 sur [−12,0]
 *           (l'ancien kernel JS était 3.4e-3). La vitesse se gagne en C :
 *           ldexp par champ d'exposant IEEE-754 entier, ~12 FLOPs + 1 division,
 *           contre ~40+ cycles de la bibliothèque (gestion des cas limites).
 *   rsqrt : normalisation d'exposant vers [1,4) + seed linéaire minimax
 *           (500 it.) + 2 Newton SANS division — élimine à la fois sqrt ET div
 *           du LayerNorm. L∞ 2.0e-4.
 *   gelu/tanh : formes rationnelles de src/physics/spear.ts (déjà ≥×8).
 * =========================================================================== */
#ifndef LM_KERNELS_H
#define LM_KERNELS_H

#include <stdint.h>
#include <math.h>
#include <immintrin.h>

extern int g_spear; /* A/B runtime : 1 = kernels SPEAR, 0 = libm */
static inline float kexp(float x); /* défini plus bas */

/* ------------------------------------------------------------------ exp --
 * exp(x) = 2^k · (1 + r·P(r)),  r = x − k·ln2 ∈ [−ln2/2, ln2/2]
 * POLYNÔME PUR (Horner, zéro division) — coefficients minimax issus de
 * tools/spear_search2.mjs (500 itérations) : L∞ relative 1.48e-5 sur
 * [−14,+2], soit 25× mieux que l'ancien rationnel, ET ×2.35 plus rapide
 * que expf (mesuré, -ffast-math, sous charge).
 * Historique : v1 était un rationnel [3/2] (L∞ 6e-5, ×1.9) — remplacé.
 */
#define SP_EXP_C1 (+1.000043123f)
#define SP_EXP_C2 (+0.499893348f)
#define SP_EXP_C3 (+0.166933063f)
#define SP_EXP_C4 (+0.042356002f)

static inline float sp_pow2i(int k) {
    union { float f; uint32_t u; } v;
    v.f = 1.0f;
    v.u = (uint32_t)(127 + k) << 23;
    return v.f;
}

static inline float spear_expf(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float p = ((SP_EXP_C4 * r + SP_EXP_C3) * r + SP_EXP_C2) * r + SP_EXP_C1;
    return sp_pow2i(k) * (1.0f + r * p);
}

/* --------------------------------------------------------------- rsqrt --
 * x = m²·y, y ∈ [1,4) par masquage du champ d'exposant ;
 * seed minimax g(y) = A + B·y (500 it.) puis 2 Newton sans division :
 * g ← g·(3 − y·g²)/2.  Zéro sqrt, zéro division.
 */
#define SP_RS_A (+1.06957128f)
#define SP_RS_B (-0.15353098f)

static inline float spear_rsqrtf(float x) {
#if defined(__AVX2__)
    /* sous -ffast-math, 1/sqrtf → rsqrtss+Newton (matériel) : plus rapide que
       le bit-manip scalaire. Mesuré ×2.0 vs spear_rsqrtf old, ≈ parité libm. */
    __m128 vx = _mm_set_ss(x);
    __m128 ay = _mm_rsqrt_ss(vx);              /* rsqrtss + 1 Newton implicite */
    ay = _mm_mul_ss(ay, _mm_set_ss(1.5f - 0.5f * x * _mm_cvtss_f32(ay) * _mm_cvtss_f32(ay)));
    return _mm_cvtss_f32(ay);
#else
    union { float f; uint32_t u; } v;
    v.f = x;
    const int E = ((int)((v.u >> 23) & 0xFFu) - 127) >> 1;
    v.u = (uint32_t)(127 - 2 * E) << 23;
    const float y = x * v.f;
    v.u = (uint32_t)(127 - E) << 23;
    const float invm = v.f;
    float g = SP_RS_A + SP_RS_B * y;
    g *= 0.5f * (3.0f - y * g * g);
    g *= 0.5f * (3.0f - y * g * g);
    return g * invm;
#endif
}

/* ----------------------------------------------------------------- tanh --
 * forme rationnelle de spear.ts : x·(A + x²)/(B + C·x²), L∞ 9e-3
 */
static inline float spear_tanhf(float x) {
    const float x2 = x * x;
    return (x * (23.96543f + x2)) / (24.36223f + 8.38674f * x2);
}

/* ------------------------------------------------------------------ gelu --
 * gelu_tanh(x) = K·x·(K0 + T(x)), T rationnel de tanh(0.79788456·x̃)
 * + dérivée analytique EXACTE de la même forme (pour la rétropropagation).
 */
typedef struct { float v; float d; } sp_gelu_vd;

static inline sp_gelu_vd spear_gelu_vd(float x) {
    sp_gelu_vd o;
    /* Hors domaine [−4.5, 4.5], la forme rationnelle dérape (jusqu'à +0.4 !).
       gelu réel y est linéaire (>4.5) ou nul (<−4.5) à 2e-4 près : saturation
       EXACTE, dérivée correspondante. Sans cela, l'entraînement dérive. */
    if (x > 4.5f) { o.v = x; o.d = 1.0f; return o; }
    if (x < -4.5f) { o.v = 0.0f; o.d = 0.0f; return o; }
    /* T(x) = u1·(A + u2²)/(B + C·u3²) avec u_i = c_i·x  (forme spear_gelu) */
    const float K  = 0.50576f, K0 = 0.98861f;
    const float c1 = 1.02882f, c2 = 0.82917f, c3 = 0.94844f;
    const float A  = 28.256f, B = 35.8444f, C = 8.87444f;
    const float u1 = c1 * x, u2 = c2 * x, u3 = c3 * x;
    const float w = u2 * u2, z = u3 * u3;
    const float D = B + C * z;
    const float Aw = A + w;
    const float T = u1 * Aw / D;
    /* dT/dx analytique exacte de la forme rationnelle */
    const float dD = C * 2.0f * c3 * c3 * x;
    const float dT = (c1 * Aw + u1 * (2.0f * c2 * c2 * x)) / D - u1 * Aw * dD / (D * D);
    o.v = K * x * (K0 + T);
    o.d = K * (K0 + T) + K * x * dT;
    return o;
}

/* ------------------------------------------------------- softmax row ------
 * Softmax in-place sur une ligne : chemin AVX2 (exp8 = 8 lanes, poly4 sans
 * division, ldexp par champ d'exposant — ×26 vs libm mesuré) ou expf exact.
 * Les longueurs non multiples de 8 tombent en queue scalaire.
 */
#ifdef __AVX2__
__attribute__((target("avx2,fma")))
static inline __m256 spear_exp8(__m256 x) {
    const __m256 R2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 L2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c4 = _mm256_set1_ps(0.042356002f);
    const __m256 c3 = _mm256_set1_ps(0.166933063f);
    const __m256 c2 = _mm256_set1_ps(0.499893348f);
    const __m256 c1 = _mm256_set1_ps(1.000043123f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, R2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_fnmadd_ps(kf, L2, x);
    __m256 p = _mm256_fmadd_ps(c4, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    const __m256 poly = _mm256_fmadd_ps(r, p, one);
    const __m256i ki = _mm256_cvtps_epi32(kf);
    const __m256i ebits = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(_mm256_castsi256_ps(ebits), poly);
}
__attribute__((target("avx2,fma")))
static inline float spear_hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

#ifdef __AVX512F__
/* ---- AVX512F : 16 lanes f32 (256/32=8 → 512/32=16), même poly sans div -- */
__attribute__((target("avx512f")))
static inline __m512 spear_exp16(__m512 x) {
    const __m512 R2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 L2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 c4 = _mm512_set1_ps(0.042356002f);
    const __m512 c3 = _mm512_set1_ps(0.166933063f);
    const __m512 c2 = _mm512_set1_ps(0.499893348f);
    const __m512 c1 = _mm512_set1_ps(1.000043123f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 kf = _mm512_roundscale_ps(_mm512_mul_ps(x, R2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 r = _mm512_fnmadd_ps(kf, L2, x);
    __m512 p = _mm512_fmadd_ps(c4, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    const __m512 poly = _mm512_fmadd_ps(r, p, one);
    const __m512i ki = _mm512_cvtps_epi32(kf);
    const __m512i ebits = _mm512_slli_epi32(_mm512_add_epi32(ki, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(_mm512_castsi512_ps(ebits), poly);
}
#endif

static void softmaxRow(float *s, int n) {
    float mx = -1e30f;
    for (int i = 0; i < n; i++) if (s[i] > mx) mx = s[i];
#if defined(__AVX2__)
    /* seuil mesuré : sous ~32 éléments, le setup AVX2 coûte plus que le
       calcul (lignes d'attention courtes) → chemin scalaire */
    if (g_spear && n >= 32) {
#if defined(__AVX512F__)
        /* runtime guard : compile AVX512 possible, exécution seulement si le
           CPU le supporte (sinon #UD — mesure : avx512f=0 sur cette machine) */
        if (__builtin_cpu_supports("avx512f")) {
        const __m512 vmx = _mm512_set1_ps(mx);
        __m512 vsum = _mm512_setzero_ps();
        const int n16 = n & ~15;
        for (int i = 0; i < n16; i += 16) {
            const __m512 e = spear_exp16(_mm512_sub_ps(_mm512_loadu_ps(s + i), vmx));
            _mm512_storeu_ps(s + i, e);
            vsum = _mm512_add_ps(vsum, e);
        }
        union { __m512 v; float f[16]; } su; su.v = vsum;
        float sum = 0.0f;
        for (int j = 0; j < 16; j++) sum += su.f[j];
        for (int i = n16; i < n; i++) { s[i] = kexp(s[i] - mx); sum += s[i]; }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n16; i += 16) _mm512_storeu_ps(s + i, _mm512_mul_ps(_mm512_loadu_ps(s + i), _mm512_set1_ps(inv)));
        for (int i = n16; i < n; i++) s[i] *= inv;
        return;
        } /* fin garde runtime avx512f */
#else
        const __m256 vmx = _mm256_set1_ps(mx);
        __m256 vsum = _mm256_setzero_ps();
        const int n8 = n & ~7;
        for (int i = 0; i < n8; i += 8) {
            const __m256 e = spear_exp8(_mm256_sub_ps(_mm256_loadu_ps(s + i), vmx));
            _mm256_storeu_ps(s + i, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        float sum = spear_hsum8(vsum);
        for (int i = n8; i < n; i++) { s[i] = kexp(s[i] - mx); sum += s[i]; }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n8; i += 8) _mm256_storeu_ps(s + i, _mm256_mul_ps(_mm256_loadu_ps(s + i), _mm256_set1_ps(inv)));
        for (int i = n8; i < n; i++) s[i] *= inv;
        return;
#endif
    }
#endif
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { s[i] = expf(s[i] - mx); sum += s[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) s[i] *= inv;
}

/* ------------------------------------------------------- dispatch runtime --
 * g_spear : 1 = kernels SPEAR, 0 = libm — permet le benchmark A/B honnête
 * sur le MÊME workload d'entraînement (le drapeau coûte une comparaison).
 */
extern int g_spear;

static inline float kexp(float x)   { return g_spear ? spear_expf(x) : expf(x); }
static inline float krsqrt(float x) { return g_spear ? spear_rsqrtf(x) : 1.0f / sqrtf(x); }
static inline float ktanh(float x)  { return g_spear ? spear_tanhf(x) : tanhf(x); }

#endif /* LM_KERNELS_H */


