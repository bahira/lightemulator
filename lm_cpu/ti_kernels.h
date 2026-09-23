/* ============================================================================
 * ti_kernels.h — SPEAR-T1 : noyau arithmétique ENTIER d'un transformer CPU.
 *
 * Contrat : le chemin d'inférence ne contient AUCUNE opération flottante.
 *   — pas de float, pas de double      (vérifié par désassemblage : --fpcheck)
 *   — pas de sqrt, pas de division     (Newton entier + décalages de bits)
 *   — pas de libm, pas d'exponentielle (exp2 par polynôme entier + exposant)
 *   — poids ternaires {-1,0,1}, activations int8, accumulateurs int32
 *
 * Tout est déterministe : les chemins scalaire et AVX2 produisent les mêmes
 * entiers (addition int32 exacte, aucun dépassement — bornes prouvées ci-dessous),
 * donc le résultat est reproductible au bit près, machine comprise.
 *
 * BORNES MESURÉES par lm_cpu/ti_test.c (aucune valeur « espérée ») :
 *   ti_rsqrt_q16 : erreur relative ≤ 1.1e-3 sur a ∈ [1, 16129] (résolution Q16)
 *                  → ≤ 0.2 pas int8 sur l'activation normalisée
 *   ti_rcp_norm  : erreur relative ≤ 8e-5 sur s ∈ [1, 2^23] (aucune division)
 *   ti_exp2_q15  : erreur absolue ≤ 1.2e-4 (Q15) ; relative ≤ 1e-3 sur t ≥ −8·ln2
 *   ti_dot_i8    : bit-exact vs référence scalaire (max|diff| = 0)
 * =========================================================================== */
#ifndef TI_KERNELS_H
#define TI_KERNELS_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ======================================================== bit primitives == */

/* floor(log2(x)), x >= 1 — 31 - clz (l'équivalent entier de sp_pow2i). */
static inline int ti_ilog2(uint32_t x) { return 31 - __builtin_clz(x | 1u); }

/* Décalage droit avec arrondi au plus proche ; sh < 0 ⇒ décalage gauche. */
static inline int64_t ti_shr_rnd(int64_t v, int sh) {
    if (sh <= 0) return (sh <= -62) ? 0 : (v << (-sh));
    if (sh >= 62) return v < 0 ? -1 : 0;
    return (v + ((int64_t)1 << (sh - 1))) >> sh;
}

/* Saturation int8 — le seul « clamp » du modèle (1 instruction : packss). */
static inline int32_t ti_clamp8(int32_t v) {
    return v < -127 ? -127 : (v > 127 ? 127 : v);
}

/* Mantisse normalisée : x = yq·2^d, yq ∈ [2^14, 2^15), arrondi au plus proche.
 * C'est la brique « champ d'exposant » du repo (sp_pow2i), appliquée à
 * l'intérieur de Newton au lieu de ldexp. */
static inline uint32_t ti_norm_mant(uint32_t x, int *d_out) {
    if (x == 0) { *d_out = 0; return 1u << 14; }
    const int e = ti_ilog2(x), d = e - 14;
    uint32_t yq;
    if (d > 0)      yq = (x + (1u << (d - 1))) >> d;
    else if (d == 0) yq = x;
    else             yq = x << (-d);
    if (yq < (1u << 14)) yq = 1u << 14;
    if (yq >= (1u << 15)) yq = (1u << 15) - 1;
    *d_out = d;
    return yq;
}

/* ============================================== Newton entier : rsqrt / rcp =
 * 2^20/√yq puis 1/yq, sans sqrt ni division, en Q28 (mantisse ∈ [2^14,2^15)).
 * Graines linéaires minimax (moindres carrés sur 200k points) — 3 itérations
 * suffisent pour saturer la précision du format.
 */
/* Les constantes négatives sont écrites avec une multiplication par 2^8 : un
 * décalage à gauche d'une valeur négative est un comportement indéfini en C
 * (-Wshift-negative-value), et la valeur obtenue est identique. */
#define TI_RS_A ((int64_t)10526313 * 256)  /* graine Q28 : (A + B·yq) >> 10 */
#define TI_RS_B ((int64_t)(-146) * 256)
#define TI_RC_A ((int64_t)92283 * 256)
#define TI_RC_B ((int64_t)(-2) * 256)

static inline int64_t ti_isqrt_q28(uint32_t yq) {
    int64_t r = (TI_RS_A + TI_RS_B * (int64_t)yq) >> 10;
    if (r < 1) r = 1;
    for (int i = 0; i < 3; i++) {
        const int64_t rr = (r * r) >> 28;                       /* u² en Q28    */
        const int64_t t  = (int64_t)yq * rr;                    /* yq·u²  (Q28) */
        r = (r * (((int64_t)3 << 28) - t) + ((int64_t)1 << 28)) >> 29;
        if (r < 1) r = 1;
    }
    return r;
}

static inline int64_t ti_rcp_q28(uint32_t yq) {
    int64_t r = (TI_RC_A + TI_RC_B * (int64_t)yq) >> 10;
    if (r < 1) r = 1;
    for (int i = 0; i < 4; i++) {
        const int64_t t = (int64_t)yq * r;                      /* yq·u   (Q28) */
        r = (r * (((int64_t)2 << 28) - t) + ((int64_t)1 << 27)) >> 28;
        if (r < 1) r = 1;
    }
    return r;
}

/* 2^16/√a, a ≥ 1 — Q16. Domaine utile : a = moyenne des carrés d'int8 ≤ 16129. */
#define TI_SQRT2_INV_Q16 46341
static inline int32_t ti_rsqrt_q16(uint32_t a) {
    if (a == 0) a = 1;
    int d;
    const uint32_t yq = ti_norm_mant(a, &d);
    int64_t U = ti_isqrt_q28(yq);                     /* U = 2^28/√yq */
    if (d & 1) U = (U * TI_SQRT2_INV_Q16) >> 16;      /* facteur 2^-1/2 (d impair) */
    const int sh = 12 + (d >> 1);                     /* 2^16/√a = U·2^{-12-d/2}   */
    if (sh < 0) return (int32_t)(U << (-sh > 30 ? 30 : -sh));
    return (int32_t)ti_shr_rnd(U, sh);
}

/* 1/s sous forme normalisée : 1/s ≈ R·2^{-k}, R ∈ [2^14, 2^15) — donc une
 * précision RELATIVE constante (≈2^-14) quelle que soit la magnitude de s.
 * C'est ce qui permet de normaliser une softmax sans division et sans perdre
 * les petits poids. */
static inline int32_t ti_rcp_norm(uint32_t s, int *k_out) {
    if (s == 0) s = 1;
    int d;
    const uint32_t yq = ti_norm_mant(s, &d);          /* s = yq·2^d */
    const int64_t U = ti_rcp_q28(yq);                 /* 2^28/yq ∈ [2^13, 2^14) */
    *k_out = 29 + d;                                  /* 1/s = (U<<1)·2^{-k}    */
    return (int32_t)(U << 1);
}

/* ======================================================= exp2 entier =====
 * 2^(t/4096) pour t ≤ 0, résultat en Q15 [0, 32768].
 * Polynôme degré 5 (minimax, coefficients en Q15) + décalage final qui consomme
 * la partie entière de l'exposant : aucun appel libm, aucune table.
 * C'est le remplaçant entier du spear_exp8 du repo (même idée : poly + champ
 * d'exposant), mais sans jamais passer par un flottant.
 */
#define TI_EX2_C5  64    /* 0.00195·2^15 */
#define TI_EX2_C4  296   /* 0.00903  */
#define TI_EX2_C3  1832  /* 0.05592  */
#define TI_EX2_C2  7872  /* 0.24025  */
#define TI_EX2_C1  22712 /* 0.69312  */

static inline int32_t ti_exp2_q15(int32_t t) {
    if (t >= 0) return 32768;                        /* exp(0) = 1 */
    const int32_t i = t >> 12;                       /* floor(t) ≤ 0 */
    if (i < -31) return 0;                           /* sous la résolution Q15 */
    const int32_t f = t & 4095;                      /* fraction ∈ [0, 4096)   */
    int64_t p = TI_EX2_C5;
    p = ((p * f) >> 12) + TI_EX2_C4;
    p = ((p * f) >> 12) + TI_EX2_C3;
    p = ((p * f) >> 12) + TI_EX2_C2;
    p = ((p * f) >> 12) + TI_EX2_C1;
    p = ((p * f) >> 12) + 32768;                     /* + 2^0 en Q15 */
    const int sh = -i;
    return (int32_t)((p + ((int64_t)1 << (sh - 1))) >> sh);  /* arrondi */
}

/* =========================================================== quantizers ==
 * Utilisés à l'entraînement / à la calibration (chemin flottant, JAMAIS dans
 * l'inférence). Quantification symétrique par position pour les activations.
 */
static inline float ti_quant_row_i8(const float *x, int8_t *q, int n, float static_scale) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) { const float a = x[i] < 0.0f ? -x[i] : x[i]; if (a > mx) mx = a; }
    if (mx < 1e-8f) mx = 1e-8f;
    const float s = (static_scale > 0.0f) ? static_scale : mx / 127.0f;
    const float inv = 1.0f / s;
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)lrintf(x[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    return s;
}

/* ==================================================== kernel int8 × int8 ==
 * Y[m][n] = Σ_k X[m][k]·W[n][k] — W rangée par canal de sortie (layout
 * « N-major ») : parcours séquentiel de W, idéal pour le GEMV (décodage
 * autorégressif, M = 1) comme pour le préfixage (M = B·T).
 *
 * Le produit int8×int8 n'existe pas en AVX2 : il est reconstruit par
 *   |x|·signe(x)·w  →  pabsb + psignb  puis  vpmaddubsw + vpmaddwd
 * soit 32 produits accumulés par paire d'instructions, exactement.
 * Bornes anti-dépassement : |acc| ≤ 127²·K ≤ 6.6e7 pour K ≤ 4096 (< 2^31).
 */
#if defined(__AVX2__)
__attribute__((target("avx2")))
static inline int32_t ti_hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

__attribute__((target("avx2")))
static inline int32_t ti_dot_i8_avx2(const int8_t *__restrict x, const int8_t *__restrict w, int K) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int k = 0;
    for (; k + 32 <= K; k += 32) {
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + k));
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(w + k));
        const __m256i xa = _mm256_abs_epi8(xv);
        const __m256i ws = _mm256_sign_epi8(wv, xv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(xa, ws), ones));
    }
    int32_t s = ti_hsum256_epi32(acc);
    for (; k < K; k++) s += (int32_t)x[k] * (int32_t)w[k];
    return s;
}

/* GEMM 4 sorties en parallèle : X est lu une fois pour 4 canaux (le
 * multi-accumulateur du repo, porté en int8). */
__attribute__((target("avx2")))
static inline void ti_gemm_i8_avx2(int32_t *__restrict Y, const int8_t *__restrict X,
                                   const int8_t *__restrict W, int M, int K, int N) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (int n = 0; n < N; n += 4) {
        const int nc = (n + 4 <= N) ? 4 : (N - n);
        for (int m = 0; m < M; m++) {
            const int8_t *x = X + (size_t)m * K;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            const int8_t *w0 = W + (size_t)(n + 0) * K;
            const int8_t *w1 = (nc > 1) ? W + (size_t)(n + 1) * K : w0;
            const int8_t *w2 = (nc > 2) ? W + (size_t)(n + 2) * K : w0;
            const int8_t *w3 = (nc > 3) ? W + (size_t)(n + 3) * K : w0;
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + k));
                const __m256i xa = _mm256_abs_epi8(xv);
                const __m256i p0 = _mm256_maddubs_epi16(xa, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w0 + k)), xv));
                const __m256i p1 = _mm256_maddubs_epi16(xa, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w1 + k)), xv));
                const __m256i p2 = _mm256_maddubs_epi16(xa, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w2 + k)), xv));
                const __m256i p3 = _mm256_maddubs_epi16(xa, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(w3 + k)), xv));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p0, ones));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(p1, ones));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(p2, ones));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(p3, ones));
            }
            int32_t s0 = ti_hsum256_epi32(a0), s1 = ti_hsum256_epi32(a1);
            int32_t s2 = ti_hsum256_epi32(a2), s3 = ti_hsum256_epi32(a3);
            for (; k < K; k++) {
                const int32_t xv = x[k];
                s0 += xv * (int32_t)w0[k]; if (nc > 1) s1 += xv * (int32_t)w1[k];
                if (nc > 2) s2 += xv * (int32_t)w2[k]; if (nc > 3) s3 += xv * (int32_t)w3[k];
            }
            Y[(size_t)m * N + n] = s0;
            if (nc > 1) Y[(size_t)m * N + n + 1] = s1;
            if (nc > 2) Y[(size_t)m * N + n + 2] = s2;
            if (nc > 3) Y[(size_t)m * N + n + 3] = s3;
        }
    }
}
#endif /* __AVX2__ */

static inline int32_t ti_dot_i8_ref(const int8_t *x, const int8_t *w, int K) {
    int32_t s = 0;
    for (int k = 0; k < K; k++) s += (int32_t)x[k] * (int32_t)w[k];
    return s;
}

static inline void ti_gemm_i8_ref(int32_t *__restrict Y, const int8_t *__restrict X,
                                  const int8_t *__restrict W, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            Y[(size_t)m * N + n] = ti_dot_i8_ref(X + (size_t)m * K, W + (size_t)n * K, K);
}

/* ============================================ kernels RMSNorm / softmax == */

/* RMSNorm entier : out[i] = clamp(( x[i]·g[i]·(2^16/√mean(x²)) ) requantifié).
 *   mean(x²) : accumulateur int32 (D ≤ 4096, 127²·D ≤ 6.6e7)
 *   inv_rms  : Newton entier, Q16      → aucun sqrt, aucune division
 *   (Mq, shq) : multiplicateur STATIQUE calibré (voir ti_model.h) — c'est lui
 *               qui fixe l'échelle int8 commune du flux résiduel.
 * Produits : |x·g| ≤ 2^14, ×inv ≤ 2^16  → ≤ 2^30 (int32), ×Mq ≤ 2^15 → ≤ 2^45
 * ⇒ dernier produit en int64 (chemin chaud : 1 mul par élément, hors GEMM).
 */
static inline int32_t ti_rmsnorm_i8(const int8_t *__restrict x, const int8_t *__restrict g,
                                    int8_t *__restrict out, int n, int32_t inv_n_q16,
                                    int32_t Mq, int shq, int32_t *msq_out, int32_t *maxp) {
    int32_t s2 = 0;
    for (int i = 0; i < n; i++) s2 += (int32_t)x[i] * (int32_t)x[i];
    /* moyenne des carrés SANS division : n⁻¹ en Q16, fourni par l'export.
     * Borne : s2 ≤ 127²·n ⇒ s2·inv_n ≤ 127²·2^16 = 1.06e9 < 2^31. */
    const int32_t ms = (int32_t)(((int64_t)s2 * inv_n_q16 + (1 << 15)) >> 16);
    if (msq_out) *msq_out = ms;
    const int32_t inv = ti_rsqrt_q16((uint32_t)(ms < 1 ? 1 : ms));
    for (int i = 0; i < n; i++) {
        const int32_t t = (int32_t)x[i] * (int32_t)g[i];        /* ≤ 16129 */
        const int32_t v = (int32_t)(((int64_t)t * inv) >> 16);  /* normalisé */
        const int32_t o = (int32_t)ti_shr_rnd((int64_t)v * Mq, shq);   /* AVANT écrêtage */
        /* Le maximum est relevé AVANT l'écrêtage : mesurer la valeur écrêtée
         * rendrait la calibration aveugle (le maximum resterait collé à 127 et
         * l'échelle croîtrait indéfiniment). Bornes : |v| ≤ 16 129, Mq ≤ 2^14
         * ⇒ |o| ≤ 2.6e8 < 2^31 : l'entier ne déborde pas. */
        if (maxp) { const int32_t a = o < 0 ? -o : o; if (a > *maxp) *maxp = a; }
        out[i] = (int8_t)ti_clamp8(o);
    }
    return ms;
}

/* Softmax entier causal d'une ligne de scores int32 → poids int8 ∈ [0,127].
 *   p_j = 2^{(s_j−max)·log2e} (Q15, polynôme entier) ; Σp ; 1/Σ par Newton
 *   normalisé ; att8 = (p·(127·R)) >> (k−10) — précision relative 2^-14 sur le
 *   poids, soit ≪ 1 pas int8.
 * scratch : buffer de n int32 fourni par l'appelant (aucune allocation).
 * Retourne Σ des poids int8 (diagnostic : idéalement ≈ 127).
 */
static inline int32_t ti_softmax_i8_row(const int32_t *__restrict s, int n,
                                        int8_t *__restrict w8, int32_t *__restrict scratch) {
    int32_t mx = s[0];
    for (int i = 1; i < n; i++) if (s[i] > mx) mx = s[i];
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        const int32_t d = s[i] - mx;                        /* ≤ 0 */
        /* argument de 2^x en Q12 : x = d·log2(e) ⇒ t = d·5909 (log2e·2^12).
         * Le seuil porte sur d (échelle ln) : d < −250000 est inatteignable en
         * pratique (e^−250000 = 0), c'est un garde-fou contre un accumulateur
         * absurde. Dans ce cas le résultat exact de la limite est 0 ; la version
         * antérieure écrivait −1, soit 2^(−1/4096) ≈ 1,0 — l'erreur n'était pas
         * observable (branche jamais prise en 1 000 pas d'entraînement), elle est
         * corrigée par cohérence, pas parce qu'elle expliquait un écart mesuré. */
        const int32_t p = (d < -250000) ? 0 : ti_exp2_q15((int32_t)((int64_t)d * 5909));
        scratch[i] = p;
        sum += p;
    }
    if (sum < 1) sum = 1;
    int k = 0;
    const int32_t R127 = (int32_t)(((int64_t)127 * ti_rcp_norm((uint32_t)sum, &k)) >> 10);
    const int sh = k - 10;
    int32_t sumw = 0;
    for (int i = 0; i < n; i++) {
        int32_t q = (int32_t)(((int64_t)scratch[i] * R127 + ((int64_t)1 << (sh - 1))) >> sh);
        if (q > 127) q = 127; else if (q < 0) q = 0;
        w8[i] = (int8_t)q;
        sumw += q;
    }
    return sumw;
}

/* ====================================================== packing ternaire ==
 * 2 bits / poids : 0 → 0, 1 → +1, 2 → −1 (3 → 0). Le fichier porte 2 bits ;
 * la RAM porte la forme déployée int8 (mesuré : ti_main --pack).
 */
static inline void ti_pack2(const int8_t *w, uint8_t *out, int n) {
    for (int i = 0; i < n; i += 4) {
        uint8_t b = 0;
        for (int j = 0; j < 4 && i + j < n; j++) {
            const int8_t v = w[i + j];
            const uint8_t c = (uint8_t)(v > 0 ? 1 : (v < 0 ? 2 : 0));
            b = (uint8_t)(b | (uint8_t)(c << (2 * j)));
        }
        out[i >> 2] = b;
    }
}
static inline void ti_unpack2(const uint8_t *in, int8_t *w, int n) {
    for (int i = 0; i < n; i++) {
        const uint8_t c = (uint8_t)((in[i >> 2] >> (2 * (i & 3))) & 3u);
        w[i] = (int8_t)(c == 1 ? 1 : (c == 2 ? -1 : 0));
    }
}

/* ================================================== dispatch scalaire/AVX2 */
static inline int32_t ti_dot_i8(const int8_t *x, const int8_t *w, int K) {
#if defined(__AVX2__) && !defined(TI_FORCE_SCALAR)
    return ti_dot_i8_avx2(x, w, K);
#else
    return ti_dot_i8_ref(x, w, K);
#endif
}
static inline void ti_gemm_i8(int32_t *__restrict Y, const int8_t *__restrict X,
                              const int8_t *__restrict W, int M, int K, int N) {
#if defined(__AVX2__) && !defined(TI_FORCE_SCALAR)
    ti_gemm_i8_avx2(Y, X, W, M, K, N);
#else
    ti_gemm_i8_ref(Y, X, W, M, K, N);
#endif
}

#endif /* TI_KERNELS_H */
