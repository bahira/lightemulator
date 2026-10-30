/* lm_quant.h — quantification RÉELLE int8 / int4 : stockage compacté + déquant
 * dans la tuile GEMM (issue #17).
 *
 * Ce qui change par rapport à l'ancien `quantTensor()` : les poids ne sont plus
 * « arrondis puis remis en float32 » (même place mémoire, même bande passante,
 * seul le bruit changeait). Ici les matrices sont **packées** en int8 (1 octet
 * par poids) ou int4 (2 poids par octet) avec une échelle par colonne de sortie,
 * et le GEMM multiplie des entiers (VNNI `dpbusd` en AVX-512, `maddubs` en AVX2,
 * scalaire ailleurs) avant de déquantifier à l'accumulation.
 *
 * Disposition des poids int8 (identique AVX2 / AVX-512) :
 *   q[((k>>2) * N + n) * 4 + (k & 3)]
 * c'est-à-dire des groupes de 4 k contigus par colonne, 16 colonnes par bloc de
 * 64 octets : un seul chargement 512 bits sert 16 colonnes, et l'activation est
 * un `set1_epi32` des mêmes 4 octets. (AVX2 : 8 colonnes par bloc de 32 octets.)
 *
 * Granularité : une échelle par **colonne** (axe d'entrée complet, K ≤ 288) et
 * une échelle par **ligne** d'activation — pas de blocs de 32 : sur ces matrices
 * les distributions sont homogènes, et 8 640 échelles fp32 (34 kB) seraient 13 %
 * de la taille int8. C'est un choix explicite, pas un oubli.
 */
#ifndef LM_QUANT_H
#define LM_QUANT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint8_t *q;      /* int8 : q[(k>>2)*N*4 + n*4 + (k&3)]  ·  int4 : q[(k>>1)*N + n] (nibble bas = k pair) */
    float   *s;      /* [N] échelle de colonne : max|colonne| / 127 (int8) ou / 7 (int4) */
    int32_t *wsum;   /* [N] somme des poids quantifiés (correction du point zéro 64) */
    int K, N, bits;  /* bits = 8 ou 4 */
    uint8_t *q8;     /* cache : int4 étendu en int8 (même disposition) au 1er usage */
    size_t bytes;    /* octets réellement occupés par les poids + échelles */
} QMat;

static size_t qMatBytes(int K, int N, int bits) {
    const size_t w = (bits == 4) ? (size_t)N * ((size_t)K / 2) : (size_t)N * (size_t)K;
    return w + (size_t)N * (sizeof(float) + sizeof(int32_t));
}

/* Packe Wt[K][N] (transposé du modèle : Wt[k*N + n]) en int8/int4. */
static void qMatPack(const float *Wt, int K, int N, int bits, QMat *Q) {
    Q->K = K; Q->N = N; Q->bits = bits; Q->q8 = NULL;
    Q->bytes = qMatBytes(K, N, bits);
    Q->q = (uint8_t *)calloc(1, Q->bytes);
    Q->s = (float *)malloc((size_t)N * sizeof(float));
    Q->wsum = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    const int qmax = (bits == 4) ? 7 : 127;
    for (int n = 0; n < N; n++) {
        float mx = 0.f;
        for (int k = 0; k < K; k++) { const float a = fabsf(Wt[(size_t)k * N + n]); if (a > mx) mx = a; }
        const float s = (mx > 1e-12f) ? mx / (float)qmax : 1e-12f;
        Q->s[n] = s;
        long sum = 0;
        for (int k = 0; k < K; k++) {
            long v = lrintf(Wt[(size_t)k * N + n] / s);
            if (v > qmax) v = qmax;
            if (v < -qmax) v = -qmax;
            sum += v;
            if (bits == 8) {
                Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)] = (uint8_t)(int8_t)v;
            } else {
                uint8_t *b = &Q->q[(size_t)(k >> 1) * N + n];
                const uint8_t nib = (uint8_t)(v & 0x0F);
                if (k & 1) *b = (uint8_t)((*b & 0x0F) | (nib << 4));
                else       *b = (uint8_t)((*b & 0xF0) | nib);
            }
        }
        Q->wsum[n] = (int32_t)sum;
    }
}

static void qMatFree(QMat *Q) {
    if (!Q) return;
    free(Q->q); free(Q->s); free(Q->wsum); free(Q->q8);
    Q->q = NULL; Q->s = NULL; Q->wsum = NULL; Q->q8 = NULL;
}

/* Quantifie une ligne d'activation en u8 avec point zéro 64 :
 *   xq[k] = clamp(round(x[k] * 63 / max|x|) + 64, 0, 127)
 * d'où  Σ xq·w = Σ round(x/sx)·w + 64·Σw  → correction 64·Σw retirée de l'accumulation.
 *
 * Pourquoi 63 niveaux et pas 127 : `maddubs` (AVX2) somme DEUX produits u8·i8 dans
 * un int16 ; avec |xq| ≤ 255 et |w| ≤ 127 on dépasse 32767 (255·127·2 = 64770) et
 * l'accumulation déborde silencieusement. 127·127·2 = 32258 tient. VNNI accumule
 * en int32 et n'aurait pas ce problème, mais une seule quantification pour les
 * deux chemins vaut mieux qu'un résultat qui dépend du CPU. */
static float qActRow(const float *x, int K, uint8_t *xq) {
    float mx = 0.f;
    for (int k = 0; k < K; k++) { const float a = fabsf(x[k]); if (a > mx) mx = a; }
    const float s = (mx > 1e-12f) ? mx / 63.0f : 1e-12f;
    for (int k = 0; k < K; k++) {
        long v = lrintf(x[k] / s) + 64;
        if (v > 127) v = 127;
        if (v < 0) v = 0;
        xq[k] = (uint8_t)v;
    }
    return s; /* y = acc_int32 * s * scale_colonne */
}

/* Extrait un poids int4 signé depuis le stockage packé. */
static inline int qGet4(const QMat *Q, int k, int n) {
    const uint8_t b = Q->q[(size_t)(k >> 1) * Q->N + n];
    const int v = (k & 1) ? (b >> 4) : (b & 0x0F);
    return (v >= 8) ? v - 16 : v; /* complément à deux sur 4 bits */
}

/* ------------------------------------------------------- référence scalaire */
static void mm_nt_t_q_scalar(float *Y, const float *X, const QMat *Q, const float *b,
                             int M, int K, int N) {
    uint8_t *xq = (uint8_t *)malloc((size_t)K);
    for (int m = 0; m < M; m++) {
        const float sx = qActRow(X + (size_t)m * K, K, xq);
        float *y = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            long acc = -64L * (long)Q->wsum[n];
            if (Q->bits == 8) {
                for (int k = 0; k < K; k++)
                    acc += (long)xq[k] * (long)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)];
            } else {
                for (int k = 0; k < K; k++) acc += (long)xq[k] * (long)qGet4(Q, k, n);
            }
            y[n] = (float)acc * sx * Q->s[n] + (b ? b[n] : 0.f);
        }
    }
    free(xq);
}

/* ------------------------------------------------------- AVX2 : maddubs */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

/* 8 colonnes par itération : chaque lane int32 = somme de 4 produits (u8 x i8). */
__attribute__((target("avx2")))
static void mm_nt_t_q8_avx2(float *Y, const float *X, const QMat *Q, const float *b,
                            int M, int K, int N) {
    const int N8 = N & ~7;
    const __m256i ones = _mm256_set1_epi16(1);
    if (K & 3) { mm_nt_t_q_scalar(Y, X, Q, b, M, K, N); return; } /* groupes de 4 k */
    uint8_t *xq = (uint8_t *)malloc((size_t)K);
    for (int mb = 0; mb + 4 <= M; mb += 4) {
        uint8_t *xq4[4];
        float sx4[4];
        for (int r = 0; r < 4; r++) {
            xq4[r] = (uint8_t *)malloc((size_t)K);
            sx4[r] = qActRow(X + (size_t)(mb + r) * K, K, xq4[r]);
        }
        const int kg = K >> 2;
        for (int nb = 0; nb + 8 <= N8; nb += 8) {
            __m256i acc[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(),
                               _mm256_setzero_si256(), _mm256_setzero_si256() };
            for (int g = 0; g < kg; g++) {
                const __m256i w = _mm256_loadu_si256((const __m256i *)(Q->q + ((size_t)g * N + nb) * 4));
                for (int r = 0; r < 4; r++) {
                    int32_t a4;
                    memcpy(&a4, xq4[r] + 4 * g, 4);
                    const __m256i a = _mm256_set1_epi32(a4);
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones));
                }
            }
            const __m256 sw = _mm256_loadu_ps(Q->s + nb);
            const __m256 bb = b ? _mm256_loadu_ps(b + nb) : _mm256_setzero_ps();
            const __m256i corr = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(Q->wsum + nb)), 6);
            for (int r = 0; r < 4; r++) acc[r] = _mm256_sub_epi32(acc[r], corr);
            for (int r = 0; r < 4; r++) {
                __m256 v = _mm256_cvtepi32_ps(acc[r]);
                v = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(v, sw), _mm256_set1_ps(sx4[r])), bb);
                _mm256_storeu_ps(Y + (size_t)(mb + r) * N + nb, v);
            }
        }
        /* colonnes restantes N % 8 : scalaire (K % 4 est traité par le fallback) */
        for (int r = 0; r < 4; r++) {
            float *y = Y + (size_t)(mb + r) * N;
            for (int nb = N8; nb < N; nb++) {
                long acc = -64L * (long)Q->wsum[nb];
                for (int k = 0; k < K; k++)
                    acc += (long)xq4[r][k] * (long)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)nb * 4 + (k & 3)];
                y[nb] = (float)acc * sx4[r] * Q->s[nb] + (b ? b[nb] : 0.f);
            }
        }
        for (int r = 0; r < 4; r++) free(xq4[r]);
    }
    /* lignes M restantes : scalaire */
    for (int m = (M & ~3); m < M; m++) {
        const float sx = qActRow(X + (size_t)m * K, K, xq);
        float *y = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            long acc = -64L * (long)Q->wsum[n];
            for (int k = 0; k < K; k++)
                acc += (long)xq[k] * (long)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)];
            y[n] = (float)acc * sx * Q->s[n] + (b ? b[n] : 0.f);
        }
    }
    free(xq);
}

/* --------------------------------------------------- AVX-512 VNNI : dpbusd */
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void mm_nt_t_q8_vnni(float *Y, const float *X, const QMat *Q, const float *b,
                            int M, int K, int N) {
    const int N16 = N & ~15;
    if (K & 3) { mm_nt_t_q_scalar(Y, X, Q, b, M, K, N); return; }
    for (int mb = 0; mb + 4 <= M; mb += 4) {
        uint8_t *xq = (uint8_t *)malloc((size_t)4 * K);
        float sx[4];
        for (int r = 0; r < 4; r++) sx[r] = qActRow(X + (size_t)(mb + r) * K, K, xq + (size_t)r * K);
        const int kg = K >> 2;
        for (int nb = 0; nb + 16 <= N16; nb += 16) {
            /* deux accumulateurs par ligne : la chaîne dpbusd (latence ~5 cycles)
               est coupée en deux, sinon 4 chaînes ne suffisent pas à couvrir la
               latence et le noyau tourne à moitié de son débit. */
            __m512i acc0 = _mm512_setzero_si512(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
            __m512i bcc0 = acc0, bcc1 = acc0, bcc2 = acc0, bcc3 = acc0;
            int g = 0;
            for (; g + 2 <= kg; g += 2) {
                const __m512i wA = _mm512_loadu_si512((const void *)(Q->q + ((size_t)g * N + nb) * 4));
                const __m512i wB = _mm512_loadu_si512((const void *)(Q->q + ((size_t)(g + 1) * N + nb) * 4));
                int32_t a4;
                memcpy(&a4, xq + 4 * g, 4);
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(a4), wA);
                memcpy(&a4, xq + 4 * (g + 1), 4);
                bcc0 = _mm512_dpbusd_epi32(bcc0, _mm512_set1_epi32(a4), wB);
                memcpy(&a4, xq + (size_t)K + 4 * g, 4);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(a4), wA);
                memcpy(&a4, xq + (size_t)K + 4 * (g + 1), 4);
                bcc1 = _mm512_dpbusd_epi32(bcc1, _mm512_set1_epi32(a4), wB);
                memcpy(&a4, xq + (size_t)2 * K + 4 * g, 4);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(a4), wA);
                memcpy(&a4, xq + (size_t)2 * K + 4 * (g + 1), 4);
                bcc2 = _mm512_dpbusd_epi32(bcc2, _mm512_set1_epi32(a4), wB);
                memcpy(&a4, xq + (size_t)3 * K + 4 * g, 4);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(a4), wA);
                memcpy(&a4, xq + (size_t)3 * K + 4 * (g + 1), 4);
                bcc3 = _mm512_dpbusd_epi32(bcc3, _mm512_set1_epi32(a4), wB);
            }
            for (; g < kg; g++) {
                const __m512i w = _mm512_loadu_si512((const void *)(Q->q + ((size_t)g * N + nb) * 4));
                int32_t a4;
                memcpy(&a4, xq + 4 * g, 4);            acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(a4), w);
                memcpy(&a4, xq + (size_t)K + 4 * g, 4); acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(a4), w);
                memcpy(&a4, xq + (size_t)2 * K + 4 * g, 4); acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(a4), w);
                memcpy(&a4, xq + (size_t)3 * K + 4 * g, 4); acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(a4), w);
            }
            acc0 = _mm512_add_epi32(acc0, bcc0); acc1 = _mm512_add_epi32(acc1, bcc1);
            acc2 = _mm512_add_epi32(acc2, bcc2); acc3 = _mm512_add_epi32(acc3, bcc3);
            const __m512 sw = _mm512_loadu_ps(Q->s + nb);
            const __m512 bb = b ? _mm512_loadu_ps(b + nb) : _mm512_setzero_ps();
            const __m512i corr = _mm512_slli_epi32(_mm512_loadu_si512((const void *)(Q->wsum + nb)), 6);
            acc0 = _mm512_sub_epi32(acc0, corr); acc1 = _mm512_sub_epi32(acc1, corr);
            acc2 = _mm512_sub_epi32(acc2, corr); acc3 = _mm512_sub_epi32(acc3, corr);
            __m512 v0 = _mm512_cvtepi32_ps(acc0), v1 = _mm512_cvtepi32_ps(acc1);
            __m512 v2 = _mm512_cvtepi32_ps(acc2), v3 = _mm512_cvtepi32_ps(acc3);
            v0 = _mm512_fmadd_ps(_mm512_mul_ps(v0, sw), _mm512_set1_ps(sx[0]), bb);
            v1 = _mm512_fmadd_ps(_mm512_mul_ps(v1, sw), _mm512_set1_ps(sx[1]), bb);
            v2 = _mm512_fmadd_ps(_mm512_mul_ps(v2, sw), _mm512_set1_ps(sx[2]), bb);
            v3 = _mm512_fmadd_ps(_mm512_mul_ps(v3, sw), _mm512_set1_ps(sx[3]), bb);
            _mm512_storeu_ps(Y + (size_t)(mb + 0) * N + nb, v0);
            _mm512_storeu_ps(Y + (size_t)(mb + 1) * N + nb, v1);
            _mm512_storeu_ps(Y + (size_t)(mb + 2) * N + nb, v2);
            _mm512_storeu_ps(Y + (size_t)(mb + 3) * N + nb, v3);
        }
        /* reste : colonnes N%16 et k % 4, scalaire */
        for (int r = 0; r < 4; r++) {
            float *y = Y + (size_t)(mb + r) * N;
            const uint8_t *xr = xq + (size_t)r * K;
            for (int nb = N16; nb < N; nb++) {
                long acc = -64L * (long)Q->wsum[nb];
                for (int k = 0; k < K; k++)
                    acc += (long)xr[k] * (long)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)nb * 4 + (k & 3)];
                y[nb] = (float)acc * sx[r] * Q->s[nb] + (b ? b[nb] : 0.f);
            }
        }
        free(xq);
    }
    for (int m = (M & ~3); m < M; m++) {
        uint8_t *xq = (uint8_t *)malloc((size_t)K);
        const float sx = qActRow(X + (size_t)m * K, K, xq);
        float *y = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            long acc = -64L * (long)Q->wsum[n];
            for (int k = 0; k < K; k++)
                acc += (long)xq[k] * (long)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)];
            y[n] = (float)acc * sx * Q->s[n] + (b ? b[n] : 0.f);
        }
        free(xq);
    }
}
#endif /* x86 */


/* ------------------------------------- int8 packé en wasm SIMD128 (A8W8) ----
 * Le noyau scalaire coûtait ~9x le FP32 dans le navigateur. Ici on garde le
 * calcul vectoriel : par panneau de 16 colonnes, on DÉQUANTIFIE les poids une
 * fois (K x 16 flottants, réutilisés sur toutes les lignes M, donc amorti), les
 * activations sont déquantifiées une fois par ligne, et la boucle interne reste
 * le GEMM v128 à 4 lignes M. Les octets stockés restent ceux du pack int8 :
 * la déquantification vit dans une tuile temporaire, jamais dans une copie
 * persistante des poids.
 */
#if defined(__wasm_simd128__) && !defined(LM_NO_WASM_SIMD)
#include <wasm_simd128.h>
static void mm_nt_t_q8_wasm(float *Y, const float *X, const QMat *Q, const float *b,
                            int M, int K, int N) {
    const int N16 = N & ~15;
    uint8_t *xq = (uint8_t *)malloc((size_t)K);
    /* activations déquantifiées UNE fois par ligne (et non par panneau) */
    float *xa = (float *)malloc(sizeof(float) * (size_t)M * K);
    for (int m = 0; m < M; m++) {
        const float sx = qActRow(X + (size_t)m * K, K, xq);
        float *d = xa + (size_t)m * K;
        for (int k = 0; k < K; k++) d[k] = (float)((int)xq[k] - 64) * sx;
    }
    float *wp = (float *)malloc(sizeof(float) * 16 * (size_t)K);

    for (int nb = 0; nb + 16 <= N16; nb += 16) {
        /* déquantification du panneau de 16 colonnes, réutilisée sur toutes les lignes */
        for (int k = 0; k < K; k++) {
            const uint8_t *src = Q->q + ((size_t)(k >> 2) * N + nb) * 4;
            const int kk = k & 3;
            float *row = wp + (size_t)k * 16;
            for (int c = 0; c < 16; c++) row[c] = (float)(int8_t)src[c * 4 + kk] * Q->s[nb + c];
        }
        v128_t bb[4];
        for (int c = 0; c < 4; c++)
            bb[c] = b ? wasm_v128_load(b + nb + 4 * c) : wasm_f32x4_splat(0.0f);

        for (int mb = 0; mb + 4 <= M; mb += 4) {
            const float *xa0 = xa + (size_t)(mb + 0) * K, *xa1 = xa + (size_t)(mb + 1) * K;
            const float *xa2 = xa + (size_t)(mb + 2) * K, *xa3 = xa + (size_t)(mb + 3) * K;
            v128_t a[4][4];
            for (int c = 0; c < 4; c++) { a[0][c] = bb[c]; a[1][c] = bb[c]; a[2][c] = bb[c]; a[3][c] = bb[c]; }
            for (int k = 0; k < K; k++) {
                const float *w = wp + (size_t)k * 16;
                const v128_t w0 = wasm_v128_load(w), w1 = wasm_v128_load(w + 4);
                const v128_t w2 = wasm_v128_load(w + 8), w3 = wasm_v128_load(w + 12);
                const v128_t v0 = wasm_f32x4_splat(xa0[k]), v1 = wasm_f32x4_splat(xa1[k]);
                const v128_t v2 = wasm_f32x4_splat(xa2[k]), v3 = wasm_f32x4_splat(xa3[k]);
                a[0][0] = wasm_f32x4_add(a[0][0], wasm_f32x4_mul(v0, w0));
                a[0][1] = wasm_f32x4_add(a[0][1], wasm_f32x4_mul(v0, w1));
                a[0][2] = wasm_f32x4_add(a[0][2], wasm_f32x4_mul(v0, w2));
                a[0][3] = wasm_f32x4_add(a[0][3], wasm_f32x4_mul(v0, w3));
                a[1][0] = wasm_f32x4_add(a[1][0], wasm_f32x4_mul(v1, w0));
                a[1][1] = wasm_f32x4_add(a[1][1], wasm_f32x4_mul(v1, w1));
                a[1][2] = wasm_f32x4_add(a[1][2], wasm_f32x4_mul(v1, w2));
                a[1][3] = wasm_f32x4_add(a[1][3], wasm_f32x4_mul(v1, w3));
                a[2][0] = wasm_f32x4_add(a[2][0], wasm_f32x4_mul(v2, w0));
                a[2][1] = wasm_f32x4_add(a[2][1], wasm_f32x4_mul(v2, w1));
                a[2][2] = wasm_f32x4_add(a[2][2], wasm_f32x4_mul(v2, w2));
                a[2][3] = wasm_f32x4_add(a[2][3], wasm_f32x4_mul(v2, w3));
                a[3][0] = wasm_f32x4_add(a[3][0], wasm_f32x4_mul(v3, w0));
                a[3][1] = wasm_f32x4_add(a[3][1], wasm_f32x4_mul(v3, w1));
                a[3][2] = wasm_f32x4_add(a[3][2], wasm_f32x4_mul(v3, w2));
                a[3][3] = wasm_f32x4_add(a[3][3], wasm_f32x4_mul(v3, w3));
            }
            for (int r = 0; r < 4; r++) {
                float *y = Y + (size_t)(mb + r) * N + nb;
                wasm_v128_store(y, a[r][0]); wasm_v128_store(y + 4, a[r][1]);
                wasm_v128_store(y + 8, a[r][2]); wasm_v128_store(y + 12, a[r][3]);
            }
        }
        for (int m = (M & ~3); m < M; m++) { /* lignes M % 4 : scalaire */
            const float *xm = xa + (size_t)m * K;
            float *y = Y + (size_t)m * N;
            for (int n = nb; n < nb + 16; n++) {
                float acc = 0.f;
                for (int k = 0; k < K; k++)
                    acc += xm[k] * (float)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)] * Q->s[n];
                y[n] = acc + (b ? b[n] : 0.f);
            }
        }
    }
    /* palier 8 colonnes (N % 16 == 8 sur toutes les formes du modèle : 72, 104, 216) */
    for (int nb = N16; nb + 8 <= (N & ~7); nb += 8) {
        for (int k = 0; k < K; k++) {
            const uint8_t *src = Q->q + ((size_t)(k >> 2) * N + nb) * 4;
            const int kk = k & 3;
            float *row = wp + (size_t)k * 16;
            for (int c = 0; c < 8; c++) row[c] = (float)(int8_t)src[c * 4 + kk] * Q->s[nb + c];
        }
        v128_t bb0 = b ? wasm_v128_load(b + nb) : wasm_f32x4_splat(0.0f);
        v128_t bb1 = b ? wasm_v128_load(b + nb + 4) : wasm_f32x4_splat(0.0f);
        for (int mb = 0; mb + 4 <= M; mb += 4) {
            const float *xa0 = xa + (size_t)(mb + 0) * K, *xa1 = xa + (size_t)(mb + 1) * K;
            const float *xa2 = xa + (size_t)(mb + 2) * K, *xa3 = xa + (size_t)(mb + 3) * K;
            v128_t a0 = bb0, a1 = bb1, b0 = bb0, b1 = bb1, c0 = bb0, c1 = bb1, d0 = bb0, d1 = bb1;
            for (int k = 0; k < K; k++) {
                const float *w = wp + (size_t)k * 16;
                const v128_t w0 = wasm_v128_load(w), w1 = wasm_v128_load(w + 4);
                const v128_t v0 = wasm_f32x4_splat(xa0[k]), v1 = wasm_f32x4_splat(xa1[k]);
                const v128_t v2 = wasm_f32x4_splat(xa2[k]), v3 = wasm_f32x4_splat(xa3[k]);
                a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(v0, w0)); a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(v0, w1));
                b0 = wasm_f32x4_add(b0, wasm_f32x4_mul(v1, w0)); b1 = wasm_f32x4_add(b1, wasm_f32x4_mul(v1, w1));
                c0 = wasm_f32x4_add(c0, wasm_f32x4_mul(v2, w0)); c1 = wasm_f32x4_add(c1, wasm_f32x4_mul(v2, w1));
                d0 = wasm_f32x4_add(d0, wasm_f32x4_mul(v3, w0)); d1 = wasm_f32x4_add(d1, wasm_f32x4_mul(v3, w1));
            }
            float *y0 = Y + (size_t)(mb + 0) * N + nb, *y1 = Y + (size_t)(mb + 1) * N + nb;
            float *y2 = Y + (size_t)(mb + 2) * N + nb, *y3 = Y + (size_t)(mb + 3) * N + nb;
            wasm_v128_store(y0, a0); wasm_v128_store(y0 + 4, a1);
            wasm_v128_store(y1, b0); wasm_v128_store(y1 + 4, b1);
            wasm_v128_store(y2, c0); wasm_v128_store(y2 + 4, c1);
            wasm_v128_store(y3, d0); wasm_v128_store(y3 + 4, d1);
        }
    }
    /* colonnes N % 16 : scalaire */
    for (int m = 0; m < M; m++) {
        const float *xm = xa + (size_t)m * K;
        float *y = Y + (size_t)m * N;
        for (int n = (N & ~7); n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += xm[k] * (float)(int8_t)Q->q[(size_t)(k >> 2) * N * 4 + (size_t)n * 4 + (k & 3)] * Q->s[n];
            y[n] = acc + (b ? b[n] : 0.f);
        }
    }
    free(xq); free(xa); free(wp);
}
#endif /* __wasm_simd128__ */

/* int4 → int8 : expansion unique (mise en cache dans Q->q8). Le stockage reste
 * 4 bits par poids ; l'expansion est un dépaquetage one-shot, comme un moteur qui
 * garde les poids quantifiés sur disque et les déplie à la première évaluation. */
static void qMatEnsureInt8(const QMat *Q) {
    QMat *M = (QMat *)Q;
    if (Q->bits != 4 || M->q8) return;
    M->q8 = (uint8_t *)malloc((size_t)Q->K * Q->N);
    for (int k = 0; k < Q->K; k++)
        for (int n = 0; n < Q->N; n++)
            M->q8[(size_t)(k >> 2) * Q->N * 4 + (size_t)n * 4 + (k & 3)] = (uint8_t)(int8_t)qGet4(Q, k, n);
}

/* int4 : stockage 4 bits, expansion int8 one-shot puis noyau int8 */
static void mm_nt_t_q4_scalar(float *Y, const float *X, const QMat *Q, const float *b,
                              int M, int K, int N) {
    if (K & 1) { mm_nt_t_q_scalar(Y, X, Q, b, M, K, N); return; } /* K impair : lecture nibble directe */
    qMatEnsureInt8(Q);
    QMat V = *Q; V.bits = 8; V.q = Q->q8;
    mm_nt_t_q_scalar(Y, X, &V, b, M, K, N);
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void mm_nt_t_q4_avx2(float *Y, const float *X, const QMat *Q, const float *b,
                            int M, int K, int N) {
    if (K & 1) { mm_nt_t_q4_scalar(Y, X, Q, b, M, K, N); return; }
    qMatEnsureInt8(Q);                       /* dépaquetage one-shot, puis noyau int8 */
    QMat V = *Q; V.bits = 8; V.q = Q->q8; V.q8 = NULL;
    mm_nt_t_q8_avx2(Y, X, &V, b, M, K, N);
}

#endif

#if defined(__wasm_simd128__) && !defined(LM_NO_WASM_SIMD)
static void mm_nt_t_q4_wasm(float *Y, const float *X, const QMat *Q, const float *b,
                            int M, int K, int N) {
    if (K & 1) { mm_nt_t_q4_scalar(Y, X, Q, b, M, K, N); return; }
    qMatEnsureInt8(Q);
    QMat V = *Q; V.bits = 8; V.q = Q->q8; V.q8 = NULL;
    mm_nt_t_q8_wasm(Y, X, &V, b, M, K, N);
}
#endif

#endif /* LM_QUANT_H */
