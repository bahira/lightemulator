/* ========================================================================== */
/* SPEAR QN01: INT8 Matrix Multiply AVX2 Micro-Kernel */
/* Quantized inference foundation: 8-bit integer matrix multiply               */
/* Tile size: 4x4 INT8, accumulation in INT32, scalable to LLM attention      */
/* ========================================================================== */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

/** INT8 micro-kernel AVX2 : 4x4 tile C = A*B, INT8*INT8 -> INT32 exact
 *  Strategie : sign-extend i8->i16 (cvtepi8_epi16), puis madd_epi16
 *  (2 produits + somme par lane i32) + hadd pour totaliser les 4 k.
 *  Pas de saturation (accumulation i32), exact a tous les ordres. */
static inline void spear_int8_mul_4x4_avx2(
    const int8_t *__restrict__ A, size_t lda,
    const int8_t *__restrict__ B, size_t ldb,
    int32_t *__restrict__ C, size_t ldc)
{
    for (int i = 0; i < 4; ++i) {
        /* A row i : 4 bytes -> i16, puis repeter [a0,a1,a2,a3] 4x */
        __m128i a16_128 = _mm_cvtepi8_epi16(
            _mm_loadl_epi64((const __m128i *)&A[i*lda]));
        __m128i rep_128 = _mm_shuffle_epi8(a16_128,
            _mm_setr_epi8(0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7));
        __m256i a_rep = _mm256_broadcastsi128_si256(rep_128);

        /* B : 4 colonnes x 4 k, packees [c0..c3] en colonnes */
        int8_t bp[16];
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                bp[j*4 + k] = B[(size_t)k*ldb + j];
        __m128i b8 = _mm_loadu_si128((const __m128i *)bp);
        __m256i b16 = _mm256_cvtepi8_epi16(b8);              /* 16 x i16 */

        __m256i prod = _mm256_madd_epi16(a_rep, b16);        /* 8 x i32 */
        __m128i lo = _mm256_castsi256_si128(prod);
        __m128i hi = _mm256_extracti128_si256(prod, 1);
        __m128i dots = _mm_hadd_epi32(lo, hi);               /* [d0..d3] */
        _mm_storeu_si128((__m128i *)&C[i*ldc], dots);
    }
}

/** INT8 16x16 matrix multiply via 4x4 tile superposition (accumulating) */
static inline void spear_int8_mul_16x16_avx2(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, size_t ldc)
{
    for (int i0 = 0; i0 < 16; i0 += 4) {
        for (int j0 = 0; j0 < 16; j0 += 4) {
            int32_t acc[16] = {0};
            for (int k0 = 0; k0 < 16; k0 += 4) {
                int32_t tile[16];
                spear_int8_mul_4x4_avx2(&A[i0*16 + k0], 16,
                                        &B[k0*16 + j0], 16, tile, 4);
                for (int t = 0; t < 16; ++t) acc[t] += tile[t];
            }
            for (int t = 0; t < 16; ++t)
                C[i0*ldc + j0 + (t%4) + (t/4)*ldc] = acc[t];
        }
    }
}

/** Reference INT8 matrix multiply 4x4 (scalar, for validation) */
static inline void spear_int8_mul_4x4_ref(
    const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 4; ++k) {
                s += (int32_t)A[i*4 + k] * (int32_t)B[k*4 + j];
            }
            C[i*4 + j] = s;
        }
    }
}

/** Print INT32 matrix (for validation) */
void spear_int8_print_matrix(const int32_t *M, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i) {
        printf("[");
        for (size_t j = 0; j < cols; ++j) {
            printf("%8d", M[i*cols + j]);
            if (j < cols - 1) printf(" ");
        }
        printf("]\n");
    }
}

/** Minimal test: 4x4 INT8 multiply - identity */
int main(void)
{
    /* Test: identity-like 4x4 INT8 multiply */
    int8_t A[16] = {1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1};

    int8_t B[16] = {1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1};

    int32_t C[16] = {0};

    /* Reference */
    spear_int8_mul_4x4_ref(A, B, C);
    printf("Reference (scalar):\n");
    spear_int8_print_matrix(C, 4, 4);

    /* INT8 kernel */
    for (int i = 0; i < 16; ++i) C[i] = 0;
    spear_int8_mul_4x4_avx2(A, 4, B, 4, C, 4);
    printf("\nINT8 AVX2 kernel:\n");
    spear_int8_print_matrix(C, 4, 4);

    /* Verify */
    int pass = 1;
    for (int i = 0; i < 16; ++i) {
        if (C[i] != (int32_t)B[i]) {
            pass = 0;
            printf("Mismatch at C[%d]: got %d, expected %d\n", i, C[i], (int32_t)B[i]);
        }
    }
    printf("\nCorrectness: %s\n", pass ? "PASS \u2705" : "FAIL \u274c");

    return pass ? 0 : 1;
}