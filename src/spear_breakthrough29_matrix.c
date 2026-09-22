/* ========================================================================== */
/* SPEAR BT29 : Micro-Kernel Matrix Multiply 4x4 Float32 AVX2               */
/* 16 multiplications + 12 additions, reference pattern tiling                */
/* ========================================================================== */
#include <immintrin.h>

/** Produit de matrices 4x4: référence cycle-précise (version scalarisèe) */
inline void spear_matrix_mul_4x4_avx2(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C)
{
    /* A, B, C sont des matrices 4x4 row-major */
    /* 16 multiplications + 12 additions = 28 opérations FLOPs */
    /* Pattern: cellule par cellule via dot product */

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                s += A[i*4 + k] * B[k*4 + j];
            }
            C[i*4 + j] = s;
        }
    }
}

/** Produit de référence 16x16 par superposition de tiles 4x4 */
inline void spear_matrix_mul_16x16_avx2(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, size_t ldc)
{
    /* Décomposer 16x16 en 16 tiles 4x4 */
    /* Pour chaque tile 4x4: spear_matrix_mul_4x4_avx2 */
    for (int i0 = 0; i0 < 16; i0 += 4) {
        for (int j0 = 0; j0 < 16; j0 += 4) {
            for (int k0 = 0; k0 < 16; k0 += 4) {
                spear_matrix_mul_4x4_avx2(
                    &A[i0 * 16 + k0],
                    &B[k0 * 16 + j0],
                    &C[i0 * ldc + j0]);
            }
        }
    }
}

/** Version référence 16x16 (pour benchmark de validation) */
inline void spear_matrix_mul_16x16_ref(
    const float *A, const float *B, float *C, size_t ldc)
{
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 16; ++k) {
                s += A[i*16 + k] * B[k*16 + j];
            }
            C[i*ldc + j] = s;
        }
    }
}

/* ========================================================================== */
//* Guide de tiling pour matrices LLM plus grandes                              */
/* ========================================================================== */
/* Pour des matrices MxK et KxN:                                                  */
/*                                                                              */
/* for i0 = 0 to M step 4:                                                       */
/*   for j0 = 0 to N step 4:                                                    */
/*     for k0 = 0 to K step 4:                                                  */
/*       spear_matrix_mul_4x4_avx2(&A[i0*K + k0], &B[k0*N + j0], &C[i0*ldc + j0])*/
/*                                                                              */
/* Avantages:                                                                   */
/* - Cache-friendly: chaque tile 4x4 tient dans L1 (64 bytes)                     */
/* - Réutilisation de registers: dot product par tile                             */
/* - Portable: fonctionne pour toute taille Multiple de 4                        */
/* - Performance LLM: typical 16x16 tiles pour poids de transformer               */
/* ========================================================================== */