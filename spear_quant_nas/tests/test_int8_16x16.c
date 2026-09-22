/* SPEAR QN02: INT8 16x16 randomized verification (tiled kernel vs reference) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline void spear_int8_mul_4x4(
    const int8_t *__restrict__ A, size_t lda,
    const int8_t *__restrict__ B, size_t ldb,
    int32_t *__restrict__ C, size_t ldc)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 4; ++k) s += (int32_t)A[i*lda+k] * (int32_t)B[k*ldb+j];
            C[i*ldc+j] = s;
        }
}

static inline void spear_int8_mul_16x16_avx2(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, size_t ldc)
{
    for (int i0 = 0; i0 < 16; i0 += 4)
        for (int j0 = 0; j0 < 16; j0 += 4) {
            int32_t acc[16] = {0};
            for (int k0 = 0; k0 < 16; k0 += 4) {
                int32_t tile[16];
                spear_int8_mul_4x4(&A[i0*16+k0], 16, &B[k0*16+j0], 16, tile, 4);
                for (int t = 0; t < 16; ++t) acc[t] += tile[t];
            }
            for (int t = 0; t < 16; ++t)
                C[i0*ldc + j0 + (t%4) + (t/4)*ldc] = acc[t];
        }
}

static void spear_int8_mul_16x16_ref(
    const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 16; ++k) s += (int32_t)A[i*16+k] * (int32_t)B[k*16+j];
            C[i*16+j] = s;
        }
}

int main(void)
{
    int8_t A[256], B[256];
    int32_t C_tile[256], C_ref[256];
    unsigned seed = 12345;
    for (int i = 0; i < 256; ++i) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (int8_t)((seed >> 16) & 0xff) - 100;  /* full INT8 range */
        B[i] = (int8_t)((seed >> 24) & 0xff) - 60;
    }

    spear_int8_mul_16x16_ref(A, B, C_ref);
    spear_int8_mul_16x16_avx2(A, B, C_tile, 16);

    int mismatches = 0;
    for (int i = 0; i < 256; ++i)
        if (C_tile[i] != C_ref[i]) {
            if (mismatches < 5)
                printf("Mismatch C[%d]: tile=%d ref=%d\n", i, C_tile[i], C_ref[i]);
            mismatches++;
        }

    printf("16x16 randomized: %d/256 mismatches -> %s\n",
           mismatches, mismatches ? "FAIL" : "PASS");
    return mismatches ? 1 : 0;
}