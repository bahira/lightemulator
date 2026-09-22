/* SPEAR QN03: Benchmark INT8 16x16 matmul vs FP32 (AVX2, -O3) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

static void matmul_fp32(const float *A, const float *B, float *C)
{
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 16; ++k) s += A[i*16+k] * B[k*16+j];
            C[i*16+j] = s;
        }
}

int main(void)
{
    static int8_t A[256], B[256];
    static int32_t C_i[256];
    static float Af[256], Bf[256], Cf[256];
    unsigned seed = 42;
    for (int i = 0; i < 256; ++i) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (int8_t)((seed >> 16) & 0xff) - 128;
        B[i] = (int8_t)((seed >> 24) & 0xff) - 128;
        Af[i] = (float)A[i]; Bf[i] = (float)B[i];
    }

    const int N = 2000000;
    clock_t t0 = clock();
    for (int i = 0; i < N; ++i) spear_int8_mul_16x16_avx2(A, B, C_i, 16);
    double t_int8 = (double)(clock() - t0) / CLOCKS_PER_SEC;

    t0 = clock();
    for (int i = 0; i < N; ++i) matmul_fp32(Af, Bf, Cf);
    double t_fp32 = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("INT8 16x16 : %.4fs (%d calls)\n", t_int8, N);
    printf("FP32 16x16 : %.4fs (%d calls)\n", t_fp32, N);
    printf("Speedup INT8 vs FP32 : %.2fx\n", t_fp32 / (t_int8 + 1e-12));

    /* Saturation : max |acc| pour confirmer pas de debordement INT32
     * valeurs [-128,127]^2 * 16 termes : max 128*128*16 = 262144, OK */
    int32_t mx = 0;
    for (int i = 0; i < 256; ++i) {
        int32_t a = C_i[i] < 0 ? -C_i[i] : C_i[i];
        if (a > mx) mx = a;
    }
    printf("Max |accumulateur INT32| : %d (limite 2^31-1 : OK)\n", mx);
    return 0;
}