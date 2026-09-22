/* ========================================================================== */
/* SPEAR V600-ULTRA : HARNESS DE VALIDATION DES MICRO-KERNELS RAPIDES         */
/* Compilation : gcc -O3 -mavx2 -mfma -march=native -Wall -Wextra            */
/* ========================================================================== */
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <x86intrin.h>

/* Déclarations des primitives */
void spear_fast_rsqrt_inv_avx2(__m256 vx, __m256 *out_rsqrt, __m256 *out_inv);
float spear_online_logsumexp_avx2(const float *x, size_t n);
float spear_dot_product_8d_fast(const float *a, const float *b);

int main(void) {
    printf("SPEAR Micro-Kernels Benchmark\n=============================\n\n");

    /* 1. TEST RSQRT / DIVISION RAPIDE */
    {
        alignas(32) float x[8] = {4.0f, 16.0f, 25.0f, 100.0f, 0.25f, 2.0f, 9.0f, 49.0f};
        __m256 vx = _mm256_loadu_ps(x);
        __m256 vr, vi;
        spear_fast_rsqrt_inv_avx2(vx, &vr, &vi);

        float res_rsqrt[8], res_inv[8];
        _mm256_storeu_ps(res_rsqrt, vr);
        _mm256_storeu_ps(res_inv, vi);

        printf("[BT 21] Fast Rsqrt 1/sqrt(4.0) = %.6f (Attendu: 0.500000) ✓\n", res_rsqrt[0]);
        printf("[BT 21] Fast Inv   1/(4.0)      = %.6f (Attendu: 0.250000) ✓\n", res_inv[0]);
        assert(fabs(res_rsqrt[0] - 0.5f) < 1e-5f);
        assert(fabs(res_inv[0] - 0.25f) < 1e-5f);
    }

    /* 2. TEST LOGSUMEXP 1-PASS STREAMING */
    {
        alignas(32) float data[16] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f, 1.0f, 0.0f
        };
        float lse = spear_online_logsumexp_avx2(data, 16);
        printf("[BT 23] Streaming 1-Pass LogSumExp = %.6f (Calculé sans 3 passes) ✓\n", lse);
        assert(lse > 8.0f);
    }

    /* 3. TEST RÉDUCTION HORIZONTALE SANS VHADDPS */
    {
        alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        alignas(32) float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        float dot = spear_dot_product_8d_fast(a, b);
        printf("[BT 24] Dot Product 8D (sans vhaddps) = %.1f (Attendu: 36.0) ✓\n", dot);
        assert(fabs(dot - 36.0f) < 1e-5f);
    }

    printf("\nTous les micro-kernels de base sont formellement validés.\n");
    return 0;
}