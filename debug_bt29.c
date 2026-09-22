#include <immintrin.h>
#include <stdio.h>
#include <math.h>

#include "src/spear_breakthrough29_matrix.c"

inline void ref_matrix_mul_4x4(const float *A, const float *B, float *C) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += A[i*4 + k] * B[k*4 + j];
            C[i*4 + j] = s;
        }
}

int main() {
    float A[16], B[16], C_bt29[16], C_ref[16];
    
    for (int i = 0; i < 16; i++) {
        A[i] = (float)(i % 4 + 1);
        B[i] = (float)(i % 4 + 2);
    }
    
    spear_matrix_mul_4x4_avx2(A, B, C_bt29);
    ref_matrix_mul_4x4(A, B, C_ref);
    
    printf("BT29 result C:\n");
    for (int i = 0; i < 16; i++) printf("  C[%d] = %f\n", i, C_bt29[i]);
    printf("\nReference result C:\n");
    for (int i = 0; i < 16; i++) printf("  C_ref[%d] = %f\n", i, C_ref[i]);
    
    printf("\nMax diff: %e\n", fabs(C_bt29[0] - C_ref[0]));
    for (int i = 0; i < 16; i++) {
        float diff = fabs(C_bt29[i] - C_ref[i]);
        if (diff > 0.001) printf("  Large diff at %d: %e\n", i, diff);
    }
    
    return 0;
}