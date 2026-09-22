/* probe_cpu.c — quelles ISA cette machine supporte-t-elle réellement ? */
#include <stdio.h>
int main(void) {
    printf("avx2   = %d\n", __builtin_cpu_supports("avx2"));
    printf("fma    = %d\n", __builtin_cpu_supports("fma"));
    printf("avx512f= %d\n", __builtin_cpu_supports("avx512f"));
    printf("avx512dq=%d\n", __builtin_cpu_supports("avx512dq"));
    printf("bmi2   = %d\n", __builtin_cpu_supports("bmi2"));
    return 0;
}
