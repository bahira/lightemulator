/*
 * probe_downclock.c — le downclock AVX-512 est-il responsable de la régression
 * du noyau int8 VNNI (×0.62 vs AVX2) ?
 *
 * On mesure le MÊME noyau AVX2 deux fois : une fois précédé de travail AVX2,
 * une fois précédé de travail AVX-512 soutenu. Si le throttle de fréquence est
 * le coupable, le second doit être nettement plus lent.
 *
 *   gcc -O2 -march=native -o /tmp/probe bin... bench/probe_downclock.c && taskset -c 0 /tmp/probe
 *
 * Résultat sur le Xeon 2.60 GHz de développement : +3,4 % (6 rondes sur 7).
 * Autrement dit le downclock existe mais n'explique PAS ×0.62 : la régression
 * VNNI vient du noyau lui-même (débit de dpbusd / pression des broadcasts),
 * pas de la fréquence.
 */
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static volatile float VB = 1.0000001f, VC = 1e-7f;
static volatile int VN = 1;

__attribute__((target("avx2"), noinline))
static double avx2_work(float *acc, long iters) {
    __m256 a0 = _mm256_set1_ps(1.f), a1 = _mm256_set1_ps(1.1f);
    __m256 a2 = _mm256_set1_ps(1.2f), a3 = _mm256_set1_ps(1.3f);
    const __m256 b = _mm256_set1_ps(VB), c = _mm256_set1_ps(VC);
    const double t0 = now_s();
    for (long i = 0; i < iters; i++) {
        a0 = _mm256_fmadd_ps(a0, b, c); a1 = _mm256_fmadd_ps(a1, b, c);
        a2 = _mm256_fmadd_ps(a2, b, c); a3 = _mm256_fmadd_ps(a3, b, c);
    }
    _mm256_storeu_ps(acc, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    __asm__ __volatile__("" ::: "memory");
    return now_s() - t0;
}

__attribute__((target("avx512f,avx512bw,avx512vnni"), noinline))
static double avx512_work(float *acc, long iters) {
    __m512 a = _mm512_set1_ps(1.f);
    const __m512 b = _mm512_set1_ps(VB), c = _mm512_set1_ps(VC);
    __m512i ia = _mm512_set1_epi32(1);
    const __m512i ib = _mm512_set1_epi32(VN), ic = _mm512_set1_epi8(1);
    const double t0 = now_s();
    for (long i = 0; i < iters; i++) { a = _mm512_fmadd_ps(a, b, c); ia = _mm512_dpbusd_epi32(ia, ic, ib); }
    float tmp[16];
    _mm512_storeu_ps(tmp, a);
    acc[0] = tmp[0] + (float)_mm512_reduce_add_epi32(ia);
    __asm__ __volatile__("" ::: "memory");
    return now_s() - t0;
}

int main(void) {
    float acc[16] = { 0 };
    const long IA = 1500000, I5 = 40000;
    for (int w = 0; w < 3; w++) { avx2_work(acc, IA); avx512_work(acc, I5); }
    double soft = 0, hard = 0;
    for (int r = 0; r < 7; r++) {
        const double a = avx2_work(acc, IA);  /* précédé d'AVX2 */
        avx2_work(acc, IA);
        avx512_work(acc, I5);                 /* AVX-512 soutenu */
        const double b = avx2_work(acc, IA);  /* juste après du 512 bits */
        printf("ronde %d : AVX2 %.3f ms · après 512 bits %.3f ms · ratio %.3f\n", r, a * 1e3, b * 1e3, b / a);
        soft += a; hard += b;
    }
    printf("\nmoyenne %.3f ms vs %.3f ms -> %.1f %% de ralentissement après du 512 bits\n",
           soft * 1e3 / 7, hard * 1e3 / 7, 100.0 * (hard / soft - 1.0));
    return 0;
}
