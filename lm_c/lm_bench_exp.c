/* lm_bench_exp.c ??? verdict C : rationnel vs poly4 vs poly3 vs libm */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) { struct timespec ts; timespec_get(&ts, TIME_UTC); return (double)ts.tv_sec + 1e-9 * ts.tv_nsec; }

static inline float sp_pow2i(int k) { union { float f; uint32_t u; } v; v.f = 1.0f; v.u = (uint32_t)(127 + k) << 23; return v.f; }

/* rationnel actuel (lm_kernels.h) */
static inline float exp_rat(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float num = 1.0f + r * (0.49889506f + r * (0.09793305f + r * 0.01029060f));
    const float den = 1.0f + r * (-0.50150042f + r * 0.09859938f);
    return sp_pow2i(k) * (num / den);
}
/* poly4 sans division */
static inline float exp_p4(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float p = ((0.042356002f * r + 0.166933063f) * r + 0.499893348f) * r + 1.000043123f;
    return sp_pow2i(k) * (1.0f + r * p);
}
/* poly3 sans division */
static inline float exp_p3(float x) {
    const int k = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    const float r = x - (float)k * 0.6931471805599453f;
    const float p = (0.165147401f * r + 0.504127785f) * r + 1.000196294f;
    return sp_pow2i(k) * (1.0f + r * p);
}

#define N 40000000
int main(void) {
    static float args[N];
    for (int i = 0; i < N; i++) args[i] = -12.0f * ((float)((i * 2654435761u) & 0xFFFFFF) / 16777216.0f);
    volatile float sink = 0.0f;

    struct { const char *name; float (*fn)(float); } ks[] = {
        { "libm expf   ", expf }, { "rat  [3/2]  ", exp_rat },
        { "poly4 no-div", exp_p4 }, { "poly3 no-div", exp_p3 },
    };
    /* exactitude d'abord */
    for (int k = 0; k < 4; k++) {
        double worst = 0;
        for (int i = 0; i <= 100000; i++) {
            const float x = -14.0f + 16.0f * (float)i / 100000.0f;
            const float e = fabsf(ks[k].fn(x) - expf(x)) / expf(x);
            if ((double)e > worst) worst = e;
        }
        printf("  %s L??? rel = %.3e\n", ks[k].name, worst);
    }
    /* vitesse */
    double tRef = 0;
    for (int k = 0; k < 4; k++) {
        const double t0 = now_s();
        float s = 0;
        for (int i = 0; i < N; i++) s += ks[k].fn(args[i]);
        const double dt = now_s() - t0;
        sink += s;
        if (k == 0) tRef = dt;
        printf("  %s %7.1f ms  (%.2f ns/call)%s\n", ks[k].name, dt * 1e3, dt * 1e9 / N, k ? "" : "  ??? r??f??rence");
        if (k) printf("      ??? ??%.2f vs libm\n", tRef / dt);
    }
    printf("  (sink=%g)\n", sink);
    return 0;
}

