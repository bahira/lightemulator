/* verify_kernels.c — autopsie des kernels SPEAR sous -ffast-math */
#define main lm_disabled_main
#include "lm_main.c"
#undef main

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("== VERIFICATION kernels (mêmes flags que lm_train.exe) ==\n");

    /* exp : grille dense [-14, +2] (au-delà du domaine nominal) */
    {
        double worst = 0; float wx = 0;
        for (int i = 0; i <= 200000; i++) {
            const float x = -14.0f + 16.0f * (float)i / 200000.0f;
            const float a = spear_expf(x), r = expf(x);
            const float e = fabsf(a - r) / r;
            if ((double)e > worst) { worst = e; wx = x; }
        }
        printf("  exp   : L∞ rel = %.3e @ x=%.3f %s\n", worst, wx, worst < 1e-3f ? "OK" : "** FAUX **");
        /* valeurs extrêmes du softmax : grands écarts négatifs */
        for (float xe = -30.0f; xe <= -20.0f; xe += 2.5f)
            printf("    exp(%.1f) : spear=%.6g libm=%.6g\n", xe, spear_expf(xe), expf(xe));
    }

    /* rsqrt : plage large incl. petits/var grands */
    {
        double worst = 0; float wx = 0;
        for (int i = 0; i <= 200000; i++) {
            const float x = (float)pow(10.0, -6.0 + 12.0 * (double)i / 200000.0);
            const float a = spear_rsqrtf(x), r = 1.0f / sqrtf(x);
            const float e = fabsf(a - r) / r;
            if ((double)e > worst) { worst = e; wx = x; }
        }
        printf("  rsqrt : L∞ rel = %.3e @ x=%.4g %s\n", worst, wx, worst < 5e-3f ? "OK" : "** FAUX **");
    }

    /* gelu : valeur + dérivée (FD en double, h adapté) */
    {
        double worstV = 0, worstD = 0;
        for (int i = 0; i <= 20000; i++) {
            const float x = -8.0f + 16.0f * (float)i / 20000.0f;
            const sp_gelu_vd gd = spear_gelu_vd(x);
            /* référence gelu_tanh exacte */
            const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
            const float refv = 0.5f * x * (1.0f + tanhf(u));
            const float eV = fabsf(gd.v - refv);
            if ((double)eV > worstV) worstV = eV;
            /* dérivée par FD double précision sur la formule de réf */
            const double hd = 1e-4;
            const double uu = 0.7978845608028654 * ((double)x + 0.044715 * (double)x * x * x);
            const double fpos = 0.5 * (double)x * (1.0 + tanh(uu + 0.7978845608028654 * (0.044715 * 3.0 * (double)x * x) * hd * 0.0)); /* placeholder évité ci-dessous */
            (void)fpos;
            /* FD simple sur refGelu (float) */
            extern float refGelu(float);
            const float num = (refGelu(x + 1e-2f) - refGelu(x - 1e-2f)) / 2e-2f;
            const float eD = fabsf(gd.d - num) / (fabsf(num) + 1e-3f);
            if ((double)eD > worstD && fabsf(num) > 0.05f) { worstD = eD; }
        }
        printf("  gelu  : écart valeur L∞ = %.3e %s · écart dérivée (vs FD, h=1e-2) rel = %.3f %s\n",
               worstV, worstV < 5e-3f ? "OK" : "** FAUX **", worstD, worstD < 0.2f ? "OK" : "** SUSPECT **");
    }
    return 0;
}
