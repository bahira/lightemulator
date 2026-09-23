/* ============================================================================
 * ti_probe.c — autopsie d'un checkpoint : où l'information est-elle perdue ?
 *
 * Pour un même checkpoint, mesure trois pertes de validation successives :
 *   1. fp32 pur            : les maîtres flottants tels quels
 *   2. fp32 à poids ternaires effectifs (s_w·w8) : ce que la ternarisation
 *      enlève, sans toucher aux activations
 *   3. moteur entier       : ce que la quantification int8 des activations
 *      enlève en plus, après calibration
 *
 * Si (1) est bonne et (3) mauvaise → la quantification est le problème.
 * Si (1) est déjà mauvaise           → l'optimisation est le problème.
 * Cette distinction évite de corriger le mauvais étage.
 * ========================================================================== */
#include "ti_model.h"

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ti_probe: %s introuvable\n", path); exit(1); }
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)ti_fa((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "lecture\n"); exit(1); }
    fclose(f); *len = n;
    return b;
}
static void upd_gain(TiModel *m) {
    const int L = m->L, D = m->D;
    for (int l = 0; l < L; l++) {
        float mx1 = 1e-9f, mx2 = 1e-9f;
        const float *g1 = ti_p(m, P_G1, (long)l * D), *g2 = ti_p(m, P_G2, (long)l * D);
        for (int i = 0; i < D; i++) {
            const float a = fabsf(g1[i]); if (a > mx1) mx1 = a;
            const float b = fabsf(g2[i]); if (b > mx2) mx2 = b;
        }
        m->S_g1n[l] = mx1 / 127.0f; m->S_g2n[l] = mx2 / 127.0f;
    }
    float mx = 1e-9f;
    const float *gf = ti_p(m, P_GF, 0);
    for (int i = 0; i < D; i++) { const float a = fabsf(gf[i]); if (a > mx) mx = a; }
    m->S_gfn[0] = mx / 127.0f;
}
static double val_loss_mode(TiModel *m, const unsigned char *data, long len, int nbatch, int eff) {
    const uint64_t seed = 4242;
    uint64_t r = seed;
    double tot = 0.0;
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m->ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m->ntok);
    const int qat_save = m->qat;
    for (int i = 0; i < nbatch; i++) {
        const int T = m->T, B = m->B;
        const long maxpos = len - T - 2;
        for (int b = 0; b < B; b++) {
            const long pos = (long)(r % (uint64_t)maxpos);
            for (int t = 0; t < T; t++) { ids[b * T + t] = data[pos + t]; tgt[b * T + t] = data[pos + t + 1]; }
        }
        if (eff == 2) {            /* moteur entier : c'est bien ti_fwd_qat */
            m->qat = 1;
            ti_fwd_qat(m, ids, T);
        } else {
            m->qat = 0;
            ti_fwd_fp32(m, ids, T, eff);
        }
        tot += ti_loss_grad(m, tgt, T);
    }
    m->qat = qat_save;
    free(ids); free(tgt);
    return tot / nbatch;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage : ti_probe <ckpt.tif> [--nbatch N]\n"); return 1; }
    int nb = 8;
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], "--nbatch")) nb = atoi(argv[i + 1]);
    long len;
    unsigned char *data = slurp("../data/shakespeare.txt", &len);
    long vlen = 0;
    const long hold = 111539;
    unsigned char *vdata = (unsigned char *)ti_fa((size_t)hold);
    memcpy(vdata, data + len - hold, (size_t)hold);
    vlen = hold;
    const long dlen = len - hold;
    TiModel m;
    ti_model_init(&m, TI_B, 7);
    const int r = ti_model_load(&m, argv[1]);
    printf("== autopsie de %s (code %d, %ld pas) ==\n", argv[1], r, m.step);
    upd_gain(&m);
    ti_export(&m);
    printf("  perte val fp32 pur                  : %.4f\n", val_loss_mode(&m, vdata, vlen, nb, 0));
    printf("  perte val fp32 à poids TERNARISÉS   : %.4f\n", val_loss_mode(&m, vdata, vlen, nb, 1));
    ti_calibrate(&m, vdata, vlen, nb, 1);
    m.qat = 1;
    ti_export(&m);
    printf("  perte val MOTEUR ENTIER (calibré)   : %.4f\n", val_loss_mode(&m, vdata, vlen, nb, 2));
    /* densité ternaire réelle (la fonction exige les quatre pointeurs) */
    long tot = 0, nz = 0, plus = 0, minus = 0;
    ti_int_ternary_stats(&m.e, &tot, &nz, &plus, &minus);
    printf("  poids ternaires non nuls            : %ld/%ld (%.1f%%)\n", nz, tot, 100.0 * (double)nz / (double)tot);
    fflush(stdout);
    ti_model_free(&m);
    free(data); free(vdata);
    return 0;
}
