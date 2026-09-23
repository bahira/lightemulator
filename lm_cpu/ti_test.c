/* ============================================================================
 * ti_test.c — Vérifications SPEAR-T1 : bornes des kernels, parité des chemins,
 * déterminisme. Chaque ligne imprimée est un chiffre MESURÉ.
 *
 *   gcc -O2 -march=native -ffast-math -o ti_test ti_test.c -lm
 *   ./ti_test --kernels        bornes exp2/rsqrt/rcp/dot/gemm/norm/softmax/pack
 *   ./ti_test --engine         moteur entier : parité préfixage/décodage,
 *                              déterminisme, comptage de MACs
 *   ./ti_test --dump fichier.ti tokens.txt   logits texte (référence Python)
 * =========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ti_infer.h"

static int g_fail = 0;
static void chk(const char *name, int ok, const char *detail) {
    printf("  %-34s %s   %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) g_fail++;
}

/* ------------------------------------------------------------ 1. kernels -- */
static void test_kernels(void) {
    printf("== bornes des kernels entiers (mesurées, pas espérées) ==\n");
    char buf[256];

    /* exp2 entier vs 2^(t/4096) exact */
    double maxabs = 0, maxrel = 0; int worst_t = 0;
    for (int t = -12 * 4096; t <= 0; t++) {
        const double got = ti_exp2_q15(t) / 32768.0;
        const double want = pow(2.0, t / 4096.0);
        const double a = fabs(got - want);
        if (a > maxabs) maxabs = a;
        if (want > 1e-2) { const double r = a / want; if (r > maxrel) { maxrel = r; worst_t = t; } }
    }
    snprintf(buf, sizeof buf, "abs ≤ %.2e  rel ≤ %.2e (t=%d, poids>1%%)", maxabs, maxrel, worst_t);
    chk("ti_exp2_q15 vs 2^x", maxabs < 2e-4, buf);

    /* rsqrt entier Q16 */
    double wr = 0; uint32_t wa = 0;
    for (uint32_t a = 1; a <= 16129; a++) {
        const double got = ti_rsqrt_q16(a), want = 65536.0 / sqrt((double)a);
        const double r = fabs(got - want) / want;
        if (r > wr) { wr = r; wa = a; }
    }
    snprintf(buf, sizeof buf, "rel ≤ %.3e (a=%u, domaine int8)", wr, wa);
    chk("ti_rsqrt_q16 (Newton, 0 sqrt)", wr < 1.5e-3, buf);

    /* réciproque normalisée */
    double wc = 0; uint32_t ws = 0;
    for (uint32_t s = 1; s <= (1u << 23); s += (1u << 22) / 4096 + 1) {
        int k = 0;
        const int32_t R = ti_rcp_norm(s, &k);
        const double got = (double)R * pow(2.0, -k), want = 1.0 / (double)s;
        const double r = fabs(got - want) / want;
        if (r > wc) { wc = r; ws = s; }
    }
    snprintf(buf, sizeof buf, "rel ≤ %.3e (s=%u, 0 division)", wc, ws);
    chk("ti_rcp_norm (Newton, 0 division)", wc < 5e-4, buf);

    /* dot / GEMM : AVX2 vs scalaire, doivent être IDENTIQUES */
    int8_t *x = (int8_t *)malloc(4096), *w = (int8_t *)malloc(4096 * 64);
    uint64_t rs = 1234567;
    for (int i = 0; i < 4096; i++) x[i] = (int8_t)(int)((rs = rs * 6364136223846793005ull + 1) >> 56);
    for (int i = 0; i < 4096 * 64; i++) w[i] = (int8_t)(((int)((rs = rs * 6364136223846793005ull + 1) >> 59) & 3) - 1);
    int32_t maxdot = 0;
    for (int K = 1; K <= 512; K = K * 2 + 1) {
        const int32_t a = ti_dot_i8_ref(x, w, K);
#if defined(__AVX2__)
        const int32_t b = ti_dot_i8_avx2(x, w, K);
        if (abs(a - b) > maxdot) maxdot = abs(a - b);
#endif
    }
    snprintf(buf, sizeof buf, "max|AVX2 − scalaire| = %d (K ∈ {1..511})", maxdot);
    chk("ti_dot_i8 bit-exact", maxdot == 0, buf);

    const int M = 5, K = 192, N = 96;
    int32_t *Y1 = (int32_t *)malloc(sizeof(int32_t) * M * N), *Y2 = (int32_t *)malloc(sizeof(int32_t) * M * N);
    ti_gemm_i8_ref(Y1, x, w, M, K, N);
#if defined(__AVX2__)
    ti_gemm_i8_avx2(Y2, x, w, M, K, N);
#else
    ti_gemm_i8_ref(Y2, x, w, M, K, N);
#endif
    int maxg = 0;
    for (int i = 0; i < M * N; i++) if (abs(Y1[i] - Y2[i]) > maxg) maxg = abs(Y1[i] - Y2[i]);
    snprintf(buf, sizeof buf, "max|AVX2 − scalaire| = %d (%dx%dx%d)", maxg, M, K, N);
    chk("ti_gemm_i8 bit-exact", maxg == 0, buf);

    /* RMSNorm : écart au calcul réel, en pas int8 */
    double wnorm = 0; int8_t g8[TI_D], xn[TI_D], out[TI_D];
    for (int i = 0; i < TI_D; i++) { g8[i] = (int8_t)(40 + (i % 20)); }
    double s2 = 0;
    for (int i = 0; i < TI_D; i++) { xn[i] = (int8_t)(((i * 37) % 240) - 120); s2 += (double)xn[i] * xn[i]; }
    int32_t msq = 0, mxpre = 0;
    const int32_t Mq = 1000, shq = 12;
    ti_rmsnorm_i8(xn, g8, out, TI_D, (int32_t)(((int64_t)1 << 16) / TI_D), Mq, shq, &msq, &mxpre);
    const double rms = sqrt(s2 / TI_D);
    for (int i = 0; i < TI_D; i++) {
        const double want = (double)xn[i] * g8[i] / rms * (double)Mq / (double)(1 << shq);
        const double wcl = want > 127 ? 127 : (want < -127 ? -127 : want);
        const double e = fabs(out[i] - wcl);
        if (e > wnorm) wnorm = e;
    }
    snprintf(buf, sizeof buf, "écart max = %.2f pas int8 (RMS=%.2f)", wnorm, rms);
    chk("ti_rmsnorm_i8 vs réel", wnorm < 1.5, buf);
    {
        /* Le maximum rapporté doit être celui d'AVANT écrêtage : sans cela la
         * calibration d'échelle ne peut jamais détecter un dépassement de
         * dynamique (elle resterait collée à 127). On force ici un gain 8× plus
         * grand que l'échelle pour que la sortie dépasse franchement 127. */
        int8_t out2[TI_D];
        int32_t mx2 = 0;
        const int32_t Mq2 = Mq * 8;
        ti_rmsnorm_i8(xn, g8, out2, TI_D, (int32_t)(((int64_t)1 << 16) / TI_D), Mq2, shq, NULL, &mx2);
        int32_t want = 0, nclip = 0;
        for (int i = 0; i < TI_D; i++) {
            const double w = (double)xn[i] * g8[i] / rms * (double)Mq2 / (double)(1 << shq);
            const double a = fabs(w);
            if (a > want) want = (int32_t)(a + 0.5);
            if (fabs(w) > 127.5) nclip++;
        }
        snprintf(buf, sizeof(buf), "max pré-écrêtage=%d, attendu≈%d (%d sorties écrêtées à ±127)",
                 mx2, want, nclip);
        chk("ti_rmsnorm_i8 max avant écrêtage", nclip > 0 && mx2 > 127 && abs(mx2 - want) <= 2, buf);
    }

    /* softmax entier vs softmax réel, en pas int8 */
    int32_t sc[64]; int8_t w8[64]; int32_t scr[64];
    double wsm = 0, dsum = 0;
    for (int rep = 0; rep < 200; rep++) {
        const int n = 1 + (rep % 64);
        double mx = -1e30;
        for (int i = 0; i < n; i++) { sc[i] = (int32_t)((((rep * 131 + i * 977) % 4000) - 2000) * 8) / 8; if (sc[i] > mx) mx = sc[i]; }
        double sw = 0; double ref[64];
        for (int i = 0; i < n; i++) { ref[i] = exp((double)(sc[i] - mx)); sw += ref[i]; }
        for (int i = 0; i < n; i++) ref[i] = ref[i] / sw * 127.0;
        const int32_t sumw = ti_softmax_i8_row(sc, n, w8, scr);
        for (int i = 0; i < n; i++) { const double e = fabs(w8[i] - ref[i]); if (e > wsm) wsm = e; }
        if (n > 4) dsum += fabs((double)sumw - 127.0);
    }
    snprintf(buf, sizeof buf, "écart max = %.2f pas int8 (Σ poids ≈ 127 ± %.2f)", wsm, dsum / 200);
    chk("ti_softmax_i8_row vs réel", wsm < 1.5, buf);

    /* packing 2 bits : aller-retour exact */
    int8_t pk_src[512], pk_dst[512]; uint8_t pk[128];
    for (int i = 0; i < 512; i++) pk_src[i] = (int8_t)((i % 3) - 1);
    ti_pack2(pk_src, pk, 512); ti_unpack2(pk, pk_dst, 512);
    int pdiff = 0;
    for (int i = 0; i < 512; i++) if (pk_src[i] != pk_dst[i]) pdiff++;
    snprintf(buf, sizeof buf, "%d octets pour 512 poids = %.2f bit/poids", 128, 128.0 * 8 / 512);
    chk("ti_pack2/unpack2 exact", pdiff == 0, buf);

    free(x); free(w); free(Y1); free(Y2);
}

/* ------------------------------------------------------- 2. moteur entier -- */
static uint64_t grng = 88172645463325252ull;
static int32_t rnd_bits(void) { grng ^= grng << 13; grng ^= grng >> 7; grng ^= grng << 17; return (int32_t)(grng >> 33); }

static void init_random_ternary(TiInt *m, int seed) {
    grng = 0x9E3779B97F4A7C15ull * (uint64_t)(seed + 1);
    /* poids ternaires ~ {−1, 0, +1} : 1/3 chacun (les modèles réels sont
     * ~1.58 bit/poids ; on quantifie volontairement plus dur ici) */
    const size_t sizes[5] = { (size_t)m->L * 3 * m->D * m->D, (size_t)m->L * m->D * m->D,
                              (size_t)m->L * m->F * m->D, (size_t)m->L * m->D * m->F,
                              (size_t)m->V * m->D };
    int8_t *base[5] = { m->wqkv, m->wo, m->w1, m->w2, m->wout };
    for (int t = 0; t < 5; t++)
        for (size_t i = 0; i < sizes[t]; i++) base[t][i] = (int8_t)((rnd_bits() % 3) - 1);
    for (size_t i = 0; i < (size_t)m->V * m->D; i++) m->emb[i] = (int8_t)((rnd_bits() % 255) - 127);
    for (size_t i = 0; i < (size_t)m->T * m->D; i++) m->pos[i] = (int8_t)((rnd_bits() % 101) - 50);
    for (int l = 0; l < m->L; l++) {
        for (int i = 0; i < m->D; i++) {
            m->g1[l * m->D + i] = (int8_t)(64 + (rnd_bits() % 64));
            m->g2[l * m->D + i] = (int8_t)(64 + (rnd_bits() % 64));
        }
        m->rq_g1[l] = (TiRq){ 4096, 16 };
        m->rq_g2[l] = (TiRq){ 4096, 16 };
        for (int h = 0; h < m->H; h++) m->rq_score[l][h] = (TiRq){ 1024, 17 };
        for (int c = 0; c < 3 * m->D; c++) m->rq_qkv[l * 3 * m->D + c] = (TiRq){ 8192, 15 };
        for (int c = 0; c < m->D; c++) {
            m->rq_wo[l * m->D + c] = (TiRq){ 8192, 15 };
            m->rq_ao[l * m->D + c] = (TiRq){ 6000, 15 };
            m->rq_w2[l * m->D + c] = (TiRq){ 8192, 15 };
        }
        for (int c = 0; c < m->F; c++) {
            m->rq_w1[l * m->F + c] = (TiRq){ 8192, 15 };
            m->rq_sq[l * m->F + c] = (TiRq){ 2000, 14 };
        }
    }
    m->rq_emb = (TiRq){ 12000, 15 };
    m->rq_gf = (TiRq){ 4096, 16 };
    for (int i = 0; i < m->D; i++) m->gf[i] = (int8_t)(64 + (rnd_bits() % 64));
}

static void test_engine(void) {
    printf("== moteur entier : parité préfixage/décodage + déterminisme ==\n");
    TiInt m;
    ti_int_alloc(&m, 1);
    init_random_ternary(&m, 7);
    char buf[256];

    int ids[64];
    for (int i = 0; i < 64; i++) ids[i] = (i * 37 + 11) & 255;
    const int n = 32;

    /* --- déterminisme : deux exécutions complètes → même empreinte --- */
    int32_t *lg = (int32_t *)malloc(sizeof(int32_t) * 256);
    ti_int_reset(&m);
    ti_int_run(&m, ids, 1, n, 0, NULL, 0);
    ti_int_step(&m, ids[n], lg);
    const int64_t h1 = ti_logits_hash(lg, 256);
    ti_int_reset(&m);
    ti_int_run(&m, ids, 1, n, 0, NULL, 0);
    ti_int_step(&m, ids[n], lg);
    const int64_t h2 = ti_logits_hash(lg, 256);
    snprintf(buf, sizeof buf, "hash=%lld deux fois identique", (long long)h1);
    chk("déterminisme du forward entier", h1 == h2, buf);

    /* --- parité préfixage ↔ décodage, position par position --- */
    ti_int_reset(&m);
    int64_t hdec[33];
    for (int t = 0; t <= n; t++) {
        ti_int_step(&m, ids[t], lg);
        hdec[t] = ti_logits_hash(lg, 256);
    }
    int mism = 0; int first_bad = -1;
    int32_t *lgp = (int32_t *)malloc(sizeof(int32_t) * 256);
    for (int t = 0; t <= n; t++) {
        ti_int_reset(&m);
        ti_int_run(&m, ids, 1, t + 1, 0, NULL, 0);   /* préfixage de t+1 jetons */
        ti_int_reset(&m);
        ti_int_run(&m, ids, 1, t, 0, NULL, 0);       /* cache 0..t-1 */
        ti_int_run(&m, &ids[t], 1, 1, t, lgp, 1);    /* décodage de la position t */
        const int64_t hp = ti_logits_hash(lgp, 256);
        if (hp != hdec[t]) { mism++; if (first_bad < 0) first_bad = t; }
    }
    snprintf(buf, sizeof buf, "%d/%d positions identiques (1re divergence: %d)", n + 1 - mism, n + 1, first_bad);
    chk("préfixage = décodage (bit-exact)", mism == 0, buf);

    /* --- bornes de dépassement : |acc| théorique vs mesuré --- */
    long tot, nz, pl, mi;
    ti_int_ternary_stats(&m, &tot, &nz, &pl, &mi);
    snprintf(buf, sizeof buf, "%ld poids, %ld non nuls (%.0f%%)", tot, nz, 100.0 * nz / tot);
    chk("statistiques ternaires", pl + mi == nz, buf);
    printf("  MACs par jeton (mesuré) : %.0f  (= 2·N·(1+T/6) approx)\n",
           (double)m.ops / (n + 1));

    /* --- échantillonnage entier : distribution bornée et reproductible --- */
    TiRng r1 = { 42 }, r2 = { 42 };
    int same = 1, inrange = 1;
    for (int i = 0; i < 200; i++) {
        const int a = ti_int_sample(&m, lg, 256, &r1);       /* T = 1.0 (Q8) */
        const int b = ti_int_sample(&m, lg, 256, &r2);
        if (a != b) same = 0;
        if (a < 0 || a >= 256) inrange = 0;
    }
    snprintf(buf, sizeof buf, "200 tirages : reproductibles=%d, dans [0,255]=%d", same, inrange);
    chk("échantillonnage entier", same && inrange, buf);

    free(lg); free(lgp);
    ti_int_free(&m);
}

/* ------------------------------------------- 3. dump pour référence Python -- */
static void dump_logits(const char *model_path, const char *tok_path) {
    TiInt m;
    ti_int_alloc(&m, 1);
    if (ti_int_load(&m, model_path, 1) != 0) { fprintf(stderr, "load: %s\n", model_path); exit(2); }
    int ids[TI_T]; int n = 0;
    FILE *f = fopen(tok_path, "r");
    if (!f) { fprintf(stderr, "tokens: %s\n", tok_path); exit(2); }
    while (n < TI_T && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    ti_int_reset(&m);
    ti_int_run(&m, ids, 1, n, 0, NULL, 0);
    int32_t lg[TI_V];
    ti_int_step(&m, ids[0], lg);
    printf("# modèle %s, %d jetons\n", model_path, n);
    for (int v = 0; v < TI_V; v++) printf("%d\n", lg[v]);
    printf("# hash %lld\n", (long long)ti_logits_hash(lg, TI_V));
    ti_int_free(&m);
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "--dump")) { dump_logits(argv[2], argv[3]); return 0; }
    const int all = (argc < 2);
    if (all || !strcmp(argv[1], "--kernels")) test_kernels();
    if (all || !strcmp(argv[1], "--engine")) test_engine();
    if (all) {
        printf("\n%s (%d échec(s))\n", g_fail ? "ÉCHEC" : "TOUT PASSE", g_fail);
        return g_fail ? 1 : 0;
    }
    return 0;
}
