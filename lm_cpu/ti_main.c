/* ============================================================================
 * ti_main.c — SPEAR-T1 : entraînement QAT, export entier, vérifications.
 *
 *   --gradcheck            différences finies vs rétroprop (graphe fp32)
 *   --train N              entraînement QAT (forward = moteur entier)
 *   --fp32                 avec --train/--gradcheck : forward flottant exact
 *   --calib P              passes de calibration avant l'entraînement
 *   --bench                débits : moteur entier, décodage, noyaux linéaires
 *   --eval                 perte de validation (mode courant)
 *   --sample "txt" T       génération autorégressive par le moteur entier
 *   --export f.ti          écrit le modèle d'inférence (entiers purs)
 *   --pack                 taille 2 bits/poids vs forme déployée
 *   --fpcheck              vérifie par désassemblage l'absence de flottant
 *                          dans le chemin d'inférence (ti_infer.h)
 * Options : --data f --steps N --lr x --bs B --seed S --ckpt f --save f
 * =========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ti_model.h"

/* --------------------------------------------------------------- utilitaires */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ti_main: impossible d'ouvrir %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *d = (unsigned char *)ti_fa((size_t)*len + 1);
    if (fread(d, 1, (size_t)*len, f) != (size_t)*len) { fprintf(stderr, "ti_main: lecture courte\n"); exit(1); }
    fclose(f);
    return d;
}

static inline uint64_t xs(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static void fill_batch(const unsigned char *data, long len, TiModel *m, uint64_t *rng,
                       int *ids, int *tgt) {
    const int T = m->T, B = m->B;
    const long maxpos = len - T - 2;
    for (int b = 0; b < B; b++) {
        const long pos = (long)(xs(rng) % (uint64_t)maxpos);
        for (int t = 0; t < T; t++) {
            ids[b * T + t] = data[pos + t];
            tgt[b * T + t] = data[pos + t + 1];
        }
    }
}

static double forward_loss(TiModel *m, const int *ids, const int *tgt, int n) {
    if (m->qat) ti_fwd_qat(m, ids, n);
    else        ti_fwd_fp32(m, ids, n, 0);
    return ti_loss_grad(m, tgt, n);
}

static double val_loss(TiModel *m, const unsigned char *data, long len, int nbatch, uint64_t seed) {
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m->ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m->ntok);
    uint64_t r = seed;
    double tot = 0.0;
    for (int i = 0; i < nbatch; i++) {
        fill_batch(data, len, m, &r, ids, tgt);
        tot += forward_loss(m, ids, tgt, m->T);
    }
    free(ids); free(tgt);
    return tot / nbatch;
}

/* échelles des gains (mesurées sur les maîtres, pas sur le moteur) */
static void ti_update_gain_scales(TiModel *m) {
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

/* --------------------------------------------------------------- gradcheck --
 * Différences finies centrées sur le graphe FLOTTANT (le graphe entier n'est
 * pas différentiable). Le même code de rétropropagation est utilisé pour les
 * deux modes : la vérification porte donc bien sur l'implémentation réelle.
 */
static int do_gradcheck(const char *data_path) {
    long len;
    unsigned char *data = slurp(data_path, &len);
    const int B = 2;
    TiModel m;
    ti_model_init(&m, B, 12345);
    m.qat = 0;
    const int n = m.T;                 /* le moteur et la rétroprop exigent n == T */
    const int D = m.D, L = m.L, F = m.F, V = m.V, T = m.T;
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    uint64_t r = 99;
    fill_batch(data, len, &m, &r, ids, tgt);
    /* --- référence : perte + gradients analytiques sur le graphe flottant --- */
    const double loss0 = forward_loss(&m, ids, tgt, n);
    memset(m.grd, 0, sizeof(float) * (size_t)m.npar);
    ti_backward(&m, ids, n);
    printf("== gradcheck : dérivées directionnelles (différences finies centrées) ==\n");
    printf("  lot B=%d T=%d (%d jetons) ; géométrie T=%d D=%d L=%d H=%d HD=%d F=%d V=%d\n",
           B, n, B * n, T, D, L, m.H, m.HD, F, V);
    printf("  perte de référence : %.6f (%.3f bit/octet)\n", loss0, loss0 / log(2.0));
    /* La perte est moyennée sur les jetons ; une perturbation individuelle de
     * poids donne un signal de l'ordre de 1e-4 pour un bruit flottant ~1e-7/ε.
     * On mesure donc une dérivée DIRECTIONNELLE par tenseur (direction clairsemée
     * {−1,0,+1}) : le signal est alors ~100× le bruit, et toute erreur de
     * rétropropagation dans la chaîne se traduit par un échec net. */
    const double eps = 1e-3;
    int fails = 0;
    const char *names[] = { "emb", "pos", "g1", "g2", "gf", "wqkv", "wo", "w1", "w2", "wout" };
    uint64_t rs = 0x243F6A8885A308D3ull;

    for (int id = 0; id < P_NUM; id++) {
        const long sz = m.sz[id];
        /* Densité de la direction : 1 % suffit pour un grand tenseur, mais pour
         * un tenseur de quelques centaines de coefficients le signal tomberait
         * sous le bruit d'arrondi fp32 des paramètres perturbés. */
        const unsigned thr = (sz < 4096) ? 30u : 1u;
        float *p = m.par + m.off[id];
        double *dir = (double *)ti_fa(sizeof(double) * (size_t)sz);
        double gv = 0.0, vn = 0.0;
        for (long i = 0; i < sz; i++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            const double v = ((rs >> 11) % 100u < thr) ? (((rs >> 20) & 1) ? 1.0 : -1.0) : 0.0;
            dir[i] = v;
            gv += (double)m.grd[m.off[id] + i] * v;
            vn += v * v;
        }
        for (long i = 0; i < sz; i++) p[i] += (float)(eps * dir[i]);
        const double lp = forward_loss(&m, ids, tgt, n);
        for (long i = 0; i < sz; i++) p[i] -= (float)(2.0 * eps * dir[i]);
        const double lm = forward_loss(&m, ids, tgt, n);
        for (long i = 0; i < sz; i++) p[i] += (float)(eps * dir[i]);
        const double fd = (lp - lm) / (2.0 * eps);
        const double rel = fabs(fd - gv) / (fabs(fd) + fabs(gv) + 1e-9);
        const int ok = rel < 2e-2;
        if (!ok) fails++;
        printf("  %-5s (n=%7ld, |v|=%6.0f) : g·v=% .6e  fd=% .6e  rel=%.2e  %s\n",
               names[id], sz, sqrt(vn), gv, fd, rel, ok ? "OK" : "ÉCHEC");
        free(dir);
    }
    /* contrôle par poids : quelques sondes individuelles (eps plus grand) */
    printf("  -- sondes individuelles (eps=5e-2) :\n");
    struct { int id; long l; long i; } pr[] = {
        { P_WOUT, 0, 3L * D + 7 }, { P_WOUT, 0, (long)(V - 5) * D + 3 },
        { P_WQKV, L - 1, (long)(D - 3) * D + 9 }, { P_WQKV, 0, 5L * D + 9 },
        { P_WO,   L - 1, 13L * D + 17 }, { P_WO,   0, 5L * D + 11 },
        { P_W1,   L - 1, (long)(F - 7) * D + 5 }, { P_W1, 0, 77L * D + 5 },
        { P_W2,   L - 1, 9L * F + F - 9 }, { P_W2, 0, 9L * F + 50 },
        { P_G1,   L - 1, 4L }, { P_G1, 0, 7L }, { P_G2, L - 1, D - 2 }, { P_G2, 0, 9L },
        { P_GF,   0, 11L }, { P_POS, 0, 5L * D + 9 }
    };
    const double eps2 = 2e-3;   /* sondes isolées : signal plus large que le bruit fp32 */
    const int NP = (int)(sizeof(pr) / sizeof(pr[0]));
    int fails2 = 0;
    for (int k = 0; k < NP; k++) {
        const int id = pr[k].id;
        const long blk = (id == P_WQKV) ? 3L * D * D : (id == P_W1) ? (long)F * D :
                         (id == P_W2) ? (long)D * F : (id == P_WO) ? (long)D * D :
                         (id == P_G1 || id == P_G2) ? (long)D : 0;
        const long idx = m.off[id] + pr[k].l * blk + pr[k].i;
        float *p = m.par + idx;
        const float save = *p;
        *p = save + (float)eps2; const double lp = forward_loss(&m, ids, tgt, n);
        *p = save - (float)eps2; const double lm = forward_loss(&m, ids, tgt, n);
        *p = save;
        const double fd = (lp - lm) / (2.0 * eps2);
        const double an = m.grd[idx];
        const double rel = fabs(fd - an) / (fabs(fd) + fabs(an) + 1e-7);
        const int ok = rel < 5e-2 || (fabs(fd) < 1e-4 && fabs(an) < 1e-4);
        if (!ok) fails2++;
        char lbl[48];
        snprintf(lbl, sizeof lbl, "%s(l=%ld)[%ld]", names[id], pr[k].l, pr[k].i);
        printf("  %-16s analytique=% .5e  FD=% .5e  rel=%.2e  %s\n",
               lbl, an, fd, rel, ok ? "OK" : "ÉCHEC");
    }
    printf("  => %s (directionnel %d/%d, sondes %d/%d)\n",
           (fails || fails2) ? "ÉCHEC" : "TOUT PASSE", P_NUM - fails, P_NUM, NP - fails2, NP);
    ti_model_free(&m);
    free(ids); free(tgt); free(data);
    return fails + fails2;
}

static void do_bench(const char *data_path, const char *ckpt) {
    long len;
    unsigned char *data = slurp(data_path, &len);
    TiModel m;
    ti_model_init(&m, TI_B, 4242);
    if (ckpt) { const int r = ti_model_load(&m, ckpt); printf("  checkpoint %s : %d flottants lus\n", ckpt, r); }
    ti_update_gain_scales(&m);
    ti_calibrate(&m, data, len, 8, 2);
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    uint64_t r = 7;
    fill_batch(data, len, &m, &r, ids, tgt);
    /* --- forward entier (préfixage) --- */
    m.e.ops = 0;                       /* remis à zéro : les MACs sont comptés en propre */
    double t0 = now_s();
    const int REP = 20;
    for (int i = 0; i < REP; i++) ti_fwd_qat(&m, ids, m.T);
    double dt = (now_s() - t0) / REP;
    printf("== débits mesurés (B=%d, T=%d → %d jetons) ==\n", m.B, m.T, m.ntok);
    printf("  forward entier (préfixage)      : %8.2f ms  →  %8.0f jetons/s\n",
           1e3 * dt, m.ntok / dt);
    printf("  MACs par jeton (préfixage)      : %8lld\n",
           (long long)(m.e.ops / (int64_t)(REP * m.ntok)));
    /* --- décodage autorégressif (1 jeton) --- */
    m.e.ops = 0;
    ti_int_reset(&m.e);
    t0 = now_s();
    const int NST = 200;
    for (int i = 0; i < NST; i++) {
        ti_int_step(&m.e, data[i], m.log32);
        if (m.e.ncache >= m.T) ti_int_reset(&m.e);
    }
    dt = now_s() - t0;
    printf("  décodage autorégressif (1 jeton): %8.2f ms/jeton → %6.0f jetons/s  (%lld MAC/jeton)\n",
           1e3 * dt / NST, NST / dt, (long long)(m.e.ops / NST));
    /* --- pas d'entraînement complet --- */
    memset(m.grd, 0, sizeof(float) * (size_t)m.npar);
    t0 = now_s();
    const double l0 = forward_loss(&m, ids, tgt, m.T);
    const double t_fwd = now_s() - t0;
    t0 = now_s();
    ti_backward(&m, ids, m.T);
    const double t_bwd = now_s() - t0;
    t0 = now_s();
    ti_export(&m);
    const double t_exp = now_s() - t0;
    t0 = now_s();
    ti_adam(&m, 0.0f, 0.0f);
    const double t_adm = now_s() - t0;
    const double t_step = t_fwd + t_bwd + t_exp + t_adm;
    printf("  pas d'entraînement              : %8.2f ms  dont forward %.1f ms (moteur entier)\n",
           1e3 * t_step, 1e3 * t_fwd);
    printf("  (rétroprop %.1f ms, export %.1f ms, Adam %.1f ms) → %.0f jetons/s en apprentissage\n",
           1e3 * t_bwd, 1e3 * t_exp, 1e3 * t_adm, m.ntok / t_step);
    printf("  perte sur le lot                : %.4f (soit %.3f bit/octet)\n",
           l0, l0 / log(2.0));
    /* --- noyaux linéaires : blocage par paquets de jetons --- */
    {
        const int n = m.ntok, K = m.D, N = 3 * m.D;
        float *X = (float *)ti_fa(sizeof(float) * (size_t)n * K);
        float *Y = (float *)ti_fa(sizeof(float) * (size_t)n * N);
        float *dW = (float *)ti_fa(sizeof(float) * (size_t)N * K);
        float *Yt = (float *)ti_fa(sizeof(float) * (size_t)N * n);
        for (int i = 0; i < n * K; i++) X[i] = 0.5f * sinf((float)i);
        for (int i = 0; i < n * N; i++) Y[i] = 0.5f * cosf((float)i);
        double tA = now_s();
        ti_lin_dw(X, Y, dW, n, K, N, NULL, Yt);
        tA = now_s() - tA;
        /* référence naïve (même ordre de sommation que la version bloquée) */
        memset(dW, 0, sizeof(float) * (size_t)N * K);
        double tB = now_s();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < N; j++) {
                const float gy = Y[(size_t)i * N + j];
                if (gy == 0.0f) continue;
                const float *x = X + (size_t)i * K;
                float *dw = dW + (size_t)j * K;
                for (int k = 0; k < K; k++) dw[k] += gy * x[k];
            }
        tB = now_s() - tB;
        free(Yt);
        printf("  gradient de poids %d×%d×%d       : %6.2f ms bloqué vs %6.2f ms naïf  (×%.2f)\n",
               n, K, N, 1e3 * tA, 1e3 * tB, (tB > 0 ? tB / tA : 0.0));
        free(X); free(Y); free(dW);
    }
    free(ids); free(tgt); free(data);
    ti_model_free(&m);
}

/* --------------------------------------------------------------- fidélité -- */
/* Compare, site par site, le moteur entier au graphe fp32 ÉQUIVALENT (mêmes
 * poids ternaires déquantifiés sw·w8, mêmes échelles) : l'écart mesuré est donc
 * exactement ce que coûte la quantification int8 des activations — ni plus
 * (aucune divergence d'algorithme) ni moins (aucune échelle masquée). */
static double rel_rms(const float *a, const float *b, long n, double *mx_a, double *mx_b) {
    double num = 0, den = 0, ma = 0, mb = 0;
    for (long i = 0; i < n; i++) {
        const double d = (double)a[i] - b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
        if (fabs((double)a[i]) > ma) ma = fabs((double)a[i]);
        if (fabs((double)b[i]) > mb) mb = fabs((double)b[i]);
    }
    *mx_a = ma; *mx_b = mb;
    return den > 0 ? sqrt(num / den) : 0.0;
}
static int do_fidelity(const char *data_path, int nbatch) {
    long len; unsigned char *data = slurp(data_path, &len);
    TiModel m;
    ti_model_init(&m, TI_B, 4242);
    ti_calibrate(&m, data, len, nbatch, 1);
    printf("== fidélité moteur entier ↔ graphe fp32 (poids ternaires effectifs) ==\n");
    printf("  échelles de résidu par couche :");
    for (int l = 0; l <= m.L; l++) printf(" %.5f", m.S_res[l]);
    printf("\n  %-8s %13s %13s %10s\n", "site", "max fp32", "max moteur", "écart RMS");
    const int D = m.D, L = m.L, F = m.F, B = m.B, T = m.T;
    const size_t BT = (size_t)m.ntok;
    int *ids = (int *)ti_fa(sizeof(int) * BT);
    int *tgt = (int *)ti_fa(sizeof(int) * BT);
    uint64_t r = 999;
    fill_batch(data, len, &m, &r, ids, tgt);
    /* référence fp32 (poids ternaires) */
    ti_fwd_fp32(&m, ids, T, 1);
    const double l32 = ti_loss_grad(&m, tgt, T);
    float *X32  = (float *)ti_fa(sizeof(float) * BT * D * (L + 1));
    float *N132 = (float *)ti_fa(sizeof(float) * BT * D * L);
    float *N232 = (float *)ti_fa(sizeof(float) * BT * D * L);
    float *Q32  = (float *)ti_fa(sizeof(float) * BT * 3 * D * L);
    float *A32  = (float *)ti_fa(sizeof(float) * BT * D * L);
    float *F32  = (float *)ti_fa(sizeof(float) * BT * F * L);
    float *S32  = (float *)ti_fa(sizeof(float) * BT * F * L);
    float *NF32 = (float *)ti_fa(sizeof(float) * BT * D);
    memcpy(X32, m.X, sizeof(float) * BT * D * (L + 1));
    memcpy(N132, m.N1, sizeof(float) * BT * D * L); memcpy(N232, m.N2, sizeof(float) * BT * D * L);
    memcpy(Q32, m.QKV, sizeof(float) * BT * 3 * D * L); memcpy(A32, m.AO, sizeof(float) * BT * D * L);
    memcpy(F32, m.F1, sizeof(float) * BT * F * L); memcpy(S32, m.FSQ, sizeof(float) * BT * F * L);
    memcpy(NF32, m.NF, sizeof(float) * BT * D);
    float *ATTN32 = (float *)ti_fa(sizeof(float) * (size_t)B * L * TI_H * TI_T * TI_T);
    memcpy(ATTN32, m.ATTN, sizeof(float) * (size_t)B * L * TI_H * TI_T * TI_T);
    /* moteur entier */
    ti_fwd_qat(&m, ids, T);
    const double lq = ti_loss_grad(&m, tgt, T);
    double ma, mb, worst = 0.0;
    for (int l = 0; l <= L; l++) {
        char tag[16]; snprintf(tag, sizeof(tag), "X[%d]", l);
        const double e = rel_rms(m.X + (size_t)l * BT * D, X32 + (size_t)l * BT * D, (long)(BT * D), &ma, &mb);
        if (e > worst) worst = e;
        printf("  %-8s %13.5f %13.5f %9.3f%%\n", tag, mb, ma, 100.0 * e);
    }
    #define LIGNE(tag, buf, ref, cnt) do { \
        const double e_ = rel_rms((buf), (ref), (long)(cnt), &ma, &mb); \
        if (e_ > worst) worst = e_; \
        printf("  %-8s %13.5f %13.5f %9.3f%%\n", tag, mb, ma, 100.0 * e_); } while (0)
    for (int l = 0; l < L; l++) {
        char tag[16];
        snprintf(tag, sizeof(tag), "N1[%d]", l); LIGNE(tag, m.N1 + (size_t)l * BT * D, N132 + (size_t)l * BT * D, BT * D);
        snprintf(tag, sizeof(tag), "N2[%d]", l); LIGNE(tag, m.N2 + (size_t)l * BT * D, N232 + (size_t)l * BT * D, BT * D);
        snprintf(tag, sizeof(tag), "Q[%d]", l);  LIGNE(tag, m.QKV + (size_t)l * BT * 3 * D, Q32 + (size_t)l * BT * 3 * D, BT * D);
        snprintf(tag, sizeof(tag), "K[%d]", l);  LIGNE(tag, m.QKV + (size_t)l * BT * 3 * D + D, Q32 + (size_t)l * BT * 3 * D + D, BT * D);
        snprintf(tag, sizeof(tag), "V[%d]", l);  LIGNE(tag, m.QKV + (size_t)l * BT * 3 * D + 2 * D, Q32 + (size_t)l * BT * 3 * D + 2 * D, BT * D);
        snprintf(tag, sizeof(tag), "AO[%d]", l); LIGNE(tag, m.AO + (size_t)l * BT * D, A32 + (size_t)l * BT * D, BT * D);
        snprintf(tag, sizeof(tag), "F[%d]", l);  LIGNE(tag, m.F1 + (size_t)l * BT * F, F32 + (size_t)l * BT * F, BT * F);
        snprintf(tag, sizeof(tag), "SQ[%d]", l); LIGNE(tag, m.FSQ + (size_t)l * BT * F, S32 + (size_t)l * BT * F, BT * F);
    }
    {   /* poids d'attention : c'est ici qu'une erreur d'échelle des scores serait
         * visible — un décalage de température ne change presque pas la somme
         * des poids (≈127) mais change complètement leur répartition */
        double mx = 0, num = 0, den = 0;
        for (int l = 0; l < L; l++)
            for (int b = 0; b < B; b++)
                for (int h = 0; h < m.H; h++) {
                    const float *pe = m.ATTN + ((((size_t)l * B) + b) * m.H + h) * TI_T * TI_T;
                    const float *pf = ATTN32 + ((((size_t)l * B) + b) * m.H + h) * TI_T * TI_T;
                    for (int t = 0; t < T; t++)
                        for (int u = 0; u <= t; u++) {
                            const double d = (double)pe[t * TI_T + u] - pf[t * TI_T + u];
                            num += d * d; den += (double)pf[t * TI_T + u] * (double)pf[t * TI_T + u];
                            if (fabs(d) > mx) mx = fabs(d);
                        }
                }
        printf("  %-8s %13s %13s %9.3f%%   (max|Δp| = %.4f)\n", "ATTN", "-", "-",
               100.0 * sqrt(num / den), mx);
    }
    {   /* norme finale (hors boucle des blocs) */
        const double e_ = rel_rms(m.NF, NF32, (long)(BT * D), &ma, &mb);
        if (e_ > worst) worst = e_;
        printf("  %-8s %13.5f %13.5f %9.3f%%\n", "NF", mb, ma, 100.0 * e_);
    }
    printf("  perte fp32(ternaire) %.4f  |  perte moteur entier %.4f  (écart %.3f%%)\n",
           l32, lq, 100.0 * fabs(lq - l32) / l32);
    printf("  pire écart RMS site : %.3f%%\n", 100.0 * worst);
    {
        long nz = 0, tot = 0;
        for (int l = 0; l <= L; l++) {
            const int8_t *x = m.e.xq + (size_t)l * BT * D;
            for (size_t i = 0; i < BT * D; i++) { tot++; if (x[i] >= 127 || x[i] <= -127) nz++; }
        }
        printf("  résidu écrêté : %ld/%ld (%.3f%%)\n", nz, tot, 100.0 * (double)nz / (double)tot);
    }
    free(ids); free(tgt); free(data);
    free(X32); free(N132); free(N232); free(Q32); free(A32); free(F32); free(S32); free(NF32); free(ATTN32);
    ti_model_free(&m);
    return 0;
}

/* ------------------------------------------------------------------- train -- */
static void do_train(int steps, const char *data_path, float lr, float lrmin, int B, uint64_t seed,
                     const char *ckpt_in, const char *ckpt_out, int calib_passes, int fp32,
                     const char *export_path, const char *val_path, long holdout,
                     int recalib, int ptq) {
    long len;
    unsigned char *data = slurp(data_path, &len);
    long vlen = 0;
    unsigned char *vdata = NULL;
    if (holdout > 0 && holdout < len / 2) {
        /* validation HONNÊTE : les derniers octets du corpus ne servent jamais à
         * l'entraînement (ni à la calibration, qui échantillonne dans `len`) */
        vlen = holdout;
        vdata = (unsigned char *)ti_fa((size_t)vlen);
        memcpy(vdata, data + len - holdout, (size_t)vlen);
        len -= holdout;
    }
    if (val_path) vdata = slurp(val_path, &vlen);
    TiModel m;
    ti_model_init(&m, B, seed);
    m.qat = fp32 ? 0 : 1;
    if (ckpt_in) {
        const int r = ti_model_load(&m, ckpt_in);
        printf("checkpoint %s rechargé (%.1f M flottants, pas %ld)\n", ckpt_in, r / 1e6, m.step);
    }
    printf("== SPEAR-T1 : entraînement %s ==\n", m.qat ? "QAT (forward = moteur entier)" : "fp32 (référence)");
    printf("  géométrie  : T=%d D=%d L=%d H=%d HD=%d F=%d V=%d B=%d\n", m.T, m.D, m.L, m.H, m.HD, m.F, m.V, m.B);
    printf("  paramètres : %ld (%.2f M)   corpus %ld octets, val %ld octets\n",
           m.npar, m.npar / 1e6, len, vlen);
    ti_update_gain_scales(&m);
    ti_export(&m);
    if (calib_passes > 0) {
        double t0 = now_s();
        ti_calibrate(&m, data, len, 12, calib_passes);
        printf("  calibration : %d passe(s) du moteur entier en %.1f s\n", calib_passes, now_s() - t0);
        printf("  échelles de résidu par couche :");
        for (int l = 0; l <= m.L; l++) printf(" %.5f", m.S_res[l]);
        printf("  (S_emb=%.5f)\n", m.S_emb);
    }
    if (val_path) printf("  perte de validation initiale    : %.4f\n", val_loss(&m, vdata, vlen, 8, 4242));
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    uint64_t rng = seed * 6364136223846793005ull + 1442695040888963407ull;
    double t_start = now_s(), t_log = t_start;
    long tok_seen = 0;
    const float wd = 0.0f;
    for (int s = 0; s < steps; s++) {
        /* échauffement (les poids maîtres bougent très vite aux premiers pas :
         * mesuré, Q(l=1) saturait à 96 % entre deux recalibrations) puis
         * décroissance cosinus jusqu'à lr·lrmin */
        const int warm = steps / 20 > 25 ? 25 : (steps / 20 < 5 ? 5 : steps / 20);
        const float t_w = (float)(s + 1) / (float)warm;
        const float lrt = lr * (t_w < 1.0f ? t_w : 1.0f) *
                          (lrmin + (1.0f - lrmin) * 0.5f * (1.0f + cosf(3.14159265f * (float)s / (float)steps)));
        fill_batch(data, len, &m, &rng, ids, tgt);
        const double loss = forward_loss(&m, ids, tgt, m.T);
        ti_count_saturation(&m);
        memset(m.grd, 0, sizeof(float) * (size_t)m.npar);
        ti_backward(&m, ids, m.T);
        ti_adam(&m, lrt, wd);
        /* re-quantification : échelles de gains à chaque pas, échelles
         * d'activation toutes les 200 pas (mesurées par le moteur) */
        ti_update_gain_scales(&m);
        /* recalibration fréquente (toutes les 50 pas) mais courte : les poids
         * bougent vite au début, une échelle figée 200 pas écrête le résidu
         * (mesuré : jusqu'à 97 % d'un site à ±127) */
        if (recalib > 0 && (s + 1) % recalib == 0) ti_calibrate(&m, data, len, 2, 1);
        else ti_export(&m);
        tok_seen += m.ntok;
        if ((s + 1) % 25 == 0 || s == 0) {
            const double t = now_s();
            const double dps = (t - t_log) / ((s + 1) % 25 == 0 ? 25 : 25);
            char nm[24];
            ti_site_name(m.sat_worst, nm, sizeof(nm), &m);
            printf("  pas %6d  perte %.4f (%.3f bit/o)  %8.0f jetons/s  %.1f s  sat %.2f%% (pire %s %.2f%%)%s\n",
                   s + 1, loss, loss / log(2.0), m.ntok / (dps > 0 ? dps : 1e-9),
                   t - t_start, 100.0 * (double)m.sat_count / (double)(m.sat_total ? m.sat_total : 1),
                   nm, m.sat_worst_frac,
                   m.grd_bad ? "  [grd non finis neutralisés]" : "");
            t_log = t;
        }
        if (ckpt_out && (s + 1) % 500 == 0) {
            ti_model_save(&m, ckpt_out);
            /* le modèle entier est exporté au même rythme : un arrêt n'enlève
             * jamais le seul artefact utilisable */
            if (export_path) ti_int_save(&m.e, export_path);
            printf("  checkpoint écrit : %s (pas %d)%s\n", ckpt_out, s + 1,
                   export_path ? " + modèle entier exporté" : "");
        }
    }
    const double total = now_s() - t_start;
    printf("  terminé : %d pas, %ld jetons en %.1f s (%.0f jetons/s)\n",
           steps, tok_seen, total, tok_seen / total);
    if (ptq) {
        /* --ptq : mesure la perte du MOTEUR ENTIER après un entraînement fp32.
         * C'est la question honnête « que coûte la quantification ? » sans le
         * biais de l'entraînement en quantification simulée. */
        unsigned char *qdata = vdata ? vdata : data;
        const long qlen = vdata ? vlen : len;
        ti_calibrate(&m, qdata, qlen, 4, 1);
        /* m.qat = 1 : SANS cela, val_loss rejoue le forward fp32 et l'on mesure
         * la perte du modèle flottant en croyant mesurer celle du moteur entier
         * (bug de mesure, pas de modèle — la valeur « 2,4650 » d'un premier run
         * était en réalité la perte fp32 telle quelle). */
        m.qat = 1;
        printf("  [ptq] perte du MOTEUR ENTIER après entraînement fp32 :");
        for (int rep = 0; rep < 2; rep++)
            printf(" %.4f", val_loss(&m, qdata, qlen, 8, 4242));
        printf("\n");
        m.qat = 0;
    }
    if (val_path) {
        const double vl = val_loss(&m, vdata, vlen, 8, 4242);
        printf("  perte de validation finale      : %.4f (%.3f bit/octet)\n", vl, vl / log(2.0));
    }
    if (ckpt_out) ti_model_save(&m, ckpt_out);
    if (export_path) {
        ti_export(&m);
        double rel, nz;
        ti_export_fidelity(&m, &rel, &nz);
        long tot, nzero, plus, minus;
        ti_int_ternary_stats(&m.e, &tot, &nzero, &plus, &minus);
        printf("  export entier → %s (Wout : erreur relative %.4f, %ld/%ld poids non nuls)\n",
               export_path, rel, nzero, tot);
        if (ti_int_save(&m.e, export_path) == 0) {
            FILE *f = fopen(export_path, "rb");
            fseek(f, 0, SEEK_END);
            printf("  fichier modèle : %ld octets (%.2f bits/poids pour %ld poids)\n",
                   ftell(f), 8.0 * (double)ftell(f) / (double)tot, tot);
            fclose(f);
        }
    }
    free(ids); free(tgt); free(data);
    if (vdata) free(vdata);
    ti_model_free(&m);
}

/* ------------------------------------------------------------------ sample -- */
static void do_sample(const char *path, const char *prompt, int ntok, int temp_q8, uint64_t seed) {
    TiInt e;
    ti_int_alloc(&e, 1);
    if (ti_int_load(&e, path, 1) < 0) { fprintf(stderr, "ti_main: modèle illisible %s\n", path); exit(1); }
    TiRng rng = { seed ? seed : 0x9E3779B97F4A7C15ull };
    ti_int_reset(&e);
    /* « T » désigne ici la LONGUEUR DE CONTEXTE du modèle, pas la température :
     * l'unité de la température est le Q8 (--temp 256 = T 1.00). */
    printf("== génération (contexte T=%d, température %.2f soit --temp %d) ==\n%s",
           e.T, temp_q8 / 256.0, temp_q8, prompt);
    const size_t plen = strlen(prompt);
    for (size_t i = 0; i < plen && e.ncache < e.T; i++) ti_int_step(&e, (unsigned char)prompt[i], e.acc);
    for (int i = 0; i < ntok && e.ncache < e.T; i++) {
        /* facteur global : logit physique = log32·(swq/4096)·logit_scale, donc
         * mult_q24 = logit_scale·2^12 (voir ti_int_sample) */
        int32_t mult = (int32_t)((double)e.logit_scale * 4096.0);
        if (mult < 1) mult = 1;
        if (mult > (int32_t)1 << 30) mult = (int32_t)1 << 30;
        const int id = ti_int_sample(&e, e.acc, temp_q8, &rng, e.swq, mult);
        ti_int_step(&e, id, e.acc);
        putchar(id >= 32 && id < 127 ? id : (id == 10 ? '\n' : '.'));
    }
    putchar('\n');
    ti_int_free(&e);
}

/* --------------------------------------------------------------- --fpcheck ---
 * Vérification INDÉPENDANTE par désassemblage : on compile un fichier qui
 * n'inclut que le moteur d'inférence, on désassemble et on cherche toute
 * instruction flottante (SSE/AVX scalaire ou vectorielle, conversions).
 * Le test échoue si ti_int_run / ti_attn_layer / ti_int_step contiennent une
 * instruction flottante.
 */
static int do_fpcheck(void) {
    const char *src = "/tmp/ti_fpcheck.c";
    const char *obj = "/tmp/ti_fpcheck.o";
    FILE *f = fopen(src, "w");
    if (!f) { fprintf(stderr, "ti_main: /tmp non inscriptible\n"); return 1; }
    fputs("#include \"ti_infer.h\"\n"
          "int probe(TiInt *m, const int *ids, int B, int n, int32_t *lg) {\n"
          "  ti_int_reset(m);\n"
          "  ti_int_run(m, ids, B, n, 0, lg, 2);\n"
          "  ti_attn_layer(m, 0, 0, 0, 1);\n"
          "  ti_int_step(m, 65, lg);\n"
          "  return ti_int_sample(m, lg, 200, (TiRng*)0, NULL, 1<<24);\n"
          "}\n", f);
    fclose(f);
    char cmd[512];
    snprintf(cmd, sizeof cmd, "gcc -O2 -march=native -I. -c %s -o %s 2>/dev/null", src, obj);
    if (system(cmd) != 0) { fprintf(stderr, "ti_main: compilation du probe échouée\n"); return 1; }
    snprintf(cmd, sizeof cmd, "objdump -d %s", obj);
    FILE *p = popen(cmd, "r");
    if (!p) { fprintf(stderr, "ti_main: objdump indisponible\n"); return 1; }
    char line[512], cur[128] = "?";
    int nfloat = 0, nins = 0;
    const char *fl[] = { "addps", "subps", "mulps", "divps", "sqrtps", "addss", "subss",
                         "mulss", "divss", "sqrtss", "addsd", "mulsd", "divsd", "sqrtsd",
                         "cvtsi2ss", "cvtsi2sd", "cvtss2sd", "cvtsd2ss", "cvttss2si",
                         "vcvt", "vfmadd", "vfnmadd", "vfmsub", "vmaxps", "vminps", "vcomis",
                         "ucomis", "comiss", "vucom", NULL };
    while (fgets(line, sizeof line, p)) {
        if (line[0] != ' ' && strchr(line, '>')) {
            char *a = strchr(line, '<'), *b = strchr(line, '>');
            if (a && b && b > a) {
                size_t n2 = (size_t)(b - a - 1);
                if (n2 > sizeof(cur) - 1) n2 = sizeof(cur) - 1;
                memcpy(cur, a + 1, n2); cur[n2] = 0;
            }
        }
        const char *tab = strchr(line, '\t');
        if (!tab) continue;
        nins++;
        for (int i = 0; fl[i]; i++)
            if (strstr(tab, fl[i])) {
                if (!strncmp(cur, "probe", 5)) {
                    printf("  FLOTTANT %s dans %s\n", fl[i], cur);
                    nfloat++;
                }
                break;
            }
    }
    pclose(p);
    printf("== fpcheck (désassemblage de ti_infer.h seul) ==\n");
    printf("  instructions examinées : %d ; instructions flottantes dans le chemin d'inférence : %d\n",
           nins, nfloat);
    printf("  => %s\n", nfloat == 0 ? "TOUT PASSE (0 flottant)" : "ÉCHEC");
    return nfloat != 0;
}

/* -------------------------------------------------------------------- pack -- */
static void do_pack(const char *path) {
    TiInt e;
    ti_int_alloc(&e, 1);
    if (ti_int_load(&e, path, 1) < 0) { fprintf(stderr, "ti_main: modèle illisible\n"); exit(1); }
    long tot, nz, plus, minus;
    ti_int_ternary_stats(&e, &tot, &nz, &plus, &minus);
    uint8_t *buf = (uint8_t *)ti_fa((size_t)(tot + 3) / 4);
    /* compactage de l'ensemble des matrices ternaires du moteur */
    long off = 0;
    ti_pack2(e.wqkv, buf + 0, (int)((long)e.L * 3 * e.D * e.D)); off += (long)e.L * 3 * e.D * e.D;
    (void)off;
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    const long fsz = ftell(f);
    fclose(f);
    printf("== packing ternaire ==\n");
    printf("  poids ternaires        : %ld (dont %ld non nuls, +%ld/−%ld)\n", tot, nz, plus, minus);
    printf("  forme déployée int8    : %ld octets (1 octet/poids, RAM d'inférence)\n", tot);
    printf("  2 bits/poids           : %ld octets (théorique) \n", (tot + 3) / 4);
    printf("  fichier MIT1           : %ld octets (%.2f bits/poids, requants inclus)\n",
           fsz, 8.0 * (double)fsz / (double)tot);
    free(buf);
    ti_int_free(&e);
}

/* -------------------------------------------------------------------- main -- */
static void usage(void) {
    puts("usage: ti_main [--train N|--gradcheck|--fidelity|--bench|--ptq|--sample \"p\" N|--pack|--fpcheck|--eval] [options]");
    puts("  --data f --val f --lr x --bs B --seed S --ckpt f --save f --export f.ti --calib P --fp32");
}

int main(int argc, char **argv) {
    const char *mode = "--train";
    const char *data_path = "../data/shakespeare.txt";
    const char *val_path = NULL, *ckpt_in = NULL, *ckpt_out = NULL, *export_path = NULL;
    const char *prompt = "To be or not";
    int steps = 200, bs = TI_B, calib = 2, fp32 = 0, temp_q8 = 192, smp = 120;
    int recalib = 50, ptq = 0;
    float lr = 3e-3f, lrmin = 1.0f;
    long holdout = 0;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gradcheck")) mode = "--gradcheck";
        else if (!strcmp(argv[i], "--train")) mode = "--train", steps = (i + 1 < argc) ? atoi(argv[++i]) : 200;
        else if (!strcmp(argv[i], "--bench")) mode = "--bench";
        else if (!strcmp(argv[i], "--pack")) mode = "--pack";
        else if (!strcmp(argv[i], "--fidelity")) mode = "--fidelity";
        else if (!strcmp(argv[i], "--fpcheck")) mode = "--fpcheck";
        else if (!strcmp(argv[i], "--eval")) mode = "--eval";
        else if (!strcmp(argv[i], "--sample")) {
            mode = "--sample";
            if (i + 1 < argc && argv[i + 1][0] != '-') prompt = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-') smp = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--data")) data_path = argv[++i];
        else if (!strcmp(argv[i], "--val")) val_path = argv[++i];
        else if (!strcmp(argv[i], "--lr")) lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--bs")) bs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calib")) calib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt")) ckpt_in = ckpt_out = argv[++i];
        else if (!strcmp(argv[i], "--save")) ckpt_out = argv[++i];
        else if (!strcmp(argv[i], "--export")) export_path = argv[++i];
        else if (!strcmp(argv[i], "--fp32")) fp32 = 1;
        else if (!strcmp(argv[i], "--temp")) temp_q8 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lrmin")) lrmin = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--recalib")) recalib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ptq")) ptq = 1;
        else if (!strcmp(argv[i], "--holdout")) holdout = atol(argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); return 0; }
        else { fprintf(stderr, "option inconnue : %s\n", argv[i]); usage(); return 1; }
    }
    if (!strcmp(mode, "--gradcheck")) return do_gradcheck(data_path);
    if (!strcmp(mode, "--fidelity")) return do_fidelity(data_path, 8);
    if (!strcmp(mode, "--fpcheck")) return do_fpcheck();
    if (!strcmp(mode, "--bench")) { do_bench(data_path, ckpt_in); return 0; }
    if (!strcmp(mode, "--pack")) { do_pack(export_path ? export_path : "/tmp/spear_t1.ti"); return 0; }
    if (!strcmp(mode, "--sample")) { do_sample(export_path ? export_path : "/tmp/spear_t1.ti", prompt, smp, temp_q8, seed); return 0; }
    if (!strcmp(mode, "--eval")) {
        long len, vlen;
        unsigned char *d = slurp(val_path ? val_path : data_path, &len);
        TiModel m;
        ti_model_init(&m, bs, seed);
        m.qat = fp32 ? 0 : 1;
        if (ckpt_in) ti_model_load(&m, ckpt_in);
        ti_update_gain_scales(&m);
        ti_export(&m);
        ti_calibrate(&m, d, len, 12, calib);
        const double l = val_loss(&m, d, len, 16, 4242);
        printf("perte (mode %s) : %.4f  soit %.3f bit/octet, perplexité %.2f\n",
               m.qat ? "QAT" : "fp32", l, l / log(2.0), exp(l));
        ti_model_free(&m);
        free(d);
        return 0;
    }
    do_train(steps, data_path, lr, lrmin, bs, seed, ckpt_in, ckpt_out, calib, fp32, export_path, val_path, holdout, recalib, ptq);
    return 0;
}
