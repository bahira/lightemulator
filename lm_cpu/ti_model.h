/* ============================================================================
 * ti_model.h — SPEAR-T1 : entraînement quantifié (QAT) + export entier.
 *
 * Principe : le forward d'entraînement EST le moteur entier (ti_infer.h) ; les
 * activations sont ensuite déquantifiées (S·q8) pour la perte et la rétroprop.
 * Conséquence : aucune divergence entraînement/inférence, par construction.
 *
 * Quantificateurs (tous à gradient direct « straight-through », STE) :
 *   poids       : ternaires {−1,0,+1}, échelle par ligne s = 1/mean|W[n,:]|
 *   gains norme : int8, échelle = max/127, gelée pendant le pas
 *   activations : int8, échelles STATIQUES par site, mesurées par le moteur
 *                 lui-même (TI_TRACK) — aucune statistique supposée
 *   résiduel    : échelle unique + addition saturante int8
 *
 * Toutes les boucles massives sont bloquées par paquets de jetons et
 * parallélisées sur des lignes disjointes (donc bit-déterministes).
 * =========================================================================== */
#ifndef TI_MODEL_H
#define TI_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ti_config.h"
#include "ti_infer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

enum { P_EMB, P_POS, P_G1, P_G2, P_GF, P_WQKV, P_WO, P_W1, P_W2, P_WOUT, P_NUM };

#define TI_TB 16            /* paquet de jetons (blocage mémoire) */

typedef struct {
    TiInt e;                        /* moteur entier */
    int T, D, L, H, HD, F, V, B, ntok;
    /* ombres fp32 (maîtres) + gradients + Adam */
    float *par, *grd, *adm1, *adm2;
    long off[P_NUM], sz[P_NUM], npar;
    /* échelles d'activation (fp32) */
    float S_emb;
    float *S_res;                   /* [L+1] une échelle de résidu PAR COUCHE : le
                                     * résidu va de ±5 (embedding) à ±250 (dernière
                                     * couche) ; une échelle unique gaspillerait
                                     * 4 bits au début et saturerait la fin */
    float *S_ln1, *S_ln2, *S_final;
    float *S_g1n, *S_g2n, *S_gfn;   /* échelles des gains (max/127) */
    float *S_q, *S_k, *S_v, *S_ao, *S_f, *S_sq;
    float *sw_qkv, *sw_wo, *sw_w1, *sw_w2, *sw_out;
    /* activations fp32 (exactes en mode FP32, déquantifiées en mode QAT) */
    float *X, *N1, *N2, *QKV, *AO, *ATTN, *F1, *FSQ, *NF, *LOGITS, *dL;
    float *XNN;   /* entrée de la norme 2 : résidu APRÈS l'attention, AVANT le FFN */
    int32_t *log32;                 /* logits entiers du moteur (mode QAT) */
    float *fmax;                    /* [TI_S_SITES] maxima mesurés sur le graphe fp32 */
    int measure_fp;                 /* 1 = accumuler les maxima fp32 (calibration) */
    float *Weff;                    /* poids ternaires déquantifiés (4 zones, voir ti_fwd_fp32) */
    int qat;
    long sat_count, sat_total;      /* saturations int8 (fraction de ±127) */
    long grd_bad;                   /* gradients non finis neutralisés (dernier pas) */
    float sat_frac[TI_S_SITES];     /* par site (mesure, pas estimation) */
    int sat_worst;                  /* site le plus saturé… */
    float sat_worst_frac;           /* …et sa fraction */
    float lr, b1, b2, eps_;
    long step;
} TiModel;

/* ------------------------------------------------------------- utilitaires -- */
static long ti_tsize(const TiModel *m, int id) {
    const long D = m->D, L = m->L, F = m->F, V = m->V, T = m->T;
    switch (id) {
        case P_EMB:  return (long)V * D;
        case P_POS:  return (long)T * D;
        case P_G1:   return L * D;
        case P_G2:   return L * D;
        case P_GF:   return D;
        case P_WQKV: return L * 3 * D * D;
        case P_WO:   return L * D * D;
        case P_W1:   return L * F * D;
        case P_W2:   return L * D * F;
        case P_WOUT: return (long)V * D;
    }
    return 0;
}
static inline float *ti_p(TiModel *m, int id, long add) { return m->par + m->off[id] + add; }
static inline float *ti_gw(TiModel *m, int id, long add) { return m->grd + m->off[id] + add; }

static void *ti_fa(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "ti_model: OOM (%zu o)\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

/* Maximum mesuré sur le graphe flottant. La calibration s'appuie UNIQUEMENT
 * sur ces mesures : elles ne dépendent pas des échelles courantes, donc aucune
 * rétroaction échelle → activations → échelle (celle-ci divergeait : mesuré
 * S_res 0.008 → 2.4e7 en huit passes). */
#define TI_FMAX(m, site, v) do { if ((m)->measure_fp) { \
        const float _a = fabsf(v); if (_a > (m)->fmax[site]) (m)->fmax[site] = _a; } } while (0)

/* ------------------------------------------------------- noyaux linéaires ----
 * Convention : W est [N][K] (rangées de sortie, K contigu).
 *   ti_lin_fwd   Y[n][N]  = X[n][K]·Wᵀ
 *   ti_lin_dw    dW[N][K] += Y[n][N]ᵀ·X[n][K]   (paquets de jetons)
 *   ti_lin_dx    dX[n][K] += Y[n][N]·W           (paquets de jetons)
 * Les paquets de jetons gardent W (et le bloc dX) en cache L1 : mesuré dans
 * ti_main --bench (débit ×5 à ×8 contre la forme naïve).
 */
static void ti_lin_fwd(const float *__restrict X, const float *__restrict W,
                       float *__restrict Y, int n, int K, int N, int add) {
    #pragma omp parallel for if(n >= 64) schedule(static)
    for (int i = 0; i < n; i++) {
        const float *x = X + (size_t)i * K;
        float *y = Y + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const float *w = W + (size_t)j * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += x[k] * w[k];
            y[j] = add ? y[j] + s : s;
        }
    }
}

static void ti_dequant_eff(const float *sw, const int8_t *w8, float *Weff, int N, int K) {
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++) Weff[(size_t)j * K + k] = sw[j] * (float)w8[(size_t)j * K + k];
}

/* Poids EFFECTIFS d'une matrice quantifiée : sw·w8 — exactement ce que le
 * moteur multiplie, à l'échelle d'activation près. En mode QAT la
 * rétropropagation DOIT utiliser ces poids-là, pas les maîtres fp32 : le
 * gradient calculé avec les maîtres (magnitude comparable mais signes
 * différents) ne correspond pas au forward réellement exécuté. */
static const float *ti_weff(TiModel *m, int id, long l, int N, int K) {
    const float *sw; const int8_t *w8;
    switch (id) {
        case P_WQKV: sw = m->sw_qkv + (long)l * 3 * m->D; w8 = m->e.wqkv + (size_t)l * 3 * m->D * m->D; break;
        case P_WO:   sw = m->sw_wo  + (long)l * m->D;     w8 = m->e.wo   + (size_t)l * m->D * m->D;     break;
        case P_W1:   sw = m->sw_w1  + (long)l * m->F;     w8 = m->e.w1   + (size_t)l * m->F * m->D;     break;
        case P_W2:   sw = m->sw_w2  + (long)l * m->D;     w8 = m->e.w2   + (size_t)l * m->D * m->F;     break;
        default:     sw = m->sw_out;                      w8 = m->e.wout;                               break;
    }
    ti_dequant_eff(sw, w8, m->Weff, N, K);
    return m->Weff;
}

static void ti_transpose(float *__restrict Yt, const float *__restrict Y, int n, int N) {
    for (int j = 0; j < N; j++) {
        float *row = Yt + (size_t)j * n;
        for (int i = 0; i < n; i++) row[i] = Y[(size_t)i * N + j];
    }
}

/* dW[N][K] += Y[n][N]ᵀ·X[n][K] ; sw != NULL ⇒ gradient ramené aux maîtres (×sw) */
static void ti_lin_dw(const float *__restrict X, const float *__restrict Y,
                      float *__restrict dW, int n, int K, int N,
                      const float *__restrict sw) {
    float *Yt = (float *)ti_fa(sizeof(float) * (size_t)N * n);
    ti_transpose(Yt, Y, n, N);
    #pragma omp parallel for if(N >= 64) schedule(static)
    for (int j = 0; j < N; j++) {
        float *dw = dW + (size_t)j * K;
        const float *yt = Yt + (size_t)j * n;
        const float sc = sw ? sw[j] : 1.0f;
        for (int i0 = 0; i0 < n; i0 += TI_TB) {
            const int i1 = (i0 + TI_TB < n) ? i0 + TI_TB : n;
            for (int i = i0; i < i1; i++) {
                const float gy = yt[i] * sc;
                if (gy == 0.0f) continue;
                const float *x = X + (size_t)i * K;
                for (int k = 0; k < K; k++) dw[k] += gy * x[k];
            }
        }
    }
    free(Yt);
}

static void ti_lin_dx(const float *__restrict Y, const float *__restrict W,
                      float *__restrict dX, int n, int K, int N, int zero) {
    if (zero) memset(dX, 0, sizeof(float) * (size_t)n * K);
    const int TB = TI_TB;
    #pragma omp parallel for if(n >= 256) schedule(static)
    for (int i0 = 0; i0 < n; i0 += TB) {
        const int i1 = (i0 + TB < n) ? i0 + TB : n;
        for (int j = 0; j < N; j++) {
            const float *w = W + (size_t)j * K;
            for (int i = i0; i < i1; i++) {
                const float gy = Y[(size_t)i * N + j];
                if (gy == 0.0f) continue;
                float *dx = dX + (size_t)i * K;
                for (int k = 0; k < K; k++) dx[k] += gy * w[k];
            }
        }
    }
}

/* RMSNorm flottante exacte (référence) */
static void ti_norm_fwd(const float *x, const float *g, float *y, int n) {
    double s2 = 0.0;
    for (int i = 0; i < n; i++) s2 += (double)x[i] * x[i];
    const float inv = (float)(1.0 / sqrt(s2 / n + 1e-6));
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * g[i];
}

/* Rétroprop d'une RMSNorm (x, g, dy) → dx (écrit) et dg (accumulé).
 *   dx_j = dy_j·g_j·inv − inv³·x_j·(Σ_i dy_i·x_i·g_i)/n ,  inv = (mean x²+ε)^-½ */
static void ti_norm_bwd(const float *x, const float *g, const float *dy,
                        float *dx, float *dg, int n) {
    double s2 = 0.0, sxy = 0.0;
    for (int i = 0; i < n; i++) s2 += (double)x[i] * x[i];
    const float inv = (float)(1.0 / sqrt(s2 / n + 1e-6));
    for (int i = 0; i < n; i++) sxy += (double)dy[i] * (double)x[i] * (double)g[i];
    const float c = (float)(inv * inv * inv * (sxy / n));
    for (int i = 0; i < n; i++) {          /* lisible même si dx == dy (en place) */
        const float d = dy[i];
        dg[i] += d * x[i] * inv;
        dx[i] = d * g[i] * inv - c * x[i];
    }
}

/* ------------------------------------------------------------------- init -- */
static void ti_model_init(TiModel *m, int B, uint64_t seed) {
    memset(m, 0, sizeof(*m));
    m->T = TI_T; m->D = TI_D; m->L = TI_L; m->H = TI_H; m->HD = TI_HD;
    m->F = TI_F; m->V = TI_V; m->B = B; m->ntok = B * TI_T;
    ti_int_alloc(&m->e, B);
    const int L = m->L, D = m->D, F = m->F, V = m->V;
    long o = 0;
    for (int id = 0; id < P_NUM; id++) { m->off[id] = o; m->sz[id] = ti_tsize(m, id); o += m->sz[id]; }
    m->npar = o;
    m->par  = (float *)ti_fa(sizeof(float) * m->npar);
    m->grd  = (float *)ti_fa(sizeof(float) * m->npar);
    m->adm1 = (float *)ti_fa(sizeof(float) * m->npar);
    m->adm2 = (float *)ti_fa(sizeof(float) * m->npar);
    uint64_t s = seed ? seed : 88172645463325252ull;
    for (long i = 0; i < m->npar; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const float u1 = (float)((s >> 11) % 1000000u) * 1e-6f + 1e-7f;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const float u2 = (float)((s >> 11) % 1000000u) * 1e-6f;
        m->par[i] = 0.02f * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    for (int i = 0; i < L * D; i++) { m->par[m->off[P_G1] + i] = 1.0f; m->par[m->off[P_G2] + i] = 1.0f; }
    for (int i = 0; i < D; i++) m->par[m->off[P_GF] + i] = 1.0f;
    const size_t BT = (size_t)m->ntok;
    m->X     = (float *)ti_fa(sizeof(float) * BT * D * (L + 1));
    m->N1    = (float *)ti_fa(sizeof(float) * BT * D * L);
    m->N2    = (float *)ti_fa(sizeof(float) * BT * D * L);
    m->QKV   = (float *)ti_fa(sizeof(float) * BT * 3 * D * L);
    m->AO    = (float *)ti_fa(sizeof(float) * BT * D * L);
    m->ATTN  = (float *)ti_fa(sizeof(float) * (size_t)B * L * TI_H * TI_T * TI_T);
    m->XNN   = (float *)ti_fa(sizeof(float) * BT * D * L);
    m->F1    = (float *)ti_fa(sizeof(float) * BT * F * L);
    m->FSQ   = (float *)ti_fa(sizeof(float) * BT * F * L);
    m->NF    = (float *)ti_fa(sizeof(float) * BT * D);
    m->LOGITS= (float *)ti_fa(sizeof(float) * BT * V);
    m->log32 = (int32_t *)ti_fa(sizeof(int32_t) * BT * V);
    m->dL    = (float *)ti_fa(sizeof(float) * BT * V);
    m->fmax  = (float *)ti_fa(sizeof(float) * TI_S_SITES);
    {   /* quatre zones : Wqkv, Wo, W1, W2 sont vivants simultanément dans la
         * boucle par jeton ; Wout réutilise la zone de Wqkv (libre alors) */
        size_t nb = (size_t)3 * D * D + (size_t)D * D + 2 * (size_t)F * D;
        if (nb < (size_t)V * D) nb = (size_t)V * D;
        m->Weff = (float *)ti_fa(sizeof(float) * nb);
    }
    m->S_emb = 1.0f;
    m->S_res = (float *)ti_fa(sizeof(float) * (L + 1));
    for (int l = 0; l <= L; l++) m->S_res[l] = 1.0f;
    m->S_g1n = (float *)ti_fa(sizeof(float) * L); m->S_g2n = (float *)ti_fa(sizeof(float) * L);
    m->S_gfn = (float *)ti_fa(sizeof(float));
    m->S_ln1 = (float *)ti_fa(sizeof(float) * L); m->S_ln2 = (float *)ti_fa(sizeof(float) * L);
    m->S_final = (float *)ti_fa(sizeof(float) * L);
    m->S_q = (float *)ti_fa(sizeof(float) * L); m->S_k = (float *)ti_fa(sizeof(float) * L);
    m->S_v = (float *)ti_fa(sizeof(float) * L);
    m->S_ao = (float *)ti_fa(sizeof(float) * L); m->S_f = (float *)ti_fa(sizeof(float) * L);
    m->S_sq = (float *)ti_fa(sizeof(float) * L);
    for (int l = 0; l < L; l++) {
        m->S_g1n[l] = m->S_g2n[l] = 1.0f / 127.0f;
        m->S_ln1[l] = m->S_ln2[l] = 2.0f;
        m->S_q[l] = m->S_k[l] = m->S_v[l] = 2.0f; m->S_ao[l] = 2.0f;
        m->S_f[l] = 4.0f; m->S_sq[l] = 2.0f;
    }
    m->S_gfn[0] = 1.0f / 127.0f;
    for (int l = 0; l < L; l++) m->S_final[l] = 2.0f;
    m->sw_qkv = (float *)ti_fa(sizeof(float) * L * 3 * D);
    m->sw_wo  = (float *)ti_fa(sizeof(float) * L * D);
    m->sw_w1  = (float *)ti_fa(sizeof(float) * L * F);
    m->sw_w2  = (float *)ti_fa(sizeof(float) * L * D);
    m->sw_out = (float *)ti_fa(sizeof(float) * V);
    for (long i = 0; i < (long)L * 3 * D; i++) m->sw_qkv[i] = 1.0f;
    for (long i = 0; i < (long)L * D; i++) { m->sw_wo[i] = 1.0f; m->sw_w2[i] = 1.0f; }
    for (long i = 0; i < (long)L * F; i++) m->sw_w1[i] = 1.0f;
    for (long i = 0; i < (long)V; i++) m->sw_out[i] = 1.0f;
    /* (sw_* ne sont que des tampons : ti_quant_ternary les remplit à l'export) */
    m->qat = 1; m->lr = 3e-3f; m->b1 = 0.9f; m->b2 = 0.95f; m->eps_ = 1e-8f;
}

static void ti_model_free(TiModel *m) {
    free(m->par); free(m->grd); free(m->adm1); free(m->adm2);
    free(m->S_res); free(m->S_g1n); free(m->S_g2n); free(m->S_gfn);
    free(m->S_ln1); free(m->S_ln2); free(m->S_final);
    free(m->S_q); free(m->S_k); free(m->S_v); free(m->S_ao); free(m->S_f); free(m->S_sq);
    free(m->sw_qkv); free(m->sw_wo); free(m->sw_w1); free(m->sw_w2); free(m->sw_out);
    free(m->X); free(m->N1); free(m->N2); free(m->QKV); free(m->AO); free(m->ATTN); free(m->XNN);
    free(m->F1); free(m->FSQ); free(m->NF); free(m->LOGITS); free(m->log32); free(m->dL);
    free(m->fmax); free(m->Weff);
    ti_int_free(&m->e);
}

/* ------------------------------------------------------------- quantifieurs -- */
static inline int32_t ti_q8(float v, float inv_s) {
    int32_t q = (int32_t)lrintf(v * inv_s);
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}
/* Ternarisation : le poids reconstruit vaut  s_w[n]·w8  avec w8 ∈ {−1,0,+1}.
 * s_w est donc la MAGNITUDE d'un poids ±1, c'est-à-dire mean|W[n,:]| — la
 * meilleure échelle au sens des moindres carrés pour un support donné. Le
 * produit sw·w8 est ensuite directement le multiplicateur des requantisations
 * d'export… ce qu'il n'était pas : la version antérieure stockait 1/mean|W|
 * comme s'il s'agissait de la magnitude, ce qui donnait des poids effectifs
 * ±1/mean|W| = ±62,5 au lieu de ±0,016 (facteur 3 900 sur TOUTES les
 * activations — la perte d'initialisation passait de 5,6 à 2071 nats). */
static void ti_quant_ternary(const float *W, float *s_w, int8_t *w8, int N, int K) {
    for (int n = 0; n < N; n++) {
        const float *w = W + (size_t)n * K;
        double sum = 0.0;
        for (int k = 0; k < K; k++) sum += (double)fabsf(w[k]);
        const float mean = (float)(sum / K);
        const float s = (mean > 1e-9f) ? mean : 0.0f;
        s_w[n] = s;
        int8_t *q = w8 + (size_t)n * K;
        if (s == 0.0f) { memset(q, 0, K); continue; }
        for (int k = 0; k < K; k++) {
            int32_t v = (int32_t)lrintf(w[k] / s);
            if (v > 1) v = 1; else if (v < -1) v = -1;
            q[k] = (int8_t)v;
        }
    }
}

/* multiplicateur entier (M, sh) : y = (acc·M + 2^(sh−1)) >> sh ≈ acc·ratio,
 * M ≤ 2^14 ⇒ |acc| ≤ 2^16 donne un produit < 2^31 (débordement exclu). */
static TiRq ti_mult(float ratio) {
    /* Recherche du décalage PAR LE HAUT puis par le bas, en double : chercher
     * d'abord avec sh = 24 débordait l'int32 dès que ratio > 128 (ratio·2^24 >
     * 2^31), ce qui rendait rq_sq = S_f²/S_sq — un ratio ~6000 — purement
     * aléatoire. Borne : ratio ≤ 2^14 et sh ≥ 1 ⇒ ratio·2^sh ≤ 2^15. */
    if (!(ratio > 0.0f)) return (TiRq){ 0, 1 };
    if (ratio > 16383.0f) ratio = 16383.0f;
    int sh = 14;
    double m = (double)ratio * (double)(1 << sh);
    while (m > 16384.0 && sh > 1) { sh--; m = (double)ratio * (double)(1 << sh); }
    while (m < 4096.0 && sh < 28) { sh++; m = (double)ratio * (double)(1 << sh); }
    int32_t M = (int32_t)(m + 0.5);
    if (M > (1 << 14)) M = 1 << 14;
    if (M < 1) M = 1;
    return (TiRq){ M, sh };
}

/* ----------------------------------------------- export vers le moteur entier */
static void ti_export(TiModel *m) {
    TiInt *e = &m->e;
    const int L = m->L, D = m->D, F = m->F, V = m->V, T = m->T, H = m->H, HD = m->HD;
    {
        float mx = 1e-9f;
        const float *emb = ti_p(m, P_EMB, 0), *pos = ti_p(m, P_POS, 0);
        for (long i = 0; i < (long)V * D; i++) { const float a = fabsf(emb[i]); if (a > mx) mx = a; }
        for (long i = 0; i < (long)T * D; i++) { const float a = fabsf(pos[i]); if (a > mx) mx = a; }
        m->S_emb = mx / 127.0f;
        for (long i = 0; i < (long)V * D; i++) e->emb[i] = (int8_t)ti_q8(emb[i], 1.0f / m->S_emb);
        for (long i = 0; i < (long)T * D; i++) e->pos[i] = (int8_t)ti_q8(pos[i], 1.0f / m->S_emb);
        e->rq_emb = ti_mult(m->S_emb / m->S_res[0]);
        for (int l = 0; l < L; l++)
            e->rq_res[l] = ti_mult(m->S_res[l] / m->S_res[l + 1]);
    }
    for (int l = 0; l < L; l++) {
        const float *g1 = ti_p(m, P_G1, (long)l * D), *g2 = ti_p(m, P_G2, (long)l * D);
        for (int i = 0; i < D; i++) {
            e->g1[l * D + i] = (int8_t)ti_q8(g1[i], 1.0f / m->S_g1n[l]);
            e->g2[l * D + i] = (int8_t)ti_q8(g2[i], 1.0f / m->S_g2n[l]);
        }
        e->rq_g1[l] = ti_mult(m->S_g1n[l] / m->S_ln1[l]);
        e->rq_g2[l] = ti_mult(m->S_g2n[l] / m->S_ln2[l]);
    }
    {
        const float *gf = ti_p(m, P_GF, 0);
        for (int i = 0; i < D; i++) e->gf[i] = (int8_t)ti_q8(gf[i], 1.0f / m->S_gfn[0]);
        e->rq_gf = ti_mult(m->S_gfn[0] / m->S_final[0]);
    }
    for (int l = 0; l < L; l++) {
        ti_quant_ternary(ti_p(m, P_WQKV, (long)l * 3 * D * D), m->sw_qkv + (long)l * 3 * D,
                         e->wqkv + (size_t)l * 3 * D * D, 3 * D, D);
        for (int c = 0; c < 3 * D; c++) {
            const float So = (c < D) ? m->S_q[l] : (c < 2 * D ? m->S_k[l] : m->S_v[l]);
            e->rq_qkv[(long)l * 3 * D + c] = ti_mult(m->sw_qkv[(long)l * 3 * D + c] * m->S_ln1[l] / So);
        }
        ti_quant_ternary(ti_p(m, P_WO, (long)l * D * D), m->sw_wo + (long)l * D,
                         e->wo + (size_t)l * D * D, D, D);
        for (int c = 0; c < D; c++)
            e->rq_wo[(long)l * D + c] = ti_mult(m->sw_wo[(long)l * D + c] * m->S_ao[l] / m->S_res[l + 1]);
        ti_quant_ternary(ti_p(m, P_W1, (long)l * F * D), m->sw_w1 + (long)l * F,
                         e->w1 + (size_t)l * F * D, F, D);
        for (int c = 0; c < F; c++)
            e->rq_w1[(long)l * F + c] = ti_mult(m->sw_w1[(long)l * F + c] * m->S_ln2[l] / m->S_f[l]);
        for (int c = 0; c < F; c++)
            e->rq_sq[(long)l * F + c] = ti_mult(m->S_f[l] * m->S_f[l] / m->S_sq[l]);
        ti_quant_ternary(ti_p(m, P_W2, (long)l * D * F), m->sw_w2 + (long)l * D,
                         e->w2 + (size_t)l * D * F, D, F);
        for (int c = 0; c < D; c++)
            e->rq_w2[(long)l * D + c] = ti_mult(m->sw_w2[(long)l * D + c] * m->S_sq[l] / m->S_res[l + 1]);
        for (int h = 0; h < H; h++) {
            /* Les scores du moteur sont en LOGARITHME NÉPÉRIEN : la softmax
             * entière fait la conversion elle-même (t = d·5909 = d·log2e·2^12,
             * l'argument de 2^x en Q12 attendu par ti_exp2_q15). Multiplier ici
             * par log2e·2^12 refroidissait la softmax d'un facteur 5 909 : le
             * moteur saturait l'attention en argmax (mesuré : AO 481 % d'écart
             * RMS avec la référence, résidu 118 % dès la première couche). */
            const float ratio = m->S_q[l] * m->S_k[l] / sqrtf((float)HD);
            e->rq_score[l][h] = ti_mult(ratio);
        }
        const TiRq r = ti_mult(m->S_v[l] / 127.0f / m->S_ao[l]);
        for (int c = 0; c < D; c++) e->rq_ao[l * D + c] = r;
    }
    ti_quant_ternary(ti_p(m, P_WOUT, 0), m->sw_out, e->wout, V, D);
}

/* ---------------------------------------------------------- calibration ------
 * Le MOTEUR ENTIER mesure les maxima |q8| de chaque site (TI_TRACK) ; les
 * échelles sont ensuite resserrées à max/127 (marge 1.05). Rien n'est supposé.
 */
/* déclarations anticipées : la calibration mesure sur le graphe fp32 */
static void ti_fwd_fp32(TiModel *m, const int *ids, int n, int eff);
static void ti_count_saturation(TiModel *m);
static const char *ti_site_name(int site, char *buf, size_t nb, const TiModel *m);

static void ti_calibrate(TiModel *m, const unsigned char *data, long len, int nbatch, int passes) {
    /* La mesure se fait sur le graphe FLOTTANT avec les poids ternaires
     * effectifs : elle ne dépend donc pas des échelles courantes. La mesurer
     * dans le moteur créait une rétroaction échelle → activations → échelle qui
     * divergeait (mesuré : S_res 0,008 → 2,4e7 en huit passes, 78 % du résidu
     * écrêté). La calibration est ici un POINT FIXE en une passe : la rejouer
     * avec le même corpus redonne exactement les mêmes nombres.
     * La passe moteur finale ne calibre rien : elle relève la saturation réelle.
     */
    const int T = m->T, B = m->B, D = m->D, L = m->L, V = m->V;
    /* Marge de 15 % : entre deux recalibrations les activations croissent (les
     * maîtres grandissent sous Adam) ; sans marge la saturation atteignait
     * 99,8 % sur Q(l=2) — mesuré — c'est-à-dire un signal détruit. */
    const float margin = 1.15f;
    (void)passes;
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m->ntok);
    uint64_t rs = 0x5DEECE66Dull;
    ti_export(m);                       /* sw, w8 : nécessaires à la déquantification */
    /* 1) embeddings : borne EXACTE sur toutes les paires (mot, position) — la
     * somme emb8+pos8 dépassait 127 (mesuré 183) si la table gardait sa propre
     * échelle ; ici l'embedding et le résidu d'entrée partagent l'échelle. */
    {
        float mx = 1e-12f;
        const float *emb = ti_p(m, P_EMB, 0), *pos = ti_p(m, P_POS, 0);
        for (int v = 0; v < V; v++) {
            const float *e = emb + (size_t)v * D;
            for (int t = 0; t < T; t++) {
                const float *p = pos + (size_t)t * D;
                for (int d = 0; d < D; d++) { const float a = fabsf(e[d] + p[d]); if (a > mx) mx = a; }
            }
        }
        m->S_emb = mx * 1.001f / 127.0f;
        m->S_res[0] = m->S_emb;
    }
    /* 2) mesure des autres sites sur le graphe fp32 (poids ternaires effectifs) */
    for (int i = 0; i < TI_S_SITES; i++) m->fmax[i] = 1e-12f;
    m->measure_fp = 1;
    const long maxpos = len - T - 2;
    for (int nb = 0; nb < nbatch; nb++) {
        for (int b = 0; b < B; b++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            const long pos = (long)((rs >> 11) % (uint64_t)maxpos);
            for (int t = 0; t < T; t++) ids[(size_t)b * T + t] = data[pos + t];
        }
        ti_fwd_fp32(m, ids, T, 1);
    }
    m->measure_fp = 0;
    /* 3) conversion en échelles : le maximum mesuré occupe 127·(1/1,05) */
    /* marge plus large sur le résidu : il somme DEUX termes eux-mêmes quantifiés
     * (résidu entrant remis à l'échelle + delta), donc son maximum peut
     * légèrement dépasser le maximum flottant mesuré */
    const float rmargin = 1.30f;
    for (int l = 1; l <= L; l++) {
        const float v = m->fmax[TI_S_RES(l)] * rmargin / 127.0f;
        if (v > 1e-12f) m->S_res[l] = v;
    }
    for (int l = 0; l < L; l++) {
        const float v[8] = {
            m->fmax[TI_S_LN1(l)] * margin / 127.0f, m->fmax[TI_S_LN2(l)] * margin / 127.0f,
            m->fmax[TI_S_Q(l)]   * margin / 127.0f, m->fmax[TI_S_K(l)]   * margin / 127.0f,
            m->fmax[TI_S_V(l)]   * margin / 127.0f, m->fmax[TI_S_AO(l)]  * margin / 127.0f,
            m->fmax[TI_S_F(l)]   * margin / 127.0f, m->fmax[TI_S_SQ(l)]  * margin / 127.0f
        };
        if (v[0] > 1e-12f) m->S_ln1[l] = v[0];
        if (v[1] > 1e-12f) m->S_ln2[l] = v[1];
        if (v[2] > 1e-12f) m->S_q[l]   = v[2];
        if (v[3] > 1e-12f) m->S_k[l]   = v[3];
        if (v[4] > 1e-12f) m->S_v[l]   = v[4];
        if (v[5] > 1e-12f) m->S_ao[l]  = v[5];
        if (v[6] > 1e-12f) m->S_f[l]   = v[6];
        if (v[7] > 1e-12f) m->S_sq[l]  = v[7];
    }
    if (m->fmax[TI_S_NF] > 1e-12f) m->S_final[0] = m->fmax[TI_S_NF] * margin / 127.0f;
    {   /* échelles des gains de norme (max/127) */
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
    /* 4) vérification : une passe du moteur entier, saturation relevée */
    ti_export(m);
    {
        TiInt *e = &m->e;
        e->measure = 1;
        for (int i = 0; i < TI_S_SITES; i++) e->sitemax[i] = 1;
        for (int b = 0; b < B; b++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            const long pos = (long)((rs >> 11) % (uint64_t)maxpos);
            for (int t = 0; t < T; t++) ids[(size_t)b * T + t] = data[pos + t];
        }
        ti_int_reset(e);
        ti_int_run(e, ids, B, T, 0, m->log32, 1);
        e->measure = 0;
    }
    ti_count_saturation(m);
    free(ids);
}

/* ------------------------------------------------------------ forward fp32 --
 * Graphe flottant exact : référence pour le gradcheck et l'A/B honnête.
 * (Le moteur entier n'y intervient pas.)
 */
static void ti_fwd_fp32(TiModel *m, const int *ids, int n, int eff) {
    const int D = m->D, L = m->L, H = m->H, HD = m->HD, F = m->F, V = m->V;
    const size_t BT = (size_t)m->ntok;
    const float *emb = ti_p(m, P_EMB, 0), *pos = ti_p(m, P_POS, 0);
    for (int b = 0; b < m->B; b++)
        for (int t = 0; t < n; t++) {
            const size_t r = (size_t)b * m->T + t;
            for (int d = 0; d < D; d++)
                m->X[r * D + d] = emb[(size_t)ids[r] * D + d] + pos[(size_t)t * D + d];
        }
    for (int l = 0; l <= L; l++)
        for (size_t i = 0; i < BT * (size_t)D; i++) TI_FMAX(m, TI_S_RES(l), m->X[(size_t)l * BT * D + i]);
    for (int l = 0; l < L; l++) {
        float *Xc = m->X + (size_t)l * BT * D, *Xn = m->X + (size_t)(l + 1) * BT * D;
        float *n1 = m->N1 + (size_t)l * BT * D, *n2 = m->N2 + (size_t)l * BT * D;
        float *qkv = m->QKV + (size_t)l * BT * 3 * D, *ao = m->AO + (size_t)l * BT * D;
        float *f1 = m->F1 + (size_t)l * BT * F, *fsq = m->FSQ + (size_t)l * BT * F;
        const float *g1 = ti_p(m, P_G1, (long)l * D), *g2 = ti_p(m, P_G2, (long)l * D);
        const float *Wq = ti_p(m, P_WQKV, (long)l * 3 * D * D), *Wo = ti_p(m, P_WO, (long)l * D * D);
        const float *W1 = ti_p(m, P_W1, (long)l * F * D), *W2 = ti_p(m, P_W2, (long)l * D * F);
        const float rscale = 1.0f / sqrtf((float)HD);
        if (eff) {   /* poids réellement vus par le moteur : sw·w8 (ternaire déquantifié) */
            float *z0 = m->Weff, *z1 = z0 + (size_t)3 * D * D;
            float *z2 = z1 + (size_t)D * D, *z3 = z2 + (size_t)F * D;
            ti_dequant_eff(m->sw_qkv + (long)l * 3 * D, m->e.wqkv + (size_t)l * 3 * D * D, z0, 3 * D, D);
            ti_dequant_eff(m->sw_wo + (long)l * D, m->e.wo + (size_t)l * D * D, z1, D, D);
            ti_dequant_eff(m->sw_w1 + (long)l * F, m->e.w1 + (size_t)l * F * D, z2, F, D);
            ti_dequant_eff(m->sw_w2 + (long)l * D, m->e.w2 + (size_t)l * D * F, z3, D, F);
            Wq = z0; Wo = z1; W1 = z2; W2 = z3;
        }
        for (int b = 0; b < m->B; b++)
        for (int t = 0; t < n; t++) {
            const size_t r = (size_t)b * m->T + t;
            ti_norm_fwd(Xc + r * D, g1, n1 + r * D, D);
            for (int d = 0; d < D; d++) TI_FMAX(m, TI_S_LN1(l), n1[r * D + d]);
            ti_lin_fwd(n1 + r * D, Wq, qkv + r * 3 * D, 1, D, 3 * D, 0);
            for (int c = 0; c < 3 * D; c++)
                TI_FMAX(m, c < D ? TI_S_Q(l) : (c < 2 * D ? TI_S_K(l) : TI_S_V(l)), qkv[r * 3 * D + c]);
            for (int h = 0; h < H; h++) {
                const float *q = qkv + r * 3 * D + h * HD;
                float *a = m->ATTN + ((((size_t)l * m->B) + b) * H + h) * TI_T * TI_T;
                double mx = -1e30;
                for (int u = 0; u <= t; u++) {
                    const float *k = qkv + ((size_t)b * m->T + u) * 3 * D + D + h * HD;
                    double s = 0.0;
                    for (int d = 0; d < HD; d++) s += (double)q[d] * k[d];
                    s *= rscale;
                    a[(size_t)t * TI_T + u] = (float)s;
                    if (s > mx) mx = s;
                }
                double se = 0.0;
                for (int u = 0; u <= t; u++) {
                    const double e = exp((double)a[(size_t)t * TI_T + u] - mx);
                    a[(size_t)t * TI_T + u] = (float)e;
                    se += e;
                }
                for (int u = 0; u <= t; u++) a[(size_t)t * TI_T + u] = (float)(a[(size_t)t * TI_T + u] / se);
                float *o = ao + r * D + h * HD;
                for (int d = 0; d < HD; d++) {
                    double s = 0.0;
                    for (int u = 0; u <= t; u++)
                        s += (double)a[(size_t)t * TI_T + u] * qkv[((size_t)b * m->T + u) * 3 * D + 2 * D + h * HD + d];
                    o[d] = (float)s;
                }
                for (int d = 0; d < HD; d++) TI_FMAX(m, TI_S_AO(l), o[d]);
            }
            ti_lin_fwd(ao + r * D, Wo, Xn + r * D, 1, D, D, 0);
            for (int d = 0; d < D; d++) Xn[r * D + d] += Xc[r * D + d];
            /* l'étape intermédiaire du résidu est ELLE AUSSI écrite en int8 : sa
             * magnitude peut dépasser celle du résidu final, et c'est elle qui
             * fixe la dynamique nécessaire à l'échelle S_res[l+1] */
            for (int d = 0; d < D; d++) TI_FMAX(m, TI_S_RES(l + 1), Xn[r * D + d]);
            /* la norme 2 voit ce résidu-ci, pas le résidu final */
            memcpy(m->XNN + (size_t)l * BT * D + r * D, Xn + r * D, sizeof(float) * D);
            ti_norm_fwd(Xn + r * D, g2, n2 + r * D, D);
            for (int d = 0; d < D; d++) TI_FMAX(m, TI_S_LN2(l), n2[r * D + d]);
            ti_lin_fwd(n2 + r * D, W1, f1 + r * F, 1, D, F, 0);
            for (int j = 0; j < F; j++) {
                const float v = f1[r * F + j];
                fsq[r * F + j] = (v > 0.0f) ? v * v : 0.0f;
                if (v > 0.0f) { TI_FMAX(m, TI_S_F(l), v); TI_FMAX(m, TI_S_SQ(l), v * v); }
            }
            /* ACCUMULATION (et non écriture) : la contribution résiduelle de
             * l'attention doit survivre au second résidu */
            ti_lin_fwd(fsq + r * F, W2, Xn + r * D, 1, F, D, 1);
            for (int d = 0; d < D; d++) TI_FMAX(m, TI_S_RES(l + 1), Xn[r * D + d]);
        }
    }
    const float *Wout = ti_p(m, P_WOUT, 0);
    if (eff) {
        ti_dequant_eff(m->sw_out, m->e.wout, m->Weff, V, D);
        Wout = m->Weff;
    }
    for (int b = 0; b < m->B; b++)
        for (int t = 0; t < n; t++) {
            const size_t r = (size_t)b * m->T + t;
            ti_norm_fwd(m->X + (size_t)L * BT * D + r * D, ti_p(m, P_GF, 0), m->NF + r * D, D);
            ti_lin_fwd(m->NF + r * D, Wout, m->LOGITS + r * V, 1, D, V, 0);
            TI_FMAX(m, TI_S_NF, 0.0f);
            for (int d = 0; d < D; d++) TI_FMAX(m, TI_S_NF, m->NF[r * D + d]);
        }
}

/* --------------------------------------------------------------- forward QAT */
static void ti_fwd_qat(TiModel *m, const int *ids, int n) {
    /* Forward en quantification simulée : TOUTES les activations et TOUS les
     * poids sont ceux du moteur entier (ti_int_run) — le gradient voit donc
     * exactement les valeurs que l'inférence utilisera. Rien n'est recalculé
     * en flottant : on ne fait que déquantifier les tampons int8 du moteur. */
    TiInt *e = &m->e;
    const int D = m->D, L = m->L, F = m->F, V = m->V, H = m->H, T = m->T;
    const size_t BT = (size_t)m->ntok;
    ti_int_reset(e);
    ti_int_run(e, ids, m->B, n, 0, m->log32, 2);
    for (int l = 0; l <= L; l++) {
        const int8_t *src = e->xq + (size_t)l * BT * D;
        float *dst = m->X + (size_t)l * BT * D;
        for (size_t i = 0; i < BT * D; i++) dst[i] = (float)src[i] * m->S_res[l];
    }
    for (int l = 0; l < L; l++) {
        const int8_t *ln = e->lnq + (size_t)l * BT * D;      /* N1 (conservé) */
        const int8_t *n2 = e->n2q + (size_t)l * BT * D;      /* N2 */
        for (size_t i = 0; i < BT * D; i++) {
            m->N1[(size_t)l * BT * D + i] = (float)ln[i] * m->S_ln1[l];
            m->N2[(size_t)l * BT * D + i] = (float)n2[i] * m->S_ln2[l];
        }
        const int8_t *qkv = e->qkv + (size_t)l * BT * 3 * D;
        float *dq = m->QKV + (size_t)l * BT * 3 * D;
        for (size_t i = 0; i < BT; i++)
            for (int c = 0; c < 3 * D; c++) {
                const float S = (c < D) ? m->S_q[l] : (c < 2 * D ? m->S_k[l] : m->S_v[l]);
                dq[i * 3 * D + c] = (float)qkv[i * 3 * D + c] * S;
            }
        const int8_t *ao = e->ao + (size_t)l * BT * D;
        for (size_t i = 0; i < BT * D; i++) m->AO[(size_t)l * BT * D + i] = (float)ao[i] * m->S_ao[l];
        /* entrée de la norme 2 : résidu reconstruit X[l] + Wo·AO (le moteur
         * requantifie cette somme ; la rétropropagation le traite en identité) */
        ti_lin_fwd(m->AO + (size_t)l * BT * D, ti_p(m, P_WO, (long)l * D * D),
                   m->XNN + (size_t)l * BT * D, (int)BT, D, D, 0);
        {
            const float *xl = m->X + (size_t)l * BT * D;
            float *xnn = m->XNN + (size_t)l * BT * D;
            for (size_t i = 0; i < BT * D; i++) xnn[i] += xl[i];
        }
        /* FFN : le moteur ne garde que la valeur APRÈS ReLU et après mise au
         * carré ; la magnitude pré-ReLU se reconstitue exactement par
         * |f1| = sqrt(fq·S_sq) (car fq = clamp(f1²/S_sq)). */
        const int8_t *fq = e->fq + (size_t)l * BT * F;
        float *df = m->F1 + (size_t)l * BT * F, *ds = m->FSQ + (size_t)l * BT * F;
        for (size_t i = 0; i < BT * F; i++) {
            const float v = (float)fq[i] * m->S_sq[l];
            ds[i] = v;
            df[i] = sqrtf(v);
        }
        /* poids d'attention (le moteur les stocke normalisés sur 127) */
        for (int b = 0; b < m->B; b++)
            for (int h = 0; h < H; h++) {
                const int8_t *at = e->att + (((size_t)b * L + l) * H + h) * T * T;
                float *dst = m->ATTN + ((((size_t)l * m->B) + b) * H + h) * TI_T * TI_T;
                for (int t = 0; t < n; t++)
                    for (int u = 0; u <= t; u++)
                        dst[(size_t)t * TI_T + u] = (float)at[(size_t)t * T + u] * (1.0f / 127.0f);
            }
    }
    for (size_t i = 0; i < BT * D; i++)
        m->NF[i] = (float)e->lnq[(size_t)L * BT * D + i] * m->S_final[0];
    for (int b = 0; b < m->B; b++)
        for (int t = 0; t < n; t++)
            for (int v = 0; v < V; v++)
                m->LOGITS[((size_t)b * T + t) * V + v] =
                    (float)m->log32[((size_t)b * n + t) * V + v] * (m->S_final[0] * m->sw_out[v]);
}

/* --------------------------------------------------------- taux de saturation --
 * Fraction d'activations int8 qui touchent le bord de la plage (±127), c'est-à-
 * dire les valeurs que la requantification a dû écrêter. Mesuré après un forward
 * QAT, sur tous les tenseurs d'activation du moteur : si ce taux devient non
 * négligeable, la calibration est trop serrée et le modèle perd de l'information.
 */
static void ti_count_saturation(TiModel *m) {
    /* Saturation PAR SITE : c'est la mesure qui dit OÙ le modèle perd de
     * l'information (une activation qui touche ±127 est écrêtée). Un total
     * global à 60 % ne dit rien ; « RES(l=3) 97 % » dit exactement quoi
     * recalibrer. */
    TiInt *e = &m->e;
    const size_t BT = (size_t)m->ntok, D = m->D, F = m->F, L = m->L;
    for (int i = 0; i < TI_S_SITES; i++) m->sat_frac[i] = -1.0f;
    long sat = 0, tot = 0;
    #define TI_SAT(SITE, PTR, N) do { const int8_t *_p = (PTR); long _s = 0; \
        for (size_t _i = 0; _i < (size_t)(N); _i++) \
            if (_p[_i] >= 127 || _p[_i] <= -127) _s++; \
        tot += (long)(N); sat += _s; m->sat_frac[SITE] = 100.0f * (float)_s / (float)(N); } while (0)
    for (int l = 0; l <= L; l++) TI_SAT(TI_S_RES(l), e->xq + (size_t)l * BT * D, BT * D);
    TI_SAT(TI_S_NF, e->lnq + (size_t)L * BT * D, BT * D);
    for (int l = 0; l < L; l++) {
        TI_SAT(TI_S_LN1(l), e->lnq + (size_t)l * BT * D, BT * D);
        TI_SAT(TI_S_LN2(l), e->n2q + (size_t)l * BT * D, BT * D);
        TI_SAT(TI_S_Q(l), e->qkv + (size_t)l * BT * 3 * D, BT * D);
        TI_SAT(TI_S_K(l), e->qkv + (size_t)l * BT * 3 * D + D, BT * D);
        TI_SAT(TI_S_V(l), e->qkv + (size_t)l * BT * 3 * D + 2 * D, BT * D);
        TI_SAT(TI_S_AO(l), e->ao + (size_t)l * BT * D, BT * D);
        TI_SAT(TI_S_SQ(l), e->fq + (size_t)l * BT * F, BT * F);
    }
    #undef TI_SAT
    m->sat_count = sat;
    m->sat_total = tot;
    /* site le plus saturé (hors sites non mesurés, marqués −1) */
    int worst = -1;
    float wf = 0.0f;
    for (int i = 0; i < TI_S_SITES; i++)
        if (m->sat_frac[i] > wf) { wf = m->sat_frac[i]; worst = i; }
    m->sat_worst = worst;
    m->sat_worst_frac = wf;
}
static const char *ti_site_name(int site, char *buf, size_t nb, const TiModel *m) {
    if (site == TI_S_NF) { snprintf(buf, nb, "NF"); return buf; }
    if (site >= TI_S_RES(0) && site <= TI_S_RES(m->L)) { snprintf(buf, nb, "RES(l=%d)", site - TI_S_RES(0)); return buf; }
    for (int l = 0; l < m->L; l++) {
        if (site == TI_S_LN1(l)) { snprintf(buf, nb, "LN1(l=%d)", l); return buf; }
        if (site == TI_S_LN2(l)) { snprintf(buf, nb, "LN2(l=%d)", l); return buf; }
        if (site == TI_S_Q(l))   { snprintf(buf, nb, "Q(l=%d)", l);   return buf; }
        if (site == TI_S_K(l))   { snprintf(buf, nb, "K(l=%d)", l);   return buf; }
        if (site == TI_S_V(l))   { snprintf(buf, nb, "V(l=%d)", l);   return buf; }
        if (site == TI_S_AO(l))  { snprintf(buf, nb, "AO(l=%d)", l);  return buf; }
        if (site == TI_S_F(l))   { snprintf(buf, nb, "F(l=%d)", l);   return buf; }
        if (site == TI_S_SQ(l))  { snprintf(buf, nb, "SQ(l=%d)", l);  return buf; }
    }
    snprintf(buf, nb, "site %d", site);
    return buf;
}
static double ti_loss_grad(TiModel *m, const int *tgt, int n) {
    const int V = m->V, B = m->B;
    const float inv = 1.0f / (float)(B * n);
    double tot = 0.0;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < n; t++) {
            const size_t i = (size_t)b * m->T + t;
            const float *lg = m->LOGITS + i * V;
            float mx = lg[0];
            for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
            double se = 0.0;
            for (int v = 0; v < V; v++) se += exp((double)lg[v] - mx);
            const double lse = log(se) + mx;
            const int y = tgt[i];
            tot += lse - (double)lg[y];
            float *dl = m->dL + i * V;
            for (int v = 0; v < V; v++) dl[v] = (float)(exp((double)lg[v] - mx) / se) * inv;
            dl[y] -= inv;
        }
    return tot / (double)(B * n);
}

/* ------------------------------------------------ rétroprop d'une attention --
 * dAO (gradient de la sortie d'attention, [B][T][D]) → dQKV (accumulé, [B][T][3D]).
 * Dépouillée des projections : elle ne voit que les scores, la softmax causale,
 * la somme pondérée des valeurs et la somme d'attention.
 *   dV[u]  += Σ_{t≥u} p[t][u]·dO[t]
 *   dP[t][u] = Σ_d dO[t][d]·V[u][d]
 *   dS[t][u] = p[t][u]·(dP[t][u] − Σ_u' p[t][u']·dP[t][u'])
 *   dQ[t] += dS[t][u]·K[u]/√HD ; dK[u] += dS[t][u]·Q[t]/√HD
 * Écrite pour être VÉRIFIÉE séparément (--attncheck) : c'est le module où une
 * erreur d'indice passe inaperçue tout en corrompant toute la chaîne.
 */
static void ti_attn_bwd(TiModel *m, int l, const float *dAO, float *dQKV, float *dP) {
    /* Accumulateurs float et boucles contiguës : cette fonction représentait
     * 60 % de la rétropropagation tant qu'elle sommait en double (mesuré :
     * 933 ms sur 1547 ms). Les têtes sont indépendantes (colonnes disjointes de
     * dQKV) donc parallélisables ; dP doit alors être privé par fil. */
    const int D = m->D, H = m->H, HD = m->HD, B = m->B, T = m->T;
    const int n = T;
    const float rs = 1.0f / sqrtf((float)HD);
    const float *qkv = m->QKV + (size_t)l * m->ntok * 3 * D;
    memset(dQKV, 0, sizeof(float) * (size_t)m->ntok * 3 * D);
    #pragma omp parallel for if(B * H >= 6) schedule(static)
    for (int bh = 0; bh < B * H; bh++) {
        const int b = bh / H, h = bh % H;
        #ifdef _OPENMP
        float *dPl = dP + (size_t)omp_get_thread_num() * TI_T;
        #else
        float *dPl = dP;
        #endif
        const float *__restrict att = m->ATTN + ((((size_t)l * B) + b) * H + h) * TI_T * TI_T;
        float *__restrict dq = dQKV + ((size_t)b * T) * 3 * D + h * HD;
        float *__restrict dk = dq + D, *__restrict dv = dq + 2 * D;
        for (int t = 0; t < n; t++) {
            const float *__restrict p = att + (size_t)t * TI_T;
            const float *__restrict dy = dAO + ((size_t)b * T + t) * D + h * HD;
            const float *__restrict qv = qkv + ((size_t)b * T + t) * 3 * D + h * HD;
            float *__restrict dqt = dq + (size_t)t * 3 * D;
            float rowsum = 0.0f;
            for (int u = 0; u <= t; u++) {
                const float *__restrict vv = qkv + ((size_t)b * T + u) * 3 * D + 2 * D + h * HD;
                float *__restrict dvu = dv + (size_t)u * 3 * D;
                float acc = 0.0f, pu = p[u];
                for (int d = 0; d < HD; d++) {
                    acc += dy[d] * vv[d];
                    dvu[d] += pu * dy[d];
                }
                dPl[u] = acc;
                rowsum += acc * pu;
            }
            for (int u = 0; u <= t; u++) {
                const float ds = p[u] * (dPl[u] - rowsum) * rs;
                const float *__restrict kv = qkv + ((size_t)b * T + u) * 3 * D + D + h * HD;
                float *__restrict dku = dk + (size_t)u * 3 * D;
                for (int d = 0; d < HD; d++) {
                    dqt[d] += ds * kv[d];
                    dku[d] += ds * qv[d];
                }
            }
        }
    }
}

/* ---------------------------------------------------------------- backward --
 * ids : identifiants d'entrée (nécessaires au gradient des embeddings).
 * Toute la rétropropagation utilise les activations STOCKÉES par le forward
 * (exactes en mode FP32, déquantifiées en mode QAT) : l'implémentation est donc
 * la même pour les deux modes, et seule la source des activations change.
 */
static void ti_backward(TiModel *m, const int *ids, int n) {
    const int D = m->D, L = m->L, H = m->H, HD = m->HD, F = m->F, V = m->V, B = m->B;
    const size_t BT = (size_t)m->ntok;
    const float rs = 1.0f / sqrtf((float)HD);
    float *dcur = (float *)ti_fa(sizeof(float) * BT * D);
    float *dA   = (float *)ti_fa(sizeof(float) * BT * D);   /* dAO puis inutilisé */
    float *dB   = (float *)ti_fa(sizeof(float) * BT * D);   /* dN2 (norme 2) */
    float *dC   = (float *)ti_fa(sizeof(float) * BT * D);   /* dN1 (norme 1) */
    float *dN   = (float *)ti_fa(sizeof(float) * BT * (F > D ? F : D));
    float *dQKV = (float *)ti_fa(sizeof(float) * BT * 3 * D);
    /* dP privé par fil : les têtes s'exécutent en parallèle dans ti_attn_bwd */
    #ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    #else
    const int nthreads = 1;
    #endif
    float *dP   = (float *)ti_fa(sizeof(float) * TI_T * (size_t)nthreads);
    /* 1) sortie : dL → dW_out, dNF
     * Le moteur écrit logits = (Σ lnf·wout8)·S_final·sw_out : le gradient par
     * rapport au logit ENTIER vaut donc dL·S_final, et la remontée vers NF
     * utilise le poids effectif. Oublier S_final (≈ 0,011 ici) tuait la tête de
     * sortie : ||g(wout)|| mesuré 1e-4 au lieu de 9e-3, le modèle ne pouvait
     * plus apprendre par sa couche de sortie et partait en vrille. */
    {
        float *dLs = (float *)ti_fa(sizeof(float) * BT * V);
        /* en mode fp32 (gradcheck) le graphe est le graphe flottant exact :
         * aucune échelle d'activation n'intervient dans les logits */
        const float Sf = m->qat ? m->S_final[0] : 1.0f;
        for (size_t i = 0; i < BT * V; i++) dLs[i] = m->dL[i] * Sf;
        ti_lin_dw(m->NF, dLs, ti_gw(m, P_WOUT, 0), BT, D, V, m->sw_out);
        ti_lin_dx(dLs, m->qat ? ti_weff(m, P_WOUT, 0, V, D) : ti_p(m, P_WOUT, 0), dN, BT, D, V, 1);
        free(dLs);
    }
    /* 2) norme finale */
    for (int i = 0; i < m->ntok; i++)
        ti_norm_bwd(m->X + (size_t)L * BT * D + (size_t)i * D, ti_p(m, P_GF, 0),
                    dN + (size_t)i * D, dcur + (size_t)i * D, ti_gw(m, P_GF, 0), D);
    /* 3) couches */
    for (int l = L - 1; l >= 0; l--) {
        const float *Xc = m->X + (size_t)l * BT * D, *Xn = m->X + (size_t)(l + 1) * BT * D;
        const float *n1 = m->N1 + (size_t)l * BT * D, *n2 = m->N2 + (size_t)l * BT * D;
        const float *qkv = m->QKV + (size_t)l * BT * 3 * D;
        const float *ao = m->AO + (size_t)l * BT * D;
        const float *f1 = m->F1 + (size_t)l * BT * F, *fsq = m->FSQ + (size_t)l * BT * F;
        const float *Wq = ti_p(m, P_WQKV, (long)l * 3 * D * D), *Wo = ti_p(m, P_WO, (long)l * D * D);
        const float *W1 = ti_p(m, P_W1, (long)l * F * D), *W2 = ti_p(m, P_W2, (long)l * D * F);
        const float *g1 = ti_p(m, P_G1, (long)l * D), *g2 = ti_p(m, P_G2, (long)l * D);
        float *swq = m->sw_qkv + (long)l * 3 * D, *swo = m->sw_wo + (long)l * D;
        float *sw1 = m->sw_w1 + (long)l * F, *sw2 = m->sw_w2 + (long)l * D;
        /* Le bloc calcule :  Xint = X[l] + ATTN ; X[l+1] = Xint + FFN(Xint).
         * Le gradient qui remonte (dcur) est celui de X[l+1]. Il faut d'abord
         * le convertir en gradient de Xint AVANT de rétropropager l'attention :
         * l'attention et le FFN partagent ce même état intermédiaire, et son
         * gradient est dcur + dxn (dxn = rétroprop de la norme 2). */
        /* 3a) FFN : W2 puis ReLU² */
        ti_lin_dw(fsq, dcur, ti_gw(m, P_W2, (long)l * D * F), BT, F, D, sw2);
        ti_lin_dx(dcur, m->qat ? ti_weff(m, P_W2, l, D, F) : W2, dN, BT, F, D, 1);   /* dN = dFSQ */
        for (int i = 0; i < m->ntok; i++)
            for (int j = 0; j < F; j++) {
                const size_t idx = (size_t)i * F + j;
                dN[idx] = (f1[idx] > 0.0f) ? 2.0f * f1[idx] * dN[idx] : 0.0f;
            }
        /* 3b) W1 puis norme 2 (entrée = résidu intermédiaire m->XNN) */
        ti_lin_dw(n2, dN, ti_gw(m, P_W1, (long)l * F * D), BT, D, F, sw1);
        ti_lin_dx(dN, m->qat ? ti_weff(m, P_W1, l, F, D) : W1, dB, BT, D, F, 1);     /* dB = dN2 */
        {
            const float *xnn = m->XNN + (size_t)l * BT * D;
            for (int i = 0; i < m->ntok; i++)
                ti_norm_bwd(xnn + (size_t)i * D, g2, dB + (size_t)i * D, dB + (size_t)i * D,
                            ti_gw(m, P_G2, (long)l * D), D);
        }
        /* 3c) gradient total de l'état intermédiaire */
        for (size_t i = 0; i < BT * D; i++) dC[i] = dcur[i] + dB[i];
        /* 3d) Wo */
        ti_lin_dw(ao, dC, ti_gw(m, P_WO, (long)l * D * D), BT, D, D, swo);
        ti_lin_dx(dC, m->qat ? ti_weff(m, P_WO, l, D, D) : Wo, dA, BT, D, D, 1);     /* dA = dAO */
        /* 3e) attention : dAO → dQ, dK, dV (+ valeurs) */
        ti_attn_bwd(m, l, dA, dQKV, dP);
        /* 3f) Wqkv puis norme 1 */
        ti_lin_dw(n1, dQKV, ti_gw(m, P_WQKV, (long)l * 3 * D * D), BT, D, 3 * D, swq);
        ti_lin_dx(dQKV, m->qat ? ti_weff(m, P_WQKV, l, 3 * D, D) : Wq, dN, BT, D, 3 * D, 1); /* dN = dN1 */
        for (int i = 0; i < m->ntok; i++)
            ti_norm_bwd(Xc + (size_t)i * D, g1, dN + (size_t)i * D, dN + (size_t)i * D,
                        ti_gw(m, P_G1, (long)l * D), D);
        /* 3g) gradient de X[l] = (gradient de Xint) + (norme 1) */
        for (size_t i = 0; i < BT * D; i++) dB[i] = dC[i] + dN[i];
        float *tmp = dcur; dcur = dB; dB = tmp;
    }
    /* 4) embeddings + positions (lignes partagées → accumulation) */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < n; t++) {
            const size_t i = (size_t)b * m->T + t;
            float *gp = ti_gw(m, P_POS, (long)t * D);
            float *ge = ti_gw(m, P_EMB, (size_t)ids[i] * D);
            const float *dc = dcur + i * D;
            for (int d = 0; d < D; d++) { gp[d] += dc[d]; ge[d] += dc[d]; }
        }
    free(dcur); free(dA); free(dB); free(dC); free(dN); free(dQKV); free(dP);
}

/* -------------------------------------------------------------------- Adam --
 * AdamW appliqué aux maîtres fp32. Déterministe et parallélisable par élément.
 */
static void ti_adam(TiModel *m, float lr, float wd) {
    m->step++;
    const float b1 = m->b1, b2 = m->b2, e = m->eps_;
    const float c1 = 1.0f - powf(b1, (float)m->step), c2 = 1.0f - powf(b2, (float)m->step);
    float *par = m->par, *grd = m->grd, *a1 = m->adm1, *a2 = m->adm2;
    const long N = m->npar;
    long bad = 0;
    /* Garde-fou : un gradient non fini (cas réel : une norme appliquée sur un
     * résidu entièrement écrêté → 1/rms = inf) est neutralisé au lieu d'être
     * propagé aux maîtres. Sans ce filtre, un seul pas suffisait à rendre tout
     * le modèle NaN — mesuré : perte figée à 5,5452 (le hasard exact) au pas
     * 125 d'un entraînement QAT. */
    #pragma omp parallel for if(N >= 65536) schedule(static) reduction(+:bad)
    for (long i = 0; i < N; i++) {
        float g = grd[i];
        if (!isfinite(g)) { g = 0.0f; bad++; }
        const float m1 = b1 * a1[i] + (1.0f - b1) * g;
        const float m2 = b2 * a2[i] + (1.0f - b2) * g * g;
        a1[i] = m1; a2[i] = m2;
        const float upd = (m1 / c1) / (sqrtf(m2 / c2) + e) + wd * par[i];
        par[i] -= lr * upd;
    }
    m->grd_bad = bad;
    if (bad) for (long i = 0; i < N; i++) if (!isfinite(m->par[i])) m->par[i] = 0.0f;
    memset(m->grd, 0, sizeof(float) * (size_t)N);
}

/* -------------------------------------------------------------- checkpoint --
 * Sauvegarde/chargement des maîtres fp32 + échelles + état Adam (format TIF1).
 * Le fichier d'inférence (MIT1, entiers purs) est produit séparément par
 * ti_export + ti_int_save.
 */
#define TI_FMAGIC 0x31464954u   /* "TIF1" */
static int ti_model_save(const TiModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const int hdr[12] = { (int)TI_FMAGIC, m->T, m->D, m->L, m->H, m->HD, m->F, m->V, m->B,
                          (int)m->npar, (int)m->step, m->qat };
    fwrite(hdr, sizeof(int), 12, f);
    float sc[4] = { m->S_emb, m->lr, (float)(m->L + 1), 0.0f };
    fwrite(sc, sizeof(float), 4, f);
    fwrite(m->S_res, sizeof(float), (size_t)(m->L + 1), f);
    fwrite(m->par, sizeof(float), (size_t)m->npar, f);
    fwrite(m->adm1, sizeof(float), (size_t)m->npar, f);
    fwrite(m->adm2, sizeof(float), (size_t)m->npar, f);
    const int L = m->L, D = m->D, F = m->F, V = m->V;
    fwrite(m->S_g1n, sizeof(float), L, f); fwrite(m->S_g2n, sizeof(float), L, f);
    fwrite(&m->S_gfn[0], sizeof(float), 1, f);
    fwrite(m->S_ln1, sizeof(float), L, f); fwrite(m->S_ln2, sizeof(float), L, f);
    fwrite(m->S_final, sizeof(float), L, f);
    fwrite(m->S_q, sizeof(float), L, f); fwrite(m->S_k, sizeof(float), L, f);
    fwrite(m->S_v, sizeof(float), L, f); fwrite(m->S_ao, sizeof(float), L, f);
    fwrite(m->S_f, sizeof(float), L, f); fwrite(m->S_sq, sizeof(float), L, f);
    fwrite(m->sw_qkv, sizeof(float), (size_t)L * 3 * D, f);
    fwrite(m->sw_wo, sizeof(float), (size_t)L * D, f);
    fwrite(m->sw_w1, sizeof(float), (size_t)L * F, f);
    fwrite(m->sw_w2, sizeof(float), (size_t)L * D, f);
    fwrite(m->sw_out, sizeof(float), (size_t)V, f);
    fclose(f);
    return 0;
}
static int ti_model_load(TiModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int hdr[12];
    if (fread(hdr, sizeof(int), 12, f) != 12 || (unsigned)hdr[0] != TI_FMAGIC) { fclose(f); return -2; }
    if (hdr[1] != m->T || hdr[2] != m->D || hdr[3] != m->L || hdr[4] != m->H ||
        hdr[5] != m->HD || hdr[6] != m->F || hdr[7] != m->V || hdr[8] != m->B ||
        hdr[9] != (int)m->npar) { fclose(f); return -3; }
    float sc[4];
    if (fread(sc, sizeof(float), 4, f) != 4) { fclose(f); return -4; }
    m->S_emb = sc[0]; m->lr = sc[1]; m->step = hdr[10]; m->qat = hdr[11];
    if ((int)sc[2] != m->L + 1) { fclose(f); return -5; }
    if (fread(m->S_res, sizeof(float), (size_t)(m->L + 1), f) != (size_t)(m->L + 1)) { fclose(f); return -4; }
    size_t got = 0;
    got += fread(m->par, sizeof(float), (size_t)m->npar, f);
    got += fread(m->adm1, sizeof(float), (size_t)m->npar, f);
    got += fread(m->adm2, sizeof(float), (size_t)m->npar, f);
    const int L = m->L, D = m->D, F = m->F, V = m->V;
    got += fread(m->S_g1n, sizeof(float), L, f); got += fread(m->S_g2n, sizeof(float), L, f);
    got += fread(m->S_gfn, sizeof(float), 1, f);
    got += fread(m->S_ln1, sizeof(float), L, f); got += fread(m->S_ln2, sizeof(float), L, f);
    got += fread(m->S_final, sizeof(float), L, f);
    got += fread(m->S_q, sizeof(float), L, f); got += fread(m->S_k, sizeof(float), L, f);
    got += fread(m->S_v, sizeof(float), L, f); got += fread(m->S_ao, sizeof(float), L, f);
    got += fread(m->S_f, sizeof(float), L, f); got += fread(m->S_sq, sizeof(float), L, f);
    got += fread(m->sw_qkv, sizeof(float), (size_t)L * 3 * D, f);
    got += fread(m->sw_wo, sizeof(float), (size_t)L * D, f);
    got += fread(m->sw_w1, sizeof(float), (size_t)L * F, f);
    got += fread(m->sw_w2, sizeof(float), (size_t)L * D, f);
    got += fread(m->sw_out, sizeof(float), (size_t)V, f);
    fclose(f);
    ti_export(m);
    return (int)got;
}

/* ------------------------------------------------------- fidélité de l'export
 * Mesure l'erreur de quantification réellement introduite : ||W − s·w8|| / ||W||
 * par matrice. Chiffre rapporté (et non supposé) sur la fidélité de l'export.
 */
static void ti_export_fidelity(const TiModel *m, double *rel_err, double *zero_frac) {
    double num = 0.0, den = 0.0;
    long nz = 0, tot = 0;
    const struct { int id; const int8_t *w8; const float *sw; int N, K; } mats[] = {
        { P_WOUT, m->e.wout, m->sw_out, m->V, m->D },
    };
    (void)mats;
    /* Wout (échelle par ligne) */
    for (int v = 0; v < m->V; v++) {
        const float *w = ti_p((TiModel *)m, P_WOUT, (long)v * m->D);
        const float s = m->sw_out[v];
        const int8_t *q = m->e.wout + (size_t)v * m->D;
        for (int k = 0; k < m->D; k++) {
            const double r = (double)w[k] - (double)s * (double)q[k];
            num += r * r; den += (double)w[k] * (double)w[k];
            tot++; if (q[k] != 0) nz++;
        }
    }
    *rel_err = sqrt(num / (den > 0 ? den : 1));
    *zero_frac = (double)nz / (double)(tot > 0 ? tot : 1);
}


#endif /* TI_MODEL_H */
