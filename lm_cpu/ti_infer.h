/* ============================================================================
 * ti_infer.h — SPEAR-T1 : moteur d'inférence ENTIER (aucun flottant ici).
 *
 * Ce fichier ne contient ni float ni double, ni sqrt, ni division :
 *   - embeddings, gains de norme, poids ternaires : int8
 *   - activations : int8 à échelles STATIQUES (calibrées hors inférence)
 *   - accumulateurs : int32 exacts (bornes prouvées, voir plus bas)
 *   - requantisation : (acc·M + arrondi) >> sh, M ≤ 2^14 entier
 *   - normalisation : Newton entier (ti_rsqrt_q16)
 *   - softmax : exp2 entier (polynôme + décalage) + réciproque normalisée
 *   - échantillonnage : softmax entier + RNG entier
 * `--fpcheck` désassemble ti_int_run/ti_int_step et exige ZÉRO instruction
 * flottante : la preuve est mécanique, pas déclarative.
 *
 * Le préfixage et le décodage partagent le même noyau d'attention (lecture du
 * cache KV) : leurs logits sont donc identiques au bit près (testé).
 * =========================================================================== */
#ifndef TI_INFER_H
#define TI_INFER_H

#include <stdio.h>
#include <stdlib.h>
#include "ti_config.h"
#include "ti_kernels.h"

/* Requantisation affine entière : y = clamp((acc·M + 2^(sh−1)) >> sh).
 * Bornes garanties par l'export : |acc| ≤ 127·K ≤ 2^16, M ≤ 2^14 ⇒ produit
 * ≤ 2^30 < 2^31 → arithmétique int32 EXACTE, zéro dépassement. */
typedef struct { int32_t M; int sh; } TiRq;

static inline int32_t ti_rq_apply(int32_t acc, TiRq r) {
    return ti_clamp8((int32_t)((acc * r.M + (int32_t)(1 << (r.sh - 1))) >> r.sh));
}
static inline int32_t ti_rq_raw(int32_t acc, TiRq r) {   /* sans saturation (logits) */
    return (int32_t)((acc * r.M + (int32_t)(1 << (r.sh - 1))) >> r.sh);
}

/* Produit attention·V : o[d] = Σ_j att[j]·vT[d][j].
 * Le cache V est stocké TRANSPOSÉ ([HD][T]) : la réduction sur j passe alors
 * par ti_dot_i8, le même kernel int8 vérifié bit-exact (maddubs/madd) que le
 * reste du modèle — aucun kernel parallèle non testé.
 * |o| ≤ 127·127·L ≤ 4.1e6 : int32 exact, aucun dépassement. */

/* --------------------------------------------------- sites de calibration --
 * Le moteur mesure lui-même les maxima |q8| de chaque site (entiers). La
 * conversion en échelle flottante se fait dans ti_model.h : aucune statistique
 * supposée, aucune estimation — ce que le corpus produit est ce qui est calibré.
 * Coût : un test par site et par position, uniquement quand measure=1.
 */
#ifndef TI_S_SITES
#define TI_S_SITES (3 + TI_L * 9)
#endif
#define TI_S_EMB   0            /* alias de TI_S_RES(0) : le résidu d'entrée */
#define TI_S_RES(l) (1 + (l))   /* L+1 échelles de résidu, une par profondeur */
#define TI_S_NF    (2 + TI_L)
#define TI_S_LN1(l) (3 + TI_L + (l) * 8 + 0)
#define TI_S_LN2(l) (3 + TI_L + (l) * 8 + 1)
#define TI_S_Q(l)   (3 + TI_L + (l) * 8 + 2)
#define TI_S_K(l)   (3 + TI_L + (l) * 8 + 3)
#define TI_S_V(l)   (3 + TI_L + (l) * 8 + 4)
#define TI_S_AO(l)  (3 + TI_L + (l) * 8 + 5)
#define TI_S_F(l)   (3 + TI_L + (l) * 8 + 6)
#define TI_S_SQ(l)  (3 + TI_L + (l) * 8 + 7)

/* ------------------------------------------------------------------ modèle -- */
typedef struct {
    int T, D, L, H, HD, F, V, B;
    int32_t inv_d_q16;               /* 1/D en Q16 : moyenne sans division */
    /* embeddings : échelle commune (documentée à l'export) */
    int8_t *emb;                     /* [V][D] */
    int8_t *pos;                     /* [T][D] */
    TiRq rq_emb;                     /* (emb8+pos8) int32 -> int8 résiduel */
    /* gains de norme + requant de sortie de norme */
    int8_t *g1, *g2;                 /* [L][D] */
    int8_t *gf;                      /* [D]    */
    TiRq *rq_g1, *rq_g2;             /* [L]    */
    TiRq rq_gf;
    /* poids ternaires déployés int8, layout N-major [N][K] */
    int8_t *wqkv;                    /* [L][3D][D] */
    int8_t *wo;                      /* [L][D][D]  */
    int8_t *w1;                      /* [L][F][D]  */
    int8_t *w2;                      /* [L][D][F]  */
    int8_t *wout;                    /* [V][D]     */
    /* requantisations par canal de sortie */
    TiRq *rq_qkv;                    /* [L][3D] */
    TiRq *rq_wo;                     /* [L][D]  */
    TiRq *rq_w1;                     /* [L][F]  */
    TiRq *rq_sq;                     /* [L][F]  */
    TiRq *rq_w2;                     /* [L][D]  */
    TiRq *rq_ao;                     /* [L][D]  */
    TiRq (*rq_score)[TI_H];          /* [L][H] -> log2 Q12 */
    TiRq *rq_res;                    /* [L] : passage S_res[l] -> S_res[l+1] (résidu) */
    /* cache KV : [B][L][H][T][HD] pour K et V */
    int8_t *kcache, *vcache;
    int ncache;
    /* activations (conservées : l'entraînement s'en sert pour la rétroprop) */
    int8_t *xq;                      /* [B][L+1][T][D]  résiduel (entrées de bloc) */
    int8_t *lnq;                     /* [B][L+1][T][D]  sorties de norme 1 (+ norme finale) */
    int8_t *n2q;                     /* [B][L][T][D]    sorties de norme 2 (séparées : la
                                      * rétropropagation a besoin des deux à la fois) */
    int8_t *qkv;                     /* [B][L][T][3D]   */
    int8_t *att;                     /* [B][L][H][T][T] poids d'attention int8 */
    int8_t *ao;                      /* [B][L][T][D]    */
    int8_t *fq;                      /* [B][L][T][F]    ReLU² int8 */
    /* scratch */
    int32_t *acc;                    /* [n][max(3D,F)] */
    int32_t *scores;                 /* [T]  */
    int32_t *scr32;                  /* [T]  */
    int32_t *oacc;                   /* [D]  */
    int64_t ops;
    int32_t sitemax[TI_S_SITES];     /* maxima |q8| mesurés (calibration) */
    int measure;                     /* 1 = accumuler les maxima */
} TiInt;

#define TI_TRACK(m, site, val) do { if ((m)->measure) { \
        const int32_t _v = (val); const int32_t _a = _v < 0 ? -_v : _v; \
        if (_a > (m)->sitemax[site]) (m)->sitemax[site] = _a; } } while (0)
#define TI_TRACK_ACC(m, site, ptr, n) do { if ((m)->measure) { \
        int32_t _mx = (m)->sitemax[site]; \
        for (int _i = 0; _i < (n); _i++) { const int32_t _v = (ptr)[_i]; \
            const int32_t _a = _v < 0 ? -_v : _v; if (_a > _mx) _mx = _a; } \
        (m)->sitemax[site] = _mx; } } while (0)

static void *ti_xalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "ti: OOM (%zu octets)\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

static void ti_int_alloc(TiInt *m, int B) {
    memset(m, 0, sizeof(*m));
    m->T = TI_T; m->D = TI_D; m->L = TI_L; m->H = TI_H; m->HD = TI_HD;
    m->F = TI_F; m->V = TI_V; m->B = B;
    m->inv_d_q16 = (int32_t)(((int64_t)1 << 16) / TI_D);
    const size_t T = m->T, D = m->D, L = m->L, H = m->H, F = m->F, V = m->V;
    const size_t btd = (size_t)B * T * D;
    const size_t maxacc = (size_t)(3 * D > F ? 3 * D : F);
    m->xq  = (int8_t *)ti_xalloc(btd * (L + 1));
    m->lnq = (int8_t *)ti_xalloc(btd * (L + 1));
    m->n2q = (int8_t *)ti_xalloc(btd * L);
    m->qkv = (int8_t *)ti_xalloc((size_t)B * L * T * 3 * D);
    m->att = (int8_t *)ti_xalloc((size_t)B * L * H * T * T);
    m->ao  = (int8_t *)ti_xalloc((size_t)B * L * T * D);
    m->fq  = (int8_t *)ti_xalloc((size_t)B * L * T * F);
    m->emb = (int8_t *)ti_xalloc((size_t)V * D);
    m->pos = (int8_t *)ti_xalloc(T * D);
    m->g1 = (int8_t *)ti_xalloc(L * D);
    m->g2 = (int8_t *)ti_xalloc(L * D);
    m->gf = (int8_t *)ti_xalloc(D);
    m->rq_g1 = (TiRq *)ti_xalloc(sizeof(TiRq) * L);
    m->rq_g2 = (TiRq *)ti_xalloc(sizeof(TiRq) * L);
    m->wqkv = (int8_t *)ti_xalloc(L * 3 * D * D);
    m->wo   = (int8_t *)ti_xalloc(L * D * D);
    m->w1   = (int8_t *)ti_xalloc(L * F * D);
    m->w2   = (int8_t *)ti_xalloc(L * D * F);
    m->wout = (int8_t *)ti_xalloc(V * D);
    m->rq_qkv = (TiRq *)ti_xalloc(sizeof(TiRq) * L * 3 * D);
    m->rq_wo  = (TiRq *)ti_xalloc(sizeof(TiRq) * L * D);
    m->rq_w1  = (TiRq *)ti_xalloc(sizeof(TiRq) * L * F);
    m->rq_sq  = (TiRq *)ti_xalloc(sizeof(TiRq) * L * F);
    m->rq_w2  = (TiRq *)ti_xalloc(sizeof(TiRq) * L * D);
    m->rq_ao  = (TiRq *)ti_xalloc(sizeof(TiRq) * L * D);
    m->rq_score = (TiRq (*)[TI_H])ti_xalloc(sizeof(TiRq) * L * H);
    m->rq_res = (TiRq *)ti_xalloc(sizeof(TiRq) * L);
    m->kcache = (int8_t *)ti_xalloc((size_t)B * L * H * T * TI_HD);
    m->vcache = (int8_t *)ti_xalloc((size_t)B * L * H * TI_HD * T);   /* transposé [HD][T] */
    m->acc    = (int32_t *)ti_xalloc(sizeof(int32_t) * T * maxacc);
    m->scores = (int32_t *)ti_xalloc(sizeof(int32_t) * T);
    m->scr32  = (int32_t *)ti_xalloc(sizeof(int32_t) * T);
    m->oacc   = (int32_t *)ti_xalloc(sizeof(int32_t) * D);
}

static void ti_int_free(TiInt *m) {
    free(m->xq); free(m->lnq); free(m->n2q); free(m->qkv); free(m->att); free(m->ao); free(m->fq);
    free(m->emb); free(m->pos); free(m->g1); free(m->g2); free(m->gf);
    free(m->rq_g1); free(m->rq_g2);
    free(m->wqkv); free(m->wo); free(m->w1); free(m->w2); free(m->wout);
    free(m->rq_qkv); free(m->rq_wo); free(m->rq_w1); free(m->rq_sq); free(m->rq_w2);
    free(m->rq_ao); free(m->rq_score); free(m->rq_res);
    free(m->kcache); free(m->vcache);
    free(m->acc); free(m->scores); free(m->scr32); free(m->oacc);
    memset(m, 0, sizeof(*m));
}

/* --------------------------------------------------------- noyau forward --
 * Attention causale entière : la ligne t attend les clés 0..t du cache.
 * Préfixage et décodage appellent CETTE fonction → parité au bit près.
 */
static void ti_attn_layer(TiInt *m, int b, int l, int row0, int nrows) {
    const int D = m->D, H = m->H, HD = m->HD, T = m->T;
    const size_t per_b = (size_t)m->L * H * T * HD;
    const int8_t *kch = m->kcache + (size_t)b * per_b + (size_t)l * H * T * HD;
    const int8_t *vch = m->vcache + (size_t)b * per_b + (size_t)l * H * HD * T;
    int8_t *qkv = m->qkv + ((size_t)b * m->L + l) * T * 3 * D;
    int8_t *ao  = m->ao  + ((size_t)b * m->L + l) * T * D;
    int8_t *attl = m->att + ((size_t)b * m->L + l) * H * T * T;
    for (int h = 0; h < H; h++) {
        const TiRq rs = m->rq_score[l][h];
        for (int r = 0; r < nrows; r++) {
            const int t = row0 + r;
            const int8_t *qr = qkv + ((size_t)t * 3 * D) + h * HD;
            int8_t *att_row = attl + ((size_t)h * T + t) * T;
            for (int j = 0; j <= t; j++) {
                const int32_t a = ti_dot_i8(qr, kch + ((size_t)h * T + j) * HD, HD);
                m->scores[j] = ti_rq_raw(a, rs);          /* log2 Q12 */
                m->ops += HD;
            }
            ti_softmax_i8_row(m->scores, t + 1, att_row, m->scr32);
            int32_t *o = m->oacc;
            for (int d = 0; d < HD; d++)
                o[d] = ti_dot_i8(att_row, vch + ((size_t)h * HD + d) * T, t + 1);
            m->ops += (int64_t)(t + 1) * HD;
            int8_t *dst = ao + (size_t)t * D + h * HD;
            for (int d = 0; d < HD; d++) {
                const int32_t raw = ti_rq_raw(o[d], m->rq_ao[l * D + h * HD + d]);
                dst[d] = (int8_t)ti_clamp8(raw);
                TI_TRACK(m, TI_S_AO(l), raw);   /* avant écrêtage */
            }
        }
    }
}

/* logits_mode : 0 = aucun, 1 = dernière position par batch, 2 = toutes.
 * logits : [B][n][V] (mode 2) ou [B][V] (mode 1), int32. */
static void ti_int_run(TiInt *m, const int *ids, int B, int n, int pos0,
                       int32_t *logits, int logits_mode) {
    const int T = m->T, D = m->D, L = m->L, F = m->F, V = m->V;
    /* 1) embeddings : emb8 + pos8 (échelle commune) → résiduel int8 */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < n; t++) {
            const int8_t *e = m->emb + (size_t)ids[(size_t)b * n + t] * D;
            const int8_t *p = m->pos + (size_t)(pos0 + t) * D;
            int8_t *x = m->xq + ((size_t)b * T + pos0 + t) * D;
            for (int d = 0; d < D; d++) {
                const int32_t raw = ti_rq_raw((int32_t)e[d] + (int32_t)p[d], m->rq_emb);
                TI_TRACK(m, TI_S_RES(0), raw);   /* avant écrêtage : l'échelle peut grandir */
                x[d] = (int8_t)ti_clamp8(raw);
            }
        }
    /* 2) blocs */
    for (int l = 0; l < L; l++) {
        int8_t *lnq = m->lnq + (size_t)l * m->B * T * D;
        int8_t *xcur = m->xq + (size_t)l * m->B * T * D;
        int8_t *xnext = m->xq + (size_t)(l + 1) * m->B * T * D;
        int8_t *qkv = m->qkv + (size_t)l * m->B * T * 3 * D;
        int8_t *ao = m->ao + (size_t)l * m->B * T * D;
        int8_t *fq = m->fq + (size_t)l * m->B * T * F;
        /* 2a) RMSNorm 1 → int8 (échelle statique) */
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++) {
                int32_t mx = m->sitemax[TI_S_LN1(l)];
                ti_rmsnorm_i8(xcur + ((size_t)b * T + pos0 + t) * D, m->g1 + (size_t)l * D,
                              lnq + ((size_t)b * T + pos0 + t) * D, D, m->inv_d_q16,
                              m->rq_g1[l].M, m->rq_g1[l].sh, NULL, m->measure ? &mx : NULL);
                if (m->measure) m->sitemax[TI_S_LN1(l)] = mx;
            }
        /* 2b) QKV = lnq · Wqkvᵀ (int8×ternaire) + requant par canal */
        for (int b = 0; b < B; b++) {
            ti_gemm_i8(m->acc, lnq + (size_t)b * T * D + (size_t)pos0 * D,
                       m->wqkv + (size_t)l * 3 * D * D, n, D, 3 * D);
            m->ops += (int64_t)n * D * 3 * D;
            int8_t *dst = qkv + (size_t)b * T * 3 * D + (size_t)pos0 * 3 * D;
            const TiRq *rq = m->rq_qkv + (size_t)l * 3 * D;
            for (int t = 0; t < n; t++) {
                int8_t *row = dst + (size_t)t * 3 * D;
                for (int c = 0; c < 3 * D; c++) {
                    const int32_t raw = ti_rq_raw(m->acc[(size_t)t * 3 * D + c], rq[c]);
                    row[c] = (int8_t)ti_clamp8(raw);
                    TI_TRACK(m, c < D ? TI_S_Q(l) : (c < 2 * D ? TI_S_K(l) : TI_S_V(l)), raw);
                }
            }
        }
        /* 2c) écriture du cache KV pour les positions courantes (int8 direct) */
        for (int b = 0; b < B; b++) {
            const size_t per_b = (size_t)L * m->H * T * m->HD;
            int8_t *kch = m->kcache + (size_t)b * per_b + (size_t)l * m->H * T * m->HD;
            int8_t *vch = m->vcache + (size_t)b * per_b + (size_t)l * m->H * m->HD * T;
            const int8_t *src = qkv + (size_t)b * T * 3 * D;
            for (int t = 0; t < n; t++) {
                const int pos = pos0 + t;                 /* position absolue */
                const int8_t *row = src + (size_t)pos * 3 * D;
                for (int h = 0; h < m->H; h++) {
                    memcpy(kch + ((size_t)h * T + pos) * m->HD, row + D + h * m->HD, m->HD);
                    const int8_t *vr = row + 2 * D + h * m->HD;      /* V transposé */
                    for (int d = 0; d < m->HD; d++) vch[((size_t)h * m->HD + d) * T + pos] = vr[d];
                }
            }
        }
        /* 2d) attention (une ou plusieurs lignes par batch) */
        for (int b = 0; b < B; b++) ti_attn_layer(m, b, l, pos0, n);
        /* 2e) projection de sortie + résidu */
        for (int b = 0; b < B; b++) {
            ti_gemm_i8(m->acc, ao + (size_t)b * T * D + (size_t)pos0 * D,
                       m->wo + (size_t)l * D * D, n, D, D);
            m->ops += (int64_t)n * D * D;
            int8_t *dst = xnext + (size_t)b * T * D + (size_t)pos0 * D;
            const int8_t *src = xcur + (size_t)b * T * D + (size_t)pos0 * D;
            const TiRq *rq = m->rq_wo + (size_t)l * D;
            for (int t = 0; t < n; t++)
                for (int c = 0; c < D; c++) {
                    const int32_t v = ti_rq_raw(m->acc[(size_t)t * D + c], rq[c]);
                    /* le résidu entrant est ramené de S_res[l] à S_res[l+1] : une
                     * seule échelle pour tout le réseau gaspillerait 4 bits aux
                     * couches fines et saturerait les couches profondes */
                    const int32_t xin = ti_rq_apply((int32_t)src[(size_t)t * D + c], m->rq_res[l]);
                    dst[(size_t)t * D + c] = (int8_t)ti_clamp8(xin + v);
                }
        }
        /* 2f) RMSNorm 2 (sur xnext) — sortie dans n2q, lnq garde N1 */
        int8_t *ln2 = m->n2q + (size_t)l * m->B * T * D;
        for (int b = 0; b < B; b++)
            for (int t = 0; t < n; t++) {
                int32_t mx = m->sitemax[TI_S_LN2(l)];
                ti_rmsnorm_i8(xnext + ((size_t)b * T + pos0 + t) * D, m->g2 + (size_t)l * D,
                              ln2 + ((size_t)b * T + pos0 + t) * D, D, m->inv_d_q16,
                              m->rq_g2[l].M, m->rq_g2[l].sh, NULL, m->measure ? &mx : NULL);
                if (m->measure) m->sitemax[TI_S_LN2(l)] = mx;
            }
        /* 2g) feed-forward ternaire : fc1 → ReLU² → fc2 */
        const TiRq *rqw1 = m->rq_w1 + (size_t)l * F;
        const TiRq *rqsq = m->rq_sq + (size_t)l * F;
        for (int b = 0; b < B; b++) {
            ti_gemm_i8(m->acc, ln2 + (size_t)b * T * D + (size_t)pos0 * D,
                       m->w1 + (size_t)l * F * D, n, D, F);
            m->ops += (int64_t)n * D * F;
            int8_t *fd = fq + (size_t)b * T * F + (size_t)pos0 * F;
            for (int t = 0; t < n; t++)
                for (int c = 0; c < F; c++) {
                    const int32_t raw = ti_rq_raw(m->acc[(size_t)t * F + c], rqw1[c]);
                    TI_TRACK(m, TI_S_F(l), raw);            /* avant écrêtage */
                    int32_t v = ti_clamp8(raw);
                    if (v < 0) v = 0;                       /* ReLU */
                    const int32_t v2 = v * v;               /* ≤ 16129 */
                    const int32_t sqraw = ti_rq_raw(v2, rqsq[c]);
                    TI_TRACK(m, TI_S_SQ(l), sqraw);         /* avant écrêtage */
                    fd[(size_t)t * F + c] = (int8_t)ti_clamp8(sqraw);
                }
            ti_gemm_i8(m->acc, fd, m->w2 + (size_t)l * D * F, n, F, D);
            m->ops += (int64_t)n * F * D;
            int8_t *dst = xnext + (size_t)b * T * D + (size_t)pos0 * D;
            const TiRq *rq2 = m->rq_w2 + (size_t)l * D;
            for (int t = 0; t < n; t++)
                for (int c = 0; c < D; c++) {
                    const int32_t v = ti_rq_raw(m->acc[(size_t)t * D + c], rq2[c]);
                    const int32_t raw = (int32_t)dst[(size_t)t * D + c] + v;
                    dst[(size_t)t * D + c] = (int8_t)ti_clamp8(raw);
                    /* mesuré AVANT écrêtage : c'est la seule façon de détecter
                     * que le résidu dépasse la dynamique allouée */
                    TI_TRACK(m, TI_S_RES(l + 1), raw);
                }
        }
    }
    /* 3) norme finale + projection de sortie (logits int32, sans requant) */
    int8_t *lnf = m->lnq + (size_t)L * m->B * T * D;
    int8_t *xlast = m->xq + (size_t)L * m->B * T * D;
    if (logits_mode) {
        for (int b = 0; b < B; b++) {
            const int t0 = (logits_mode == 2) ? 0 : n - 1;
            for (int t = t0; t < n; t++) {
                int32_t mxf = m->sitemax[TI_S_NF];
                ti_rmsnorm_i8(xlast + ((size_t)b * T + pos0 + t) * D, m->gf, lnf, D, m->inv_d_q16,
                              m->rq_gf.M, m->rq_gf.sh, NULL, m->measure ? &mxf : NULL);
                if (m->measure) m->sitemax[TI_S_NF] = mxf;
                int32_t *out = logits + ((logits_mode == 2) ? ((size_t)b * n + t) * V : (size_t)b * V);
                for (int v = 0; v < V; v++) {
                    out[v] = ti_dot_i8(lnf, m->wout + (size_t)v * D, D);
                    m->ops += D;
                }
            }
        }
    }
}

/* Décodage autorégressif : un jeton, position = m->ncache. */
static void ti_int_step(TiInt *m, int id, int32_t *logits) {
    const int pos = m->ncache;
    ti_int_run(m, &id, 1, 1, pos, logits, 1);
    m->ncache = pos + 1;
}

static void ti_int_reset(TiInt *m) { m->ncache = 0; }

/* -------------------------------------------------------------- échantillon --
 * Échantillonnage entier : softmax entier sur les logits int32 (température
 * appliquée par un décalage de l'exposant, aucun flottant), puis tirage par
 * RNG entier déterministe. temperature_q8 : 0 = argmax (glouton). */
typedef struct { uint64_t s; } TiRng;
static inline uint64_t ti_rng_u64(TiRng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static int ti_int_sample(const TiInt *m, const int32_t *logits, int temperature_q8, TiRng *rng) {
    const int V = m->V;
    if (temperature_q8 <= 0) {                       /* glouton : argmax entier */
        int best = 0;
        for (int v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
    /* score_j = logit_j·(1/T)·log2e·2^12 ; 1/T en Q8 */
    int32_t mx = logits[0];
    for (int v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
    int32_t *sc = (int32_t *)ti_xalloc(sizeof(int32_t) * V);
    int8_t *w8 = (int8_t *)ti_xalloc(V);
    for (int v = 0; v < V; v++) {
        /* (logit − mx)·log2e·2^12·(256/T) / 256, le tout en arithmétique entière */
        /* (logit−mx)·log2e·2^12 / T  — réciproque normalisée : aucune division */
        int kk = 0;
        const int32_t rc = ti_rcp_norm((uint32_t)temperature_q8, &kk);
        const int64_t num = (int64_t)(logits[v] - mx) * 5909 * rc;
        sc[v] = (int32_t)(num >> (12 + kk + 8));
    }
    ti_softmax_i8_row(sc, V, w8, m->scr32);
    int32_t tot = 0;
    for (int v = 0; v < V; v++) tot += w8[v];
    if (tot <= 0) { free(sc); free(w8); return 0; }
    const uint64_t pick = ti_rng_u64(rng) % (uint64_t)tot;
    uint64_t acc = 0;
    int out = V - 1;
    for (int v = 0; v < V; v++) { acc += (uint64_t)w8[v]; if (pick < acc) { out = v; break; } }
    free(sc); free(w8);
    return out;
}

/* -------------------------------------------------------------- sauvegarde --
 * Le fichier ne contient que des entiers (aucun flottant non plus). */
#define TI_MAGIC 0x3154494Du   /* "MIT1" */
static int ti_int_save(const TiInt *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* hdr[9] = version du format : 1 = rq_res présent (échelles de résidu par
     * couche). Un fichier sans rq_res mettait le résidu entrant à zéro à chaque
     * couche (rq_res = {0,1} ⇒ xin = 0) : les logits devenaient indépendants de
     * la position et la génération ne produisait qu'un seul octet répété.
     * v2 : matrices ternaires compactées à 2 bits/poids au lieu d'1 octet. */
    const int hdr[10] = { (int)TI_MAGIC, m->T, m->D, m->L, m->H, m->HD, m->F, m->V, 1, 2 };
    fwrite(hdr, sizeof(int), 10, f);
    const size_t D = m->D, L = m->L, F = m->F, V = m->V, T = m->T, H = m->H;
    fwrite(m->emb, 1, (size_t)V * D, f);
    fwrite(m->pos, 1, T * D, f);
    fwrite(&m->rq_emb, sizeof(TiRq), 1, f);
    fwrite(m->g1, 1, L * D, f); fwrite(m->g2, 1, L * D, f); fwrite(m->gf, 1, D, f);
    fwrite(m->rq_g1, sizeof(TiRq), L, f); fwrite(m->rq_g2, sizeof(TiRq), L, f);
    fwrite(&m->rq_gf, sizeof(TiRq), 1, f);
    /* Matrices ternaires en 2 bits/poids (4 poids par octet) : le format déployé
     * en RAM reste 1 octet/poids — c'est le rayon d'accès du moteur — mais le
     * FICHIER n'a aucune raison de payer ce facteur 4. ti_pack2/unpack2 sont
     * vérifiés exacts par ti_test. */
    {
        const size_t nb = (size_t)(L * 3 * D * D + L * D * D + L * F * D + L * D * F + V * D);
        uint8_t *pk = (uint8_t *)ti_xalloc((nb + 3) / 4);
        const int8_t *srcs[5] = { m->wqkv, m->wo, m->w1, m->w2, m->wout };
        const int cnt[5] = { (int)(L * 3 * D * D), (int)(L * D * D), (int)(L * F * D),
                             (int)(L * D * F), (int)(V * D) };
        size_t off = 0;
        for (int t = 0; t < 5; t++) { ti_pack2(srcs[t], pk + off / 4, cnt[t]); off += (size_t)cnt[t]; }
        fwrite(pk, 1, (nb + 3) / 4, f);
        free(pk);
    }
    fwrite(m->rq_qkv, sizeof(TiRq), L * 3 * D, f);
    fwrite(m->rq_wo, sizeof(TiRq), L * D, f);
    fwrite(m->rq_w1, sizeof(TiRq), L * F, f);
    fwrite(m->rq_sq, sizeof(TiRq), L * F, f);
    fwrite(m->rq_w2, sizeof(TiRq), L * D, f);
    fwrite(m->rq_ao, sizeof(TiRq), L * D, f);
    fwrite(m->rq_score, sizeof(TiRq), L * H, f);
    fwrite(m->rq_res, sizeof(TiRq), L, f);
    fclose(f);
    return 0;
}

static int ti_int_load(TiInt *m, const char *path, int B) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int hdr[10];
    if (fread(hdr, sizeof(int), 10, f) != 10 || hdr[0] != (int)TI_MAGIC) { fclose(f); return -2; }
    if (hdr[1] != TI_T || hdr[2] != TI_D || hdr[3] != TI_L || hdr[4] != TI_H || hdr[6] != TI_F) {
        fprintf(stderr, "ti: checkpoint incompatible avec la géométrie compilée "
                        "(T=%d D=%d L=%d H=%d F=%d)\n", hdr[1], hdr[2], hdr[3], hdr[4], hdr[6]);
        fclose(f); return -3;
    }
    if (m->emb == NULL) ti_int_alloc(m, B);
    const size_t D = m->D, L = m->L, F = m->F, V = m->V, T = m->T, H = m->H;
    size_t got = 0;
    got += fread(m->emb, 1, V * D, f);
    got += fread(m->pos, 1, T * D, f);
    got += fread(&m->rq_emb, sizeof(TiRq), 1, f);
    got += fread(m->g1, 1, L * D, f) + fread(m->g2, 1, L * D, f) + fread(m->gf, 1, D, f);
    got += fread(m->rq_g1, sizeof(TiRq), L, f) + fread(m->rq_g2, sizeof(TiRq), L, f);
    got += fread(&m->rq_gf, sizeof(TiRq), 1, f);
    {   /* poids ternaires : 2 bits/poids dans le fichier (format v2) */
        const size_t nb = (size_t)(L * 3 * D * D + L * D * D + L * F * D + L * D * F + V * D);
        uint8_t *pk = (uint8_t *)ti_xalloc((nb + 3) / 4);
        got += fread(pk, 1, (nb + 3) / 4, f);
        int8_t *dsts[5] = { m->wqkv, m->wo, m->w1, m->w2, m->wout };
        const int cnt[5] = { (int)(L * 3 * D * D), (int)(L * D * D), (int)(L * F * D),
                             (int)(L * D * F), (int)(V * D) };
        size_t off = 0;
        for (int t = 0; t < 5; t++) { ti_unpack2(pk + off / 4, dsts[t], cnt[t]); off += (size_t)cnt[t]; }
        free(pk);
    }
    got += fread(m->rq_qkv, sizeof(TiRq), L * 3 * D, f) + fread(m->rq_wo, sizeof(TiRq), L * D, f);
    got += fread(m->rq_w1, sizeof(TiRq), L * F, f) + fread(m->rq_sq, sizeof(TiRq), L * F, f);
    got += fread(m->rq_w2, sizeof(TiRq), L * D, f) + fread(m->rq_ao, sizeof(TiRq), L * D, f);
    got += fread(m->rq_score, sizeof(TiRq), L * H, f);
    if (hdr[9] == 1) {
        fprintf(stderr, "ti: fichier au format v1 (poids déployés) — réexportez-le\n");
        fclose(f); return -5;
    }
    if (hdr[9] >= 2) {
        got += fread(m->rq_res, sizeof(TiRq), L, f);
    } else {
        /* ancien fichier : pas d'échelle de résidu. On refuse plutôt que de
         * faire tourner un modèle aux logits indépendants de la position. */
        fprintf(stderr, "ti: fichier au format v0 (sans échelles de résidu) — "
                        "réexportez-le avec la version courante\n");
        fclose(f); return -5;
    }
    fclose(f);
    return got ? 0 : -4;
}

/* Empreinte de déterminisme : somme int64 des logits (test « deux exécutions
 * → même entier », et comparaison scalaire/AVX2). */
static int64_t ti_logits_hash(const int32_t *logits, int n) {
    int64_t h = 1469598103934665603LL;
    for (int i = 0; i < n; i++) {
        h ^= (int64_t)logits[i];
        h *= 1099511628211LL;
    }
    return h;
}

/* Diagnostics de densité ternaire (appelé par --stats). */
static void ti_int_ternary_stats(const TiInt *m, long *total, long *nonzero, long *plus, long *minus) {
    const size_t sz[5] = { (size_t)m->L * 3 * m->D * m->D, (size_t)m->L * m->D * m->D,
                           (size_t)m->L * m->F * m->D, (size_t)m->L * m->D * m->F,
                           (size_t)m->V * m->D };
    const int8_t *base[5] = { m->wqkv, m->wo, m->w1, m->w2, m->wout };
    long tot = 0, nz = 0, pl = 0, mi = 0;
    for (int t = 0; t < 5; t++)
        for (size_t i = 0; i < sz[t]; i++) {
            const int8_t v = base[t][i];
            tot++;
            if (v > 0) { nz++; pl++; } else if (v < 0) { nz++; mi++; }
        }
    *total = tot; *nonzero = nz; *plus = pl; *minus = mi;
}

#endif /* TI_INFER_H */
