/* ============================================================================
 * ti_dump.c — vidage du graphe pour la référence INDÉPENDANTE numpy.
 *
 * Écrit dans un fichier texte la géométrie, le lot (ids/tgt), la perte, tous les
 * paramètres et tous les gradients, dans le format attendu par
 * `tools/ti_reference.py`. Cette référence réimplémente le graphe entier en
 * numpy/float64 sans partager une ligne de code avec ce dépôt : c'est elle qui
 * arbitre en cas de désaccord, et c'est la seule vérification qui ne peut pas
 * être fausse « du même côté » que l'implémentation C.
 *
 * Géométrie minuscule par défaut (le script numpy est quadratique en T) :
 *   cc -O2 -ffast-math -I. -o ti_dump ti_dump.c -lm \
 *      -DTI_T=8 -DTI_D=8 -DTI_L=2 -DTI_H=2 -DTI_F=8 -DTI_V=8 -DTI_B=2
 *   ./ti_dump > vidage.txt && python3 ../tools/ti_reference.py vidage.txt
 * ========================================================================== */
#include "ti_model.h"

static const char *NOMS[P_NUM] = { "emb", "pos", "g1", "g2", "gf", "wqkv", "wo", "w1", "w2", "wout" };

int main(void) {
    TiModel m;
    ti_model_init(&m, TI_B, 88172645463325252ull);
    m.qat = 0;                        /* graphe flottant exact : c'est lui la référence */
    int *ids = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    int *tgt = (int *)ti_fa(sizeof(int) * (size_t)m.ntok);
    /* lot déterministe et non trivial : plusieurs positions, plusieurs octets,
     * des cibles distinctes (une cible uniforme masquerait des erreurs) */
    for (int i = 0; i < m.ntok; i++) {
        ids[i] = (i * 3 + 1) % m.V;
        tgt[i] = (i * 5 + 2) % m.V;
    }
    ti_fwd_fp32(&m, ids, m.T, 0);
    const double loss = ti_loss_grad(&m, tgt, m.T);
    memset(m.grd, 0, sizeof(float) * (size_t)m.npar);
    ti_backward(&m, ids, m.T);

    printf("geom %d %d %d %d %d %d %d %d\n", m.T, m.D, m.L, m.H, m.HD, m.F, m.V, m.B);
    printf("loss %.9g\n", loss);
    /* attention au format : le lecteur du script démarre la section APRÈS la
     * ligne d'en-tête, donc les valeurs vont sur la ligne suivante */
    printf("ids\n");
    for (int i = 0; i < m.ntok; i++) printf("%d ", ids[i]);
    printf("\n");
    printf("tgt\n");
    for (int i = 0; i < m.ntok; i++) printf("%d ", tgt[i]);
    printf("\n");
    for (int id = 0; id < P_NUM; id++) {
        printf("par %s %ld\n", NOMS[id], m.sz[id]);
        for (long i = 0; i < m.sz[id]; i++) printf("%.9g ", m.par[m.off[id] + i]);
        printf("\n");
    }
    for (int id = 0; id < P_NUM; id++) {
        printf("grd %s %ld\n", NOMS[id], m.sz[id]);
        for (long i = 0; i < m.sz[id]; i++) printf("%.9g ", m.grd[m.off[id] + i]);
        printf("\n");
    }
    free(ids); free(tgt);
    ti_model_free(&m);
    return 0;
}
