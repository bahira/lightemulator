/* ============================================================================
 * ti_config.h — géométrie du modèle SPEAR-T1 (compilée au plus juste).
 *
 * Toutes les dimensions sont des constantes de compilation : les boucles des
 * noyaux sont déroulées, les tampons sont des tableaux statiques et aucun
 * calcul de dimensions n'a lieu à l'exécution. Chaque macro est surchargeable
 * en ligne de commande (-DTI_T=128 …) pour balayer une géométrie sans toucher
 * au code — c'est ainsi que le gradcheck tourne en petite géométrie.
 *
 * (les dimensions sont lisibles dans les nombres du fichier .ti exporté : le
 *  chargeur refuse tout modèle dont la géométrie diffère du binaire.)
 * =========================================================================== */
#ifndef TI_CONFIG_H
#define TI_CONFIG_H

#ifndef TI_T
#define TI_T 256          /* longueur de contexte (jetons par séquence) */
#endif
#ifndef TI_D
#define TI_D 192          /* dimension du modèle */
#endif
#ifndef TI_L
#define TI_L 4            /* nombre de blocs attention/FFN */
#endif
#ifndef TI_H
#define TI_H 3            /* têtes d'attention */
#endif
#ifndef TI_HD
#define TI_HD (TI_D / TI_H)   /* dimension par tête (doit rester multiple de 32) */
#endif
#ifndef TI_F
#define TI_F (2 * TI_D)   /* dimension cachée du FFN (2·D : tenseur ternaire) */
#endif
#ifndef TI_V
#define TI_V 256          /* vocabulaire OCTET (byte-level LM) */
#endif
#ifndef TI_B
#define TI_B 8            /* lot par défaut */
#endif

#endif /* TI_CONFIG_H */
