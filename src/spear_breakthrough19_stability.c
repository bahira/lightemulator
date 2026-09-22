/* ========================================================================== */
/* SPEAR BT19 : Stabilité Smart-Grid / Ott-Antonsen O(1) réduction Kuramoto  */
/* Évaluation globale réseau en 1 cycle, pas d'assemblage O(N²)               */
/* ========================================================================== */
#include <math.h>

typedef struct {
    float gamma;     /* Dispersion de fréquence du réseau (rad/s) */
    float alpha;     /* Déphasage de ligne moyen (rad) */
    float omega0;    /* Fréquence nominale synchrone (ex: 2*pi*50 Hz) */
} spear_grid_kuramoto_cfg_t;

/** Évaluation instantanée O(1) de la stabilité globale d'un réseau électrique */
void spear_smartgrid_stability_eval(
    const spear_grid_kuramoto_cfg_t *cfg, float K_coupling, float *out_coherence_R, float *out_margin_to_trip)
{
    const float cos_alpha = cosf(cfg->alpha);
    const float K_eff = K_coupling * cos_alpha;
    const float Kc = 2.0f * cfg->gamma;

    /* Marge de sécurité avant décrochage */
    *out_margin_to_trip = K_eff - Kc;

    /* Paramètre d'ordre d'Ott-Antonsen stationnaire */
    if (K_eff > Kc) {
        *out_coherence_R = sqrtf(1.0f - (Kc / K_eff));
    } else {
        *out_coherence_R = 0.0f;
    }
}