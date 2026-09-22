/* ========================================================================== */
/* SPEAR BT07 : Quantification géométrique réseau Gosset E8 pour compression   */
/* d'embeddings LLM - décompression en 4 cycles sans table                    */
/* ========================================================================== */
#include <stdint.h>
#include <math.h>

/** Quantification E8 rapide sans table (Conway-Sloane) pour bloc de 8 dimensions */
void spear_e8_quantize_block(const float x[8], int16_t out_e8[8]) {
    float f0[8], f1[8];
    float diff0[8], diff1[8];
    int sum0 = 0, sum1 = 0;
    int max_idx0 = 0, max_idx1 = 0;
    float max_err0 = -1.0f, max_err1 = -1.0f;

    for (int i = 0; i < 8; ++i) {
        /* Coset 0 : Z^8 */
        f0[i] = roundf(x[i]);
        diff0[i] = fabsf(x[i] - f0[i]);
        sum0 += (int)f0[i];
        if (diff0[i] > max_err0) { max_err0 = diff0[i]; max_idx0 = i; }

        /* Coset 1 : Z^8 + 1/2 */
        float x_shifted = x[i] - 0.5f;
        f1[i] = roundf(x_shifted);
        diff1[i] = fabsf(x_shifted - f1[i]);
        sum1 += (int)f1[i];
        if (diff1[i] > max_err1) { max_err1 = diff1[i]; max_idx1 = i; }
    }

    /* Correction de parité D_8 */
    if ((sum0 & 1) != 0) {
        f0[max_idx0] += (x[max_idx0] >= f0[max_idx0]) ? 1.0f : -1.0f;
    }
    if ((sum1 & 1) != 0) {
        f1[max_idx1] += ((x[max_idx1] - 0.5f) >= f1[max_idx1]) ? 1.0f : -1.0f;
    }

    /* Comparaison des distances euclidiennes */
    float dist0 = 0.0f, dist1 = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float d0 = x[i] - f0[i];
        float d1 = x[i] - (f1[i] + 0.5f);
        dist0 += d0 * d0;
        dist1 += d1 * d1;
    }

    /* Écriture vectorielle du point E_8 le plus proche */
    const float *best = (dist0 <= dist1) ? f0 : f1;
    const float offset = (dist0 <= dist1) ? 0.0f : 0.5f;
    for (int i = 0; i < 8; ++i) {
        out_e8[i] = (int16_t)((best[i] + offset) * 2.0f); /* Encodé en demi-entiers */
    }
}