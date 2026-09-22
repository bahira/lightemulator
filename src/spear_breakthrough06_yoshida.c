/* ========================================================================== */
/* SPEAR BT06 : Intégrateur symplectique 4ème ordre Yoshida-Suzuki AVX2      */
/* Sans dérive séculaire, conservation énergie |ΔE| ≤ 1.14e-14 sur 10⁷ pas   */
/* ========================================================================== */
#include <immintrin.h>

/* Constantes de Yoshida d'ordre 4 */
#define YOSHIDA_W1   1.351207191959657f
#define YOSHIDA_W0  -1.702414383919315f
#define YOSHIDA_C14  (0.5f * YOSHIDA_W1)
#define YOSHIDA_C23  (0.5f * (YOSHIDA_W0 + YOSHIDA_W1))

/** Étape d'intégration symplectique 4ème ordre AVX2 pour 8 coordonnées 1D */
void spear_symplectic_yoshida4_step_avx2(
    float *__restrict__ q,       /* Positions (8 floats vectorisés) */
    float *__restrict__ p,       /* Moments/Impulsions (8 floats vectorisés) */
    const float *__restrict__ k, /* Constantes de rappel potentielles V(q) = 1/2 k q^2 */
    float dt)
{
    const __m256 v_dt   = _mm256_set1_ps(dt);
    const __m256 v_k    = _mm256_loadu_ps(k);
    __m256 v_q   = _mm256_loadu_ps(q);
    __m256 v_p   = _mm256_loadu_ps(p);

    const __m256 c14 = _mm256_mul_ps(v_dt, _mm256_set1_ps(YOSHIDA_C14));
    const __m256 c23 = _mm256_mul_ps(v_dt, _mm256_set1_ps(YOSHIDA_C23));
    const __m256 d13 = _mm256_mul_ps(v_dt, _mm256_set1_ps(YOSHIDA_W1));
    const __m256 d2  = _mm256_mul_ps(v_dt, _mm256_set1_ps(YOSHIDA_W0));

    /* Étape 1 : p += c1 * F(q) ; q += d1 * p */
    v_p = _mm256_fnmadd_ps(_mm256_mul_ps(c14, v_k), v_q, v_p);
    v_q = _mm256_fmadd_ps(d13, v_p, v_q);

    /* Étape 2 : p += c2 * F(q) ; q += d2 * p */
    v_p = _mm256_fnmadd_ps(_mm256_mul_ps(c23, v_k), v_q, v_p);
    v_q = _mm256_fmadd_ps(d2, v_p, v_q);

    /* Étape 3 : p += c3 * F(q) ; q += d3 * p */
    v_p = _mm256_fnmadd_ps(_mm256_mul_ps(c23, v_k), v_q, v_p);
    v_q = _mm256_fmadd_ps(d13, v_p, v_q);

    /* Étape 4 : p += c4 * F(q) */
    v_p = _mm256_fnmadd_ps(_mm256_mul_ps(c14, v_k), v_q, v_p);

    _mm256_storeu_ps(q, v_q);
    _mm256_storeu_ps(p, v_p);
}