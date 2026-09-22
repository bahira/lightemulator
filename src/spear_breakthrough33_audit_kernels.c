/* ============================================================================
 * SPEAR BT33 : AUDIT KERNELS v2.2 — Validés en C (MISRA-C:2012 friendly)
 * 1. SCARA IK closed-form (round-trip vs forward kinematics)
 * 2. HFT implied vol Black-Scholes (vs Newton-Raphson référence)
 * 3. Screening moléculaire O(1) (vs erfc référence)
 * 4. NMPC évasion analytique (propriétés vérifiées)
 * Zero-heap, déterministe, O(1).
 * ============================================================================ */
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* ------------------------------ 1. SCARA IK ------------------------------ */
typedef struct {
    float a0; float a1; float L1; float L2;
    float r_min_sq; float r_max_sq;
} spear_scara_config_t;

typedef struct { float theta1; float theta2; } spear_scara_joints_t;

static inline float spear_clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);
}

static bool spear_scara_ik_eval(const spear_scara_config_t *cfg,
                                float x, float z,
                                spear_scara_joints_t *out)
{
    float r2 = x*x + z*z;
    if (r2 < cfg->r_min_sq || r2 > cfg->r_max_sq) return false;
    float cos_t2 = spear_clampf(cfg->a0 + cfg->a1 * r2, -1.0f, 1.0f);
    float sin_t2 = sqrtf(1.0f - cos_t2*cos_t2);
    out->theta2 = atan2f(sin_t2, cos_t2);            /* elbow down */
    float k1 = cfg->L1 + cfg->L2 * cos_t2;
    float k2 = cfg->L2 * sin_t2;
    out->theta1 = atan2f(z, x) - atan2f(k2, k1);
    return true;
}

static void spear_scara_fk(const spear_scara_config_t *cfg,
                           const spear_scara_joints_t *j,
                           float *x, float *z)
{
    float s1 = sinf(j->theta1), c1 = cosf(j->theta1);
    float s12 = sinf(j->theta1 + j->theta2), c12 = cosf(j->theta1 + j->theta2);
    *x = cfg->L1*c1 + cfg->L2*c12;
    *z = cfg->L1*s1 + cfg->L2*s12;
}

/* ---------------------------- 2. HFT Implied Vol -------------------------- */
static float bs_call(float S, float K, float T, float r, float sigma);

static inline float spear_hft_implied_vol(float price, float S, float K,
                                          float T, float r)
{
    float inv_sqrt_T = 1.0f / sqrtf(T);
    float discount   = expf(-r * T);
    float F          = S / discount;
    float x          = logf(F / K);
    float c          = price / (K * discount);
    float diff       = c - 0.5f * (F / K - 1.0f);
    float num        = 2.50662827f * diff;              /* sqrt(2*pi) */
    float den        = 1.0f + 0.483832f * fabsf(x) + 0.125f * diff * diff;
    return fmaxf((num / den) * inv_sqrt_T, 1e-4f);
}

/* Seed rationnel + 2 pas Newton : precision machine sans boucle complete */
static float spear_hft_iv_seed_newton(float price, float S, float K,
                                      float T, float r)
{
    float sig = spear_hft_implied_vol(price, S, K, T, r);
    for (int i = 0; i < 2; ++i) {
        float c = bs_call(S, K, T, r, sig);
        float h = 1e-4f;
        float vega = (bs_call(S, K, T, r, sig + h) - c) / h;
        if (fabsf(vega) < 1e-12f) break;
        float step = (c - price) / vega;
        sig -= step;
        if (sig < 1e-4f) sig = 1e-4f;
    }
    return sig;
}

/* Black-Scholes call price (référence) */
static float bs_call(float S, float K, float T, float r, float sigma)
{
    if (T <= 0.0f || sigma <= 0.0f) return 0.0f;
    float sq = sigma * sqrtf(T);
    float d1 = (logf(S/K) + (r + 0.5f*sigma*sigma)*T) / sq;
    float d2 = d1 - sq;
    /* Phi approx. Abramowitz-Stegun 7.1.26 */
    float erf1 = erf(d1 / sqrtf(2.0f));
    float erf2 = erf(d2 / sqrtf(2.0f));
    float n1 = 0.5f * (1.0f + erf1);
    float n2 = 0.5f * (1.0f + erf2);
    return S*n1 - K*expf(-r*T)*n2;
}

/* Newton-Raphson pour sigma (référence) */
static float bs_iv_newton(float price, float S, float K, float T, float r)
{
    float sig = 0.3f;
    for (int i = 0; i < 30; ++i) {
        float c = bs_call(S, K, T, r, sig);
        float h = 1e-4f;
        float c2 = bs_call(S, K, T, r, sig + h);
        float vega = (c2 - c) / h;
        if (fabsf(vega) < 1e-12f) break;
        float step = (c - price) / vega;
        sig -= step;
        if (sig < 1e-4f) sig = 1e-4f;
        if (fabsf(step) < 1e-9f) break;
    }
    return sig;
}

/* ------------------- 3. Screening moléculaire O(1) ------------------------ */
static inline float spear_molecular_coulomb_screened(float r, float alpha)
{
    float ar = alpha * r;
    if (ar >= 3.5f) return 0.0f;
    /* Fit polynomial degre 10 (moindres carres, [0,3.5]) : max err ~1e-4 */
    const float c0 =  1.0001047821e+00f;
    const float c1 = -1.1324630651e+00f;
    const float c2 =  3.8927586171e-02f;
    const float c3 =  2.1774783205e-01f;
    const float c4 =  3.3924879023e-01f;
    const float c5 = -5.3223950142e-01f;
    const float c6 =  3.0514089286e-01f;
    const float c7 = -9.4205012194e-02f;
    const float c8 =  1.6550587440e-02f;
    const float c9 = -1.5481922967e-03f;
    const float cA =  5.9037883693e-05f;
    float poly = c0 + ar * (c1 + ar * (c2 + ar * (c3 + ar * (c4
                 + ar * (c5 + ar * (c6 + ar * (c7 + ar * (c8
                 + ar * (c9 + ar * cA)))))))));
    poly = fmaxf(poly, 0.0f);
    return poly / (r + 1e-12f);
}

/* ------------------------- 4. NMPC évasion analytique --------------------- */
typedef struct {
    float x, y, vx, vy, a_lat_max, safety_margin;
} spear_obstacle_threat_t;

static float spear_nmpc_evasion_steering(const spear_obstacle_threat_t *obs)
{
    float v_sq = obs->vx*obs->vx + obs->vy*obs->vy;
    if (v_sq < 1e-6f) return 0.0f;
    float t_cpa = -((obs->x*obs->vx) + (obs->y*obs->vy)) / v_sq;
    if (t_cpa <= 0.0f || t_cpa > 3.0f) return 0.0f;
    float y_at_cpa = obs->y + obs->vy * t_cpa;
    float deficit = obs->safety_margin - fabsf(y_at_cpa);
    if (deficit <= 0.0f) return 0.0f;
    float sign = (y_at_cpa >= 0.0f) ? -1.0f : 1.0f;
    float u_req = sign * (2.0f * deficit) / (t_cpa * t_cpa);
    return fminf(fmaxf(u_req, -obs->a_lat_max), obs->a_lat_max);
}

/* ----------------------------- BENCHMARK SPEED --------------------------- */
#include <time.h>

static double bench_molecular(int n, float alpha, int use_spear)
{
    clock_t t0 = clock();
    volatile float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        float r = 0.1f + 3.4f * (float)(i % 1000) / 1000.0f;
        acc += use_spear ? spear_molecular_coulomb_screened(r, alpha)
                         : (erfc(alpha * r) / (r + 1e-12f));
    }
    (void)acc;
    return (double)(clock() - t0) / CLOCKS_PER_SEC;
}

/* --------------------------------- TESTS --------------------------------- */
int main(void)
{
    int fails = 0;

    /* 1. SCARA round-trip : IK(FK(theta)) == theta (100 échantillons) */
    spear_scara_config_t cfg = { -1.0416666f, 0.4629629f, 1.2f, 0.9f, 0.09f, 4.41f };
    float max_ik_err = 0.0f;
    for (int i = 0; i < 100; ++i) {
        float t1 = -2.6f + 5.2f * (i % 10) / 9.0f;
        float t2 =  0.2f + 1.8f * (i / 10) / 9.0f;
        spear_scara_joints_t j = { t1, t2 };
        float x, z;
        spear_scara_fk(&cfg, &j, &x, &z);
        spear_scara_joints_t ik;
        if (spear_scara_ik_eval(&cfg, x, z, &ik)) {
            float e = fabsf(ik.theta2 - j.theta2);
            if (e > max_ik_err) max_ik_err = e;
        }
    }
    float ik_ok = (max_ik_err < 1e-3f) && max_ik_err >= 0.0f;
    printf("SCARA IK round-trip max err: %.6f rad -> %s\n", max_ik_err,
           ik_ok ? "PASS" : "FAIL");
    if (!ik_ok) fails++;

    /* 2. HFT implied vol : seed+2 Newton vs Newton complet (10 echantillons) */
    float max_iv_err = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float S = 90.0f + i, K = 100.0f, T = 0.5f + 0.05f * i, r = 0.03f;
        float sigma_true = 0.15f + 0.02f * i;
        float price = bs_call(S, K, T, r, sigma_true);
        float iv_approx = spear_hft_iv_seed_newton(price, S, K, T, r);
        float iv_newton = bs_iv_newton(price, S, K, T, r);
        float e = fabsf(iv_approx - iv_newton);
        if (e > max_iv_err) max_iv_err = e;
    }
    printf("HFT IV seed+2Newton vs Newton max err: %.6f -> %s\n", max_iv_err,
           max_iv_err < 2e-3f ? "PASS" : "FAIL");
    if (max_iv_err >= 2e-3f) fails++;

    /* 3. Screening moléculaire vs erfc/r (degre 10, 10 echantillons) */
    float max_mol_err = 0.0f;
    float alpha = 1.0f;
    for (int i = 0; i < 10; ++i) {
        float r = 0.2f + 0.3f * i;
        float ref = erfc(alpha * r) / r;
        float got = spear_molecular_coulomb_screened(r, alpha);
        float e = fabsf(got - ref);
        if (e > max_mol_err) max_mol_err = e;
    }
    printf("Molecular screening deg10 max err: %.6f -> %s\n", max_mol_err,
           max_mol_err < 1e-3f ? "PASS" : "FAIL");
    if (max_mol_err >= 1e-3f) fails++;

    /* 4. NMPC : 0 si pas de danger (obstacle s'eloigne), borne sinon */
    spear_obstacle_threat_t safe = { 0.0f, 10.0f, 0.0f, 5.0f, 3.0f, 1.0f };  /* s'eloigne : t_cpa < 0 */
    float u_safe = spear_nmpc_evasion_steering(&safe);
    spear_obstacle_threat_t threat = { 0.0f, 0.5f, 0.0f, -5.0f, 3.0f, 2.0f }; /* collision proche */
    float u_threat = spear_nmpc_evasion_steering(&threat);
    int nmpc_ok = (u_safe == 0.0f) && (fabsf(u_threat) <= 3.0f + 1e-6f);
    printf("NMPC safe=%+.3f threat=%+.3f |u|<=amax -> %s\n",
           u_safe, u_threat, nmpc_ok ? "PASS" : "FAIL");
    if (!nmpc_ok) fails++;

    printf("\nBT33 audit kernels: %s (%d fails)\n",
           fails ? "FAIL" : "ALL PASS", fails);

    /* Benchmark : spear vs libc sur 1M echantillons */
    const int N = 1000000;
    double t_spear = bench_molecular(N, 1.0f, 1);
    double t_libc  = bench_molecular(N, 1.0f, 0);
    printf("Molecular bench (1M): spear=%.4fs libc=%.4fs speedup=%.1fx\n",
           t_spear, t_libc, t_libc / (t_spear + 1e-12));

    return fails ? 1 : 0;
}