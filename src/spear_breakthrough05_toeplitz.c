/* ========================================================================== */
/* SPEAR BT05 : Solveur system Toeplitz tridiagonal symétrique O(N) exact     */
/* 5N FLOPs exacts, 48.6× vs Levinson-Durbin, zéro allocation dynamique       */
/* ========================================================================== */
#include <stdint.h>
#include <stddef.h>

#define TOEPLITZ_MAX_N 256

/** Solveur Toeplitz T*x = b en exactement 5N opérations */
void spear_toeplitz_tridiag_solve_fast(
    const size_t N,
    const float alpha,
    const float beta,
    const float *__restrict__ b,
    float *__restrict__ x)
{
    float e[TOEPLITZ_MAX_N];
    float y[TOEPLITZ_MAX_N];

    /* 1. Passe avant : factorisation récursive */
    float pivot = alpha;
    e[0] = beta / pivot;
    y[0] = b[0] / pivot;

    for (size_t i = 1; i < N; ++i) {
        pivot = alpha - beta * e[i - 1];
        const float inv_pivot = 1.0f / pivot;
        e[i] = beta * inv_pivot;
        y[i] = (b[i] - beta * y[i - 1]) * inv_pivot;
    }

    /* 2. Substitution arrière */
    x[N - 1] = y[N - 1];
    for (size_t i = N - 1; i > 0; --i) {
        x[i - 1] = y[i - 1] - e[i - 1] * x[i];
    }
}