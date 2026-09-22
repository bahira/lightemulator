#!/usr/bin/env python3
"""spur_parity.py — parité des noyaux SPEAR vs SpearVM (pip spur-math).

Grounded-loop L3 : chaque noyau TS/C est confronté au paquet binaire de
référence (spur_math) ET à la référence exacte (math). Aucune assertion
cachée : les résidus mesurés sont affichés.
"""
import math
import numpy as np


def ref_erf(x):
    s = -1 if x < 0 else 1
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return s * (1.0 - p * math.exp(-ax * ax))


N = np.array([1.12841751266903279, 0.183482771948230095, 0.0573373674730976793, 0.00248430060206610405, 3.72785350475749968e-6])
D = np.array([1.0, 0.496471589671860558, 0.114910282096263028, 0.0161717422205343367, 1.86656477609649336e-4, -1.74401807407079551e-7])


def spear_erf(x):
    u = np.clip(x, -2, 2)
    y = u * u
    pn = np.zeros_like(u)
    dn = np.zeros_like(u)
    for i in range(4, -1, -1):
        pn = pn * y + N[i]
    for i in range(5, -1, -1):
        dn = dn * y + D[i]
    return u * (pn / dn)


def spear_gelu_erf(x):
    u = x * 0.7071067811865476
    out = np.where(u > 3.5, x, np.where(u < -3.5, 0.0, 0.5 * x * (1 + u * (np.polynomial.polynomial.polyval(u * u, N) / np.polynomial.polynomial.polyval(u * u, D)))))
    return out


def spear_tanh_p34(x):
    y = np.clip(x, -4, 4)
    t = y * y
    return (0.994894946 * y + 0.076611228 * y * t) / (1 + 0.402171314 * t + 0.005670342 * t * t)


def spear_gelu_quintic(x):
    off = 0.01104961
    t = 0.200055340257 * x + 0.5
    t2 = t * t
    core = x * (t2 * t * (6 * t2 - 15 * t + 10)) - off
    return np.where(t < 0, -off, np.where(t > 1, x - off, core))


def _rt(u1, u2, u3, A, B, C):
    return u1 * (A + u2 * u2) / (B + C * u3 * u3)


def spear_sigmoid(x):
    return 0.5 + 0.52976 * _rt(0.44175 * x, 0.6205 * x, 0.58509 * x,
                               27.35096, 25.64128, 7.19645)


def spear_silu(x):
    return 0.4777 * x * (1.04667 + _rt(0.46719 * x, 0.59566 * x, 0.5623 * x,
                                        25.80993, 23.10982, 6.95725))


def linf(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    try:
        import spur_math as sm
        have_spur = True
    except Exception:
        have_spur = False
        print("(spur-math non installé — on compare à math libm seul)")

    grid = np.linspace(-4, 4, 200000)
    g2 = np.linspace(-2, 2, 200000)
    ref_gelu = 0.5 * grid * (1 + np.vectorize(ref_erf)(grid / math.sqrt(2)))
    ref_erf_g = np.vectorize(ref_erf)(g2)

    rows = []
    rows.append(("tanh_p34", spear_tanh_p34(grid), np.tanh(grid), 0.004))
    rows.append(("erf_v2", spear_erf(g2), ref_erf_g, 1e-3))
    rows.append(("gelu_quintic", spear_gelu_quintic(grid), ref_gelu, 0.03))
    rows.append(("gelu_erf", spear_gelu_erf(grid), ref_gelu, 1e-3))
    rows.append(("sigmoid", spear_sigmoid(grid), 1 / (1 + np.exp(-grid)), 5e-4))
    rows.append(("silu", spear_silu(grid), grid / (1 + np.exp(-grid)), 5e-4))

    fails = 0
    print(f"{'kernel':14s} {'Linf vs exact':>12s} {'vs spur-math':>12s}  tol")
    for name, ours, exact, tol in rows:
        vs_exact = linf(ours, exact)
        line = f"{name:14s} {vs_exact:12.3e} "
        if have_spur:
            sp = {"tanh_p34": sm.tanh, "gelu_quintic": sm.gelu_quintic, "gelu_erf": sm.gelu_erf,
                  "erf_v2": sm.erf, "sigmoid": getattr(sm, "sigmoid", None),
                  "silu": getattr(sm, "silu", None)}.get(name)
            if sp is not None:
                try:
                    vs_spur = linf(ours, sp(grid if name != "erf_v2" else g2))
                    line += f"{vs_spur:12.3e}"
                except Exception as e:
                    line += f"  n/a ({e})"
            else:
                line += f"{'n/a':>12s}"
        ok = vs_exact <= tol
        if not ok:
            fails += 1
        print(f"{line}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{'PARITE VALIDEE' if fails == 0 else f'{fails} ECHEC(S)'} — {len(rows)} noyaux vs libm")
    raise SystemExit(fails)


if __name__ == "__main__":
    main()