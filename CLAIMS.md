# CLAIMS — re-mesuré sur cette machine

Généré par `npm run claims -- --full` le 2026-10-30.

| environnement | valeur |
|---|---|
| CPU | Intel(R) Xeon(R) Processor @ 2.60GHz |
| cœurs | 2 |
| OS | linux 6.1.158+ |
| Node | v22.22.3 |
| gcc | 12.2.0 |

| affirmation | mesuré ici | source |
|---|---|---|
| boucle grounded (tests) | **44/44 PASS en 1273 ms** | npm run test |
| exactitude kernel exp | **L∞ 1.492e-05** | lm_c/verify_kernels.c |
| exactitude kernel rsqrt | **L∞ 2.709e-07** | lm_c/verify_kernels.c |
| exactitude kernel gelu | **L∞ 2.912e-03** | lm_c/verify_kernels.c |
| exp8 AVX2 vs libm expf | **×6.08** | 20M appels, lm_c/lm_bench_exp8.c |
| exp scalaire vs libm expf | **×1.00** | dépend du CPU — voir README |
| exp8 AVX2 — erreur | **L∞ 1.484e-05** | tolérance CI 1e-3 |
| inférence wasm — débit | **275 tok/s** | V=104, 270000 params, 52.2 kB |
| build web (single-file) | **563 kB** | dist/index.html (npm run build) |
| gradcheck (FD vs backward analytique) | **erreur moyenne 0.0253** | 40 paramètres |
| training smoke 60 steps (32 kB) | **val loss 3.566** | hasard ≈ 4,08 |

> Les gains des kernels dépendent du CPU, du compilateur et des flags. Un chiffre
> du README qui ne se reproduit pas ici n'est pas faux : il est daté. C'est la
> raison d'être de ce fichier.
