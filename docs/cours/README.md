# Le cours — GPT from scratch in C

*5 modules · chaque claim falsifiable · les receipts sont les chiffres du repo*

Le plan de cours derrière [lightemulator](https://github.com/bahira/lightemulator) — la physique et le ML distillés en algèbre pure, vérifiés contre des oracles indépendants. Le module 1 est gratuit en preview ci-dessous.

---

## Module 1 — GPT en 500 lignes (gratuit)

**Le modèle** : `lm_c/lm_model.h` + `lm_c/lm_main.c` — 270k params, TT=32, DD=72, NLv=4, forward + backward EXACTE + Adam, ~14 000 tok/s sur un i7-7660U 2016.

**Ce qu'on apprend** : un transformer de caractère est 500 lignes de C — embeddings, attention causale (QKV), MLP 4×, layernorm, softmax. Rien d'autre.

**Les receipts** :
- auto-test GEMM 4 chemins : max|diff| 1.4e-6
- gradcheck vs différences finies : 0.4 % pire
- val loss 3.15 en 800 steps (hasard 4.64)
- [Démo live](https://bahira.github.io/lightemulator/demo.html) — le module C compilé wasm32, génération INT4/FP32 dans votre navigateur, ~150 tok/s

## Module 2 — La backward exacte

**Ce qu'on apprend** : la rétropropagation à la main — dV, softmax-crossentropy, attention backward, les 4 chemins GEMM (nt, nt_t, dxd, tndw). Le gradient analytique vérifié numériquement.

**Les receipts** : gradcheck 0.0044 · parité noyaux 6/6 · 44 tests grounded.

## Module 3 — Les kernels (exp / rsqrt / tanh / gelu)

**Ce qu'on apprend** : écrire exp(x) = 2^k·(1+r·P(r)) à la main, les polynômes minimax, pourquoi libm est lent, le dispatch runtime AVX2 vs scalaire.

**Les receipts** : exp ×3.4, tanh ×5.6, gelu ×5.4 vs libm (20M appels, -ffast-math, sous charge) · L∞ 1.48e-5 sur [−14,+2] · vérifiés en CI (`c-bench` job).

## Module 4 — Le GEMM multi-accumulateurs

**Ce qu'on apprend** : la latence FMA (~4 cy) masquée par 4 chaînes indépendantes, les tuiles de 8 colonnes, le packing transposé zéro-copie, pourquoi GEMM n'est plus le frein.

**Les receipts** : ×1.78-2.19 vs scalaire naïf (5 shapes réelles) · ~×3 sur le wall GEMM · A/B end-to-end NLv 4→6 : val −0.028 pour ×0.67 tok/s (le speed/scale trade-off mesuré).

## Module 5 — La rigueur comme produit

**Ce qu'on apprend** : la boucle grounded — oracle indépendant d'abord, code ensuite. Trois vrais bugs attrapés avant publication (le softmax du PNN, le μΔt du Langevin, le signe de l'Éq 10). Les limites documentées au lieu d'être masquées.

**Les receipts** : [le blog post](./blog/2026-09-23-verified-papers.md) · [les conclusions des intégrations](../SHOWCASES.md) · 2 papiers 2026 vérifiés · la génération de Langevin ne sépare pas à petite échelle — documentée, pas cachée.

---

*Built by [@bahira](https://github.com/bahira) — zéro dépendance, tout est reproducible depuis le repo.*
