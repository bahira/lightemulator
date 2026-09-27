# Changelog

Toutes les modifications notables de ce projet. Format basé sur [Keep a Changelog](https://keepachangelog.com/), semver.

## [1.0.0] — 2026-09-27

Premier tag de série. Tout est vérifié par la boucle grounded (44 tests), reproduit par la CI (2 jobs : `test` + `c-bench`).

### Physics — 15 labs, modules zéro-dépendance
- **Propagation** : BPM split-step Fourier, maillage MZI (unitaire N×N exacte), machine d'Ising (CIM vs optimum exact), KAN photonique, énergie libre (FEP), optique (Sellmeier, GDD/TOD, Fresnel biaxial fermé), contrôle (IK fermée, trajectoire jerk-bornée, pendule inversé).
- **Playground & showcases** : drones lumineux (phototaxie 1/d², trilatération exacte, budget de Friis), robot sur rail (SCARA, waypoints, plan jerk).
- **Papiers 2026 vérifiés** :
  - PNN — *Nature Com. 17, 1059* : superposition N+C (linéarité 2.12e-16), gradient AVM (1.76e-5 vs FD), entraînement 100% sur tâche séparable.
  - Langevin — *arXiv:2506.15121* (Whitelam, LBNL) : gradient inverse Éqs 10-11 exact (1.13e-6), relation de fluctuation ordre 1 (ratio 0.029), objectif sous l'entropie du bruit. Limite d'échelle documentée.
- **Nouveauté v1.0** : optique quantique (états de Fock, HOM, CHSH), holographie (Gerchberg-Saxton), **Opto-Transformer** (attention MZI + FFN rationnel), **inference edge** (le module C compilé wasm32-wasip1, SIMD v128, ~150 tok/s dans le navigateur, quantification FP32/E8/INT8/INT4 depuis un vrai checkpoint).
- **AETHERFALL** : JRPG action zéro-asset (exécutable dans le navigateur).

### Modèle GPT (C)
- Mini-GPT 270k params (TT=32, DD=72, NLv=4), forward + backward exacte + Adam, ~14 000 tok/s (i7-7660U 2016).
- GEMM micro-kernels 4-accumulateurs (AVX2, dispatch runtime) : ×1.78-2.19 vs scalaire.
- Kernels d'activation SPEAR : exp ×3.4, tanh ×5.6, gelu ×5.4 vs libm (20M appels, L∞ 1.48e-5).
- Gradcheck vs différences finies 0.0044 ; auto-test GEMM 4 chemins max|diff| 1.4e-6.
- A/B apparié NLv 4→6 : val 3.122 vs 3.150 (−0.028) pour ×0.67 tok/s → NLv=4 par défaut.
- Export wasm32 (52.2 kB, SIMD v128) : génération offline/privée dans le navigateur.

### Distribution
- `@bahira/spear-physics-core` v0.1.0 : 14 modules namespaced, types .d.ts, zéro dépendance (npm publish en attente de commit).
- Licence MIT, CONTRIBUTING, code de conduite, templates d'issues.

### Qualité & outillage
- **Boucle grounded** : 44 tests, chaque affirmation adossée à un oracle indépendant (FD, forme fermée exacte, identité machine).
- **CI** : 2 jobs — `test` (tsc + 44 tests) et `c-bench` (exactitude kernels + microbench exp8 avec tolérance L∞ < 1e-3 + training C smoke 60 steps sur échantillon 32 KB commité). Timings publiés dans le job summary.
- **Cours** : `docs/cours/README.md` — 5 modules, module 1 jouable en ligne.
- **Documentation** : SHOWCASES.md, blog post, conclusions des intégrations, rapports de performance.

### Limites documentées (non masquées)
- La génération bruit→structure du Langevin ne sépare pas du hasard à échelle réduite (784+512 requis).
- La quantification INT4 est simulée (poids fp32 en mémoire) : la claim est qualité-par-bit, pas taille mémoire.
- Le modèle 270k est char-level : texte pseudo-cohérent, pas un LLM utile.

[1.0.0]: https://github.com/bahira/lightemulator/releases/tag/v1.0.0
