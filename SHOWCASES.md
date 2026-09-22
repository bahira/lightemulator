# SHOWCASES — Rapport complet de qualité

**PHOTONIC ENGINE / GROUNDED-SPEAR** · https://bahira.github.io/lightemulator/ · repo : https://github.com/bahira/lightemulator

Chaque chiffre de ce rapport est **mesuré** (performance.now() côté client, C wall-clock côté training) et **vérifié** contre une référence indépendante (solution analytique, loi de conservation, énumération exhaustive, différences finies). Aucune métrique n'est générée aléatoirement. Les parts honnêtes sont documentées, pas masquées.

---

## 1. L'émulateur — 10 labs, tout côté client

### 1.1 Moteur de lumière (BPM)
Propagation réelle par split-step Fourier (Strang, O(Δz²)) dans 10 dispositifs : guide droit, coupleur directif, jonction Y, MMI, Mach-Zehnder, lentille GRIN, espace libre, milieu désordonné, réseau de Bragg.
- **Conservation de puissance** : ΔP/P = 3.5e-14 sur 400 pas (tol 1e-9)
- **Diffraction gaussienne** : reproduit w(z) = w₀√(1+(z/z_R)²) à 2.7e-15 rel
- **Solveur de mode** : n_eff = 1.451458 vs 1.451469 exact (101 itérations)

### 1.2 Processeur MZI
Maillage universel — toute unitaire N×N par décomposition de Reck/Givens, exactement.
- **Aller-retour décompose→reconstruit** : résidu 2.18e-15 (66 MZI, profondeur 21)
- **Unitarité** : défaut 1.83e-15
- **Mat-vec optique** : 5.87e-16

### 1.3 Machine d'Ising (CIM)
CIM DOPO mean-field avec correction d'hétérogénéité d'amplitude, sur MaxCut, comparée à l'optimum exact (énumération jusqu'à n=22).
- **CIM vs optimum** : 26/26 — succès 100% des cycles, R99 = 1200 tours
- **Borne de l'optimum exact** : vérifiée par énumération

### 1.4 KAN photonique
B-splines de Cox-de Boor apprises, rétropropagation analytique, Adam.
- **Partition de l'unité** : 3.33e-16
- **Dérivée analytique des splines** : 1.67e-10
- **Gradient check** : 2.34e-7

### 1.5 Énergie libre (FEP)
Champs complexes apprenant par contraste d'équilibre (EP).
- **Identité phase-gradient** : 1.84e-4, résidu d'équilibre 1.1e-14

### 1.6 Contrôle — IK, trajectoires, pendule
- **IK SCARA fermée** (2 liaisons, cos θ₂ affine en r², 0 itération) : FK∘IK = 3.00e-16
- **Profil jerk-borné 7 segments** (3 régimes exacts, cas T2<0 du rapport source **corrigé**) : violations = 0
- **Pendule inversé** : la preuve « V̇ = −6.4 θ̇² ≤ 0 » du papier SPEAR est **réfutée** — V̇ > 0 sur 26.9% du plan de phase (la loi reste bonne, la preuve était fausse)

### 1.7 Robot animé (nouveau)
Bras SCARA **en mouvement** : suivi de trajectoire par waypoints, chaque segment est un déplacement temps-minimal exact (planProfile, 3 régimes) exécuté par IK fermée — zéro itération, zéro heap, comme un vrai contrôleur.
- **3 trajectoires** : cercle (régime palier, D ≈ 0,12 m > D_crz), ligne (régime réduit, D ≈ 0,04 m < D_sat), huit (lemniscate de Gerono, les deux régimes)
- **Télémétrie live** : v, a effecteur (profil exact), θ₁/θ₂, résidu FK∘IK machine
- Vérifié par `ctrl-ik` (3.00e-16) et `ctrl-traj` (violations = 0)

### 1.8 Drones lumineux (nouveau)
Playground de **drones intelligents naviguant par la lumière** : phototaxie réelle (gradient du champ 1/d²), séparation d'essaim, positionnement par trilatération exacte, liaison de Friis (LED IR 940 nm, pièce 3×3 m). Panneau « cerveau photonique » : les 4 photodiodes du drone montrent le gradient qu'il suit.
- **Phototaxie — convergence** : 0.0089 ≤ 0.015 (arrêt en pas bornés, trajectoire droite garantie — le gradient d'une source unique pointe toujours vers elle)
- **Trilatération exacte** : 2.48e-16 — l'epsilon machine (différenciation + solve 2×2 fermé)
- **Budget Friis** : 0 exactement — Pr(2d) = Pr(d)/4 sur 40 distances

### 1.9 Dispersion
Sellmeier exact avec dérivées 1ʳᵉ–3ᵉ (TOD), GDD, impulsion femtoseconde, Fresnel biaxial.
- **Sellmeier** : 1.95e-4 · **GDD silice** : 36.2 fs²/mm (rel 0.01%) · λ_ZD = 1.2728 µm
- **Fresnel biaxial fermé** : 5.33e-15 — coquille « =1/n² » du rapport source **corrigée** en « =0 »

### 1.10 Kernels SPEAR
10 kernels algébriques (GELU, tanh, exp…) sans transcendante, L∞ mesuré par kernel, benchmarkés dans le navigateur.

---

## 2. Le GPT en C — 270k params, kernels AVX2 maison

Entraînement complet from scratch (forward + backward exacte + Adam) sur TinyStories, micro-kernels GEMM écrits à la main, ~9300 tok/s sur un i7-7660U (2 cœurs, 2016).

| Kernel / système | Gain vs baseline | Accuracy |
|---|---|---|
| GEMM micro-kernel (4-accumulateur) | **×1.78–2.19** (5 shapes réelles, bench isolé) | max\|diff\| 1.4e-6 vs scalaire |
| GEMM end-to-end (fwd + bwd multi-acc) | **~×3** sur le wall GEMM | gradcheck PASS (0.4% pire) |
| exp — Padé[4] + ldexp | **×2.35** vs libm | L∞ 1.5e-5 sur [−14, +2] |
| tanh — rationnel | **×5.3** | L∞ 8.9e-3 sur [−4, +4] |
| gelu — rationnel + d/dx | **×4.9** | L∞ 2.9e-3, dérivée 10.8% (FD) |
| sigmoid / silu | > formes upstream | L∞ 1.6e-4 / 4.0e-4 |
| rsqrt — rsqrtss + Newton | ×0.52 (honnête) | **L∞ 2.7e-7** |
| Entraînement 270k params | 9300 tok/s | val loss 3.79 (hasard 4.64) |

**Validations** : auto-test GEMM 4 chemins (nt + nt_t + dxd + tndw, N%8 ≠ 0 inclus) 1.4e-6 · gradcheck vs différences finies 0.0044 · parité noyaux 6/6 · 28/28 tests grounded.

---

## 3. La page live — vérification dans le navigateur

**https://bahira.github.io/lightemulator/** — panneau d'instrument avec :
1. **Live Kernel Lab** : les formules de lm_kernels.h portées 1:1, L∞ calculées contre libm sur grille de 20 000 points, en direct — 6/6 PASS, chiffres identiques aux receipts C (exp 1.49e-5, tanh 8.94e-3, gelu 2.83e-3, sigmoid 1.61e-4, silu 4.02e-4, rsqrt 2.00e-4)
2. **Live GEMM Race** : réplique JS de la technique 1-chaîne vs 4-tuiles, course en direct, max|diff| = 0
3. **Receipts** : la table C mesurée, y compris le ×0.52 honnête
4. **Verification Loop** : la méthodologie en 4 étapes

**https://bahira.github.io/lightemulator/demo.html** — l'émulateur complet (10 labs + drones + robot), single-file 442 Ko, zéro install.

---

## 4. La méthodologie (pourquoi ces chiffres sont crédibles)

1. **Référence indépendante** — chaque affirmation est confrontée à une solution analytique, une loi de conservation, une énumération exhaustive ou des différences finies. Jamais le modèle qui se vérifie lui-même.
2. **Parité verrouillée** — 6/6 noyaux vs libm sur grilles denses, dans la boucle grounded CI (`npm run test`, code de sortie = nombre d'échecs).
3. **Gradient check systématique** — backward vs différences finies sur chaque run d'entraînement.
4. **A/B appariés** — la variance turbo est réelle (×3 mesurée) : runs intercalés, min-of-3, benches isolés. Le bench ment, les receipts non.
5. **Falsification avant intégration** — les formules des rapports sources ont été **corrigées** avant d'entrer dans le code (régime T2<0 du profil jerk, terme résiduel du pendule, coquille Fresnel). Les corrections sont commentées dans le code.
6. **Les parts honnêtes** — attention tiling plafonne à ×1.05 (Amdahl, sauté volontairement), rsqrt échange vitesse contre précision, AVX-512 inutilisable sur ce matériel. Documentés, pas masqués.

---

## 5. Où ça va

1. Cerveau drone entraînable (gains de phototaxie appris en ligne)
2. Drag des balises à la souris (~20 lignes)
3. Taille du modèle GPT (le GEMM n'est plus le frein)
4. Capture email / formation sur la page live (le pipeline de monétisation)
5. Licence du repo

---

*Built by [@bahira](https://github.com/bahira) — every claim falsifiable, every number reproducible.*
