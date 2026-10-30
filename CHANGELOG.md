# Changelog

Toutes les modifications notables de ce projet. Format basé sur [Keep a Changelog](https://keepachangelog.com/), semver.

## [1.2.0] — 2026-10-30

Perf, mesurée et falsifiable. Aucune régression de justesse : autotest GEMM `max|diff| = 0`,
gradcheck PASS (0.0253), 44/44 tests JS, loss identique à 3 décimales.

### Ajouté
- **GEMM 4 lignes M** : `mm_nt_t_avx2_mb4` (4M×16) et `mm_nt_t_avx512_mb4` (4M×64). Le noyau
  1-ligne relisait W une fois par ligne M (formes réelles : 62–83 kB, L2) — il était limité par
  le trafic, pas par les FMA. Résultat : **×1.53–2.37 vs le noyau 4×8**, ×38.5 vs scalaire.
- **Backward bloqué** : `mm_dxd_avx2_mb4` (4 lignes) et `mm_tndw_avx2_mb4` (4 colonnes), même
  raisonnement sur W (dxd) et X (tndw). GEMM end-to-end **×1.44** (6.9σ, 4 rondes appariées).
- **Dispatch 3 niveaux** : scalaire / AVX2 / AVX-512 (`__builtin_cpu_supports("avx512f")`),
  fonctions `target(...)` compilables partout (vérifié en `-march=x86-64`). `--no512` force le
  chemin AVX2 pour l'A/B.
- `bench/bench_mm_isa.c` : 5 variantes, rondes entrelacées, médiane, ratios appariés en σ, parité
  `max|diff|` vs scalaire. CI : parité vérifiée, tableau dans le job summary.

### Corrigé
- **`npm run smoke:dom` mourait en OOM** (6 min, 1 000 001 entrées User Timing). Le stub
  `requestAnimationFrame` se réarmait en `setTimeout(0)` : ~25 000 images/s sous jsdom, chaque
  image étant un render React, et React 19 (build dev) émet un `performance.measure` par render.
  Le stub simule maintenant un 60 Hz virtuel avec un **budget de frames par section** → 4,4 s,
  16 labs + shell + galerie + partage au vert.
- **`bench/bench_mm_isa.c` sur un runner sans AVX-512** : les variantes 512 sautaient le calcul
  mais entraient quand même dans les totaux (5 000 000 000 ms affichés, « meilleur » faux). Elles
  sont désormais omises ligne et total. `LM_NO512=1` reproduit ce profil à volonté ; la CI le
  mesure aussi.
- Ancien fichier `lm_c/bench_mm_isa.c` déplacé en `bench/` (les benchmarks ne sont pas des sources
  du moteur).

### Mesuré (et ce que ça corrige)
- **AVX-512 : la claim du README était fausse à moitié.** Porter le *même* noyau 1-ligne en
  16 voies est **plus lent** (×0.87 : pression ports 512 bits + downclock). C'est le **blocage
  4 lignes** qui paie — la largeur SIMD n'a jamais été le goulot.
- Débit d'entraînement : 11,8–14,6 k → **15,6–19,1 k tok/s** (×1.28, 5 rondes appariées, 5.0σ).
- GEMM wall time (40 steps, 5 rondes appariées) : 553–590 ms → **361–383 ms**
  (×1.44 médian, 6.9σ ; 1.28–1.85 selon les rondes).

[1.2.0]: https://github.com/bahira/lightemulator/releases/tag/v1.2.0

## [1.1.0] — 2026-10-30

Aucune régression de physique (44/44 tests inchangés). Cette version rend la démo
**partageable**, la publication **reproductible** et les chiffres **re-mesurables**.

### Ajouté
- **Galerie d'exemples partageables** (`src/gallery.ts` + `src/ui/Gallery.tsx`) : 10 scènes
  réglées (coupleur, maillage 6×6 et 18×18, Ising au-dessus/sous le seuil, zéro de dispersion,
  HOM, CHSH, Gerchberg–Saxton 80 itérations). Chaque carte = une URL.
- **État dans l'URL** : `useHashNumber` (kit) + `#lab=<id>` épinglé par le shell. Les paramètres
  principaux de BPM, MZI, Ising, dispersion, quantique et holographie sont désormais exportés ;
  un lien partagé rouvre le bon lab avec la bonne scène. Bouton **⧉ partager**, raccourci `g`.
- `npm run smoke:dom` — les 16 labs sont montés **et dessinés** dans jsdom (canvas factice),
  plus des tests de la galerie, du partage et du deep-link.
- `npm run verify:inference` — assets wasm, dimensions, génération, effectivité de la quantification.
- `npm run claims` / `claims:write` — re-mesure sur la machine locale et écrit `CLAIMS.md`
  (provenance datée : CPU, OS, Node, gcc).
- `npm run docs:sync` / `docs:check` — la démo publiée est un build reproductible, et la CI échoue
  si `docs/` diverge des sources.
- Workflow **`pages.yml`** : à chaque push sur `master`, `docs/` est reconstruit et commité
  (Pages reste en mode legacy `/docs`, aucun réglage de dépôt à changer).

### Corrigé
- **Lab « Inference edge » opérationnel** : `lm_checkpoint.bin` / `lm_vocab.bin` n'avaient jamais
  été livrés ; le fallback SPA renvoyait `index.html` sur ces 404, donc le lab lisait du HTML comme
  checkpoint. Un checkpoint réel **V=104 / 270 000 params** (500 steps, val loss 2,977) est livré,
  et le chargement vérifie `content-type`, `r.ok`, `V == |vocab|` et la taille exacte.
- `docs/demo.html` n'était rattachable à aucune commande : il l'est (`docs:sync`).
- 0 vulnérabilité npm (vite 7.3.2 → 7.3.7, esbuild, source-map-js).

### Modifié (hygiène)
- Racine dégraissée : 24 fichiers de bench/harnais → `bench/`, 5 prototypes → `experiments/`
  (avec README), `experiment/` supprimé (doublon de `src/`, sous-ensemble strict).
- README : provenance datée des mesures + commandes de vérification.

[1.1.0]: https://github.com/bahira/lightemulator/releases/tag/v1.1.0

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
