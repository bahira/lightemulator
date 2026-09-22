# Roadmap Complete — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform lightemulator d'un prototype rigoureux en projet open-source mature — refactor de la dette technique, portage WASM/npm, docs/cours, release v1.0.

**Architecture:** Monorepo implicite (émulateur React + GPT C + kernels). Le principe directeur reste : chaque chiffre mesuré, chaque affirmation vérifiée par une référence indépendante. Les refactors sont **neutres** (git diff vide ou tests inchangés) avant toute feature.

**Tech Stack:** TypeScript strict + Vite (émulateur), C99/AVX2 + OpenMP (GPT), esbuild (tests), GitHub Actions (CI), WASM SIMD v128 (portage navigateur).

**État audité (mesuré)** : 29/29 tests grounded · CI verte 22s · 10 labs · GPT 270k params @9300 tok/s · page live + demo déployées · 7 issues ouvertes (#4,5,6,8,9,10,12) · MIT.

---

## Phase A — Refactor de la dette (neutre, d'abord)

### Task 1: Générateur de kernels unique — finir la triplication

**Problème** : les mêmes formules (exp, tanh, gelu, sigmoid, silu) vivent en **3 copies** — `lm_c/lm_kernels.h`, `src/kernels/spear_kernels.h`, `src/physics/spear.ts`. Risque : divergence silencieuse entre copies (un coeff corrigé ici, pas ailleurs).

**Files:**
- Create: `tools/spear_emit.mjs` (émetteur : lit les constantes canoniques → injecte entre marqueurs)
- Modify: `lm_c/lm_kernels.h`, `src/kernels/spear_kernels.h`, `src/physics/spear.ts` (ajouter `// BEGIN GENERATED (tools/spear_emit.mjs) — NE PAS ÉDITER` / `// END GENERATED` autour des blocs de formules)

**Steps:**
1. Entourer les blocs de formules des 3 fichiers par les marqueurs GENERATED.
2. Écrire `spear_emit.mjs` : table de constantes (exp C1-C4, rsqrt A/B, gelu K/K0/c1-c3/A/B/C, sigmoid, silu) → regex-replace entre marqueurs dans les 3 formats (C float, TS const).
3. Exécuter → `git diff` doit être **vide** (refactor neutre, mêmes bytes).
4. `npm run test` (29/29) + `python src/kernels/spur_parity.py` (6/6) + rebuild C + auto-test.
5. Commit: `refactor: single-source kernel emission (neutral)`

**Effort** : 1-2 h. **Garantie** : toute future correction de coeff se fait à UN endroit.

### Task 2: Encoding sweep — zéro mojibake

**Problème** : caractères cp1252 corrompus (`�`) dans `validate.ts` (case opt-fresnel), `lm_model.h`, `lm_main.c` (strings printf), `lm_kernels.h` (header). Bloque les edits outillés et rend les diffs illisibles.

**Files:**
- Modify: `src/physics/validate.ts`, `lm_c/lm_model.h`, `lm_c/lm_main.c`, `lm_c/lm_kernels.h`

**Steps:**
1. `grep -c $'\ufffd'` sur chaque fichier source → liste des coupables.
2. Réécrire les lignes coupables en UTF-8 propre (les strings printf C peuvent passer en ASCII pur).
3. Vérifier : `rg '\ufffd' src/ lm_c/` → 0 match. `npx tsc --noEmit` OK. `npm run test` 29/29. Rebuild C + `--steps 20 --gradcheck` PASS.
4. Commit: `refactor: clean UTF-8 across sources (zero mojibake)`

**Effort** : 30-45 min.

---

## Phase B — v0.3 Performance & portage

### Task 3: Package npm des modules physics (#8)

**Files:**
- Modify: `package.json` (name `@bahira/spear-physics-core`, exports map, files, prepack)
- Create: `src/physics/index.ts` ( barrel : ré-exporte control, ising, spear, fep, fft, mzi, bpm, drones, optics, kan )

**Steps:**
1. Écrire `src/physics/index.ts` — barrel pur, zéro logique.
2. package.json : `"files": ["src/physics", "LICENSE", "README.md"]`, `"exports": { ".": "./src/physics/index.ts" }`, `"prepack": "npx tsc --noEmit && npm run test"`.
3. `npm pack` → installer dans `%TEMP%\opencode\pkgtest` → `import { ikScara } from '@bahira/spear-physics-core'` fonctionne.
4. `npm publish --access public`.
5. Commit: `feat: npm package @bahira/spear-physics-core (zero-dep physics modules)`
6. Issue #8 close avec le lien npm.

**Effort** : 1-2 h. **Vérification** : install propre hors repo + les 29 tests passent au prepack.

### Task 4: WASM kernels — mm_nt_t multi-acc dans le navigateur (#6)

**Files:**
- Create: `tools/emit_mm_wasm.mjs` (générateur de binaire WASM SIMD, pattern E8 du projet)
- Create: `src/kernels/mm_wasm.ts` (loader + bridge typé)
- Modify: `docs/index.html` (section Live GEMM Race : 3ᵉ lane WASM)

**Architecture** : WASM SIMD = v128 = **4 lanes f32** (vs AVX2 8). Le multi-acc garde son intérêt : 4 tuiles × 4 accs. Générer le binaire à la main (le projet sait déjà émettre du WASM — exports du MCP spear-kernels).

**Steps:**
1. Écrire l'émetteur : module WASM minimal (memory, export `mm_nt_t(M,K,N)` avec 4 accs v128 + boucle k + queue scalar).
2. Loader TS : `WebAssembly.instantiate` + fallback JS si échec.
3. Test : max|diff| WASM vs JS scalaire < 1e-6 sur les 5 shapes ; race JS vs WASM même protocole que la page.
4. Intégrer à la page live (3ᵉ lane) + SHOWCASES.md.
5. Commit: `feat: WASM SIMD kernels in browser (4-lane multi-acc)`
6. Issue #6 close.

**Effort** : 3-4 h, le plus risqué — fallback JS obligatoire. `ponytail:` si l'émetteur dérape : WASM sans SIMD d'abord (scalar 1-chaîne), SIMD ensuite.

### Task 5: Taille du modèle GPT (#5)

**Files:**
- Modify: `lm_c/lm_model.h:16` (`#define NLv 4` → `6`, ou DD 72→96 avec NH 3→4)

**Steps:**
1. Bump NLv → 6. Rebuild + auto-test GEMM + gradcheck (`--steps 20 --gradcheck`).
2. A/B **apparié** : NLv=4 vs NLv=6 en deux builds, runs intercalés min-of-3, `--steps 100` — le e2e simple est bruité (variance turbo ×3).
3. Documenter SHOWCASES.md §2 (params, tok/s, val loss).
4. Commit: `feat: GPT scale-up NLv=6 (paired A/B measured)`
5. Issue #5 close.

**Effort** : 30 min + 20 min de bench.

---

## Phase C — v1.0 Release sérieuse

### Task 6: CI C benchmarks (#10)

**Files:**
- Modify: `.github/workflows/ci.yml` (job `c-kernels` séparé)
- Create: `data/sample.txt` (échantillon 1 Mo commite — le corpus 18.5 Mo reste gitigné)

**Steps:**
1. Créer `data/sample.txt` = premiers 1 Mo de tinystories_valid.txt (reproductible : `head -c 1048576`).
2. Job CI `c-kernels` : ubuntu + apt gcc, `gcc -O2 -ffast-math -o lm_c/lm_train.exe lm_c/lm_main.c -lm` (OpenMP ubuntu ok), `lm_train.exe --steps 30 --gradcheck` + micro-bench kernels avec tolérances larges (variance des runners).
3. `DATA_PATH` : flag `--data` existe déjà (`lm_main.c:325` — hardcodé, ajouter le flag si absent).
4. CI verte sur push → summary avec les timings.
5. Commit: `ci: C kernel benchmarks on runners (sample data)`
6. Issue #10 close.

**Effort** : 1 h.

### Task 7: Docs site / cours (#9) + Release v1.0 (#12)

**Files:**
- Create: `docs/guide/index.html` (le cours 5 modules — plan déjà écrit : GPT en 500 lignes, backward exacte, kernels, GEMM multi-acc, la rigueur comme produit)
- Create: `CHANGELOG.md`
- Modify: `docs/index.html` (lien vers le guide)

**Steps:**
1. Guide statique dans le style de la page live (module 1 gratuit en preview — l'appât).
2. CHANGELOG.md : timeline (fondations → robot → drones → page live → open-source → v0.2 → v0.3).
3. `gh release create v1.0.0 --notes "<receipts>"` + tag.
4. Issue #9 + #12 close.
5. Commit: `docs: course guide + CHANGELOG + release v1.0.0`

**Effort** : 3-4 h.

---

## Use cases (la demande, pas la dette)

Le pipeline existe déjà (démo → vérification live → repo receipts). Les use cases que ce plan débloque :

1. **Enseignement universitaire** — les 10 labs comme simulateurs de cours (photonique, contrôle, optique). Débloqué par : docs/guide (Task 7) + licence MIT (fait).
2. **Cours dev « GPT from scratch in C »** — Gumroad $39 ou YouTube. Débloqué par : Task 7 (le guide = la vitrine du cours).
3. **Consulting kernel/HPC** — « GEMM ×3 sans BLAS, chiffres pour le prouver ». Débloqué par : Task 3 (package npm = crédibilité de distribution) + Task 4 (WASM = démonstration navigateur du kernel).
4. **SaaS photonique niche** — émulateur en abonnement pour labos. Débloqué par : Task 7 (docs) + galerie d'exemples (#4, restée ouverte volontairement).

## Ordre d'exécution recommandé

Phase A (Tasks 1-2) **d'abord** — les refactors neutres protègent tout le reste. Puis Task 3 (npm, le plus rapide à valeur de distribution). Puis Task 5 (model size, 30 min). Task 4 (WASM) quand les deux sont verts. Phase C en dernier.

**Skipped volontairement** : #4 galerie d'exemples (état complet par lab — toucher 10 labs, post-v1.0), analytics, domaine custom.
