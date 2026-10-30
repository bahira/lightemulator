# Vérification — 2026-10-30

Tout ce qui suit a été **exécuté dans ce checkout**, sur la branche `arena/f19b90c9-lightemulator`
(commit de départ `33244ca`, `master`). Les chiffres sont ceux de cette machine, pas des copies du README.

Environnement : Node 22.22.3 · npm 10.9.8 · gcc 12.2.0 (AMD64, `avx2` + `avx512f` + `fma`) · Linux.

---

## 1. Verdict en une ligne

**Le projet fonctionne.** Les 44 tests passent, le build passe, le C compile et s'entraîne, les 16 labs se montent,
le module wasm d'inférence tourne — **à une exception réelle près** : le lab *Inference edge* était cassé faute
d'assets livrés (corrigé ici, voir §4).

| Étage | Commande | Résultat |
|---|---|---|
| Dépendances | `npm install` | 96 paquets, 3 s |
| Typage | `npx tsc --noEmit` | 0 erreur |
| Boucle grounded | `npm run test` | **44/44 PASS** en 1,5–1,8 s |
| Build web | `npm run build` | 68 modules → `dist/index.html` 567 kB (181 kB gzip), single-file |
| Labs headless | `npm run smoke:labs` *(ajouté)* | **16/16** labs + shell |
| Kernels C | `gcc … verify_kernels.c && ./lm_verify` | exp L∞ 1,49e-5 · rsqrt 2,71e-7 · gelu 2,91e-3 → OK |
| Microbench exp8 | `./lm_bench` | libm 20,1 ms → **SPEAR AVX2 x8 : 4,3 ms (×4,70)** |
| Entraînement C | `./lm_train --data ci-data/… --steps 60` | loss 4,078 → 3,85 (ema) · val 3,566 · 16–18 k tok/s |
| Gradcheck | `./lm_train --gradcheck` | 40 params · erreur moyenne norm. **0,0253** · pire abs 0,0108 → PASS |
| Auto-test GEMM | (au démarrage) | avx2 vs scalaire `max|diff| = 0` → OK |
| Inférence wasm | `npm run verify:inference` *(ajouté)* | 270 000 params · 236–272 tok/s · quantification effective |
| CI GitHub | `gh run list` | 3 derniers runs `master` : **success** · release `v1.0.0` publiée |

Les 44 tests couvrent Reck/MZI, Ising vs optimum exact, KAN, FEP, contrôle, dispersion, drones, PNN, Langevin,
holographie GS, optique quantique (HOM, CHSH 2√2 à 4,4e-16), Opto-Transformer. Aucun test n'est vide : chacun
compare à un oracle indépendant (forme fermée, conservation, FD, énumération) avec une tolérance numérique.

---

## 2. Ce qui a été mesuré (et qui diverge du README)

Le README annonce des gains mesurés sur un i7-7660U de 2016. Ici, sur un CPU récent, **certains gains ne se
reproduisent pas** — c'est cohérent avec la philosophie du projet (« honnête »), mais ça mérite une note :

| Claim README | Mesuré ici | Lecture |
|---|---|---|
| `exp` ×2,35 vs libm (scalaire) | **×0,96–0,98** | glibc a une `exp` très rapide ici ; la version SPEAR scalaire ne gagne pas |
| `rsqrt` ≈ ×0,52 (assumé) | ×0,67–0,70 | conforme à l'esprit du README (plus lent, plus précis) |
| `tanh` ×5,3 | ×2,26–2,39 | gain réel, amplitude dépendante du CPU |
| `gelu` ×4,9 | ×4,33–4,39 | conforme |
| `exp8` AVX2 (8 voies) | **×4,70** | le gain vient du vectoriel, pas du scalaire |
| ~9 300 tok/s training | **16–19 k tok/s** | machine plus récente |
| ~150 tok/s wasm | **236–272 tok/s** | idem |
| val loss 3,79 | 3,57 (60 steps, échantillon 32 kB) | pas comparable : corpus différent |

**Conséquence pratique** : les nombres du README devraient être étiquetés *« mesuré sur i7-7660U, gcc X, date Y »*.
Le projet le fait déjà en partie ; c'est la seule claim qui vieillit mal (`exp`).

Autre point : sur 3 modes de quantification testés avec des **poids aléatoires**, les sorties étaient identiques
au token près. Vérification faite : ce n'est pas un bug — avec des logits quasi uniformes le tirage est dominé par
la seed. Avec le **vrai checkpoint**, FP32 / E8 / INT8 / INT4 divergent bien (voir §4).

---

## 3. Ce qui ne marchait pas

### 3.1 `InferenceLab` (« Inference edge ») — cassé en production

Le lab appelle :

```ts
fetch('./lm_infer.wasm')      // ✔ présent (52,2 kB)
fetch('./lm_checkpoint.bin')  // ✘ introuvable dans public/ ET docs/
fetch('./lm_vocab.bin')       // ✘ introuvable (le fichier s'appelle vocab.bin)
```

Le serveur (Vite, comme GitHub Pages) répond `index.html` en `200 text/html` sur un 404 : le code lisait donc
1 000 octets de HTML comme un checkpoint et affichait `checkpoint incompatible (V=…, params=…)`. Sur la démo
publiée, le lab était **mort en silence** — sans erreur visible pour l'auteur.

En prime, la vérification était codée en dur sur `V !== 104`, et le vocabulaire livré (`vocab.bin`, 104 octets)
correspond à un **corpus complet** alors que les checkpoints régénérables depuis `ci-data/` ont V=59 → même en
recréant un checkpoint, le couple vocab/checkpoint était incompatible.

### 3.2 `docs/demo.html` impossible à reproduire

Le build Vite produit `dist/index.html`, identique octet pour octet à `docs/demo.html` — mais aucune commande,
aucune CI ne documentait la copie. Après une correction du code, la démo publiée restait donc périmée.

### 3.3 Aucun test d'UI

Les 44 tests couvrent la physique ; **les 16 labs n'étaient validés par rien**. Deux labs (`Ising`, `Dispersion`)
touchent `window.location.hash` via `useHashState` (état partageable dans l'URL) et échouent hors DOM.

### 3.4 Hygiène

- 3 alertes `npm audit` : `vite` (≤7.3.6), `esbuild`, `source-map-js` — **dev uniquement** (dev-server Windows,
  DoS source-map). Fix : `vite@7.3.7`.
- Racine encombrée : `test_debug.c`, `test_debug2.c`, `test_debug3.c`, `test_debug4.c`, `bt29_quick_bench.py`,
  `benchmark_output.txt`, `debug_bn*.c`… ~20 fichiers de session qui n'ont rien à faire à la racine.
- `experiment/` duplique 5 labs + 6 modules physics de `src/` (risque de divergence silencieuse).

---

## 4. Corrections apportées (ce commit)

1. **Assets d'inférence réels livrés** — un checkpoint **V=104, 270 000 params** a été entraîné localement :
   500 steps sur `ci-data/tinystories_ci.txt` (32 kB) préfixé du vocabulaire complet,
   **val loss 2,977** (hasard = 4,644), ~19 k tok/s à l'entraînement.
   - `public/lm_checkpoint.bin` (1,03 MiB) + `public/lm_vocab.bin`
   - idem dans `docs/` → la démo publiée répare le lab sans rebuild
2. **`tools/prepare_inference_assets.mjs`** (`npm run assets:inference`) : copie et **vérifie le contrat**
   `V == |vocab|` et `taille == 4 + 4·params`, sinon message explicite (réentraîner) au lieu d'un plantage muet.
3. **`src/labs/InferenceLab.tsx` durci** : contrôle `r.ok` + `content-type` (un 404 déguisé en `index.html`
   n'est plus pris pour un checkpoint), V comparé au vocabulaire réel, message d'erreur actionnable, et
   compteurs de la section « honnêteté » mis à jour avec les chiffres réellement mesurés.
4. **`tools/sync_docs.mjs`** (`npm run docs:sync`) : `assets → vite build → docs/demo.html + assets`.
   `docs/demo.html` est régénéré, la page de vérification `docs/index.html` n'est pas touchée.
5. **`npm run smoke:labs`** : les 16 labs + le shell se rendent en headless (stubs `window`/`location`/`document`),
   sortie non nulle en cas de crash.
6. **`npm run verify:inference`** : vérifie assets, dimensions, génération et **effectivité de la quantification**
   (‖Δw‖/‖w‖ = 0,26 % INT8 · 4,31 % E8 · 4,71 % INT4, et INT4 change bien la sortie vs FP32).
7. **CI étendue** : les deux vérifications ci-dessus tournent désormais dans le job `test`.

---

## 5. Ce qu'on peut faire avec ce projet

### Tel quel (valeur immédiate)
- **Démo pédagogique cliquable** : 16 labs client-side, zéro serveur, `npm run dev` ou la page Pages.
  BPM, maillages MZI, Ising/CIM, holographie, HOM/CHSH : de quoi illustrer un cours d'optique/calcul photonique.
- **Cours payant** (5 modules, 149 €, module 1 gratuit) déjà branché sur `docs/` — c'est le dernier commit.
- **Modèle jouet exécutable dans le navigateur** : 270 k params, wasm 52 kB, 236–272 tok/s, offline.
  Idéal pour un atelier « un LLM tient dans 52 kB ».
- **Méthode réutilisable** : la boucle « 1 claim = 1 oracle indépendant + tolérance » est le vrai actif du dépôt.
  Elle se copie dans n'importe quel projet scientifique.

### Ce que je ferais ensuite (par ordre de rapport/valeur)

| Priorité | Chantier | Pourquoi |
|---|---|---|
| 1 | **CI → Pages** (workflow `docs:sync` + `pages`), au lieu de commits manuels de `docs/` | supprime la classe de bug §3.1/§3.2 |
| 2 | **Galerie d'exemples partageables** (issue #4) : l'état est déjà dans le hash d'URL — il ne manque qu'un bouton « copier le lien » + une page qui liste 10 réglages | fonctionnalité la plus visible pour un visiteur |
| 3 | **Quantification réelle** (stockage int8/int4 + dequant au chargement) | aujourd'hui seule la *qualité par bit* est démontrée, la taille mémoire ne baisse pas |
| 4 | **Export WASM des kernels GEMM** (issue #6) + kernel `v128` manuel sur W transposé | gain ×2–3 attendu sur le lab d'inférence, aujourd'hui ×1,25 |
| 5 | **Nettoyage** : `bench/`, `experiments/` datés, suppression des `test_debug*.c`, dédoublonnage `experiment/` vs `src/` | un visiteur qui arrive sur le dépôt voit 20 fichiers de debug avant le code |
| 6 | **Fraîcheur des claims** : dater + machine + `gcc --version` dans les tableaux du README ; script `npm run claims` qui re-mesure | c'est la promesse du projet, elle doit être rejouable |
| 7 | **Publication npm** de `@bahira/spear-physics-core` (déjà packagé) | distribue les modules physics zéro-dépendance |
| 8 | **Playwright** sur 3–4 labs (un test de pixels sur holographie/Ising) + `npm audit fix` (vite 7.3.7) | ferme le trou de couverture UI et les 2 alertes *high* |

### Limites à assumer (elles sont déjà écrites dans le dépôt, et c'est bien)
- Modèle **char-level 270 k**, corpus 32 kB → texte pseudo-cohérent, pas un LLM utile.
- Quantification **simulée** (poids fp32 en mémoire).
- Tout est mesuré sur x86 + gcc ; rien n'est validé sur ARM/Windows (le GEMM AVX2 se dégrade en scalaire).
- `rsqrt` est plus lent que libm : choix assumé (précision 2,7e-7).

---

## 6. Reproduire

```bash
npm install
npx tsc --noEmit && npm run test        # 44 tests
npm run smoke:labs                      # 16 labs montent
npm run verify:inference                # wasm + checkpoint + quantif
npm run build                           # 567 kB single-file
npm run docs:sync                       # rebuild de docs/demo.html
npm run dev                             # http://localhost:5173

# le C
gcc -O2 -march=native -ffast-math -fopenmp -o /tmp/lm_verify lm_c/verify_kernels.c -lm && /tmp/lm_verify
gcc -O2 -march=native -ffast-math -fopenmp -o /tmp/lm_train  lm_c/lm_main.c       -lm
/tmp/lm_train --data ci-data/tinystories_ci.txt --steps 60 --gradcheck --nogen
/tmp/lm_train --data DATA_V=104.txt --steps 500 --nogen && npm run assets:inference
```
