# Rapport SPEAR — session 2026-09-21

## 1. Perimetre
Integration SpearVM (github.com/bahira/SpearVM), boucle grounded v9 (100 iterations/cible),
3 use cases reels, audit + ameliorations du module Energie libre (EP).
Tout est mesure (L3) : gcc -O2 + node + npm test + tsc.

## 2. Noyaux integres (SpearVM championnes)

| Noyau | Forme | L∞ (mesure) | Note |
|---|---|---|---|
| tanh_p34 | Pade[3/4] clamp ±4 | 1.562e-3 | ×29.6 vs spear_tanh ; ×5.5 vs spur_math.tanh |
| erf_v2 | Horner 5/5 clamp ±2 | 2.348e-5 | 100% ALU, 1 div |
| gelu_quintic | smoothstep, 5 mul 0 div | 1.742e-2 | fuse-friendly |
| gelu_erf | 0.5x(1+u·P/Q), u=x/√2 | 2.098e-5 | training/backprop |
| spearGeluBackward | derivee linaire-clamp v1 | gradcheck 2.5e-4 | UC3 |

100 iterations evolutives : aucun gain sur les 4 formes (minimums locaux deja converges).
Honnete : pas de gain force.

## 3. Use cases mesures (src/kernels/spear_usecases.c)

| Cas | Resultat |
|---|---|
| UC1 FFN fusionne gelu_erf (1024×768×3072) | 203.7 ms / 23.7 GFLOPS vs 1072.6 ms two-pass → ×5.26 wall ; err 2.24e-8 |
| UC2 porte MLP tanh_p34 | L∞ 1.562e-3 vs 4.622e-2 → ×29.6 |
| UC3 backprop gelu_erf | gradcheck : \|err\|max 2.53e-4, 0 cellule hors tol |
| UC4 Opto-SPEAR attention photonique MZI + FFN (S=64, D=64, dFfn=256) | 3.11 ms, temps de vol 160 ps ; gain énergie physique ×5.48 vs CMOS 7nm ; err 4.44e-8 |

Qualite d'entrainement (spear_train_quality.c, MLP 2→16→2, spirale, meme seed) :
exact 2.3289 < gelu_erf 2.3406 < gelu_quintic 2.3650.
Verdict : la precision du forward determine la qualite (gelu_erf recupere ~82% du gap).

## 4. FEP / attention (src/physics/fepAttn.ts, fep.ts)

| Feature | Mesure |
|---|---|
| A : gradient EP analytique des pompes (chain complete : −P·ΣRe, facteur ½×2, Jacobien softmax) | accord EP-vs-FD 1.07e-2 ; ×3.4 plus rapide (2 settles vs 8) ; loss 0.026 ≤ oracle 0.030 |
| B : AdamW sur couplages EP | 6.2× vs 11.9× SGD → non adopte, option `adam:true` |
| C : machine bande 2 couches (nHidden 8, bands 2) | reduction 27.5× (vs 11.9× plat), 0 couplage non-adjacent, identite valide 3.94e-3 |

Hypotheses refutees par mesure :
- le residu de relaxation a 800 pas est 8.19e-14 (identite valide au budget d'entrainement) ;
- l'ancien facteur 8.5 du gradient de pompe vient de la chaine incomplete (facteur symetrique + normalisation).

Claims app verifies : identite 1.39e-4, residu 2.0e-15, reduction 11.9×, attention ≤ oracle.

## 5. Validation globale
- npm run test : 25/25 PASS
- npx tsc --noEmit : OK
- npm run build : dist/index.html 454.78 kB (gzip 143.49 kB)
- C : spear_kernels.h compile + L∞ matches datasheet
- parity python vs pip spur-math : 4/4

## 6. Fichiers
| Fichier | Role |
|---|---|
| src/physics/spear.ts | noyaux + specs TS |
| src/kernels/spear_kernels.h | memes noyaux C99 |
| src/kernels/operator_benchmark.py | bench A/B (corrige) |
| src/kernels/spur_parity.py | parite vs pip spur-math |
| src/kernels/spear_usecases.c | UC1-UC4 |
| src/kernels/spear_train_quality.c | precision→qualite |
| tools/spear_grounded.mjs | boucle grounded 100 it. |
| tools/fep_audit.ts | audit FEP A/B/C |
| src/kernels/spear_champions_found.json | ledger recherche |

## 7. Suite immediate (2026-09-21, phase 2)

| Element | Mesure |
|---|---|
| Certificat de monotonie ΔE ≤ 0 (energyTrace stride 64 + flowResid ; VerifyResult.dEmax/stepsUsed) | ΔE max = 0.00e+0 certifie (D1), 0 NaN, seuil residu 1e-10 atteint en 1024 pas adaptatifs |
| verifyPhaseGradient adaptatif (blocs de 512, stop residu<1e-10 OU stabilite Δ<1e-12, plafond 20480) | ×20 de vitesse vs 20000 pas fixes (1024 pas effectifs) ; UI affiche ΔE max + pas |
| Residu d'equilibre final | 1.11e-14 (0.1072 → 1.0002) |
| Table bandes (nHidden=8, epochs=12) : L=1 → 19.2× ; L=2 → 27.5× ; L=4 → 38.2× | la profondeur bandee domine les iterations supplementaires |

## 8. Bench C (QPC, 8M elements, exp)

| Kernel | Temps | ns/el | x vs libm |
|---|---|---|---|
| libm expf | 38-86 ms | 4.8-10.7 | — |
| spear scalar (poly + ldexp champ entier) | 13-39 ms | 1.6-4.8 | x2.7-5.3 |
| spear AVX2 x8 (fma) | 4.0-8.1 ms | 0.50-1.02 | x10-25 |
| spear AVX512 x16 | **N/A — pas d'AVX-512 sur cette machine** | — | — |

CORRECTION (mesuree ce jour) : le CPU est un **Intel i7-7660U (Kaby Lake),
`__builtin_cpu_supports("avx512f") = 0`**. Toute instruction zmm leve #UD
(exit 0xC000001D = -1073741795). Les chiffres 512 anterieurs (0.13-0.28 ms,
"x2 vs avx2") sont **INVALIDES** — ce n'etait pas des intrinsics MinGW
casses, c'etait du code executable sur du hardware absent. `-march=native`
ne definit pas `__AVX512F__` ici ; garde runtime `__builtin_cpu_supports`
ajoute dans softmaxRow (lm_kernels.h) pour tout build force `-mavx512f`.

## 9. lm_train end-to-end (A/B libm vs SPEAR, 200-300 steps)

Build canonique : `gcc -O2 -march=native -ffast-math -fopenmp lm_main.c -lm`.

| Check | Resultat |
|---|---|
| verify_kernels (memes flags) | exp L∞ 1.49e-5 OK · rsqrt 2.00e-4 OK · gelu OK |
| gradcheck (40 params vs FD) | pire abs 0.0031 PASS |
| auto-test GEMM avx2/scal | max diff 1e-6 OK |
| loss | 4.644 (hasard) → 3.09 ema, val 3.07 |

**Profil du temps (`--profile`)** : GEMMs 89.5% · attention 8.8% · layernorm 1.7%.

**Debit A/B (6 runs, 200-300 steps)** : x0.73 a x1.23 selon l'ordre des phases
(1er = avantage turbo froid, 2e = throttled — biais confirme par LM_REV=1 :
ordre normal median x0.87, ordre inverse median x1.14). **Corrige d'ordre :
gain end-to-end ≈ x1.00**, coherent avec Amdahl : les kernels ne touchent que
~10% du temps (softmax+rsqrt), le GEMM AVX2 (89.5%) est identique des deux cotes.

Cote microbench (20M appels) : exp x1.83 · tanh x5.20 · gelu x4.75 — mais
**rsqrt x0.65** : sous -ffast-math, `1/sqrtf` devient `rsqrtss` materiel +
Newton du compilateur, plus rapide que spear_rsqrtf (bit-manip scalaire).
C'est le seul kernel SPEAR perdant, et il ne pese que 1.7% du wall-time.

## 10. GEMM micro-kernel (PR #1) + cartographie des goulots

### 10.0 Cartographie des goulots (profil mesuré)

| Rang | Goulot | Part | Étatement | Action |
|---|---|---|---|---|
| 1 | **GEMM forward** `mm_nt_t` | ~50% GEMM | 1 chaîne → **multi-acc 4 tuiles** | **FAIT ×1.78–2.19** |
| 2 | **GEMM backward** `mm_dxd`+`mm_tndw` | ~50% GEMM | 1 chaîne / load-store mémoire → **multi-acc 4 tuiles + acc registres sur M** | **FAIT** |
| 3 | attention (QKᵀ, PV, softmax) | 3.1% | scalaire + exp8 AVX2 | laisser (Amdahl) |
| 4 | layernorm `krsqrt` | 0.6% | spear_rsqrtf ×0.52 vs fast-math | non bloquant (1.7% wall) |
| 5 | CE + embeddings + adam | reste | — | hors scope |

**Profil post-optimisation** : GEMM ~92% · attention ~5% · ln ~1% (du sous-total
gemm+attn+ln). Absolu (200 steps, 1 phase, min 3) : **GEMM 3668 ms** · attn 425 ms · ln 104 ms.
Avant multi-acc fwd+bwd (même machine) : GEMM ~11 s/phase → **≈ ×3 sur le GEMM e2e**.

Champions MCP consultés : `gelu`, `layernorm_scale`, `fast_exp_alu` — aucun gain
vs les formes locales déjà en place. Discover (`1/sqrt(x)`, `exp(x)`) : rien de
mieux que le local (exp Padé[3/2] L∞ 1.5e-5 ; rsqrt bit-manip ou rsqrtss+Newton).

### 10.1 Sigmoid / siLU — rien à intégrer (local > upstream)

Upstream SpearVM (CHAMPIONS.md) : sigmoid Pade[3/4]-dérivé L∞ 7.8e-4, siLU 9.3e-4.
Local (spear.ts / spear_kernels.h, fondés sur le rt-core) : **sigmoid 1.6e-4, siLU 3.4e-4**
→ 5× / 3× meilleurs. Les formes upstream sont une **régression** : on garde le local.
Ajoutés à `spur_parity.py` (6 noyaux, tol 5e-4) pour verrouiller la parité.

### 10.2 mm_nt_t + backward multi-accumulateurs (PR #1)

Profil : GEMMs 89.5% du wall-time. Anciens micro-kernels : **une seule chaîne
d'accumulateur** par tuile 8-col → latence FMA (~4 cy) non masquée ≈ 0.25 FMA/cy
(pic AVX2 = 2/cy).

Nouveaux micro-kernels (4 tuiles de 8, broadcast partagé, charges contigues) :
- `mm_nt_t_avx2` (forward, W [K][N]) — queue N%8 scalar (fix OOB latent)
- `mm_dxd_avx2` (backward dX = dY·W) — 4 tuiles sur K
- `mm_tndw_avx2` (backward dW) — accs en registres sur tout M (fini load/store dw/m)

Auto-test étendu : `mm_nt` + `mm_nt_t` + `mm_dxd` + `mm_tndw` (GN=44).
`--gemm-old` force l'ancienne chaîne unique pour A/B e2e.

| Check | Résultat |
|---|---|
| auto-test 4 chemins avx2/scal | max\|diff\| = 1.9e-6 OK |
| gradcheck (40 params vs FD) | pire abs 0.0087 PASS |
| verify_kernels exp/rsqrt/gelu | 1.5e-5 / 2.7e-7 / 2.9e-3 OK |
| **A/B isolé old→new mm_nt_t** | **×1.78–2.19** (5 shapes) |
| **GEMM e2e absolu (200 steps)** | **3668 ms** (vs ~11 s avant fwd+bwd) |
| avx2 vs scal (lm_train, OMP) | ×3.3–12.6 |

Parité `spur_parity.py` : **6/6 PASS** (sigmoid 1.61e-4, silu 4.02e-4).

### 10.3 Next steps restants
1. rsqrt : ×0.52 vs fast-math — seulement 1.7% du wall, laisser (Amdahl).
2. attention 3–5% : tiling AVX2 de QKᵀ/PV — plafond wall ×1.05.
3. AVX-512 : inutilisable (Kaby Lake) — re-sauter sur machine avx512f=1.
4. **Taille du modèle (NLv/DD)** : à envisager **maintenant** que le GEMM est ×3.
5. Prolonger le ledger : re-runner spear_grounded.mjs sur nouveaux seeds.
