# SPEAR-T1 — un modèle **entier** qui tourne sur CPU

Ce dossier contient un modèle de langage d'un type nouveau : **l'inférence n'utilise
aucun flottant**. Pas « quantifié sur GPU avec des noyaux mixtes » : zéro instruction
flottante dans la boucle de génération, vérifié mécaniquement par désassemblage
(`--fpcheck`). Tout ce qui suit est mesuré sur cette machine et reproductible en une
commande ; la section « Vérifications » dit comment chaque chiffre a été obtenu.

```
cc -O3 -funroll-loops -march=native -ffast-math -fopenmp -o ti_main ti_main.c -lm
./ti_main --train 3000 --data ../data/shakespeare.txt --holdout 111539 \
          --save m.ckpt --export m.ti ; ./ti_main --sample "To be or not"
```

---

## 1. Ce qui rend le modèle différent

| Question | Réponse dans SPEAR-T1 |
|---|---|
| Où tourne-t-il ? | CPU nu (AVX2 + FMA), un cœur suffit, deux cœurs pour l'entraînement |
| Type des poids ? | **Ternaires** {−1, 0, +1} — 1 bit de signe + 1 bit de présence, sélectionnés par ligne |
| Type des activations ? | **int8 à échelle statique**, calibrées sur le corpus par le modèle lui-même |
| Accumulateurs ? | **int32 exacts**, bornes prouvées (aucun débordement possible) |
| Normalisation ? | RMSNorm entière : Newton sur 32 bits, **zéro `sqrt`** |
| Softmax ? | `exp2` entier (polynôme degré 5 + décalage) + réciproque normalisée, **zéro division** |
| Résidu ? | int8 **par couche** : une échelle de résidu par profondeur (le résidu va de ±5 à l'embedding à ±250 en couche 4) |
| Mélange train/inférence ? | **aucun** : le forward d'entraînement *est* le moteur entier |

Géométrie par défaut (`ti_config.h`, tout est surchargeable à la compilation) :

```
T=256  D=192  L=4  H=3  HD=64  F=384  V=256 (octets)  B=8
1 328 832 paramètres · 1 228 800 poids ternaires · 44,4 M MAC par jeton (préfixage)
```

## 2. Architecture

```
X0 = emb(id) + pos(t)
  ↓ (× L)
  X1 = X0 + Wo  · attn( rmsnorm₁(X0) )          attention causale, 3 têtes
  X2 = X1 + W2  · relu²( W1 · rmsnorm₂(X1) )    FFN à activation quadratique
  ↓
logits = Wout · rmsnorm_f(X_L)
```

`rmsnorm₂` normalise `X1` (**pas** `X2`), et le second résidu **accumule** — deux
détails qui, inversés, cassent silencieusement la rétropropagation ; ils sont
figés par le gradcheck.

## 3. Précision des noyaux (bornes mesurées, pas espérées)

| Noyau | Erreur mesurée | Méthode |
|---|---|---|
| `ti_exp2_q15` vs `2^x` | abs ≤ 1,15e‑4 · rel ≤ 1,49e‑3 | balayage exhaustif vs `exp2` de libm |
| `ti_rsqrt_q16` (Newton, 0 `sqrt`) | rel ≤ 1,02e‑3 | domaine int8 complet |
| `ti_rcp_norm` (Newton, 0 division) | rel ≤ 7,4e‑5 | 8,3 M d'arguments |
| `ti_dot_i8` AVX2 vs scalaire | **bit-exact** (0 écart) | K ∈ {1…511} |
| `ti_gemm_i8` AVX2 vs scalaire | **bit-exact** | 5×192×96 |
| `ti_rmsnorm_i8` vs réel | 0,73 pas int8 | RMS = 69,4 |
| `ti_softmax_i8_row` vs réel | 0,00 pas int8 | Σ poids = 127 exactement |
| `ti_pack2/unpack2` | exact | 2,00 bit/poids |

`--fpcheck` désassemble `ti_int_run`/`ti_int_step` et **échoue** si une seule
instruction flottante y apparaît : la revendication « aucune instruction
flottante » n'est pas déclarative, elle est vérifiée à l'exécution.

## 4. Fidélité : ce que coûte la quantification

`ti_probe` fait l'autopsie d'un checkpoint en trois mesures successives : les maîtres
flottants tels quels, les mêmes à **poids ternaires effectifs** (`s_w·w8`), puis le
**moteur entier**. La différence entre la première et la dernière est exactement le
prix de la quantification.

Deux modèles entraînés **1 000 pas** sur le même corpus (1,0 Mo), même géométrie,
même validation mise de côté (111 539 octets jamais vus à l'entraînement) :

| | fp32 pur | poids ternarisés | **moteur entier** | coût de la quantification |
|---|---|---|---|---|
| entraîné **en entier** (QAT) | 3,7576 | 3,7423 | **3,7494** | **+0,008** |
| entraîné en fp32 puis quantifié | 2,3924 | 2,6654 | **3,3483** | +0,956 |

Deux conclusions mesurées, dans deux directions opposées :

* **Entraîner dans le moteur rend la quantification gratuite** : 0,008 nat entre le
  modèle flottant et le moteur entier qui sert à l'inférence. C'est le point qui
  distingue ce dépôt d'un « post-training quantization » classique.
* **Mais cet entraînement converge moins bien** : la ternarisation étant un
  gradient biaisé (les mises à jour doivent franchir un seuil, et l'échelle
  `mean|W|` se recalcule avec les poids), le QAT atteint 3,75 là où le fp32
  atteint 2,39. Le meilleur modèle entier des deux est donc **celui qui vient du
  fp32** (3,35), malgré ses 0,96 nat de pénalité de quantification.

Pour comparaison, la baseline mini-GPT de 270 k paramètres de ce dépôt atteint
3,79 avec 20 % des paramètres et **sans contrainte d'inférence entière**.

Échantillon du modèle à 3,35 (génération entière, 220 jetons, le fichier `m.ti`
étant celui exporté ci-dessus) :

`./ti_main --sample "To be or not" 220 --export m.ti --temp 160 --seed 7`

```
To be or notinoryouechaspey,
I'd s!
Ann mave Wis benelllyou.
Wh Yone be be ALer, seathayadseond anstend be s SABbedigreisp,
Fow
ADor s I:
Aleoulaghtelindarnowot chansss s!
I t ste, ge fowe tove s mad. spot ps is the veise inialingo
```

Un modèle de 1,33 M paramètres ayant vu 2 M jetons produit cela — ce n'est pas
une démonstration de qualité linguistique, c'est la trace d'un moteur qui
fonctionne : la syntaxe y est (majuscules, apostrophes, fins de vers), le
vocabulaire non.

La **température** est en Q8 : `--temp 256` vaut T = 1,00, `--temp 160` vaut
T = 0,62 (d'où l'argument ci-dessus). Elle agit réellement sur le tirage — la
distribution du moteur a été comparée à la softmax exacte en float64
(`p(top)` = 0,0552 des deux côtés) et `./ti_test --sampling` vérifie l'accord sur
60 000 tirages à deux températures. Avant correction, la température était
appliquée aux logits entiers *bruts*, sans leur échelle physique : tous les
tirages se ressemblaient et la génération était dégénérée.

La fidélité est aussi vérifiée site par site (`--fidelity`) : le moteur entier est
comparé au graphe flottant *qui lui correspond exactement* (mêmes poids effectifs,
mêmes échelles), donc l'écart mesuré est du coût de quantification pur.

```
perte fp32 (poids ternaires) 5,5833  |  perte moteur entier 5,5812   écart 0,038 %
résidu écrêté : 0,000 %
```

## 5. Débits mesurés (i7 2 cœurs, AVX2, `--bench`)

| | |
|---|---|
| Préfixage entier (B=8, T=256) | **10 835 jetons/s** (1,43 M MAC/jeton) |
| Décodage autorégressif (1 jeton) | **11 576 jetons/s** (1,38 M MAC/jeton) |
| Pas d'entraînement complet | 3 671 jetons/s (forward 207 ms, rétroprop 348 ms) |
| Entraînement fp32 (référence) | ≈2 200 jetons/s (mesuré sur les runs fp32 de 1 000 pas) |

(Préfixage et décodage ont été mesurés à 11 380 et 12 007 jetons/s dans la mesure
immédiatement suivante : la machine n'a que 2 cœurs et ces chiffres bougent de
±20 % avec la charge. Les MAC/jeton, eux, sont exacts — c'est un compteur, pas un
chronomètre.)

Le gain de ×3,8 sur la version -O2 vient de `-O3 -funroll-loops` : les boucles
d'accumulation des noyaux de rétropropagation ne sont **pas** vectorisées à -O2
(1 seule boucle vectorisée sur 5, 4,6 GMAC/s contre 20 GMAC/s à -O3).

## 6. Vérifications (comment chaque chiffre est obtenu)

| Commande | Ce qu'elle prouve |
|---|---|
| `./ti_test --kernels` | bornes du tableau §3, comparaison à libm |
| `./ti_test --engine` | déterminisme (hash), **préfixage = décodage au bit près** sur 33/33 positions, statistiques ternaires |
| `./ti_test --sampling` | la distribution tirée suit la softmax exacte (écart max 3e‑4 à T = 0,5 et 2e‑3 à T = 1,0) sur 60 000 tirages |
| `./ti_main --gradcheck` | rétropropagation : 10 directions conjuguées + 16 sondes individuelles vs différences finies centrées — **TOUT PASSE** en petite *et* en pleine géométrie |
| `./ti_main --fidelity` | §4, site par site |
| `./ti_main --fpcheck` | aucune instruction flottante dans le moteur |
| `./ti_main --pack` | format de fichier modèle, 2 bits/poids |
| `./ti_dump` + `tools/ti_reference.py` | référence numpy float64 **indépendante** (aucun code C partagé) : perte à 2e‑09, gradients à 2,5e‑07, différences finies float64 concordantes |

La référence numpy se rejoue en deux commandes, sur une petite géométrie
(`-DTI_T=8 -DTI_D=8 -DTI_L=2 -DTI_H=2 -DTI_F=8 -DTI_V=8 -DTI_B=2`) : `ti_dump`
écrit les entrées, les sorties et tous les gradients tenseur par tenseur,
`ti_reference.py` recalcule les mêmes quantités en float64 à partir des mêmes
entrées et compare. Il conclut « TOUT PASSE » ou « ÉCHEC » et sort en code non
nul si un seuil est franchi — la capacité du test à échouer a été vérifiée en
faussant volontairement un gradient de 0,1 %. C'est aussi ce rejeu qui a remplacé
`--attncheck` (mode supprimé) : il vérifie les poids d'attention avec le reste du
graphe, et non par un seuil de convergence arbitraire.

Le gradcheck tourne aussi en **petite géométrie** (`-DTI_T=32 -DTI_D=64 -DTI_L=2
-DTI_H=2 -DTI_F=128`) : les deux échelles passent, ce qui écarte l'hypothèse d'un
accord qui ne tiendrait qu'à une taille particulière.

## 7. Fichiers

| Fichier | Rôle |
|---|---|
| `ti_config.h` | géométrie, surchargeable en ligne de commande |
| `ti_kernels.h` | noyaux entiers (dot/gemm AVX2, exp2, rsqrt, rcp, rmsnorm, softmax) |
| `ti_infer.h` | moteur d'inférence : **aucun flottant**, cache KV, préfixage/décodage |
| `ti_model.h` | maîtres fp32, ternarisation, calibration, rétroprop, Adam, export/checkpoint |
| `ti_main.c` | harnais : `--train --bench --gradcheck --fidelity --ptq --sample --export --pack --fpcheck` |
| `ti_test.c` | bornes des noyaux et invariants du moteur |
| `ti_probe.c` | autopsie d'un checkpoint : fp32 / ternaire / entier, pour savoir **quel** étage perd l'information |
| `ti_dump.c` | vidage d'une petite géométrie (entrées, sorties, tous les gradients) pour la référence numpy `tools/ti_reference.py` |

Statuts de fichiers : `MIT1` = modèle entier exporté (embeddings int8, **poids
ternaires compactés à 2 bits**, requantifications, échelles de résidu par couche,
échelle de sortie `swq[V]` et `logit_scale`) — 469 948 octets pour 3,06 bits/poids,
requants inclus ; `TIF1` = checkpoint d'entraînement (maîtres fp32 + moments Adam
+ échelles). Le format est versionné (`hdr[9]`) : v1 et v0 sont **refusés** au
chargement plutôt que lus de travers, v2 est lu avec un avertissement (il n'a pas
d'échelle de sortie, donc l'échantillonnage y est à une température
approximative), v3 est la version courante.

La température d'échantillonnage dépend de `swq`/`logit_scale` : c'est ce qui a
fait passer le format de v2 à v3. Un fichier v3 décrit donc à la fois les poids
et l'échelle physique des logits, et `--temp` y est une température au sens
habituel (unité Q8).

## 8. Défauts connus (non cachés)

* Le moteur ne fait pas de `top-k`/`top-p` : seul le tirage à température est
  implémenté. L'accord avec la softmax exacte est mesuré sur une distribution
  synthétique à 8 classes (`lg[v] = −4v`, le cas que le test sait calculer en
  float64) : écart maximal 3e‑4 à T = 0,5 et 2e‑3 à T = 1,0. Rien ne prouve que
  l'accord tienne aussi bien sur la queue d'une distribution à 256 classes.
* L'entraînement QAT converge plus lentement que le fp32 ; le pas d'apprentissage
  doit décroître vite (les runs à pas constant plafonnent ~0,9 nat au-dessus).
  Ce n'est pas corrigé, c'est mesuré.
* Le format de fichier compacte les poids à 2 bits mais redéploie 1 octet/poids en
  RAM ; un moteur qui décoderait directement les 2 bits gagnerait 2,4 Mo de RAM
  (non fait, aucune mesure à l'appui).
