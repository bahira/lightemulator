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

Échantillon du modèle à 3,35 (génération entière, T=0,625, 220 jetons) :

```
To be or nothand,
I beandeand theand theand, theand,
Wheand theand beand theand thand thereand the thererd there therereand theand the......
```

Un modèle de 1,33 M paramètres ayant vu 2 M jetons produit cela — ce n'est pas
une démonstration de qualité linguistique, c'est la trace d'un moteur qui
fonctionne (le modèle QAT, entraîné par gradient biaisé, produit des « e »
répétés : la différence de perte, 0,4 nat, s'entend).

La fidélité est aussi vérifiée site par site (`--fidelity`) : le moteur entier est
comparé au graphe flottant *qui lui correspond exactement* (mêmes poids effectifs,
mêmes échelles), donc l'écart mesuré est du coût de quantification pur.

```
perte fp32 (poids ternaires) 5,5833  |  perte moteur entier 5,5810   écart 0,04 %
résidu écrêté : 0,000 %
```

## 5. Débits mesurés (i7 2 cœurs, AVX2, `--bench`)

| | |
|---|---|
| Préfixage entier (B=8, T=256) | **10 725 jetons/s** (1,43 M MAC/jeton) |
| Décodage autorégressif (1 jeton) | **9 449 jetons/s** (1,38 M MAC/jeton) |
| Pas d'entraînement complet | 3 476 jetons/s (forward 205 ms, rétroprop 380 ms) |
| Entraînement fp32 (référence) | 2 207 jetons/s |

(Préfixage et décodage ont été mesurés à 13 310 et 12 361 jetons/s à d'autres
moments : la machine n'a que 2 cœurs et ces chiffres bougent de ±20 % avec la
charge. Les MAC/jeton, eux, sont exacts — c'est un compteur, pas un chronomètre.)

Le gain de ×3,8 sur la version -O2 vient de `-O3 -funroll-loops` : les boucles
d'accumulation des noyaux de rétropropagation ne sont **pas** vectorisées à -O2
(1 seule boucle vectorisée sur 5, 4,6 GMAC/s contre 20 GMAC/s à -O3).

## 6. Vérifications (comment chaque chiffre est obtenu)

| Commande | Ce qu'elle prouve |
|---|---|
| `./ti_test --kernels` | bornes du tableau §3, comparaison à libm |
| `./ti_test --engine` | déterminisme (hash), **préfixage = décodage au bit près** sur 33/33 positions, statistiques ternaires, échantillonnage |
| `./ti_main --gradcheck` | rétropropagation : 10 directions conjuguées + 16 sondes individuelles vs différences finies centrées — **TOUT PASSE** en petite *et* en pleine géométrie |
| `./ti_main --fidelity` | §4, site par site |
| `./ti_main --fpcheck` | aucune instruction flottante dans le moteur |
| `./ti_main --pack` | format de fichier modèle, 2 bits/poids |
| `tools/ti_reference.py` | référence numpy float64 **indépendante** (aucun code C partagé) : perte à 2e‑10, gradients à 3e‑7 |

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

Statuts de fichiers : `MIT1` = modèle entier exporté (embeddings int8, **poids
ternaires compactés à 2 bits**, requantifications, échelles de résidu par couche)
— 468 920 octets pour 3,05 bits/poids, requants inclus ; `TIF1` = checkpoint
d'entraînement (maîtres fp32 + moments Adam + échelles). Le format est versionné :
un fichier au format antérieur est **refusé** au chargement plutôt que lu de
travers.

## 8. Défauts connus (non cachés)

* `ti_lin_dw` alloue/libère le transposé à chaque appel (perf, pas correction).
* Mode `--attncheck` : seuil de convergence obsolète, remplacé par
  `tools/ti_reference.py` ; ne pas s'en servir comme critère.
* L'unité de `sc[]` dans `ti_int_sample` (réciproque normalisée) n'a pas été
  revérifiée depuis la réécriture de la softmax — à ce stade elle ne change que
  la température effective de l'échantillonnage.
* L'entraînement QAT converge plus lentement que le fp32 ; le pas d'apprentissage
  doit décroître vite (les runs à pas constant plafonnent ~0,9 nat au-dessus).
  Ce n'est pas corrigé, c'est mesuré.
* Le format de fichier compacte les poids à 2 bits mais redéploie 1 octet/poids en
  RAM ; un moteur qui décoderait directement les 2 bits gagnerait 2,4 Mo de RAM
  (non fait, aucune mesure à l'appui).
