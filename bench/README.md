# bench/ — micro-benchmarks, harnais et traces

Tout ce qui **mesure** sans faire partie du produit. Rien ici n'est compilé par la CI
(qui utilise `lm_c/verify_kernels.c` et `lm_c/lm_bench_exp8.c`), et rien ne doit être
importé par `src/`. Historique de la phase d'exploration SPEAR (BT23 → BT31).

## Kernels & harnais C

| Fichier | Rôle |
|---|---|
| `spear_bench.c` | micro-bench GEMM/activations vs libm |
| `test_spear_microkernels_harness.c` | harnais d'exactitude des micro-kernels |
| `test_spear_phase5_harness.c` | harnais phase 5 (intégration) |
| `test_debug.c`, `test_debug2.c`, `test_debug3.c`, `test_debug4.c`, `debug_bn.c`, `debug_bn3.c` | sessions de mise au point (batch-norm, accumulateurs) — **conservés comme traces**, pas comme tests |
| `test_bt29.c`, `debug_bt29.c`, `test_exp.c`, `test_exp2.c`, `test_clamp.c`, `test_minimal.c`, `test_standard.c` | reproductions minimales d'un bug, puis correctif |

## Python

| Fichier | Rôle |
|---|---|
| `benchmark_token_speed.py` | tok/s du GPT vs références |
| `quick_bench.py`, `bt29_bench2.py`, `bt29_quick_bench.py`, `bt29_test.py` | benchs de la percée BT29 |
| `test_spear_api.py` | test de l'API de `experiments/spear_api_server.py` |
| `test_llm_attention_integration.py` | parité attention SPEAR vs référence |
| `benchmark_output.txt` | sortie brute d'un run (traçabilité) |

## Règle

Un fichier entre ici quand il **documente une mesure** ou **une impasse**. Il sort d'ici
(ou disparaît) quand il devient un test de CI — la CI ne doit dépendre que de `src/`,
`lm_c/`, `tools/` et `ci-data/`.
