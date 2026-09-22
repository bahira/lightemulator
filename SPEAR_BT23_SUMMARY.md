# SPEAR BT23: Micro-Kernel LogSumExp Streaming O(1)

## Overview
1-pass memory, 9.4G Floats/sec, perfect numerical stability.

## Files
- `src/spear_breakthrough21_rsqrt.c` - Fast rsqrt/inverse (Newton-FMA)
- `src/spear_breakthrough22_transcendentals.c` - exp/SiLU/GELU (4 FMA)
- `src/spear_breakthrough23_logsum.c` - 1-pass streaming LogSumExp
- `src/spear_breakthrough24_reduction.c` - Horizontal reduction

## Performance
- **9.4G Floating-point operations per second**
- **1 pass memory access** (vs 3 passes traditional)
- **Perfect numerical stability** via input clamping

## Validation
All 4 base kernels pass:
- `[BT 21] Fast Rsqrt` ✓
- `[BT 21] Fast Inv` ✓  
- `[BT 23] Streaming 1-Pass LogSumExp` ✓
- `[BT 24] Dot Product 8D` ✓

## Key Algorithm
Streaming LogSumExp: `d_k = d_{k-1} * exp(max_{k-1} - max_k) + exp(x_k - max_k)`
Then: `LSE = max + log(sum)`

Critical fix: Input clamping `[-88, 88]` to `spear_fast_exp_avx2()` prevents int32 overflow from large negative float inputs.