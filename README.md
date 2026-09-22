<div align="center">

# ⚡ GROUNDED-SPEAR

**Physics emulator + GPT from scratch in C — every number measured, every claim verified.**

[![tests](https://img.shields.io/badge/tests-25%2F25-brightgreen)]() [![parity](https://img.shields.io/badge/kernel%20parity-6%2F6-brightgreen)]() [![gradcheck](https://img.shields.io/badge/gradcheck-PASS-brightgreen)]() [![license](https://img.shields.io/badge/license-MIT-blue)]()

**[🔬 Live verification page →](https://bahira.github.io/lightemulator/)** — run the kernels in your browser, watch the receipts appear.

</div>

---

## The pitch

A hand-tuned GPT written from scratch in pure C, with AVX2 kernels faster than libm — next to a browser emulator that simulates photonic computing (beam propagation, Mach-Zehnder meshes, coherent Ising machines), control theory, and optics. No BLAS. No PyTorch. No hand-waving: every number in this README is measured and reproducible.

## Receipts

| Kernel / system | Gain vs baseline | Accuracy |
|---|---|---|
| GEMM micro-kernel (`mm_nt_t`, 4-accumulator) | **×1.78–2.19** (5 real shapes, isolated bench) | max\|diff\| 1.4e-6 vs scalar |
| GEMM end-to-end (fwd + bwd multi-acc) | **~×3** on GEMM wall time | gradcheck PASS (0.4% worst) |
| `exp` — Padé-refined polynomial + ldexp | **×2.35** vs libm | L∞ 1.5e-5 on [−14, +2] |
| `tanh` — rational form | **×5.3** vs libm | L∞ 9e-3 |
| `gelu` — rational + analytic derivative | **×4.9** vs libm | L∞ 2.9e-3, derivative 10.8% (FD) |
| `sigmoid` / `silu` | > upstream reference forms | L∞ 1.6e-4 / 4.0e-4 |
| `rsqrt` — rsqrtss + Newton | ≈ libm (×0.52, honest) | **L∞ 2.7e-7** |
| Full training loop | 270k params, ~9300 tok/s on an i7-7660U | val loss 3.79 (chance = 4.64) |

## What's inside

```
├── src/            the emulator — 9 labs (BPM, MZI, Ising/CIM, KAN, control,
│                   free energy, dispersion, SPEAR kernels, validation)
├── lm_c/           the GPT — pure C, forward + exact backward + Adam,
│                   hand-written AVX2 GEMM micro-kernels, exp/rsqrt/gelu/tanh
├── tools/          evolutionary kernel search (500+ iters/target), grounded loop
└── docs/           the live verification page (GitHub Pages)
```

Everything in `src/physics/*.ts` is **pure TypeScript, zero dependencies** — copy the file, it works. Every numeric claim in the app is checked against an independent reference (analytic solution, conservation law, exhaustive enumeration, or finite differences) in the **Validation** tab.

## Quickstart

```bash
# the emulator
npm install
npm run dev      # 9 labs, all client-side
npm run test     # grounded loop headless: 25 tests

# the GPT (needs gcc + OpenMP, TinyStories data in data/)
gcc -O2 -march=native -ffast-math -fopenmp -o lm_c/lm_train.exe lm_c/lm_main.c -lm
lm_c/lm_train.exe --steps 300 --gradcheck
```

## The GEMM story

The original micro-kernel used **one accumulator chain** per 8-column tile → FMA latency (~4 cy) unmasked, ≈ 0.25 FMA/cy against a peak of 2. The fix: **4 tiles of 8 columns sharing a broadcast**, contiguous loads, 4 independent FMA chains — ×1.78–2.19 on real shapes (qkv, attn, fc1, fc2, out). The backward pass got the same treatment (`mm_dxd`, `mm_tndw` with register accumulators over the full M dimension).

The honest parts: attention tiling caps at ×1.05 wall time (Amdahl — skipped on purpose), AVX-512 is unreachable on this hardware, and `rsqrt` currently trades speed for accuracy.

## The verification loop

25 tests, each with a number: Parseval, power conservation, MZI unitarity, CIM vs exact optimum, spline derivatives, gradient checks, Sellmeier dispersion, Fresnel biaxial. Plus kernel parity 6/6 against libm, GEMM auto-tests (forward + backward, N%8 ≠ 0 included), and a gradient check against finite differences on every training run. See [`SPEAR_REPORT_2026-09-21.md`](SPEAR_REPORT_2026-09-21.md) for the full measured report.

---

<div align="center">

**Built by [@bahira](https://github.com/bahira) — every claim falsifiable, every number reproducible.**

</div>
