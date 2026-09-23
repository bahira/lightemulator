<div align="center">

# ⚡ GROUNDED-SPEAR

**Physics emulator + GPT from scratch in C — every number measured, every claim verified.**

[![CI](https://github.com/bahira/lightemulator/actions/workflows/ci.yml/badge.svg)](https://github.com/bahira/lightemulator/actions/workflows/ci.yml) [![tests](https://img.shields.io/badge/tests-38%2F38-brightgreen)]() [![parity](https://img.shields.io/badge/kernel%20parity-6%2F6-brightgreen)]() [![gradcheck](https://img.shields.io/badge/gradcheck-PASS-brightgreen)]() [![license](https://img.shields.io/badge/license-MIT-blue)]()

**[🔬 Live verification page →](https://bahira.github.io/lightemulator/)** — run the kernels in your browser, watch the receipts appear.
**[🎮 Full emulator demo →](https://bahira.github.io/lightemulator/demo.html)** — the 9 labs, running client-side, no install.

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
| Gerchberg–Saxton holography (2D FFT phase retrieval) | RMSE 0.217 → 0.069 in 40 it (×3.1) | η = 87% measured, Parseval defect 2.8e-15 |
| Quantum optics — HOM dip & CHSH (closed-form Fock algebra) | S_QM = 2√2 at 4.4e-16, local hidden variables ≤ 2 (200k pairs) | P(1,1)|₅₀/₅₀ = 0 exact, Σ P = 1 at 1e-16 |
| Full training loop | 270k params, ~9300 tok/s on an i7-7660U | val loss 3.79 (chance = 4.64) |

## What's inside

```
├── src/            the emulator — 11 labs (BPM, MZI, Ising/CIM, KAN, control,
│                   free energy, dispersion, SPEAR kernels, validation,
│                   holography, quantum optics)
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
npm run test     # grounded loop headless: 38 tests

# the GPT (needs gcc + OpenMP, TinyStories data in data/)
gcc -O2 -march=native -ffast-math -fopenmp -o lm_c/lm_train.exe lm_c/lm_main.c -lm
lm_c/lm_train.exe --steps 300 --gradcheck
```

## The GEMM story

The original micro-kernel used **one accumulator chain** per 8-column tile → FMA latency (~4 cy) unmasked, ≈ 0.25 FMA/cy against a peak of 2. The fix: **4 tiles of 8 columns sharing a broadcast**, contiguous loads, 4 independent FMA chains — ×1.78–2.19 on real shapes (qkv, attn, fc1, fc2, out). The backward pass got the same treatment (`mm_dxd`, `mm_tndw` with register accumulators over the full M dimension).

The honest parts: attention tiling caps at ×1.05 wall time (Amdahl — skipped on purpose), AVX-512 is unreachable on this hardware, and `rsqrt` currently trades speed for accuracy.

## The verification loop

38 tests, each with a number: Parseval, power conservation, MZI unitarity, CIM vs exact optimum, spline derivatives, gradient checks, Sellmeier dispersion, Fresnel biaxial, Fraunhofer vs the closed-form Dirichlet kernel, Gerchberg–Saxton convergence & energy conservation, HOM dip from Fock algebra, and a measured CHSH violation (quantum S = 2√2 vs a simulated local hidden-variable model capped at 2). Plus kernel parity 6/6 against libm, GEMM auto-tests (forward + backward, N%8 ≠ 0 included), and a gradient check against finite differences on every training run. See [`SPEAR_REPORT_2026-09-21.md`](SPEAR_REPORT_2026-09-21.md) for the full measured report and [`SHOWCASES.md`](SHOWCASES.md) for the complete quality showcase catalog.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **v0.1 — Fondations vérifiées** | 10 labs, kernels AVX2, GPT en C, page live, robot animé, drones, CI | ✅ shipped |
| **v0.2 — Interactivité & playground** | drag des balises, cerveau drone entraînable, éditeur de trajectoire robot, galerie d'exemples partageables | 🔜 |
| **v0.3 — Performance & portage** | taille du modèle GPT, export WASM des kernels, quantization int8 documentée, package npm des modules physics | 🔜 |
| **v1.0 — Release sérieuse** | docs site (cours), suite de benchmarks reproductible en CI, communauté, changelog | 🔜 |

Full issue tracker on the [milestones page](https://github.com/bahira/lightemulator/milestones).

## Contributing

PRs welcome — one rule: **every number is measured, every claim is verified**. Any contribution that adds a numeric claim must come with an independent reference (analytic solution, conservation law, exhaustive enumeration, finite differences). New physics modules must be pure TypeScript, zero dependencies, with a test in `src/physics/validate.ts`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow, and the [`good first issue`](https://github.com/bahira/lightemulator/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label for an entry point.

## License

[MIT](LICENSE) — free to use, modify, and sell. The physics modules are dependency-free single files: copy one, it works.

---

<div align="center">

**Built by [@bahira](https://github.com/bahira) — every claim falsifiable, every number reproducible.**

</div>
