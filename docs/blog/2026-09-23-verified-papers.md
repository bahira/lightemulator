# I verified two research papers with a grounded loop — and caught 3 real bugs

*2026-09-23 · [lightemulator](https://github.com/bahira/lightemulator) · 41 tests, every claim falsifiable*

Most "paper emulation" repos don't verify anything. You get a notebook, a plot, and a claim you have to take on faith. We did the opposite: before publishing our emulation of two 2026 papers, we built the verification first — and it caught three real bugs, including a sign error that would have silently shipped.

## The setup

[lightemulator](https://github.com/bahira/lightemulator) distills physics into pure algebra: 14 zero-dependency TypeScript modules, each validated against an independent oracle (finite differences, exact closed forms, machine-precision identities). Two papers got the same treatment:

**Nature Com. 17, 1059 (2026)** — an inverse-designed photonic neural network. We emulated its two algebraic structures: the **N+C trick** (N basis fields reconstruct every sample by linearity — N+C simulations per epoch instead of L, verified to 2.12e-16) and the **adjoint-variable method** gradient (verified against finite differences: 1.76e-5). Training: 100% accuracy on held-out samples.

**arXiv:2506.15121 (Whitelam, LBNL)** — generative thermodynamic computing. We emulated the Langevin computer and its training rule: maximize the probability of the *reverse* trajectory (Onsager-Machlup action). The analytic gradient verified to 1.13e-6; the fluctuation relation's order-1 convergence confirmed (ratio 0.029); the training objective descends below the pure-noise entropy (47.85 → 46.85, N/2 = 48).

## The 3 bugs the loop caught

**1. The softmax that wasn't (PNN).** Our first gradient used raw powers instead of the softmaxed distribution. Relative error: 1.985. The finite-difference check caught it before anything shipped.

**2. The misplaced μΔt (Langevin).** The paper's Eq. 10 is terse: `(−Δx + μ∂iV·Δt)/(2kBT)`. Our first version multiplied the *entire* bracket by μΔt — including the −Δx. Gradient wrong by orders of magnitude until the factors were separated.

**3. The sign (Langevin).** Eq. 10 is *minus* ∂ln P̃/∂J. Our update went the wrong way — and the objective **exploded**: 47.85 → 81.71, measured. With the sign fixed: 47.85 → 46.85, below the noise entropy. The experimental proof of the wrong sign is itself a result: the paper's equations are correct but laconic, and the sign convention is the trap.

Plus one numerical finding: dt=0.05 violates Euler stability for ∂²V = 20+120x² (stable only |x| < 0.41) → NaN explosion.

## The most interesting scientific result

**The paper's noise→structure generation is scale-sensitive.** At 64+32 units (the paper uses 784+512), generation does not separate from chance (mean |Pearson| 0.17 vs 0.17). The analysis: at θ=0 the expected gradient is *exactly zero* — the bootstrap comes from the noise, then self-amplification shapes J at a rate that depends on the pattern. That signal is second-order (hidden covariances), drowned at small scale. We documented the limit in the module header instead of hiding it.

## The takeaway

The loop is the product. Every number reproducible, every claim falsifiable, every limit visible. The full write-up with live, in-browser verification: [bahira.github.io/lightemulator](https://bahira.github.io/lightemulator/) — the demo runs the actual test suite (41 tests, <2s) in your browser.
