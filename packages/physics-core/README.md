# @bahira/spear-physics-core

Physics primitives distilled to pure algebra — verified modules, zero dependencies.

Part of [lightemulator](https://github.com/bahira/lightemulator) (12 physics labs, 35 grounded tests, every claim falsifiable).

## Modules

`spear` (LLM activations as pure algebra) · `kan` · `ising` · `fep` / `fepAttn` (free-energy principle) · `fft` · `mzi` · `bpm` (beam propagation) · `optics` (Sellmeier, GDD/TOD, Fresnel biaxial) · `control` (IK, jerk-bounded trajectories, inverted pendulum) · `latency` · `drones` (phototaxis, trilateration, Friis budget) · `pnn` (photonic neural network emulation, Nat. Com. 2026) · `langevin` (generative thermodynamic computing, arXiv:2506.15121)

## Usage

```js
import { spear, langevin, pnn } from '@bahira/spear-physics-core';

const err = langevin.langevinGradError(); // 1.1e-6 — analytic inverse-trajectory gradient
```

Namespaced exports (`export * as module`) — no name collisions across the 14 modules.

Zero dependencies. ESM only. MIT.
