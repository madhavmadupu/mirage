# ADR 0002: Stratified Latin Hypercube Sampling (LHS) for Domain Randomization

- **Status**: Accepted
- **Date**: 2026-10-31
- **Author**: Madhav Madupu (`madhav.madupu@gmail.com`)

---

## Context and Problem Statement

On a single-GPU workstation (NVIDIA RTX 4060 8 GB), synthetic dataset generation budgets typically range from $N = 250$ to $N = 5,000$ distinct physical scenes across a $D$-dimensional domain randomization space ($D \ge 9$: camera radius, elevation, azimuth, focal length, HDRI exposure, HDRI azimuth, PBR metallic, PBR roughness, and distractor clutter count).

When sampling $N$ configurations using naive independent and identically distributed (i.i.d.) uniform random numbers ($x_{i, d} \sim \mathcal{U}(a_d, b_d)$):
- Sample points exhibit high discrepancy and clustering in $D$ dimensions.
- One-dimensional marginal projections (e.g., extreme camera elevations or high specular metallic shaders combined with low roughness) contain large unvisited gaps at moderate $N$, causing trained detectors in [R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor) to fail on edge-case lighting regimes.

## Decision

We implement **Deterministic Stratified Latin Hypercube Sampling (LHS)** in `src/mirage/sampler.py`, seeded by a 64-bit `PCG64` master seed (`DomainRandomizationSpec.master_seed`).

For $N$ scenes and $D$ continuous randomization dimensions:
1. Each unit interval $[0, 1]$ is partitioned into $N$ equal non-overlapping strata $\left[\frac{i-1}{N}, \frac{i}{N}\right]$.
2. Each dimension $d \in \{1, \dots, D\}$ undergoes an independent random permutation $\pi_d \in S_N$ with intra-stratum uniform jitter $u_{i,d} \sim \mathcal{U}(0, 1)$:
   $$\hat{x}_{i, d} = \frac{\pi_d(i) - 1 + u_{i, d}}{N}$$
3. The normalized coordinate $\hat{x}_{i, d}$ is mapped via the target parameter's inverse CDF $F_d^{-1}(\hat{x}_{i, d})$ to its physical range $[a_d, b_d]$.

## Consequences

### Positive
- **Guaranteed Marginal Stratification**: Every 1D parameter projection is guaranteed to have exactly one sample in each of the $N$ intervals $\left[\frac{i-1}{N}, \frac{i}{N}\right]$, ensuring uniform coverage of rare specular, grazing-angle, and low-exposure regimes even when $N = 100$.
- **Variance Reduction**: Reduces the variance of downstream loss estimates over the domain randomization space by $O(N^{-1})$ on additive components relative to i.i.d. uniform sampling.
- **Bit-Exact Reproducibility**: Pre-computing the $N \times D$ LHS matrix from `master_seed` allows any single scene $i$ to be regenerated in isolation for debugging without running preceding scenes $0 \dots i-1$.

### Negative & Mitigations
- **Fixed Horizon $N$**: Classic LHS requires specifying the number of scenes $N$ upfront to define the $N$ strata. *Mitigation*: To expand an existing dataset of size $N$, users can generate an independent LHS block of size $N$ with an incremented seed offset (`master_seed + 1`), preserving full stratification within each block.
