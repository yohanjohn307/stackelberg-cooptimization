# Stochastic Surveillance Stackelberg Game

Code accompanying the paper:

> **A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**
> Yohan John, Gilberto Díaz-García, Xiaoming Duan, Jason R. Marden, Francesco Bullo
> *IEEE Transactions on Automatic Control*, vol. 70, no. 8, pp. 5468–5474, August 2025.
> DOI: [10.1109/TAC.2025.3549295](https://doi.org/10.1109/TAC.2025.3549295)

## Overview

A surveillance agent patrols a graph using a stochastic (Markov chain) strategy against a worst-case omniscient attacker. The attacker selects a node and begins an attack when the agent departs; a successful defense requires the agent to return within the node's **attack duration** τ_i. This work extends the Stackelberg game formulation of [Duan et al., 2021] to **heterogeneous attack durations**, enabling co-optimization of the patrol strategy and defense placement.

## Contents

| File | Description |
|------|-------------|
| `stackelberg_surveillance.py` | Core library — all algorithms from the paper |
| `demo.py` | Reproduces the numerical example from Section V (Fig. 1b) |

## Installation

The only dependencies are NumPy and SciPy.

```bash
pip install numpy scipy
```

## Usage

```python
import numpy as np
from stackelberg_surveillance import (
    capture_probability,
    complete_graph_strategy,
    complete_bipartite_strategy,
    complete_graph_defense,
    complete_bipartite_defense,
)

# --- Capture probability for an arbitrary transition matrix (Eq. 4) ---
P   = np.array([[0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]])
tau = np.array([3, 2, 4])
mu  = capture_probability(P, tau)

# --- Optimised patrol strategy for a complete graph (Sec. III-B) ---
tau = np.array([2, 3, 4, 5])
P_opt, pi_opt, mu_opt = complete_graph_strategy(tau)

# --- Optimised patrol strategy for a complete bipartite graph (Sec. III-C) ---
tau_p = np.array([6, 4, 4])   # attack durations for partition P
tau_q = np.array([4, 2])      # attack durations for partition Q
P_opt, p_star, q_star, mu_opt = complete_bipartite_strategy(tau_p, tau_q)

# --- Optimal defense placement for a complete graph (Theorem 6) ---
tau = complete_graph_defense(n=4, B=10)

# --- Optimal defense placement for a complete bipartite graph (Algorithm 1) ---
tau_p, tau_q = complete_bipartite_defense(np_=3, nq=2, B=20)
```

Run the demo to reproduce the paper's Section V example:

```bash
python demo.py
```

## API Reference

### `capture_probability(P, tau)`
Computes the capture probability μ for a given row-stochastic transition matrix `P` and attack-duration vector `tau` using the F_k hitting-time recursion (Eq. 4).

### `complete_graph_strategy(tau)`
Returns `(P_opt, pi_opt, mu_opt)` — the optimised patrol strategy for a complete graph with heterogeneous attack durations (Sec. III-B, Eqs. 12–13). Solves a single nonlinear equation via bisection, then sets P_opt = **1**π_opt^T.

### `complete_bipartite_strategy(tau_p, tau_q)`
Returns `(P_opt, p_star, q_star, mu_opt)` — the optimised patrol strategy for a complete bipartite graph (Sec. III-C, Eqs. 19–21). Nodes are ordered as [P-partition, Q-partition] in the transition matrix.

### `complete_graph_defense(n, B)`
Returns the optimal integer attack-duration vector for a complete graph with `n` nodes and total budget `B` (Theorem 6). Entries differ by at most 1.

### `complete_bipartite_defense(np_, nq, B)`
Returns `(tau_p, tau_q)` — the optimal even-integer defense allocation for a complete bipartite graph (Algorithm 1 + Theorem 7). Uses modified bisection on the sub-budget B_p.

### `upper_bound(P, tau)`
Computes the upper bound μ* ≤ min_i π_i τ_i from Theorem 1, given a transition matrix `P` whose stationary distribution π is used in the bound.

## Citation

```bibtex
@article{john2025stackelberg,
  author  = {John, Yohan and D{\'i}az-Garc{\'i}a, Gilberto and Duan, Xiaoming
             and Marden, Jason R. and Bullo, Francesco},
  title   = {A Stochastic Surveillance {S}tackelberg Game: {C}o-Optimizing
             Defense Placement and Patrol Strategy},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {70},
  number  = {8},
  pages   = {5468--5474},
  year    = {2025},
  doi     = {10.1109/TAC.2025.3549295},
}
```

## License

This work was supported in part by the Air Force Office of Scientific Research under Grant FA9550-21-1-0203 and in part by the National Natural Science Foundation of China under Grant 62303314.
