"""
Reproduces the numerical example from Section V of the paper:
  np_ = 3, nq = 2, B = 20.
Expected results (Fig. 1b): τ^p = [6,4,4], τ^q = [4,2], μ = 0.601.
"""

import numpy as np
from stackelberg_surveillance import (
    capture_probability,
    complete_graph_strategy,
    complete_bipartite_strategy,
    complete_graph_defense,
    complete_bipartite_defense,
    upper_bound,
)

np.set_printoptions(precision=4, suppress=True)


def section_separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# -----------------------------------------------------------------------
# Example 1 – Complete graph (n=4, uniform τ=3)
# -----------------------------------------------------------------------
section_separator("Complete Graph  (n=4, τ=[3,3,3,3])")

tau_cg = np.array([3, 3, 3, 3])
P_cg, pi_cg, mu_cg = complete_graph_strategy(tau_cg)
print(f"π_opt   = {pi_cg}")
print(f"μ_opt   = {mu_cg:.6f}")
mu_direct = capture_probability(P_cg, tau_cg)
print(f"μ (via F_k recursion) = {mu_direct:.6f}  [should match]")
ub = upper_bound(P_cg, tau_cg)
print(f"Upper bound Thm 1     = {ub:.6f}")


# -----------------------------------------------------------------------
# Example 2 – Complete graph (n=4, heterogeneous τ)
# -----------------------------------------------------------------------
section_separator("Complete Graph  (n=4, τ=[2,3,4,5])")

tau_cg2 = np.array([2, 3, 4, 5])
P_cg2, pi_cg2, mu_cg2 = complete_graph_strategy(tau_cg2)
print(f"π_opt   = {pi_cg2}")
print(f"μ_opt   = {mu_cg2:.6f}")
mu_direct2 = capture_probability(P_cg2, tau_cg2)
print(f"μ (via F_k recursion) = {mu_direct2:.6f}  [should match]")
ub2 = upper_bound(P_cg2, tau_cg2)
print(f"Upper bound Thm 1     = {ub2:.6f}")


# -----------------------------------------------------------------------
# Example 3 – Complete bipartite (paper's Fig. 1b)
# -----------------------------------------------------------------------
section_separator("Complete Bipartite (np=3, nq=2) – Paper Fig. 1b")

tau_p_paper = np.array([6, 4, 4])
tau_q_paper = np.array([4, 2])

P_bp, p_star, q_star, mu_bp = complete_bipartite_strategy(tau_p_paper, tau_q_paper)
print(f"p* (transition probs to P-nodes) = {p_star}")
print(f"q* (transition probs to Q-nodes) = {q_star}")
print(f"μ_opt   = {mu_bp:.6f}  (paper reports 0.601)")
print(f"\nTransition matrix P (rows: P-nodes then Q-nodes):\n{P_bp}")

mu_fk = capture_probability(P_bp, np.concatenate([tau_p_paper, tau_q_paper]))
print(f"\nμ (via F_k recursion) = {mu_fk:.6f}  [should ≈ 0.601]")
ub_bp = upper_bound(P_bp, np.concatenate([tau_p_paper, tau_q_paper]))
print(f"Upper bound Thm 1     = {ub_bp:.6f}")


# -----------------------------------------------------------------------
# Example 4 – Complete bipartite defense placement (paper's example)
# -----------------------------------------------------------------------
section_separator("Bipartite Defense Placement (np=3, nq=2, B=20)")

tau_p_opt, tau_q_opt = complete_bipartite_defense(np_=3, nq=2, B=20)
print(f"τ^p = {tau_p_opt}  (paper: [6, 4, 4])")
print(f"τ^q = {tau_q_opt}  (paper: [4, 2])")

_, _, _, mu_opt_placed = complete_bipartite_strategy(tau_p_opt, tau_q_opt)
print(f"μ with optimal placement  = {mu_opt_placed:.6f}")

# Compare with uniform allocation
tau_p_unif = np.array([4, 4, 4])
tau_q_unif = np.array([4, 4])
_, _, _, mu_unif = complete_bipartite_strategy(tau_p_unif, tau_q_unif)
print(f"μ with uniform allocation = {mu_unif:.6f}")
improvement_abs = (mu_opt_placed - mu_unif) * 100
improvement_rel = (mu_opt_placed - mu_unif) / mu_unif * 100
print(f"Absolute gain:  {improvement_abs:.2f} pp  (paper reports ~4.5 pp)")
print(f"Relative gain:  {improvement_rel:.1f}%")


# -----------------------------------------------------------------------
# Example 5 – Complete graph defense placement
# -----------------------------------------------------------------------
section_separator("Complete Graph Defense Placement (n=4, B=10)")

tau_cg_def = complete_graph_defense(n=4, B=10)
print(f"τ = {tau_cg_def}  (should sum to 10, differ by ≤1)")
print(f"Sum = {tau_cg_def.sum()}")

P_def, pi_def, mu_def = complete_graph_strategy(tau_cg_def)
print(f"μ with optimal placement = {mu_def:.6f}")
mu_fk_def = capture_probability(P_def, tau_cg_def)
print(f"μ (via F_k recursion)    = {mu_fk_def:.6f}  [should match]")
