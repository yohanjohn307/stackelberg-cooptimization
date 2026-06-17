"""
Stochastic Surveillance Stackelberg Game
=========================================
Implements the algorithms from:

  Y. John, G. Díaz-García, X. Duan, J. R. Marden, F. Bullo,
  "A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense
  Placement and Patrol Strategy," IEEE TAC, vol. 70, no. 8, 2025.

Public API
----------
capture_probability(P, tau)
    Eq. (4): compute μ for any transition matrix and attack-duration vector.

complete_graph_strategy(tau)
    Sec. III-B: solve for π_opt via bisection (Eq. 12–13), return P_opt = 1π^T.

complete_bipartite_strategy(tau_p, tau_q)
    Sec. III-C: solve for p*, q* via bisection (Eq. 20), return block P (Eq. 15).

complete_graph_defense(n, B)
    Theorem 6: optimal even-spread integer τ for a complete graph.

complete_bipartite_defense(np_, nq, B)
    Algorithm 1 + Theorem 7: optimal even τ^p, τ^q for a complete bipartite graph.
"""

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fk_matrices(P: np.ndarray, tau_max: int) -> list[np.ndarray]:
    """Return [F_1, ..., F_{tau_max}] using Eq. (1): F_{k+1} = P(F_k - diag(F_k))."""
    n = P.shape[0]
    Fk = [None] * (tau_max + 1)   # 1-indexed; Fk[0] unused
    Fk[1] = P.copy()
    for k in range(1, tau_max):
        diag_k = np.diag(np.diag(Fk[k]))
        Fk[k + 1] = P @ (Fk[k] - diag_k)
    return Fk


def _check_stochastic(P: np.ndarray) -> None:
    n = P.shape[0]
    if P.shape != (n, n):
        raise ValueError("P must be square")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P must be row-stochastic (rows must sum to 1)")
    if np.any(P < -1e-10):
        raise ValueError("P must be non-negative")


# ---------------------------------------------------------------------------
# (i) Capture probability for arbitrary P and τ
# ---------------------------------------------------------------------------

def capture_probability(P: np.ndarray, tau: np.ndarray) -> float:
    """
    Compute the capture probability μ for transition matrix P and attack
    duration vector τ (Eq. 4).

    μ = min_{i,j}  Σ_{k=1}^{τ_j}  F_k(i, j)

    Parameters
    ----------
    P   : (n, n) row-stochastic transition matrix.
    tau : (n,) integer array of attack durations, one per node.

    Returns
    -------
    μ ∈ [0, 1]
    """
    P = np.asarray(P, dtype=float)
    tau = np.asarray(tau, dtype=int)
    _check_stochastic(P)
    n = P.shape[0]
    if tau.shape != (n,):
        raise ValueError("tau must have length n")

    tau_max = int(tau.max())
    Fk = _fk_matrices(P, tau_max)

    # Build the column-summed matrix M where M[:, j] = Σ_{k=1}^{τ_j} F_k[:, j]
    M = np.zeros((n, n))
    for j in range(n):
        for k in range(1, tau[j] + 1):
            M[:, j] += Fk[k][:, j]

    return float(np.min(M))


# ---------------------------------------------------------------------------
# Bisection subroutine shared by Sections III-B and III-C
# ---------------------------------------------------------------------------

def _solve_capture_eq(tau_eff: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Solve  Σ_i [1 - (-v)^{1/τ_i}] = 1  for v ∈ (-1, 0)   (Eq. 12 / Eq. 20).

    `tau_eff` is the vector of *effective* attack durations (τ_i for complete
    graphs, ⌊τ_i/2⌋ for each partition of a complete bipartite graph).

    Returns
    -------
    v_opt   : optimal objective value (< 0)
    pi_opt  : optimal transition weights, π_i = 1 - (-v_opt)^{1/τ_i}
    """
    tau_eff = np.asarray(tau_eff, dtype=float)
    if np.any(tau_eff < 1):
        raise ValueError("All effective attack durations must be >= 1")

    def g(v_neg):
        # v_neg = -v_opt > 0; g should equal 1 at the solution
        return np.sum(1.0 - v_neg ** (1.0 / tau_eff)) - 1.0

    # g is strictly increasing in v_neg; bracket it in (0, 1)
    v_opt_neg = brentq(g, 1e-12, 1.0 - 1e-12)
    pi_opt = 1.0 - v_opt_neg ** (1.0 / tau_eff)
    return -v_opt_neg, pi_opt


# ---------------------------------------------------------------------------
# (iii-a) Complete graph strategy  (Section III-B)
# ---------------------------------------------------------------------------

def complete_graph_strategy(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the optimised patrol strategy P_opt = 1 π_opt^T for a complete
    graph with heterogeneous attack durations τ  (Sec. III-B, Eqs. 12–13).

    Parameters
    ----------
    tau : (n,) integer array, τ_i ≥ 1 for all i.

    Returns
    -------
    P_opt   : (n, n) row-stochastic transition matrix.
    pi_opt  : (n,) stationary distribution.
    mu_opt  : capture probability min_i [1 - (1-π_i)^{τ_i}].
    """
    tau = np.asarray(tau, dtype=int)
    n = len(tau)
    _v_opt, pi_opt = _solve_capture_eq(tau.astype(float))
    P_opt = np.ones((n, 1)) @ pi_opt.reshape(1, n)
    mu_opt = float(np.min(1.0 - (1.0 - pi_opt) ** tau))
    return P_opt, pi_opt, mu_opt


# ---------------------------------------------------------------------------
# (iii-b) Complete bipartite graph strategy  (Section III-C)
# ---------------------------------------------------------------------------

def complete_bipartite_strategy(
    tau_p: np.ndarray, tau_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute the optimised patrol strategy (Eq. 15) for a complete bipartite
    graph with heterogeneous attack durations  (Sec. III-C, Eqs. 19–21).

    The graph has partition P of size n_p and partition Q of size n_q.

    Parameters
    ----------
    tau_p : (n_p,) integer array, τ^P_i ≥ 2 for all i.
    tau_q : (n_q,) integer array, τ^Q_j ≥ 2 for all j.

    Returns
    -------
    P_opt  : (n_p+n_q, n_p+n_q) row-stochastic block transition matrix.
             Rows/columns ordered as [P-nodes, Q-nodes].
    p_star : (n_p,) optimal transition probabilities from Q-nodes to P-nodes.
    q_star : (n_q,) optimal transition probabilities from P-nodes to Q-nodes.
    mu_opt : capture probability (Eq. 16).
    """
    tau_p = np.asarray(tau_p, dtype=int)
    tau_q = np.asarray(tau_q, dtype=int)
    np_ = len(tau_p)
    nq = len(tau_q)

    # Effective durations are ⌊τ/2⌋ (Lemma 3)
    tau_p_eff = np.floor(tau_p / 2.0).astype(float)
    tau_q_eff = np.floor(tau_q / 2.0).astype(float)

    if np.any(tau_p_eff < 1) or np.any(tau_q_eff < 1):
        raise ValueError("All attack durations must be >= 2 for bipartite graphs")

    _vp, p_star = _solve_capture_eq(tau_p_eff)
    _vq, q_star = _solve_capture_eq(tau_q_eff)

    # Build block transition matrix (Eq. 15)
    n = np_ + nq
    P_opt = np.zeros((n, n))
    # P-nodes (rows 0..np_-1) transition to Q-nodes (cols np_..n-1)
    P_opt[:np_, np_:] = np.ones((np_, 1)) @ q_star.reshape(1, nq)
    # Q-nodes (rows np_..n-1) transition to P-nodes (cols 0..np_-1)
    P_opt[np_:, :np_] = np.ones((nq, 1)) @ p_star.reshape(1, np_)

    mu_p = np.min(1.0 - (1.0 - p_star) ** tau_p_eff)
    mu_q = np.min(1.0 - (1.0 - q_star) ** tau_q_eff)
    mu_opt = float(min(mu_p, mu_q))
    return P_opt, p_star, q_star, mu_opt


# ---------------------------------------------------------------------------
# Defense placement helpers
# ---------------------------------------------------------------------------

def _even_spread(n: int, B: int) -> np.ndarray:
    """
    Allocate integer budget B among n items using the near-even rule
    (Theorem 6 for complete graphs).  Returns τ with Σ τ_i = B.
    """
    r = B % n
    tau = np.full(n, B // n, dtype=int)
    tau[:r] += 1
    return tau


def _even_spread_bipartite(n: int, B: int) -> np.ndarray:
    """
    Allocate even-integer budget B among n items, each τ_i must be even
    (Theorem 7 for complete bipartite graphs, restricted to even values).

    Uses the smallest two consecutive even values whose weighted sum equals B.
    """
    # Find the two even neighbours of B/n
    frac = B / n
    lo = int(np.floor(frac))
    lo = lo if lo % 2 == 0 else lo - 1     # round down to even
    hi = lo + 2

    if lo == hi or hi == lo:                # perfectly divisible by even
        return np.full(n, lo, dtype=int)

    # s * hi + (n - s) * lo = B  →  s = (B - n*lo) / 2
    s = (B - n * lo) // 2
    tau = np.full(n, lo, dtype=int)
    tau[:s] = hi
    return tau


# ---------------------------------------------------------------------------
# Defense placement – complete graphs  (Theorem 6)
# ---------------------------------------------------------------------------

def complete_graph_defense(n: int, B: int) -> np.ndarray:
    """
    Optimal defense allocation for a complete graph with n nodes and integer
    budget B ∈ (n, n²)  (Theorem 6, Eq. 33).

    Returns
    -------
    tau : (n,) integer array, Σ τ_i = B, entries differ by at most 1.
    """
    if not (n < B < n ** 2):
        raise ValueError(f"Budget B={B} must satisfy n < B < n² ({n} < B < {n**2})")
    return _even_spread(n, B)


# ---------------------------------------------------------------------------
# Defense placement – complete bipartite graphs  (Algorithm 1 + Theorem 7)
# ---------------------------------------------------------------------------

def _wp_from_subbudget(np_: int, Bp: int) -> float:
    """Attack-success probability for the P-partition given sub-budget Bp."""
    tau_p_eff = _even_spread_bipartite(np_, Bp) / 2.0
    v_opt, _ = _solve_capture_eq(tau_p_eff)
    return -v_opt   # w_p = -v^p_opt > 0


def _wq_from_subbudget(nq: int, Bq: int) -> float:
    """Attack-success probability for the Q-partition given sub-budget Bq."""
    tau_q_eff = _even_spread_bipartite(nq, Bq) / 2.0
    v_opt, _ = _solve_capture_eq(tau_q_eff)
    return -v_opt


def complete_bipartite_defense(
    np_: int, nq: int, B: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimal defense allocation for a complete bipartite graph (Algorithm 1 +
    Theorem 7) with |P| = np_, |Q| = nq, and even integer budget B.

    Bisects on the sub-budget B_p to equalise the attack-success probabilities
    w_p and w_q, then applies Theorem 7 to each partition.

    Parameters
    ----------
    np_ : number of nodes in partition P.
    nq  : number of nodes in partition Q.
    B   : total even defense budget, 2(np_+nq) < B < 2(np_²+nq²).

    Returns
    -------
    tau_p : (np_,) even integer attack durations for P-nodes.
    tau_q : (nq,) even integer attack durations for Q-nodes.
    """
    if B % 2 != 0:
        raise ValueError("Budget B must be even for the bipartite defense problem")
    B_lo = 2 * np_
    B_hi = B - 2 * nq

    if B_lo >= B_hi:
        raise ValueError(
            f"Budget B={B} is too small: need B > 2(np_+nq)={2*(np_+nq)}"
        )

    # Bisect on B_p (Algorithm 1)
    LB, UB = B_lo, B_hi
    while UB - LB > 2:
        Bp = (LB + UB) // 2
        if Bp % 2 != 0:        # keep Bp even
            Bp += 1
        Bq = B - Bp
        wp = _wp_from_subbudget(np_, Bp)
        wq = _wq_from_subbudget(nq, Bq)
        if wp < wq:
            UB = Bp
        else:
            LB = Bp

    # Evaluate both candidates and pick the better one
    best = None
    for Bp_cand in [LB, UB]:
        Bp_cand = int(Bp_cand)
        if Bp_cand % 2 != 0:
            Bp_cand += 1
        Bq_cand = B - Bp_cand
        if Bq_cand < 2 * nq or Bp_cand < 2 * np_:
            continue
        wp = _wp_from_subbudget(np_, Bp_cand)
        wq = _wq_from_subbudget(nq, Bq_cand)
        obj = max(wp, wq)
        if best is None or obj < best[0]:
            best = (obj, Bp_cand, Bq_cand)

    _, Bp_opt, Bq_opt = best
    tau_p = _even_spread_bipartite(np_, Bp_opt)
    tau_q = _even_spread_bipartite(nq, Bq_opt)
    return tau_p, tau_q


# ---------------------------------------------------------------------------
# Upper bound on optimal capture probability  (Theorem 1)
# ---------------------------------------------------------------------------

def upper_bound(P: np.ndarray, tau: np.ndarray) -> float:
    """
    Compute the upper bound μ* ≤ min_i π_i τ_i from Theorem 1.

    Uses the stationary distribution of the provided transition matrix P.
    For a given optimised P the bound is tight up to the suboptimality
    established in Eqs. (14) and (22).
    """
    P = np.asarray(P, dtype=float)
    tau = np.asarray(tau, dtype=float)
    _check_stochastic(P)
    # Stationary distribution: left eigenvector for eigenvalue 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi) / np.sum(np.abs(pi))
    return float(np.min(pi * tau))
