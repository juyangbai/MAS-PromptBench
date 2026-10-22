"""Exact MAP inference for MAPRO's joint quality score.

Objective (paper Eq. for T(P)):

    T(P) = ∏_i g(p_i) · ∏_{(i,j)∈E} g(p_i, p_j),   g(·) ∈ (0, 1]

We seek argmax_P T(P), the MAP assignment. Equivalently, in log-space,
argmax_P [ Σ_i log g(p_i) + Σ_{(i,j)} log g(p_i,p_j) ].

The paper solves this with a language-guided variant of *max-product belief
propagation* on the (moralised, triangulated) graph, i.e. junction-tree
max-product with a backtracking downward pass. We realise exactly that
computation via **max-product variable elimination with traceback**:

  * eliminating a variable = the paper's upward message
        m_{i→j}(p_j) = max_{p_i} [ ... ]   (Eq. 6, the `max` over the eliminated
        variable of the combined potential), and
  * the stored argmax pointers = the downward backtracking pass that recovers
        the globally-optimal assignment.

The sets of variables that co-occur during elimination are the junction-tree
cliques; the largest one determines the treewidth (cost K^{w+1}). For the target
topologies this is tiny (Chain: a 3-clique; DMAD: pairwise/star). Correctness is
checked against brute force in ``tests/test_core.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Problem container
# --------------------------------------------------------------------------- #
@dataclass
class MAPProblem:
    """A pairwise MAP problem.

    var_names:        ordered list of variables (agents).
    cardinalities:    K per variable (pool size).
    node_potentials:  var -> array shape (K_v,)     of g(p_v)      in (0,1].
    edge_potentials:  (u,v) -> array shape (K_u,K_v) of g(p_u,p_v) in (0,1].
                      Direction is the DAG edge u->v; the factor couples u and v.
    """

    var_names: list[str]
    cardinalities: dict[str, int]
    node_potentials: dict[str, np.ndarray]
    edge_potentials: dict[tuple[str, str], np.ndarray]

    def log_factors(self) -> list[tuple[tuple[str, ...], np.ndarray]]:
        """Node and edge potentials as log-space factors over their variables."""
        factors: list[tuple[tuple[str, ...], np.ndarray]] = []
        for v in self.var_names:
            factors.append(((v,), _safe_log(np.asarray(self.node_potentials[v], float))))
        for (u, v), tab in self.edge_potentials.items():
            factors.append(((u, v), _safe_log(np.asarray(tab, float))))
        return factors


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))


# --------------------------------------------------------------------------- #
# Broadcasting a factor onto a superset of variables (log-space additive)
# --------------------------------------------------------------------------- #
def _broadcast(sub_vars: tuple[str, ...], sub_arr: np.ndarray, full_vars: list[str], K: dict[str, int]) -> np.ndarray:
    """Return an array over `full_vars` (⊇ sub_vars) equal to sub_arr broadcast."""
    pos = [full_vars.index(v) for v in sub_vars]
    order = np.argsort(pos)  # move sub axes into ascending order
    arr_t = np.transpose(sub_arr, axes=tuple(order))
    shape = [1] * len(full_vars)
    for k, p in enumerate(sorted(pos)):
        shape[p] = arr_t.shape[k]
    full_shape = tuple(K[v] for v in full_vars)
    return np.broadcast_to(arr_t.reshape(shape), full_shape)


# --------------------------------------------------------------------------- #
# Min-fill elimination order
# --------------------------------------------------------------------------- #
def min_fill_order(var_names: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Greedy min-fill elimination order (ties broken by name)."""
    adj: dict[str, set[str]] = {v: set() for v in var_names}
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    remaining = set(var_names)
    order: list[str] = []
    while remaining:
        best, best_fill = None, None
        for v in sorted(remaining):  # sorted → deterministic
            nb = adj[v] & remaining
            nb_list = list(nb)
            fill = 0
            for a in range(len(nb_list)):
                for b in range(a + 1, len(nb_list)):
                    if nb_list[b] not in adj[nb_list[a]]:
                        fill += 1
            if best_fill is None or fill < best_fill:
                best, best_fill = v, fill
        # eliminate `best`: connect its remaining neighbours
        nb = list(adj[best] & remaining)
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                adj[nb[a]].add(nb[b])
                adj[nb[b]].add(nb[a])
        remaining.remove(best)
        order.append(best)
    return order


# --------------------------------------------------------------------------- #
# MAP via max-product variable elimination with traceback
# --------------------------------------------------------------------------- #
@dataclass
class MAPResult:
    """MAP assignment (candidate index per variable), its log score and the elimination treewidth."""

    assignment: dict[str, int]
    log_score: float
    treewidth: int  # largest elimination clique size - 1
    messages: dict = field(default_factory=dict)  # elim_var -> combined-clique vars


def map_infer(problem: MAPProblem) -> MAPResult:
    """Exact MAP by max-product variable elimination with traceback."""
    K = problem.cardinalities
    factors = problem.log_factors()
    edges = [tuple(vs) for (vs, _) in factors if len(vs) == 2]
    order = min_fill_order(problem.var_names, [(u, v) for (u, v) in edges])

    back: list[tuple[str, list[str], np.ndarray]] = []
    max_clique = 1
    clique_info: dict[str, tuple[str, ...]] = {}

    active = list(factors)
    for v in order:
        involved = [(vs, arr) for (vs, arr) in active if v in vs]
        rest = [(vs, arr) for (vs, arr) in active if v not in vs]
        # union of variables in the involved factors (the elimination clique)
        union: list[str] = []
        for vs, _ in involved:
            for x in vs:
                if x not in union:
                    union.append(x)
        clique_info[v] = tuple(union)
        max_clique = max(max_clique, len(union))
        # combine (log-space sum) over the clique
        combined = np.zeros(tuple(K[x] for x in union))
        for vs, arr in involved:
            combined = combined + _broadcast(vs, arr, union, K)
        # max / argmax over v
        vaxis = union.index(v)
        maxed = np.max(combined, axis=vaxis)
        argmaxed = np.argmax(combined, axis=vaxis)
        remaining = [x for x in union if x != v]
        back.append((v, remaining, argmaxed))
        active = rest + [(tuple(remaining), maxed)]

    # everything eliminated → sum of remaining scalars = optimal log score
    total = 0.0
    for _vs, arr in active:
        total += float(np.asarray(arr).reshape(()))

    # traceback (downward pass)
    assignment: dict[str, int] = {}
    for v, remaining, argmaxed in reversed(back):
        if remaining:
            idx = tuple(assignment[r] for r in remaining)
            assignment[v] = int(argmaxed[idx])
        else:
            assignment[v] = int(argmaxed)

    return MAPResult(assignment=assignment, log_score=total, treewidth=max_clique - 1, messages=clique_info)


# --------------------------------------------------------------------------- #
# Brute-force reference (tests only)
# --------------------------------------------------------------------------- #
def brute_force_map(problem: MAPProblem) -> MAPResult:
    """MAP by enumerating every joint assignment (reference for tests)."""
    K = problem.cardinalities
    names = problem.var_names
    factors = problem.log_factors()
    best_score, best_assign = -np.inf, None
    for combo in product(*[range(K[v]) for v in names]):
        assign = dict(zip(names, combo))
        s = 0.0
        for vs, arr in factors:
            idx = tuple(assign[x] for x in vs)
            s += float(arr[idx])
        if s > best_score:
            best_score, best_assign = s, dict(assign)
    return MAPResult(assignment=best_assign, log_score=best_score, treewidth=-1)
