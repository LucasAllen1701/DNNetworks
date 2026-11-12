"""Causal variant of the discrete neural network trainer.

Described in docs/Causal_Variant.md. dnn.py is the baseline and is not
modified; this module reuses its RUN, FEAS, SUBBOX and weight sampling.

There are three kinds of neuron:

- "visible": as in dnn.py (cohort 0);
- "feature": a hidden neuron with fixed weights (its box is a single point)
  that reads only visible neurons, feature neurons and itself. It is 0 at
  time 0 and computed by the network afterwards, so it needs no key;
- "chain": predictor / anti-predictor helpers exactly as in dnn.py
  (cohort r >= 1 for round r, condition (D), stored in keys).

Feature neurons fire on every sequence, old ones included, so the
no-forgetting part of the Theorem no longer holds. Learning still holds:
every sequence of the latest round replays exactly from its key.
"""
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack as sp_hstack, identity

from dnn import INF, Net, _bounds, _pairs, feas, run, sample_weights, sigma, subbox  # noqa: F401

VISIBLE, FEATURE, CHAIN = "visible", "feature", "chain"
FEATURE_COHORT = -1             # cohort label of feature neurons


class CausalNet(Net):
    """A Net that also records the kind of every neuron and, for feature
    neurons, a short description of what they compute."""

    def __init__(self, n_visible, inputs=(), outputs=()):
        super().__init__(n_visible, inputs, outputs)
        self.kind = [VISIBLE] * n_visible
        self.feature_info = {}  # neuron index -> (template, i, j)

    def count(self, kind):
        return sum(k == kind for k in self.kind)


# ---------------------------------------------------------------- candidates

TEMPLATES = ("copy", "and", "or", "and_not", "latch", "set_reset")


def candidate_weights(template, i, j, self_index, m):
    """Weight row of length m + 1 (the new neuron is index m == self_index)."""
    w = np.zeros(m + 1)
    if template == "copy":
        w[i] = 1
    elif template == "and":
        w[i] += 0.5
        w[j] += 0.5
    elif template == "or":
        w[i] += 1
        w[j] += 1
    elif template == "and_not":
        w[i] += 1
        w[j] -= 1
    elif template == "latch":
        w[i] += 1
        w[self_index] += 1
    elif template == "set_reset":
        w[i] += 1
        w[self_index] += 1
        w[j] -= 2
    else:
        raise ValueError(template)
    return w


def candidates(sources):
    """(template, i, j) for every candidate over the source neurons."""
    out = [("copy", i, None) for i in sources] + [("latch", i, None) for i in sources]
    for i, j in combinations(sources, 2):
        out += [("and", i, j), ("or", i, j)]
    for i, j in permutations(sources, 2):
        out += [("and_not", i, j), ("set_reset", i, j)]
    return out


def simulate_column(seq, w):
    """Activations of a new neuron with weight row w (last entry: itself),
    started at 0, reading the known columns of seq at the previous step."""
    col = np.zeros(len(seq), int)
    base = seq @ w[:-1]
    for t in range(len(seq) - 1):
        col[t + 1] = sigma(np.array([base[t] + w[-1] * col[t]]))[0]
    return col


def simulate_features(net, seq_visible):
    """Values of every neuron on a new sequence: visible values from the
    sequence, feature neurons simulated from 0, chain neurons 0."""
    s = np.zeros((len(seq_visible), net.n), int)
    s[:, :net.n_visible] = seq_visible
    feats = [i for i, k in enumerate(net.kind) if k == FEATURE]
    if feats:
        W = net.L[feats]        # point boxes: L == U
        for t in range(len(s) - 1):
            s[t + 1, feats] = sigma(W @ s[t])
    return s


# ------------------------------------------------------------- slack LP

def total_slack(seqs, g, L, U, eps):
    """Minimum total slack of FEAS(g): one non-negative slack per
    constraint, counted with multiplicity (identical pairs are merged)."""
    X, y = _pairs(seqs, g)
    return _slack(X, y, L[g], U[g], eps)


def _slack(X, y, lo, hi, eps):
    rows, counts = np.unique(np.c_[X, y], axis=0, return_counts=True)
    X, y = rows[:, :-1], rows[:, -1].astype(int)
    p, m = X.shape
    A = np.where(y[:, None] == 1, -X, X)
    b = np.where(y == 1, -1.0, 1.0 - eps)
    A_ub = sp_hstack([csr_matrix(A), -identity(p, format="csr")], format="csr")
    cost = np.r_[np.zeros(m), counts.astype(float)]
    res = linprog(cost, A_ub=A_ub, b_ub=b, bounds=_bounds(lo, hi) + [(0, None)] * p)
    if res.status != 0:
        raise RuntimeError(f"slack LP failed: {res.message}")
    return float(res.fun)


# ------------------------------------------------------------- training

def train(net, V, eps=0.5, max_features=4, slack_tol=1e-9):
    """One training round of the causal variant. Modifies `net` in place.

    V: list of sequences on the visible neurons (input neurons included).
    Returns the extended sequences on all neurons; row 0 of each is its key.
    """
    net.round += 1
    r = net.round
    seqs = [simulate_features(net, np.array(s, int)) for s in V]
    L, U, c, kind = net.L, net.U, net.c, net.kind

    def add(kind_new, columns, row_box=None):
        """Append a neuron; lift every box by the rules of the variant."""
        nonlocal L, U, seqs
        m = len(c)
        new_L = np.full((m + 1, m + 1), -INF)
        new_U = np.full((m + 1, m + 1), INF)
        new_L[:m, :m], new_U[:m, :m] = L, U
        for j in range(m):              # weight from the new neuron into j
            if kind[j] == FEATURE:
                new_L[j, m] = new_U[j, m] = 0.0
            elif kind[j] == CHAIN and (kind_new == FEATURE or c[j] != r):
                new_U[j, m] = 0.0
        if kind_new == FEATURE:
            new_L[m], new_U[m] = row_box, row_box
            c.append(FEATURE_COHORT)
        else:                           # chain neuron of cohort r
            for j in range(m):
                if c[j] != r:
                    new_U[m, j] = 0.0
            c.append(r)
        kind.append(kind_new)
        L, U = new_L, new_U
        seqs = [np.hstack([s, col[:, None]]) for s, col in zip(seqs, columns)]
        return m

    def feature_search(g):
        """Greedy search over the candidate pool (step 3 of the variant)."""
        best = total_slack(seqs, g, L, U, eps)
        for _ in range(max_features):
            if best <= slack_tol:
                return
            m = len(c)
            sources = [i for i in range(m) if kind[i] != CHAIN]
            X, y = _pairs(seqs, g)
            known = {np.concatenate([s[:, i] for s in seqs]).tobytes() for i in range(m)}
            lo = np.r_[L[g], -INF]          # new coordinate of a visible g is unrestricted
            hi = np.r_[U[g], INF]
            choice = None
            for template, i, j in candidates(sources):
                w = candidate_weights(template, i, j, m, m)
                cols = [simulate_column(s, w) for s in seqs]
                key = np.concatenate(cols).tobytes()
                if key in known or not any(col.any() for col in cols):
                    continue            # duplicate of an existing or earlier candidate, or silent
                known.add(key)
                xcol = np.concatenate([col[:-1] for col in cols])
                value = _slack(np.c_[X, xcol], y, lo, hi, eps)
                if value < best - slack_tol:
                    best, choice = value, (template, i, j, w, cols)
            if choice is None:
                return
            template, i, j, w, cols = choice
            idx = add(FEATURE, cols, row_box=w)
            net.feature_info[idx] = (template, i, j)

    def helper_columns(g, anti=False):
        columns = []
        for s in seqs:
            col = np.zeros(len(s), int)
            col[:-1] = (1 - s[1:, g]) if anti else s[1:, g]
            columns.append(col)
        return columns

    def snapshot():
        return L, U, list(c), list(kind), seqs

    def restore(snap):
        nonlocal L, U, seqs
        L, U, labels, kinds, seqs = snap
        c[:] = labels
        kind[:] = kinds

    worklist = [g for g in range(net.n_visible) if g not in net.I]
    refit = list(worklist)
    while worklist:
        g = worklist.pop(0)
        if feas(seqs, g, L, U, eps):
            continue
        if kind[g] == VISIBLE:
            feature_search(g)
            if feas(seqs, g, L, U, eps):
                continue
        # Fallback: predictor / anti-predictor chains, as in dnn.train.
        snap = snapshot()
        added = [add(CHAIN, helper_columns(g))]
        if kind[g] == VISIBLE and not feas(seqs, g, L, U, eps):
            restore(snap)
            added = [add(CHAIN, helper_columns(g, anti=True))]
            if not feas(seqs, g, L, U, eps):
                restore(snap)
                added = [add(CHAIN, helper_columns(g)), add(CHAIN, helper_columns(g, anti=True))]
        if not feas(seqs, g, L, U, eps):
            raise RuntimeError(f"neuron {g} still infeasible after adding helpers")
        worklist += added
        refit += added
    for g in refit:
        L[g], U[g] = subbox(seqs, g, L, U, eps)
    net.L, net.U, net.n = L, U, len(c)
    return seqs


def hand_built(n_visible, inputs, outputs, features, visible_rows):
    """A CausalNet with given feature neurons and point boxes for visible rows.

    features: list of weight rows of length n_visible + len(features).
    visible_rows: dict neuron -> weight row of the same length.
    Used to show that a task is realisable with feature neurons.
    """
    net = CausalNet(n_visible, inputs, outputs)
    n = n_visible + len(features)
    net.L = np.zeros((n, n))
    net.U = np.zeros((n, n))
    for g, row in visible_rows.items():
        net.L[g] = net.U[g] = row
    for k, row in enumerate(features):
        net.L[n_visible + k] = net.U[n_visible + k] = row
    net.kind += [FEATURE] * len(features)
    net.c += [FEATURE_COHORT] * len(features)
    net.n = n
    return net
