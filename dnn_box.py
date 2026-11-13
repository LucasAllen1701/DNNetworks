"""Other rules for SUBBOX: which box to keep inside P_g and the current box.

TRAIN keeps, for every neuron g it refits, one box inside the polyhedron P_g
of weight vectors that fit the round's sequences (notes, section 4.2). The
Theorem only needs the box to lie inside P_g and inside the box kept before,
so the choice among such boxes is free. It matters for later rounds: if the
kept box contains no weight vector that fits the next round, TRAIN falls back
on helper chains and the neuron stops learning (experiments 8.2, 9.2, 10.4).
docs/Box_Rule.md describes the rules and what was measured.

`dnn.py` is unchanged. `using(rule)` swaps the rule in for the duration of a
`with` block:

    with dnn_box.using(dnn_box.volume_rule(8)):
        dnn.train(net, V)

A hidden neuron's box is never refitted after its own round, so the choice
only matters for visible neurons; `only_for` applies a rule to a given set of
neurons and dnn.subbox to the rest.
"""
from contextlib import contextmanager

import numpy as np
from scipy.optimize import linprog

import dnn

_original = dnn.subbox


def _problem(seqs, g, L, U, eps):
    """Constraints shared by every rule. LP variables: lower endpoints (m),
    upper endpoints (m), widths (m)."""
    X, y = dnn._pairs(seqs, g)
    m = X.shape[1]
    n = 3 * m
    A, b = [], []
    for x, target in zip(X, y):
        row = np.zeros(n)
        if target == 1:
            row[:m] = -x                 # sum of lower endpoints >= 1
            b.append(-1.0)
        else:
            row[m:2 * m] = x             # sum of upper endpoints <= 1 - eps
            b.append(1.0 - eps)
        A.append(row)
    for j in range(m):                   # w_j <= u_j - l_j
        row = np.zeros(n)
        row[j], row[m + j], row[2 * m + j] = 1, -1, 1
        A.append(row)
        b.append(0.0)
    in_pos = X[y == 1].sum(0) > 0 if (y == 1).any() else np.zeros(m, bool)
    in_neg = X[y == 0].sum(0) > 0 if (y == 0).any() else np.zeros(m, bool)
    return X, y, m, A, b, in_pos, in_neg


def _finish(res, g, L, U, m, in_pos, in_neg):
    if res.status != 0:
        raise RuntimeError(f"SUBBOX failed for neuron {g}: {res.message}")
    lo, hi = res.x[:m].copy(), res.x[m:2 * m].copy()
    lo, hi = np.clip(lo, L[g], U[g]), np.clip(hi, L[g], U[g])   # guard against solver round-off
    hi = np.maximum(hi, lo)
    lo[~in_pos] = L[g][~in_pos]          # expansion step, as in dnn.subbox
    hi[~in_neg] = U[g][~in_neg]
    return lo, hi


def capped_rule(cap=1.0):
    """dnn.subbox with another cap on each width (the notes use 1)."""
    def chosen(seqs, g, L, U, eps):
        return _original(seqs, g, L, U, eps, cap=cap)

    chosen.__name__ = f"capped_{cap:g}"
    return chosen


def bounded_rule(bound=8.0):
    """Widest box with every endpoint between -bound and bound.

    dnn.subbox caps each width. When a coordinate may be unbounded on one
    side, the cap is reached wherever the other endpoint is put, so that
    endpoint is whatever the solver returns. Here the endpoints are bounded
    and the total width is maximised with no cap, so each endpoint is pushed
    as far out as the constraints allow. The expansion step is unchanged, so
    endpoints that no constraint touches are still unbounded. If no box fits
    inside the bound, the bound is doubled.
    """
    def chosen(seqs, g, L, U, eps):
        X, y, m, A, b, in_pos, in_neg = _problem(seqs, g, L, U, eps)
        cost = np.r_[np.zeros(2 * m), -np.ones(m)]
        M = bound
        for _ in range(40):
            lo_b, hi_b = np.maximum(L[g], -M), np.minimum(U[g], M)
            if (lo_b <= hi_b).all():
                ends = [(lo_b[j], hi_b[j]) for j in range(m)]
                res = linprog(cost, A_ub=np.array(A), b_ub=np.array(b), bounds=ends * 2 + [(0, None)] * m)
                if res.status == 0:
                    return _finish(res, g, L, U, m, in_pos, in_neg)
            M *= 2
        raise RuntimeError(f"SUBBOX failed for neuron {g}")

    chosen.__name__ = f"bounded_{bound:g}"
    return chosen


def volume_rule(bound=8.0, iters=12, delta=1e-3):
    """Box of (approximately) the largest volume with every endpoint between
    -bound and bound.

    Maximises the sum of the logarithms of the widths, over the coordinates
    that are free to vary and that some constraint touches, by the
    Frank-Wolfe method: each step solves one linear program and moves part of
    the way towards its solution. A product of widths is 0 as soon as one
    width is 0, so no coordinate is left without room, and the box covers as
    much as it can of the weight vectors that fit the data so far. The
    expansion step is unchanged. If no box fits inside the bound, the bound
    is doubled.
    """
    def chosen(seqs, g, L, U, eps):
        X, y, m, A, b, in_pos, in_neg = _problem(seqs, g, L, U, eps)
        A, b = np.array(A)[:len(y), :2 * m], np.array(b)[:len(y)]     # endpoints only
        free = (U[g] > L[g]) & (in_pos | in_neg)
        order = np.zeros((m, 2 * m))                                  # l_j <= u_j
        order[np.arange(m), np.arange(m)], order[np.arange(m), m + np.arange(m)] = 1, -1
        A, b = np.vstack([A, order]), np.r_[b, np.zeros(m)]
        M = bound
        for _ in range(40):
            lo_b, hi_b = np.maximum(L[g], -M), np.minimum(U[g], M)
            ends = [(lo_b[j], hi_b[j]) for j in range(m)] * 2
            start = None
            if (lo_b <= hi_b).all():
                # start from the most balanced box: maximise the smallest free width
                At = np.hstack([A, np.zeros((len(A), 1))])
                rows = np.zeros((int(free.sum()), 2 * m + 1))
                for k, j in enumerate(np.flatnonzero(free)):
                    rows[k, j], rows[k, m + j], rows[k, 2 * m] = 1, -1, 1     # t <= u_j - l_j
                cost = np.zeros(2 * m + 1)
                cost[2 * m] = -1
                first = linprog(cost, A_ub=np.vstack([At, rows]), b_ub=np.r_[b, np.zeros(len(rows))],
                                bounds=ends + [(0, 2 * M)])
                if first.status == 0:
                    start = first.x[:2 * m]
            if start is not None:
                break
            M *= 2
        else:
            raise RuntimeError(f"SUBBOX failed for neuron {g}")
        z = start
        idx = np.flatnonzero(free)

        def value(z):
            return np.log(z[m + idx] - z[idx] + delta).sum()

        for _ in range(iters if len(idx) else 0):
            grad = np.zeros(2 * m)
            grad[m + idx] = 1.0 / (z[m + idx] - z[idx] + delta)
            grad[idx] = -grad[m + idx]
            step = linprog(-grad, A_ub=A, b_ub=b, bounds=ends)
            if step.status != 0:
                break
            d = step.x - z
            if grad @ d < 1e-9:
                break
            lo_g, hi_g = 0.0, 1.0                       # line search: value is concave along d
            for _ in range(30):
                mid1, mid2 = lo_g + (hi_g - lo_g) / 3, hi_g - (hi_g - lo_g) / 3
                if value(z + mid1 * d) < value(z + mid2 * d):
                    lo_g = mid1
                else:
                    hi_g = mid2
            z = z + (lo_g + hi_g) / 2 * d

        class _Res:                                    # same shape as a linprog result
            status, x = 0, z
        return _finish(_Res, g, L, U, m, in_pos, in_neg)

    chosen.__name__ = f"volume_{bound:g}"
    return chosen


def only_for(box_rule, neurons):
    """`box_rule` for the given neurons, dnn.subbox for every other neuron."""
    neurons = set(neurons)

    def chosen(seqs, g, L, U, eps):
        return (box_rule if g in neurons else _original)(seqs, g, L, U, eps)

    chosen.__name__ = box_rule.__name__
    return chosen


@contextmanager
def using(box_rule):
    """Make dnn.train use `box_rule` inside the `with` block."""
    previous = dnn.subbox
    dnn.subbox = box_rule
    try:
        yield
    finally:
        dnn.subbox = previous
