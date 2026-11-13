"""A constraint buffer: keep a neuron's first constraints exactly, and only
turn them into a box when there are too many.

In the notes each weight region F(g) is a box. TRAIN must then pick one box
inside the polyhedron of weight vectors that fit a round, and after a small
round that choice can exclude every weight vector that fits the next one
(docs/Box_Rule.md, Proposition 6: no rule for choosing the box avoids this).

Here F(g), for the chosen neurons, is

    F(g) = box(g)  intersected with  {a : a fits every stored constraint},

where a stored constraint is a pair (x, y): an activation pattern x of the
neurons that existed when it was stored, and the target y of neuron g at the
next step. At most `capacity` constraints are stored per neuron. When a round
would exceed that, the stored constraints and the round's constraints are
turned into a box by a SUBBOX rule and the buffer is emptied, as in the notes.
With capacity 0 this is exactly the notes' algorithm.

The Theorem needs only that the new F(g) lies inside P_g and inside the old
F(g), which holds in both cases. A pattern stored before a hidden neuron was
added has 0 for that neuron, which is that neuron's value on every earlier
sequence (Lemma 1). `dnn.py` is unchanged: `Buffer.active()` swaps FEAS and
SUBBOX for the duration of a `with` block.
"""
from contextlib import contextmanager

import numpy as np
from scipy.optimize import linprog

import dnn


class Buffer:
    def __init__(self, neurons, capacity, eps=0.5, box_rule=None):
        """neurons: the neurons that get a buffer (normally the visible
        non-input neurons that are trained). box_rule: the SUBBOX rule used
        when a buffer overflows (default dnn.subbox)."""
        self.capacity = capacity
        self.eps = eps
        self.rows = {g: [] for g in neurons}        # g -> [(pattern tuple, target)]
        self.box_rule = box_rule or dnn.subbox
        self.compressions = 0

    def stored(self):
        return sum(len(r) for r in self.rows.values())

    def _padded(self, g, width):
        return {(x + (0,) * (width - len(x)), y) for x, y in self.rows[g]}

    def _as_sequences(self, g, width):
        """The stored constraints of g as two-step sequences, so that the
        FEAS and SUBBOX of dnn.py read them like any other constraint."""
        out = []
        for x, y in self._padded(g, width):
            s = np.zeros((2, width), int)
            s[0] = x
            s[1, g] = y
            out.append(s)
        return out

    @contextmanager
    def active(self):
        """Use the buffer in every dnn.train call inside the `with` block."""
        feas0, subbox0 = dnn.feas, dnn.subbox
        pending, overflow = {}, []

        def feas(seqs, g, L, U, eps):
            if self.rows.get(g):
                seqs = list(seqs) + self._as_sequences(g, seqs[0].shape[1])
            return feas0(seqs, g, L, U, eps)

        def subbox(seqs, g, L, U, eps):
            if g not in self.rows:
                return subbox0(seqs, g, L, U, eps)
            width = seqs[0].shape[1]
            X, y = dnn._pairs(seqs, g)
            rows = self._padded(g, width) | {(tuple(int(v) for v in x), int(t)) for x, t in zip(X, y)}
            if len(rows) <= self.capacity:
                pending[g] = sorted(rows)
                return L[g].copy(), U[g].copy()         # the box is kept as it is
            pending[g] = []
            overflow.append(g)
            return self.box_rule(list(seqs) + self._as_sequences(g, width), g, L, U, eps)

        dnn.feas, dnn.subbox = feas, subbox
        try:
            yield self
        finally:
            dnn.feas, dnn.subbox = feas0, subbox0
        self.rows.update(pending)
        self.compressions += len(overflow)


def _nearest(a0, rows, lo, hi, eps):
    """The point closest to a0 (sum of absolute differences) that fits the
    stored constraints and lies between lo and hi. None if there is none."""
    n = len(a0)
    A, b = [], []
    for x, y in rows:
        row = np.zeros(2 * n)
        row[:len(x)] = x
        if y == 1:
            A.append(-row)
            b.append(-1.0)
        else:
            A.append(row)
            b.append(1.0 - eps)
    eye = np.eye(n)
    A = np.vstack([np.array(A), np.hstack([eye, -eye]), np.hstack([-eye, -eye])])
    b = np.r_[b, a0, -a0]
    bounds = [(None if np.isinf(l) else l, None if np.isinf(h) else h) for l, h in zip(lo, hi)]
    res = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=A, b_ub=b, bounds=bounds + [(0, None)] * n)
    return res.x[:n] if res.status == 0 else None


def sample_weights(net, buffer, rng=None, mode="random"):
    """A weight matrix whose rows lie in the regions F. Rows without stored
    constraints are sampled as by dnn.sample_weights ("mid" gives the midpoint
    of the box). A row with stored constraints is the point of its region
    closest to that sample."""
    lo, hi = dnn.sample_weights(net, mode="lower"), dnn.sample_weights(net, mode="upper")
    A = (lo + hi) / 2 if mode == "mid" else dnn.sample_weights(net, rng, mode)
    for g, rows in buffer.rows.items():
        if rows:
            a = _nearest(A[g], rows, lo[g], hi[g], buffer.eps)
            if a is None:                           # the region lies outside the sampling cut-off
                a = _nearest(A[g], rows, net.L[g], net.U[g], buffer.eps)
            if a is None:
                raise RuntimeError(f"no weight vector found in the region of neuron {g}")
            A[g] = a
    return A
