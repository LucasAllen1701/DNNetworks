"""Discrete neural networks with box-valued weight regions.

Reference implementation of the RUN and TRAIN algorithms described in
docs/DiscreteNeuralNetworkNotes151126.md. Section numbers in the
comments refer to that document. Neurons are indexed from 0 here and from 1
in the notes.
"""
import numpy as np
from scipy.optimize import linprog

INF = np.inf
TOL = 1e-7  # floating-point tolerance on the firing threshold (notes, section 8.6)


class Net:
    """A discrete neural network (n, I, O, F, c) (Definition 3).

    F(i) is the box prod_j [L[i, j], U[i, j]] of allowed weight vectors into
    neuron i. c[i] is the cohort label: 0 for visible neurons, r for hidden
    neurons created in training round r.
    """

    def __init__(self, n_visible, inputs=(), outputs=()):
        self.n = n_visible
        self.n_visible = n_visible
        self.I = set(inputs)
        self.O = set(outputs)
        self.L = np.full((n_visible, n_visible), -INF)
        self.U = np.full((n_visible, n_visible), INF)
        self.c = [0] * n_visible
        self.round = 0

    def space(self):
        """Numbers stored: 2n^2 interval endpoints plus n cohort labels."""
        return 2 * self.n * self.n + self.n


def sigma(x):
    """Step activation (Definition 1): 1 where x >= 1, else 0."""
    return (x >= 1 - TOL).astype(int)


def run(A, inputs, v0, input_sets, steps):
    """Algorithm RUN (section 3).

    A: weight matrix whose row i is a point of F(i).
    inputs: the set I of input neurons.
    v0: initial activation vector (for a trained sequence, its key).
    input_sets: input_sets[t] is the set S_t of input neurons active at time t.
    Returns the activation vectors v(0), ..., v(steps).
    """
    v = np.array(v0, int).copy()
    for i in inputs:
        v[i] = int(i in input_sets[0])
    out = [v.copy()]
    for t in range(steps):
        w = sigma(A @ v)
        for i in inputs:
            w[i] = int(i in input_sets[t + 1])
        v = w
        out.append(v.copy())
    return out


def _pairs(seqs, g):
    """Constraint pairs (x, y) for neuron g (section 4.1)."""
    X, y = [], []
    for s in seqs:
        for t in range(len(s) - 1):
            X.append(s[t])
            y.append(s[t + 1][g])
    return np.array(X, float), np.array(y, int)


def _bounds(lo, hi):
    return [(None if l == -INF else l, None if u == INF else u) for l, u in zip(lo, hi)]


def feas(seqs, g, L, U, eps):
    """FEAS(g) (section 4.2): is P_g intersected with the box of g non-empty?"""
    X, y = _pairs(seqs, g)
    A = np.where(y[:, None] == 1, -X, X)
    b = np.where(y == 1, -1.0, 1.0 - eps)
    res = linprog(np.zeros(X.shape[1]), A_ub=A, b_ub=b, bounds=_bounds(L[g], U[g]))
    return res.status == 0


def subbox(seqs, g, L, U, eps, cap=1.0):
    """SUBBOX(g) (section 4.2): a box inside P_g and the current box of g.

    Maximises the total width, each width capped at `cap`, then expands the
    endpoints that no constraint touches.
    """
    X, y = _pairs(seqs, g)
    m = X.shape[1]
    # LP variables: lower endpoints (m), upper endpoints (m), capped widths (m).
    A, b = [], []
    for x, target in zip(X, y):
        row = np.zeros(3 * m)
        if target == 1:
            row[:m] = -x                 # sum of lower endpoints >= 1
            b.append(-1.0)
        else:
            row[m:2 * m] = x             # sum of upper endpoints <= 1 - eps
            b.append(1.0 - eps)
        A.append(row)
    for j in range(m):                   # w_j <= u_j - l_j
        row = np.zeros(3 * m)
        row[j], row[m + j], row[2 * m + j] = 1, -1, 1
        A.append(row)
        b.append(0.0)
    bounds = _bounds(L[g], U[g]) + _bounds(L[g], U[g]) + [(0, cap)] * m
    cost = np.r_[np.zeros(2 * m), -np.ones(m)]
    res = linprog(cost, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds)
    if res.status != 0:
        raise RuntimeError(f"SUBBOX failed for neuron {g}: {res.message}")
    lo, hi = res.x[:m].copy(), res.x[m:2 * m].copy()
    in_pos = X[y == 1].sum(0) > 0 if (y == 1).any() else np.zeros(m, bool)
    in_neg = X[y == 0].sum(0) > 0 if (y == 0).any() else np.zeros(m, bool)
    lo[~in_pos] = L[g][~in_pos]          # expansion step
    hi[~in_neg] = U[g][~in_neg]
    return lo, hi


def train(net, V, eps=0.5):
    """Algorithm TRAIN, one round (section 4.4). Modifies `net` in place.

    V: list of sequences, each an array of shape (k + 1, n_visible) of 0/1
       activations on the visible neurons (input neurons included).
    eps: margin in (0, 1].
    Returns the extended sequences on all neurons. Row 0 of each is that
    sequence's key: the initial vector to pass to `run` to replay it.
    """
    net.round += 1
    r = net.round
    n_hidden = net.n - net.n_visible
    seqs = [np.hstack([np.array(s, int), np.zeros((len(s), n_hidden), int)]) for s in V]
    L, U, c = net.L, net.U, net.c

    def add_neuron(columns):
        """ADD (section 4.3): append a hidden neuron of cohort r."""
        nonlocal L, U, seqs
        m = len(c)
        new_L = np.full((m + 1, m + 1), -INF)
        new_U = np.full((m + 1, m + 1), INF)
        new_L[:m, :m] = L
        new_U[:m, :m] = U
        for j in range(m):
            if c[j] != 0 and c[j] != r:  # hidden neuron of an earlier round
                new_U[j, m] = 0.0
            if c[j] != r:                # the new neuron is only excited by its own cohort
                new_U[m, j] = 0.0
        L, U = new_L, new_U
        c.append(r)
        seqs = [np.hstack([s, col[:, None]]) for s, col in zip(seqs, columns)]
        return m

    def helper_columns(g, anti=False):
        """Predictor (or anti-predictor) activations of neuron g (section 4.3)."""
        columns = []
        for s in seqs:
            col = np.zeros(len(s), int)
            col[:-1] = (1 - s[1:, g]) if anti else s[1:, g]
            columns.append(col)
        return columns

    def restore(snapshot):
        nonlocal L, U, seqs
        L, U, labels, seqs = snapshot
        c[:] = labels

    worklist = [g for g in range(net.n_visible) if g not in net.I]
    refit = list(worklist)
    while worklist:
        g = worklist.pop(0)
        if feas(seqs, g, L, U, eps):
            continue
        snapshot = (L, U, list(c), seqs)
        added = [add_neuron(helper_columns(g))]                    # (a) predictor
        if c[g] == 0 and not feas(seqs, g, L, U, eps):
            restore(snapshot)
            added = [add_neuron(helper_columns(g, anti=True))]     # (b) anti-predictor
            if not feas(seqs, g, L, U, eps):
                restore(snapshot)                                  # (c) both
                added = [add_neuron(helper_columns(g)),
                         add_neuron(helper_columns(g, anti=True))]
        if not feas(seqs, g, L, U, eps):
            raise RuntimeError(f"neuron {g} still infeasible after adding helpers")
        worklist += added
        refit += added
    for g in refit:
        L[g], U[g] = subbox(seqs, g, L, U, eps)
    net.L, net.U, net.n = L, U, len(c)
    return seqs


def sample_weights(net, rng=None, mode="random", span=5.0):
    """A weight matrix with row i in F(i).

    mode: "random", "lower" (lower corners) or "upper" (upper corners).
    Unbounded sides are cut off `span` away from the finite side, or at
    +-span/2 when both sides are unbounded.
    """
    L, U = net.L.copy(), net.U.copy()
    free = np.isinf(L) & np.isinf(U)
    L[free], U[free] = -span / 2, span / 2
    L = np.where(np.isinf(L), U - span, L)
    U = np.where(np.isinf(U), L + span, U)
    if mode == "lower":
        return L
    if mode == "upper":
        return U
    rng = rng or np.random.default_rng()
    return L + rng.random(L.shape) * (U - L)
