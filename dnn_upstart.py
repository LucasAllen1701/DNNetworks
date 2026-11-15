"""Helper neurons computed from the inputs, in the style of the upstart
algorithm (Frean 1990), for Boolean functions at depth h = 2.

Task: the output at time t + 2 is f(x(t)), a fixed Boolean function of the
input vector x(t). The helpers form one hidden layer: each helper is a fixed
neuron that reads only the inputs (and the always-on neuron), so its value at
t + 1 is a function of x(t). They are built from the training data before
round 1 and installed with `Layout.add_fixed`, so docs/Fixed_Neurons.md
applies unchanged and the Theorem (no forgetting) still holds. The output is
then trained by `dnn.train`, unchanged.

Building the helpers (`build`), on the distinct training pairs (x, f(x)):
1. Fit the output as well as it can be fitted over the helpers so far (an LP
   that minimises the total violation, then the sum of |weights|), with a
   free bias. With no helpers this is a constant.
2. If the fit is exact, stop.
3. Otherwise the wrongly-off patterns get a helper that should be on for them
   and off for every pattern whose answer is 0; the wrongly-on patterns get a
   helper that should be on for them and off for every pattern whose answer
   is 1. Each helper is itself a best-fit perceptron of the inputs, so it may
   be imperfect; step 1 then sees the new errors.
4. A helper that adds nothing new is replaced by the indicator of one wrong
   pattern (always realisable). After `max_helpers`, indicators of all
   remaining positive patterns are added, which makes the fit exact.

The first helper is therefore the best perceptron of f, as in upstart.
`lookup` gives the comparison layer: one indicator per positive pattern seen.
"""
import numpy as np
from scipy.optimize import linprog

import dnn_fixed as F

EPS = 0.5


def best_fit(X, y, eps=EPS, bound=None, slack_cost=1000.0):
    """Weights w and bias b minimising the total violation of
    w.x + b >= 1 (y = 1) and w.x + b <= 1 - eps (y = 0), and among those the
    sum of |w_j| (which favours few, small weights). Returns (w, b)."""
    X = np.asarray(X, float)
    k, m = X.shape
    bound = bound or m + 1
    sign = np.where(y == 1, -1.0, 1.0)
    # variables: w (m), b (1), slack (k), a (m) with a_j >= |w_j|
    A = np.hstack([sign[:, None] * X, sign[:, None], -np.eye(k), np.zeros((k, m))])
    I, Z = np.eye(m), np.zeros((m, 1 + k))
    A = np.vstack([A, np.hstack([I, Z, -I]), np.hstack([-I, Z, -I])])
    rhs = np.r_[np.where(y == 1, -1.0, 1.0 - eps), np.zeros(2 * m)]
    cost = np.r_[np.zeros(m + 1), slack_cost * np.ones(k), np.ones(m)]
    bounds = [(-bound, bound)] * (m + 1) + [(0, None)] * (k + m)
    res = linprog(cost, A_ub=A, b_ub=rhs, bounds=bounds)
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:m], res.x[m]


def integer_rule(w, b, eps=EPS):
    """Integer weights and bias whose step function (sum >= 1) agrees with
    w.x + b on every 0/1 point that keeps the margin (>= 1 or <= 1 - eps)."""
    m = len(w)
    K = int(np.ceil((m + 2) / eps)) + 1          # rounding error < K * eps / 2
    cut = int(np.ceil(K * (1 - eps / 2)))
    return np.round(K * np.asarray(w)).astype(int), int(round(K * b)) - cut + 1


def fire(rule, X):
    wi, bi = rule
    return (np.asarray(X) @ wi + bi >= 1).astype(int)


def indicator(p):
    """The rule that fires exactly on the 0/1 pattern p."""
    p = np.asarray(p, int)
    return np.where(p == 1, 1, -len(p)), 1 - int(p.sum())


def _output_errors(H, y, eps):
    """Best fit of the output over the helper values H, with a free bias (in
    the network it comes from the "started" neuron). Returns the predictions
    and whether the fit is exact."""
    w, b = best_fit(H, y, eps)
    pred = fire(integer_rule(w, b, eps), H)
    return pred, bool((pred == y).all())


def build(X, y, eps=EPS, max_helpers=20):
    """Helper rules for the distinct patterns X with answers y."""
    X, y = np.asarray(X, int), np.asarray(y, int)
    rules = []
    cols = np.zeros((len(X), 0), int)

    def add(rule):
        nonlocal cols
        c = fire(rule, X)
        if any((c == cols[:, j]).all() for j in range(cols.shape[1])) or not c.any():
            return False
        rules.append(rule)
        cols = np.c_[cols, c]
        return True

    while True:
        pred, exact = _output_errors(cols, y, eps)
        if exact:
            return rules
        if len(rules) >= max_helpers:
            for p in X[y == 1]:
                add(indicator(p))
            return rules
        progress = False
        for wrong, other in (((y == 1) & (pred == 0), y == 0), ((y == 0) & (pred == 1), y == 1)):
            if not wrong.any():
                continue
            rows = wrong | other
            w, b = best_fit(X[rows], wrong[rows].astype(int), eps)
            r = integer_rule(w, b, eps)
            if (fire(r, X)[wrong] == 1).any() and add(r):
                progress = True
            elif add(indicator(X[np.flatnonzero(wrong)[0]])):
                progress = True
        if not progress:                         # cannot happen with indicators; be safe
            for p in X[y == 1]:
                add(indicator(p))
            return rules


def lookup(X, y):
    """One indicator per positive pattern."""
    return [indicator(p) for p in np.asarray(X, int)[np.asarray(y) == 1]]


def make_layout(n_in, rules=()):
    """Inputs 0 ... n_in - 1, output n_in, the always-on neuron, the "started"
    neuron (0 at time 0, then 1: the output's bias from time 2 on, while the
    output at time 1 must be 0), then one fixed neuron per helper rule
    (reading the inputs and the always-on neuron)."""
    lay = F.Layout(n_in + 1, range(n_in), {n_in})
    lay.add_on()
    lay.add_fixed({lay.on: 1}, "started")
    for k, (wi, bi) in enumerate(rules):
        row = {j: int(wi[j]) for j in range(n_in) if wi[j]}
        if bi:
            row[lay.on] = int(bi)
        lay.add_fixed(row, f"helper{k}")
    return lay
