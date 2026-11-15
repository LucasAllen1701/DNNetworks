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

Added for experiment 16 (OR-of-AND functions):
- `build_cover`: a covering search. Each new helper must be off on every
  pattern whose answer is 0 and on for one uncovered positive pattern (the
  seed), with the smallest sum of |weights|; that makes it as general as the
  negatives allow. The output is then the OR of the helpers. On the full
  truth table this finds the 3 terms of dnf3 and the 4 terms of mux.
- `pair_rules` and `make_layout_h3`: a second hidden layer, at h = 3. Layer 1
  is a fixed library (every AND of two literals); layer 2 is built by `build`
  over the inputs and layer 1. Layer 2 reads the inputs through one-step
  copies (past shifts), so both layers see the same x(t). Layer-2 neurons are
  held at 0 until the "started" neuron is 1, so that the output is 0 at
  times 0 to 2. All are fixed neurons, so docs/Fixed_Neurons.md applies.
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


# ---- experiment 16: a covering search, and a second hidden layer

def pure_fit(X, y, cost_pos, eps=EPS, bound=None, weight_cost=0.01):
    """Weights (w, b) with w.x + b <= 1 - eps on every pattern with y = 0
    (hard), and w.x + b >= 1 on the positive patterns as far as possible: the
    slack of positive i costs cost_pos[i], and each |w_j| costs
    `weight_cost`."""
    X = np.asarray(X, float)
    neg, pos = X[y == 0], X[y == 1]
    m, k = X.shape[1], len(pos)
    bound = bound or m + 1
    I, Zn, Zp = np.eye(m), np.zeros((len(neg), k)), np.zeros((k, m))
    A = np.vstack([
        np.hstack([neg, np.ones((len(neg), 1)), Zn, np.zeros((len(neg), m))]),
        np.hstack([-pos, -np.ones((k, 1)), -np.eye(k), Zp]),
        np.hstack([I, np.zeros((m, 1 + k)), -I]),
        np.hstack([-I, np.zeros((m, 1 + k)), -I])])
    rhs = np.r_[np.full(len(neg), 1.0 - eps), -np.ones(k), np.zeros(2 * m)]
    cost = np.r_[np.zeros(m + 1), cost_pos, weight_cost * np.ones(m)]
    bounds = [(-bound, bound)] * (m + 1) + [(0, None)] * (k + m)
    res = linprog(cost, A_ub=A, b_ub=rhs, bounds=bounds)
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:m], res.x[m]


def pure_helper(X, y, covered, seed, eps=EPS):
    """One helper for the covering search: the rule with the smallest sum of
    |w_j| that is off on every negative pattern and on for the seed pattern
    (slack cost 1000), with a small reward for also covering other positives
    (slack cost 0.01 if uncovered, 0.001 if covered). Small weights make the
    rule as general as the negatives allow: for an OR of ANDs it is a term
    containing the seed."""
    pos = np.flatnonzero(y == 1)
    cost = np.where(covered[pos], 0.001, 0.01)
    cost[np.searchsorted(pos, seed)] = 1000.0
    r = integer_rule(*pure_fit(X, y, cost, eps, weight_cost=1.0), eps)
    c = fire(r, X)
    return r if c[seed] and not c[y == 0].any() else indicator(X[seed])


def build_cover(X, y, eps=EPS, max_helpers=20):
    """Helpers that are each off on every negative pattern and together on
    for every positive one (sequential covering). Each new helper is seeded
    with the first uncovered positive pattern (`pure_helper`)."""
    X, y = np.asarray(X, int), np.asarray(y, int)
    rules = []
    covered = np.zeros(len(X), bool)
    while not covered[y == 1].all():
        seed = np.flatnonzero((y == 1) & ~covered)[0]
        r = pure_helper(X, y, covered, seed, eps) if len(rules) < max_helpers else indicator(X[seed])
        rules.append(r)
        covered |= fire(r, X).astype(bool)
    return rules


def pair_rules(n_in):
    """Every AND of two literals: 4 sign patterns for each pair of inputs.
    Returns [(rule over the inputs, name)]."""
    out = []
    for i in range(n_in):
        for j in range(i + 1, n_in):
            for si in (1, 0):
                for sj in (1, 0):
                    w = np.zeros(n_in, int)
                    w[i], w[j] = (1 if si else -1), (1 if sj else -1)
                    out.append(((w, 1 - si - sj), f"{'' if si else '~'}x{i}&{'' if sj else '~'}x{j}"))
    return out


def layer1_values(X, layer1):
    """Features seen by layer 2: the inputs, then the layer-1 helpers."""
    X = np.asarray(X, int)
    return np.c_[X, np.array([fire(r, X) for r in layer1]).T] if layer1 else X


def gate(rule):
    """Weights K and bias shift so that the rule fires only when the gating
    neuron is 1: w.x + b + K g - K >= 1. K is at least the largest value of
    w.x + b on 0/1 points."""
    w, b = rule
    return max(int(np.maximum(w, 0).sum()) + int(b), 1)


def make_layout_h3(n_in, layer1, layer2):
    """Depth h = 3. Inputs 0 ... n_in - 1, output n_in, the always-on neuron,
    "started" (0 at time 0), "started2" (0 at times 0 and 1), one-step copies
    of the inputs, the layer-1 helpers (rules over the inputs), then the
    layer-2 helpers (rules over the copies and layer 1, held at 0 while
    "started" is 0). Returns (layout, columns the output may read)."""
    lay = F.Layout(n_in + 1, range(n_in), {n_in})
    lay.add_on()
    started = lay.add_fixed({lay.on: 1}, "started")
    started2 = lay.add_fixed({started: 1}, "started2")
    copies = [lay.add_fixed({i: 1}, f"x{i}-1") for i in range(n_in)]
    l1 = []
    for k, (wi, bi) in enumerate(layer1):
        row = {j: int(wi[j]) for j in range(n_in) if wi[j]}
        if bi:
            row[lay.on] = int(bi)
        l1.append(lay.add_fixed(row, f"layer1 {k}"))
    sources = copies + l1
    l2 = []
    for k, (wi, bi) in enumerate(layer2):
        K = gate((wi, bi))
        row = {sources[j]: int(wi[j]) for j in range(len(sources)) if wi[j]}
        row[started] = K
        if int(bi) - K:
            row[lay.on] = int(bi) - K
        l2.append(lay.add_fixed(row, f"layer2 {k}"))
    return lay, [started2] + l2
