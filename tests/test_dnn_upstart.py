from itertools import product

import numpy as np

import dnn_upstart as U


def test_integer_rule_keeps_the_margin():
    rng = np.random.default_rng(0)
    X = np.array(list(product([0, 1], repeat=5)))
    for _ in range(50):
        w, b = rng.normal(size=5), rng.normal()
        s = X @ w + b
        keep = (s >= 1) | (s <= 1 - U.EPS)
        assert (U.fire(U.integer_rule(w, b), X)[keep] == (s[keep] >= 1)).all()


def test_build_makes_the_output_exact():
    X = np.array(list(product([0, 1], repeat=6)))
    rng = np.random.default_rng(1)
    for f in (lambda x: x.sum() % 2, lambda x: x[2 + 2 * x[0] + x[1]], lambda x: rng.integers(0, 2)):
        y = np.array([int(f(x)) for x in X])
        rules = U.build(X, y)
        H = np.array([U.fire(r, X) for r in rules]).T if rules else np.zeros((len(X), 0))
        _, exact = U._output_errors(H, y, U.EPS)
        assert exact
        lay = U.make_layout(6, rules)
        assert len(lay.fixed) == len(rules) + 1      # + the started neuron


def test_cover_finds_the_terms_of_an_or_of_ands():
    X = np.array(list(product([0, 1], repeat=6)))
    for f, terms in ((lambda x: (x[0] & x[1]) | (x[2] & x[3]) | (x[4] & x[5]), 3),
                     (lambda x: x[2 + 2 * x[0] + x[1]], 4)):
        y = np.array([int(f(x)) for x in X])
        rules = U.build_cover(X, y)
        H = np.array([U.fire(r, X) for r in rules]).T
        assert len(rules) == terms
        assert (H.max(1) == y).all()


def test_cover_is_exact_on_partial_data():
    rng = np.random.default_rng(3)
    X = np.unique(rng.integers(0, 2, (30, 6)), axis=0)
    y = rng.integers(0, 2, len(X))
    H = np.array([U.fire(r, X) for r in U.build_cover(X, y)]).T
    assert (H.max(1) == y).all()


def test_h3_layout_computes_layer2_two_steps_later():
    import dnn_fixed as F
    X = np.array(list(product([0, 1], repeat=6)))
    y = np.array([int((x[:3] == x[3:]).all()) for x in X])
    l1 = [r for r, _ in U.pair_rules(6)]
    Fv = U.layer1_values(X, l1)
    l2 = U.build(Fv, y)
    lay, allowed = U.make_layout_h3(6, l1, l2)
    xs = np.random.default_rng(0).integers(0, 2, (13, 6))
    e = F.extend(lay, np.c_[xs, np.zeros(13, int)])
    rows = [int("".join(map(str, x)), 2) for x in xs]
    want = np.array([U.fire(r, Fv[rows]) for r in l2]).T
    assert (e[2:, allowed[1:]] == want[:-2]).all()
    assert not e[:2, allowed].any()
