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
        assert len(lay.fixed) == len(rules)
