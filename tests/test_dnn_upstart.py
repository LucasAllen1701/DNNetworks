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


def test_neighbour_pairs_are_next_to_each_other():
    rules = U.neighbour_pair_rules(4)
    assert len(rules) == 24 * 4
    for (w, b), _ in rules:
        i, j = np.flatnonzero(w)
        assert (j - i == 1 and i % 4 != 3) or j - i == 4


def test_translates_of_pixel_and_pair_rules():
    import dnn_upstart as Up
    side = 3
    src = Up.pixel_sources(9)
    w = np.zeros(9, int)
    w[[0, 1]] = [2, -1]                                   # pixels (0,0), (0,1)
    tr = Up.translates((w, 1), src, side)
    assert len(tr) == 3 * 2 - 1                           # 3 row shifts x 2 column shifts, minus itself
    assert all(b == 1 and sorted(t.tolist()).count(0) == 7 for t, b in tr)
    layer1 = [r for r, _ in Up.neighbour_pair_rules(side)]
    srcs = src + layer1
    v = np.zeros(len(srcs), int)
    v[9] = 1                                              # the first layer-1 pair, x0 & x1
    tr2 = Up.translates((v, 0), srcs, side)
    assert len(tr2) == 5
    X = np.random.default_rng(0).integers(0, 2, (20, 9))
    Z = Up.layer1_values(X, layer1)
    for t, b in tr2:
        j = int(np.flatnonzero(t)[0])
        assert j >= 9 and np.flatnonzero(layer1[j - 9][0]).tolist() in ([0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8])
    assert Up.fire((v, 0), Z).tolist() == (X[:, 0] & X[:, 1]).tolist()



def test_build_cover_local_shares_a_block_detector():
    import dnn_upstart as Up
    assert len(Up.windows(3, ((2, 2),))) == 4 and len(Up.windows(4)) == 9 + 4 + 4 + 4
    rng = np.random.default_rng(2)
    X = np.unique((rng.random((120, 16)) < 0.3).astype(int), axis=0)
    blk = lambda x: int(any(x.reshape(4, 4)[i:i + 2, j:j + 2].all() for i in range(3) for j in range(3)))
    y = np.array([blk(x) for x in X])
    for share in (False, True):
        rules = Up.build_cover_local(X, y, 4, Up.windows(4), share=share)
        on = np.any([Up.fire(r, X) for r in rules], axis=0).astype(int)
        assert (on == y).all()
    allx = np.array([[(v >> i) & 1 for i in range(16)] for v in range(1 << 16)])
    on = np.any([Up.fire(r, allx) for r in rules], axis=0).astype(int)
    assert (on == np.array([blk(x) for x in allx])).mean() > 0.99
