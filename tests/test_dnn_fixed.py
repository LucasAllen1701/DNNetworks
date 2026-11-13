"""Tests for dnn_fixed.py. Run with `python -m pytest` from the repository root.

Proposition numbers refer to docs/Fixed_Neurons.md."""
import numpy as np
import pytest

import dnn
import dnn_fixed as F
from tests.test_dnn import replays


def random_layout(rng):
    n0 = int(rng.integers(1, 4))
    inputs = {0} if (n0 > 1 and rng.random() < 0.5) else set()
    depth = int(rng.integers(1, 4))
    extra = str(rng.choice(["none", "clock", "twins"]))
    return F.make_layout(n0, inputs, {n0 - 1}, depth, extra=extra)


def test_extend_shifts_clock_and_twins():
    """The extended columns are what their names say (section 2)."""
    rng = np.random.default_rng(0)
    s = rng.integers(0, 2, (9, 2))
    for extra in ("none", "clock", "twins"):
        lay = F.make_layout(2, {0}, {1}, depth=3, extra=extra)
        v = F.extend(lay, s)
        col = {name: v[:, j] for j, name in enumerate(lay.names)}
        assert (v[:, :2] == s).all()
        for i in range(2):
            for d in range(1, 4):
                past = np.r_[np.zeros(d, int), s[:-d, i]]
                assert (col[f"v{i}-{d}"] == past).all()
                if extra == "twins":
                    started = np.r_[np.zeros(d, int), np.ones(len(s) - d, int)]
                    assert (col[f"not v{i}-{d}"] == started - past).all()
        if extra != "none":
            assert (col["on"] == 1).all()
        if extra == "clock":
            for d in range(1, 4):
                assert (col[f"on-{d}"] == np.r_[np.zeros(d, int), np.ones(len(s) - d, int)]).all()


def test_example_1_needs_no_key():
    """Notes, Example 1 (1,1,1,0) with two past shifts of the neuron: no hidden
    neuron is added and the sequence replays from its first value alone."""
    lay = F.make_layout(1, outputs={0}, depth=2)
    net = F.make_net(lay)
    (key_seq,) = F.train(net, lay, [np.array([[1], [1], [1], [0]])])
    assert net.n == lay.n_visible == 3
    assert list(key_seq[0]) == [1, 0, 0]
    for mode in ("lower", "upper", "random"):
        A = dnn.sample_weights(net, np.random.default_rng(0), mode)
        out = dnn.run(A, net.I, F.start(lay, net, [1]), [set()] * 4, 3)
        assert [int(v[0]) for v in out] == [1, 1, 1, 0]


@pytest.mark.parametrize("seed", range(4))
def test_theorem_with_fixed_neurons(seed):
    """Proposition 1: with fixed neurons, every sequence trained so far still
    replays from its key after every round, the fixed boxes stay points on
    the visible neurons, and condition (D) holds."""
    rng = np.random.default_rng(seed)
    for _ in range(25):
        lay = random_layout(rng)
        net = F.make_net(lay)
        eps = float(rng.choice([0.1, 0.5, 1.0]))
        trained = []
        for _ in range(int(rng.integers(1, 5))):
            V = [rng.integers(0, 2, (int(rng.integers(2, 7)), lay.n_task))
                 for _ in range(int(rng.integers(1, 4)))]
            if trained and rng.random() < 0.3:
                old = trained[int(rng.integers(len(trained)))][:, :lay.n_task]
                s = rng.integers(0, 2, old.shape)
                s[0] = old[0]
                V.append(s)
            trained += F.train(net, lay, V, eps)
            assert replays(net, trained, rng, trials=8)
            assert F.fixed_boxes_intact(net, lay)
        assert all(net.c[m] == 0 for m in lay.fixed)
        for i in range(net.n):
            for j in range(net.n):
                if net.c[i] != 0 and net.c[j] != net.c[i]:
                    assert net.U[i, j] <= 0


@pytest.mark.parametrize("seed", range(4))
def test_fixed_neurons_follow_their_rule_on_any_run(seed):
    """Proposition 2: in a run started with the hidden neurons at 0, on inputs
    and from a state never trained, every fixed neuron applies its rule to the
    run's own values, for every weight matrix in the boxes."""
    rng = np.random.default_rng(100 + seed)
    for _ in range(25):
        lay = random_layout(rng)
        net = F.make_net(lay)
        for _ in range(int(rng.integers(1, 4))):
            F.train(net, lay, [rng.integers(0, 2, (int(rng.integers(2, 7)), lay.n_task))
                               for _ in range(2)])
        weights = lay.weights
        nv = lay.n_visible
        for mode in ("lower", "upper", "random"):
            A = dnn.sample_weights(net, rng, mode)
            steps = 8
            input_sets = [{i for i in net.I if i == lay.on or rng.random() < 0.5}
                          for _ in range(steps + 1)]
            v0 = np.zeros(net.n, int)
            v0[:nv] = rng.integers(0, 2, nv)        # any visible state, fixed neurons included
            out = np.array(dnn.run(A, net.I, v0, input_sets, steps))
            assert (out[:, nv:] == 0).all()         # hidden neurons stay silent (Lemma 1)
            for t in range(steps):
                for m, w in weights.items():
                    assert out[t + 1, m] == int(w @ out[t, :nv] >= 1)


def test_twins_and_clock_span_the_same_columns():
    """Proposition 3: on every extended sequence, shift + twin = clock shift."""
    rng = np.random.default_rng(3)
    s = rng.integers(0, 2, (10, 3))
    clock = F.make_layout(3, {0}, {2}, depth=3, extra="clock")
    twins = F.make_layout(3, {0}, {2}, depth=3, extra="twins")
    c = dict(zip(clock.names, F.extend(clock, s).T))
    w = dict(zip(twins.names, F.extend(twins, s).T))
    for d in range(1, 4):
        for i in range(3):
            assert (w[f"v{i}-{d}"] + w[f"not v{i}-{d}"] == c[f"on-{d}"]).all()


def test_tomita_2_is_learned_with_a_shift_of_the_output():
    """(10)* needs the output from one step earlier. With past shifts of depth
    1 of the input and the output it is learned with no hidden neuron and is
    exact on every stream of length 8, trained or not."""
    def target(x):
        ok = lambda p: len(p) % 2 == 0 and all(b == 1 - i % 2 for i, b in enumerate(p))
        return np.array([int(ok(list(x[:t]))) for t in range(len(x))])

    rng = np.random.default_rng(0)
    lay = F.make_layout(2, {0}, {1}, depth=1)
    net = F.make_net(lay)
    V = []
    for _ in range(64):
        x = rng.integers(0, 2, 13)
        V.append(np.c_[x, target(x)])
    F.train(net, lay, V)
    assert net.n == lay.n_visible
    for mode in ("lower", "upper", "random"):
        A = dnn.sample_weights(net, rng, mode)
        for code in range(2 ** 8):
            x = np.array([(code >> k) & 1 for k in range(8)])
            y = target(x)
            out = np.array(dnn.run(A, net.I, F.start(lay, net, [x[0], y[0]]),
                                   [{0} if b else set() for b in x], len(x) - 1))
            assert (out[:, 1] == y).all()
