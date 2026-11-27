"""Tests for dnn_clock: clock-neuron wake rounds and merge-based sleep."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnn import Net, run, sample_weights  # noqa: E402
from dnn_clock import merge_costs, sleep_merge, wake_clock  # noqa: E402


def visible_replays(net, keys, seqs, rng, trials=4):
    """Every sequence's visible part replays from its (zero-padded) key."""
    for mode in ["lower", "upper"] + ["random"] * (trials - 2):
        A = sample_weights(net, rng, mode)
        for key, s in zip(keys, seqs):
            s = np.asarray(s)
            sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
            v0 = np.r_[key, np.zeros(net.n - len(key), int)]
            out = np.array(run(A, net.I, v0, sets, len(s) - 1))
            if not (out[:, :s.shape[1]] == s).all():
                return False
    return True


def random_case(rng):
    nv = int(rng.integers(1, 4))
    net = Net(nv, inputs={0} if nv > 1 and rng.random() < 0.5 else set())
    rounds = [[rng.integers(0, 2, (rng.integers(2, 6), nv)) for _ in range(rng.integers(1, 3))]
              for _ in range(rng.integers(1, 3))]
    return net, rounds


def test_wake_clock_recalls_everything():
    rng = np.random.default_rng(0)
    for _ in range(40):
        net, rounds = random_case(rng)
        keys, seqs = [], []
        for V in rounds:
            ext = wake_clock(net, V, 1e-4)
            keys += [e[0] for e in ext]
            seqs += [np.asarray(s) for s in V]
            assert visible_replays(net, keys, seqs, rng)


def test_wake_clock_adds_one_neuron_per_vector_or_none():
    net = Net(1, outputs={0})
    wake_clock(net, [np.array([[1], [1], [1], [0]])], 0.5)
    assert net.n == 4                       # three clock neurons
    n = net.n
    wake_clock(net, [np.array([[1], [1]])], 0.5)   # weight 1 on itself is in its box
    assert net.n == n                       # fits already: nothing added


def test_sleep_merge_exact_at_zero_and_shrinks_at_infinity():
    rng = np.random.default_rng(1)
    for _ in range(20):
        net, rounds = random_case(rng)
        keys, seqs = [], []
        for V in rounds:
            ext = wake_clock(net, V, 1e-4)
            keys += [e[0] for e in ext]
            seqs += [np.asarray(s) for s in V]
        keys = [np.r_[k, np.zeros(net.n - len(k), int)] for k in keys]
        n0 = net.n
        K, _ = sleep_merge(net, keys, 0.0)
        assert visible_replays(net, K, seqs, rng)
        K, _ = sleep_merge(net, K, np.inf)
        assert net.n <= n0
        assert net.L.shape == (net.n, net.n) and len(net.c) == net.n
        assert not np.isfinite(merge_costs(net.L, net.U, K, net.n_visible)).any()


def test_wake_tree_recalls_without_keys():
    from dnn_clock import wake_tree
    rng = np.random.default_rng(2)
    for _ in range(10):
        T, n = 6, 5
        V = []
        for _ in range(n):
            x = rng.integers(0, 2, T)
            y = np.r_[0, 0, np.cumsum(x)[:-2] % 2]          # parity, answered at depth 2
            V.append(np.c_[(np.arange(T) == 0).astype(int), x, y])
        net = Net(3, inputs={0, 1}, outputs={2})
        ext, nodes, fallback = wake_tree(net, V, 1e-4)
        assert not fallback and nodes > 0
        zero = [np.r_[s[0], np.zeros(net.n - 3, int)] for s in V]
        assert visible_replays(net, zero, V, rng)           # no keys needed
