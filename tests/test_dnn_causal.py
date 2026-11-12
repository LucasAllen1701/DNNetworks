"""Tests for dnn_causal.py. Run with `python -m pytest` from the repository root."""
import numpy as np
import pytest

from dnn_causal import CHAIN, FEATURE, CausalNet, hand_built, run, sample_weights, train
from test_dnn import replays


@pytest.mark.parametrize("seed", range(4))
def test_current_round_replays_from_keys(seed):
    """Learning (part 1 of the Theorem) still holds: after every round, every
    sequence of that round replays exactly, on all neurons, from its key."""
    rng = np.random.default_rng(seed)
    for _ in range(25):
        n0 = int(rng.integers(1, 5))
        inputs = {0} if (n0 > 1 and rng.random() < 0.5) else set()
        net = CausalNet(n0, inputs, {n0 - 1})
        eps = float(rng.choice([0.1, 0.5, 1.0]))
        for _ in range(int(rng.integers(1, 4))):
            V = [rng.integers(0, 2, (int(rng.integers(2, 7)), n0))
                 for _ in range(int(rng.integers(1, 4)))]
            trained = train(net, V, eps, max_features=2)
            assert replays(net, trained, rng)


def test_feature_rows_are_points_and_read_no_chain_neuron():
    rng = np.random.default_rng(7)
    net = CausalNet(3, {0}, {2})
    for _ in range(3):
        train(net, [rng.integers(0, 2, (6, 3)) for _ in range(3)])
    for i, k in enumerate(net.kind):
        if k == FEATURE:
            assert (net.L[i] == net.U[i]).all()
            assert all(net.L[i, j] == 0 for j in range(net.n) if net.kind[j] == CHAIN)


def _streams(rng, f, count, T=12):
    out = []
    for _ in range(count):
        x = rng.integers(0, 2, T + 1)
        out.append(np.c_[x, f(x)])
    return out


def _exact_from_zero(net, seqs, rng):
    """Every stream is reproduced on the visible neurons when run from its
    visible initial state with all hidden neurons at 0."""
    for mode in ["lower", "upper", "random", "random"]:
        A = sample_weights(net, rng, mode)
        for s in seqs:
            v0 = np.r_[s[0], np.zeros(net.n - net.n_visible, int)]
            sets = [{0} if s[t, 0] else set() for t in range(len(s))]
            out = np.array(run(A, net.I, v0, sets, len(s) - 1))
            if not (out[:, :net.n_visible] == s).all():
                return False
    return True


@pytest.mark.parametrize("task", [
    lambda x: np.r_[0, 0, x[:-2]],                    # y(t+1) = x(t-1)
    lambda x: np.r_[0, 0, x[1:-1] & x[:-2]],          # y(t+1) = x(t) AND x(t-1)
], ids=["delay", "and"])
def test_tasks_learned_with_features_only(task):
    rng = np.random.default_rng(0)
    net = CausalNet(2, {0}, {1})
    train(net, _streams(rng, task, 16))
    assert net.count(FEATURE) >= 1 and net.count(CHAIN) == 0
    assert _exact_from_zero(net, _streams(rng, task, 50), rng)


def test_hand_built_two_step_parity():
    """h1 = s(x - h1 - h2), h2 = s(h1 + h2 - x), y = s(h1 + h2) computes
    y(t+2) = parity of x(0..t) (experiment 6a)."""
    net = hand_built(2, {0}, {1},
                     features=[[1, 0, -1, -1], [-1, 0, 1, 1]],
                     visible_rows={1: [0, 0, 1, 1]})
    parity2 = lambda x: np.r_[0, 0, np.cumsum(x[:-2]) % 2]
    assert _exact_from_zero(net, _streams(np.random.default_rng(1), parity2, 50), np.random.default_rng(2))
