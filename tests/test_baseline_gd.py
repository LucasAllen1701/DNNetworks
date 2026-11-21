import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import baseline_gd as G


def test_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = X[:, 0] ^ X[:, 1]
    assert (G.MLP(2, hidden=8, seed=0).fit(X, y).predict(X) == y).all()


def test_rnn_learns_running_parity():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (16, 8, 1))
    Y = np.cumsum(X[:, :, 0], 1) % 2
    assert (G.RNN(1, hidden=16, seed=0).fit(X, Y, steps=3000, lr=0.01).predict(X) == Y).mean() > 0.95


def test_gru_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, (3, 5, 2)).astype(float)
    Y = rng.integers(0, 2, (3, 5)).astype(float)
    net = G.GRU(2, hidden=4, seed=1)
    _, grads = net.grads(X, Y)
    for p, g in zip(net.params(), grads):
        for i in range(min(p.size, 6)):
            idx = np.unravel_index(i, p.shape)
            old = p[idx]
            p[idx] = old + 1e-6
            up = net.grads(X, Y)[0]
            p[idx] = old - 1e-6
            down = net.grads(X, Y)[0]
            p[idx] = old
            assert abs((up - down) / 2e-6 - g[idx]) < 1e-6


def test_gru_learns_running_parity():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (16, 8, 1))
    Y = np.cumsum(X[:, :, 0], 1) % 2
    assert (G.GRU(1, hidden=16, seed=0).fit(X, Y, steps=1500, lr=0.01).predict(X) == Y).mean() > 0.95
