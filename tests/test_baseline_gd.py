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
