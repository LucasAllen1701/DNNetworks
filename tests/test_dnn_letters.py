import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import dnn_fixed as F                          # noqa: E402
import dnn_letters as L                        # noqa: E402
import exp10_library_and_states as e10         # noqa: E402


def _streams(f, delta, out, rng, n, T=12):
    V = []
    for _ in range(n):
        x = rng.integers(0, 2, (T + 1, 6))
        y = np.zeros(T + 1, int)
        q = 0
        for t, l in enumerate(f(x)[:T + 1 - L.H]):
            y[t + L.H] = out[q][(int(l),)]
            q = delta[q][(int(l),)]
        V.append(np.c_[x, y])
    return V


def test_state_neurons_track_the_machine():
    delta, out, _ = e10.machine("tomita4")
    for f in (lambda X: (X[:, :3].sum(1) >= 2).astype(int), lambda X: X[:, 0] ^ X[:, 1]):
        rules, lr = L.letter_helpers(f, 6)
        lay, idx = L.make_layout(6, rules, lr, delta)
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.integers(0, 2, (13, 6))
            e = F.extend(lay, np.c_[x, np.zeros(13, int)])
            assert not e[:2, list(idx.values())].any()
            q = 0
            for t in range(11):
                l = int(f(x[t:t + 1])[0])
                assert [k for k, m in idx.items() if e[t + 2, m]] == [(q, l)]
                q = delta[q][(l,)]


def test_choose_letter_finds_the_letter():
    delta, out, _ = e10.machine("parity")
    f = lambda X: X[:, 2] ^ X[:, 4]
    V = _streams(f, delta, out, np.random.default_rng(1), 16)
    name, g, d, o, _ = L.choose_letter(V, 6)
    assert name == "x2^x4"
    assert L.same_composite(d, o, lambda x: (int(g(np.array([x]))[0]),), delta, out,
                            lambda x: (int(f(np.array([x]))[0]),), [tuple(p) for p in np.ndindex(*[2] * 6)])
