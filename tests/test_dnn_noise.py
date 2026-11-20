"""Tests for dnn_noise.py."""
import numpy as np

import dnn_merge as M
import dnn_noise as N


def parity_streams(rng, n, length, flip=0.0):
    """Streams of one input bit; output at t + 2 is the parity of inputs 0..t."""
    out = []
    for _ in range(n):
        x = rng.integers(0, 2, length)
        y = np.zeros(length, int)
        y[2:] = np.cumsum(x)[:-2] % 2
        noise = rng.random(length) < flip
        noise[:2] = False
        out.append(np.c_[x, y ^ noise])
    return out


def test_noisy_is_nested_and_fixed():
    f = lambda x: int(sum(x) >= 2)
    xs = [np.array(v) for v in np.ndindex(2, 2, 2, 2)]
    flipped = {p: {i for i, x in enumerate(xs) if N.noisy(f, p, 3)(x) != f(x)} for p in (0, 0.2, 0.5)}
    assert flipped[0] == set() and flipped[0.2] <= flipped[0.5]
    assert all(N.noisy(f, 0.5, 3)(x) == N.noisy(f, 0.5, 3)(x) for x in xs)


PARITY_DELTA = [{(0,): 0, (1,): 1}, {(0,): 1, (1,): 0}]
PARITY_OUT = [{(0,): 0, (1,): 1}, {(0,): 1, (1,): 0}]


def test_fit_machine_exact_on_clean_parity():
    rng = np.random.default_rng(0)
    V = parity_streams(rng, 6, 12)
    delta, out, err = N.fit_machine(V, 1, 1, 2, 2)
    assert err == 0
    assert M.same_machine(delta, [{a: o[0] for a, o in row.items()} for row in out], PARITY_DELTA, PARITY_OUT)


def test_heldout_recovers_parity_under_noise():
    rng = np.random.default_rng(1)
    V = parity_streams(rng, 20, 30, flip=0.1)
    delta, out, err, n = N.learn_machine_heldout(V, 1, 1, 2, max_states=3)
    assert n == 2 and err > 0
    assert M.same_machine(delta, [{a: o[0] for a, o in row.items()} for row in out], PARITY_DELTA, PARITY_OUT)


def test_local_search_matches_exhaustive_cost():
    rng = np.random.default_rng(3)
    V = parity_streams(rng, 10, 20, flip=0.1)
    S, Y = N._encode(V, 1, 1, 2)
    D, cost = N._local_search(S, Y, 2, 2, 2, np.random.default_rng(0))
    assert cost == N.fit_machine(V, 1, 1, 2, 2)[2]


def test_relabel_gives_machine_outputs():
    rng = np.random.default_rng(2)
    noisy_V = parity_streams(rng, 5, 10, flip=0.3)
    delta = [{(0,): 0, (1,): 1}, {(0,): 1, (1,): 0}]
    out = [{(0,): (0,), (1,): (1,)}, {(0,): (1,), (1,): (0,)}]
    clean = N.relabel(noisy_V, 1, 1, 2, [lambda X: X[:, 0]], delta, out)
    for s in clean:
        assert (s[2:, 1] == np.cumsum(s[:, 0])[:-2] % 2).all()
