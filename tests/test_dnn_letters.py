import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import dnn_fixed as F                          # noqa: E402
import dnn_letters as L                        # noqa: E402
import dnn_merge as M                          # noqa: E402
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


def _dnf2(X):
    return (X[:, 0] & X[:, 1]) | (X[:, 2] & X[:, 3])


def test_compose_letter_finds_an_or_of_two_library_letters():
    delta, out, _ = e10.machine("tomita4")
    V = _streams(_dnf2, delta, out, np.random.default_rng(1), 32)
    assert L.choose_letter(V, 6) is None or L.choose_letter(V, 6)[0] != "(x0&x1)|(x2&x3)"
    name, f, d, o, level, _ = L.compose_letter(V, 6)
    X = np.array(list(np.ndindex(*(2,) * 6)))
    assert level == 2 and (f(X) == _dnf2(X)).all() and len(d) == len(delta)


def test_joint_search_reproduces_every_output():
    delta, out, _ = e10.machine("parity")
    f = lambda X: X[:, 0]
    V = _streams(f, delta, out, np.random.default_rng(2), 6)
    X, seqs = L._indexed(V, 6)
    assert L.joint_search(seqs, len(X), 1, np.zeros(len(X), int)) is None
    z, d, o = L.joint_search(seqs, len(X), 2, np.zeros(len(X), int))
    for w, outs in seqs:
        q = 0
        for p, y in zip(w, outs):
            assert o[q][z[p]] == y
            q = d[q][z[p]]
    found = L.learn_letter(V, 6)
    assert found["states_search"] == 2 and len(L._machine(found["f"](X), seqs)[0]) == len(found["delta"])


def test_bit_state_neurons_track_the_machine():
    rules = [(np.array([1, 1, 0, 0, 0, 0]), 0), (np.array([0, 0, 1, 0, 0, 1]), -1)]   # x0 or x1; x2 and x5
    letters = [(0, 0), (0, 1), (1, 0), (1, 1)]
    delta = [{a: (q + a[0] + 2 * a[1]) % 3 for a in letters} for q in range(3)]
    lay, idx = L.make_layout_bits(6, rules, delta)
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = rng.integers(0, 2, (13, 6))
        e = F.extend(lay, np.c_[x, np.zeros(13, int)])
        assert not e[:2, list(idx.values())].any()
        q = 0
        for t in range(11):
            a = tuple(int((x[t] @ w + b) >= 1) for w, b in rules)
            assert [k for k, m in idx.items() if e[t + 2, m]] == [(q, a)]
            q = delta[q][a]


def test_two_output_machine_and_layout():
    rules = [(np.array([1, 1, 0, 0, 0, 0]), 0)]                    # x0 or x1
    letters = [(0,), (1,)]
    delta = [{a: (q + a[0]) % 2 for a in letters} for q in range(2)]
    out = [{a: (delta[q][a], 1 - a[0]) for a in letters} for q in range(2)]
    rng = np.random.default_rng(4)
    V = []
    for _ in range(12):
        x = rng.integers(0, 2, (14, 6))
        y = np.zeros((14, 2), int)
        q = 0
        for t in range(11):
            a = (int(x[t, 0] or x[t, 1]),)
            y[t + 3] = out[q][a]
            q = delta[q][a]
        V.append(np.c_[x, y])
    d, o = L.learn_machine_multi(L.bit_streams(V, 6, [lambda X: X[:, 0] | X[:, 1]], 2), 1, 2, 3)
    assert len(d) == 2 and o[0][(1,)] == (1, 0)
    lay, idx = L.make_layout_bits(6, rules, M.complete(d, 1), n_out=2)
    assert lay.outputs == {6, 7} and lay.n_task == 8


def test_choose_bits_fast_equals_choose_bits():
    import exp33_bigger_screens as e33
    rng = np.random.default_rng(3)
    V = [e33.episode(rng, 10, 4) for _ in range(6)]
    lib = e33.features(4, e33.regions)[:12]
    assert L.choose_bits_fast(V, 32, lib, 3, n_out=2) == L.choose_bits(V, 32, lib, 3, n_out=2)
    V1 = [np.c_[s[:, :32], s[:, 32]] for s in V]
    assert L.choose_bits_fast(V1, 32, lib, 3) == L.choose_bits(V1, 32, lib, 3)
