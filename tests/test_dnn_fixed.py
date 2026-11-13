"""Tests for dnn_fixed.py. Run with `python -m pytest` from the repository root.

Propositions 1-3 are in docs/Fixed_Neurons.md; Propositions 4-5 and Lemma S
are in docs/Latency_And_State.md."""
from itertools import product

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


# ---- training path, library, machines (docs/Latency_And_State.md)

def test_skipping_fixed_neurons_gives_the_same_boxes():
    """`train` leaves the fixed neurons out of TRAIN's worklist by default.
    Every other neuron must get the same box as when they are included."""
    rng = np.random.default_rng(7)
    for _ in range(15):
        lay = random_layout(rng)
        fast, slow = F.make_net(lay), F.make_net(lay)
        for _ in range(3):
            V = [rng.integers(0, 2, (int(rng.integers(2, 7)), lay.n_task)) for _ in range(2)]
            a = F.train(fast, lay, V)
            b = F.train(slow, lay, V, refit_fixed=True)
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
        others = [i for i in range(fast.n) if i not in lay.fixed]
        assert fast.n == slow.n and fast.c == slow.c
        assert np.allclose(fast.L[others], slow.L[others]) and np.allclose(fast.U[others], slow.U[others])
        nv = lay.n_visible
        assert all((fast.L[m, nv:] == 0).all() and (fast.U[m, nv:] == 0).all() for m in lay.fixed)


def random_machine(rng):
    k = int(rng.integers(1, 3))
    patterns = list(product((0, 1), repeat=k))
    n = int(rng.integers(1, 5))
    delta = [{a: int(rng.integers(n)) for a in patterns} for _ in range(n)]
    out = [{a: int(rng.integers(2)) for a in patterns} for _ in range(n)]
    return k, delta, out, int(rng.integers(2))


def machine_stream(rng, k, delta, out, y0, steps, latency):
    """A task sequence [inputs, output] of the machine on random inputs."""
    x = rng.integers(0, 2, (steps + 1, k))
    y, q = [y0], 0
    for t in range(steps):
        a = tuple(int(v) for v in x[t])
        y.append(out[q][a])
        q = delta[q][a]
    y = np.array(y)
    if latency:
        y = np.r_[y[0], y[:-1]]
    return np.c_[x, y]


def state_layout(k, delta):
    lay = F.Layout(k + 1, range(k), {k})
    lay.add_on()
    return lay, F.add_automaton(lay, delta)


@pytest.mark.parametrize("seed", range(3))
def test_state_neurons_follow_the_machine(seed):
    """Lemma S: neuron (q, a) is on at time t + 1 exactly when the machine was
    in state q and read a at time t."""
    rng = np.random.default_rng(200 + seed)
    for _ in range(20):
        k, delta, out, y0 = random_machine(rng)
        lay, index = state_layout(k, delta)
        s = machine_stream(rng, k, delta, out, y0, 20, 0)
        v = F.extend(lay, s)
        q = 0
        for t in range(20):
            a = tuple(int(b) for b in s[t, :k])
            on_now = {key for key, m in index.items() if v[t + 1, m]}
            assert on_now == {(q, a)}
            q = delta[q][a]


@pytest.mark.parametrize("seed", range(3))
def test_output_is_learned_from_state_neurons(seed):
    """Proposition 5 on random machines. With the state neurons installed,
    the first round adds no hidden neuron at latency 1; at latency 0 it adds
    none exactly when the machine passes `realisable_without_latency`. When
    none is added, the trained network is exact on unseen streams."""
    rng = np.random.default_rng(300 + seed)
    for _ in range(12):
        k, delta, out, y0 = random_machine(rng)
        for latency in (0, 1):
            lay, _ = state_layout(k, delta)
            net = F.make_net(lay)
            F.train(net, lay, [machine_stream(rng, k, delta, out, y0, 14, latency) for _ in range(150)])
            expected = latency == 1 or F.realisable_without_latency(out, delta)
            assert (net.n == lay.n_visible) == expected
            if not expected:
                continue
            for mode in ("lower", "upper"):
                A = dnn.sample_weights(net, mode=mode)
                for _ in range(20):
                    s = machine_stream(rng, k, delta, out, y0, 14, latency)
                    inputs = [{i for i in range(k) if row[i]} | {lay.on} for row in s]
                    run = np.array(dnn.run(A, net.I, F.start(lay, net, s[0]), inputs, 14))
                    assert (run[:, k] == s[:, k]).all()


def parity(x):
    return np.r_[0, np.cumsum(x[:-1, 0]) % 2]


def no_three_zeros(x):          # Tomita 4
    bits = "".join(map(str, x[:, 0]))
    return np.array([int("000" not in bits[:t]) for t in range(len(bits))])


def test_machine_from_task_and_the_latency_0_limit():
    """Proposition 4 on two named tasks: parity cannot be computed at latency
    0 and Tomita 4 can. The machines found have 2 and 4 states."""
    delta, out, y0 = F.machine_from_task(parity, 1)
    assert len(delta) == 2 and y0 == 0
    assert not F.realisable_without_latency(out)
    delta, out, y0 = F.machine_from_task(no_three_zeros, 1)
    assert len(delta) == 4 and y0 == 1
    assert F.realisable_without_latency(out)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, (15, 1))
        y, q = no_three_zeros(x), 0
        for t in range(14):
            a = (int(x[t, 0]),)
            assert out[q][a] == y[t + 1]
            q = delta[q][a]


def test_library_is_task_independent_and_solves_delayed_xor():
    """The library has no duplicate or silent features, is the same on every
    call, and with it y(t+2) = x(t) xor x(t-1) needs no hidden neuron."""
    def layout():
        lay = F.Layout(2, {0}, {1})
        lay.add_on()
        prev = {0: 0, 1: 1}
        for s in (1, 2):
            for i in (0, 1):
                prev[i] = lay.add_fixed({prev[i]: 1}, f"v{i}-{s}")
        F.add_library(lay)
        return lay

    lay = layout()
    assert lay.names == layout().names
    rng = np.random.default_rng(1)
    probes = np.vstack([F.extend(lay, rng.integers(0, 2, (30, 2))) for _ in range(30)])
    columns = {probes[:, j].tobytes() for j in range(lay.n_visible)}
    assert len(columns) == lay.n_visible and probes.any(axis=0).all()

    V = []
    for _ in range(64):
        x = rng.integers(0, 2, 13)
        y = np.r_[0, 0, x[:11] ^ np.r_[0, x[:10]]]      # y(t+2) = x(t) xor x(t-1)
        V.append(np.c_[x, y])
    net = F.make_net(lay)
    F.train(net, lay, V)
    assert net.n == lay.n_visible and F.fixed_boxes_intact(net, lay)
