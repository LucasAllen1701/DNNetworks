"""Tests for dnn_box.py and dnn_buffer.py. Run with `python -m pytest` from the
repository root. Proposition numbers refer to docs/Box_Rule.md."""
import copy
from itertools import product

import numpy as np
import pytest
from scipy.optimize import linprog

import dnn
import dnn_box as B
import dnn_buffer as BUF
from tests.test_dnn import replays

RULES = [B.capped_rule(16.0), B.bounded_rule(8.0), B.volume_rule(8.0)]


def random_rounds(rng, n0, rounds):
    return [[rng.integers(0, 2, (int(rng.integers(2, 6)), n0)) for _ in range(int(rng.integers(1, 4)))]
            for _ in range(rounds)]


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.__name__)
def test_every_rule_keeps_the_theorem(rule):
    """Any box inside P_g and inside the box kept before will do: with each
    rule, every sequence trained so far replays from its key after every
    round, and the boxes of the visible neurons only shrink."""
    rng = np.random.default_rng(0)
    for _ in range(12):
        n0 = int(rng.integers(1, 4))
        inputs = {0} if (n0 > 1 and rng.random() < 0.5) else set()
        net = dnn.Net(n0, inputs)
        trained = []
        for V in random_rounds(rng, n0, 3):
            old_L, old_U, old_n = net.L.copy(), net.U.copy(), net.n
            with B.using(rule):
                trained += dnn.train(net, V)
            assert replays(net, trained, rng, trials=6)
            for g in range(n0):
                assert (net.L[g, :old_n] >= old_L[g] - 1e-7).all()
                assert (net.U[g, :old_n] <= old_U[g] + 1e-7).all()
    assert dnn.subbox is B._original


def pattern_sequence(x, y):
    """Three inputs and one output: the inputs show x, and the output is y at
    the next step."""
    return np.array([list(x) + [0], [0, 0, 0, y]])


def test_no_box_is_safe():
    """Proposition 6. After the single constraint (1,1,1) -> 1, no box inside
    P_g contains a correct weight vector for each of the tasks y = x1, y = x2
    and y = x3. So every rule goes wrong on one of them, and a buffer on none."""
    # a box needs l1 + l2 + l3 >= 1; to fit task i it needs the sum of the other two <= 1/2
    A = [[-1, -1, -1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert linprog([0, 0, 0], A_ub=A, b_ub=[-1, 0.5, 0.5, 0.5], bounds=[(None, None)] * 3).status == 2

    patterns = list(product((0, 1), repeat=3))
    first = [pattern_sequence((1, 1, 1), 1)]

    def hidden_added(train_round):
        added = []
        for i in range(3):
            net = dnn.Net(4, inputs={0, 1, 2}, outputs={3})
            state = train_round(net, None, first)
            rest = [pattern_sequence(x, x[i]) for x in patterns if x != (1, 1, 1)]
            train_round(net, state, rest)
            added.append(net.n > 4)
        return added

    for rule in [dnn.subbox] + RULES:
        def with_rule(net, state, V):
            with B.using(rule):
                dnn.train(net, V)
        assert any(hidden_added(with_rule))

    def with_buffer(net, state, V):
        state = state or BUF.Buffer([3], capacity=8)
        with state.active():
            dnn.train(net, V)
        return state
    assert not any(hidden_added(with_buffer))


def test_buffer_of_capacity_0_is_the_notes_algorithm():
    rng = np.random.default_rng(1)
    for _ in range(10):
        n0 = int(rng.integers(1, 4))
        plain, buffered = dnn.Net(n0), dnn.Net(n0)
        buf = BUF.Buffer(range(n0), capacity=0)
        for V in random_rounds(rng, n0, 3):
            dnn.train(plain, V)
            with buf.active():
                dnn.train(buffered, V)
        assert plain.n == buffered.n and buf.stored() == 0
        assert np.array_equal(plain.L, buffered.L) and np.array_equal(plain.U, buffered.U)


def replays_with_buffer(net, buf, trained, rng):
    for mode in ["lower", "upper", "mid", "random", "random", "random"]:
        A = BUF.sample_weights(net, buf, rng, mode)
        for s in trained:
            inputs = [{i for i in net.I if s[t][i]} for t in range(len(s))]
            v0 = np.r_[s[0], np.zeros(net.n - len(s[0]), int)]
            run = np.array(dnn.run(A, net.I, v0, inputs, len(s) - 1))
            if not (run[:, :s.shape[1]] == s).all():
                return False
    return True


@pytest.mark.parametrize("seed", range(4))
def test_buffer_keeps_the_theorem(seed):
    """Proposition 7: with a buffer of any capacity, every sequence trained so
    far replays from its key after every round, for weight matrices drawn from
    the regions. Small capacities make the buffer overflow."""
    rng = np.random.default_rng(10 + seed)
    for _ in range(10):
        n0 = int(rng.integers(1, 4))
        inputs = {0} if (n0 > 1 and rng.random() < 0.5) else set()
        net = dnn.Net(n0, inputs)
        capacity = int(rng.choice([1, 3, 8, 50]))
        buf = BUF.Buffer([g for g in range(n0) if g not in inputs], capacity)
        trained = []
        for V in random_rounds(rng, n0, 4):
            with buf.active():
                trained += dnn.train(net, V)
            assert all(len(rows) <= capacity for rows in buf.rows.values())
            assert replays_with_buffer(net, buf, trained, rng)
    assert dnn.feas.__module__ == "dnn" and dnn.subbox is B._original


@pytest.mark.parametrize("seed", range(3))
def test_buffer_never_adds_a_helper_for_a_threshold_task(seed):
    """Proposition 7: if the buffer does not overflow and the output is a
    threshold function of the visible state, no hidden neuron is ever added,
    even with one short stream per round."""
    rng = np.random.default_rng(20 + seed)
    k = 4
    w = rng.integers(-2, 3, k)                              # y(t+1) = [w . x(t) >= 1]
    net = dnn.Net(k + 1, inputs=range(k), outputs={k})
    buf = BUF.Buffer([k], capacity=2 ** (k + 1))
    for _ in range(40):
        x = rng.integers(0, 2, (5, k))
        y = np.r_[0, (x[:-1] @ w >= 1).astype(int)]
        with buf.active():
            dnn.train(net, [np.c_[x, y]])
        assert net.n == k + 1
    assert buf.compressions == 0
    A = BUF.sample_weights(net, buf, rng, "mid")
    seen = {x for x, _ in buf.rows[k]}
    for x in seen:                                          # right on every pattern it has seen
        assert int(A[k, :k + 1] @ np.array(x[:k + 1]) >= 1 - 1e-7) == int(np.array(x[:k]) @ w >= 1)
