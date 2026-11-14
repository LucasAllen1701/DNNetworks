"""Tests for dnn_merge.py. Run with `python -m pytest` from the repository root.

Proposition 8 is in docs/State_Merging.md."""
from itertools import product

import numpy as np
import pytest

import dnn
import dnn_fixed as F
import dnn_merge as M


def random_machine(rng, max_states=4):
    k = int(rng.integers(1, 3))
    patterns = list(product((0, 1), repeat=k))
    n = int(rng.integers(1, max_states + 1))
    delta = [{a: int(rng.integers(n)) for a in patterns} for _ in range(n)]
    out = [{a: int(rng.integers(2)) for a in patterns} for _ in range(n)]
    return k, delta, out


def stream(rng, k, delta, out, steps, h=2):
    """A training sequence [inputs, output] of the machine at output timing h."""
    x = rng.integers(0, 2, (steps + h, k))
    y, q = [0] * h, 0
    for t in range(steps):
        a = tuple(int(v) for v in x[t])
        y.append(out[q][a])
        q = delta[q][a]
    return np.c_[x, y]


def outputs_of(delta, out, word):
    q, result = 0, []
    for a in word:
        result.append(out[q][a])
        q = delta[q][a]
    return result


def minimal(delta, out):
    """Number of states of the smallest machine equal to (delta, out)."""
    reach = sorted(F.reachable_states(delta))
    block = {q: tuple(sorted(out[q].items())) for q in reach}
    while True:
        new = {q: (block[q], tuple(sorted((a, block[n]) for a, n in delta[q].items()))) for q in reach}
        if len(set(new.values())) == len(set(block.values())):
            return len(set(new.values()))
        block = new


@pytest.mark.parametrize("seed", range(4))
def test_merged_machine_reproduces_the_training_streams(seed):
    """Proposition 8(1): whatever the data, the tree and the merged machine
    give the observed outputs on every training stream, and merging never
    gives more states than the tree has nodes."""
    rng = np.random.default_rng(seed)
    for _ in range(25):
        k, delta, out = random_machine(rng)
        seqs = [stream(rng, k, delta, out, int(rng.integers(1, 9))) for _ in range(int(rng.integers(1, 12)))]
        obs = [M.observations(s, k) for s in seqs]
        tree = M.learn_machine(seqs, k, merging=False)
        merged = M.learn_machine(seqs, k)
        assert len(merged[0]) <= len(tree[0])
        for word, outputs in obs:
            assert outputs_of(*tree, word) == outputs
            assert outputs_of(*merged, word) == outputs


@pytest.mark.parametrize("seed", range(4))
def test_enough_data_gives_the_true_machine(seed):
    """With many streams the merged machine is the smallest machine of the
    task: same outputs on every input stream, same number of states."""
    rng = np.random.default_rng(100 + seed)
    for _ in range(10):
        k, delta, out = random_machine(rng)
        seqs = [stream(rng, k, delta, out, 14) for _ in range(400)]
        found = M.learn_machine(seqs, k)
        assert M.same_machine(*found, delta, out)
        assert len(found[0]) == minimal(delta, out)


@pytest.mark.parametrize("merging", [True, False])
def test_training_streams_replay_without_a_key(merging):
    """Proposition 8(2): with the state neurons of the tree or of the merged
    machine, training adds no hidden neuron and every training stream is
    reproduced from its first state alone, for sampled weight matrices."""
    rng = np.random.default_rng(5)
    for _ in range(8):
        k, delta, out = random_machine(rng)
        seqs = [stream(rng, k, delta, out, 6) for _ in range(6)]
        lay, _, _ = M.make_layout(seqs, k, merging=merging)
        net = F.make_net(lay)
        F.train(net, lay, seqs)
        assert net.n == lay.n_visible
        for mode in ("lower", "upper", "random"):
            A = dnn.sample_weights(net, rng, mode)
            for s in seqs:
                inputs = [{i for i in range(k) if row[i]} | {lay.on} for row in s]
                run = np.array(dnn.run(A, net.I, F.start(lay, net, s[0]), inputs, len(s) - 1))
                assert (run[:, k] == s[:, k]).all()


def test_parity_is_found_from_the_data():
    """Parity from 32 streams of 13 vectors: two states are found, no hidden
    neuron is added, and streams of 60 vectors are exact."""
    rng = np.random.default_rng(0)

    def parity_stream(length):
        x = rng.integers(0, 2, length)
        return np.c_[x, np.r_[0, 0, np.cumsum(x[:-2]) % 2]]     # y(t+2) = parity of x(0..t)

    seqs = [parity_stream(13) for _ in range(32)]
    lay, delta, out = M.make_layout(seqs, 1)
    assert len(delta) == 2
    net = F.make_net(lay)
    F.train(net, lay, seqs)
    assert net.n == lay.n_visible
    for mode in ("lower", "upper"):
        A = dnn.sample_weights(net, mode=mode)
        for _ in range(30):
            s = parity_stream(60)
            inputs = [({0} if row[0] else set()) | {lay.on} for row in s]
            run = np.array(dnn.run(A, net.I, F.start(lay, net, s[0]), inputs, len(s) - 1))
            assert (run[:, 1] == s[:, 1]).all()


def test_inconsistent_data_are_rejected():
    """Two streams with the same inputs and different outputs are not a
    function of the input history."""
    a = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    b = np.array([[1, 0], [1, 0], [0, 0], [0, 1]])
    with pytest.raises(ValueError):
        M.learn_machine([a, b], 1)
