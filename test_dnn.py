"""Tests for dnn.py. Run with `python -m pytest` from the repository root."""
import numpy as np
import pytest

from dnn import Net, run, sample_weights, train


def replays(net, sequences, rng, trials=20):
    """True if every sequence replays exactly from its key, for the two
    extreme corners of the boxes and for random weight matrices."""
    for s in sequences:
        s = np.hstack([s, np.zeros((len(s), net.n - s.shape[1]), int)])  # zero-pad later cohorts
        input_sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
        for k in range(trials):
            mode = ["lower", "upper"][k] if k < 2 else "random"
            A = sample_weights(net, rng, mode)
            out = run(A, net.I, s[0], input_sets, len(s) - 1)
            if not all((o == x).all() for o, x in zip(out, s)):
                return False
    return True


def test_chain_example():
    """Notes, Example 1: the sequence 1,1,1,0 needs a chain of two predictors."""
    net = Net(1, outputs={0})
    trained = train(net, [np.array([[1], [1], [1], [0]])])
    assert net.n == 3
    assert list(trained[0][0]) == [1, 1, 1]
    assert replays(net, trained, np.random.default_rng(0))


def test_anti_predictor_example():
    """Notes, Example 2: 1,1 then 1,0 needs one anti-predictor."""
    net = Net(1, outputs={0})
    first = train(net, [np.array([[1], [1]])])
    assert net.n == 1
    second = train(net, [np.array([[1], [0]])])
    assert net.n == 2 and net.c == [0, 2]
    assert list(second[0][0]) == [1, 1]
    assert replays(net, first + second, np.random.default_rng(0))


def test_feasible_sequence_adds_no_neurons():
    """Training a sequence that fits the visible neurons twice changes nothing in size."""
    net = Net(2, outputs={1})
    s = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    train(net, [s])
    train(net, [s])
    assert net.n == 2


@pytest.mark.parametrize("seed", range(6))
def test_random_multi_round_replay(seed):
    """Learning and no forgetting (the Theorem): after every round, every
    sequence trained so far replays for all sampled weight matrices."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        n0 = int(rng.integers(1, 5))
        inputs = {0} if (n0 > 1 and rng.random() < 0.5) else set()
        net = Net(n0, inputs, {n0 - 1})
        eps = float(rng.choice([0.1, 0.5, 1.0]))
        trained = []
        for _ in range(int(rng.integers(1, 5))):
            V = [rng.integers(0, 2, (int(rng.integers(2, 7)), n0))
                 for _ in range(int(rng.integers(1, 4)))]
            if trained and rng.random() < 0.3:
                # same start as an old sequence, different continuation
                old = trained[int(rng.integers(len(trained)))][:, :n0]
                s = rng.integers(0, 2, (len(old), n0))
                s[0] = old[0]
                V.append(s)
            trained += train(net, V, eps)
            assert replays(net, trained, rng)


def test_neuron_bound_and_dormancy():
    """Lemma 4 (at most 2k(n0 - |I|) new neurons per round) and condition (D)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        n0 = int(rng.integers(1, 5))
        k = int(rng.integers(1, 7))
        net = Net(n0, outputs={0})
        for _ in range(3):
            before = net.n
            train(net, [rng.integers(0, 2, (k + 1, n0)) for _ in range(2)])
            assert net.n - before <= 2 * k * n0
        for i in range(net.n):
            for j in range(net.n):
                if net.c[i] != 0 and net.c[j] != net.c[i]:
                    assert net.U[i, j] <= 0


def test_space_is_quadratic_in_neurons():
    net = Net(3, {0}, {2})
    train(net, [np.random.default_rng(2).integers(0, 2, (6, 3))])
    assert net.L.shape == net.U.shape == (net.n, net.n)
    assert net.space() == 2 * net.n ** 2 + net.n
