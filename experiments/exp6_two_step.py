"""Experiment 6: tasks given one extra network step.

(a) Two-step parity: one input x, one output y, y(t+2) = parity of x(0..t),
    y(0) = y(1) = 0. First a hand-built network with two feature neurons,
        h1 = s(x - h1 - h2),  h2 = s(h1 + h2 - x),  y = s(h1 + h2),
    is checked on random streams to show that the task is realisable. Then
    the baseline and the causal variant are trained on random streams.
(b) Rule 90 on a ring of 6 cells, one visible neuron per cell, no inputs.
    Each automaton step takes two network steps and the cells are 0 on odd
    steps: c(0), 0, c(1), 0, ..., c(K).

Unseen data is run from its true visible state at time 0 with every hidden
neuron at 0.

Run from the repository root:  python experiments/exp6_two_step.py
Writes experiments/results/exp6_two_step.csv.
"""
import time

import numpy as np

import harness as H
from dnn_causal import hand_built
from exp1_cellular_automata import ca_step

T = 12                          # network steps per parity stream
PARITY_TRAIN = [8, 32, 128]
N_UNSEEN = 100
CELLS = 6
K = 5                           # automaton steps per trajectory (2K network steps)
CA_TRAIN = [2, 4, 8, 16, 32]
SEEDS = range(3)
VARIANTS = ("baseline", "causal")


def parity2(x):
    return np.r_[0, 0, np.cumsum(x[:-2]) % 2]


def parity_streams(rng, count, exclude=()):
    seen = {s.tobytes() for s in exclude}
    out = []
    while len(out) < count:
        x = rng.integers(0, 2, T + 1)
        if x.tobytes() not in seen:
            seen.add(x.tobytes())
            out.append(x)
    return out


def two_step_trajectory(x0, rule=90):
    rows, x = [np.array(x0, int)], np.array(x0, int)
    for _ in range(K):
        x = ca_step(x, rule)
        rows += [np.zeros(CELLS, int), x]
    return np.array(rows)


def all_states():
    return [np.array([(k >> i) & 1 for i in range(CELLS)]) for k in range(2 ** CELLS)]


def describe_features(net):
    info = getattr(net, "feature_info", {})
    return "; ".join(f"{i}:{t}({a},{b})" if b is not None else f"{i}:{t}({a})"
                     for i, (t, a, b) in sorted(info.items()))


def main():
    rows = []
    # (a) hand-built two-step parity
    rng = np.random.default_rng(0)
    hand = hand_built(2, {0}, {1}, features=[[1, 0, -1, -1], [-1, 0, 1, 1]],
                      visible_rows={1: [0, 0, 1, 1]})
    test = [np.c_[x, parity2(x)] for x in parity_streams(rng, 1000)]
    ex, acc = H.score(hand, H.zero_hidden_starts(test), test, [1], rng, modes=["lower"])
    rows.append(dict(task="parity2", variant="hand_built", n_train=0, seed=0, features=2,
                     chain_neurons=0, unseen_exact=ex, unseen_acc=acc))
    print(f"hand-built two-step parity: exact on {ex:.3f} of 1000 random streams, accuracy {acc:.3f}")

    # (a) trained
    for variant in VARIANTS:
        for n_train in PARITY_TRAIN:
            for seed in SEEDS:
                rng = np.random.default_rng(100 * seed + n_train)
                xs = parity_streams(rng, n_train)
                V = [np.c_[x, parity2(x)] for x in xs]
                unseen = [np.c_[x, parity2(x)] for x in parity_streams(rng, N_UNSEEN, exclude=xs)]
                net = H.make_net(variant, 2, inputs={0}, outputs={1})
                t0 = time.perf_counter()
                trained = H.train(variant, net, V)
                secs = time.perf_counter() - t0
                f, c = H.hidden_counts(net)
                un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [1], rng)
                rows.append(dict(task="parity2", variant=variant, n_train=n_train, seed=seed,
                                 features=f, chain_neurons=c, train_seconds=round(secs, 4),
                                 train_exact_from_key=H.score(net, [k[0] for k in trained], V, [1], rng)[0],
                                 unseen_exact=un_ex, unseen_acc=un_acc,
                                 majority_baseline=H.majority_baseline(unseen, 1),
                                 feature_list=describe_features(net)))

    # (b) two-step rule 90 on a ring of 6
    states = all_states()
    for variant in VARIANTS:
        for n_train in CA_TRAIN:
            for seed in SEEDS:
                rng = np.random.default_rng(100 * seed + n_train)
                pick = rng.permutation(len(states))
                train_states = [states[i] for i in pick[:n_train]]
                unseen_states = [states[i] for i in pick[n_train:]]
                V = [two_step_trajectory(x) for x in train_states]
                unseen = [two_step_trajectory(x) for x in unseen_states]
                net = H.make_net(variant, CELLS)
                t0 = time.perf_counter()
                trained = H.train(variant, net, V)
                secs = time.perf_counter() - t0
                f, c = H.hidden_counts(net)
                cells = list(range(CELLS))
                un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, cells, rng)
                rows.append(dict(task="rule90_two_step", variant=variant, n_train=n_train, seed=seed,
                                 features=f, chain_neurons=c, train_seconds=round(secs, 4),
                                 train_exact_from_key=H.score(net, [k[0] for k in trained], V, cells, rng)[0],
                                 unseen_exact=un_ex, unseen_acc=un_acc,
                                 majority_baseline=float(np.mean([(s[1:] == 0).mean() for s in unseen])),
                                 n_unseen=len(unseen), feature_list=describe_features(net)))
    H.write_csv("exp6_two_step.csv", rows)
    summarise(rows)


def summarise(rows):
    cols = ["features", "chain_neurons", "train_seconds", "train_exact_from_key",
            "unseen_exact", "unseen_acc", "majority_baseline"]
    print("| task | variant | n_train | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 3))
    for task, sizes in (("parity2", PARITY_TRAIN), ("rule90_two_step", CA_TRAIN)):
        for variant in VARIANTS:
            for n in sizes:
                g = [r for r in rows if r["task"] == task and r["variant"] == variant and r["n_train"] == n]
                print(f"| {task} | {variant} | {n} | " + " | ".join(f"{H.mean(g, c):.3g}" for c in cols) + " |")


if __name__ == "__main__":
    main()
