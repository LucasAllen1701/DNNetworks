"""Experiment 1: elementary cellular automata on a ring of 8 cells.

One visible neuron per cell, no input neurons. Rule 232 (majority of
left, self, right) is linearly separable without a bias; rule 90 (XOR of
the two neighbours) is not. For each rule and each number of training
trajectories we train one round, then:

- check that every training trajectory replays from its key (the Theorem);
- run trajectories from unseen starting states with every hidden neuron at
  0, and compare the visible cells with the true automaton;
- run the training starting states the same way (hidden neurons at 0), to
  separate what is stored in the keys from what is in the visible weights.

Run from the repository root:  python experiments/exp1_cellular_automata.py
Writes experiments/results/exp1_cellular_automata.csv.
"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dnn import Net, run, sample_weights, train  # noqa: E402
from tests.test_dnn import replays  # noqa: E402

N_CELLS = 8
STEPS = 10                      # transitions per trajectory
N_TRAIN = [1, 2, 4, 8, 16, 32]
SEEDS = range(5)
N_UNSEEN = 100
WEIGHT_MODES = ["lower", "upper", "random", "random", "random"]
EPS = 0.5
OUT = os.path.join(ROOT, "experiments", "results", "exp1_cellular_automata.csv")


def ca_step(x, rule):
    left, right = np.roll(x, 1), np.roll(x, -1)
    return (rule >> (4 * left + 2 * x + right)) & 1


def trajectory(x0, rule, steps=STEPS):
    s = [np.array(x0, int)]
    for _ in range(steps):
        s.append(ca_step(s[-1], rule))
    return np.array(s)


def random_starts(rng, count, exclude=()):
    seen = {tuple(x) for x in exclude}
    out = []
    while len(out) < count:
        x = rng.integers(0, 2, N_CELLS)
        if tuple(x) not in seen:
            seen.add(tuple(x))
            out.append(x)
    return out


def score_from_zero_hidden(net, starts, rule, rng):
    """Run each start with hidden neurons at 0. Returns (fraction of
    trajectories reproduced exactly, fraction of first steps correct,
    fraction of cell-steps correct), averaged over the weight samples."""
    exact, first, cells = [], [], []
    for mode in WEIGHT_MODES:
        A = sample_weights(net, rng, mode)
        for x0 in starts:
            truth = trajectory(x0, rule)
            v0 = np.r_[x0, np.zeros(net.n - N_CELLS, int)]
            out = np.array(run(A, net.I, v0, [set()] * (STEPS + 1), STEPS))
            assert not out[:, N_CELLS:].any(), "hidden neuron fired from a zero cohort"
            pred = out[:, :N_CELLS]
            exact.append(bool((pred == truth).all()))
            first.append(bool((pred[1] == truth[1]).all()))
            cells.append(float((pred[1:] == truth[1:]).mean()))
    return float(np.mean(exact)), float(np.mean(first)), float(np.mean(cells))


def zero_baseline(starts, rule):
    """Cell accuracy of predicting every cell as 0 at every step."""
    return float(np.mean([(trajectory(x, rule)[1:] == 0).mean() for x in starts]))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for rule in (232, 90):
        for n_train in N_TRAIN:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * rule + 100 * n_train + seed)
                starts = random_starts(rng, n_train)
                unseen = random_starts(rng, N_UNSEEN, exclude=starts)
                net = Net(N_CELLS)
                t0 = time.perf_counter()
                trained = train(net, [trajectory(x, rule) for x in starts], EPS)
                train_s = time.perf_counter() - t0
                rows.append(dict(
                    rule=rule, n_train=n_train, seed=seed,
                    hidden_added=net.n - N_CELLS,
                    lemma4_bound=2 * STEPS * N_CELLS,
                    train_seconds=round(train_s, 4),
                    train_replay_from_key=replays(net, trained, rng, trials=5),
                    train_starts_zero_hidden_exact=score_from_zero_hidden(net, starts, rule, rng)[0],
                    unseen_exact=(u := score_from_zero_hidden(net, unseen, rule, rng))[0],
                    unseen_first_step_exact=u[1],
                    unseen_cell_acc=u[2],
                    all_zero_baseline_cell_acc=zero_baseline(unseen, rule),
                ))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    """Print a Markdown table of means over seeds."""
    cols = ["hidden_added", "train_seconds", "train_replay_from_key",
            "train_starts_zero_hidden_exact", "unseen_exact",
            "unseen_first_step_exact", "unseen_cell_acc", "all_zero_baseline_cell_acc"]
    print("| rule | n_train | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for rule in (232, 90):
        for n_train in N_TRAIN:
            g = [r for r in rows if r["rule"] == rule and r["n_train"] == n_train]
            vals = [np.mean([float(r[c]) for r in g]) for c in cols]
            print(f"| {rule} | {n_train} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")


if __name__ == "__main__":
    main()
