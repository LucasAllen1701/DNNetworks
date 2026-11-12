"""Experiment 2: continual learning over 10 rounds.

A small network (4 visible neurons; one configuration without inputs and one
where neuron 0 is an input) is trained for 10 rounds, each on 3 new random
sequences of 6 activation vectors. After every round, every sequence trained
so far is replayed from its zero-padded key with 20 weight matrices sampled
from the boxes (lower corner, upper corner, 18 random points), using
`replays` from tests/test_dnn.py.

Run from the repository root:  python experiments/exp2_continual_learning.py
Writes experiments/results/exp2_continual_learning.csv.
"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dnn import Net, train  # noqa: E402
from tests.test_dnn import replays  # noqa: E402

N_VISIBLE = 4
ROUNDS = 10
SEQS_PER_ROUND = 3
SEQ_LEN = 6
SEEDS = range(5)
CONFIGS = {"no_input": set(), "one_input": {0}}
EPS = 0.5
OUT = os.path.join(ROOT, "experiments", "results", "exp2_continual_learning.csv")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for config, inputs in CONFIGS.items():
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            net = Net(N_VISIBLE, inputs, {N_VISIBLE - 1})
            trained, round_of = [], []
            for r in range(1, ROUNDS + 1):
                V = [rng.integers(0, 2, (SEQ_LEN, N_VISIBLE)) for _ in range(SEQS_PER_ROUND)]
                before = net.n
                t0 = time.perf_counter()
                new = train(net, V, EPS)
                train_s = time.perf_counter() - t0
                trained += new
                round_of += [r] * len(new)
                ok = [replays(net, [s], rng, trials=20) for s in trained]
                old = [o for o, q in zip(ok, round_of) if q < r]
                rows.append(dict(
                    config=config, seed=seed, round=r,
                    neurons_added=net.n - before, total_neurons=net.n,
                    lemma4_bound=2 * (SEQ_LEN - 1) * (N_VISIBLE - len(inputs)),
                    train_seconds=round(train_s, 4),
                    sequences_so_far=len(trained),
                    recall_all=float(np.mean(ok)),
                    recall_current_round=float(np.mean(ok[-SEQS_PER_ROUND:])),
                    recall_earlier_rounds=float(np.mean(old)) if old else "",
                    space_numbers=net.space(),
                ))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    """Print a Markdown table of means over seeds."""
    cols = ["neurons_added", "total_neurons", "train_seconds", "recall_all",
            "recall_current_round", "recall_earlier_rounds", "space_numbers"]
    print("| config | round | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for config in CONFIGS:
        for r in range(1, ROUNDS + 1):
            g = [x for x in rows if x["config"] == config and x["round"] == r]
            vals = []
            for c in cols:
                v = [float(x[c]) for x in g if x[c] != ""]
                vals.append(f"{np.mean(v):.3g}" if v else "–")
            print(f"| {config} | {r} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
