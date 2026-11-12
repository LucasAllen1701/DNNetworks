"""Experiment 3: capacity of random sequences.

4 visible neurons, no inputs. For each number of sequences N and sequence
length (k + 1 activation vectors, k transitions), N random sequences are
trained either

- "one_round": all N in a single round, or
- "per_sequence": one round per sequence (N rounds).

We record the neurons added against the total number of transitions N*k, and
compare with the Lemma 4 bound 2k(n0 - |I|) per round, i.e. 2k*n0 for one
round and N*2k*n0 summed over N rounds. Every sequence is also checked to
replay from its key (5 weight matrices: both corners and 3 random points).

Run from the repository root:  python experiments/exp3_capacity.py
Writes experiments/results/exp3_capacity.csv.
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
N_SEQS = [1, 2, 4, 8, 16]
K = [2, 4, 8, 16]               # transitions per sequence
SEEDS = range(3)
EPS = 0.5
OUT = os.path.join(ROOT, "experiments", "results", "exp3_capacity.csv")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for k in K:
        for n_seqs in N_SEQS:
            for seed in SEEDS:
                rng = np.random.default_rng(10_000 * k + 100 * n_seqs + seed)
                V = [rng.integers(0, 2, (k + 1, N_VISIBLE)) for _ in range(n_seqs)]
                for mode in ("one_round", "per_sequence"):
                    net = Net(N_VISIBLE, outputs={N_VISIBLE - 1})
                    t0 = time.perf_counter()
                    if mode == "one_round":
                        trained = train(net, V, EPS)
                        rounds = 1
                    else:
                        trained = [s for v in V for s in train(net, [v], EPS)]
                        rounds = n_seqs
                    train_s = time.perf_counter() - t0
                    added = net.n - N_VISIBLE
                    bound = rounds * 2 * k * N_VISIBLE
                    rows.append(dict(
                        mode=mode, k=k, n_seqs=n_seqs, seed=seed,
                        total_transitions=n_seqs * k,
                        neurons_added=added,
                        lemma4_bound=bound,
                        fraction_of_bound=round(added / bound, 4),
                        added_per_transition=round(added / (n_seqs * k), 4),
                        train_seconds=round(train_s, 4),
                        replay_ok=replays(net, trained, rng, trials=5),
                    ))
                    assert added <= bound, rows[-1]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    """Print a Markdown table of means over seeds, both modes side by side."""
    print("| k | N | N*k | one round: added | bound | per sequence: added | bound "
          "| one round: added per transition | per sequence: added per transition "
          "| one round s | per sequence s | replay ok |")
    print("|" + "---|" * 12)
    for k in K:
        for n_seqs in N_SEQS:
            g = {m: [r for r in rows if r["k"] == k and r["n_seqs"] == n_seqs and r["mode"] == m]
                 for m in ("one_round", "per_sequence")}
            m = {mo: {c: np.mean([float(r[c]) for r in g[mo]]) for c in
                      ("neurons_added", "lemma4_bound", "added_per_transition", "train_seconds")}
                 for mo in g}
            ok = all(r["replay_ok"] for mo in g for r in g[mo])
            print(f"| {k} | {n_seqs} | {n_seqs * k} | {m['one_round']['neurons_added']:.3g} "
                  f"| {m['one_round']['lemma4_bound']:.0f} | {m['per_sequence']['neurons_added']:.3g} "
                  f"| {m['per_sequence']['lemma4_bound']:.0f} "
                  f"| {m['one_round']['added_per_transition']:.3g} "
                  f"| {m['per_sequence']['added_per_transition']:.3g} "
                  f"| {m['one_round']['train_seconds']:.3g} | {m['per_sequence']['train_seconds']:.3g} "
                  f"| {ok} |")


if __name__ == "__main__":
    main()
