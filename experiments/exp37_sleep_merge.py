"""Experiment 37: wake with clock neurons, then sleep by merging neurons (Lucas's merge, 27 November).

Phase 1 (wake, notes of 26 November, section 4.4): one round on the training
streams at epsilon = 1e-4; if a neuron does not fit, one clock neuron per
vector of each stream (dnn_clock.wake_clock). Phase 2 (sleep, Lucas's idea of
27 November): repeatedly merge the two hidden neurons whose boxes need the
least stretching to meet, while that stretch is at most delta
(dnn_clock.sleep_merge). Incoming boxes meet; outgoing weights add (the two
neurons fire together); key values must agree. No linear program is solved.

Tasks from experiment 4 (12 transitions per stream, output y(t+1) from inputs
up to t): prev_input (needs no memory: a visible neuron can do it), parity
and adder (need memory). 8 training streams, 100 unseen, 5 seeds. For each
delta: hidden neurons left, merges, trained streams recalled exactly from
their keys (5 weight samples), trained and unseen streams run without keys
(hidden neurons at 0): whole streams with every output right, and fraction
of outputs right. Question: does merging shrink the network, and does it
make the network right on streams it never saw?
"""
import copy
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from dnn import Net  # noqa: E402
from dnn_clock import sleep_merge, wake_clock  # noqa: E402

TASKS = ["prev_input", "parity", "adder"]
N_TRAIN = 8
SEEDS = range(5)
DELTAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
EPS = 1e-4


def row(task, seed, delta, net, keys, train, unseen, n_in, merges, hidden_before):
    rng = np.random.default_rng(seed)
    out = [n_in]
    recall = np.mean([H.recalled(net, k, s, rng, trials=5) for k, s in zip(keys, train)])
    tr_exact, tr_acc = H.score(net, H.zero_hidden_starts(train), train, out, rng)
    un_exact, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, out, rng)
    return {"task": task, "seed": seed, "delta": delta, "hidden_before": hidden_before,
            "hidden_after": net.n - net.n_visible, "merges": merges,
            "recall_keys": round(float(recall), 4),
            "train_nokey_exact": round(tr_exact, 4), "train_nokey_acc": round(tr_acc, 4),
            "unseen_exact": round(un_exact, 4), "unseen_acc": round(un_acc, 4),
            "majority_acc": round(H.majority_baseline(unseen, n_in), 4)}


def job(args):
    task, seed, deltas = args
    n_in, train, unseen = H.stream_dataset(task, N_TRAIN, seed)
    net = Net(n_in + 1, inputs=set(range(n_in)), outputs={n_in})
    ext = wake_clock(net, train, EPS)
    keys = [e[0] for e in ext]
    hb = net.n - net.n_visible
    rows = [row(task, seed, "wake", net, keys, train, unseen, n_in, 0, hb)]
    for d in deltas:
        m = copy.deepcopy(net)
        K, merges = sleep_merge(m, keys, d)
        rows.append(row(task, seed, d, m, K, train, unseen, n_in, merges, hb))
    return rows


def main(tasks=tuple(TASKS), seeds=SEEDS, deltas=DELTAS):
    jobs = [(t, s, deltas) for t in tasks for s in seeds]
    with Pool() as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    H.write_csv("exp37_sleep_merge.csv", rows)
    return rows


if __name__ == "__main__":
    main()
