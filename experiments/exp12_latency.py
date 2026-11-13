"""Experiment 12: how many steps should the output be given?

The stream tasks define an output for every input prefix. A training
sequence has to say at which step that output appears: h steps after the
last input it depends on. Experiments 4-9 used h = 1 and experiment 10 added
h = 2 ("latency 1"). Here h runs from 1 to 4 for the two layouts of
experiment 10 that go beyond shifts:

- library: shifts of depth 2 and the feature library (the same for every task);
- states:  the state neurons of the task's own machine.

The data are those of experiment 8 with the output column moved h - 1 steps
later (the first output value is repeated). 128 training streams, 3 seeds,
`dnn.py` unchanged.

Run from the repository root:  python experiments/exp12_latency.py
Writes experiments/results/exp12_latency.csv.
"""
import os
from multiprocessing import Pool

import numpy as np

import harness as H
import exp4_input_tasks as e4
import exp8_past_shift_inputs as e8
import exp10_library_and_states as e10
import dnn_fixed as F

H_VALUES = [1, 2, 3, 4]
LAYOUTS = ["library", "states"]
N_TRAIN = 128
SEEDS = e4.SEEDS


def one_run(args):
    task, h, kind, seed = args
    rng = np.random.default_rng(seed)
    out, V, unseen = e8.dataset(task, N_TRAIN, seed)
    for _ in range(h - 1):
        V, unseen = [e10.delayed(s, out) for s in V], [e10.delayed(s, out) for s in unseen]
    lay = e10.make_layout(kind, task)
    net = F.make_net(lay)
    F.train(net, lay, V, H.EPS)
    unseen = [F.extend(lay, s) for s in unseen]
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
    return dict(task=task, h=h, layout=kind, seed=seed, n_train=N_TRAIN,
                chain_neurons=net.n - lay.n_visible, unseen_exact=un_ex, unseen_acc=un_acc,
                unseen_exact_mid=e8.score_mid(net, unseen, out)[0],
                unseen_majority_baseline=H.majority_baseline(unseen, out))


def summarise(rows):
    print(f"\n{N_TRAIN} training streams, {len(SEEDS)} seeds. 'yes': no hidden neuron in any seed "
          "(fraction of unseen streams exact with sampled weights, if below 1)\n")
    head = [f"{kind}, h = {h}" for kind in LAYOUTS for h in H_VALUES]
    print("| task | " + " | ".join(head) + " |")
    print("|" + "---|" * (len(head) + 1))
    solved = {(kind, h): 0 for kind in LAYOUTS for h in H_VALUES}
    for task in e8.TASKS:
        cells = []
        for kind in LAYOUTS:
            for h in H_VALUES:
                g = [r for r in rows if r["task"] == task and r["layout"] == kind and r["h"] == h]
                if all(r["chain_neurons"] == 0 for r in g):
                    solved[kind, h] += 1
                    ex = H.mean(g, "unseen_exact")
                    cells.append("yes" if ex == 1 else f"yes ({ex:.3f})")
                else:
                    cells.append("–")
        print(f"| {task} | " + " | ".join(cells) + " |")
    print("| tasks with no hidden neuron, of 15 | " + " | ".join(str(solved[k, h]) for k in LAYOUTS for h in H_VALUES) + " |")
    ok = [r for r in rows if r["chain_neurons"] == 0]
    print(f"\nruns with no hidden neuron: {len(ok)} of {len(rows)}; of these, exact on every unseen stream "
          f"with midpoint weights: {sum(r['unseen_exact_mid'] == 1.0 for r in ok)}")
    bad = [r for r in rows if r["chain_neurons"] > 0]
    print(f"runs with hidden neurons: {len(bad)}; of these, exact on every unseen stream: "
          f"{sum(r['unseen_exact'] == 1.0 for r in bad)}")


def main():
    jobs = [(task, h, kind, seed) for task in e8.TASKS for h in H_VALUES for kind in LAYOUTS for seed in SEEDS]
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        rows = pool.map(one_run, jobs, chunksize=4)
    H.write_csv("exp12_latency.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
