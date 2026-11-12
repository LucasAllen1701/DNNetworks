"""Experiment 5: the baseline (dnn.py) with an always-on input neuron.

Experiments 1 (cellular automata) and 4 (input-driven tasks) are rerun on
exactly the same data, with and without an extra input neuron that is 1 at
every step. That neuron gives every threshold unit a bias weight.

Run from the repository root:  python experiments/exp5_bias.py
Writes experiments/results/exp5_bias.csv.
"""
import numpy as np

import harness as H
import exp1_cellular_automata as e1
import exp4_input_tasks as e4


def main():
    rows = []
    for bias in (False, True):
        for rule in (232, 90):
            for n_train in e1.N_TRAIN:
                for seed in e1.SEEDS:
                    rng = np.random.default_rng(seed)
                    rows.append(dict(experiment="1_ca", task=f"rule{rule}", n_train=n_train, seed=seed,
                                     bias=bias, **H.run_ca("baseline", bias, rule, n_train, seed, rng)))
        for task in e4.TASKS:
            for n_train in e4.N_TRAIN:
                for seed in e4.SEEDS:
                    rng = np.random.default_rng(seed)
                    rows.append(dict(experiment="4_inputs", task=task, n_train=n_train, seed=seed,
                                     bias=bias, **H.run_stream_task("baseline", bias, task, n_train, seed, rng)))
    H.write_csv("exp5_bias.csv", rows)
    summarise(rows)


def summarise(rows):
    def g(exp, task, n, bias):
        return [r for r in rows if r["experiment"] == exp and r["task"] == task
                and r["n_train"] == n and r["bias"] == bias]
    print("| task | n_train | hidden: no bias | hidden: bias | unseen exact: no bias | unseen exact: bias "
          "| unseen cell acc: no bias | unseen cell acc: bias | all-zero baseline |")
    print("|" + "---|" * 9)
    for rule in (232, 90):
        for n in e1.N_TRAIN:
            a, b = g("1_ca", f"rule{rule}", n, False), g("1_ca", f"rule{rule}", n, True)
            print(f"| rule {rule} | {n} | {H.mean(a, 'chain_neurons'):.3g} | {H.mean(b, 'chain_neurons'):.3g} "
                  f"| {H.mean(a, 'unseen_exact'):.3g} | {H.mean(b, 'unseen_exact'):.3g} "
                  f"| {H.mean(a, 'unseen_cell_acc'):.3g} | {H.mean(b, 'unseen_cell_acc'):.3g} "
                  f"| {H.mean(a, 'all_zero_baseline_cell_acc'):.3g} |")
    print()
    print("| task | n_train | hidden: no bias | hidden: bias | train acc from key: bias "
          "| unseen acc: no bias | unseen acc: bias | unseen exact: bias | majority baseline |")
    print("|" + "---|" * 9)
    for task in e4.TASKS:
        for n in e4.N_TRAIN:
            a, b = g("4_inputs", task, n, False), g("4_inputs", task, n, True)
            print(f"| {task} | {n} | {H.mean(a, 'chain_neurons'):.3g} | {H.mean(b, 'chain_neurons'):.3g} "
                  f"| {H.mean(b, 'train_acc_from_key'):.3g} "
                  f"| {H.mean(a, 'unseen_acc'):.3g} | {H.mean(b, 'unseen_acc'):.3g} "
                  f"| {H.mean(b, 'unseen_exact'):.3g} | {H.mean(a, 'unseen_majority_baseline'):.3g} |")


if __name__ == "__main__":
    main()
