"""Experiment 7: experiments 1-4 rerun with the causal variant next to the
baseline, each with and without an always-on input neuron.

The variant's candidate search is much slower than the baseline, so the
problem sizes are reduced (see RESULTS.md, section 7):
- exp 1: 3 seeds, 4/16/32 training trajectories (same data as experiment 1);
- exp 2: the no-input configuration, 3 seeds, 10 rounds of 3 sequences;
- exp 3: k in {2, 4, 8}, N in {1, 2, 4, 8}, 2 seeds;
- exp 4: unchanged (all tasks, 8/32/128 streams, 3 seeds).
Configurations run in parallel worker processes; every configuration seeds
its own generator, so results do not depend on scheduling.

Run from the repository root:  python experiments/exp7_causal.py [part ...]
(parts: 1 2 3 4; default all). Writes experiments/results/exp7_causal_exp{1..4}.csv.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp4_input_tasks as e4

VARIANTS = ("baseline", "causal")
BIASES = (False, True)

CA_TRAIN = [4, 16, 32]
CA_SEEDS = range(3)
CL_SEEDS = range(3)
CL_ROUNDS, CL_SEQS, CL_LEN, CL_VISIBLE = 10, 3, 6, 4
CAP_K = [2, 4, 8]
CAP_N = [1, 2, 4, 8]
CAP_SEEDS = range(2)
CAP_VISIBLE = 4


# ---------------------------------------------------------------- part 1

def job_ca(args):
    variant, bias, rule, n, seed = args
    rng = np.random.default_rng(seed)
    return dict(variant=variant, bias=bias, rule=rule, n_train=n, seed=seed,
                **H.run_ca(variant, bias, rule, n, seed, rng))


def part1(pool):
    jobs = [(v, b, rule, n, s) for v in VARIANTS for b in BIASES for rule in (232, 90)
            for n in CA_TRAIN for s in CA_SEEDS]
    rows = pool.map(job_ca, jobs, chunksize=1)
    H.write_csv("exp7_causal_exp1.csv", rows)
    print("### Experiment 1 (cellular automata)")
    cols = ["features", "chain_neurons", "train_seconds", "train_replay_from_key",
            "train_starts_zero_hidden_exact", "unseen_exact", "unseen_first_step_exact",
            "unseen_cell_acc", "all_zero_baseline_cell_acc"]
    table(rows, ["rule", "n_train", "bias", "variant"],
          [(232, n) for n in CA_TRAIN] + [(90, n) for n in CA_TRAIN], cols)


# ---------------------------------------------------------------- part 2

def job_cl(args):
    variant, bias, seed = args
    rng = np.random.default_rng(seed)
    net = H.make_net(variant, CL_VISIBLE, outputs={CL_VISIBLE - 1}, bias=bias)
    keys, seqs, round_of, rows = [], [], [], []
    for r in range(1, CL_ROUNDS + 1):
        V = [rng.integers(0, 2, (CL_LEN, CL_VISIBLE)) for _ in range(CL_SEQS)]
        if bias:
            V = H.with_bias(V)
        f0, c0 = H.hidden_counts(net)
        t0 = time.perf_counter()
        trained = H.train(variant, net, V)
        secs = time.perf_counter() - t0
        f1, c1 = H.hidden_counts(net)
        keys += [k[0] for k in trained]
        seqs += V
        round_of += [r] * len(V)
        ok = [H.recalled(net, k, s, rng) for k, s in zip(keys, seqs)]
        old = [o for o, q in zip(ok, round_of) if q < r]
        first = [o for o, q in zip(ok, round_of) if q == 1]
        rows.append(dict(variant=variant, bias=bias, seed=seed, round=r,
                         features_added=f1 - f0, chain_added=c1 - c0, total_neurons=net.n,
                         train_seconds=round(secs, 4),
                         recall_current_round=float(np.mean(ok[-CL_SEQS:])),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         recall_round1=float(np.mean(first)),
                         recall_all=float(np.mean(ok))))
    return rows


def part2(pool):
    jobs = [(v, b, s) for v in VARIANTS for b in BIASES for s in CL_SEEDS]
    rows = [r for rs in pool.map(job_cl, jobs, chunksize=1) for r in rs]
    H.write_csv("exp7_causal_exp2.csv", rows)
    print("### Experiment 2 (continual learning)")
    cols = ["features_added", "chain_added", "total_neurons", "train_seconds",
            "recall_current_round", "recall_earlier_rounds", "recall_round1"]
    table(rows, ["round", "bias", "variant"], [(r,) for r in range(1, CL_ROUNDS + 1)], cols)


# ---------------------------------------------------------------- part 3

def job_cap(args):
    variant, bias, k, n, seed = args
    rng = np.random.default_rng(10_000 * k + 100 * n + seed)
    V = [rng.integers(0, 2, (k + 1, CAP_VISIBLE)) for _ in range(n)]
    if bias:
        V = H.with_bias(V)
    out = []
    for mode in ("one_round", "per_sequence"):
        net = H.make_net(variant, CAP_VISIBLE, outputs={CAP_VISIBLE - 1}, bias=bias)
        t0 = time.perf_counter()
        if mode == "one_round":
            keys = [s[0] for s in H.train(variant, net, V)]
        else:
            keys = [s[0] for v in V for s in H.train(variant, net, [v])]
        secs = time.perf_counter() - t0
        f, c = H.hidden_counts(net)
        rounds = 1 if mode == "one_round" else n
        out.append(dict(variant=variant, bias=bias, mode=mode, k=k, n_seqs=n, seed=seed,
                        total_transitions=n * k, features=f, chain_neurons=c,
                        lemma4_bound=rounds * 2 * k * CAP_VISIBLE,
                        feature_cap=rounds * 4 * CAP_VISIBLE,
                        train_seconds=round(secs, 4),
                        recall_all=float(np.mean([H.recalled(net, key, s, rng) for key, s in zip(keys, V)]))))
    return out


def part3(pool):
    jobs = [(v, b, k, n, s) for v in VARIANTS for b in BIASES for k in CAP_K for n in CAP_N
            for s in CAP_SEEDS]
    rows = [r for rs in pool.map(job_cap, jobs, chunksize=1) for r in rs]
    H.write_csv("exp7_causal_exp3.csv", rows)
    print("### Experiment 3 (capacity)")
    cols = ["features", "chain_neurons", "lemma4_bound", "train_seconds", "recall_all"]
    table(rows, ["k", "n_seqs", "mode", "bias", "variant"],
          [(k, n) for k in CAP_K for n in CAP_N], cols)


# ---------------------------------------------------------------- part 4

def job_task(args):
    variant, bias, task, n, seed = args
    rng = np.random.default_rng(seed)
    return dict(variant=variant, bias=bias, task=task, n_train=n, seed=seed,
                **H.run_stream_task(variant, bias, task, n, seed, rng))


def part4(pool):
    jobs = [(v, b, t, n, s) for v in VARIANTS for b in BIASES for t in e4.TASKS
            for n in e4.N_TRAIN for s in e4.SEEDS]
    rows = pool.map(job_task, jobs, chunksize=1)
    H.write_csv("exp7_causal_exp4.csv", rows)
    print("### Experiment 4 (input-driven tasks)")
    cols = ["features", "chain_neurons", "train_seconds", "train_exact_from_key",
            "train_acc_zero_hidden", "unseen_acc", "unseen_exact", "unseen_majority_baseline"]
    table(rows, ["task", "n_train", "bias", "variant"],
          [(t, n) for t in e4.TASKS for n in e4.N_TRAIN], cols)


# ---------------------------------------------------------------- output

def table(rows, keys, groups, cols):
    """Markdown table of means over seeds. `groups` fixes the leading keys;
    the remaining keys (mode, bias, variant) are expanded in order."""
    rest = keys[len(groups[0]):]
    values = {k: list(dict.fromkeys(r[k] for r in rows)) for k in rest}
    print("| " + " | ".join(keys + cols) + " |")
    print("|" + "---|" * (len(keys) + len(cols)))
    for g in groups:
        combos = [()]
        for k in rest:
            combos = [c + (v,) for c in combos for v in values[k]]
        for combo in combos:
            sel = [r for r in rows if all(r[k] == v for k, v in zip(keys, g + combo))]
            if sel:
                print("| " + " | ".join(map(str, g + combo)) + " | "
                      + " | ".join(f"{H.mean(sel, c):.3g}" for c in cols) + " |")
    print(flush=True)


def main():
    parts = sys.argv[1:] or ["1", "2", "3", "4"]
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        for p in parts:
            t0 = time.perf_counter()
            {"1": part1, "2": part2, "3": part3, "4": part4}[p](pool)
            print(f"(part {p}: {time.perf_counter() - t0:.0f} s)\n", flush=True)


if __name__ == "__main__":
    main()
