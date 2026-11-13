"""Experiment 9: fixed neurons inside the network (dnn_fixed.py).

Past-shift neurons, the start clock and twins are fixed-weight visible
neurons present before round 1 (docs/Fixed_Neurons.md). `dnn.py` is unchanged
and trains on the extended sequences. Unlike experiment 8, the shifts are
computed by the network, and they can follow the output neuron as well as the
inputs.

Configurations:
- sources: "inputs" (shifts of the input neurons only; the in-network version
  of experiment 8) or "all" (shifts of the output neuron too);
- extra: "none", "clock" (always-on input and its shifts) or "twins"
  (always-on input and, for every shift neuron, a twin that is 1 exactly when
  the shifted value is 0 and the stream has started).

Part A (one round): the 15 tasks and the data of experiment 8, depths 0-3.
Part B (several rounds): the 32 training streams fed 4 per round for 8 rounds
or 16 per round for 2 rounds, 10 seeds, shifts of all task neurons.
Part C (random sequences): the design of experiment 2 (4 task neurons, 10
rounds of 3 random sequences of 6 vectors) with shifts of all task neurons at
depths 0-3. "Recall without a key" starts a trained sequence from its first
task state with every fixed and hidden neuron at 0.

Run from the repository root:  python experiments/exp9_fixed_neurons.py
Writes experiments/results/exp9_fixed_{one_round,rounds,random}.csv.
"""
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp4_input_tasks as e4
import exp8_past_shift_inputs as e8
import dnn_fixed as F
from tests.test_dnn import replays

dnn = H.dnn

DEPTHS = [1, 2, 3]
CONFIGS = [("inputs", "none"), ("inputs", "clock"),
           ("all", "none"), ("all", "clock"), ("all", "twins")]
N_TRAIN = e4.N_TRAIN
SEEDS = e4.SEEDS

ROUND_CASES = [("tomita2", 1, "none"), ("tomita4", 2, "clock"), ("tomita4", 2, "twins"),
               ("maj3", 2, "none"), ("parity", 2, "none")]
ROUND_STREAMS = 32
SCHEDULES = [(8, 4), (2, 16)]
ROUND_SEEDS = range(10)

RANDOM_INPUTS = {"no_input": set(), "one_input": {0}}
RANDOM_DEPTHS = [0, 1, 2, 3]
RANDOM_SEEDS = range(5)
N_TASK, ROUNDS, SEQS_PER_ROUND, SEQ_LEN = 4, 10, 3, 6


def build(task, n_train, seed, depth, sources, extra):
    n_in, V, unseen = e8.dataset(task, n_train, seed)
    src = None if sources == "all" else range(n_in)
    lay = F.make_layout(n_in + 1, range(n_in), {n_in}, depth, src, extra)
    return lay, F.make_net(lay), V, [F.extend(lay, s) for s in unseen], n_in


def diagnostics(seqs, out):
    """(is the next output a function of the visible state?, is it a threshold
    function of it at margin EPS?) over all the given transitions."""
    table = {}
    for s in seqs:
        for t in range(len(s) - 1):
            table.setdefault(tuple(s[t]), set()).add(int(s[t + 1][out]))
    m = seqs[0].shape[1]
    L, U = np.full((m, m), -np.inf), np.full((m, m), np.inf)
    return all(len(v) == 1 for v in table.values()), bool(dnn.feas(seqs, out, L, U, H.EPS))


# ---- part A

def one_round(args):
    task, n_train, seed, depth, sources, extra = args
    rng = np.random.default_rng(seed)
    lay, net, V, unseen, out = build(task, n_train, seed, depth, sources, extra)
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    secs = time.perf_counter() - t0
    Vx = [s[:, :lay.n_visible] for s in trained]
    tr_ex, tr_acc = H.score(net, [s[0] for s in trained], Vx, [out], rng)
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
    tz_acc = H.score(net, H.zero_hidden_starts(Vx), Vx, [out], rng)[1]
    mid_ex, mid_acc = e8.score_mid(net, unseen, out)
    function, separable = diagnostics(Vx + unseen, out)
    return dict(task=task, n_train=n_train, seed=seed, depth=depth, sources=sources, extra=extra,
                visible_neurons=lay.n_visible, fixed_neurons=len(lay.fixed),
                chain_neurons=net.n - net.n_visible, train_seconds=round(secs, 4),
                train_acc_from_key=tr_acc, train_exact_from_key=tr_ex,
                train_acc_zero_hidden=tz_acc,
                unseen_acc=un_acc, unseen_exact=un_ex,
                unseen_acc_mid=mid_acc, unseen_exact_mid=mid_ex,
                unseen_majority_baseline=H.majority_baseline(unseen, out),
                next_output_function_of_visible=function, visible_separable=separable,
                fixed_boxes_intact=F.fixed_boxes_intact(net, lay))


def part_a(pool):
    jobs = []
    for task in e8.TASKS:
        for n_train in N_TRAIN:
            for seed in SEEDS:
                jobs += [(task, n_train, seed, 0, "all", extra) for extra in ("none", "clock")]
                jobs += [(task, n_train, seed, depth, sources, extra)
                         for depth in DEPTHS for sources, extra in CONFIGS]
    rows = pool.map(one_round, jobs, chunksize=8)
    H.write_csv("exp9_fixed_one_round.csv", rows)
    return rows


# ---- part B

def rounds(args):
    task, depth, extra, n_rounds, per_round, seed = args
    rng = np.random.default_rng(seed)
    lay, net, V, unseen, out = build(task, ROUND_STREAMS, seed, depth, "all", extra)
    rows, trained = [], []
    for r in range(1, n_rounds + 1):
        before = net.n
        trained += F.train(net, lay, V[(r - 1) * per_round: r * per_round], H.EPS)
        ok = [replays(net, [s], rng, trials=20) for s in trained]
        old = ok[:-per_round]
        un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
        rows.append(dict(task=task, depth=depth, extra=extra, rounds=n_rounds, per_round=per_round,
                         seed=seed, round=r, chain_added=net.n - before, total_neurons=net.n,
                         recall_this_round=float(np.mean(ok[-per_round:])),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         unseen_acc=un_acc, unseen_exact=un_ex,
                         unseen_majority_baseline=H.majority_baseline(unseen, out),
                         fixed_boxes_intact=F.fixed_boxes_intact(net, lay),
                         space_numbers=net.space()))
    return rows


def part_b(pool):
    jobs = [(task, depth, extra, n_rounds, per_round, seed)
            for task, depth, extra in ROUND_CASES
            for n_rounds, per_round in SCHEDULES for seed in ROUND_SEEDS]
    rows = [r for group in pool.map(rounds, jobs) for r in group]
    H.write_csv("exp9_fixed_rounds.csv", rows)
    return rows


# ---- part C

def random_rounds(args):
    config, depth, seed = args
    data = np.random.default_rng(seed).integers(0, 2, (ROUNDS, SEQS_PER_ROUND, SEQ_LEN, N_TASK))
    rng = np.random.default_rng(1000 + seed)
    lay = F.make_layout(N_TASK, RANDOM_INPUTS[config], {N_TASK - 1}, depth)
    net = F.make_net(lay)
    nv = lay.n_visible
    rows, trained = [], []
    for r in range(1, ROUNDS + 1):
        before = net.n
        t0 = time.perf_counter()
        trained += F.train(net, lay, list(data[r - 1]), H.EPS)
        secs = time.perf_counter() - t0
        ok = [replays(net, [s], rng, trials=20) for s in trained]
        old = ok[:-SEQS_PER_ROUND]
        no_key = [H.recalled(net, s[0, :nv], s[:, :nv], rng) for s in trained]
        rows.append(dict(config=config, depth=depth, seed=seed, round=r,
                         fixed_neurons=len(lay.fixed), chain_added=net.n - before,
                         total_neurons=net.n, train_seconds=round(secs, 4),
                         recall_all=float(np.mean(ok)),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         recall_without_key=float(np.mean(no_key)),
                         fixed_boxes_intact=F.fixed_boxes_intact(net, lay),
                         space_numbers=net.space()))
    return rows


def part_c(pool):
    jobs = [(config, depth, seed) for config in RANDOM_INPUTS
            for depth in RANDOM_DEPTHS for seed in RANDOM_SEEDS]
    rows = [r for group in pool.map(random_rounds, jobs) for r in group]
    H.write_csv("exp9_fixed_random.csv", rows)
    return rows


# ---- tables

def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _cells(rows, col, fmt, **kw):
    return " / ".join(fmt(H.mean(_sel(rows, depth=d, **kw), col)) for d in [0] + DEPTHS)


def _f(x):
    return f"{x:.3g}" if x in (0, 1) else f"{x:.3f}"


def summarise_a(rows, n_train=128):
    print(f"\nPart A, {n_train} training streams, means over seeds; cells are depths 0 / 1 / 2 / 3\n")
    zero = {"none": "none", "clock": "clock", "twins": "clock"}   # depth 0 has no fixed neurons
    for col, title, fmt in (("chain_neurons", "chain neurons", lambda x: f"{x:.3g}"),
                            ("unseen_exact", "unseen exact", _f)):
        print(f"{title}\n")
        print("| task | " + " | ".join(f"{s}, {e}" for s, e in CONFIGS) + " |")
        print("|" + "---|" * (len(CONFIGS) + 1))
        for task in e8.TASKS:
            cells = []
            for sources, extra in CONFIGS:
                g = [r for r in _sel(rows, task=task, n_train=n_train)
                     if (r["depth"] == 0 and r["extra"] == zero[extra])
                     or (r["depth"] > 0 and r["sources"] == sources and r["extra"] == extra)]
                cells.append(_cells(g, col, fmt))
            print(f"| {task} | " + " | ".join(cells) + " |")
        print()
    sep = [r for r in rows if r["visible_separable"]]
    non = [r for r in rows if not r["visible_separable"]]
    print(f"runs: {len(rows)}; threshold function of the visible state: {len(sep)}, of which "
          f"{sum(r['chain_neurons'] > 0 for r in sep)} added chain neurons")
    for n in N_TRAIN:
        g = _sel(sep, n_train=n)
        print(f"  {n} streams: {sum(r['unseen_exact'] == 1.0 for r in g)} of {len(g)} exact on every unseen stream")
    print(f"not a threshold function: {len(non)}, of which "
          f"{sum(r['unseen_exact'] == 1.0 for r in non)} exact on every unseen stream")
    print("training exact from key, min:", min(r["train_exact_from_key"] for r in rows),
          "; fixed boxes intact in every run:", all(r["fixed_boxes_intact"] for r in rows))


def summarise_b(rows):
    n = len(ROUND_SEEDS)
    print(f"\nPart B, {ROUND_STREAMS} training streams, shifts of all task neurons, {n} seeds\n")
    print("| task | depth | extra | rounds x streams | chain: round 1 | runs that add chains late "
          "| neurons at end | recall, earlier rounds (min) | unseen exact: round 1 "
          "| unseen exact: last round | runs exact on all unseen streams at the end |")
    print("|" + "---|" * 11)
    for task, depth, extra in ROUND_CASES:
        for n_rounds, per_round in SCHEDULES:
            g = _sel(rows, task=task, extra=extra, rounds=n_rounds)
            first, last = _sel(g, round=1), _sel(g, round=n_rounds)
            late = {r["seed"] for r in g if r["round"] > 1 and r["chain_added"] > 0}
            print(f"| {task} | {depth} | {extra} | {n_rounds} x {per_round} "
                  f"| {H.mean(first, 'chain_added'):.3g} | {len(late)} of {n} "
                  f"| {H.mean(last, 'total_neurons'):.3g} "
                  f"| {min(float(r['recall_earlier_rounds']) for r in g if r['round'] > 1):.3g} "
                  f"| {H.mean(first, 'unseen_exact'):.3f} | {H.mean(last, 'unseen_exact'):.3f} "
                  f"| {sum(r['unseen_exact'] == 1.0 for r in last)} of {n} |")
    old = [float(r["recall_earlier_rounds"]) for r in rows if r["recall_earlier_rounds"] != ""]
    print(f"\nPart B, recall: min over {len(old)} checks of earlier rounds = {min(old)}, "
          f"min over {len(rows)} checks of the current round = {min(r['recall_this_round'] for r in rows)}")


def summarise_c(rows):
    print(f"\nPart C, random sequences, means over {len(RANDOM_SEEDS)} seeds\n")
    print("| inputs | depth | fixed neurons | chain neurons added per round | chain neurons after 10 rounds "
          "| total neurons after 10 rounds | stored numbers | recall from keys, all rounds (min) "
          "| recall without a key after 10 rounds | train s, round 10 |")
    print("|" + "---|" * 10)
    for config in RANDOM_INPUTS:
        for depth in RANDOM_DEPTHS:
            g = _sel(rows, config=config, depth=depth)
            last = _sel(g, round=ROUNDS)
            chain = [r["total_neurons"] - N_TASK - r["fixed_neurons"] for r in last]
            print(f"| {config} | {depth} | {last[0]['fixed_neurons']} | {H.mean(g, 'chain_added'):.3g} "
                  f"| {np.mean(chain):.3g} | {H.mean(last, 'total_neurons'):.3g} "
                  f"| {H.mean(last, 'space_numbers'):.0f} | {min(r['recall_all'] for r in g):.3g} "
                  f"| {H.mean(last, 'recall_without_key'):.3f} | {H.mean(last, 'train_seconds'):.3f} |")
    print("\nPart C, fixed boxes intact in every round:", all(r["fixed_boxes_intact"] for r in rows))


def main():
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        a = part_a(pool)
        b = part_b(pool)
        c = part_c(pool)
    summarise_a(a)
    summarise_b(b)
    summarise_c(c)


if __name__ == "__main__":
    main()
