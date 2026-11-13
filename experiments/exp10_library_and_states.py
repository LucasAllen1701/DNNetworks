"""Experiment 10: a feature library and state neurons, installed before round 1.

Everything here is a fixed neuron in the sense of docs/Fixed_Neurons.md, so
the Theorem holds and `dnn.py` trains unchanged. The propositions tested are
in docs/Latency_And_State.md.

Latency. The stream tasks of experiment 8 ask for y(t+1) = out(x(0..t)): the
output must react to the input of the same step. "Latency 1" gives the
network one more step: the same outputs, one step later (y(0) is unchanged).

Layouts, all with the always-on input:
- shifts:  past shifts of depth 2 of every task neuron (the reference);
- library: the same, plus every feature template of the causal variant (copy,
  latch, AND, OR, AND-NOT, set-reset) over all those neurons and pairs of
  them. The library is the same for every task;
- states:  the state neurons of the task's own finite machine (Minsky's
  construction). This layout is built from the task's definition, so it shows
  what can be represented, not what is found.

Part A: the 15 tasks, latency 0 and 1, the three layouts, one round.
Part B: two-step rule 90 on a ring of 6 (the task and data of experiment 6)
        with the library over the cells.
Part C: several rounds (32 streams, 4 per round for 8 rounds or 16 per round
        for 2 rounds, 10 seeds), with every stream trained so far replayed
        from its key with 20 weight matrices after every round.
Part D: networks trained on 128 streams of 13 vectors, tested on 200 streams
        of 101 vectors, to see whether what was learned is the machine or
        something that only works for short streams.

Run from the repository root:  python experiments/exp10_library_and_states.py
Writes experiments/results/exp10_{one_round,rule90,rounds,long_streams}.csv.
"""
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp4_input_tasks as e4
import exp6_two_step as e6
import exp8_past_shift_inputs as e8
import dnn_fixed as F
from exp9_fixed_neurons import diagnostics
from tests.test_dnn import replays

dnn = H.dnn

LAYOUTS = ["shifts", "library", "states"]
N_TRAIN = e4.N_TRAIN
SEEDS = e4.SEEDS
DEPTH = 2

ROUND_CASES = [("parity", 1, "library"), ("parity", 1, "states"), ("xor2", 1, "library"),
               ("tomita7", 0, "states"), ("tomita4", 0, "library")]
ROUND_STREAMS = 32
SCHEDULES = [(8, 4), (2, 16)]
ROUND_SEEDS = range(10)

LONG_CASES = ([(t, 1, "library") for t in ("parity", "xor2", "tomita2", "tomita4", "maj3", "no000")]
              + [(t, 1, "states") for t in ("parity", "adder", "tomita3", "tomita5", "tomita6")]
              + [("tomita7", 0, "states")])
LONG_STEPS, LONG_STREAMS = 100, 200

_machines = {}


def machine(task):
    if task not in _machines:
        n_in, f = e8.TASKS[task]
        _machines[task] = F.machine_from_task(f, n_in)
    return _machines[task]


def delayed(s, out):
    """The same sequence with the output column one step later."""
    s = np.array(s, int)
    s[1:, out] = s[:-1, out].copy()
    return s


def make_layout(kind, task):
    n_in = e8.TASKS[task][0]
    lay = F.Layout(n_in + 1, range(n_in), {n_in})
    lay.add_on()
    if kind == "states":
        F.add_automaton(lay, machine(task)[0])
        return lay
    prev = {i: i for i in range(n_in + 1)}
    for s in range(1, DEPTH + 1):
        for i in range(n_in + 1):
            prev[i] = lay.add_fixed({prev[i]: 1}, f"v{i}-{s}")
    if kind == "library":
        F.add_library(lay)
    return lay


def data(task, n_train, seed, latency):
    n_in, V, unseen = e8.dataset(task, n_train, seed)
    if latency:
        V, unseen = [delayed(s, n_in) for s in V], [delayed(s, n_in) for s in unseen]
    return n_in, V, unseen


# ---- part A

def one_round(args):
    task, latency, kind, n_train, seed = args
    rng = np.random.default_rng(seed)
    out, V, unseen = data(task, n_train, seed, latency)
    lay = make_layout(kind, task)
    net = F.make_net(lay)
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    unseen = [F.extend(lay, s) for s in unseen]
    tr_ex = H.score(net, [s[0] for s in trained], Vx, [out], rng)[0]
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
    mid_ex, mid_acc = e8.score_mid(net, unseen, out)
    function, separable = diagnostics(Vx + unseen, out)
    return dict(task=task, latency=latency, layout=kind, n_train=n_train, seed=seed,
                machine_states=len(machine(task)[0]),
                realisable_at_latency_0=F.realisable_without_latency(machine(task)[1], machine(task)[0]),
                visible_neurons=nv, fixed_neurons=len(lay.fixed),
                chain_neurons=net.n - nv, train_seconds=round(secs, 4),
                train_exact_from_key=tr_ex, unseen_acc=un_acc, unseen_exact=un_ex,
                unseen_acc_mid=mid_acc, unseen_exact_mid=mid_ex,
                unseen_majority_baseline=H.majority_baseline(unseen, out),
                next_output_function_of_visible=function, visible_separable=separable,
                fixed_boxes_intact=F.fixed_boxes_intact(net, lay))


def part_a(pool):
    jobs = [(task, latency, kind, n_train, seed)
            for task in e8.TASKS for latency in (0, 1) for kind in LAYOUTS
            for n_train in N_TRAIN for seed in SEEDS]
    rows = pool.map(one_round, jobs, chunksize=4)
    H.write_csv("exp10_one_round.csv", rows)
    return rows


# ---- part B

def rule90(args):
    kind, n_train, seed = args
    rng = np.random.default_rng(100 * seed + n_train)       # the data of experiment 6
    states = e6.all_states()
    pick = rng.permutation(len(states))
    V = [e6.two_step_trajectory(states[i]) for i in pick[:n_train]]
    unseen = [e6.two_step_trajectory(states[i]) for i in pick[n_train:]]
    cells = list(range(e6.CELLS))
    lay = F.Layout(e6.CELLS)
    lay.add_on()
    if kind == "library":
        F.add_library(lay, sources=cells)
    net = F.make_net(lay)
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    secs = time.perf_counter() - t0
    nv = lay.n_visible
    unseen = [F.extend(lay, s) for s in unseen]
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, cells, rng)
    A = e8.midpoint(net)
    mid = []
    for s in unseen:
        v0 = np.r_[s[0], np.zeros(net.n - nv, int)]
        run = np.array(dnn.run(A, net.I, v0, [{lay.on}] * len(s), len(s) - 1))
        mid.append((run[:, cells] == s[:, cells]).all())
    return dict(layout=kind, n_train=n_train, seed=seed, n_unseen=len(unseen),
                visible_neurons=nv, fixed_neurons=len(lay.fixed),
                chain_neurons=net.n - nv, train_seconds=round(secs, 4),
                train_exact_from_key=H.score(net, [s[0] for s in trained],
                                             [s[:, :nv] for s in trained], cells, rng)[0],
                unseen_exact=un_ex, unseen_acc=un_acc, unseen_exact_mid=float(np.mean(mid)),
                majority_baseline=float(np.mean([(s[1:, cells] == 0).mean() for s in unseen])))


def part_b(pool):
    jobs = [(kind, n, seed) for kind in ("none", "library") for n in e6.CA_TRAIN for seed in e6.SEEDS]
    rows = pool.map(rule90, jobs)
    H.write_csv("exp10_rule90.csv", rows)
    return rows


# ---- part C

def rounds(args):
    task, latency, kind, n_rounds, per_round, seed = args
    rng = np.random.default_rng(seed)
    out, V, unseen = data(task, ROUND_STREAMS, seed, latency)
    lay = make_layout(kind, task)
    net = F.make_net(lay)
    unseen = [F.extend(lay, s) for s in unseen]
    rows, trained = [], []
    for r in range(1, n_rounds + 1):
        before = net.n
        trained += F.train(net, lay, V[(r - 1) * per_round: r * per_round], H.EPS)
        ok = [replays(net, [s], rng, trials=20) for s in trained]
        old = ok[:-per_round]
        un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
        rows.append(dict(task=task, latency=latency, layout=kind, rounds=n_rounds,
                         per_round=per_round, seed=seed, round=r,
                         chain_added=net.n - before, total_neurons=net.n,
                         recall_this_round=float(np.mean(ok[-per_round:])),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         unseen_acc=un_acc, unseen_exact=un_ex,
                         unseen_exact_mid=e8.score_mid(net, unseen, out)[0],
                         fixed_boxes_intact=F.fixed_boxes_intact(net, lay)))
    return rows


def part_c(pool):
    jobs = [(task, latency, kind, n_rounds, per_round, seed)
            for task, latency, kind in ROUND_CASES
            for n_rounds, per_round in SCHEDULES for seed in ROUND_SEEDS]
    rows = [r for group in pool.map(rounds, jobs) for r in group]
    H.write_csv("exp10_rounds.csv", rows)
    return rows


# ---- part D

def long_streams(args):
    task, latency, kind, seed = args
    out, V, _ = data(task, 128, seed, latency)
    lay = make_layout(kind, task)
    net = F.make_net(lay)
    F.train(net, lay, V, H.EPS)
    n_in, f = e8.TASKS[task]
    rng = np.random.default_rng(10_000 + seed)
    seqs = []
    for _ in range(LONG_STREAMS):
        s = e4.make_sequence(rng.integers(0, 2, (LONG_STEPS + 1, n_in)), f)
        seqs.append(F.extend(lay, delayed(s, n_in) if latency else s))
    ex, acc = H.score(net, H.zero_hidden_starts(seqs), seqs, [out], rng)
    mid_ex, mid_acc = e8.score_mid(net, seqs, out)
    return dict(task=task, latency=latency, layout=kind, seed=seed, steps=LONG_STEPS,
                chain_neurons=net.n - lay.n_visible, long_exact=ex, long_acc=acc,
                long_exact_mid=mid_ex, long_acc_mid=mid_acc)


def part_d(pool):
    rows = pool.map(long_streams, [case + (seed,) for case in LONG_CASES for seed in SEEDS])
    H.write_csv("exp10_long_streams.csv", rows)
    return rows


# ---- tables

def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _f(x):
    return f"{x:.3g}" if x in (0, 1) else f"{x:.3f}"


def summarise_a(rows, n_train=128):
    print(f"\nPart A, {n_train} training streams, means over seeds: chain neurons, then the fraction "
          "of unseen streams exact (sampled weights / midpoint weights)\n")
    head = [f"{kind}, latency {lat}" for lat in (0, 1) for kind in LAYOUTS]
    print("| task | states | realisable at latency 0 | " + " | ".join(head) + " |")
    print("|" + "---|" * (len(head) + 3))
    for task in e8.TASKS:
        g0 = _sel(rows, task=task)
        cells = []
        for lat in (0, 1):
            for kind in LAYOUTS:
                g = _sel(g0, latency=lat, layout=kind, n_train=n_train)
                cells.append(f"{H.mean(g, 'chain_neurons'):.3g}; {_f(H.mean(g, 'unseen_exact'))} / "
                             f"{_f(H.mean(g, 'unseen_exact_mid'))}")
        print(f"| {task} | {g0[0]['machine_states']} | {'yes' if g0[0]['realisable_at_latency_0'] else 'no'} | "
              + " | ".join(cells) + " |")
    print()
    for kind in LAYOUTS:
        g = _sel(rows, layout=kind)
        print(f"{kind}: {g[0]['fixed_neurons']}-{max(r['fixed_neurons'] for r in g)} fixed neurons; "
              f"max training time {max(r['train_seconds'] for r in g):.1f} s")
    st = _sel(rows, layout="states")
    agree = sum((r["chain_neurons"] == 0) == (r["latency"] == 1 or r["realisable_at_latency_0"]) for r in st)
    print(f"states layout: 'no hidden neuron added' agrees with the prediction in {agree} of {len(st)} runs")
    sep = [r for r in rows if r["visible_separable"]]
    print(f"all layouts: {len(rows)} runs; threshold function of the visible state in {len(sep)}, of which "
          f"{sum(r['chain_neurons'] > 0 for r in sep)} added chain neurons; "
          f"not a threshold function in {len(rows) - len(sep)}, of which "
          f"{sum(r['unseen_exact'] == 1.0 for r in rows if not r['visible_separable'])} exact on every unseen stream")
    print("training exact from key, min:", min(r["train_exact_from_key"] for r in rows),
          "; fixed boxes intact:", all(r["fixed_boxes_intact"] for r in rows))


def summarise_b(rows):
    print("\nPart B, two-step rule 90 on a ring of 6, means over seeds\n")
    print("| layout | training states | fixed neurons | chain neurons | train s | training exact from key "
          "| unseen exact | unseen exact (midpoint) | unseen cell accuracy | baseline |")
    print("|" + "---|" * 10)
    for kind in ("none", "library"):
        for n in e6.CA_TRAIN:
            g = _sel(rows, layout=kind, n_train=n)
            print(f"| {kind} | {n} | {g[0]['fixed_neurons']} | {H.mean(g, 'chain_neurons'):.3g} "
                  f"| {H.mean(g, 'train_seconds'):.2f} | {H.mean(g, 'train_exact_from_key'):.3g} "
                  f"| {H.mean(g, 'unseen_exact'):.3f} | {H.mean(g, 'unseen_exact_mid'):.3f} "
                  f"| {H.mean(g, 'unseen_acc'):.3f} | {H.mean(g, 'majority_baseline'):.3f} |")


def summarise_c(rows):
    n = len(ROUND_SEEDS)
    print(f"\nPart C, {ROUND_STREAMS} training streams, {n} seeds\n")
    print("| task | latency | layout | rounds x streams | chain: round 1 | runs that add chains late "
          "| neurons at end | recall, earlier rounds (min) | unseen exact: round 1 "
          "| unseen exact: last round | runs exact on all unseen streams at the end (sampled / midpoint) |")
    print("|" + "---|" * 11)
    for task, latency, kind in ROUND_CASES:
        for n_rounds, per_round in SCHEDULES:
            g = _sel(rows, task=task, layout=kind, rounds=n_rounds)
            first, last = _sel(g, round=1), _sel(g, round=n_rounds)
            late = {r["seed"] for r in g if r["round"] > 1 and r["chain_added"] > 0}
            print(f"| {task} | {latency} | {kind} | {n_rounds} x {per_round} "
                  f"| {H.mean(first, 'chain_added'):.3g} | {len(late)} of {n} "
                  f"| {H.mean(last, 'total_neurons'):.4g} "
                  f"| {min(float(r['recall_earlier_rounds']) for r in g if r['round'] > 1):.3g} "
                  f"| {H.mean(first, 'unseen_exact'):.3f} | {H.mean(last, 'unseen_exact'):.3f} "
                  f"| {sum(r['unseen_exact'] == 1.0 for r in last)} / "
                  f"{sum(r['unseen_exact_mid'] == 1.0 for r in last)} of {n} |")
    old = [float(r["recall_earlier_rounds"]) for r in rows if r["recall_earlier_rounds"] != ""]
    print(f"\nPart C, recall: min over {len(old)} checks of earlier rounds = {min(old)}, "
          f"min over {len(rows)} checks of the current round = {min(r['recall_this_round'] for r in rows)}")


def summarise_d(rows):
    print(f"\nPart D, trained on 128 streams of 13 vectors, tested on {LONG_STREAMS} streams of "
          f"{LONG_STEPS + 1} vectors, means over seeds\n")
    print("| task | latency | layout | chain neurons | long streams exact (sampled) | long streams exact (midpoint) "
          "| output accuracy (sampled) |")
    print("|" + "---|" * 7)
    for task, latency, kind in LONG_CASES:
        g = _sel(rows, task=task, layout=kind)
        print(f"| {task} | {latency} | {kind} | {H.mean(g, 'chain_neurons'):.3g} | {H.mean(g, 'long_exact'):.3f} "
              f"| {H.mean(g, 'long_exact_mid'):.3f} | {H.mean(g, 'long_acc'):.4f} |")


def main():
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        a = part_a(pool)
        b = part_b(pool)
        c = part_c(pool)
        d = part_d(pool)
    summarise_a(a)
    summarise_b(b)
    summarise_c(c)
    summarise_d(d)


if __name__ == "__main__":
    main()
