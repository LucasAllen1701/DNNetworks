"""Experiment 13: finding the state neurons from the training streams.

dnn_merge.py builds the prefix tree of the training streams, merges its
nodes into states, and installs the state neurons of the merged machine
before round 1 (docs/State_Merging.md). Nothing about the task is given to
it. The output is then trained by `dnn.py`, unchanged. Output timing is
h = 2 throughout (experiment 10's "latency 1").

Part A (one round): the 15 tasks, 4 to 128 training streams, 10 seeds.
  Layouts: "merged" (the machine after merging) and, up to 32 streams,
  "tree" (the prefix tree with no merging: the streams memorised, but
  without keys). Unseen streams of 13 vectors, and for "merged" also 100
  streams of 101 vectors.
Part B (several rounds): 32 training streams. The first S of them are
  trained together as round 1 and give the machine; the rest follow 4 per
  round. After every round every stream trained so far is replayed from its
  key with 20 weight matrices.

Run from the repository root:  python experiments/exp13_state_merging.py
Writes experiments/results/exp13_{one_round,rounds}.csv.
"""
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp4_input_tasks as e4
import exp8_past_shift_inputs as e8
import exp10_library_and_states as e10
import dnn_fixed as F
import dnn_merge as M
from tests.test_dnn import replays

dnn = H.dnn

SIZES = [4, 8, 16, 32, 64, 128]
TREE_SIZES = [4, 8, 16, 32]
SEEDS = range(10)
LONG_STEPS, LONG_STREAMS = 100, 100

ROUND_TASKS = ["parity", "adder", "tomita3", "tomita4", "tomita5", "tomita7"]
ROUND_STREAMS, PER_ROUND = 32, 4
STRUCTURE_STREAMS = [4, 8, 16, 32]


def data(task, n_train, seed):
    n_in, V, unseen = e8.dataset(task, n_train, seed)
    return n_in, [e10.delayed(s, n_in) for s in V], [e10.delayed(s, n_in) for s in unseen]


def score_mid(net, lay, seqs, out):
    """Fraction of streams exact with midpoint weights, run without a key."""
    A = e8.midpoint(net)
    exact = []
    for s in seqs:
        inputs = [{i for i in net.I if row[i]} for row in s]
        run = np.array(dnn.run(A, net.I, np.r_[s[0], np.zeros(net.n - len(s[0]), int)], inputs, len(s) - 1))
        exact.append((run[:, out] == s[:, out]).all())
    return float(np.mean(exact))


# ---- part A

def one_round(args):
    task, n_train, seed, kind = args
    rng = np.random.default_rng(seed)
    out, V, unseen = data(task, n_train, seed)
    t0 = time.perf_counter()
    lay, delta, m_out = M.make_layout(V, out, merging=(kind == "merged"))
    merge_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    true_delta, true_out, _ = e10.machine(task)
    tree_nodes = len(M.prefix_tree([M.observations(s, out) for s in V])[0])
    # recall without a key: every training stream from its first state alone
    Vx = [s[:, :nv] for s in trained]
    no_key = H.score(net, H.zero_hidden_starts(Vx), Vx, [out], rng)[0]
    unseen = [F.extend(lay, s) for s in unseen]
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind,
               tree_nodes=tree_nodes, states_found=len(delta), states_true=len(true_delta),
               machine_correct=M.same_machine(delta, m_out, true_delta, true_out),
               fixed_neurons=len(lay.fixed), chain_neurons=net.n - nv,
               merge_seconds=round(merge_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=no_key,
               unseen_acc=un_acc, unseen_exact=un_ex,
               unseen_exact_mid=score_mid(net, lay, unseen, out),
               unseen_majority_baseline=H.majority_baseline(unseen, out), long_exact_mid="")
    if kind == "merged":
        n_in, f = e8.TASKS[task]
        lrng = np.random.default_rng(10_000 + seed)
        longs = [F.extend(lay, e10.delayed(e4.make_sequence(lrng.integers(0, 2, (LONG_STEPS + 1, n_in)), f), n_in))
                 for _ in range(LONG_STREAMS)]
        row["long_exact_mid"] = score_mid(net, lay, longs, out)
    return row


def part_a(pool):
    jobs = [(task, n, seed, kind) for task in e8.TASKS for n in SIZES for seed in SEEDS
            for kind in (["merged", "tree"] if n in TREE_SIZES else ["merged"])]
    rows = pool.map(one_round, jobs, chunksize=4)
    H.write_csv("exp13_one_round.csv", rows)
    return rows


# ---- part B

def rounds(args):
    task, structure, seed = args
    rng = np.random.default_rng(seed)
    out, V, unseen = data(task, ROUND_STREAMS, seed)
    lay, delta, m_out = M.make_layout(V[:structure], out)
    true_delta, true_out, _ = e10.machine(task)
    net = F.make_net(lay)
    unseen = [F.extend(lay, s) for s in unseen]
    batches = [V[:structure]] + [V[i: i + PER_ROUND] for i in range(structure, ROUND_STREAMS, PER_ROUND)]
    rows, trained = [], []
    for r, batch in enumerate(batches, 1):
        before = net.n
        trained += F.train(net, lay, batch, H.EPS)
        ok = [replays(net, [s], rng, trials=20) for s in trained]
        old = ok[:-len(batch)]
        rows.append(dict(task=task, structure_streams=structure, seed=seed, round=r, streams_in_round=len(batch),
                         states_found=len(delta), states_true=len(true_delta),
                         machine_correct=M.same_machine(delta, m_out, true_delta, true_out),
                         chain_added=net.n - before, total_neurons=net.n,
                         recall_this_round=float(np.mean(ok[-len(batch):])),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         unseen_exact=H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)[0],
                         unseen_exact_mid=score_mid(net, lay, unseen, out)))
    return rows


def part_b(pool):
    jobs = [(task, s, seed) for task in ROUND_TASKS for s in STRUCTURE_STREAMS for seed in SEEDS]
    rows = [r for group in pool.map(rounds, jobs) for r in group]
    H.write_csv("exp13_rounds.csv", rows)
    return rows


# ---- tables

def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def summarise_a(rows):
    n = len(SEEDS)
    print(f"\nPart A, merged layout: runs of {n} that are exact on every unseen stream "
          "(midpoint weights), by number of training streams\n")
    print("| task | true states | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for task in e8.TASKS:
        g = _sel(rows, task=task, layout="merged")
        print(f"| {task} | {g[0]['states_true']} | "
              + " | ".join(str(sum(r["unseen_exact_mid"] == 1.0 for r in _sel(g, n_train=s))) for s in SIZES) + " |")
    print("\nPart A, totals over the 15 tasks\n")
    print("| layout | training streams | runs | true machine found | exact on every unseen stream (sampled) "
          "| (midpoint) | exact on every long stream (midpoint) | mean unseen exact (midpoint) "
          "| training streams exact without a key | hidden neurons added | states or nodes (mean) | fixed neurons (mean) |")
    print("|" + "---|" * 12)
    for kind in ("merged", "tree"):
        for s in (SIZES if kind == "merged" else TREE_SIZES):
            g = _sel(rows, layout=kind, n_train=s)
            longs = sum(r["long_exact_mid"] == 1.0 for r in g) if kind == "merged" else "–"
            print(f"| {kind} | {s} | {len(g)} | {sum(r['machine_correct'] for r in g)} "
                  f"| {sum(r['unseen_exact'] == 1.0 for r in g)} | {sum(r['unseen_exact_mid'] == 1.0 for r in g)} "
                  f"| {longs} | {H.mean(g, 'unseen_exact_mid'):.3f} "
                  f"| {sum(r['train_exact_without_key'] == 1.0 for r in g)} | {sum(r['chain_neurons'] for r in g)} "
                  f"| {H.mean(g, 'states_found'):.1f} | {H.mean(g, 'fixed_neurons'):.1f} |")
    m = _sel(rows, layout="merged")
    ok = [r for r in m if r["machine_correct"]]
    print(f"\nmerged: true machine found in {len(ok)} of {len(m)} runs; of these "
          f"{sum(r['unseen_exact_mid'] == 1.0 for r in ok)} exact on every unseen stream and "
          f"{sum(r['long_exact_mid'] == 1.0 for r in ok)} on every long stream (midpoint weights); "
          f"of the other {len(m) - len(ok)}, {sum(r['long_exact_mid'] == 1.0 for r in m if not r['machine_correct'])} "
          "exact on every long stream")
    print(f"states found vs true, merged: more in {sum(r['states_found'] > r['states_true'] for r in m)}, "
          f"fewer in {sum(r['states_found'] < r['states_true'] for r in m)}, equal in "
          f"{sum(r['states_found'] == r['states_true'] for r in m)}")
    print(f"max seconds: merging {max(r['merge_seconds'] for r in rows):.2f}, "
          f"training merged {max(r['train_seconds'] for r in m):.2f}, "
          f"training tree {max(r['train_seconds'] for r in _sel(rows, layout='tree')):.2f}")


def summarise_b(rows):
    n = len(SEEDS)
    print(f"\nPart B, {ROUND_STREAMS} training streams, {n} seeds: the first S streams give the machine\n")
    print("| task | S | true machine found | runs that add hidden neurons after round 1 "
          "| recall of earlier rounds (min) | hidden neurons at the end (mean) "
          "| runs exact on every unseen stream at the end (sampled / midpoint) |")
    print("|" + "---|" * 7)
    for task in ROUND_TASKS:
        for s in STRUCTURE_STREAMS:
            g = _sel(rows, task=task, structure_streams=s)
            last = [max(_sel(g, seed=seed), key=lambda r: r["round"]) for seed in SEEDS]
            late = {r["seed"] for r in g if r["round"] > 1 and r["chain_added"] > 0}
            old = [float(r["recall_earlier_rounds"]) for r in g if r["recall_earlier_rounds"] != ""]
            hidden = np.mean([sum(r["chain_added"] for r in _sel(g, seed=seed)) for seed in SEEDS])
            print(f"| {task} | {s} | {sum(r['machine_correct'] for r in last)} of {n} | {len(late)} of {n} "
                  f"| {min(old) if old else '–'} | {hidden:.3g} "
                  f"| {sum(r['unseen_exact'] == 1.0 for r in last)} / {sum(r['unseen_exact_mid'] == 1.0 for r in last)} |")
    old = [float(r["recall_earlier_rounds"]) for r in rows if r["recall_earlier_rounds"] != ""]
    print(f"\nPart B, recall: min over {len(old)} checks of earlier rounds = {min(old)}, "
          f"min over {len(rows)} checks of the current round = {min(r['recall_this_round'] for r in rows)}")
    for s in STRUCTURE_STREAMS:
        g = _sel(rows, structure_streams=s)
        last = [max(_sel(g, task=t, seed=seed), key=lambda r: r["round"]) for t in ROUND_TASKS for seed in SEEDS]
        late = {(r["task"], r["seed"]) for r in g if r["round"] > 1 and r["chain_added"] > 0}
        print(f"S = {s}: true machine in {sum(r['machine_correct'] for r in last)} of {len(last)}; "
              f"add hidden neurons late: {len(late)}; exact at the end (midpoint): "
              f"{sum(r['unseen_exact_mid'] == 1.0 for r in last)}")


def main():
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        a = part_a(pool)
        b = part_b(pool)
    summarise_a(a)
    summarise_b(b)


if __name__ == "__main__":
    main()
