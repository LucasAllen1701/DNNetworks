"""Experiment 11: the wrong-box problem, other box rules, and a constraint buffer.

A learnable task trained in small rounds can lock in a box that contains no
correct weight vector (experiments 8.2, 9.2, 10.4). docs/Box_Rule.md shows
that no rule for choosing the box avoids this (Proposition 6) and that a
bounded buffer of exact constraints does, as long as it does not overflow
(Proposition 7). This experiment measures both. `dnn.py` is unchanged.

A run "goes wrong" if any round after the first adds a hidden neuron: the
tasks here are all threshold functions of the visible state, so no hidden
neuron is ever needed.

Cases (fixed-neuron layouts of experiments 8-10, all learnable):
  tomita4 with input shifts and clock; with all shifts and clock; with all
  shifts and twins; tomita2 with all shifts; tomita7 with state neurons;
  parity at latency 1 with state neurons; and, with the feature library,
  parity at latency 1, tomita4, and xor2 at latency 1.

Part A (box rules, no buffer): the notes' rule, the same with cap 16, the
  bounded rule and the largest-volume rule (dnn_box.py). 32 training streams
  fed 4 per round or 16 per round, 10 seeds.
Part B (buffer, the notes' rule on overflow): capacities 0, 16, 64 and 256
  constraints. The same schedules, plus one stream per round for the cases
  without the library. After every round every stream trained so far is
  replayed from its key with 5 weight matrices drawn from the regions.

Run from the repository root:  python experiments/exp11_box_rule.py
Writes experiments/results/exp11_{rules,buffer}.csv.
"""
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp8_past_shift_inputs as e8
import exp10_library_and_states as e10
import dnn_box as B
import dnn_buffer as BUF
import dnn_fixed as F

dnn = H.dnn

CASES = {
    "tomita4, input shifts + clock": ("shifts", "tomita4", 0, 2, "inputs", "clock"),
    "tomita4, all shifts + clock": ("shifts", "tomita4", 0, 2, "all", "clock"),
    "tomita4, all shifts + twins": ("shifts", "tomita4", 0, 2, "all", "twins"),
    "tomita2, all shifts": ("shifts", "tomita2", 0, 1, "all", "none"),
    "tomita7, states": ("e10", "tomita7", 0, "states"),
    "parity (latency 1), states": ("e10", "parity", 1, "states"),
    "parity (latency 1), library": ("e10", "parity", 1, "library"),
    "tomita4, library": ("e10", "tomita4", 0, "library"),
    "xor2 (latency 1), library": ("e10", "xor2", 1, "library"),
}
RULES = {
    "notes (cap 1)": lambda: dnn.subbox,
    "cap 16": lambda: B.capped_rule(16.0),
    "bounded 8": lambda: B.bounded_rule(8.0),
    "volume 8": lambda: B.volume_rule(8.0),
    "volume 32": lambda: B.volume_rule(32.0),
}
CAPACITIES = [0, 16, 64, 256]
STREAMS = 32
SEEDS = range(10)
MODES = ["lower", "upper", "random", "random", "random"]


def setup(case, seed):
    spec = CASES[case]
    if spec[0] == "shifts":
        _, task, latency, depth, sources, extra = spec
        n_in = e8.TASKS[task][0]
        lay = F.make_layout(n_in + 1, range(n_in), {n_in}, depth,
                            None if sources == "all" else range(n_in), extra)
    else:
        _, task, latency, kind = spec
        lay = e10.make_layout(kind, task)
    out, V, unseen = e10.data(task, STREAMS, seed, latency)
    return lay, out, V, [F.extend(lay, s) for s in unseen]


def runs_of(net, A, seqs):
    """Run every sequence from its first row (zero-padded) with weights A."""
    for s in seqs:
        inputs = [{i for i in net.I if s[t][i]} for t in range(len(s))]
        v0 = np.r_[s[0], np.zeros(net.n - len(s[0]), int)]
        yield s, np.array(dnn.run(A, net.I, v0, inputs, len(s) - 1))


def unseen_exact(net, buf, unseen, out, rng):
    """Fraction of unseen streams exact: mean over the five sampled matrices,
    and for the midpoint matrix."""
    sampled = []
    for mode in MODES:
        A = BUF.sample_weights(net, buf, rng, mode)
        sampled.append(np.mean([(r[:, out] == s[:, out]).all() for s, r in runs_of(net, A, unseen)]))
    A = BUF.sample_weights(net, buf, rng, "mid")
    mid = np.mean([(r[:, out] == s[:, out]).all() for s, r in runs_of(net, A, unseen)])
    return float(np.mean(sampled)), float(mid)


def recall(net, buf, trained, rng):
    """Fraction of trained sequences replayed exactly, on all neurons, from
    their keys, by every one of 5 weight matrices from the regions."""
    ok = np.ones(len(trained), bool)
    for mode in ["lower", "upper", "mid", "random", "random"]:
        A = BUF.sample_weights(net, buf, rng, mode)
        ok &= [bool((r[:, :s.shape[1]] == s).all()) for s, r in runs_of(net, A, trained)]
    return float(ok.mean())


def one_run(args):
    part, case, rule_name, capacity, n_rounds, per_round, seed = args
    lay, out, V, unseen = setup(case, seed)
    net = F.make_net(lay)
    rng = np.random.default_rng(seed)
    rule = B.only_for(RULES[rule_name](), [out])
    buf = BUF.Buffer([out], capacity, eps=H.EPS, box_rule=rule)
    trained, late, first, min_recall, max_rows = [], 0, None, 1.0, 0
    t0 = time.perf_counter()
    for r in range(n_rounds):
        before = net.n
        with B.using(rule), buf.active():
            trained += F.train(net, lay, V[r * per_round:(r + 1) * per_round], H.EPS)
        late += r > 0 and net.n > before
        max_rows = max(max_rows, buf.stored())
        if part == "buffer":
            min_recall = min(min_recall, recall(net, buf, trained, rng))
        if r == 0:
            first = unseen_exact(net, buf, unseen, out, rng)[0]
    secs = time.perf_counter() - t0
    if part == "rules":
        min_recall = recall(net, buf, trained, rng)
    ex, mid = unseen_exact(net, buf, unseen, out, rng)
    return dict(case=case, rule=rule_name, capacity=capacity, rounds=n_rounds, per_round=per_round,
                seed=seed, visible_neurons=lay.n_visible, went_wrong=bool(late),
                rounds_adding_chains_late=int(late), chain_neurons=net.n - lay.n_visible,
                unseen_exact_round_1=first, unseen_exact=ex, unseen_exact_mid=mid,
                recall_min=min_recall, constraints_stored_end=buf.stored(),
                constraints_stored_max=max_rows, overflows=buf.compressions if capacity else "",
                seconds=round(secs, 3))


def part_a(pool):
    jobs = [("rules", case, rule, 0, n_rounds, per_round, seed)
            for case in CASES for rule in RULES
            for n_rounds, per_round in ((8, 4), (2, 16)) for seed in SEEDS]
    rows = pool.map(one_run, jobs, chunksize=2)
    H.write_csv("exp11_rules.csv", rows)
    return rows


def part_b(pool):
    jobs = []
    for case in CASES:
        schedules = [(8, 4), (2, 16)] + ([] if "library" in case else [(32, 1)])
        jobs += [("buffer", case, "notes (cap 1)", cap, n_rounds, per_round, seed)
                 for cap in CAPACITIES for n_rounds, per_round in schedules for seed in SEEDS]
    rows = pool.map(one_run, jobs, chunksize=2)
    H.write_csv("exp11_buffer.csv", rows)
    return rows


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def summarise_a(rows):
    n = len(SEEDS)
    for n_rounds, per_round in ((8, 4), (2, 16)):
        print(f"\nPart A, {n_rounds} rounds of {per_round} streams: runs that go wrong, of {n} "
              "(runs exact on every unseen stream at the end, midpoint weights)\n")
        print("| case | visible neurons | " + " | ".join(RULES) + " |")
        print("|" + "---|" * (len(RULES) + 2))
        totals = {rule: 0 for rule in RULES}
        for case in CASES:
            cells = []
            for rule in RULES:
                g = _sel(rows, case=case, rule=rule, rounds=n_rounds)
                bad = sum(r["went_wrong"] for r in g)
                totals[rule] += bad
                cells.append(f"{bad} ({sum(r['unseen_exact_mid'] == 1.0 for r in g)})")
            print(f"| {case} | {g[0]['visible_neurons']} | " + " | ".join(cells) + " |")
        print(f"| total, of {n * len(CASES)} | | " + " | ".join(str(totals[r]) for r in RULES) + " |")
    print("\nPart A, recall of all trained streams at the end, min over runs:",
          min(r["recall_min"] for r in rows))


def summarise_b(rows):
    n = len(SEEDS)
    for n_rounds, per_round in ((32, 1), (8, 4), (2, 16)):
        print(f"\nPart B, {n_rounds} rounds of {per_round}: runs that go wrong, of {n} "
              "(runs exact on every unseen stream at the end, midpoint weights; "
              "largest number of constraints stored)\n")
        print("| case | visible neurons | " + " | ".join(f"capacity {c}" for c in CAPACITIES) + " |")
        print("|" + "---|" * (len(CAPACITIES) + 2))
        totals = {c: 0 for c in CAPACITIES}
        count = 0
        for case in CASES:
            if not _sel(rows, case=case, rounds=n_rounds):
                continue
            count += 1
            cells = []
            for cap in CAPACITIES:
                g = _sel(rows, case=case, capacity=cap, rounds=n_rounds)
                bad = sum(r["went_wrong"] for r in g)
                totals[cap] += bad
                cells.append(f"{bad} ({sum(r['unseen_exact_mid'] == 1.0 for r in g)}; "
                             f"{max(r['constraints_stored_max'] for r in g)})")
            print(f"| {case} | {g[0]['visible_neurons']} | " + " | ".join(cells) + " |")
        print(f"| total, of {n * count} | | " + " | ".join(str(totals[c]) for c in CAPACITIES) + " |")
    print("\nPart B, recall of every stream trained so far after every round, min over runs:",
          min(r["recall_min"] for r in rows))
    for cap in CAPACITIES:
        g = _sel(rows, capacity=cap)
        print(f"capacity {cap}: unseen exact at the end, sampled weights, mean over all runs = "
              f"{H.mean(g, 'unseen_exact'):.3f}; midpoint = {H.mean(g, 'unseen_exact_mid'):.3f}")


def main():
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        a = part_a(pool)
        b = part_b(pool)
    summarise_a(a)
    summarise_b(b)


if __name__ == "__main__":
    main()
