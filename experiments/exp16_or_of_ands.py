"""Experiment 16: helpers for OR-of-AND functions (a covering search, and a second hidden layer).

Experiment 14's upstart helpers were exact on majority, and3 and xor2 but
reached only 0.89 on dnf3 and 0.67 on mux, and were worse than the constant
answer on equal3. Same tasks, data and scoring as experiment 14 (six inputs,
1 to 16 training streams of 13 vectors, 10 seeds, 200 random test streams),
three layouts, all trained by `dnn.py` unchanged:
  "upstart": experiment 14's helpers (h = 2). These rows should equal
             experiment 14's upstart rows.
  "cover":   a covering search (h = 2, dnn_upstart.build_cover): each
             helper is off on every negative pattern seen and on for one
             uncovered positive, with the smallest sum of |weights|. On the
             full truth table it finds the terms of dnf3 and mux.
  "pairs3":  a second hidden layer (h = 3, dnn_upstart.make_layout_h3):
             layer 1 is every AND of two literals (60 fixed neurons, the same
             for every task), layer 2 is built by upstart over the inputs and
             layer 1. The output at t + 3 is f(x(t)).
The output reads only the helpers (and its bias neuron). Each output from
t = h on is scored, split by whether its input pattern was seen in training.
Run from the repository root:  python experiments/exp16_or_of_ands.py
Writes experiments/results/exp16_or_of_ands.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp14_boolean_functions as e14
import dnn_fixed as F
import dnn_upstart as U

dnn = H.dnn
N_IN, T = e14.N_IN, e14.T
SIZES = e14.SIZES
SEEDS = e14.SEEDS
TASKS = e14.TASKS
LAYOUTS = ["upstart", "cover", "pairs3"]
TEST_STREAMS = e14.TEST_STREAMS
MODES = e14.MODES
PAIRS = [r for r, _ in U.pair_rules(N_IN)]


def stream(x, f, h):
    """Task neurons: inputs x (T + 1 rows), output f(x(t - h)) at t >= h."""
    out = np.zeros(len(x), int)
    out[h:] = [f(v) for v in x[:-h]]
    return np.c_[x, out]


def run_out(net, A, s, out):
    """Output column of a run started with every non-task neuron at 0."""
    v0 = np.zeros(net.n, int)
    v0[:s.shape[1]] = s[0]
    sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
    return np.array(dnn.run(A, net.I, v0, sets, len(s) - 1))[:, out]


def evaluate(net, lay, tests, seen, rng, h, n_in=N_IN):
    """Accuracy at t >= h on seen and unseen input patterns, for the midpoint
    weights and averaged over sampled weights. The output is neuron n_in."""
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    res = {}
    for label, mats in (("mid", [mid]), ("sampled", [dnn.sample_weights(net, rng, m) for m in MODES])):
        hit = {True: [], False: []}
        for A in mats:
            for s in tests:
                pred = run_out(net, A, F.extend(lay, s), n_in)
                for t in range(h, len(s)):
                    hit[s[t - h, :n_in].tobytes() in seen].append(pred[t] == s[t, n_in])
        res[f"seen_acc_{label}"] = float(np.mean(hit[True])) if hit[True] else ""
        res[f"unseen_acc_{label}"] = float(np.mean(hit[False])) if hit[False] else ""
    return res


def one_run(args):
    task, n_train, seed, kind = args
    f = TASKS[task]
    h = 3 if kind == "pairs3" else 2
    rng = np.random.default_rng(seed)
    xs = [rng.integers(0, 2, (T + 1, N_IN)) for _ in range(n_train)]
    trng = np.random.default_rng(50_000 + seed)
    test_x = [trng.integers(0, 2, (T + 1, N_IN)) for _ in range(TEST_STREAMS)]
    V = [stream(x, f, h) for x in xs]
    tests = [stream(x, f, h) for x in test_x]
    # the labelled patterns: 11 per stream at h = 2 (as experiment 14), 10 at h = 3
    pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T + 1 - h)}
    X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    t0 = time.perf_counter()
    if kind == "pairs3":
        layer2 = U.build(U.layer1_values(X, PAIRS), y, H.EPS)
        lay, allowed = U.make_layout_h3(N_IN, PAIRS, layer2)
        helpers = len(layer2)
    else:
        rules = U.build(X, y, H.EPS) if kind == "upstart" else U.build_cover(X, y, H.EPS)
        lay = U.make_layout(N_IN, rules)
        helpers = len(rules)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    if kind == "pairs3":
        blocked = [j for j in range(lay.n_visible) if j not in allowed]
        net.L[N_IN, blocked] = net.U[N_IN, blocked] = 0.0
    else:                                  # as experiment 14
        net.L[N_IN, :N_IN + 1] = net.U[N_IN, :N_IN + 1] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind, depth=h,
               patterns_seen=len(pats), helpers=helpers, hidden_added=net.n - nv,
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [N_IN], rng)[0],
               train_exact_with_key=H.score(net, [s[0] for s in trained], trained, [N_IN], rng)[0])
    row.update(evaluate(net, lay, tests, set(pats), rng, h))
    return row


def summarise(rows):
    sel, mean = e14._sel, e14._mean
    print("\nMean accuracy on unseen input patterns (midpoint weights), by number of training streams\n")
    print("| task | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for task in TASKS:
        for kind in LAYOUTS:
            g = sel(rows, task=task, layout=kind)
            print(f"| {task} | {kind} | " + " | ".join(mean(sel(g, n_train=s), "unseen_acc_mid") for s in SIZES) + " |")
    print("\nTotals over the 8 tasks\n")
    print("| layout | training streams | runs | helpers (mean) | hidden neurons added | seen acc (mid) "
          "| unseen acc (mid) | unseen acc (sampled) | training streams exact without key |")
    print("|" + "---|" * 9)
    for kind in LAYOUTS:
        for s in SIZES:
            g = sel(rows, layout=kind, n_train=s)
            print(f"| {kind} | {s} | {len(g)} | {mean(g, 'helpers')} | {sum(r['hidden_added'] for r in g)} "
                  f"| {mean(g, 'seen_acc_mid')} | {mean(g, 'unseen_acc_mid')} | {mean(g, 'unseen_acc_sampled')} "
                  f"| {sum(r['train_exact_without_key'] == 1.0 for r in g)} |")
    print(f"\nmax seconds: build {max(r['build_seconds'] for r in rows):.2f}, "
          f"train {max(r['train_seconds'] for r in rows):.2f}")


def main(sizes=SIZES, seeds=SEEDS, tasks=TASKS):
    jobs = [(t, n, s, k) for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=2)
    H.write_csv("exp16_or_of_ands.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
