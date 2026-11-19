"""Experiment 21: more helpers for shapes on 4 x 4 images (follow-up to experiment 19).

In experiment 19 the helper search stopped at 20 helpers and then added one
indicator per remaining positive image. Upstart then fell to the lookup
table on lines and blocks, and cover reached 0.79-0.88. Same tasks, images,
seeds and scoring as experiment 19. Layouts:
  "upstart", "cover":  experiment 19's (limit 20). These rows should equal
                       experiment 19's.
  "upstart200", "cover200": the same searches with a limit of 200 helpers
                       (more than the positive images seen, so in effect no
                       limit).
  "nbr3_upstart", "nbr3_cover": a second hidden layer (h = 3, as experiment
                       16's pairs3): layer 1 is the AND of two literals for
                       every pair of neighbouring pixels (24 pairs x 4 = 96
                       fixed neurons), layer 2 is built over the inputs and
                       layer 1 by upstart or cover (limit 200).
  "pairs3_cover":      the same with every pair of pixels in layer 1 (120
                       pairs x 4 = 480 fixed neurons), layer 2 by cover.
The output reads only its helpers (and its bias neuron). At h = 3 the output
at t + 3 is f(x(t)), so each stream has one labelled image fewer.
Run from the repository root:  python experiments/exp21_more_helpers_images.py
Writes experiments/results/exp21_more_helpers.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp14_boolean_functions as e14
import exp16_or_of_ands as e16
import exp19_small_images as e19
import dnn_fixed as F
import dnn_upstart as U

dnn = H.dnn
N_IN, T = e19.N_IN, e19.T
SIZES = e19.SIZES
SEEDS = e19.SEEDS
TASKS = e19.TASKS
LAYOUTS = ["upstart", "cover", "upstart200", "cover200", "nbr3_upstart", "nbr3_cover", "pairs3_cover"]
LIMIT = 200
LAYER1 = {"nbr3": [r for r, _ in U.neighbour_pair_rules(e19.SIDE)],
          "pairs3": [r for r, _ in U.pair_rules(N_IN)]}


def one_run(args):
    task, n_train, seed, kind = args
    if kind in ("upstart", "cover"):
        return e19.one_run(args)
    f = TASKS[task]
    h = 3 if "3" in kind else 2
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [e19.images(task, rng, T + 1) for _ in range(n_train)]
    trng = np.random.default_rng(90_000 + seed)
    tests = [e16.stream(e19.images(task, trng, T + 1), f, h) for _ in range(e19.TEST_STREAMS)]
    V = [e16.stream(x, f, h) for x in xs]
    pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T + 1 - h)}
    X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    search = U.build if kind.endswith("upstart") or kind == "upstart200" else U.build_cover
    t0 = time.perf_counter()
    if h == 3:
        layer1 = LAYER1[kind.split("_")[0]]
        rules = search(U.layer1_values(X, layer1), y, H.EPS, max_helpers=LIMIT)
        lay, allowed = U.make_layout_h3(N_IN, layer1, rules)
    else:
        rules = search(X, y, H.EPS, max_helpers=LIMIT)
        lay = U.make_layout(N_IN, rules)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    if h == 3:
        blocked = [j for j in range(lay.n_visible) if j not in allowed]
        net.L[N_IN, blocked] = net.U[N_IN, blocked] = 0.0
    else:
        net.L[N_IN, :N_IN + 1] = net.U[N_IN, :N_IN + 1] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    answers = np.concatenate([s[h:, N_IN] for s in tests])
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind, patterns_seen=len(pats),
               positives_seen=int(y.sum()), helpers=len(rules), hidden_added=net.n - nv,
               base_rate=float(max(answers.mean(), 1 - answers.mean())),
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [N_IN], rng)[0])
    row.update(e16.evaluate(net, lay, tests, set(pats), rng, h, N_IN))
    row["depth"] = h
    return row


def summarise(rows):
    sel, mean = e14._sel, e14._mean
    print("\nMean accuracy on unseen images (midpoint weights), by number of training streams\n")
    print("| task | base rate | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 3))
    for task in TASKS:
        base = mean(sel(rows, task=task, layout="upstart"), "base_rate")
        for kind in LAYOUTS:
            g = sel(rows, task=task, layout=kind)
            print(f"| {task} | {base} | {kind} | " + " | ".join(mean(sel(g, n_train=s), "unseen_acc_mid") for s in SIZES) + " |")
    print("\nAt 32 training streams: mean helpers (layer 2 at h = 3) / mean unseen accuracy (midpoint)\n")
    print("| layout | " + " | ".join(TASKS) + " |")
    print("|" + "---|" * (len(TASKS) + 1))
    for kind in LAYOUTS:
        print(f"| {kind} | " + " | ".join(f"{mean(sel(rows, task=t, layout=kind, n_train=SIZES[-1]), 'helpers')} / "
                                         f"{mean(sel(rows, task=t, layout=kind, n_train=SIZES[-1]), 'unseen_acc_mid')}"
                                         for t in TASKS) + " |")
    print("\nTotals over the 6 tasks\n")
    print("| layout | training streams | helpers (mean) | hidden neurons added | seen acc (mid) | unseen acc (mid) "
          "| unseen acc (sampled) | training streams exact without key |")
    print("|" + "---|" * 8)
    for kind in LAYOUTS:
        for s in SIZES:
            g = sel(rows, layout=kind, n_train=s)
            print(f"| {kind} | {s} | {mean(g, 'helpers')} | {sum(r['hidden_added'] for r in g)} | {mean(g, 'seen_acc_mid')} "
                  f"| {mean(g, 'unseen_acc_mid')} | {mean(g, 'unseen_acc_sampled')} "
                  f"| {sum(r['train_exact_without_key'] == 1.0 for r in g)} of {len(g)} |")
    print(f"\nmax seconds: build {max(r['build_seconds'] for r in rows):.2f}, "
          f"train {max(r['train_seconds'] for r in rows):.2f}")


def main(sizes=SIZES, seeds=SEEDS, tasks=TASKS):
    jobs = [(t, n, s, k) for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    jobs.sort(key=lambda j: (-j[1], "pairs3" not in j[3]))          # big runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), r["n_train"], r["seed"], LAYOUTS.index(r["layout"])))
    H.write_csv("exp21_more_helpers.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
