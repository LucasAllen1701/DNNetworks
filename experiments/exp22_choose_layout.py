"""Experiment 22: choosing between helper layouts automatically.

In experiments 16, 19 and 21 different layouts win different tasks (upstart
on threshold functions, cover on OR-of-ANDs and lines, a layer of pair ANDs
on equal3 and some shapes). Here the layout is chosen from the training
streams alone, by two rules:
  "heldout":  cross-validation over training streams. The streams are split
              into k = min(streams, 4) folds; for each fold every layout is
              built and trained on the other streams and scored on the held
              out ones (outputs whose input pattern the other streams do not
              contain, midpoint weights). The layout with the best pooled
              score wins (ties: fewer helpers, then the order below). With
              1 training stream there is nothing to hold out, and the
              fewest-helpers rule is used.
  "fewest":   the layout with the fewest helpers built from the data (for
              h = 3, layer 2 only; ties: the order below).
Then the chosen layout, built on all the training streams, is scored on the
test streams, and compared with each fixed layout and with "oracle" (the
best layout for that run on the test streams, which a learner cannot know).
Tasks and data: experiment 16's eight Boolean functions of 6 inputs (1 to 16
streams; layouts upstart, cover, pairs3) and experiment 19's six image tasks
(2 to 32 streams; layouts upstart, cover and experiment 21's nbr3_cover).
Each row is one layout of one run. Its test columns come from experiment
16's, 19's or 21's own code, so they should equal those experiments' rows;
the held-out columns are new.
Run from the repository root:  python experiments/exp22_choose_layout.py
Writes experiments/results/exp22_choose_layout.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp14_boolean_functions as e14
import exp16_or_of_ands as e16
import exp19_small_images as e19
import exp21_more_helpers_images as e21
import dnn_fixed as F
import dnn_upstart as U

dnn = H.dnn
T = 12
FAMILIES = {
    "boolean": dict(tasks=e16.TASKS, sizes=e16.SIZES, layouts=["upstart", "cover", "pairs3"], n_in=e16.N_IN),
    "images": dict(tasks=e19.TASKS, sizes=e19.SIZES, layouts=["upstart", "cover", "nbr3_cover"], n_in=e19.N_IN),
}
SEEDS = range(10)
MAX_FOLDS = 4
SELECTORS = ["heldout", "fewest", "oracle"]


def training_inputs(family, task, n_train, seed):
    """The training input streams of experiment 16 or 19 for this run."""
    if family == "boolean":
        rng = np.random.default_rng(seed)
        return [rng.integers(0, 2, (T + 1, e16.N_IN)) for _ in range(n_train)]
    rng = np.random.default_rng(seed * 1000 + n_train)
    return [e19.images(task, rng, T + 1) for _ in range(n_train)]


def depth(kind):
    return 3 if kind in ("pairs3", "nbr3_cover") else 2


def build_and_train(family, kind, xs, f):
    """The network of layout `kind` built and trained on input streams xs,
    as experiments 16, 19 and 21 build it. Returns (net, layout, patterns)."""
    n_in, h = FAMILIES[family]["n_in"], depth(kind)
    V = [e16.stream(x, f, h) for x in xs]
    pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T + 1 - h)}
    X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    allowed = None
    if kind == "pairs3":
        layer2 = U.build(U.layer1_values(X, e16.PAIRS), y, H.EPS)
        lay, allowed = U.make_layout_h3(n_in, e16.PAIRS, layer2)
    elif kind == "nbr3_cover":
        layer1 = e21.LAYER1["nbr3"]
        layer2 = U.build_cover(U.layer1_values(X, layer1), y, H.EPS, max_helpers=e21.LIMIT)
        lay, allowed = U.make_layout_h3(n_in, layer1, layer2)
    else:
        rules = (U.build if kind == "upstart" else U.build_cover)(X, y, H.EPS)
        lay = U.make_layout(n_in, rules)
    net = F.make_net(lay)
    if allowed is None:
        net.L[n_in, :n_in + 1] = net.U[n_in, :n_in + 1] = 0.0
    else:
        blocked = [j for j in range(lay.n_visible) if j not in allowed]
        net.L[n_in, blocked] = net.U[n_in, blocked] = 0.0
    F.train(net, lay, V, H.EPS)
    return net, lay, set(pats)


def heldout(family, task, n_train, seed, kind):
    """(pooled accuracy on held-out outputs, number of outputs scored)."""
    if n_train < 2:
        return "", 0
    f = FAMILIES[family]["tasks"][task]
    n_in, h = FAMILIES[family]["n_in"], depth(kind)
    xs = training_inputs(family, task, n_train, seed)
    k = min(n_train, MAX_FOLDS)
    hits, total = 0, 0
    for fold in range(k):
        train = [x for i, x in enumerate(xs) if i % k != fold]
        held = [e16.stream(x, f, h) for i, x in enumerate(xs) if i % k == fold]
        net, lay, seen = build_and_train(family, kind, train, f)
        mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
        res = []
        for s in held:
            pred = e16.run_out(net, mid, F.extend(lay, s), n_in)
            res += [(s[t - h, :n_in].tobytes() in seen, pred[t] == s[t, n_in]) for t in range(h, len(s))]
        new = [ok for was_seen, ok in res if not was_seen]
        use = new if new else [ok for _, ok in res]
        hits, total = hits + sum(use), total + len(use)
    return hits / total, total


def one_run(args):
    family, task, n_train, seed, kind = args
    run = {"boolean": e16.one_run, "images": e21.one_run if kind == "nbr3_cover" else e19.one_run}[family]
    row = dict(family=family)
    row.update(run((task, n_train, seed, kind)))
    t0 = time.perf_counter()
    row["heldout_acc"], row["heldout_outputs"] = heldout(family, task, n_train, seed, kind)
    row["heldout_seconds"] = round(time.perf_counter() - t0, 4)
    return row


def choose(group, rule, layouts):
    """The row chosen by `rule` among the rows of one run (one per layout)."""
    order = {k: i for i, k in enumerate(layouts)}
    if rule == "oracle":
        return max(group, key=lambda r: (r["unseen_acc_mid"], -order[r["layout"]]))
    if rule == "heldout" and group[0]["heldout_acc"] != "":
        return min(group, key=lambda r: (-r["heldout_acc"], r["helpers"], order[r["layout"]]))
    return min(group, key=lambda r: (r["helpers"], order[r["layout"]]))


def runs(rows, family):
    groups = {}
    for r in rows:
        if r["family"] == family:
            groups.setdefault((r["task"], r["n_train"], r["seed"]), []).append(r)
    return groups


def summarise(rows):
    for family, spec in FAMILIES.items():
        groups = runs(rows, family)
        if not groups:
            continue
        layouts, sizes = spec["layouts"], spec["sizes"]
        print(f"\n{family}: mean accuracy on unseen patterns (midpoint), over all tasks and seeds, "
              "by number of training streams\n")
        print("| layout or rule | " + " | ".join(str(s) for s in sizes) + " |")
        print("|" + "---|" * (len(sizes) + 1))
        for name in layouts + SELECTORS:
            cells = []
            for s in sizes:
                g = [v for k, v in groups.items() if k[1] == s]
                if name in layouts:
                    vals = [r["unseen_acc_mid"] for grp in g for r in grp if r["layout"] == name]
                else:
                    vals = [choose(grp, name, layouts)["unseen_acc_mid"] for grp in g]
                cells.append(f"{np.mean(vals):.3f}" if vals else "–")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print(f"\n{family}: layout chosen by the held-out rule, runs per task (all sizes), and runs where it "
              "is the oracle's layout\n")
        print("| task | " + " | ".join(layouts) + " | same as oracle | of |")
        print("|" + "---|" * (len(layouts) + 3))
        for task in spec["tasks"]:
            g = [v for k, v in groups.items() if k[0] == task]
            picks = [choose(grp, "heldout", layouts)["layout"] for grp in g]
            same = sum(choose(grp, "heldout", layouts)["unseen_acc_mid"] == choose(grp, "oracle", layouts)["unseen_acc_mid"]
                       for grp in g)
            print(f"| {task} | " + " | ".join(str(picks.count(k)) for k in layouts) + f" | {same} | {len(g)} |")
    print(f"\nmax seconds: held-out {max(r['heldout_seconds'] for r in rows):.2f}")


def main(seeds=SEEDS, families=tuple(FAMILIES)):
    jobs = [(fam, t, n, s, k) for fam in families for t in FAMILIES[fam]["tasks"] for n in FAMILIES[fam]["sizes"]
            for s in seeds for k in FAMILIES[fam]["layouts"]]
    jobs.sort(key=lambda j: (-j[2], j[4] != "nbr3_cover"))          # big runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    fam_order = list(FAMILIES)
    rows.sort(key=lambda r: (fam_order.index(r["family"]), list(FAMILIES[r["family"]]["tasks"]).index(r["task"]),
                             r["n_train"], r["seed"], FAMILIES[r["family"]]["layouts"].index(r["layout"])))
    H.write_csv("exp22_choose_layout.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
