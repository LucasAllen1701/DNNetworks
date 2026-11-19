"""Experiment 24: an ordinary network trained by gradient descent, on the same data as experiments 16-23.

The baseline the discrete networks have never been compared with. Same
training and test data, seeds and scores as the earlier experiments:
  boolean:  experiment 16's eight functions of 6 inputs, 1 to 16 streams;
  images:   experiment 19's six 4 x 4 image tasks, 2 to 32 streams;
            for both, an MLP is trained on the labelled input patterns, and
            scored by accuracy on test outputs whose pattern is unseen;
  state:    experiment 20's seven finite-state tasks on 6-bit letters, 2 to
            32 streams;
  game:     experiment 23's six grid-game tasks, 8 to 32 episodes (the first
            n of its 32);
            for both, an Elman RNN reads x(t) and is trained to give the
            output that answers x(0..t); score: fraction of unseen test
            streams with every output right.
Networks (baseline_gd.py, numpy only): one hidden layer of 64 tanh units,
sigmoid output, cross-entropy, full-batch Adam; MLP 2000 steps at rate 0.01,
RNN 3000 steps at rate 0.005. The hyperparameters are fixed, not tuned per
task. The summary sets them next to the discrete networks' results from the
committed CSVs: experiment 22's held-out choice (boolean, images),
experiment 20's composed letters (state) and experiment 23's learned bits
with the whole buffer (game).
Run from the repository root:  python experiments/exp24_gradient_baseline.py
Writes experiments/results/exp24_gradient_baseline.csv.
"""
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp16_or_of_ands as e16
import exp17_helpers_and_state as e17
import exp19_small_images as e19
import exp20_learned_letter as e20
import exp22_choose_layout as e22
import exp23_grid_game as e23
import baseline_gd as G

T = 12
SEEDS = range(10)
FAMILIES = {
    "boolean": dict(tasks=list(e16.TASKS), sizes=e16.SIZES, model="mlp"),
    "images": dict(tasks=list(e19.TASKS), sizes=e19.SIZES, model="mlp"),
    "state": dict(tasks=list(e20.TASKS), sizes=e20.SIZES, model="rnn"),
    "game": dict(tasks=list(e23.TASKS), sizes=[8, 16, 24, 32], model="rnn"),
}


def pattern_data(family, task, n_train, seed):
    """(training input streams, test input streams, f) of experiment 16 or 19."""
    if family == "boolean":
        rng, trng = np.random.default_rng(seed), np.random.default_rng(50_000 + seed)
        xs = [rng.integers(0, 2, (T + 1, e16.N_IN)) for _ in range(n_train)]
        tests = [trng.integers(0, 2, (T + 1, e16.N_IN)) for _ in range(e16.TEST_STREAMS)]
        return xs, tests, e16.TASKS[task]
    rng, trng = np.random.default_rng(seed * 1000 + n_train), np.random.default_rng(90_000 + seed)
    xs = [e19.images(task, rng, T + 1) for _ in range(n_train)]
    tests = [e19.images(task, trng, T + 1) for _ in range(e19.TEST_STREAMS)]
    return xs, tests, e19.TASKS[task]


def stream_data(family, task, n_train, seed):
    """(training streams, test streams, n_in, h) of experiment 20 or 23."""
    if family == "state":
        h = 3
        rng, trng = np.random.default_rng(seed * 1000 + n_train), np.random.default_rng(70_000 + seed)
        V = [e17.stream(rng.integers(0, 2, (T + 1, e17.N_IN)), task, h) for _ in range(n_train)]
        tests = [e17.stream(trng.integers(0, 2, (T + 1, e17.N_IN)), task, h) for _ in range(e17.TEST_STREAMS)]
        return V, tests, e17.N_IN, h
    rng, trng = np.random.default_rng(seed), np.random.default_rng(80_000 + seed)
    V = [e23.stream(e23.screens(rng, e23.T + 1), task) for _ in range(e23.STREAMS)][:n_train]
    tests = [e23.stream(e23.screens(trng, e23.T + 1), task) for _ in range(e23.TEST_STREAMS)]
    return V, tests, e23.N_IN, e23.h


def one_run(args):
    family, task, n_train, seed = args
    row = dict(family=family, task=task, n_train=n_train, seed=seed, model=FAMILIES[family]["model"])
    t0 = time.perf_counter()
    if row["model"] == "mlp":
        h = 2                                   # as experiments 16 and 19 at h = 2
        xs, tests, f = pattern_data(family, task, n_train, seed)
        pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T + 1 - h)}
        X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
        y = np.array(list(pats.values()))
        net = G.MLP(X.shape[1], seed=seed).fit(X, y)
        TX = np.concatenate([x[:T + 1 - h] for x in tests])
        unseen = np.array([x.tobytes() not in pats for x in TX])
        right = net.predict(TX) == np.array([f(x) for x in TX])
        row.update(patterns_seen=len(pats), train_acc=float((net.predict(X) == y).mean()),
                   unseen_acc=float(right[unseen].mean()) if unseen.any() else "",
                   seen_acc=float(right[~unseen].mean()) if (~unseen).any() else "")
    else:
        V, tests, n_in, h = stream_data(family, task, n_train, seed)
        X = np.array([s[:-h, :n_in] for s in V])
        Y = np.array([s[h:, n_in] for s in V])
        net = G.RNN(n_in, seed=seed).fit(X, Y)
        TX = np.array([s[:-h, :n_in] for s in tests])
        TY = np.array([s[h:, n_in] for s in tests])
        P = net.predict(TX)
        row.update(train_exact=float((net.predict(X) == Y).all(1).mean()),
                   unseen_exact=float((P == TY).all(1).mean()), unseen_step_acc=float((P == TY).mean()))
    row["seconds"] = round(time.perf_counter() - t0, 3)
    return row


def _read(name):
    path = os.path.join(H.RESULTS, name)
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def dnn_reference():
    """{(family, task, n_train): mean score} of the discrete networks."""
    ref = {}
    rows22 = _read("exp22_choose_layout.csv")
    for family in ("boolean", "images"):
        layouts = e22.FAMILIES[family]["layouts"]
        groups = {}
        for r in rows22:
            if r["family"] == family:
                groups.setdefault((r["task"], int(r["n_train"]), int(r["seed"])), []).append(r)
        acc = {}
        for (task, n, _), g in groups.items():
            acc.setdefault((family, task, n), []).append(e22.choose(g, "heldout", layouts)["unseen_acc_mid"])
        ref.update({k: float(np.mean(v)) for k, v in acc.items()})
    for r in _read("exp20_learned_letter.csv"):
        if r["layout"] == "composed" and r.get("unseen_exact_mid", "") != "":
            ref.setdefault(("state", r["task"], int(r["n_train"])), []).append(r["unseen_exact_mid"])
    for r in _read("exp23_grid_game.csv"):
        if r["layout"] == "bits" and r["policy"] == "all":
            ref.setdefault(("game", r["task"], int(r["round"]) * e23.PER_ROUND), []).append(r["unseen_exact_mid"])
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in ref.items()}


def summarise(rows):
    ref = dnn_reference()
    for family, spec in FAMILIES.items():
        g = [r for r in rows if r["family"] == family]
        if not g:
            continue
        score = "unseen_acc" if spec["model"] == "mlp" else "unseen_exact"
        what = ("accuracy on unseen patterns" if spec["model"] == "mlp"
                else "fraction of unseen streams exact")
        print(f"\n{family}: {spec['model'].upper()} / discrete network, mean {what}, by number of training streams\n")
        print("| task | " + " | ".join(str(s) for s in spec["sizes"]) + " |")
        print("|" + "---|" * (len(spec["sizes"]) + 1))
        for task in spec["tasks"] + ["mean"]:
            cells = []
            for s in spec["sizes"]:
                tasks = spec["tasks"] if task == "mean" else [task]
                vals = [r[score] for r in g if r["task"] in tasks and r["n_train"] == s and r[score] != ""]
                refs = [ref[(family, t, s)] for t in tasks if (family, t, s) in ref]
                d = f"{np.mean(refs):.2f}" if len(refs) == len(tasks) else "?"
                cells.append(f"{np.mean(vals):.2f} / {d}" if vals else "–")
            print(f"| {task} | " + " | ".join(cells) + " |")
        fit = "train_acc" if spec["model"] == "mlp" else "train_exact"
        print(f"\n{family}: mean fit to the training data {np.mean([r[fit] for r in g]):.3f}; "
              f"max seconds {max(r['seconds'] for r in g):.1f}")
    if not ref:
        print("\n(no committed CSVs of experiments 20, 22, 23 found: the discrete column is '?')")


def main(seeds=SEEDS, families=tuple(FAMILIES)):
    jobs = [(f, t, n, s) for f in families for t in FAMILIES[f]["tasks"] for n in FAMILIES[f]["sizes"] for s in seeds]
    jobs.sort(key=lambda j: (FAMILIES[j[0]]["model"] != "rnn", -j[2]))          # slow runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    order = list(FAMILIES)
    rows.sort(key=lambda r: (order.index(r["family"]), FAMILIES[r["family"]]["tasks"].index(r["task"]),
                             r["n_train"], r["seed"]))
    H.write_csv("exp24_gradient_baseline.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
