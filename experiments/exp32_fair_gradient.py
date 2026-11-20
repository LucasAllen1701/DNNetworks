"""Experiment 32: a fairer gradient-descent baseline for the tasks with memory.

Experiment 24's Elman RNN was untuned and scored only on whole streams. Same
data, seeds and tasks as experiment 24's state (experiment 20's seven
finite-state tasks, 2 to 32 streams) and game (experiment 23's six grid-game
tasks, 8 to 32 episodes) families. Models:
  rnn:        experiment 24's RNN (64 tanh units, Adam 3000 steps at 0.005);
              its rows must equal experiment 24's.
  rnn_tuned:  the RNN with (hidden units, learning rate) chosen from
              (64, 0.005), (32, 0.005), (128, 0.005), (64, 0.02) by held-out
              training streams (every 4th stream held out; with 2 streams,
              the second), on per-step accuracy, then refitted on all.
  gru_tuned:  a GRU (baseline_gd.GRU) chosen the same way.
3000 Adam steps, full batch, gradient clipping at 5. Scores on unseen test
streams: whole streams exact and per-step accuracy. The summary sets them
next to the discrete networks (experiment 20's composed letters, h = 3;
experiment 23's learned bits; committed CSVs).
Run from the repository root:  python experiments/exp32_fair_gradient.py
Writes experiments/results/exp32_fair_gradient.csv.
"""
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import exp24_gradient_baseline as e24
import baseline_gd as G

SEEDS = e24.SEEDS
FAMILIES = {k: v for k, v in e24.FAMILIES.items() if v["model"] == "rnn"}
MODELS = ["rnn", "rnn_tuned", "gru_tuned"]
GRID = [(64, 0.005), (32, 0.005), (128, 0.005), (64, 0.02)]


def split(n):
    """Indices (fit, held out) of n training streams."""
    held = [1] if n < 4 else [i for i in range(n) if i % 4 == 3]
    return [i for i in range(n) if i not in held], held


def one_run(args):
    family, task, n_train, seed, model = args
    V, tests, n_in, h = e24.stream_data(family, task, n_train, seed)
    X = np.array([s[:-h, :n_in] for s in V])
    Y = np.array([s[h:, n_in] for s in V])
    TX = np.array([s[:-h, :n_in] for s in tests])
    TY = np.array([s[h:, n_in] for s in tests])
    row = dict(family=family, task=task, n_train=n_train, seed=seed, model=model)
    t0 = time.perf_counter()
    cls = G.GRU if model.startswith("gru") else G.RNN
    if model == "rnn":
        hidden, lr, held_acc = 64, 0.005, ""
    else:
        fit, held = split(n_train)
        scores = []
        for k, (hid, rate) in enumerate(GRID):
            P = cls(n_in, hidden=hid, seed=seed).fit(X[fit], Y[fit], lr=rate).predict(X[held])
            scores.append(((P == Y[held]).mean(), (P == Y[held]).all(1).mean(), -k))
        best = max(range(len(GRID)), key=lambda k: scores[k])
        (hidden, lr), held_acc = GRID[best], float(scores[best][0])
    net = cls(n_in, hidden=hidden, seed=seed).fit(X, Y, lr=lr)
    P = net.predict(TX)
    row.update(hidden=hidden, lr=lr, heldout_step_acc=held_acc,
               train_exact=float((net.predict(X) == Y).all(1).mean()),
               unseen_exact=float((P == TY).all(1).mean()), unseen_step_acc=float((P == TY).mean()),
               seconds=round(time.perf_counter() - t0, 3))
    return row


def _read(name):
    path = os.path.join(H.ROOT, "experiments", "results", name)      # committed CSVs, not H.RESULTS
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def dnn_reference():
    """{(family, n_train): (whole streams exact, per-step accuracy or '')} of the discrete networks."""
    acc = {}
    for r in _read("exp20_learned_letter.csv"):
        if r["layout"] == "composed" and r["unseen_exact_mid"] != "":
            acc.setdefault(("state", int(r["n_train"])), []).append(
                (float(r["unseen_exact_mid"]), float(r["unseen_step_acc_mid"])))
    for r in _read("exp23_grid_game.csv"):
        if r["layout"] == "bits" and r["policy"] == "all" and r["unseen_exact_mid"] != "":
            acc.setdefault(("game", int(r["round"]) * e23.PER_ROUND), []).append((float(r["unseen_exact_mid"]), None))
    return {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v]) if v[0][1] is not None else None)
            for k, v in acc.items()}


def summarise(rows):
    ref = dnn_reference()
    for family, spec in FAMILIES.items():
        g = [r for r in rows if r["family"] == family]
        if not g:
            continue
        sizes = spec["sizes"]
        print(f"\n{family}: unseen test streams with every output right / per-step accuracy, all tasks and seeds\n")
        print("| model | " + " | ".join(str(s) for s in sizes) + " |")
        print("|" + "---|" * (len(sizes) + 1))
        for m in MODELS:
            cells = []
            for s in sizes:
                c = [r for r in g if r["model"] == m and r["n_train"] == s]
                cells.append(f"{np.mean([r['unseen_exact'] for r in c]):.2f} / {np.mean([r['unseen_step_acc'] for r in c]):.3f}"
                             if c else "–")
            print(f"| {m} | " + " | ".join(cells) + " |")
        cells = []
        for s in sizes:
            v = ref.get((family, s))
            cells.append("–" if v is None else f"{v[0]:.2f} / " + ("–" if v[1] is None else f"{v[1]:.3f}"))
        print("| discrete | " + " | ".join(cells) + " |")
        for m in MODELS[1:]:
            c = [r for r in g if r["model"] == m]
            picks = {}
            for r in c:
                picks[(r["hidden"], r["lr"])] = picks.get((r["hidden"], r["lr"]), 0) + 1
            print(f"\n{family} {m}: settings chosen " + ", ".join(f"{k}: {v}" for k, v in sorted(picks.items())))
    print(f"\nall training streams fitted exactly: "
          + ", ".join(f"{m} {np.mean([r['train_exact'] for r in rows if r['model'] == m]):.3f}" for m in MODELS))
    print(f"max seconds: {max(r['seconds'] for r in rows):.1f}")


def main(seeds=SEEDS, families=tuple(FAMILIES), models=tuple(MODELS), sizes=None):
    jobs = [(f, t, n, s, m) for f in families for t in FAMILIES[f]["tasks"] for n in (sizes or FAMILIES[f]["sizes"])
            for s in seeds for m in models]
    jobs.sort(key=lambda j: (j[4] != "gru_tuned", -j[2]))           # slow runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    fams = list(FAMILIES)
    rows.sort(key=lambda r: (fams.index(r["family"]), FAMILIES[r["family"]]["tasks"].index(r["task"]),
                             r["n_train"], r["seed"], MODELS.index(r["model"])))
    H.write_csv("exp32_fair_gradient.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
