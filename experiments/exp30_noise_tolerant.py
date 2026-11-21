"""Experiment 30: a first noise-tolerant training, on experiment 29's noisy data.

Same data, noise levels (p = 0, 0.05, 0.1, 0.2), sizes and seeds as
experiment 29. One change per family, before the linear programs:
  game:   the letter and the machine are found by search instead of exact
          merging (dnn_noise): the letter is the set of at most 2 of the 32
          regions whose best machine of at most 2 states disagrees least
          with the training moves; then its number of states (1 to 3) is
          chosen on held-out episodes and the machine refitted on all
          (every transition table tried, or local search for 3 states;
          majority output per state and letter). The training episodes are
          relabelled with the machine's moves (dnn_noise.relabel) and
          trained as in experiment 28's wait game. Layouts bits and
          true_bits (letter given; states still chosen).
  images: a disagreement filter. The distinct training images are split
          into 4 folds; a network built on 3 folds predicts the 4th; images
          whose (noisy) label disagrees with the prediction are dropped,
          and the network is built on the rest. Layouts nbr3_upstart
          (experiment 21, h = 3) and experiment 24's MLP, each filtered by
          its own kind of network.
The summary compares with experiment 29's unchanged rows (committed CSV).
Run from the repository root:  python experiments/exp30_noise_tolerant.py
Writes experiments/results/exp30_noise_tolerant.csv.
"""
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp16_or_of_ands as e16
import exp19_small_images as e19
import exp23_grid_game as e23
import exp28_acting_latency as e28
import exp29_noisy_labels as e29
import baseline_gd as G
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_noise as N

MAX_STATES = 3
FOLDS = 4
NOISE, SEEDS = e29.NOISE, e29.SEEDS
IMG_TASKS, IMG_SIZES, IMG_LAYOUTS = e29.IMG_TASKS, e29.IMG_SIZES, ["nbr3_upstart", "mlp"]
GAME_SIZES, GAME_LAYOUTS = e29.GAME_SIZES, e29.GAME_LAYOUTS
N_IMG = e29.N_IMG


# ---- game

def search_choice(layout):
    n_in, n_out = e28.N_IN, e28.N_OUT

    def choose(V):
        if layout == "true_bits":
            names = e28.e26.TRUE
            rules = [e23.FEATURE[n] for n in names]
            seqs = Lm.bit_streams(V, n_in, [lambda X, r=r: e23.U_fire(r, X) for r in rules], n_out)
            delta, out, err, n = N.learn_machine_heldout(seqs, len(rules), n_out, 3, MAX_STATES)
        else:
            idx, delta, out, err, n = N.choose_bits_search(V, n_in, e23.FEATURES, 3, n_out=n_out,
                                                           max_states=MAX_STATES)
            names, rules = [e23.FEATURES[i][0] for i in idx], [e23.FEATURES[i][1] for i in idx]
        bits = [lambda X, r=r: e23.U_fire(r, X) for r in rules]
        V[:] = N.relabel(V, n_in, n_out, 3, bits, delta, out)
        choose.err = err
        return names, rules, delta, out
    return choose


def game_run(layout, n_train, seed, p):
    V, tests, wrong, rng = e29.game_episodes(seed, n_train, p)
    clean, _, _, _ = e29.game_episodes(seed, n_train, 0.0)
    row = dict(family="game", task="key_door", n_train=n_train, seed=seed, layout=layout, noise=p,
               labels_flipped=wrong)
    choose = search_choice(layout)
    e29.game_net(V, tests, layout, seed, row, rng, choose=choose)
    n_in = e28.N_IN
    diff = [(a[3:, n_in:] != b[3:, n_in:]).any(1) for a, b in zip(V, clean)]
    row.update(train_disagreements=choose.err,
               relabelled_wrong_steps=int(sum(d.sum() for d in diff)))
    return row


# ---- images

def pattern_stream(X, labels, h):
    """One stream showing the patterns X in order, label of X[t] at t + h."""
    x = np.vstack([X, np.zeros((h, X.shape[1]), X.dtype)])
    s = np.zeros((len(x), X.shape[1] + 1), int)
    s[:, :X.shape[1]] = x
    s[h:, X.shape[1]] = labels
    return s


def fit_predict(kind, X, y, Xq, seed):
    """Predictions on Xq of a network of `kind` built on X, y."""
    if kind == "mlp":
        return G.MLP(X.shape[1], seed=seed).fit(X, y).predict(Xq)
    h = 3
    net, lay, _, _, _, _ = e29.build_net(kind, X, y, [pattern_stream(X, y, h)])
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    return np.array(e16.run_out(net, mid, F.extend(lay, pattern_stream(Xq, np.zeros(len(Xq), int), h)),
                                N_IMG))[h:h + len(Xq)]


def image_run(task, n_train, seed, kind, p):
    h = 3 if kind == "nbr3_upstart" else 2
    xs, V, tests, pats, flipped = e29.image_data(task, n_train, seed, p, h)
    f = e19.TASKS[task]
    X = np.array([np.frombuffer(b, dtype=xs[0].dtype) for b in pats])
    y = np.array(list(pats.values()))
    truth = np.array([f(x) for x in X])
    row = dict(family="images", task=task, n_train=n_train, seed=seed, layout=kind, noise=p,
               patterns_seen=len(pats), labels_flipped=int(flipped))
    t0 = time.perf_counter()
    fold = np.arange(len(X)) % FOLDS
    pred = np.zeros(len(X), int)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        if te.any():
            pred[te] = fit_predict(kind, X[tr], y[tr], X[te], seed) if tr.any() else y[te]
    keep = pred == y
    if not keep.any():
        keep[:] = True
    filter_secs = time.perf_counter() - t0
    row.update(dropped=int((~keep).sum()), dropped_flipped=int(((~keep) & (y != truth)).sum()),
               kept_flipped=int((keep & (y != truth)).sum()), filter_seconds=round(filter_secs, 4))
    Xk, yk = X[keep], y[keep]
    rng = np.random.default_rng(seed * 1000 + n_train)
    for _ in range(n_train):
        e19.images(task, rng, e19.T + 1)
    t0 = time.perf_counter()
    if kind == "mlp":
        net = G.MLP(X.shape[1], seed=seed).fit(Xk, yk)
        TX = np.concatenate([s[:e19.T + 1 - h, :N_IMG] for s in tests])
        unseen = np.array([x.tobytes() not in pats for x in TX])
        right = net.predict(TX) == np.array([f(x) for x in TX])
        row.update(unseen_acc_mid=float(right[unseen].mean()), seen_acc_mid=float(right[~unseen].mean()),
                   train_seconds=round(time.perf_counter() - t0, 4))
        return row
    net, lay, trained, helpers, bs, ts = e29.build_net(kind, Xk, yk, [pattern_stream(Xk, yk, h)])
    row.update(helpers=helpers, hidden_added=net.n - lay.n_visible, build_seconds=round(bs, 4),
               train_seconds=round(ts, 4))
    row.update(e16.evaluate(net, lay, tests, set(pats), rng, h, N_IMG))
    return row


def one_run(args):
    family, *rest = args
    return image_run(*rest) if family == "images" else game_run(*rest)


# ---- summary

def _read(name):
    path = os.path.join(H.ROOT, "experiments", "results", name)      # committed CSVs, not H.RESULTS
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


def _mean(g, col, fmt="{:.3f}"):
    vals = [r[col] for r in g if r.get(col, "") != ""]
    return fmt.format(np.mean(vals)) if vals else "–"


def summarise(rows):
    old = _read("exp29_noisy_labels.csv")
    img = [r for r in rows if r["family"] == "images"]
    if img:
        print("\nimages: mean accuracy on unseen patterns (midpoint), filtered (experiment 29 unfiltered), by streams\n")
        print("| layout | p | " + " | ".join(str(s) for s in IMG_SIZES) + " |")
        print("|" + "---|" * (len(IMG_SIZES) + 2))
        for kind in IMG_LAYOUTS:
            for p in NOISE:
                cells = []
                for s in IMG_SIZES:
                    g = [r for r in img if (r["layout"], r["noise"], r["n_train"]) == (kind, p, s)]
                    o = [r for r in old if (r["layout"], r["noise"], r["n_train"]) == (kind, p, s)]
                    cells.append(f"{_mean(g, 'unseen_acc_mid')} ({_mean(o, 'unseen_acc_mid')})")
                print(f"| {kind} | {p} | " + " | ".join(cells) + " |")
        print("\nimages at the largest size: patterns flipped / dropped / dropped that were flipped / "
              "flipped kept (means)\n")
        print("| layout | p | flipped | dropped | dropped flipped | flipped kept |")
        print("|---|---|---|---|---|---|")
        for kind in IMG_LAYOUTS:
            for p in NOISE:
                g = [r for r in img if (r["layout"], r["noise"], r["n_train"]) == (kind, p, IMG_SIZES[-1])]
                print(f"| {kind} | {p} | " + " | ".join(_mean(g, c, "{:.1f}") for c in
                                                     ("labels_flipped", "dropped", "dropped_flipped", "kept_flipped"))
                      + " |")
    game = [r for r in rows if r["family"] == "game"]
    if game:
        print("\ngame: runs with the true machine / mean unseen exact / mean laps (experiment 29: runs trained), "
              "by episodes\n")
        print("| layout | p | " + " | ".join(str(s) for s in GAME_SIZES) + " |")
        print("|" + "---|" * (len(GAME_SIZES) + 2))
        for lay in GAME_LAYOUTS:
            for p in NOISE:
                cells = []
                for s in GAME_SIZES:
                    g = [r for r in game if (r["layout"], r["noise"], r["n_train"]) == (lay, p, s)]
                    o = [r for r in old if (r["family"], r["layout"], r["noise"], r["n_train"]) == ("game", lay, p, s)]
                    cells.append(f"{sum(r['machine_correct'] for r in g)} / {_mean(g, 'unseen_exact_mid', '{:.2f}')} / "
                                 f"{_mean(g, 'laps_network', '{:.2f}')} ({sum(r['letter'] != 'none' for r in o)})")
                print(f"| {lay} | {p} | " + " | ".join(cells) + " |")
        print(f"\nletters chosen by bits at 32 episodes: "
              + ", ".join(sorted({r['letter'] for r in game if r['layout'] == 'bits' and r['n_train'] == 32})))
    print(f"\nmax seconds: filter {max([r.get('filter_seconds', 0) for r in rows]):.2f}, "
          f"train {max(r.get('train_seconds', 0) for r in rows):.2f}")


def main(seeds=SEEDS, img_sizes=IMG_SIZES, game_sizes=GAME_SIZES, noise=NOISE, img_tasks=IMG_TASKS):
    jobs = [("images", t, n, s, k, p) for t in img_tasks for n in img_sizes for s in seeds
            for k in IMG_LAYOUTS for p in noise]
    jobs += [("game", l, n, s, p) for l in GAME_LAYOUTS for n in game_sizes for s in seeds for p in noise]
    jobs.sort(key=lambda j: (-j[2], -j[-1]))
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    fam = ["images", "game"]
    rows.sort(key=lambda r: (fam.index(r["family"]), IMG_TASKS.index(r["task"]) if r["family"] == "images" else 0,
                             (IMG_LAYOUTS if r["family"] == "images" else GAME_LAYOUTS).index(r["layout"]),
                             r["noise"], r["n_train"], r["seed"]))
    H.write_csv("exp30_noise_tolerant.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
