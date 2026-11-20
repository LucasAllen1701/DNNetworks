"""Experiment 29: training labels with noise.

Path to Doom, step 3. Every training set so far was exact. Here a fraction
p = 0, 0.05, 0.1, 0.2 of the training labels is wrong; test labels are
exact. Measurement only, no change to the algorithms.
  images: experiment 19's six 4 x 4 image tasks, 2 to 32 streams. The label
          of a training image is flipped when a hash of (image, seed) is
          below p (dnn_noise.noisy): the same image always gets the same
          label, and the flipped images at p = 0.05 are among those at 0.1.
          Layouts: cover (experiment 19, h = 2) and nbr3_upstart
          (experiment 21, h = 3, 200 helpers at most), and experiment 24's
          MLP (64 tanh units, Adam) on the same noisy patterns.
          At p = 0 the rows equal experiments 19/21 (and 24 for the MLP).
  game:   experiment 28's "wait" game (each screen shown 3 network steps),
          layouts bits and true_bits, 8 to 32 episodes. With probability p
          the expert's move for a screen is replaced by one of the 3 other
          moves (all 3 network steps of that screen). At p = 0 the rows equal
          experiment 28's wait rows.
10 seeds.
Run from the repository root:  python experiments/exp29_noisy_labels.py
Writes experiments/results/exp29_noisy_labels.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp16_or_of_ands as e16
import exp19_small_images as e19
import exp21_more_helpers_images as e21
import exp23_grid_game as e23
import exp26_imitate_expert as e26
import exp28_acting_latency as e28
import baseline_gd as G
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M
import dnn_noise as N
import dnn_upstart as U

NOISE = [0.0, 0.05, 0.1, 0.2]
SEEDS = range(10)
IMG_TASKS, IMG_SIZES, IMG_LAYOUTS = list(e19.TASKS), e19.SIZES, ["cover", "nbr3_upstart", "mlp"]
GAME_SIZES, GAME_LAYOUTS = e28.SIZES, e28.LAYOUTS
T, N_IMG = e19.T, e19.N_IN
MOVE_LIST = [(1, 1), (1, 0), (0, 1), (0, 0)]


# ---- images

def image_data(task, n_train, seed, p, h):
    """(training streams with noisy labels, test streams, patterns {bytes: noisy label}, flipped seen)."""
    f = e19.TASKS[task]
    fn = N.noisy(f, p, seed)
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [e19.images(task, rng, T + 1) for _ in range(n_train)]
    trng = np.random.default_rng(90_000 + seed)
    tests = [e16.stream(e19.images(task, trng, T + 1), f, h) for _ in range(e19.TEST_STREAMS)]
    V = [e16.stream(x, fn, h) for x in xs]
    pats = {x[t].tobytes(): fn(x[t]) for x in xs for t in range(T + 1 - h)}
    flipped = sum(fn(np.frombuffer(b, dtype=xs[0].dtype)) != f(np.frombuffer(b, dtype=xs[0].dtype)) for b in pats)
    return xs, V, tests, pats, flipped


def build_net(kind, X, y, V):
    """Experiment 19's cover (h = 2) or experiment 21's nbr3_upstart (h = 3)
    network on patterns X, y, trained on V. Returns (net, layout, trained, helpers, build_secs, train_secs)."""
    t0 = time.perf_counter()
    if kind == "cover":
        rules = U.build_cover(X, y)
        lay, allowed = U.make_layout(N_IMG, rules), None
    else:
        layer1 = e21.LAYER1["nbr3"]
        rules = U.build(U.layer1_values(X, layer1), y, H.EPS, max_helpers=e21.LIMIT)
        lay, allowed = U.make_layout_h3(N_IMG, layer1, rules)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    if allowed is None:
        net.L[N_IMG, :N_IMG + 1] = net.U[N_IMG, :N_IMG + 1] = 0.0
    else:
        blocked = [j for j in range(lay.n_visible) if j not in allowed]
        net.L[N_IMG, blocked] = net.U[N_IMG, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    return net, lay, trained, len(rules), build_secs, time.perf_counter() - t0


def image_run(task, n_train, seed, kind, p):
    h = 3 if kind == "nbr3_upstart" else 2
    xs, V, tests, pats, flipped = image_data(task, n_train, seed, p, h)
    X = np.array([np.frombuffer(b, dtype=xs[0].dtype) for b in pats])
    y = np.array(list(pats.values()))
    row = dict(family="images", task=task, n_train=n_train, seed=seed, layout=kind, noise=p,
               patterns_seen=len(pats), labels_flipped=int(flipped))
    rng = np.random.default_rng(seed * 1000 + n_train)          # e19/e21 reuse their data rng for scoring
    for _ in range(n_train):
        e19.images(task, rng, T + 1)
    if kind == "mlp":
        f = e19.TASKS[task]
        t0 = time.perf_counter()
        net = G.MLP(X.shape[1], seed=seed).fit(X, y)
        TX = np.concatenate([s[:T + 1 - h, :N_IMG] for s in tests])
        unseen = np.array([x.tobytes() not in pats for x in TX])
        right = net.predict(TX) == np.array([f(x) for x in TX])
        row.update(train_acc=float((net.predict(X) == y).mean()),
                   unseen_acc_mid=float(right[unseen].mean()), seen_acc_mid=float(right[~unseen].mean()),
                   train_seconds=round(time.perf_counter() - t0, 4))
        return row
    net, lay, trained, helpers, bs, ts = build_net(kind, X, y, V)
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    row.update(helpers=helpers, hidden_added=net.n - nv, build_seconds=round(bs, 4), train_seconds=round(ts, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [N_IMG], rng)[0])
    row.update(e16.evaluate(net, lay, tests, set(pats), rng, h, N_IMG))
    return row


# ---- game

def game_episodes(seed, n, p, rep=3, h_out=3):
    """Experiment 28's wait episodes (first n of 32) and tests; in training,
    each screen's move is replaced by another move with probability p."""
    rng = np.random.default_rng(seed)
    V = [e28.episode(rng, e28.T + 1, rep, h_out) for _ in range(max(GAME_SIZES))][:n]
    trng = np.random.default_rng(60_000 + seed)
    tests = [e28.episode(trng, e28.T + 1, rep, h_out) for _ in range(e28.TEST_STREAMS)]
    nrng = np.random.default_rng(20_000 + seed)
    noise = [(nrng.random(e28.T + 1), nrng.integers(1, 4, e28.T + 1)) for _ in range(max(GAME_SIZES))][:n]
    wrong = 0
    for s, (u, r) in zip(V, noise):
        for k in range(e28.T + 1):
            ticks = [t + h_out for t in range(k * rep, (k + 1) * rep) if t + h_out < len(s)]
            if u[k] < p and ticks:
                cur = tuple(s[ticks[0], e28.N_IN:])
                s[ticks, e28.N_IN:] = MOVE_LIST[(MOVE_LIST.index(cur) + r[k]) % 4]
                wrong += 1
    return V, tests, wrong, rng


def game_net(V, tests, layout, seed, row, rng, choose=None, h_out=3, rep=3):
    """Experiment 28's wait pipeline on (possibly noisy) episodes (rng: the
    data rng after the episodes, as experiment 28 scores recall). `choose`
    (V) -> (names, rules, delta, out) or None replaces the exact letter and
    machine search. Fills `row`."""
    n_in, n_out = e28.N_IN, e28.N_OUT
    true_delta, true_out = e26.machine()
    t0 = time.perf_counter()
    if choose is not None:
        found = choose(V)
    elif layout == "true_bits":
        rules = [e23.FEATURE[n] for n in e26.TRUE]
        try:
            delta, out = Lm.learn_machine_multi(
                Lm.bit_streams(V, n_in, [lambda X, r=r: e23.U_fire(r, X) for r in rules], n_out), 2, n_out, h_out)
            found = (e26.TRUE, rules, delta, out)
        except ValueError:
            found = None
    else:
        f = Lm.choose_bits(V, n_in, e23.FEATURES, h_out, n_out=n_out)
        found = None if f is None else ([e23.FEATURES[i][0] for i in f[0]], [e23.FEATURES[i][1] for i in f[0]],
                                        f[1], f[2])
    if found is None:
        row.update(letter="none", machine_correct=False)
        return None
    names, rules, delta, out = found
    lay, index = Lm.make_layout_bits(n_in, rules, M.complete(delta, len(rules)), n_out=n_out)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in index.values()]
    for o in range(n_in, n_in + n_out):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    key = lambda x: tuple(int(e23.U_fire(r, np.array([x]))[0]) for r in rules)
    true_key = lambda x: tuple(int(e23.U_fire(e23.FEATURE[n], np.array([x]))[0]) for n in e26.TRUE)
    exact = []
    for s in tests:
        ext = F.extend(lay, s)
        v0 = np.zeros(net.n, int)
        v0[:ext.shape[1]] = ext[0]
        sets = [{i for i in net.I if ext[t][i]} for t in range(len(ext))]
        pred = np.array(dnn.run(mid, net.I, v0, sets, len(ext) - 1))[:, n_in:n_in + n_out]
        exact.append((pred[h_out:] == s[h_out:, n_in:]).all(1).all())
    game_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=e28.GAMES)
    laps = [e26.play(e28.wait_policy(net, lay, mid, rep), np.random.default_rng(g)) for g in game_seeds]
    row.update(letter="+".join(names), states_found=len(delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key, e23.SCREENS),
               hidden_added=net.n - lay.n_visible, build_seconds=round(build_secs, 4),
               train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, list(range(n_in, n_in + n_out)), rng)[0],
               unseen_exact_mid=float(np.mean(exact)), laps_network=float(np.mean(laps)))
    return net


def game_run(layout, n_train, seed, p):
    V, tests, wrong, rng = game_episodes(seed, n_train, p)
    row = dict(family="game", task="key_door", n_train=n_train, seed=seed, layout=layout, noise=p,
               labels_flipped=wrong)
    game_net(V, tests, layout, seed, row, rng)
    return row


def one_run(args):
    family, *rest = args
    return image_run(*rest) if family == "images" else game_run(*rest)


# ---- summary

def _cell(g, col, fmt="{:.3f}"):
    vals = [r[col] for r in g if r.get(col, "") != ""]
    return fmt.format(np.mean(vals)) if vals else "–"


def summarise(rows):
    img = [r for r in rows if r["family"] == "images"]
    if img:
        print("\nimages: mean accuracy on unseen patterns (midpoint), all tasks and seeds, by streams\n")
        print("| layout | p | " + " | ".join(str(s) for s in IMG_SIZES) + " |")
        print("|" + "---|" * (len(IMG_SIZES) + 2))
        for kind in IMG_LAYOUTS:
            for p in NOISE:
                g = [r for r in img if r["layout"] == kind and r["noise"] == p]
                print(f"| {kind} | {p} | " + " | ".join(_cell([r for r in g if r["n_train"] == s], "unseen_acc_mid")
                                                     for s in IMG_SIZES) + " |")
        print("\nimages at the largest size: helpers / hidden neurons added (mean) / seen acc vs true labels\n")
        print("| layout | p | helpers | hidden added | seen acc |")
        print("|---|---|---|---|---|")
        for kind in IMG_LAYOUTS:
            for p in NOISE:
                g = [r for r in img if r["layout"] == kind and r["noise"] == p and r["n_train"] == IMG_SIZES[-1]]
                print(f"| {kind} | {p} | {_cell(g, 'helpers', '{:.1f}')} | {_cell(g, 'hidden_added', '{:.1f}')} "
                      f"| {_cell(g, 'seen_acc_mid')} |")
    game = [r for r in rows if r["family"] == "game"]
    if game:
        print("\ngame: runs with a consistent letter / true machine / mean unseen exact (trained runs) / "
              "mean laps (trained runs), by episodes\n")
        print("| layout | p | " + " | ".join(str(s) for s in GAME_SIZES) + " |")
        print("|" + "---|" * (len(GAME_SIZES) + 2))
        for lay in GAME_LAYOUTS:
            for p in NOISE:
                cells = []
                for s in GAME_SIZES:
                    g = [r for r in game if (r["layout"], r["noise"], r["n_train"]) == (lay, p, s)]
                    t = [r for r in g if r["letter"] != "none"]
                    cells.append(f"{len(t)} / {sum(r['machine_correct'] for r in g)} / "
                                 f"{_cell(t, 'unseen_exact_mid', '{:.2f}')} / {_cell(t, 'laps_network', '{:.2f}')}")
                print(f"| {lay} | {p} | " + " | ".join(cells) + " |")
    done = [r for r in rows if "train_seconds" in r]
    print(f"\nmax seconds: build {max(r.get('build_seconds', 0) for r in done):.2f}, "
          f"train {max(r['train_seconds'] for r in done):.2f}")


def main(seeds=SEEDS, img_sizes=IMG_SIZES, game_sizes=GAME_SIZES, noise=NOISE, img_tasks=IMG_TASKS):
    jobs = [("images", t, n, s, k, p) for t in img_tasks for n in img_sizes for s in seeds
            for k in IMG_LAYOUTS for p in noise]
    jobs += [("game", l, n, s, p) for l in GAME_LAYOUTS for n in game_sizes for s in seeds for p in noise]
    jobs.sort(key=lambda j: (-j[2] if j[0] == "images" else -j[2], -j[-1]))      # big, noisy runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    fam = ["images", "game"]
    rows.sort(key=lambda r: (fam.index(r["family"]), IMG_TASKS.index(r["task"]) if r["family"] == "images" else 0,
                             (IMG_LAYOUTS if r["family"] == "images" else GAME_LAYOUTS).index(r["layout"]),
                             r["noise"], r["n_train"], r["seed"]))
    H.write_csv("exp29_noisy_labels.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
