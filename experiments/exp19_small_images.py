"""Experiment 19: 4 x 4 black-and-white images at depth 2 (ladder rung 2).

The input x(t) is a 4 x 4 image (16 inputs, row by row) and the output at
t + 2 is a fixed property of it; 0 at times 0 and 1. Tasks:
  hline:  some row is all black;      vline:  some column is all black;
  block:  some 2 x 2 square is all black;
  ink8:   at least 8 black pixels;    top:    more black in the top half than
          the bottom half;
  mirror: the image is its own left-right mirror image.
Images: each pixel black with probability d, d uniform in [0.15, 0.5] per
image (in [0.2, 0.8] for ink8 and top); for hline, vline, block and mirror,
half the images get the shape planted (a random row, column or square made
black, or the left half mirrored), so that both answers are common.
Layouts as in experiments 14 and 16 (h = 2, output reads only its helpers):
"lookup" (one indicator per positive image seen: the constant-answer
reference), "upstart" (experiment 14) and "cover" (experiment 16).
Training: 2 to 32 streams of 13 images, 10 seeds. Testing: 200 streams of 13
images; nearly every test image is unseen. "base rate" is the accuracy of the
better constant answer on the test images.
Run from the repository root:  python experiments/exp19_small_images.py
Writes experiments/results/exp19_images.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp14_boolean_functions as e14
import exp16_or_of_ands as e16
import dnn_fixed as F
import dnn_upstart as U

dnn = H.dnn
SIDE = 4
N_IN, T = SIDE * SIDE, 12
SIZES = [2, 4, 8, 16, 32]
SEEDS = range(10)
LAYOUTS = ["lookup", "upstart", "cover"]
TEST_STREAMS = 200


def _img(x):
    return np.asarray(x).reshape(SIDE, SIDE)


def _has_block(im):
    return any(im[i:i + 2, j:j + 2].all() for i in range(SIDE - 1) for j in range(SIDE - 1))


TASKS = {
    "hline": lambda x: int(_img(x).all(1).any()),
    "vline": lambda x: int(_img(x).all(0).any()),
    "block": lambda x: int(_has_block(_img(x))),
    "ink8": lambda x: int(np.sum(x) >= 8),
    "top": lambda x: int(_img(x)[:SIDE // 2].sum() > _img(x)[SIDE // 2:].sum()),
    "mirror": lambda x: int((_img(x) == _img(x)[:, ::-1]).all()),
}


def image(task, rng):
    lo, hi = (0.2, 0.8) if task in ("ink8", "top") else (0.15, 0.5)
    im = (rng.random((SIDE, SIDE)) < rng.uniform(lo, hi)).astype(int)
    if task in ("hline", "vline", "block", "mirror") and rng.random() < 0.5:
        k = rng.integers(0, SIDE)
        if task == "hline":
            im[k] = 1
        elif task == "vline":
            im[:, k] = 1
        elif task == "block":
            i, j = rng.integers(0, SIDE - 1, 2)
            im[i:i + 2, j:j + 2] = 1
        else:
            im[:, SIDE // 2:] = im[:, :SIDE // 2][:, ::-1]
    return im.reshape(-1)


def images(task, rng, n):
    return np.array([image(task, rng) for _ in range(n)])


def one_run(args):
    task, n_train, seed, kind = args
    f = TASKS[task]
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [images(task, rng, T + 1) for _ in range(n_train)]
    trng = np.random.default_rng(90_000 + seed)
    tests = [e16.stream(images(task, trng, T + 1), f, 2) for _ in range(TEST_STREAMS)]
    V = [e16.stream(x, f, 2) for x in xs]
    pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T - 1)}
    X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    t0 = time.perf_counter()
    rules = {"lookup": U.lookup, "upstart": U.build, "cover": U.build_cover}[kind](X, y)
    build_secs = time.perf_counter() - t0
    lay = U.make_layout(N_IN, rules)
    net = F.make_net(lay)
    net.L[N_IN, :N_IN + 1] = net.U[N_IN, :N_IN + 1] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    answers = np.concatenate([s[2:, N_IN] for s in tests])
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind, patterns_seen=len(pats),
               positives_seen=int(y.sum()), helpers=len(rules), hidden_added=net.n - nv,
               base_rate=float(max(answers.mean(), 1 - answers.mean())),
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [N_IN], rng)[0])
    row.update(e16.evaluate(net, lay, tests, set(pats), rng, 2, N_IN))
    return row


def summarise(rows):
    sel, mean = e14._sel, e14._mean
    print("\nMean accuracy on unseen images (midpoint weights), by number of training streams\n")
    print("| task | base rate | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 3))
    for task in TASKS:
        base = mean(sel(rows, task=task), "base_rate")
        for kind in LAYOUTS:
            g = sel(rows, task=task, layout=kind)
            print(f"| {task} | {base} | {kind} | " + " | ".join(mean(sel(g, n_train=s), "unseen_acc_mid") for s in SIZES) + " |")
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
    jobs.sort(key=lambda j: -j[1])          # big runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), r["n_train"], r["seed"], LAYOUTS.index(r["layout"])))
    H.write_csv("exp19_images.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
