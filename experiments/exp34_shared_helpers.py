"""Experiment 34: helpers shared across positions, on 4 x 4 and 8 x 8 images.

Experiment 21's image tasks (hline, vline, block, ink, top, mirror; ink is
"at least half the pixels black"), scaled to side S = 4 and 8 (16 and 64
inputs), with experiment 19's image generator, 12 images per stream, 2 to 32
streams, 10 seeds. Upstart and cover fit each helper to the whole image, so a
helper learned at one position does nothing at another. Layouts (all h = 2,
the output reads its helpers; limit 200 helpers or helper groups):
  upstart       experiment 21's upstart200 (side-4 rows must be reproduced);
  cover         experiment 21's cover200 (side-4 rows must be reproduced);
  cover_local   cover with each helper fitted inside one window: every
                2 x 2 and 3 x 3 square, row and column is tried
                (dnn_upstart.build_cover_local; window shapes chosen by
                Claude), the window covering the most positives wins;
  cover_shared  the same, each helper joined by all its translates (the
                same weights moved to every position where they fit) that
                are off for every negative training image.
Score: accuracy on unseen images (200 test streams, midpoint weights), as
experiments 19-21; also helpers and build time.
Run from the repository root:  python experiments/exp34_shared_helpers.py
Writes experiments/results/exp34_shared_helpers.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp16_or_of_ands as e16
import dnn_fixed as F
import dnn_upstart as U

T = 12
SIDES = [4, 8]
SIZES = [2, 4, 8, 16, 32]
SEEDS = range(10)
TEST_STREAMS = 200
LIMIT = 200
TASK_NAMES = ["hline", "vline", "block", "ink", "top", "mirror"]
LAYOUTS = ["upstart", "cover", "cover_local", "cover_shared"]


def task_fn(task, S):
    img = lambda x: np.asarray(x).reshape(S, S)
    return {
        "hline": lambda x: int(img(x).all(1).any()),
        "vline": lambda x: int(img(x).all(0).any()),
        "block": lambda x: int(any(img(x)[i:i + 2, j:j + 2].all() for i in range(S - 1) for j in range(S - 1))),
        "ink": lambda x: int(np.sum(x) >= S * S // 2),
        "top": lambda x: int(img(x)[:S // 2].sum() > img(x)[S // 2:].sum()),
        "mirror": lambda x: int((img(x) == img(x)[:, ::-1]).all()),
    }[task]


def image(task, rng, S):
    """Experiment 19's generator (same random draws) at side S."""
    lo, hi = (0.2, 0.8) if task in ("ink", "top") else (0.15, 0.5)
    im = (rng.random((S, S)) < rng.uniform(lo, hi)).astype(int)
    if task in ("hline", "vline", "block", "mirror") and rng.random() < 0.5:
        k = rng.integers(0, S)
        if task == "hline":
            im[k] = 1
        elif task == "vline":
            im[:, k] = 1
        elif task == "block":
            i, j = rng.integers(0, S - 1, 2)
            im[i:i + 2, j:j + 2] = 1
        else:
            im[:, S // 2:] = im[:, :S // 2][:, ::-1]
    return im.reshape(-1)


def images(task, rng, n, S):
    return np.array([image(task, rng, S) for _ in range(n)])


def one_run(args):
    S, task, n_train, seed, kind = args
    n_in = S * S
    f = task_fn(task, S)
    h = 2
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [images(task, rng, T + 1, S) for _ in range(n_train)]
    trng = np.random.default_rng(90_000 + seed)
    tests = [e16.stream(images(task, trng, T + 1, S), f, h) for _ in range(TEST_STREAMS)]
    V = [e16.stream(x, f, h) for x in xs]
    pats = {x[t].tobytes(): f(x[t]) for x in xs for t in range(T + 1 - h)}
    X = np.array([np.frombuffer(p, dtype=xs[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    t0 = time.perf_counter()
    if kind == "upstart":
        rules = U.build(X, y, H.EPS, max_helpers=LIMIT)
    elif kind == "cover":
        rules = U.build_cover(X, y, H.EPS, max_helpers=LIMIT)
    else:
        rules = U.build_cover_local(X, y, S, U.windows(S), share=kind == "cover_shared", eps=H.EPS,
                                    max_helpers=LIMIT)
    lay = U.make_layout(n_in, rules)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    net.L[n_in, :n_in + 1] = net.U[n_in, :n_in + 1] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    answers = np.concatenate([s[h:, n_in] for s in tests])
    row = dict(side=S, task=task, n_train=n_train, seed=seed, layout=kind, patterns_seen=len(pats),
               positives_seen=int(y.sum()), helpers=len(rules), hidden_added=net.n - nv,
               base_rate=float(max(answers.mean(), 1 - answers.mean())),
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [n_in], rng)[0])
    row.update(e16.evaluate(net, lay, tests, set(pats), rng, h, n_in))
    row["depth"] = h
    return row


def _mean(rows, col):
    v = [r[col] for r in rows if r[col] != ""]
    return f"{np.mean(v):.3f}" if v else "–"


def summarise(rows):
    sizes = sorted({r["n_train"] for r in rows})
    for S in sorted({r["side"] for r in rows}):
        print(f"\nSide {S}: mean unseen accuracy (midpoint) over the 6 tasks / mean helpers, by training streams\n")
        print("| layout | " + " | ".join(str(s) for s in sizes) + " |")
        print("|" + "---|" * (len(sizes) + 1))
        for kind in LAYOUTS:
            g = [r for r in rows if r["side"] == S and r["layout"] == kind]
            print(f"| {kind} | " + " | ".join(f"{_mean([r for r in g if r['n_train'] == n], 'unseen_acc_mid')} / "
                                             f"{_mean([r for r in g if r['n_train'] == n], 'helpers')}"
                                             for n in sizes) + " |")
        print(f"\nSide {S}, {sizes[-1]} streams: unseen accuracy by task (base rate)\n")
        print("| layout | " + " | ".join(TASK_NAMES) + " |")
        print("|" + "---|" * (len(TASK_NAMES) + 1))
        last = [r for r in rows if r["side"] == S and r["n_train"] == sizes[-1]]
        print("| base rate | " + " | ".join(_mean([r for r in last if r["task"] == t and r["layout"] == "upstart"],
                                                  "base_rate") for t in TASK_NAMES) + " |")
        for kind in LAYOUTS:
            print(f"| {kind} | " + " | ".join(_mean([r for r in last if r["task"] == t and r["layout"] == kind],
                                                   "unseen_acc_mid") for t in TASK_NAMES) + " |")
    print(f"\nhidden neurons added: {sum(r['hidden_added'] for r in rows)}; training streams exact without key: "
          f"{sum(r['train_exact_without_key'] == 1.0 for r in rows)} of {len(rows)}")
    for kind in LAYOUTS:
        g = [r for r in rows if r["layout"] == kind]
        print(f"{kind}: max build seconds {max(r['build_seconds'] for r in g):.1f}, "
              f"max helpers {max(r['helpers'] for r in g)}")


def main(sides=SIDES, sizes=SIZES, seeds=SEEDS, tasks=TASK_NAMES):
    jobs = [(S, t, n, s, k) for S in sides for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    jobs.sort(key=lambda j: (-j[0] * j[2], not j[4].startswith("cover_")))     # big runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (r["side"], TASK_NAMES.index(r["task"]), r["n_train"], r["seed"],
                             LAYOUTS.index(r["layout"])))
    H.write_csv("exp34_shared_helpers.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
