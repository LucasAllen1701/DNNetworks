"""Experiment 14: Boolean functions of six inputs at depth 2 (ladder rung 1).

The output at time t + 2 is f(x(t)) for a fixed Boolean function f of the
six inputs; it is 0 at times 0 and 1. The network is layered: the output's
weights from the inputs and from itself are fixed at 0 (x(t + 1) and the
last answer carry nothing about this answer). Three layouts, all trained by `dnn.py` unchanged:
  "baseline": the always-on neuron only; TRAIN adds predictor chains.
  "lookup":   one fixed helper per positive input pattern seen in training.
  "upstart":  fixed helpers built from the training data by dnn_upstart.py.
Training: 1 to 16 streams of 13 vectors (11 labelled patterns each), 10
seeds. Testing: 200 random streams of 13 vectors, run with every hidden and
fixed neuron at 0 and no key. Each output from t = 2 on is scored, split by
whether its input pattern x(t - 2) was seen in training. There are 64
patterns; "random" is a random function, where nothing can generalise.
Run from the repository root:  python experiments/exp14_boolean_functions.py
Writes experiments/results/exp14_boolean.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import dnn_fixed as F
import dnn_upstart as U

dnn = H.dnn
N_IN, T = 6, 12
SIZES = [1, 2, 4, 8, 16]
SEEDS = range(10)
LAYOUTS = ["baseline", "lookup", "upstart"]
TEST_STREAMS = 200
MODES = ["lower", "upper", "random", "random", "random"]

_RANDOM = np.random.default_rng(1234).integers(0, 2, 2 ** N_IN)
TASKS = {
    "majority": lambda x: int(x.sum() >= 3),
    "and3": lambda x: int(x[0] & x[1] & x[2]),
    "xor2": lambda x: int(x[0] ^ x[1]),
    "dnf3": lambda x: int((x[0] & x[1]) | (x[2] & x[3]) | (x[4] & x[5])),
    "mux": lambda x: int(x[2 + 2 * x[0] + x[1]]),
    "equal3": lambda x: int((x[:3] == x[3:]).all()),
    "parity6": lambda x: int(x.sum() % 2),
    "random": lambda x: int(_RANDOM[int("".join(map(str, x)), 2)]),
}


def stream(x, f):
    """Task neurons: inputs x (T + 1 rows), output f(x(t - 2)) at t >= 2."""
    out = np.zeros(len(x), int)
    out[2:] = [f(v) for v in x[:-2]]
    return np.c_[x, out]


def run_out(net, A, s):
    """Output column of a run started with every non-task neuron at 0."""
    v0 = np.zeros(net.n, int)
    v0[:s.shape[1]] = s[0]
    sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
    return np.array(dnn.run(A, net.I, v0, sets, len(s) - 1))[:, N_IN]


def evaluate(net, lay, tests, seen, rng):
    """Accuracy at t >= 2 on seen and unseen patterns, for the midpoint weights
    and averaged over sampled weights."""
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    res = {}
    for label, mats in (("mid", [mid]), ("sampled", [dnn.sample_weights(net, rng, m) for m in MODES])):
        hit = {True: [], False: []}
        for A in mats:
            for s in tests:
                pred = run_out(net, A, F.extend(lay, s))
                for t in range(2, len(s)):
                    hit[s[t - 2, :N_IN].tobytes() in seen].append(pred[t] == s[t, N_IN])
        res[f"seen_acc_{label}"] = float(np.mean(hit[True])) if hit[True] else ""
        res[f"unseen_acc_{label}"] = float(np.mean(hit[False])) if hit[False] else ""
    return res


def one_run(args):
    task, n_train, seed, kind = args
    f = TASKS[task]
    rng = np.random.default_rng(seed)
    V = [stream(rng.integers(0, 2, (T + 1, N_IN)), f) for _ in range(n_train)]
    trng = np.random.default_rng(50_000 + seed)
    tests = [stream(trng.integers(0, 2, (T + 1, N_IN)), f) for _ in range(TEST_STREAMS)]
    pats = {s[t, :N_IN].tobytes(): s[t + 2, N_IN] for s in V for t in range(T - 1)}
    X = np.array([np.frombuffer(p, dtype=V[0].dtype) for p in pats])
    y = np.array(list(pats.values()))
    t0 = time.perf_counter()
    rules = {"baseline": [], "lookup": U.lookup(X, y), "upstart": U.build(X, y, H.EPS)}[kind]
    build_secs = time.perf_counter() - t0
    lay = U.make_layout(N_IN, rules)
    net = F.make_net(lay)
    # layered: the output reads neither x(t + 1) nor its own last value
    net.L[N_IN, :N_IN + 1] = net.U[N_IN, :N_IN + 1] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    nv = lay.n_visible
    Vx = [s[:, :nv] for s in trained]
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind,
               patterns_seen=len(pats), helpers=len(rules), hidden_added=net.n - nv,
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               train_exact_without_key=H.score(net, H.zero_hidden_starts(Vx), Vx, [N_IN], rng)[0],
               train_exact_with_key=H.score(net, [s[0] for s in trained], trained, [N_IN], rng)[0])
    row.update(evaluate(net, lay, tests, set(pats), rng))
    return row


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _mean(rows, col):
    vals = [r[col] for r in rows if r[col] != ""]
    return f"{np.mean(vals):.3f}" if vals else "–"


def summarise(rows):
    print("\nMean accuracy on unseen input patterns (midpoint weights), by number of training streams\n")
    print("| task | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for task in TASKS:
        for kind in LAYOUTS:
            g = _sel(rows, task=task, layout=kind)
            print(f"| {task} | {kind} | " + " | ".join(_mean(_sel(g, n_train=s), "unseen_acc_mid") for s in SIZES) + " |")
    print("\nTotals over the 8 tasks\n")
    print("| layout | training streams | runs | patterns seen (mean) | helpers (mean) | hidden neurons added "
          "| seen acc (mid) | unseen acc (mid) | unseen acc (sampled) | training streams exact with key "
          "| without key |")
    print("|" + "---|" * 11)
    for kind in LAYOUTS:
        for s in SIZES:
            g = _sel(rows, layout=kind, n_train=s)
            print(f"| {kind} | {s} | {len(g)} | {_mean(g, 'patterns_seen')} | {_mean(g, 'helpers')} "
                  f"| {sum(r['hidden_added'] for r in g)} | {_mean(g, 'seen_acc_mid')} | {_mean(g, 'unseen_acc_mid')} "
                  f"| {_mean(g, 'unseen_acc_sampled')} | {sum(r['train_exact_with_key'] == 1.0 for r in g)} "
                  f"| {sum(r['train_exact_without_key'] == 1.0 for r in g)} |")
    up = _sel(rows, layout="upstart")
    print(f"\nupstart: runs that needed hidden neurons from TRAIN: {sum(r['hidden_added'] > 0 for r in up)} of {len(up)}")
    print(f"max seconds: build {max(r['build_seconds'] for r in rows):.2f}, train {max(r['train_seconds'] for r in rows):.2f}")


def main(sizes=SIZES, seeds=SEEDS, tasks=TASKS):
    jobs = [(t, n, s, k) for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=2)
    H.write_csv("exp14_boolean.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
