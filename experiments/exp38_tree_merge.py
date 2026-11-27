"""Experiment 38: memorising without keys (input-driven clock neurons), then sleep by merging.

Wake (dnn_clock.wake_tree, epsilon = 1e-4): if a neuron does not fit, add one
neuron for each distinct input history x(0..t) of the training streams, on
at time t + 1 in exactly the streams with that history. These neurons may be
excited by the inputs (condition (D) is dropped for them), so a stream
replays without a key. Depth 2: the output at t + 2 answers the inputs up to
t; a start input is on at time 0 only.

Sleep, four ways:
  merge-meet / merge-sum: dnn_clock.sleep_merge for delta in DELTAS, with the
      weights out of a merged neuron met (never fire together) or added;
  forced-merge: the grouping a correct machine needs is given (oracle):
      neurons are grouped by (machine state before the last input, last
      input), and each group is merged by meeting boxes (no LP);
  forced-resolve: the same grouping, but the boxes are found again by linear
      programs on the training streams (a fresh network: visible neurons
      plus one neuron per group).
Tasks of experiment 4 at depth 2: prev_input (state: none), parity (state:
parity so far), adder (state: carry). 8 and 32 training streams, 100
unseen, 5 seeds. Scores: hidden neurons; training and unseen streams run
without keys (hidden neurons at 0): whole streams right, outputs right.
"""
import copy
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from dnn import INF, Net, feas, subbox  # noqa: E402
from dnn_clock import merge, sleep_merge, wake_tree  # noqa: E402

TASKS = ["prev_input", "parity", "adder"]
N_TRAIN = [8, 32]
SEEDS = range(5)
DELTAS = [0.0, 1e-4, 1.0, float("inf")]
EPS = 1e-4


def state(task, x):
    """State of the task's machine after the inputs x (rows x(0..t-1))."""
    if task == "parity":
        return int(x[:, 0].sum() % 2)
    if task == "adder":
        carry = 0
        for a, b in x:
            carry = (int(a) + int(b) + carry) // 2
        return carry
    return 0


def depth2(seqs, n_in):
    """[start, inputs, output delayed by one step]: the output at t + 2
    answers the inputs up to t."""
    out = []
    for s in seqs:
        s = np.asarray(s)
        start = (np.arange(len(s)) == 0).astype(int)[:, None]
        out.append(np.hstack([start, s[:, :n_in], np.r_[0, s[:-1, n_in]][:, None]]))
    return out


def group_of(task, x):
    """Group of the neuron for history x(0..t): (state before x(t), x(t))."""
    return (state(task, x[:-1]),) + tuple(int(v) for v in x[-1])


def scores(net, tr, un, out, seed):
    rng = np.random.default_rng(seed)
    st = H.zero_hidden_starts
    te, ta = H.score(net, st(tr), tr, [out], rng)
    ue, ua = H.score(net, st(un), un, [out], rng)
    return {"hidden": net.n - net.n_visible, "train_exact": round(te, 4), "train_acc": round(ta, 4),
            "unseen_exact": round(ue, 4), "unseen_acc": round(ua, 4)}


def forced(task, net, ext, tr, n_in, how):
    """Merge the tree neurons by the oracle grouping, by meeting boxes
    ('merge') or by solving linear programs again ('resolve')."""
    nv = net.n_visible
    xs = slice(1, 1 + n_in)
    hist = {}
    for s, e in zip(tr, ext):
        for t in range(1, len(e)):
            for h in np.flatnonzero(e[t, nv:]):
                hist[nv + int(h)] = group_of(task, s[:t, xs])
    if how == "merge":
        L, U, K = net.L.copy(), net.U.copy(), np.array([e[0] for e in ext])
        labels = [None] * nv + [hist[i] for i in range(nv, net.n)]
        while True:
            pair = next(((i, j) for i in range(nv, len(labels)) for j in range(i + 1, len(labels))
                         if labels[i] == labels[j]), None)
            if pair is None:
                break
            L, U, K = merge(L, U, K, *pair, combine="meet")
            labels = [labels[m] for m in range(len(labels)) if m not in pair] + [labels[pair[0]]]
        m = Net(nv, inputs=net.I, outputs=net.O)
        m.L, m.U, m.n, m.c = L, U, L.shape[0], [0] * nv + [1] * (L.shape[0] - nv)
        return m, True
    groups = sorted(set(hist.values()))
    seqs = []
    for s in tr:
        cols = np.zeros((len(s), len(groups)), int)
        for t in range(1, len(s)):
            cols[t, groups.index(group_of(task, s[:t, xs]))] = 1
        seqs.append(np.hstack([s, cols]))
    N = nv + len(groups)
    m = Net(nv, inputs=net.I, outputs=net.O)
    m.L, m.U, m.n, m.c = np.full((N, N), -INF), np.full((N, N), INF), N, [0] * nv + [1] * len(groups)
    targets = [g for g in range(nv) if g not in net.I] + list(range(nv, N))
    ok = all(feas(seqs, g, m.L, m.U, EPS) for g in targets)
    if ok:
        for g in targets:
            m.L[g], m.U[g] = subbox(seqs, g, m.L, m.U, EPS)
    return m, ok


def job(args):
    task, n, seed, deltas = args
    n_in, tr, un = H.stream_dataset(task, n, seed)
    tr, un = depth2(tr, n_in), depth2(un, n_in)
    out = n_in + 1
    net = Net(n_in + 2, inputs=set(range(n_in + 1)), outputs={out})
    ext, nodes, fallback = wake_tree(net, tr, EPS)
    base = {"task": task, "n_train": n, "seed": seed, "tree_neurons": nodes, "fallback": fallback}
    rows = [dict(base, sleep="wake", delta="", feasible=True, **scores(net, tr, un, out, seed))]
    keys = [e[0] for e in ext]
    for comb in ["meet", "sum"]:
        for d in deltas:
            m = copy.deepcopy(net)
            sleep_merge(m, keys, d, comb)
            rows.append(dict(base, sleep=f"merge-{comb}", delta=d, feasible=True, **scores(m, tr, un, out, seed)))
    for how in ["merge", "resolve"]:
        m, ok = forced(task, net, ext, tr, n_in, how)
        rows.append(dict(base, sleep=f"forced-{how}", delta="", feasible=ok, **scores(m, tr, un, out, seed)))
    return rows


def main(tasks=tuple(TASKS), sizes=N_TRAIN, seeds=SEEDS, deltas=DELTAS):
    jobs = [(t, n, s, deltas) for t in tasks for n in sizes for s in seeds]
    jobs.sort(key=lambda j: -j[1])                       # big jobs first
    with Pool() as pool:
        rows = [r for rs in pool.map(job, jobs, chunksize=1) for r in rs]
    rows.sort(key=lambda r: (TASKS.index(r["task"]), r["n_train"], r["seed"]))
    H.write_csv("exp38_tree_merge.csv", rows)
    return rows


if __name__ == "__main__":
    main()
