"""Shared helpers for experiments 5-7: building, training and scoring the
baseline (dnn.py) and the causal variant (dnn_causal.py), with or without
an always-on input neuron."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import dnn  # noqa: E402
import dnn_causal  # noqa: E402

WEIGHT_MODES = ["lower", "upper", "random", "random", "random"]
EPS = 0.5
RESULTS = os.path.join(ROOT, "experiments", "results")


def with_bias(seqs):
    """Append an always-on column (the bias neuron is the last visible neuron)."""
    return [np.c_[np.asarray(s, int), np.ones(len(s), int)] for s in seqs]


def make_net(variant, n_visible, inputs=(), outputs=(), bias=False):
    """n_visible counts the task's neurons; with bias, one more always-on input
    neuron is appended at index n_visible."""
    inputs = set(inputs) | ({n_visible} if bias else set())
    cls = dnn.Net if variant == "baseline" else dnn_causal.CausalNet
    return cls(n_visible + int(bias), inputs, outputs)


def train(variant, net, V, **kw):
    if variant == "baseline":
        return dnn.train(net, V, EPS)
    return dnn_causal.train(net, V, EPS, **kw)


def hidden_counts(net, n_visible_before=None):
    """(feature neurons, chain neurons) in the network."""
    if isinstance(net, dnn_causal.CausalNet):
        return net.count(dnn_causal.FEATURE), net.count(dnn_causal.CHAIN)
    return 0, net.n - net.n_visible


def _run(net, A, v0, s):
    sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
    v0 = np.r_[v0, np.zeros(net.n - len(v0), int)]
    return np.array(dnn.run(A, net.I, v0, sets, len(s) - 1))


def score(net, starts, seqs, cols, rng, modes=WEIGHT_MODES):
    """Run each sequence from its start vector (zero-padded to all neurons)
    and compare the given visible columns at t >= 1 with the sequence.

    Returns (fraction of runs exact on those columns, fraction of values
    correct), averaged over weight samples."""
    exact, acc = [], []
    for mode in modes:
        A = dnn.sample_weights(net, rng, mode)
        for v0, s in zip(starts, seqs):
            out = _run(net, A, v0, s)[1:, cols]
            truth = np.asarray(s)[1:, cols]
            exact.append(bool((out == truth).all()))
            acc.append(float((out == truth).mean()))
    return float(np.mean(exact)), float(np.mean(acc))


def recalled(net, key, s, rng, trials=5):
    """True if the visible part of s replays from its (zero-padded) key for
    every one of `trials` weight samples (both corners, then random)."""
    cols = list(range(len(s[0])))
    modes = (["lower", "upper"] + ["random"] * trials)[:trials]
    for mode in modes:
        A = dnn.sample_weights(net, rng, mode)
        out = _run(net, A, key, s)
        if not (out[:, cols] == np.asarray(s)).all():
            return False
    return True


def zero_hidden_starts(seqs):
    return [np.asarray(s)[0] for s in seqs]


def write_csv(name, rows):
    import csv
    os.makedirs(RESULTS, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(rows, col):
    v = [float(r[col]) for r in rows if r.get(col, "") != ""]
    return float(np.mean(v)) if v else float("nan")


# ---- datasets, generated exactly as in experiments 1 and 4

def ca_dataset(rule, n_train, seed):
    """(training trajectories, unseen trajectories) as in experiment 1."""
    import exp1_cellular_automata as e1
    rng = np.random.default_rng(1000 * rule + 100 * n_train + seed)
    starts = e1.random_starts(rng, n_train)
    unseen = e1.random_starts(rng, e1.N_UNSEEN, exclude=starts)
    return [e1.trajectory(x, rule) for x in starts], [e1.trajectory(x, rule) for x in unseen]


def stream_dataset(task, n_train, seed):
    """(n_inputs, training sequences, unseen sequences) as in experiment 4."""
    import exp4_input_tasks as e4
    n_in, f = e4.TASKS[task]
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = e4.random_streams(rng, n_train, n_in)
    unseen_x = e4.random_streams(rng, e4.N_UNSEEN, n_in, exclude=xs)
    return n_in, [e4.make_sequence(x, f) for x in xs], [e4.make_sequence(x, f) for x in unseen_x]


def majority_baseline(seqs, col):
    y = np.concatenate([np.asarray(s)[1:, col] for s in seqs])
    return float(max(y.mean(), 1 - y.mean()))


def run_ca(variant, bias, rule, n_train, seed, rng, **kw):
    """Experiment 1 for one configuration. Returns a dict of metrics."""
    import time
    V, unseen = ca_dataset(rule, n_train, seed)
    n_cells = V[0].shape[1]
    if bias:
        V, unseen = with_bias(V), with_bias(unseen)
    net = make_net(variant, n_cells, bias=bias)
    t0 = time.perf_counter()
    trained = train(variant, net, V, **kw)
    secs = time.perf_counter() - t0
    feats, chain = hidden_counts(net)
    cells = list(range(n_cells))
    ex, acc = score(net, zero_hidden_starts(unseen), unseen, cells, rng)
    first, _ = score(net, zero_hidden_starts(unseen), [s[:2] for s in unseen], cells, rng)
    return dict(features=feats, chain_neurons=chain, train_seconds=round(secs, 4),
                train_replay_from_key=float(np.mean([recalled(net, k[0], s, rng) for k, s in zip(trained, V)])),
                train_starts_zero_hidden_exact=score(net, zero_hidden_starts(V), V, cells, rng)[0],
                unseen_exact=ex, unseen_first_step_exact=first, unseen_cell_acc=acc,
                all_zero_baseline_cell_acc=float(np.mean([(np.asarray(s)[1:, cells] == 0).mean() for s in unseen])))


def run_stream_task(variant, bias, task, n_train, seed, rng, **kw):
    """Experiment 4 for one configuration. Returns a dict of metrics."""
    import time
    n_in, V, unseen = stream_dataset(task, n_train, seed)
    out = n_in
    if bias:
        V, unseen = with_bias(V), with_bias(unseen)
    net = make_net(variant, n_in + 1, inputs=range(n_in), outputs={out}, bias=bias)
    t0 = time.perf_counter()
    trained = train(variant, net, V, **kw)
    secs = time.perf_counter() - t0
    feats, chain = hidden_counts(net)
    tr_ex, tr_acc = score(net, [k[0] for k in trained], V, [out], rng)
    un_ex, un_acc = score(net, zero_hidden_starts(unseen), unseen, [out], rng)
    return dict(features=feats, chain_neurons=chain, train_seconds=round(secs, 4),
                train_acc_from_key=tr_acc, train_exact_from_key=tr_ex,
                train_acc_zero_hidden=score(net, zero_hidden_starts(V), V, [out], rng)[1],
                unseen_acc=un_acc, unseen_exact=un_ex,
                unseen_majority_baseline=majority_baseline(unseen, out))
