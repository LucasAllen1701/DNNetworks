"""Experiment 4: input-driven tasks.

Visible neurons: the input neuron(s) first, then one output neuron. A stream
x_0, ..., x_T of inputs gives the training sequence v(t) = (x_t, y_t), where
y_t = f(x_0 ... x_{t-1}) is the target for the input read so far (so the
output at time t+1 answers for the string up to and including x_t, one step
after it is read, which is the earliest a single synapse layer can respond).

Tasks:
- prev_input: y_t = x_{t-1} (a separable control);
- parity: y_t = x_0 xor ... xor x_{t-1};
- adder: two inputs a, b (least significant bit first); y_t is sum bit t-1
  of a + b, which depends on the carry;
- tomita1 ... tomita7: y_t = 1 if x_0 ... x_{t-1} is in the Tomita grammar.

For each task and number of training streams, all training streams are trained
in one round. Then:
- training accuracy: each training stream run from its key;
- unseen accuracy: 100 unseen input streams, run from the true visible state
  at time 0 with every hidden neuron at 0;
- training streams with hidden neurons at 0, to show what the keys contribute.
Accuracy is the fraction of output values y_1 ... y_T predicted correctly
(y_0 is part of the given initial state), averaged over 5 weight matrices.

The script also records whether the next output is a function of the visible
state (inputs and output) at all, and if so whether it is linearly separable
without a bias at margin eps (FEAS on the visible neurons with an unbounded
box). A network run with every hidden neuron at 0 can only be right on every
stream if both hold.

Run from the repository root:  python experiments/exp4_input_tasks.py
Writes experiments/results/exp4_input_tasks.csv.
"""
import csv
import itertools
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dnn import Net, feas, run, sample_weights, train  # noqa: E402

T = 12                          # transitions per stream
N_TRAIN = [8, 32, 128]
N_UNSEEN = 100
SEEDS = range(3)
WEIGHT_MODES = ["lower", "upper", "random", "random", "random"]
EPS = 0.5
OUT = os.path.join(ROOT, "experiments", "results", "exp4_input_tasks.csv")


# ---- tasks: each maps an input stream (array of shape (T+1, n_in)) to y_0..y_T

def _prefix_task(accept):
    def f(x):
        bits = [int(b) for b in x[:, 0]]
        return np.array([int(accept(bits[:t])) for t in range(len(bits))])
    return f


def _runs(bits):
    return [(b, len(list(g))) for b, g in itertools.groupby(bits)]


def tomita3(bits):
    """Reject iff an odd-length run of 1s is immediately followed by an
    odd-length run of 0s (runs are maximal within the string read so far)."""
    r = _runs(bits)
    return not any(a[0] == 1 and a[1] % 2 and b[0] == 0 and b[1] % 2
                   for a, b in zip(r, r[1:]))


TOMITA = {
    1: lambda s: all(b == 1 for b in s),                                   # 1*
    2: lambda s: len(s) % 2 == 0 and all(b == 1 - i % 2 for i, b in enumerate(s)),  # (10)*
    3: tomita3,
    4: lambda s: "000" not in "".join(map(str, s)),
    5: lambda s: s.count(0) % 2 == 0 and s.count(1) % 2 == 0,
    6: lambda s: (s.count(0) - s.count(1)) % 3 == 0,
    7: lambda s: re.fullmatch("0*1*0*1*", "".join(map(str, s))) is not None,
}


def prev_input(x):
    return np.r_[0, x[:-1, 0]]


def parity(x):
    return np.r_[0, np.cumsum(x[:-1, 0]) % 2]


def adder(x):
    y, carry = [0], 0
    for a, b in x[:-1]:
        total = int(a) + int(b) + carry
        y.append(total % 2)
        carry = total // 2
    return np.array(y)


TASKS = {"prev_input": (1, prev_input), "parity": (1, parity), "adder": (2, adder)}
TASKS.update({f"tomita{i}": (1, _prefix_task(f)) for i, f in TOMITA.items()})


# ---- helpers

def make_sequence(x, f):
    return np.c_[x, f(x)]


def random_streams(rng, count, n_in, exclude=()):
    seen = {s.tobytes() for s in exclude}
    out = []
    while len(out) < count:
        x = rng.integers(0, 2, (T + 1, n_in))
        if x.tobytes() not in seen:
            seen.add(x.tobytes())
            out.append(x)
    return out


def output_accuracy(net, starts, seqs, rng):
    """Mean fraction of outputs y_1..y_T correct and fraction of streams fully
    correct, over the weight samples. starts[i] is the initial vector on all
    neurons for seqs[i] (a key, or the visible state padded with zeros)."""
    out_idx = net.n_visible - 1
    acc, exact = [], []
    for mode in WEIGHT_MODES:
        A = sample_weights(net, rng, mode)
        for v0, s in zip(starts, seqs):
            input_sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
            out = np.array(run(A, net.I, v0, input_sets, len(s) - 1))
            pred = out[1:, out_idx]
            acc.append(float((pred == s[1:, out_idx]).mean()))
            exact.append(bool((pred == s[1:, out_idx]).all()))
    return float(np.mean(acc)), float(np.mean(exact))


def key_ignores_inputs(net, trained, V, rng):
    """Fraction of (training stream, weight sample) runs in which the key,
    run with the inputs of a different training stream, still emits exactly
    the trained output sequence. 1.0 means the outputs are replayed from the
    key alone, whatever the inputs."""
    out_idx = net.n_visible - 1
    same = []
    for mode in WEIGHT_MODES:
        A = sample_weights(net, rng, mode)
        for i, (key, s) in enumerate(zip(trained, V)):
            other = V[(i + 1 + int(rng.integers(len(V) - 1))) % len(V)]
            input_sets = [{j for j in net.I if other[t][j]} for t in range(len(other))]
            out = np.array(run(A, net.I, np.r_[other[0][:-1], key[0][out_idx:]], input_sets, len(s) - 1))
            same.append(bool((out[:, out_idx] == s[:, out_idx]).all()))
    return float(np.mean(same))


def zero_hidden(net, s):
    return np.r_[s[0], np.zeros(net.n - net.n_visible, int)]


def visible_diagnostics(seqs, n_in):
    """(is y_{t+1} a function of v(t)?, is it bias-free separable at EPS?)"""
    table = {}
    for s in seqs:
        for t in range(len(s) - 1):
            table.setdefault(tuple(s[t]), set()).add(int(s[t + 1][-1]))
    function = all(len(v) == 1 for v in table.values())
    m = n_in + 1
    L, U = np.full((m, m), -np.inf), np.full((m, m), np.inf)
    return function, bool(feas(seqs, m - 1, L, U, EPS))


def majority_baseline(seqs):
    """Accuracy of always predicting the most common output value."""
    y = np.concatenate([s[1:, -1] for s in seqs])
    return float(max(y.mean(), 1 - y.mean()))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for task, (n_in, f) in TASKS.items():
        for n_train in N_TRAIN:
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + n_train)
                xs = random_streams(rng, n_train, n_in)
                unseen_x = random_streams(rng, N_UNSEEN, n_in, exclude=xs)
                V = [make_sequence(x, f) for x in xs]
                unseen = [make_sequence(x, f) for x in unseen_x]
                net = Net(n_in + 1, inputs=range(n_in), outputs={n_in})
                t0 = time.perf_counter()
                trained = train(net, V, EPS)
                train_s = time.perf_counter() - t0
                train_acc, train_exact = output_accuracy(net, [s[0] for s in trained], V, rng)
                tz_acc, tz_exact = output_accuracy(net, [zero_hidden(net, s) for s in V], V, rng)
                un_acc, un_exact = output_accuracy(net, [zero_hidden(net, s) for s in unseen], unseen, rng)
                ignores = key_ignores_inputs(net, trained, V, rng)
                is_function, separable = visible_diagnostics(V + unseen, n_in)
                rows.append(dict(
                    task=task, n_train=n_train, seed=seed,
                    hidden_added=net.n - net.n_visible,
                    lemma4_bound=2 * T,
                    train_seconds=round(train_s, 4),
                    train_acc_from_key=train_acc,
                    train_exact_from_key=train_exact,
                    train_acc_zero_hidden=tz_acc,
                    key_output_unchanged_with_other_inputs=ignores,
                    unseen_acc=un_acc,
                    unseen_exact=un_exact,
                    unseen_majority_baseline=majority_baseline(unseen),
                    next_output_function_of_visible=is_function,
                    visible_separable=separable,
                ))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    cols = ["hidden_added", "train_seconds", "train_acc_from_key", "train_exact_from_key",
            "train_acc_zero_hidden", "key_output_unchanged_with_other_inputs", "unseen_acc", "unseen_exact", "unseen_majority_baseline"]
    print("| task | n_train | " + " | ".join(cols) + " | function of visible | separable |")
    print("|" + "---|" * (len(cols) + 4))
    for task in TASKS:
        for n_train in N_TRAIN:
            g = [r for r in rows if r["task"] == task and r["n_train"] == n_train]
            vals = [f"{np.mean([float(r[c]) for r in g]):.3g}" for c in cols]
            fn = {r["next_output_function_of_visible"] for r in g}
            sep = {r["visible_separable"] for r in g}
            print(f"| {task} | {n_train} | " + " | ".join(vals)
                  + f" | {'/'.join(map(str, sorted(fn)))} | {'/'.join(map(str, sorted(sep)))} |")


if __name__ == "__main__":
    main()
