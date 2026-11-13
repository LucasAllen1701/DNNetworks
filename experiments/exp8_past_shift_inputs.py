"""Experiment 8: past shifts of the inputs, installed before round 1.

The baseline (dnn.py, unchanged) is given extra *input* neurons that carry the
inputs of earlier steps: for depth d and s = 1 ... d, a neuron whose value at
time t is x(t - s), and 0 for t < s. They are ordinary input neurons, clamped
by RUN, so the Theorem of the notes applies as written; the caller supplies the
sliding window. An in-network copy neuron with weight 1 from its predecessor
would have exactly the same activations.

Visible layout: the task's inputs (then the always-on input, if any), their
shifts 1 ... d, and the output neuron last. With `bias`, the always-on input is
shifted like any other input, so its shifts are 0 for the first steps and 1
afterwards: they tell the network how long ago the stream started.

Part A (one round): the tasks and data of experiment 4, plus five tasks that
are functions of the last few inputs:
- delay2: y(t+1) = x(t-1);
- and2:   y(t+1) = x(t) and x(t-1);
- maj3:   y(t+1) = majority of x(t), x(t-1), x(t-2);
- xor2:   y(t+1) = x(t) xor x(t-1)   (a window function that is not a
  threshold function);
- no000:  y(t+1) = 0 iff x(t) = x(t-1) = x(t-2) = 0, counting only inputs that
  were really read (the stateless core of Tomita 4).
Scores are as in experiment 4; unseen streams start with every hidden neuron
at 0. "mid" columns use the midpoint of every box instead of the five sampled
weight matrices.

Part B (several rounds): the 32 training streams of part A, depth 2, fed
either 4 per round for 8 rounds or 16 per round for 2 rounds, for 10 seeds
(seeds 0-2 use the data of part A). After every round, every stream trained so
far is replayed from its zero-padded key with 20 weight matrices, and the
unseen streams are scored. A run "adds chains late" if any round after the
first adds a hidden neuron.

Run from the repository root:  python experiments/exp8_past_shift_inputs.py
Writes experiments/results/exp8_shift_one_round.csv and exp8_shift_rounds.csv.
"""
import time

import numpy as np

import harness as H
import exp4_input_tasks as e4

dnn = H.dnn

DEPTHS = [0, 1, 2, 3]
N_TRAIN = e4.N_TRAIN
SEEDS = e4.SEEDS

ROUND_TASKS = ["and2", "maj3", "no000", "tomita4", "parity"]
ROUND_DEPTH = 2
ROUND_STREAMS = 32
SCHEDULES = [(8, 4), (2, 16)]   # (rounds, streams per round)
ROUND_SEEDS = range(10)


# ---- extra tasks: functions of the last few inputs (y_0 = 0, or 1 for no000)

def _window_task(rule, first=0):
    def f(x):
        bits = [int(b) for b in x[:, 0]]
        return np.array([first] + [int(rule(bits[:t])) for t in range(1, len(bits))])
    return f


def _last(bits, k):
    """The last k inputs read, most recent first, padded with 0."""
    return (bits[::-1] + [0] * k)[:k]


EXTRA = {
    "delay2": _window_task(lambda b: _last(b, 2)[1]),
    "and2": _window_task(lambda b: all(_last(b, 2))),
    "maj3": _window_task(lambda b: sum(_last(b, 3)) >= 2),
    "xor2": _window_task(lambda b: _last(b, 2)[0] ^ _last(b, 2)[1]),
    "no000": _window_task(lambda b: len(b) < 3 or any(b[-3:]), first=1),
}
TASKS = dict(e4.TASKS)
TASKS.update({name: (1, f) for name, f in EXTRA.items()})


# ---- data

def dataset(task, n_train, seed):
    """(n_inputs, training sequences, unseen sequences); the input streams are
    those of experiment 4 for the same seed and size."""
    n_in, f = TASKS[task]
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = e4.random_streams(rng, n_train, n_in)
    unseen_x = e4.random_streams(rng, e4.N_UNSEEN, n_in, exclude=xs)
    return n_in, [e4.make_sequence(x, f) for x in xs], [e4.make_sequence(x, f) for x in unseen_x]


def with_shifts(s, n_in, depth, bias):
    """Columns: inputs (and the always-on input), their shifts 1..depth, output."""
    s = np.asarray(s, int)
    inp = s[:, :n_in]
    if bias:
        inp = np.c_[inp, np.ones(len(s), int)]
    blocks = [inp]
    for d in range(1, depth + 1):
        blocks.append(np.vstack([np.zeros((d, inp.shape[1]), int), inp[:-d]]))
    return np.hstack(blocks + [s[:, n_in:]])


def build(task, n_train, seed, depth, bias):
    n_in, V, unseen = dataset(task, n_train, seed)
    V = [with_shifts(s, n_in, depth, bias) for s in V]
    unseen = [with_shifts(s, n_in, depth, bias) for s in unseen]
    out = V[0].shape[1] - 1
    net = dnn.Net(out + 1, inputs=range(out), outputs={out})
    return net, V, unseen, out


# ---- scoring

def midpoint(net):
    return (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2


def score_mid(net, seqs, out):
    """(fraction of streams exact, fraction of outputs correct) for the
    midpoint weights, started with every hidden neuron at 0."""
    A = midpoint(net)
    hits = []
    for s in seqs:
        s = np.asarray(s)
        input_sets = [{i for i in net.I if s[t][i]} for t in range(len(s))]
        v0 = np.r_[s[0], np.zeros(net.n - len(s[0]), int)]
        pred = np.array(dnn.run(A, net.I, v0, input_sets, len(s) - 1))[1:, out]
        hits.append(pred == s[1:, out])
    return float(np.mean([h.all() for h in hits])), float(np.mean(hits))


def diagnostics(seqs, out):
    """(is y(t+1) a function of the visible state v(t)?, is it a threshold
    function of it at margin EPS?) over all the given transitions."""
    table = {}
    for s in seqs:
        for t in range(len(s) - 1):
            table.setdefault(tuple(s[t]), set()).add(int(s[t + 1][out]))
    m = out + 1
    L, U = np.full((m, m), -np.inf), np.full((m, m), np.inf)
    return all(len(v) == 1 for v in table.values()), bool(dnn.feas(seqs, out, L, U, H.EPS))


# ---- part A

def one_round(task, n_train, seed, depth, bias):
    rng = np.random.default_rng(seed)
    net, V, unseen, out = build(task, n_train, seed, depth, bias)
    t0 = time.perf_counter()
    trained = dnn.train(net, V, H.EPS)
    secs = time.perf_counter() - t0
    tr_ex, tr_acc = H.score(net, [k[0] for k in trained], V, [out], rng)
    un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
    tz_acc = H.score(net, H.zero_hidden_starts(V), V, [out], rng)[1]
    mid_ex, mid_acc = score_mid(net, unseen, out)
    function, separable = diagnostics(V + unseen, out)
    return dict(task=task, n_train=n_train, seed=seed, depth=depth, bias=bias,
                visible_neurons=out + 1, chain_neurons=net.n - net.n_visible,
                train_seconds=round(secs, 4),
                train_acc_from_key=tr_acc, train_exact_from_key=tr_ex,
                train_acc_zero_hidden=tz_acc,
                unseen_acc=un_acc, unseen_exact=un_ex,
                unseen_acc_mid=mid_acc, unseen_exact_mid=mid_ex,
                unseen_majority_baseline=H.majority_baseline(unseen, out),
                next_output_function_of_visible=function, visible_separable=separable)


def part_a():
    rows = []
    for task in TASKS:
        for n_train in N_TRAIN:
            for seed in SEEDS:
                for depth in DEPTHS:
                    for bias in (False, True):
                        rows.append(one_round(task, n_train, seed, depth, bias))
    H.write_csv("exp8_shift_one_round.csv", rows)
    return rows


# ---- part B

def rounds(task, seed, bias, n_rounds, per_round, depth=ROUND_DEPTH):
    rng = np.random.default_rng(seed)
    net, V, unseen, out = build(task, ROUND_STREAMS, seed, depth, bias)
    rows, trained, round_of = [], [], []
    for r in range(1, n_rounds + 1):
        batch = V[(r - 1) * per_round: r * per_round]
        before = net.n
        t0 = time.perf_counter()
        new = dnn.train(net, batch, H.EPS)
        secs = time.perf_counter() - t0
        trained += [(k[0], s) for k, s in zip(new, batch)]
        round_of += [r] * len(batch)
        ok = [H.recalled(net, key, s, rng, trials=20) for key, s in trained]
        old = [o for o, q in zip(ok, round_of) if q < r]
        un_ex, un_acc = H.score(net, H.zero_hidden_starts(unseen), unseen, [out], rng)
        mid_ex, mid_acc = score_mid(net, unseen, out)
        rows.append(dict(task=task, seed=seed, bias=bias, depth=depth,
                         rounds=n_rounds, per_round=per_round, round=r,
                         chain_added=net.n - before, total_neurons=net.n,
                         train_seconds=round(secs, 4),
                         recall_this_round=float(np.mean(ok[-per_round:])),
                         recall_earlier_rounds=float(np.mean(old)) if old else "",
                         unseen_acc=un_acc, unseen_exact=un_ex,
                         unseen_acc_mid=mid_acc, unseen_exact_mid=mid_ex,
                         unseen_majority_baseline=H.majority_baseline(unseen, out),
                         space_numbers=net.space()))
    return rows


def part_b():
    rows = []
    for task in ROUND_TASKS:
        for bias in (False, True):
            for n_rounds, per_round in SCHEDULES:
                for seed in ROUND_SEEDS:
                    rows += rounds(task, seed, bias, n_rounds, per_round)
    H.write_csv("exp8_shift_rounds.csv", rows)
    return rows


# ---- tables

def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _flags(rows, col):
    v = {bool(r[col]) for r in rows}
    return "yes" if v == {True} else "no" if v == {False} else "mixed"


def summarise_a(rows, n_train=128):
    print(f"\nPart A, {n_train} training streams (means over seeds)\n")
    print("| task | bias | depth | chain | unseen acc | unseen exact | unseen exact (mid) "
          "| baseline | function | threshold |")
    print("|" + "---|" * 10)
    for task in TASKS:
        for bias in (False, True):
            for depth in DEPTHS:
                g = _sel(rows, task=task, n_train=n_train, depth=depth, bias=bias)
                print(f"| {task} | {'yes' if bias else 'no'} | {depth} | {H.mean(g, 'chain_neurons'):.3g} "
                      f"| {H.mean(g, 'unseen_acc'):.3f} | {H.mean(g, 'unseen_exact'):.3f} "
                      f"| {H.mean(g, 'unseen_exact_mid'):.3f} | {H.mean(g, 'unseen_majority_baseline'):.3f} "
                      f"| {_flags(g, 'next_output_function_of_visible')} | {_flags(g, 'visible_separable')} |")
    print("\nPart A, training exact from key: min over all runs =",
          min(r["train_exact_from_key"] for r in rows))


def summarise_a_sizes(rows, tasks, depth, bias):
    print(f"\nPart A, by training size (depth {depth}, bias {'yes' if bias else 'no'})\n")
    print("| task | streams | chain | unseen exact | unseen exact (mid) |")
    print("|---|---|---|---|---|")
    for task in tasks:
        for n in N_TRAIN:
            g = _sel(rows, task=task, n_train=n, depth=depth, bias=bias)
            print(f"| {task} | {n} | {H.mean(g, 'chain_neurons'):.3g} | {H.mean(g, 'unseen_exact'):.3f} "
                  f"| {H.mean(g, 'unseen_exact_mid'):.3f} |")


def summarise_b(rows):
    n = len(ROUND_SEEDS)
    print(f"\nPart B, depth {ROUND_DEPTH}, {ROUND_STREAMS} training streams, {n} seeds\n")
    print("| task | bias | rounds x streams | chain: round 1 | runs that add chains late "
          "| neurons at end | recall, earlier rounds (min) | unseen exact: round 1 "
          "| unseen exact: last round | runs exact on all unseen streams at the end |")
    print("|" + "---|" * 10)
    for task in ROUND_TASKS:
        for bias in (False, True):
            for n_rounds, per_round in SCHEDULES:
                g = _sel(rows, task=task, bias=bias, rounds=n_rounds)
                first, last = _sel(g, round=1), _sel(g, round=n_rounds)
                late = {r["seed"] for r in g if r["round"] > 1 and r["chain_added"] > 0}
                solved = sum(r["unseen_exact"] == 1.0 for r in last)
                print(f"| {task} | {'yes' if bias else 'no'} | {n_rounds} x {per_round} "
                      f"| {H.mean(first, 'chain_added'):.3g} | {len(late)} of {n} "
                      f"| {H.mean(last, 'total_neurons'):.3g} "
                      f"| {min(float(r['recall_earlier_rounds']) for r in g if r['round'] > 1):.3g} "
                      f"| {H.mean(first, 'unseen_exact'):.3f} | {H.mean(last, 'unseen_exact'):.3f} "
                      f"| {solved} of {n} |")
    old = [float(r["recall_earlier_rounds"]) for r in rows if r["recall_earlier_rounds"] != ""]
    now = [float(r["recall_this_round"]) for r in rows]
    print(f"\nPart B, recall: min over {len(old)} checks of earlier rounds = {min(old)}, "
          f"min over {len(now)} checks of the current round = {min(now)}")


def main():
    a = part_a()
    summarise_a(a)
    summarise_a_sizes(a, ["delay2", "and2", "maj3", "no000", "tomita4"], 2, True)
    summarise_a_sizes(a, ["delay2", "and2", "maj3", "no000", "tomita4"], 2, False)
    b = part_b()
    summarise_b(b)


if __name__ == "__main__":
    main()
