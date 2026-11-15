"""Experiment 17: state neurons that read helpers (finite-state tasks on six input bits).

Each task is a machine of experiment 13 (parity or a Tomita language) run on a
binary letter l(x(t)) of six input bits, for example tomita4 on the majority
of x0, x1, x2. The other inputs are distractors. Training: 2 to 32 streams of
13 vectors, 10 seeds. Testing: 200 random streams of 13 vectors, run with no
key. Layouts:
  "merged":  experiment 13 (h = 2): state merging with the 64 raw input
             patterns as letters (dnn_merge.py).
  "letters": h = 3, dnn_letters.py: the letter is chosen from a fixed library
             (each input, each AND of two literals, each XOR of two inputs,
             each majority of three: 101 candidates) as the consistent one
             whose merged machine has the fewest states; helpers compute it
             and the state neurons read the helpers.
  "true_letter": h = 3, as "letters" with the true letter given; only the
             machine is merged from the data. It separates errors in the
             letter from errors in the machine.
The last task's letter, (x0 and x1) or (x2 and x3), is not in the library.
Columns: whether the chosen letter is the true one (as a function on all 64
patterns, up to swapping 0 and 1), whether the machine found gives the true
outputs on every input stream, and the fraction of unseen streams that are
exact (midpoint weights) and of their outputs that are right.
Run from the repository root:  python experiments/exp17_helpers_and_state.py
Writes experiments/results/exp17_helpers_state.csv.
"""
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

import harness as H
import exp10_library_and_states as e10
import exp16_or_of_ands as e16
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

dnn = H.dnn
N_IN, T = 6, 12
SIZES = [2, 4, 8, 16, 32]
SEEDS = range(10)
LAYOUTS = ["merged", "letters", "true_letter"]
TEST_STREAMS = 200
PATTERNS = np.array(list(product((0, 1), repeat=N_IN)))


def _maj(i, j, k):
    return lambda X: (X[:, [i, j, k]].sum(1) >= 2).astype(int)


TASKS = {                                   # name: (machine, letter)
    "parity/maj012": ("parity", _maj(0, 1, 2)),
    "tomita3/xor01": ("tomita3", lambda X: X[:, 0] ^ X[:, 1]),
    "tomita4/and01": ("tomita4", lambda X: X[:, 0] & X[:, 1]),
    "tomita5/x0": ("tomita5", lambda X: X[:, 0]),
    "tomita7/maj345": ("tomita7", _maj(3, 4, 5)),
    "tomita4/dnf2": ("tomita4", lambda X: (X[:, 0] & X[:, 1]) | (X[:, 2] & X[:, 3])),
}


def stream(x, task, h):
    """Inputs x (T + 1 rows) and the output after reading x(0..t) at t + h."""
    delta, out, _ = e10.machine(TASKS[task][0])
    letters = TASKS[task][1](x)
    y = np.zeros(len(x), int)
    q = 0
    for t in range(len(x) - h):
        a = (int(letters[t]),)
        y[t + h] = out[q][a]
        q = delta[q][a]
    return np.c_[x, y]


def run_out(net, lay, s):
    A = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    return e16.run_out(net, A, F.extend(lay, s), N_IN)


def one_run(args):
    task, n_train, seed, kind = args
    h = 2 if kind == "merged" else 3
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [rng.integers(0, 2, (T + 1, N_IN)) for _ in range(n_train)]
    trng = np.random.default_rng(70_000 + seed)
    V = [stream(x, task, h) for x in xs]
    tests = [stream(trng.integers(0, 2, (T + 1, N_IN)), task, h) for _ in range(TEST_STREAMS)]
    true_delta, true_out, _ = e10.machine(TASKS[task][0])
    true_f = TASKS[task][1]
    true_key = lambda x: (int(true_f(np.array([x]))[0]),)
    t0 = time.perf_counter()
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind, depth=h)
    if kind == "merged":
        lay, delta, out = M.make_layout(V, N_IN, h)
        key = lambda x: tuple(int(v) for v in x)
        row.update(letter="raw", letter_correct="", candidates_consistent="", helpers=0)
        allowed = None
    else:
        if kind == "letters":
            found = Lm.choose_letter(V, N_IN, h)
            if found is None:
                row.update(letter="none", letter_correct=False, candidates_consistent=0)
                return row
            name, f, delta, out, n_ok = found
            fv, tv = f(PATTERNS), true_f(PATTERNS)
            row.update(letter=name, letter_correct=bool((fv == tv).all() or (fv == 1 - tv).all()),
                       candidates_consistent=n_ok)
        else:
            f = true_f
            delta, out = M.learn_machine(Lm.letter_streams(V, N_IN, f), 1, h)
            row.update(letter="true", letter_correct=True, candidates_consistent="")
        rules, lr = Lm.letter_helpers(f, N_IN)
        lay, index = Lm.make_layout(N_IN, rules, lr, M.complete(delta, 1))
        key = lambda x: (int(f(np.array([x]))[0]),)
        row["helpers"] = len(rules)
        allowed = list(index.values())
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    if allowed is not None:
        blocked = [j for j in range(lay.n_visible) if j not in allowed]
        net.L[N_IN, blocked] = net.U[N_IN, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    preds = [run_out(net, lay, s) for s in tests]
    exact = [(p[h:] == s[h:, N_IN]).all() for p, s in zip(preds, tests)]
    steps = np.concatenate([p[h:] == s[h:, N_IN] for p, s in zip(preds, tests)])
    seen = {x[t].tobytes() for x in xs for t in range(T + 1)}
    row.update(states_found=len(delta), states_true=len(true_delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key,
                                                 [tuple(p) for p in PATTERNS]),
               patterns_seen=len(seen), hidden_added=net.n - lay.n_visible,
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, [N_IN], rng)[0],
               unseen_exact_mid=float(np.mean(exact)), unseen_step_acc_mid=float(np.mean(steps)))
    return row


def _sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def summarise(rows):
    def cell(g):
        g = [r for r in g if "unseen_exact_mid" in r]
        if not g:
            return "–"
        return (f"{sum(r['machine_correct'] for r in g)} / {np.mean([r['unseen_exact_mid'] for r in g]):.2f}")
    print("\nRuns with the true machine (of 10) / mean fraction of unseen streams exact (midpoint), "
          "by number of training streams\n")
    print("| task | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for task in TASKS:
        for kind in LAYOUTS:
            print(f"| {task} | {kind} | " + " | ".join(cell(_sel(rows, task=task, layout=kind, n_train=s))
                                                     for s in SIZES) + " |")
    print("\nletters: runs whose chosen letter is the true one, by number of training streams\n")
    print("| task | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 1))
    for task in TASKS:
        print(f"| {task} | " + " | ".join(str(sum(r["letter_correct"] is True for r in _sel(rows, task=task, layout="letters", n_train=s)))
                                         for s in SIZES) + " |")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"\nruns with no consistent letter: {sum(r.get('letter') == 'none' for r in rows)}; "
          f"hidden neurons added by TRAIN: {sum(r['hidden_added'] > 0 for r in done)} runs; "
          f"recall from keys, min: {min(r['recall_with_key'] for r in done):.3f}")
    print(f"max seconds: build {max(r['build_seconds'] for r in done):.2f}, "
          f"train {max(r['train_seconds'] for r in done):.2f}")


def main(sizes=SIZES, seeds=SEEDS, tasks=TASKS):
    jobs = [(t, n, s, k) for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    H.write_csv("exp17_helpers_state.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
