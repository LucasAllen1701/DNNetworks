"""Experiment 20: learning the letter from the data (follow-up to experiment 17).

Experiment 17 chose the letter of a finite-state task from a fixed library of
101 functions of six input bits, and built its helpers on the letter's full
truth table. Its letter (x0 and x1) or (x2 and x3) (task tomita4/dnf2) is not
in the library and was never found. Two ways to learn the letter, h = 3:
  "composed": the library grown by one step (dnn_letters.compose_letter):
             the OR and the AND of every pair of library letters (about
             10,000 letters). The consistent letter (of either level) whose
             merged machine has the fewest states wins (ties: the library
             letter); helpers as in experiment 17.
  "search":  no library (dnn_letters.learn_letter): the smallest machine and
             a letter label for every training pattern, found together by
             backtracking so that every training output is reproduced (the
             library only orders the free label choices; at most 6 states,
             300,000 search nodes per size, a size that runs out is passed
             over). The labels the machine forces are fitted by the covering
             search of experiment 16, and the letter is the OR of those
             helpers, so unseen patterns get the helpers' answer.
"letters" and "true_letter" are experiment 17's layouts; these rows should
equal experiment 17's. Tasks: experiment 17's six, plus tomita7/or3 (the OR
of x0, x1, x2, also outside the library). Training: 2 to 32 streams of 13
vectors, 10 seeds; testing: 200 random streams, run with no key (midpoint
weights).
Run from the repository root:  python experiments/exp20_learned_letter.py
Writes experiments/results/exp20_learned_letter.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp10_library_and_states as e10
import exp17_helpers_and_state as e17
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

dnn = H.dnn
N_IN, T = e17.N_IN, e17.T
SIZES = e17.SIZES
SEEDS = e17.SEEDS
LAYOUTS = ["letters", "true_letter", "composed", "search"]
TEST_STREAMS = e17.TEST_STREAMS
PATTERNS = e17.PATTERNS
# experiment 17's helpers look tasks up in its own table, so the new task goes there
e17.TASKS["tomita7/or3"] = ("tomita7", lambda X: (X[:, :3].sum(1) >= 1).astype(int))
TASKS = e17.TASKS


def learned_run(task, n_train, seed, kind):
    h = 3
    rng = np.random.default_rng(seed * 1000 + n_train)
    xs = [rng.integers(0, 2, (T + 1, N_IN)) for _ in range(n_train)]
    trng = np.random.default_rng(70_000 + seed)
    V = [e17.stream(x, task, h) for x in xs]
    tests = [e17.stream(trng.integers(0, 2, (T + 1, N_IN)), task, h) for _ in range(TEST_STREAMS)]
    true_delta, true_out, _ = e10.machine(TASKS[task][0])
    true_f = TASKS[task][1]
    true_key = lambda x: (int(true_f(np.array([x]))[0]),)
    row = dict(task=task, n_train=n_train, seed=seed, layout=kind, depth=h)
    t0 = time.perf_counter()
    if kind == "composed":
        found = Lm.compose_letter(V, N_IN, h)
        if found is None:
            row.update(letter="none", letter_correct=False, candidates_consistent=0)
            return row
        name, f, delta, out, level, n_ok = found
        rules, lr = Lm.letter_helpers(f, N_IN)
        row.update(letter=name, level=level, candidates_consistent=n_ok)
    else:
        found = Lm.learn_letter(V, N_IN, h)
        if found is None:
            row.update(letter="none", letter_correct=False)
            return row
        f, delta, out, rules, lr = found["f"], found["delta"], found["out"], found["rules"], found["rule"]
        row.update(letter=found["name"], states_search=found["states_search"],
                   budget_hits=found["budget_hits"], labels_forced=found["labels_forced"])
    lay, index = Lm.make_layout(N_IN, rules, lr, M.complete(delta, 1))
    build_secs = time.perf_counter() - t0
    fv, tv = f(PATTERNS), true_f(PATTERNS)
    row.update(letter_correct=bool((fv == tv).all() or (fv == 1 - tv).all()),
               letter_agreement=float(max((fv == tv).mean(), (fv != tv).mean())), helpers=len(rules))
    net = F.make_net(lay)
    allowed = list(index.values())
    blocked = [j for j in range(lay.n_visible) if j not in allowed]
    net.L[N_IN, blocked] = net.U[N_IN, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    preds = [e17.run_out(net, lay, s) for s in tests]
    exact = [(p[h:] == s[h:, N_IN]).all() for p, s in zip(preds, tests)]
    steps = np.concatenate([p[h:] == s[h:, N_IN] for p, s in zip(preds, tests)])
    seen = {x[t].tobytes() for x in xs for t in range(T + 1)}
    key = lambda x: (int(f(np.array([x]))[0]),)
    row.update(states_found=len(delta), states_true=len(true_delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key,
                                                 [tuple(p) for p in PATTERNS]),
               patterns_seen=len(seen), hidden_added=net.n - lay.n_visible,
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, [N_IN], rng)[0],
               unseen_exact_mid=float(np.mean(exact)), unseen_step_acc_mid=float(np.mean(steps)))
    return row


def one_run(args):
    task, n_train, seed, kind = args
    if kind in ("composed", "search"):
        return learned_run(task, n_train, seed, kind)
    return e17.one_run(args)


def summarise(rows):
    sel = e17._sel

    def cell(g):
        g = [r for r in g if "unseen_exact_mid" in r]
        if not g:
            return "–"
        return f"{sum(r['machine_correct'] for r in g)} / {np.mean([r['unseen_exact_mid'] for r in g]):.2f}"
    print("\nRuns with the true machine (of 10) / mean fraction of unseen streams exact (midpoint), "
          "by number of training streams ('–': no letter found in any run)\n")
    print("| task | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for task in TASKS:
        for kind in LAYOUTS:
            print(f"| {task} | {kind} | " + " | ".join(cell(sel(rows, task=task, layout=kind, n_train=s))
                                                     for s in SIZES) + " |")
    print("\nRuns whose letter is the true one on all 64 patterns: letters / composed / search\n")
    print("| task | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 1))
    for task in TASKS:
        print(f"| {task} | " + " | ".join(" / ".join(
            str(sum(r.get("letter_correct") is True for r in sel(rows, task=task, layout=k, n_train=s)))
            for k in ("letters", "composed", "search")) for s in SIZES) + " |")
    print("\nRuns with no letter found: " + ", ".join(
        f"{k} {sum(r.get('letter') == 'none' for r in sel(rows, layout=k))}" for k in LAYOUTS))
    comp = [r for r in sel(rows, layout="composed") if "unseen_exact_mid" in r]
    print(f"composed: {sum(r['level'] == 2 for r in comp)} of {len(comp)} runs chose a two-letter combination")
    srch = [r for r in sel(rows, layout="search") if "unseen_exact_mid" in r]
    if srch:
        print(f"search: mean helpers {np.mean([r['helpers'] for r in srch]):.1f}, runs with a size passed over "
              f"{sum(r['budget_hits'] > 0 for r in srch)}, mean agreement with the true letter "
              f"{np.mean([r['letter_agreement'] for r in srch]):.2f}")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"hidden neurons added by TRAIN: {sum(r['hidden_added'] > 0 for r in done)} runs; "
          f"recall from keys, min: {min(r['recall_with_key'] for r in done):.3f}")
    print(f"max seconds: build {max(r['build_seconds'] for r in done):.2f}, "
          f"train {max(r['train_seconds'] for r in done):.2f}")


def main(sizes=SIZES, seeds=SEEDS, tasks=TASKS):
    jobs = [(t, n, s, k) for t in tasks for n in sizes for s in seeds for k in LAYOUTS]
    jobs.sort(key=lambda j: (-j[1], j[3] != "search"))          # big runs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), r["n_train"], r["seed"], LAYOUTS.index(r["layout"])))
    H.write_csv("exp20_learned_letter.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
