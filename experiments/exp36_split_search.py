"""Experiment 36: a machine search that grows by splitting states, for bigger machines and noisy moves.

Experiment 30 found the letter and machine of noisy demonstrations by
search: every transition table for up to 2 states, local search from 8
random starts for more. Here the machine needs more states. Games
(experiment 28's wait game, 4 x 4, each screen shown 3 network steps,
h = 3): the player must visit K keys in order and then the door:
  keys1: key top-right, door bottom-left (experiment 28's game, 2 states);
  keys2: keys top-right then top-left, then the door (3 states);
  keys3: keys top-right, top-left, bottom-right, then the door (4 states).
The expert heads for the next goal: vertically first, then sideways. The
letter is given (P:top, P:right), so only the machine is searched. With
probability p a screen's move is replaced by one of the 3 others (as
experiment 29). Searches (at most 5 states, number of states chosen on
held-out episodes, majority outputs):
  local: dnn_noise.learn_machine_heldout (experiment 30's search);
  split: dnn_noise.learn_machine_split: from 1 state, each step sends one
         transition to a new state (the best of all such splits) and then
         improves single transitions while the disagreements fall;
  merge: exact merging (dnn_letters.learn_machine_multi), p = 0 only.
Then experiment 30's pipeline: relabel, build the layout, linear programs.
Training 16 to 64 episodes of 25 screens, p = 0, 0.1, 0.2, 10 seeds. Scores:
the true machine, unseen clean episodes with every move right (200),
laps per game of 60 steps for the network and the expert (100 games),
and search time.
Run from the repository root:  python experiments/exp36_split_search.py
Writes experiments/results/exp36_split_search.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import exp26_imitate_expert as e26
import exp28_acting_latency as e28
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M
import dnn_noise as N

N_IN, N_OUT, REP, H_OUT, T = e28.N_IN, e28.N_OUT, 3, 3, e28.T
TASKS = {"keys1": [(1, 1)], "keys2": [(1, 1), (1, 0)], "keys3": [(1, 1), (1, 0), (0, 1)]}
DOOR = (0, 0)
METHODS = ["merge", "local", "split"]
NOISE = [0.0, 0.1, 0.2]
SIZES = [16, 32, 64]
SEEDS = range(10)
MAX_STATES = 5
GAMES, GAME_STEPS, TEST_STREAMS = e26.GAMES, e26.GAME_STEPS, e26.TEST_STREAMS
MOVE_LIST = [(1, 1), (1, 0), (0, 1), (0, 0)]       # up, down, right, left
LETTERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def expert_step(task, q, a):
    """(next state, move, lap): state = keys taken; a = (top, right)."""
    goals = TASKS[task] + [DOOR]
    lap = 0
    if a == goals[q]:
        q, lap = (0, 1) if q == len(goals) - 1 else (q + 1, 0)
    g = goals[q]
    if g[0] != a[0]:
        mv = (1, 1) if g[0] else (1, 0)
    elif g[1] != a[1]:
        mv = (0, 1) if g[1] else (0, 0)
    else:
        mv = (1, 1)
    return q, mv, lap


def machine(task):
    n = len(TASKS[task]) + 1
    return ([{a: expert_step(task, q, a)[0] for a in LETTERS} for q in range(n)],
            [{a: expert_step(task, q, a)[1] for a in LETTERS} for q in range(n)])


def episode(task, rng, n):
    ps, ms = e23.walk(rng, n), e23.walk(rng, n)
    ps, ms = [p for p in ps for _ in range(REP)], [m for m in ms for _ in range(REP)]
    x = np.array([e26.screen(p, m) for p, m in zip(ps, ms)])
    y = np.zeros((len(ps), N_OUT), int)
    q = 0
    for t in range(len(ps) - H_OUT):
        q, mv, _ = expert_step(task, q, e26.letter(ps[t]))
        y[t + H_OUT] = mv
    return np.c_[x, y]


def episodes(task, seed, n, p):
    """Training (noisy as experiment 29) and clean test episodes."""
    rng = np.random.default_rng(seed)
    V = [episode(task, rng, T + 1) for _ in range(max(SIZES))][:n]
    trng = np.random.default_rng(60_000 + seed)
    tests = [episode(task, trng, T + 1) for _ in range(TEST_STREAMS)]
    nrng = np.random.default_rng(20_000 + seed)
    noise = [(nrng.random(T + 1), nrng.integers(1, 4, T + 1)) for _ in range(max(SIZES))][:n]
    wrong = 0
    for s, (u, r) in zip(V, noise):
        for k in range(T + 1):
            ticks = [t + H_OUT for t in range(k * REP, (k + 1) * REP) if t + H_OUT < len(s)]
            if u[k] < p and ticks:
                cur = tuple(s[ticks[0], N_IN:])
                s[ticks, N_IN:] = MOVE_LIST[(MOVE_LIST.index(cur) + r[k]) % 4]
                wrong += 1
    return V, tests, wrong


def play(task, policy, rng):
    p, m = rng.integers(0, e23.SIDE, 2), rng.integers(0, e23.SIDE, 2)
    screens, laps, q = [], 0, 0
    for t in range(GAME_STEPS):
        screens.append((p.copy(), m.copy()))
        q, _, lap = expert_step(task, q, e26.letter(p))
        laps += lap
        p = e26.move(p, e26.MOVES[policy(t, screens)])
        m = e26.move(m, [(-1, 0), (1, 0), (0, -1), (0, 1)][rng.integers(0, 4)])
    return laps


def expert_policy(task):
    def pol(t, screens):
        q = 0
        for p, _ in screens[:t + 1]:
            q, mv, _ = expert_step(task, q, e26.letter(p))
        return mv
    return pol


def one_run(args):
    task, method, p, n_train, seed = args
    V, tests, wrong = episodes(task, seed, n_train, p)
    rules = [e23.FEATURE[nm] for nm in e26.TRUE]
    bits = [lambda X, r=r: e23.U_fire(r, X) for r in rules]
    seqs = Lm.bit_streams(V, N_IN, bits, N_OUT)
    row = dict(task=task, method=method, p=p, n_train=n_train, seed=seed, wrong_moves=wrong)
    t0 = time.perf_counter()
    if method == "merge":
        try:
            delta, out = Lm.learn_machine_multi(seqs, 2, N_OUT, H_OUT)
        except ValueError:
            row.update(machine_correct=False, states_found="", search_seconds=round(time.perf_counter() - t0, 4))
            return row
        err = 0
    elif method == "local":
        delta, out, err, _ = N.learn_machine_heldout(seqs, 2, N_OUT, H_OUT, max_states=MAX_STATES, seed=seed)
    else:
        delta, out, err, _ = N.learn_machine_split(seqs, 2, N_OUT, H_OUT, max_states=MAX_STATES)
    search_secs = time.perf_counter() - t0
    delta = M.complete(delta, 2)
    out = [{a: row_.get(a, (1, 1)) for a in LETTERS} for row_ in out]
    V2 = N.relabel(V, N_IN, N_OUT, H_OUT, bits, delta, out)
    lay, index = Lm.make_layout_bits(N_IN, rules, delta, n_out=N_OUT)
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in index.values()]
    for o in range(N_IN, N_IN + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    t0 = time.perf_counter()
    F.train(net, lay, V2, H.EPS)
    train_secs = time.perf_counter() - t0
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    exact = []
    for s in tests:
        ext = F.extend(lay, s)
        v0 = np.zeros(net.n, int)
        v0[:ext.shape[1]] = ext[0]
        sets = [{i for i in net.I if ext[t][i]} for t in range(len(ext))]
        pred = np.array(dnn.run(mid, net.I, v0, sets, len(ext) - 1))[:, N_IN:N_IN + N_OUT]
        exact.append(bool((pred[H_OUT:] == s[H_OUT:, N_IN:]).all()))
    true_delta, true_out = machine(task)
    key = lambda x: tuple(int(e23.U_fire(r, np.array([x]))[0]) for r in rules)
    game_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=GAMES)
    games = {"network": [play(task, e28.wait_policy(net, lay, mid, REP), np.random.default_rng(g)) for g in game_seeds],
             "expert": [play(task, expert_policy(task), np.random.default_rng(g)) for g in game_seeds]}
    row.update(states_found=len(delta), states_true=len(true_delta), train_disagreements=err,
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, key, e23.SCREENS),
               hidden_added=net.n - lay.n_visible, search_seconds=round(search_secs, 4),
               train_seconds=round(train_secs, 4), unseen_exact_mid=float(np.mean(exact)),
               **{f"laps_{k}": float(np.mean(v)) for k, v in games.items()})
    return row


def summarise(rows):
    sizes = sorted({r["n_train"] for r in rows})
    print("\nRuns with the true machine (of 10) / unseen episodes all right / laps network (expert), "
          "by training episodes\n")
    print("| task | p | method | " + " | ".join(str(s) for s in sizes) + " |")
    print("|" + "---|" * (len(sizes) + 3))
    for task in TASKS:
        for p in NOISE:
            for m in METHODS:
                g = [r for r in rows if (r["task"], r["p"], r["method"]) == (task, p, m)]
                if not g:
                    continue
                cells = []
                for n in sizes:
                    c = [r for r in g if r["n_train"] == n]
                    d = [r for r in c if "unseen_exact_mid" in r]
                    cells.append(f"{sum(r['machine_correct'] for r in c)} / "
                                 + (f"{np.mean([r['unseen_exact_mid'] for r in d]):.2f} / "
                                    f"{np.mean([r['laps_network'] for r in d]):.2f} "
                                    f"({np.mean([r['laps_expert'] for r in d]):.2f})" if d else "none"))
                print(f"| {task} | {p} | {m} | " + " | ".join(cells) + " |")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"\nmerge with no consistent machine: {sum('unseen_exact_mid' not in r for r in rows)} runs; "
          f"hidden added: {sum(r['hidden_added'] for r in done)}")
    for m in METHODS:
        g = [r for r in rows if r["method"] == m]
        if g:
            print(f"{m}: mean / max search seconds {np.mean([r['search_seconds'] for r in g]):.1f} / "
                  f"{max(r['search_seconds'] for r in g):.1f}")


def main(tasks=tuple(TASKS), sizes=SIZES, seeds=SEEDS, noise=NOISE):
    jobs = [(t, m, p, n, s) for t in tasks for m in METHODS for p in noise for n in sizes for s in seeds
            if m != "merge" or p == 0]
    jobs.sort(key=lambda j: (j[1] != "local", -len(TASKS[j[0]]), -j[3]))     # long jobs first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), r["p"], METHODS.index(r["method"]), r["n_train"],
                             r["seed"]))
    H.write_csv("exp36_split_search.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
