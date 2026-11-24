"""Experiment 33: the wait game on bigger screens (4 x 4, 8 x 8, 16 x 16).

Experiment 28's "wait" variant of the key/door game (each screen shown for 3
network steps, the move read 3 steps after the screen first appears, h = 3),
with the screen side S = 4, 8 or 16 (2 S^2 inputs: player and monster
planes). The expert is the same 2-state machine reading (player in top
half, player in right half); key = top-right quadrant, door = bottom-left.
Layouts (dnn_letters.make_layout_bits, 2 output neurons reading only the
state neurons):
  bits:      letter of 1-2 regions from experiment 23's library scaled to S
             (rows, columns, halves, quadrants of each plane: 4 S + 16
             regions); at S = 4 this is experiment 28 (its 80 "wait" rows
             must be reproduced);
  cuts:      letter of 1-2 regions from a library of every straight cut
             (row < k, row >= k, column < k, column >= k, k = 1 .. S-1, each
             plane: 8 (S - 1) regions), a library not built around halves;
  true_bits: given (P:top, P:right).
Training 8 to 64 episodes of 25 screens (random walks; the expert's moves are
labels), 10 seeds. Scores as experiment 28: the true machine (checked on
every screen class), unseen episodes with every move right (200), and laps
per game of 60 game steps for the network, the expert and a random player
(100 games). Also the letter search time.
Run from the repository root:  python experiments/exp33_bigger_screens.py
Writes experiments/results/exp33_bigger_screens.csv.
"""
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

import harness as H
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

h, N_OUT, T = 3, 2, 24
REPEAT = 3
SIDES = [4, 8, 16]
SIZES = [8, 16, 24, 32, 64]
SEEDS = range(10)
TEST_STREAMS = 200
GAMES, GAME_STEPS = 100, 60
LAYOUTS = ["bits", "cuts", "true_bits"]
TRUE = ["P:top", "P:right"]
MOVES = {(1, 1): (-1, 0), (1, 0): (1, 0), (0, 1): (0, 1), (0, 0): (0, -1)}   # up, down, right, left
STEPS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def regions(S):
    """Experiment 23's region library for side S (same order)."""
    rows = {f"row{r}": [(r, c) for c in range(S)] for r in range(S)}
    cols = {f"col{c}": [(r, c) for r in range(S)] for c in range(S)}
    half = S // 2
    halves = {"top": [(r, c) for r in range(half) for c in range(S)],
              "bottom": [(r, c) for r in range(half, S) for c in range(S)],
              "left": [(r, c) for r in range(S) for c in range(half)],
              "right": [(r, c) for r in range(S) for c in range(half, S)]}
    quads = {f"{a}{b}": [(r, c) for r in rr for c in cc]
             for a, rr in (("T", range(half)), ("B", range(half, S)))
             for b, cc in (("L", range(half)), ("R", range(half, S)))}
    return {**rows, **cols, **halves, **quads}


def cuts(S):
    """Every straight cut; 'top' and 'right' are named as in regions()."""
    out = {}
    for k in range(1, S):
        out["top" if 2 * k == S else f"r<{k}"] = [(r, c) for r in range(k) for c in range(S)]
        out[f"r>={k}"] = [(r, c) for r in range(k, S) for c in range(S)]
        out[f"c<{k}"] = [(r, c) for r in range(S) for c in range(k)]
        out["right" if 2 * k == S else f"c>={k}"] = [(r, c) for r in range(S) for c in range(k, S)]
    return out


def features(S, library):
    """[(name, rule)]: for each plane and region, the OR of its pixels."""
    n_pix = S * S
    out = []
    for p, off in (("P", 0), ("M", n_pix)):
        for name, cells in library(S).items():
            w = np.zeros(2 * n_pix, int)
            w[[off + r * S + c for r, c in cells]] = 1
            out.append((f"{p}:{name}", (w, 0)))
    return out


def fire(rule, X):
    w, b = rule
    return (np.asarray(X) @ w + b >= 1).astype(int)


def expert_step(q, a):
    """Experiment 26's expert: (next state, move bits, lap) after a = (top, right)."""
    top, right = a
    lap = 0
    if not q and top and right:
        q = 1
    elif q and not top and not right:
        q, lap = 0, 1
    if not q:
        mv = (1, 1) if not top else (0, 1)
    else:
        mv = (1, 0) if top else (0, 0)
    return q, mv, lap


def machine():
    letters = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return ([{a: expert_step(q, a)[0] for a in letters} for q in (0, 1)],
            [{a: expert_step(q, a)[1] for a in letters} for q in (0, 1)])


def letter(pos, S):
    return int(pos[0] < S // 2), int(pos[1] >= S // 2)


def screen(p, m, S):
    x = np.zeros(2 * S * S, int)
    x[p[0] * S + p[1]] = 1
    x[S * S + m[0] * S + m[1]] = 1
    return x


def walk(rng, n, S):
    """Experiment 23's random walk (same random draws)."""
    pos = rng.integers(0, S, 2)
    path = []
    for _ in range(n):
        path.append(pos.copy())
        pos = np.clip(pos + STEPS4[rng.integers(0, 4)], 0, S - 1)
    return path


def episode(rng, n, S):
    """Experiment 28's wait episode: each screen shown REPEAT times, the
    expert's move h steps after each step."""
    ps, ms = walk(rng, n, S), walk(rng, n, S)
    ps, ms = [p for p in ps for _ in range(REPEAT)], [m for m in ms for _ in range(REPEAT)]
    x = np.array([screen(p, m, S) for p, m in zip(ps, ms)])
    y = np.zeros((len(ps), N_OUT), int)
    q = 0
    for t in range(len(ps) - h):
        q, mv, _ = expert_step(q, letter(ps[t], S))
        y[t + h] = mv
    return np.c_[x, y]


def play(policy, rng, S):
    p, m = rng.integers(0, S, 2), rng.integers(0, S, 2)
    screens, laps, q = [], 0, 0
    for t in range(GAME_STEPS):
        screens.append((p.copy(), m.copy()))
        q, _, lap = expert_step(q, letter(p, S))
        laps += lap
        p = np.clip(np.asarray(p) + MOVES[policy(t, screens)], 0, S - 1)
        m = np.clip(np.asarray(m) + STEPS4[rng.integers(0, 4)], 0, S - 1)
    return laps


def expert_policy(S):
    def pol(t, screens):
        q = 0
        for p, _ in screens[:t + 1]:
            q, mv, _ = expert_step(q, letter(p, S))
        return mv
    return pol


def wait_policy(net, lay, A, S):
    n_in = 2 * S * S
    outs = list(range(n_in, n_in + N_OUT))
    state = {}

    def pol(t, screens):
        x = screen(*screens[-1], S)
        v = np.zeros(net.n, int) if t == 0 else state["next"]
        for i in range(REPEAT):
            v[:n_in] = x
            if lay.on is not None:
                v[lay.on] = 1
            nxt = dnn.sigma(A @ v).astype(int)
            v = nxt if i < REPEAT - 1 else v
        state["next"] = nxt
        return tuple(int(b) for b in nxt[outs])
    return pol


def screen_classes(S, rules, true_rules):
    """One screen per distinct (found letter, true letter) pair over every
    screen: same_composite on these equals same_composite on all screens."""
    n_pix = S * S
    reps = {}
    for p, m in product(range(n_pix), repeat=2):
        x = np.zeros(2 * n_pix, int)
        x[p] = x[n_pix + m] = 1
        k = (tuple(int(fire(r, x)) for r in rules), tuple(int(fire(r, x)) for r in true_rules))
        reps.setdefault(k, tuple(x))
    return list(reps.values())


def one_run(args):
    S, layout, n_train, seed = args
    n_in = 2 * S * S
    rng = np.random.default_rng(seed)
    V = [episode(rng, T + 1, S) for _ in range(max(SIZES))][:n_train]
    trng = np.random.default_rng(60_000 + seed)
    tests = [episode(trng, T + 1, S) for _ in range(TEST_STREAMS)]
    true_delta, true_out = machine()
    feats = dict(features(S, regions))
    row = dict(side=S, layout=layout, n_train=n_train, seed=seed)
    t0 = time.perf_counter()
    if layout == "true_bits":
        names = TRUE
        rules = [feats[n] for n in names]
        delta, out = Lm.learn_machine_multi(
            Lm.bit_streams(V, n_in, [lambda X, r=r: fire(r, X) for r in rules], N_OUT), 2, N_OUT, h)
        n_ok, n_cand = "", ""
    else:
        lib = features(S, regions if layout == "bits" else cuts)
        n_cand = len(lib) + len(lib) * (len(lib) - 1) // 2
        found = Lm.choose_bits_fast(V, n_in, lib, h, n_out=N_OUT)
        if found is None:
            row.update(letter="none", machine_correct=False, candidates=n_cand,
                       search_seconds=round(time.perf_counter() - t0, 4))
            return row
        idx, delta, out, n_ok = found
        names, rules = [lib[i][0] for i in idx], [lib[i][1] for i in idx]
    search_secs = time.perf_counter() - t0
    lay, index = Lm.make_layout_bits(n_in, rules, M.complete(delta, len(rules)), n_out=N_OUT)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in set(index.values())]
    for o in range(n_in, n_in + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    true_rules = [feats[n] for n in TRUE]
    key = lambda x: tuple(int(fire(r, np.array([x]))[0]) for r in rules)
    true_key = lambda x: tuple(int(fire(r, np.array([x]))[0]) for r in true_rules)
    exact, steps = [], []
    for s in tests:
        ext = F.extend(lay, s)
        v0 = np.zeros(net.n, int)
        v0[:ext.shape[1]] = ext[0]
        sets = [{i for i in net.I if ext[t][i]} for t in range(len(ext))]
        pred = np.array(dnn.run(mid, net.I, v0, sets, len(ext) - 1))[:, n_in:n_in + N_OUT]
        ok = (pred[h:] == s[h:, n_in:]).all(1)
        exact.append(ok.all())
        steps.append(ok.mean())
    game_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=GAMES)
    prng = np.random.default_rng(30_000 + seed)
    makers = {"network": lambda: wait_policy(net, lay, mid, S), "expert": lambda: expert_policy(S),
              "random": lambda: lambda t, sc: tuple(prng.integers(0, 2, 2))}
    games = {k: [play(make(), np.random.default_rng(g), S) for g in game_seeds] for k, make in makers.items()}
    row.update(letter="+".join(names), letter_correct=sorted(names) == sorted(TRUE), candidates=n_cand,
               candidates_consistent=n_ok, states_found=len(delta), states_true=len(true_delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key,
                                                 screen_classes(S, rules, true_rules)),
               hidden_added=net.n - lay.n_visible, search_seconds=round(search_secs, 4),
               build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, list(range(n_in, n_in + N_OUT)), rng)[0],
               unseen_exact_mid=float(np.mean(exact)), unseen_move_acc_mid=float(np.mean(steps)),
               **{f"laps_{k}": float(np.mean(v)) for k, v in games.items()})
    return row


def summarise(rows, sizes):
    print("\nRuns with the true machine (of 10) / unseen episodes all right / laps network (expert), "
          "by number of training episodes\n")
    print("| side | layout | " + " | ".join(str(s) for s in sizes) + " |")
    print("|" + "---|" * (len(sizes) + 2))
    sides = sorted({r["side"] for r in rows})
    for S in sides:
        for lay in LAYOUTS:
            cells = []
            for n in sizes:
                g = [r for r in rows if (r["side"], r["layout"], r["n_train"]) == (S, lay, n)]
                d = [r for r in g if "unseen_exact_mid" in r]
                if not g:
                    cells.append("–")
                    continue
                cells.append(f"{sum(r['machine_correct'] for r in g)} / "
                             + (f"{np.mean([r['unseen_exact_mid'] for r in d]):.2f} / "
                                f"{np.mean([r['laps_network'] for r in d]):.2f} "
                                f"({np.mean([r['laps_expert'] for r in d]):.2f})" if d else "none"))
            print(f"| {S} | {lay} | " + " | ".join(cells) + " |")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"\nno consistent letter: {sum(r.get('letter') == 'none' for r in rows)} runs; recall from keys, min: "
          f"{min(r['recall_with_key'] for r in done):.3f}; max hidden added: {max(r['hidden_added'] for r in done)}")
    for S in sides:
        for lay in ("bits", "cuts"):
            g = [r for r in rows if (r["side"], r["layout"]) == (S, lay)]
            if g:
                print(f"side {S} {lay}: candidates {g[0]['candidates']}, max search seconds "
                      f"{max(r['search_seconds'] for r in g):.1f}; letters at {max(sizes)}: "
                      + ", ".join(sorted({r['letter'] for r in g if r['n_train'] == max(sizes)})))


def main(sides=SIDES, sizes=SIZES, seeds=SEEDS):
    jobs = [(S, l, n, s) for S in sides for l in LAYOUTS for n in sizes for s in seeds]
    jobs.sort(key=lambda j: -j[0] * j[2])        # longest first
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (r["side"], LAYOUTS.index(r["layout"]), r["n_train"], r["seed"]))
    H.write_csv("exp33_bigger_screens.csv", rows)
    summarise(rows, sizes)


if __name__ == "__main__":
    main()
