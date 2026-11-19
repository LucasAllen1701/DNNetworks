"""Experiment 26: acting in the grid game by copying an expert.

The game of experiment 23 (4 x 4, player and monster planes, random-walking
monster). An expert fetches the key (top-right quadrant) and takes it to the
door (bottom-left quadrant), again and again; key and door are not drawn, so
the expert must remember whether it holds the key. Its move is 2 output
bits: (vertical, up-or-right). Holding no key, it goes up until it is in the
top half, then right; holding the key, down until the bottom half, then
left. Reaching the key quadrant takes the key; reaching the door quadrant
with the key scores a lap and drops it. This is a 2-state machine reading
the letter (player in top half, player in right half) with 2 output bits.
Training: 8 to 32 episodes of 25 screens in which the player walks at
random (the expert's moves are only labels), 10 seeds. Layouts (h = 3,
dnn_letters.make_layout_bits with 2 output neurons, each reading only the
state neurons): "bits" chooses the letter from experiment 23's 32 regions;
"true_bits" is given (P:top, P:right).
Scores: (1) imitation on 200 unseen random-walk episodes: episodes with
every move right, and moves right; (2) play: the network drives the player
for 60 steps from a random start, the same 100 games for every player. The network's move at step t
answers the screen of step t - 3 (h = 3), so it acts on old screens; the
same lag is given to an "expert with lag" reference. Reported: mean laps
per game for the network, the expert with lag, the expert without lag, and
a random player.
Run from the repository root:  python experiments/exp26_imitate_expert.py
Writes experiments/results/exp26_imitate_expert.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

SIDE, N_PIX, N_IN, h = e23.SIDE, e23.N_PIX, e23.N_IN, e23.h
N_OUT = 2
T = 24
SIZES = [8, 16, 24, 32]
SEEDS = range(10)
TEST_STREAMS = 200
GAMES, GAME_STEPS = 100, 60
LAYOUTS = ["bits", "true_bits"]
TRUE = ["P:top", "P:right"]
MOVES = {(1, 1): (-1, 0), (1, 0): (1, 0), (0, 1): (0, 1), (0, 0): (0, -1)}   # up, down, right, left


def expert_step(q, a):
    """(next state, move bits, lap) after reading a = (top, right)."""
    top, right = a
    lap = 0
    if not q and top and right:
        q = 1                                   # took the key
    elif q and not top and not right:
        q, lap = 0, 1                           # through the door
    if not q:
        move = (1, 1) if not top else (0, 1)
    else:
        move = (1, 0) if top else (0, 0)
    return q, move, lap


def machine():
    letters = [(0, 0), (0, 1), (1, 0), (1, 1)]
    delta = [{a: expert_step(q, a)[0] for a in letters} for q in (0, 1)]
    out = [{a: expert_step(q, a)[1] for a in letters} for q in (0, 1)]
    return delta, out


def letter(pos):
    r, c = pos
    return int(r < SIDE // 2), int(c >= SIDE // 2)


def screen(p, m):
    x = np.zeros(N_IN, int)
    x[p[0] * SIDE + p[1]] = 1
    x[N_PIX + m[0] * SIDE + m[1]] = 1
    return x


def episode(rng, n):
    """Random-walk screens with the expert's move h steps later."""
    ps, ms = e23.walk(rng, n), e23.walk(rng, n)
    x = np.array([screen(p, m) for p, m in zip(ps, ms)])
    y = np.zeros((n, N_OUT), int)
    q = 0
    for t in range(n - h):
        q, move, _ = expert_step(q, letter(ps[t]))
        y[t + h] = move
    return np.c_[x, y]


def move(pos, d):
    return np.clip(np.asarray(pos) + d, 0, SIDE - 1)


def play(policy, rng):
    """Laps in one game. policy(t, screens so far) -> move bits."""
    p, m = rng.integers(0, SIDE, 2), rng.integers(0, SIDE, 2)
    screens, laps, q = [], 0, 0
    for t in range(GAME_STEPS):
        screens.append((p.copy(), m.copy()))
        q, _, lap = expert_step(q, letter(p))  # the game's own key and door
        laps += lap
        p = move(p, MOVES[policy(t, screens)])
        m = move(m, [(-1, 0), (1, 0), (0, -1), (0, 1)][rng.integers(0, 4)])
    return laps


def expert_policy(lag):
    def pol(t, screens):
        if t < lag:
            return (0, 0)
        q = 0
        for p, _ in screens[:t - lag + 1]:
            q, mv, _ = expert_step(q, letter(p))
        return mv
    return pol


def network_policy(net, lay, A):
    """Runs the network one step per screen; the move is its output now."""
    outs = list(range(N_IN, N_IN + N_OUT))
    state = {}

    def pol(t, screens):
        x = screen(*screens[-1])
        if t == 0:
            v = np.zeros(net.n, int)
        else:
            v = dnn.sigma(A @ state["v"]).astype(int)
        v[:N_IN] = x                            # the input neurons are clamped
        if lay.on is not None:
            v[lay.on] = 1
        state["v"] = v
        return tuple(int(b) for b in v[outs])
    return pol


def one_run(args):
    layout, n_train, seed = args
    rng = np.random.default_rng(seed)
    V = [episode(rng, T + 1) for _ in range(max(SIZES))][:n_train]
    trng = np.random.default_rng(60_000 + seed)
    tests = [episode(trng, T + 1) for _ in range(TEST_STREAMS)]
    true_delta, true_out = machine()
    row = dict(layout=layout, n_train=n_train, seed=seed)
    t0 = time.perf_counter()
    if layout == "true_bits":
        names = TRUE
        rules = [e23.FEATURE[n] for n in names]
        delta, out = Lm.learn_machine_multi(
            Lm.bit_streams(V, N_IN, [lambda X, r=r: e23.U_fire(r, X) for r in rules], N_OUT), 2, N_OUT, h)
        n_ok = ""
    else:
        found = Lm.choose_bits(V, N_IN, e23.FEATURES, h, n_out=N_OUT)
        if found is None:
            row.update(letter="none", machine_correct=False)
            return row
        idx, delta, out, n_ok = found
        names, rules = [e23.FEATURES[i][0] for i in idx], [e23.FEATURES[i][1] for i in idx]
    lay, index = Lm.make_layout_bits(N_IN, rules, M.complete(delta, len(rules)), n_out=N_OUT)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in index.values()]
    for o in range(N_IN, N_IN + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    key = lambda x: tuple(int(e23.U_fire(r, np.array([x]))[0]) for r in rules)
    true_key = lambda x: tuple(int(e23.U_fire(e23.FEATURE[n], np.array([x]))[0]) for n in TRUE)
    exact, steps = [], []
    for s in tests:
        ext = F.extend(lay, s)
        v0 = np.zeros(net.n, int)
        v0[:ext.shape[1]] = ext[0]
        sets = [{i for i in net.I if ext[t][i]} for t in range(len(ext))]
        pred = np.array(dnn.run(mid, net.I, v0, sets, len(ext) - 1))[:, N_IN:N_IN + N_OUT]
        ok = (pred[h:] == s[h:, N_IN:]).all(1)
        exact.append(ok.all())
        steps.append(ok.mean())
    game_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=GAMES)
    prng = np.random.default_rng(30_000 + seed)
    makers = {"network": lambda: network_policy(net, lay, mid), "expert_lag": lambda: expert_policy(h),
              "expert": lambda: expert_policy(0), "random": lambda: lambda t, sc: tuple(prng.integers(0, 2, 2))}
    games = {k: [play(make(), np.random.default_rng(g)) for g in game_seeds] for k, make in makers.items()}
    row.update(letter="+".join(names), letter_correct=sorted(names) == sorted(TRUE), candidates_consistent=n_ok,
               states_found=len(delta), states_true=len(true_delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key, e23.SCREENS),
               hidden_added=net.n - lay.n_visible, build_seconds=round(build_secs, 4),
               train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, list(range(N_IN, N_IN + N_OUT)), rng)[0],
               unseen_exact_mid=float(np.mean(exact)), unseen_move_acc_mid=float(np.mean(steps)),
               **{f"laps_{k}": float(np.mean(v)) for k, v in games.items()})
    return row


def summarise(rows):
    print("\nRuns with the true machine (of 10) / mean fraction of unseen episodes with every move right, "
          "by number of training episodes\n")
    print("| layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 1))
    for lay in LAYOUTS:
        cells = []
        for s in SIZES:
            g = [r for r in rows if r["layout"] == lay and r["n_train"] == s and "unseen_exact_mid" in r]
            cells.append(f"{sum(r['machine_correct'] for r in g)} / {np.mean([r['unseen_exact_mid'] for r in g]):.2f}"
                         if g else "–")
        print(f"| {lay} | " + " | ".join(cells) + " |")
    print(f"\nMean laps per game of {GAME_STEPS} steps (network / expert with lag {h} / expert / random)\n")
    print("| layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 1))
    for lay in LAYOUTS:
        cells = []
        for s in SIZES:
            g = [r for r in rows if r["layout"] == lay and r["n_train"] == s and "laps_network" in r]
            cells.append(" / ".join(f"{np.mean([r[f'laps_{k}'] for r in g]):.2f}"
                                    for k in ("network", "expert_lag", "expert", "random")) if g else "–")
        print(f"| {lay} | " + " | ".join(cells) + " |")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"\nletters chosen by bits at {SIZES[-1]} episodes: "
          + ", ".join(sorted({r['letter'] for r in done if r['layout'] == 'bits' and r['n_train'] == SIZES[-1]})))
    print(f"hidden neurons added by TRAIN: {sum(r['hidden_added'] > 0 for r in done)} runs; recall from keys, min: "
          f"{min(r['recall_with_key'] for r in done):.3f}; no consistent letter: "
          f"{sum(r.get('letter') == 'none' for r in rows)} runs")
    print(f"max seconds: build {max(r['build_seconds'] for r in done):.2f}, "
          f"train {max(r['train_seconds'] for r in done):.2f}")


def main(sizes=SIZES, seeds=SEEDS):
    jobs = [(l, n, s) for l in LAYOUTS for n in sizes for s in seeds]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (LAYOUTS.index(r["layout"]), r["n_train"], r["seed"]))
    H.write_csv("exp26_imitate_expert.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
