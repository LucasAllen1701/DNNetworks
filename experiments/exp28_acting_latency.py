"""Experiment 28: acting without the lag, in experiment 26's game.

Experiment 26's network imitates the key/door expert exactly but plays like
an expert with a 3-step lag (2.74 laps per 60 steps against 14.28): with
h = 3 its move answers the screen of 3 steps earlier. Three variants, each
with layouts "bits" (letter chosen from experiment 23's 32 regions) and
"true_bits" (given P:top, P:right):
  lag3:   experiment 26 unchanged (its 80 rows must be reproduced).
  direct: the output neurons read the letter-bit neurons as well as the
          state neurons, and are trained to answer 2 steps after the input
          (h = 2 for the outputs; letter and state neurons as before).
          TRAIN adds hidden neurons if a single output neuron cannot do it.
  wait:   the game waits for the move: each screen is shown for 3 network
          steps and the move is read 3 steps after the screen first appears,
          so the move answers the current screen (no lag in game time).
          Training episodes show each screen 3 times as well.
Training 8 to 32 episodes of 25 screens, 10 seeds, as experiment 26.
Scores: unseen episodes with every move right; laps per game of 60 game
steps for the network, the expert with the variant's lag (0 for wait), the
expert without lag, and a random player.
Run from the repository root:  python experiments/exp28_acting_latency.py
Writes experiments/results/exp28_acting_latency.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import exp26_imitate_expert as e26
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

N_IN, N_OUT, h = e26.N_IN, e26.N_OUT, e26.h
T, SIZES, SEEDS, TEST_STREAMS = e26.T, e26.SIZES, e26.SEEDS, e26.TEST_STREAMS
GAMES, GAME_STEPS = e26.GAMES, e26.GAME_STEPS
LAYOUTS = e26.LAYOUTS
VARIANTS = {"lag3": dict(repeat=1, h_out=3, read_bits=False),
            "direct": dict(repeat=1, h_out=2, read_bits=True),
            "wait": dict(repeat=3, h_out=3, read_bits=False)}


def episode(rng, n, repeat, h_out):
    """Experiment 26's random-walk episode with each screen shown `repeat`
    times; the expert's move answers each step h_out steps later."""
    ps, ms = e23.walk(rng, n), e23.walk(rng, n)
    ps, ms = [p for p in ps for _ in range(repeat)], [m for m in ms for _ in range(repeat)]
    x = np.array([e26.screen(p, m) for p, m in zip(ps, ms)])
    y = np.zeros((len(ps), N_OUT), int)
    q = 0
    for t in range(len(ps) - h_out):
        q, move, _ = e26.expert_step(q, e26.letter(ps[t]))
        y[t + h_out] = move
    return np.c_[x, y]


def wait_policy(net, lay, A, repeat):
    """The game waits: each screen is clamped for `repeat` network steps and
    the move is the output one step later (answering the screen's first
    step when repeat = h)."""
    outs = list(range(N_IN, N_IN + N_OUT))
    state = {}

    def pol(t, screens):
        x = e26.screen(*screens[-1])
        v = np.zeros(net.n, int) if t == 0 else state["next"]
        for _ in range(repeat):
            v[:N_IN] = x
            if lay.on is not None:
                v[lay.on] = 1
            nxt = dnn.sigma(A @ v).astype(int)
            v = nxt if _ < repeat - 1 else v
        state["next"] = nxt
        return tuple(int(b) for b in nxt[outs])
    return pol


def one_run(args):
    variant, layout, n_train, seed = args
    spec = VARIANTS[variant]
    rep, h_out = spec["repeat"], spec["h_out"]
    rng = np.random.default_rng(seed)
    V = [episode(rng, T + 1, rep, h_out) for _ in range(max(SIZES))][:n_train]
    trng = np.random.default_rng(60_000 + seed)
    tests = [episode(trng, T + 1, rep, h_out) for _ in range(TEST_STREAMS)]
    true_delta, true_out = e26.machine()
    row = dict(variant=variant, layout=layout, n_train=n_train, seed=seed)
    t0 = time.perf_counter()
    if layout == "true_bits":
        names = e26.TRUE
        rules = [e23.FEATURE[n] for n in names]
        delta, out = Lm.learn_machine_multi(
            Lm.bit_streams(V, N_IN, [lambda X, r=r: e23.U_fire(r, X) for r in rules], N_OUT), 2, N_OUT, h_out)
        n_ok = ""
    else:
        found = Lm.choose_bits(V, N_IN, e23.FEATURES, h_out, n_out=N_OUT)
        if found is None:
            row.update(letter="none", machine_correct=False)
            return row
        idx, delta, out, n_ok = found
        names, rules = [e23.FEATURES[i][0] for i in idx], [e23.FEATURES[i][1] for i in idx]
    lay, index = Lm.make_layout_bits(N_IN, rules, M.complete(delta, len(rules)), n_out=N_OUT)
    build_secs = time.perf_counter() - t0
    net = F.make_net(lay)
    allowed = set(index.values())
    if spec["read_bits"]:
        allowed |= {j for j, nm in enumerate(lay.names) if nm.startswith("bit")}
    blocked = [j for j in range(lay.n_visible) if j not in allowed]
    for o in range(N_IN, N_IN + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    t0 = time.perf_counter()
    trained = F.train(net, lay, V, H.EPS)
    train_secs = time.perf_counter() - t0
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    key = lambda x: tuple(int(e23.U_fire(r, np.array([x]))[0]) for r in rules)
    true_key = lambda x: tuple(int(e23.U_fire(e23.FEATURE[n], np.array([x]))[0]) for n in e26.TRUE)
    exact, steps = [], []
    for s in tests:
        ext = F.extend(lay, s)
        v0 = np.zeros(net.n, int)
        v0[:ext.shape[1]] = ext[0]
        sets = [{i for i in net.I if ext[t][i]} for t in range(len(ext))]
        pred = np.array(dnn.run(mid, net.I, v0, sets, len(ext) - 1))[:, N_IN:N_IN + N_OUT]
        ok = (pred[h_out:] == s[h_out:, N_IN:]).all(1)
        exact.append(ok.all())
        steps.append(ok.mean())
    game_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=GAMES)
    prng = np.random.default_rng(30_000 + seed)
    lag = 0 if rep > 1 else h_out
    net_policy = (lambda: e26.network_policy(net, lay, mid)) if rep == 1 else (lambda: wait_policy(net, lay, mid, rep))
    makers = {"network": net_policy, "expert_lag": lambda: e26.expert_policy(lag),
              "expert": lambda: e26.expert_policy(0), "random": lambda: lambda t, sc: tuple(prng.integers(0, 2, 2))}
    games = {k: [e26.play(make(), np.random.default_rng(g)) for g in game_seeds] for k, make in makers.items()}
    row.update(letter="+".join(names), letter_correct=sorted(names) == sorted(e26.TRUE), candidates_consistent=n_ok,
               states_found=len(delta), states_true=len(true_delta),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key, e23.SCREENS),
               hidden_added=net.n - lay.n_visible, build_seconds=round(build_secs, 4),
               train_seconds=round(train_secs, 4),
               recall_with_key=H.score(net, [s[0] for s in trained], trained, list(range(N_IN, N_IN + N_OUT)), rng)[0],
               unseen_exact_mid=float(np.mean(exact)), unseen_move_acc_mid=float(np.mean(steps)),
               **{f"laps_{k}": float(np.mean(v)) for k, v in games.items()}, lag=lag)
    return row


def summarise(rows):
    print("\nRuns with the true machine (of 10) / unseen episodes all right / hidden neurons added (runs), "
          "by number of training episodes\n")
    print("| variant | layout | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 2))
    for var in VARIANTS:
        for lay in LAYOUTS:
            cells = []
            for s in SIZES:
                g = [r for r in rows if (r["variant"], r["layout"], r["n_train"]) == (var, lay, s)
                     and "unseen_exact_mid" in r]
                cells.append(f"{sum(r['machine_correct'] for r in g)} / {np.mean([r['unseen_exact_mid'] for r in g]):.2f}"
                             f" / {sum(r['hidden_added'] > 0 for r in g)}" if g else "–")
            print(f"| {var} | {lay} | " + " | ".join(cells) + " |")
    print(f"\nMean laps per game of {GAME_STEPS} steps (network / expert with the same lag / expert / random)\n")
    print("| variant | layout | lag | " + " | ".join(str(s) for s in SIZES) + " |")
    print("|" + "---|" * (len(SIZES) + 3))
    for var in VARIANTS:
        for lay in LAYOUTS:
            cells, lag = [], ""
            for s in SIZES:
                g = [r for r in rows if (r["variant"], r["layout"], r["n_train"]) == (var, lay, s) and "laps_network" in r]
                lag = g[0]["lag"] if g else lag
                cells.append(" / ".join(f"{np.mean([r[f'laps_{k}'] for r in g]):.2f}"
                                        for k in ("network", "expert_lag", "expert", "random")) if g else "–")
            print(f"| {var} | {lay} | {lag} | " + " | ".join(cells) + " |")
    done = [r for r in rows if "unseen_exact_mid" in r]
    print(f"\nno consistent letter: {sum(r.get('letter') == 'none' for r in rows)} runs; recall from keys, min: "
          f"{min(r['recall_with_key'] for r in done):.3f}; max hidden added: {max(r['hidden_added'] for r in done)}")
    print(f"max seconds: build {max(r['build_seconds'] for r in done):.2f}, "
          f"train {max(r['train_seconds'] for r in done):.2f}")


def main(sizes=SIZES, seeds=SEEDS, variants=tuple(VARIANTS)):
    jobs = [(v, l, n, s) for v in variants for l in LAYOUTS for n in sizes for s in seeds]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    order = list(VARIANTS)
    rows.sort(key=lambda r: (order.index(r["variant"]), LAYOUTS.index(r["layout"]), r["n_train"], r["seed"]))
    H.write_csv("exp28_acting_latency.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
