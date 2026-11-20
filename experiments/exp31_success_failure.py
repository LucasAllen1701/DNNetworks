"""Experiment 31: learning to play from success and failure, without an expert.

Lucas's idea 3 (15 November), in experiment 28's "wait" game (key/door, each
screen shown 3 network steps, 60 game steps per game). No expert moves are
ever seen. Each run plays 50 rounds of one game:
  - the player is a random mover until the first success, then the network;
  - a game is a success if it scores at least `target` laps, where target
    is the most laps of any success so far (at least 1);
  - success: the game (screens and the moves the player made) joins the
    buffer and the network is rebuilt and trained on the whole buffer, as
    in experiment 30's game pipeline (letter and machine by search, at
    most 3 states, then relabelling and the linear programs), and plays
    with the midpoint of its boxes;
  - failure: every weight is redrawn uniformly inside its box
    (dnn.sample_weights mode "random"), so the network still replays its
    buffer but acts differently on screens it has not seen.
After every round the current network (midpoint weights) is scored on
experiment 26's 100 test games. References: imitation of the expert
(experiment 28, 14.28 laps) and the random player (1.45). 10 seeds.
Run from the repository root:  python experiments/exp31_success_failure.py
Writes experiments/results/exp31_success_failure.csv.
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

ROUNDS = 50
SEEDS = range(10)
REP, H_OUT = 3, 3
N_IN, N_OUT = e28.N_IN, e28.N_OUT
GAMES, GAME_STEPS = e26.GAMES, e26.GAME_STEPS
REPORT = [1, 5, 10, 20, 30, 40, 50]


def own_stream(screens, moves):
    """A wait-game training stream from a played game: each screen 3 steps,
    the move made on screen k answering each of its steps 3 steps later."""
    ticks = [s for s in screens for _ in range(REP)]
    x = np.array([e26.screen(p, m) for p, m in ticks])
    y = np.zeros((len(ticks), N_OUT), int)
    for t in range(len(ticks) - H_OUT):
        y[t + H_OUT] = moves[t // REP]
    return np.c_[x, y]


def recorded(policy):
    """policy plus the list of moves it returned."""
    moves = []

    def pol(t, screens):
        mv = policy(t, screens)
        moves.append(mv)
        return mv
    return pol, moves


def build(buffer):
    """(net, layout, letter, states) trained on the buffer, as experiment 30."""
    V = [s.copy() for s in buffer]
    idx, delta, out, err, _ = N.choose_bits_search(V, N_IN, e23.FEATURES, H_OUT, n_out=N_OUT, max_states=3)
    rules = [e23.FEATURES[i][1] for i in idx]
    V = N.relabel(V, N_IN, N_OUT, H_OUT, [lambda X, r=r: e23.U_fire(r, X) for r in rules], delta, out)
    lay, index = Lm.make_layout_bits(N_IN, rules, M.complete(delta, len(rules)), n_out=N_OUT)
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in index.values()]
    for o in range(N_IN, N_IN + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    F.train(net, lay, V, H.EPS)
    return net, lay, "+".join(e23.FEATURES[i][0] for i in idx), len(delta), err


def midpoint(net):
    return (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2


def one_run(args):
    seed, rounds = args
    rng = np.random.default_rng(31_000 + seed)
    eval_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=GAMES)
    buffer, target, net, lay, A = [], 1, None, None, None
    letter, states, err = "", 0, ""
    rows = []
    t_build = 0.0
    for rnd in range(1, rounds + 1):
        if net is None:
            base = lambda t, sc: tuple(int(b) for b in rng.integers(0, 2, 2))
        else:
            base = e28.wait_policy(net, lay, A, REP)
        pol, moves = recorded(base)
        game = []
        laps = e26.play(lambda t, sc: (game.__setitem__(slice(None), sc), pol(t, sc))[1],
                        np.random.default_rng(int(rng.integers(1 << 30))))
        success = laps >= target
        if success:
            target = max(target, laps)
            buffer.append(own_stream(game, moves))
            t0 = time.perf_counter()
            net, lay, letter, states, err = build(buffer)
            t_build += time.perf_counter() - t0
            A = midpoint(net)
        elif net is not None:
            A = dnn.sample_weights(net, rng, mode="random")
        if net is None:
            eval_laps = ""
        else:
            mid = midpoint(net)
            eval_laps = float(np.mean([e26.play(e28.wait_policy(net, lay, mid, REP), np.random.default_rng(g))
                                       for g in eval_seeds]))
        rows.append(dict(seed=seed, round=rnd, laps=laps, success=success, target=target, buffer=len(buffer),
                         letter=letter, states=states, train_disagreements=err,
                         hidden=(net.n - lay.n_visible) if net is not None else "",
                         eval_laps=eval_laps, build_seconds=round(t_build, 4)))
    return rows


def summarise(rows):
    print("\nMean laps of the current network on the 100 test games (midpoint weights; random player = 0 "
          "before the first success) / successes so far, by round\n")
    print("| | " + " | ".join(str(r) for r in REPORT) + " |")
    print("|" + "---|" * (len(REPORT) + 1))
    cells = []
    for k in REPORT:
        g = [r for r in rows if r["round"] == k]
        if not g:
            cells.append("–")
            continue
        laps = [r["eval_laps"] if r["eval_laps"] != "" else 0.0 for r in g]
        cells.append(f"{np.mean(laps):.2f} / {np.mean([r['buffer'] for r in g]):.1f}")
    print("| mean of seeds | " + " | ".join(cells) + " |")
    print("\nPer seed at the last round: test laps, buffer, target, letter, states\n")
    last = max(r["round"] for r in rows)
    for r in [r for r in rows if r["round"] == last]:
        print(f"seed {r['seed']}: {r['eval_laps']} laps, buffer {r['buffer']}, target {r['target']}, "
              f"{r['letter']}, {r['states']} states")
    best = [max((r["eval_laps"] or 0.0) for r in rows if r["seed"] == s) for s in sorted({r["seed"] for r in rows})]
    print(f"\nbest test laps reached in any round, per seed: " + ", ".join(f"{b:.2f}" for b in best))
    print(f"max build seconds per run: {max(r['build_seconds'] for r in rows):.1f}")


def main(seeds=SEEDS, rounds=ROUNDS):
    with Pool() as pool:
        rows = [r for rs in pool.map(one_run, [(s, rounds) for s in seeds], chunksize=1) for r in rs]
    rows.sort(key=lambda r: (r["seed"], r["round"]))
    H.write_csv("exp31_success_failure.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
