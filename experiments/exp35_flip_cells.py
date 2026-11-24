"""Experiment 35: learning to play from success and failure by flipping one cell of the machine.

Lucas's idea 3, second form, in experiment 28's wait game (key/door, 60 game
steps). Experiment 31 redrew weights inside their boxes on failure and did
not learn (0.17 laps). Here the player is the network built from a machine
(letter bits -> state neurons -> 2 output neurons, dnn_letters
.make_layout_bits), and exploration changes the machine itself: each round,
one (state, letter) cell is changed at random, either its move (to one of
the 3 others) or its next state (to another state). The changed player and
the current one play the same 8 new games; if the changed one scores at
least as many laps it is a success and is kept, otherwise a failure and the
change is undone. 300 rounds. Starts:
  scratch2, scratch3: a random machine of 2 or 3 states over the given
                      letter (P:top, P:right): no demonstrations, but the
                      letter is designed;
  imit2, imit4:       experiment 30's pipeline on the first 2 or 4
                      demonstration episodes of experiment 28 (letter from
                      experiment 23's 32 regions and machine by search, at
                      most 3 states), padded to 3 states (a spare state
                      copies state 0's moves and stays put; a flip of a
                      next state can reach it), then rounds of flips.
During the rounds the machine is played directly, as the network runs it
(the move answers the current screen; one transition per network step, so
3 per screen). At the end the network is built (64 random-walk episodes and
64 new games played by the machine, relabelled by it, then the output
linear programs) and plays the
test games with midpoint weights: net_laps must equal the machine's laps.
Scores: laps per game on experiment 26's 100 test games after 0 to 300
rounds; references expert 14.28, random 1.45, experiment 31 0.17. 10 seeds.
Run from the repository root:  python experiments/exp35_flip_cells.py
Writes experiments/results/exp35_flip_cells.csv.
"""
import time
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import exp26_imitate_expert as e26
import exp28_acting_latency as e28
import exp31_success_failure as e31
import dnn
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M
import dnn_noise as N

N_IN, N_OUT, REP, H_OUT = e28.N_IN, e28.N_OUT, 3, 3
ROUNDS, GAMES_PER_ROUND = 300, 8
REPORT = [0, 10, 25, 50, 100, 200, 300]
SEEDS = range(10)
STARTS = ["scratch2", "scratch3", "imit2", "imit4"]
MAX_STATES = 3
MOVE_LIST = [(1, 1), (1, 0), (0, 1), (0, 0)]
LETTERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def machine_policy(rules, delta, out):
    """The machine as the network runs it in the wait game: the move answers
    the current screen's letter, and the state then takes one transition per
    network step, so 3 per screen. Unseen letters keep the state and move up."""
    state = {"q": 0}

    def pol(t, screens):
        if t == 0:
            state["q"] = 0
        x = e26.screen(*screens[-1])
        a = tuple(int(e23.U_fire(r, x[None])[0]) for r in rules)
        q = state["q"]
        mv = out[q].get(a, (1, 1))
        for _ in range(REP):
            q = delta[q].get(a, q)
        state["q"] = q
        return mv
    return pol


def laps(rules, delta, out, seeds):
    return [e26.play(machine_policy(rules, delta, out), np.random.default_rng(int(g))) for g in seeds]


def start_machine(start, seed):
    """(rules, letter name, delta, out) of the first player."""
    rng = np.random.default_rng(35_000 + seed)
    if start.startswith("scratch"):
        n = int(start[-1])
        rules = [e23.FEATURE[nm] for nm in e26.TRUE]
        delta = [{a: int(rng.integers(0, n)) for a in LETTERS} for _ in range(n)]
        out = [{a: MOVE_LIST[rng.integers(0, 4)] for a in LETTERS} for _ in range(n)]
        return rules, "+".join(e26.TRUE), delta, out, n
    k = int(start[-1])
    drng = np.random.default_rng(seed)               # experiment 28's episodes
    V = [e28.episode(drng, e28.T + 1, REP, H_OUT) for _ in range(max(e28.SIZES))][:k]
    idx, delta, out, _, _ = N.choose_bits_search(V, N_IN, e23.FEATURES, H_OUT, n_out=N_OUT, max_states=3)
    rules = [e23.FEATURES[i][1] for i in idx]
    letters = sorted({a for row in delta for a in row} | set(LETTERS if len(idx) == 2 else [(0,), (1,)]))
    n_found = len(delta)
    delta = M.complete(delta, len(idx))
    out = [{a: row.get(a, (1, 1)) for a in letters} for row in out]
    for q in range(len(delta), MAX_STATES):         # spare states, reachable only after a flip
        delta.append({a: q for a in letters})
        out.append(dict(out[0]))
    return rules, "+".join(e23.FEATURES[i][0] for i in idx), delta, out, n_found


def flip(delta, out, rng):
    """A copy with one cell changed: its move (prob 1/2, always with one
    state) or its next state."""
    delta, out = [dict(r) for r in delta], [dict(r) for r in out]
    n = len(delta)
    q = int(rng.integers(0, n))
    cells = sorted(delta[q])
    a = cells[int(rng.integers(0, len(cells)))]
    if n == 1 or rng.random() < 0.5:
        others = [m for m in MOVE_LIST if m != out[q][a]]
        out[q][a] = others[int(rng.integers(0, 3))]
    else:
        others = [p for p in range(n) if p != delta[q][a]]
        delta[q][a] = others[int(rng.integers(0, n - 1))]
    return delta, out


def own_games(rules, delta, out, seed, n=64):
    """Wait-game streams of n new games played by the machine (its own
    games: screens shown 3 steps, its moves as outputs)."""
    seeds = np.random.default_rng(80_000 + seed).integers(1 << 30, size=n)
    V = []
    for g in seeds:
        pol, moves = e31.recorded(machine_policy(rules, delta, out))
        game = []
        e26.play(lambda t, sc: (game.__setitem__(slice(None), sc), pol(t, sc))[1], np.random.default_rng(int(g)))
        V.append(e31.own_stream(game, moves))
    return V


def build_net(rules, delta, out, seed):
    """The network for the machine: 64 random-walk wait episodes and 64 of
    its own games, relabelled by it, then experiment 28's layout and linear
    programs."""
    rng = np.random.default_rng(70_000 + seed)
    V = [e28.episode(rng, e28.T + 1, REP, H_OUT) for _ in range(64)] + own_games(rules, delta, out, seed)
    bits = [lambda X, r=r: e23.U_fire(r, X) for r in rules]
    V = N.relabel(V, N_IN, N_OUT, H_OUT, bits, delta, out)
    lay, index = Lm.make_layout_bits(N_IN, rules, delta, n_out=N_OUT)
    net = F.make_net(lay)
    blocked = [j for j in range(lay.n_visible) if j not in index.values()]
    for o in range(N_IN, N_IN + N_OUT):
        net.L[o, blocked] = net.U[o, blocked] = 0.0
    F.train(net, lay, V, H.EPS)
    return net, lay


def one_run(args):
    start, seed = args
    t0 = time.perf_counter()
    rules, name, delta, out, start_states = start_machine(start, seed)
    start_secs = time.perf_counter() - t0
    test_seeds = np.random.default_rng(40_000 + seed).integers(1 << 30, size=e26.GAMES)
    rng = np.random.default_rng(31_000 + seed)
    row = dict(start=start, seed=seed, letter=name, states=len(delta), start_states=start_states, start_seconds=round(start_secs, 4))
    successes = 0
    t0 = time.perf_counter()
    for rnd in range(ROUNDS + 1):
        if rnd in REPORT:
            row[f"laps_{rnd}"] = float(np.mean(laps(rules, delta, out, test_seeds)))
        if rnd == ROUNDS:
            break
        games = rng.integers(1 << 30, size=GAMES_PER_ROUND)
        d2, o2 = flip(delta, out, rng)
        if sum(laps(rules, d2, o2, games)) >= sum(laps(rules, delta, out, games)):
            delta, out = d2, o2
            successes += 1
    row["rounds_seconds"] = round(time.perf_counter() - t0, 4)
    true_delta, true_out = e26.machine()
    key = lambda x: tuple(int(e23.U_fire(r, np.array([x]))[0]) for r in rules)
    true_key = lambda x: tuple(int(e23.U_fire(e23.FEATURE[n], np.array([x]))[0]) for n in e26.TRUE)
    t0 = time.perf_counter()
    net, lay = build_net(rules, delta, out, seed)
    mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
    net_laps = [e26.play(e28.wait_policy(net, lay, mid, REP), np.random.default_rng(int(g))) for g in test_seeds]
    row.update(successes=successes, hidden_added=net.n - lay.n_visible,
               net_laps=float(np.mean(net_laps)), net_equals_machine=bool(np.isclose(np.mean(net_laps),
                                                                                    row[f"laps_{ROUNDS}"])),
               machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key, e23.SCREENS),
               build_seconds=round(time.perf_counter() - t0, 4))
    return row


def summarise(rows):
    rep = [r for r in REPORT if f"laps_{r}" in rows[0]]
    print("\nMean laps per game on the 100 test games after each number of rounds (expert 14.28, random 1.45, "
          "experiment 31: 0.17)\n")
    print("| start | " + " | ".join(str(r) for r in rep) + " | runs >= 14 at end | true machine |")
    print("|" + "---|" * (len(rep) + 3))
    for st in STARTS:
        g = [r for r in rows if r["start"] == st]
        if g:
            print(f"| {st} | " + " | ".join(f"{np.mean([r[f'laps_{k}'] for r in g]):.2f}" for k in rep)
                  + f" | {sum(r[f'laps_{rep[-1]}'] >= 14 for r in g)} | {sum(r['machine_correct'] for r in g)} |")
    print(f"\nnetwork plays like its machine: {sum(r['net_equals_machine'] for r in rows)} of {len(rows)}; "
          f"hidden added: {sum(r['hidden_added'] for r in rows)}; mean successes: "
          f"{np.mean([r['successes'] for r in rows]):.0f}")
    print("letters of imitation starts: " + ", ".join(sorted({r['letter'] for r in rows if r['start'].startswith('imit')})))
    print(f"max seconds: start {max(r['start_seconds'] for r in rows):.1f}, "
          f"rounds {max(r['rounds_seconds'] for r in rows):.1f}, build {max(r['build_seconds'] for r in rows):.1f}")


def main(seeds=SEEDS, starts=STARTS):
    jobs = [(st, s) for st in starts for s in seeds]
    with Pool() as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    rows.sort(key=lambda r: (STARTS.index(r["start"]), r["seed"]))
    H.write_csv("exp35_flip_cells.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
