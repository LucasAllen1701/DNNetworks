"""Experiment 23: a tiny grid game (ladder rung 3): screen helpers, state neurons and a witness buffer.

The screen is a 4 x 4 grid in two planes (32 inputs): the player's plane and
a monster's plane, each with one black pixel. Both walk at random (up, down,
left or right, each with probability 1/4; a move into a wall stays put)
from random starts. An episode (stream) is 25 screens. The output, 3 steps later, is a game signal that needs memory:
  key_door:  the player is on the door (bottom-left quadrant) after having
             been on the key (top-right quadrant);
  lap:       the player enters the right half after being in the left half;
  count2:    every second time the player enters the top row;
  monster_key: the monster is in the left half and the player has been on
             the key;
  parity_top: the number of steps the player has spent in the top half is
             odd;
  both_bottom: the game is still on: the player and the monster have never
             been in the bottom row at the same time.
Each task is a machine (2 to 4 states) reading a letter of 1 or 2 bits, each
bit a region of one plane. Layout (h = 3, dnn_letters.make_layout_bits): each
letter bit is one fixed neuron (the OR of the region's pixels); the state
neurons read the bits. "bits" chooses the letter from a library of 32 regions
(rows, columns, halves and quadrants of each plane): every set of 1 or 2
regions (528 letters) whose letter streams are consistent is merged, and the
fewest states win. "true_bits" is given the true regions. Raw state merging
(experiment 13) is not run: its letters would be 2^32 input patterns.
Data arrive as 32 streams of 25 screens, 4 per round; after every second
round the network is rebuilt from the buffer (a sleep, experiment 15). The
buffer is cut to its cap before each sleep (experiment 18): "all" (no cap),
"recent" or "witness" with caps 16 and 8. "witness" first chooses the
letter from the whole buffer, then keeps the streams that its merged
machine needs (those that ruled out a merge or showed a transition; the
oldest go first), then rebuilds from what is kept. If no letter is
consistent on the whole buffer, it keeps the most recent streams. 10 seeds; after every sleep: whether
the letter is the true regions, whether the machine is the true one (on
every stream of screens: a different letter can give the same outputs),
unseen test streams exact (200
streams, no key, midpoint weights), and recall without key of every stream
seen so far.
Run from the repository root:  python experiments/exp23_grid_game.py
Writes experiments/results/exp23_grid_game.csv.
"""
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

import harness as H
import exp16_or_of_ands as e16
import dnn_fixed as F
import dnn_letters as Lm
import dnn_merge as M

dnn = H.dnn
SIDE = 4
N_PIX = SIDE * SIDE
N_IN, T, h = 2 * N_PIX, 24, 3
STREAMS, PER_ROUND, SLEEP_EVERY = 32, 4, 2
SEEDS = range(10)
TEST_STREAMS = 200
POLICIES = [("bits", "all", None), ("true_bits", "all", None), ("bits", "recent", 16), ("bits", "witness", 16),
            ("bits", "recent", 8), ("bits", "witness", 8)]


def _regions():
    rows = {f"row{r}": [(r, c) for c in range(SIDE)] for r in range(SIDE)}
    cols = {f"col{c}": [(r, c) for r in range(SIDE)] for c in range(SIDE)}
    half = SIDE // 2
    halves = {"top": [(r, c) for r in range(half) for c in range(SIDE)],
              "bottom": [(r, c) for r in range(half, SIDE) for c in range(SIDE)],
              "left": [(r, c) for r in range(SIDE) for c in range(half)],
              "right": [(r, c) for r in range(SIDE) for c in range(half, SIDE)]}
    quads = {f"{a}{b}": [(r, c) for r in rr for c in cc]
             for a, rr in (("T", range(half)), ("B", range(half, SIDE)))
             for b, cc in (("L", range(half)), ("R", range(half, SIDE)))}
    return {**rows, **cols, **halves, **quads}


def features():
    """[(name, rule)]: for each plane and region, the OR of its pixels."""
    out = []
    for p, off in (("P", 0), ("M", N_PIX)):
        for name, cells in _regions().items():
            w = np.zeros(N_IN, int)
            w[[off + r * SIDE + c for r, c in cells]] = 1
            out.append((f"{p}:{name}", (w, 0)))
    return out


FEATURES = features()
FEATURE = dict(FEATURES)


def _key_door(q, a):
    key, door = a
    return int(q or key), int(q and door)


def _lap(q, a):
    return (0, q) if a[0] else (1, 0)


def _count2(q, a):
    c, inside = q
    if not a[0]:
        return (c, 0), 0
    if inside:
        return (c, 1), 0
    c = (c + 1) % 2
    return (c, 1), int(c == 0)


def _monster_key(q, a):
    key, mleft = a
    q2 = int(q or key)
    return q2, int(q2 and mleft)


def _parity(q, a):
    return q ^ a[0], q ^ a[0]


def _both_bottom(q, a):
    q2 = int(q or (a[0] and a[1]))
    return q2, 1 - q2


TASKS = {                    # name: (true regions, step, start state)
    "key_door": (["P:TR", "P:BL"], _key_door, 0),
    "lap": (["P:right"], _lap, 0),
    "count2": (["P:row0"], _count2, (0, 0)),
    "monster_key": (["P:TR", "M:left"], _monster_key, 0),
    "parity_top": (["P:top"], _parity, 0),
    "both_bottom": (["P:row3", "M:row3"], _both_bottom, 0),
}


def machine(task):
    """(delta, out) of the task over its letter tuples, start state 0."""
    regions, step, start = TASKS[task]
    letters = list(product((0, 1), repeat=len(regions)))
    index, todo, delta, out = {start: 0}, [start], [], []
    while todo:
        q = todo.pop(0)
        drow, orow = {}, {}
        for a in letters:
            q2, o = step(q, a)
            if q2 not in index:
                index[q2] = len(index)
                todo.append(q2)
            drow[a], orow[a] = index[q2], o
        delta.append(drow)
        out.append(orow)
    return delta, out


def walk(rng, n):
    pos = rng.integers(0, SIDE, 2)
    path = []
    for _ in range(n):
        path.append(pos.copy())
        d = [(-1, 0), (1, 0), (0, -1), (0, 1)][rng.integers(0, 4)]
        pos = np.clip(pos + d, 0, SIDE - 1)
    return path


def screens(rng, n):
    x = np.zeros((n, N_IN), int)
    for off in (0, N_PIX):
        for t, (r, c) in enumerate(walk(rng, n)):
            x[t, off + r * SIDE + c] = 1
    return x


def stream(x, task):
    """Screens x and the output after reading x(0..t), at t + h."""
    delta, out = machine(task)
    bits = [U_fire(FEATURE[r], x) for r in TASKS[task][0]]
    y = np.zeros(len(x), int)
    q = 0
    for t in range(len(x) - h):
        a = tuple(int(b[t]) for b in bits)
        y[t + h] = out[q][a]
        q = delta[q][a]
    return np.c_[x, y]


def U_fire(rule, X):
    w, b = rule
    return (np.asarray(X) @ w + b >= 1).astype(int)


def all_screens():
    out = []
    for p, m in product(range(N_PIX), repeat=2):
        x = np.zeros(N_IN, int)
        x[p] = x[N_PIX + m] = 1
        out.append(tuple(x))
    return out


SCREENS = all_screens()


def witness_evict(buffer, cap, bit_rules):
    """experiment 18's witness policy on the letter streams of the chosen
    bits (h = 3): drop the oldest streams while the merged machine stays the
    same. Returns (kept streams, witnesses dropped)."""
    if len(buffer) <= cap:
        return list(buffer), 0
    k = len(bit_rules)
    ls = [np.c_[np.array([U_fire(r, s[:, :N_IN]) for r in bit_rules]).T, s[:, N_IN]] for s in buffer]
    ref = M.learn_machine(ls, k, h)
    keep = list(range(len(buffer)))
    for i in range(len(buffer)):
        if len(keep) <= cap:
            break
        trial = [j for j in keep if j != i]
        m = M.learn_machine([ls[j] for j in trial], k, h)
        if M.same_machine(*m, *ref) and M.same_machine(*ref, *m):
            keep = trial
    return [buffer[j] for j in keep[-cap:]], max(len(keep) - cap, 0)


def cut_recent(buffer, cap, layout, task):
    return buffer[-cap:], ""


def cut_witness(buffer, cap, layout, task):
    """Witnesses of the whole buffer's machine (most recent if no letter)."""
    full = build(buffer, layout, task)
    if full is None:
        return buffer[-cap:], ""
    return witness_evict(buffer, cap, full[1])


def build(buffer, layout, task):
    """(bit names, bit rules, delta, out, consistent letters) from the buffer."""
    if layout == "true_bits":
        names = TASKS[task][0]
        rules = [FEATURE[n] for n in names]
        delta, out = M.learn_machine(Lm.bit_streams(buffer, N_IN, [lambda X, r=r: U_fire(r, X) for r in rules]),
                                     len(rules), h)
        return names, rules, delta, out, ""
    found = Lm.choose_bits(buffer, N_IN, FEATURES, h)
    if found is None:
        return None
    idx, delta, out, n_ok = found
    return [FEATURES[i][0] for i in idx], [FEATURES[i][1] for i in idx], delta, out, n_ok


def one_run(args):
    task, layout, policy, cap, seed = args
    rng = np.random.default_rng(seed)
    V = [stream(screens(rng, T + 1), task) for _ in range(STREAMS)]
    trng = np.random.default_rng(80_000 + seed)
    tests = [stream(screens(trng, T + 1), task) for _ in range(TEST_STREAMS)]
    true_delta, true_out = machine(task)
    true_rules = [FEATURE[n] for n in TASKS[task][0]]
    true_key = lambda x: tuple(int(U_fire(r, np.array([x]))[0]) for r in true_rules)
    seen, buffer, rows = [], [], []
    for r in range(1, STREAMS // PER_ROUND + 1):
        batch = V[(r - 1) * PER_ROUND: r * PER_ROUND]
        seen += batch
        buffer += batch
        if r % SLEEP_EVERY:
            continue
        row = dict(task=task, layout=layout, policy=policy, cap=cap or "", seed=seed, round=r,
                   streams_seen=len(seen))
        dropped = ""
        if cap and len(buffer) > cap:
            buffer, dropped = CUTS[policy](buffer, cap, layout, task)
        t0 = time.perf_counter()
        found = build(buffer, layout, task)
        row.update(buffer=len(buffer), witnesses_dropped=dropped)
        if found is None:
            row.update(letter="none", letter_correct=False, machine_correct=False)
            rows.append(row)
            continue
        names, rules, delta, out, n_ok = found
        lay, index = Lm.make_layout_bits(N_IN, rules, M.complete(delta, len(rules)))
        build_secs = time.perf_counter() - t0
        net = F.make_net(lay)
        blocked = [j for j in range(lay.n_visible) if j not in index.values()]
        net.L[N_IN, blocked] = net.U[N_IN, blocked] = 0.0
        t0 = time.perf_counter()
        trained = F.train(net, lay, buffer, H.EPS)
        train_secs = time.perf_counter() - t0
        key = lambda x: tuple(int(U_fire(rr, np.array([x]))[0]) for rr in rules)
        mid = (dnn.sample_weights(net, mode="lower") + dnn.sample_weights(net, mode="upper")) / 2
        preds = [e16.run_out(net, mid, F.extend(lay, s), N_IN) for s in tests]
        exact = [(p[h:] == s[h:, N_IN]).all() for p, s in zip(preds, tests)]
        seen_ext = [F.extend(lay, s) for s in seen]
        row.update(letter="+".join(names), candidates_consistent=n_ok,
                   letter_correct=sorted(names) == sorted(TASKS[task][0]),
                   states_found=len(delta), states_true=len(true_delta),
                   machine_correct=Lm.same_composite(delta, out, key, true_delta, true_out, true_key, SCREENS),
                   hidden_added=net.n - lay.n_visible,
                   build_seconds=round(build_secs, 4), train_seconds=round(train_secs, 4),
                   recall_without_key=H.score(net, H.zero_hidden_starts(seen_ext), seen_ext, [N_IN], rng)[0],
                   unseen_exact_mid=float(np.mean(exact)))
        rows.append(row)
    return rows


# buffer policies: (buffer, cap, layout, task) -> (kept streams, witnesses dropped)
CUTS = {"recent": cut_recent, "witness": cut_witness}


def _sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def summarise(rows):
    sleeps = list(range(SLEEP_EVERY, STREAMS // PER_ROUND + 1, SLEEP_EVERY))
    print("\nAfter each sleep: runs with the true machine (of 10) / mean fraction of unseen streams exact\n")
    print("| task | layout | buffer | " + " | ".join(f"{r * PER_ROUND} streams" for r in sleeps) + " |")
    print("|" + "---|" * (len(sleeps) + 3))
    for task in TASKS:
        for layout, policy, cap in POLICIES:
            g = _sel(rows, task=task, layout=layout, policy=policy, cap=cap or "")
            cells = []
            for r in sleeps:
                gr = _sel(g, round=r)
                acc = [x["unseen_exact_mid"] for x in gr if "unseen_exact_mid" in x]
                cells.append(f"{sum(x['machine_correct'] for x in gr)} / {np.mean(acc):.2f}" if acc else "–")
            print(f"| {task} | {layout} | {policy} {cap or ''} | " + " | ".join(cells) + " |")
    last = sleeps[-1]
    n = len(_sel(rows, layout="bits", policy="all", round=last))
    print(f"\nAt the end (of {n} runs): runs whose letter is the true regions, runs with the true machine "
          "(a different letter can give it), mean unseen exact, mean recall of all streams seen (no key), "
          "runs that dropped a witness\n")
    print("| layout | buffer | true regions | true machine | unseen exact | recall | dropped a witness |")
    print("|" + "---|" * 7)
    for layout, policy, cap in POLICIES:
        g = _sel(rows, layout=layout, policy=policy, cap=cap or "", round=last)
        ok = [x for x in g if "unseen_exact_mid" in x]
        drop = sum(1 for x in g if x["witnesses_dropped"] not in ("", 0))
        print(f"| {layout} | {policy} {cap or ''} | {sum(x['letter_correct'] for x in g)} | "
              f"{sum(x['machine_correct'] for x in g)} | {np.mean([x['unseen_exact_mid'] for x in ok]):.3f} | "
              f"{np.mean([x['recall_without_key'] for x in ok]):.3f} | {drop if policy == 'witness' else '–'} |")
    done = [x for x in rows if "unseen_exact_mid" in x]
    print(f"\nhidden neurons added by TRAIN: {sum(x['hidden_added'] > 0 for x in done)} sleeps; "
          f"no consistent letter: {sum(x['letter'] == 'none' for x in rows)} sleeps")
    print(f"max seconds: build {max(x['build_seconds'] for x in done):.2f}, "
          f"train {max(x['train_seconds'] for x in done):.2f}")


def main(tasks=TASKS, seeds=SEEDS):
    jobs = [(t, l, p, c, s) for t in tasks for l, p, c in POLICIES for s in seeds]
    with Pool() as pool:
        rows = [r for group in pool.map(one_run, jobs, chunksize=1) for r in group]
    H.write_csv("exp23_grid_game.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
