"""Experiment 25: witnesses for the letter choice (follow-up to experiment 23).

In experiment 23, a bounded buffer that kept the witnesses of the merged
machine still ended wrong in 18 runs, and in 17 of them because the letter
chosen from the kept streams differed from the one chosen from the whole
buffer: the streams that ruled out other letters had been dropped. The new
policy "letter_witness" keeps those too:
  1. choose the letter on the whole buffer (as "witness");
  2. keep the machine's witnesses: drop streams, oldest first, while the
     merged machine of that letter stays the same (no cap yet);
  3. if they exceed the cap, keep the newest of them (the rest count as
     witnesses dropped); else fill up to the cap with the most recent
     streams;
  4. while the letter chosen from the kept streams (the rival) is not that
     letter, swap the oldest filler stream for the stream that makes the
     rival inconsistent, or else gives it the most states (newest first on
     ties), until no filler is left or no stream helps.
Same game, tasks, seeds and schedule as experiment 23 (32 episodes, 4 per
round, a sleep every 2 rounds). Buffers: all; witness and letter_witness
with caps 16, 8 and 4; recent with cap 4. The all and witness rows (caps 16
and 8) should equal experiment 23's.
Run from the repository root:  python experiments/exp25_letter_witnesses.py
Writes experiments/results/exp25_letter_witnesses.csv.
"""
from multiprocessing import Pool

import numpy as np

import harness as H
import exp23_grid_game as e23
import dnn_letters as Lm
import dnn_merge as M

TASKS = e23.TASKS
SEEDS = e23.SEEDS
POLICIES = [("bits", "all", None), ("bits", "witness", 16), ("bits", "letter_witness", 16),
            ("bits", "witness", 8), ("bits", "letter_witness", 8),
            ("bits", "recent", 4), ("bits", "witness", 4), ("bits", "letter_witness", 4)]


def _letter_streams(streams, rules):
    return [np.c_[np.array([e23.U_fire(r, s[:, :e23.N_IN]) for r in rules]).T, s[:, e23.N_IN]] for s in streams]


def _states(streams, rules):
    """Merged states of the letter streams, or None if inconsistent."""
    try:
        return len(M.learn_machine(_letter_streams(streams, rules), len(rules), e23.h)[0])
    except ValueError:
        return None


def cut_letter_witness(buffer, cap, layout, task):
    full = e23.build(buffer, layout, task)
    if full is None:
        return buffer[-cap:], ""
    names, rules = full[0], full[1]
    ls = _letter_streams(buffer, rules)
    k = len(rules)
    ref = M.learn_machine(ls, k, e23.h)
    keep = list(range(len(buffer)))
    for i in range(len(buffer)):               # 2. the machine's witnesses
        trial = [j for j in keep if j != i]
        m = M.learn_machine([ls[j] for j in trial], k, e23.h)
        if M.same_machine(*m, *ref) and M.same_machine(*ref, *m):
            keep = trial
    keep = sorted(keep)
    if len(keep) >= cap:                       # the witnesses do not fit
        return [buffer[j] for j in keep[-cap:]], len(keep) - cap
    W = set(keep)
    for j in reversed(range(len(buffer))):     # 3. fill with the most recent
        if len(keep) >= cap:
            break
        if j not in W:
            keep.append(j)
    letter_w = set()
    for _ in range(len(buffer)):               # 4. the letter's witnesses
        rival = e23.build([buffer[j] for j in sorted(keep)], layout, task)
        if rival is None or rival[0] == names:
            break
        base = _states([buffer[j] for j in keep], rival[1])
        best = None
        for j in reversed(range(len(buffer))):
            if j in keep:
                continue
            n = _states([buffer[i] for i in keep + [j]], rival[1])
            score = float("inf") if n is None else n
            if score > base and (best is None or score > best[1]):
                best = (j, score)
        free = sorted(j for j in keep if j not in W and j not in letter_w)
        if best is None or not free:
            break
        keep.remove(free[0])                   # the oldest filler makes room
        keep.append(best[0])
        letter_w.add(best[0])
    return [buffer[j] for j in sorted(keep)], 0


e23.CUTS["letter_witness"] = cut_letter_witness


def one_run(args):
    return e23.one_run(args)


def summarise(rows):
    sleeps = sorted({r["round"] for r in rows})
    print("\nAfter each sleep: runs with the true machine (of 10) / mean fraction of unseen streams exact\n")
    print("| task | buffer | " + " | ".join(f"{r * e23.PER_ROUND} streams" for r in sleeps) + " |")
    print("|" + "---|" * (len(sleeps) + 2))
    for task in TASKS:
        for layout, policy, cap in POLICIES:
            g = e23._sel(rows, task=task, layout=layout, policy=policy, cap=cap or "")
            cells = []
            for r in sleeps:
                gr = e23._sel(g, round=r)
                acc = [x["unseen_exact_mid"] for x in gr if "unseen_exact_mid" in x]
                cells.append(f"{sum(x['machine_correct'] for x in gr)} / {np.mean(acc):.2f}" if acc else "–")
            print(f"| {task} | {policy} {cap or ''} | " + " | ".join(cells) + " |")
    last = sleeps[-1]
    n = len(e23._sel(rows, policy="all", round=last))
    print(f"\nAt the end (of {n} runs): true machine, mean unseen exact, mean recall of all streams seen "
          "(no key), runs that dropped a witness, mean witnesses dropped\n")
    print("| buffer | true machine | unseen exact | recall | dropped a witness | mean dropped |")
    print("|" + "---|" * 6)
    for layout, policy, cap in POLICIES:
        g = e23._sel(rows, layout=layout, policy=policy, cap=cap or "", round=last)
        ok = [x for x in g if "unseen_exact_mid" in x]
        drops = [x["witnesses_dropped"] for x in g if x["witnesses_dropped"] != ""]
        print(f"| {policy} {cap or ''} | {sum(x['machine_correct'] for x in g)} | "
              f"{np.mean([x['unseen_exact_mid'] for x in ok]):.3f} | "
              f"{np.mean([x['recall_without_key'] for x in ok]):.3f} | "
              f"{sum(d > 0 for d in drops) if drops else '–'} | {np.mean(drops) if drops else 0:.1f} |")
    done = [x for x in rows if "build_seconds" in x]
    print(f"\nmax seconds: build {max(x['build_seconds'] for x in done):.2f}")


def main(tasks=TASKS, seeds=SEEDS):
    jobs = [(t, l, p, c, s) for t in tasks for l, p, c in POLICIES for s in seeds]
    jobs.sort(key=lambda j: j[2] != "letter_witness")          # slow runs first
    with Pool() as pool:
        rows = [r for group in pool.map(one_run, jobs, chunksize=1) for r in group]
    pol = [(p, c or "") for _, p, c in POLICIES]
    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), pol.index((r["policy"], r["cap"])), r["seed"], r["round"]))
    H.write_csv("exp25_letter_witnesses.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
