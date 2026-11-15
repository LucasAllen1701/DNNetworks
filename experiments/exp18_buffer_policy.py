"""Experiment 18: which streams may leave a bounded buffer (witnesses).

Experiment 15 rebuilt the network from a buffer at every sleep; with the 16
most recent streams, 3 of 60 runs (all tomita4) ended with a wrong machine,
because the streams that had ruled it out had left the buffer. Here the
buffer is cut to its cap just before every sleep, by one of these policies
(dnn_sleep.evict):
  "all":     no cap (experiment 15's "sleep").
  "recent":  the most recent streams (experiment 15's "sleep16" for cap 16).
  "random":  a random subset.
  "witness": streams are dropped oldest first, but only while the machine
             merged from the rest equals the machine merged from the whole
             buffer; the streams that cannot go are that machine's
             witnesses. If there are more witnesses than the cap, the oldest
             go.
The six tasks of experiment 15, 32 streams arriving 4 per round, a sleep
after every second round (4 sleeps), caps 16 and 8, 20 seeds (seeds 0-9 are
experiment 15's data). Wake rounds are left out: a sleep rebuilds from the
buffer alone, so they do not change the machine. After every sleep: whether
the true machine was found, unseen streams exact (midpoint), recall without
key of every stream seen so far, and for "witness" how many witnesses had to
be dropped.
Run from the repository root:  python experiments/exp18_buffer_policy.py
Writes experiments/results/exp18_buffer_policy.csv.
"""
from multiprocessing import Pool

import numpy as np

import harness as H
import exp10_library_and_states as e10
import exp13_state_merging as e13
import dnn_fixed as F
import dnn_merge as M
import dnn_sleep as S

TASKS = e13.ROUND_TASKS
SEEDS = range(20)
STREAMS, PER_ROUND, SLEEP_EVERY = 32, 4, 2
POLICIES = [("all", None), ("recent", 16), ("random", 16), ("witness", 16),
            ("recent", 8), ("random", 8), ("witness", 8)]


def one_run(args):
    task, policy, cap, seed = args
    rng = np.random.default_rng(seed)
    out, V, unseen = e13.data(task, STREAMS, seed)
    true_delta, true_out, _ = e10.machine(task)
    seen, buffer, rows = [], [], []
    for r in range(1, STREAMS // PER_ROUND + 1):
        batch = V[(r - 1) * PER_ROUND: r * PER_ROUND]
        seen += batch
        buffer += batch
        if r % SLEEP_EVERY:
            continue
        dropped = ""
        if cap:
            buffer, dropped = S.evict(buffer, cap, out, policy, rng)
        net, lay, trained, delta, m_out, _, _, _ = S.sleep(buffer, out, H.EPS, pruning=False)
        seen_ext = [F.extend(lay, s) for s in seen]
        unseen_ext = [F.extend(lay, s) for s in unseen]
        rows.append(dict(task=task, policy=policy, cap=cap or "", seed=seed, round=r, streams_seen=len(seen),
                         buffer=len(buffer), witnesses_dropped="" if dropped is None else dropped,
                         states_found=len(delta), states_true=len(true_delta),
                         machine_correct=M.same_machine(delta, m_out, true_delta, true_out),
                         hidden_neurons=net.n - lay.n_visible,
                         recall_without_key=H.score(net, H.zero_hidden_starts(seen_ext), seen_ext, [out], rng)[0],
                         unseen_exact_mid=e13.score_mid(net, lay, unseen_ext, out)))
    return rows


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def summarise(rows):
    last = STREAMS // PER_ROUND
    n = len(_sel(rows, policy="all", round=last))
    print(f"\nAfter each sleep: runs with the true machine (of {n}: 6 tasks x {n // len(TASKS)} seeds)\n")
    sleeps = list(range(SLEEP_EVERY, last + 1, SLEEP_EVERY))
    print("| policy | cap | " + " | ".join(f"round {r}" for r in sleeps)
          + " | unseen streams exact at the end (mean) | recall of all streams seen, no key, at the end (mean) "
            "| runs that dropped a witness |")
    print("|" + "---|" * (len(sleeps) + 5))
    for policy, cap in POLICIES:
        g = _sel(rows, policy=policy, cap=cap or "")
        end = _sel(g, round=last)
        dropped = len({(x["task"], x["seed"]) for x in g if x["witnesses_dropped"] not in ("", 0)})
        print(f"| {policy} | {cap or '–'} | " + " | ".join(str(sum(x["machine_correct"] for x in _sel(g, round=r))) for r in sleeps)
              + f" | {np.mean([x['unseen_exact_mid'] for x in end]):.3f} | {np.mean([x['recall_without_key'] for x in end]):.3f} "
              f"| {dropped if policy == 'witness' else '–'} |")
    print("\nAt the end: runs with the true machine, by task\n")
    print("| task | " + " | ".join(f"{p} {c or ''}".strip() for p, c in POLICIES) + " |")
    print("|" + "---|" * (len(POLICIES) + 1))
    for task in TASKS:
        print(f"| {task} | " + " | ".join(str(sum(x["machine_correct"] for x in _sel(rows, task=task, policy=p, cap=c or "", round=last)))
                                         for p, c in POLICIES) + " |")
    ref = _sel(rows, policy="recent", cap=16, round=last)
    ref = [x for x in ref if x["seed"] < 10]
    print(f"\ncheck: recent 16, seeds 0-9, true machine at the end: {sum(x['machine_correct'] for x in ref)} of {len(ref)} "
          f"(experiment 15, sleep16: 57 of 60)")


def main(tasks=TASKS, seeds=SEEDS):
    jobs = [(t, p, c, s) for t in tasks for p, c in POLICIES for s in seeds]
    with Pool() as pool:
        rows = [r for group in pool.map(one_run, jobs) for r in group]
    H.write_csv("exp18_buffer_policy.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
