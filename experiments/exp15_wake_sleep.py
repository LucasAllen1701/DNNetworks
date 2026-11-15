"""Experiment 15: wake-and-sleep cycles with a stream buffer and safe deletion.

Lucas's idea 2 (expansion and compression). The six tasks of experiment
13 part B, 32 training streams arriving 4 per round (8 rounds), 10 seeds.
Round 1 builds the network from its 4 streams by state merging (as in
experiment 13). Then, per strategy:
  "fixed":   wake only. Later streams are trained into the same network; the
             machine found from the first 4 streams is kept (experiment 13
             part B with S = 4; its rows should match).
  "sleep":   wake, and after every even round sleep: state merging on the
             buffer of every stream so far, a fresh network trained on the
             buffer, then safe deletion (dnn_sleep.py).
  "sleep16": the same with a buffer of the 16 most recent streams. Streams
             that leave the buffer have no key in the rebuilt network.
Scored after every round: recall from keys of the streams trained into the
current network; recall without keys of every stream seen so far (output
column, sampled weights); unseen streams exact (sampled and midpoint);
whether the true machine was found; neurons deleted at sleep, and the unseen
score just before deletion; neurons the rule could delete after every round
(counted, not deleted, outside sleeps).
Run from the repository root:  python experiments/exp15_wake_sleep.py
Writes experiments/results/exp15_wake_sleep.csv.
"""
from multiprocessing import Pool

import numpy as np

import harness as H
import exp10_library_and_states as e10
import exp13_state_merging as e13
import dnn_fixed as F
import dnn_merge as M
import dnn_sleep as S
from tests.test_dnn import replays

TASKS = e13.ROUND_TASKS
SEEDS = range(10)
STREAMS, PER_ROUND, SLEEP_EVERY = 32, 4, 2
STRATEGIES = {"fixed": None, "sleep": STREAMS, "sleep16": 16}


def unseen_scores(net, lay, unseen, out, rng):
    ext = [F.extend(lay, s) for s in unseen]
    return (H.score(net, H.zero_hidden_starts(ext), ext, [out], rng)[0], e13.score_mid(net, lay, ext, out))


def one_run(args):
    task, strategy, seed = args
    cap = STRATEGIES[strategy]
    rng = np.random.default_rng(seed)
    out, V, unseen = e13.data(task, STREAMS, seed)
    true_delta, true_out, _ = e10.machine(task)
    batches = [V[i: i + PER_ROUND] for i in range(0, STREAMS, PER_ROUND)]
    seen, buffer, rows = [], [], []
    for r, batch in enumerate(batches, 1):
        seen += batch
        if cap:
            buffer = (buffer + batch)[-cap:]
        row = dict(task=task, strategy=strategy, seed=seed, round=r, streams_seen=len(seen),
                   buffer=len(buffer), deleted_fixed=0, deleted_hidden=0, unseen_exact_mid_before_delete="")
        if r == 1 or (cap and r % SLEEP_EVERY == 0):
            source = batch if r == 1 else buffer
            net, lay, trained, delta, m_out, df, dh, (net0, lay0) = S.sleep(source, out, H.EPS, pruning=bool(cap))
            row.update(phase="build" if r == 1 else "sleep", deleted_fixed=df, deleted_hidden=dh,
                       unseen_exact_mid_before_delete=unseen_scores(net0, lay0, unseen, out, rng)[1])
        else:
            before = net.n
            trained += F.train(net, lay, batch, H.EPS)
            row.update(phase="wake", hidden_added=net.n - before)
        _, _, _, cf, ch = S.prune(net, lay, [])      # what deletion would remove now (not applied)
        row.update(deletable_fixed=cf, deletable_hidden=ch)
        seen_ext = [F.extend(lay, s) for s in seen]
        u_ex, u_mid = unseen_scores(net, lay, unseen, out, rng)
        row.update(states_found=len(delta), states_true=len(true_delta),
                   machine_correct=M.same_machine(delta, m_out, true_delta, true_out),
                   fixed_neurons=len(lay.fixed), hidden_neurons=net.n - lay.n_visible,
                   recall_keys=float(np.mean([replays(net, [s], rng, trials=10) for s in trained])),
                   recall_without_key=H.score(net, H.zero_hidden_starts(seen_ext), seen_ext, [out], rng)[0],
                   unseen_exact=u_ex, unseen_exact_mid=u_mid)
        rows.append(row)
    return rows


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def summarise(rows):
    n = len(_sel(rows, strategy="fixed", round=1))
    print(f"\nBy strategy and round ({n} runs each: 6 tasks x {n // len(TASKS)} seeds)\n")
    print("| strategy | round | streams seen | buffer | true machine found | runs exact on every unseen stream "
          "(sampled / midpoint) | recall from keys (min) | recall of all streams seen, without key (mean) "
          "| hidden neurons (mean) | fixed neurons (mean) | deleted at this sleep (fixed / hidden, mean) |")
    print("|" + "---|" * 11)
    for strategy in STRATEGIES:
        for r in range(1, STREAMS // PER_ROUND + 1):
            g = _sel(rows, strategy=strategy, round=r)
            print(f"| {strategy} | {r}{'*' if g[0]['phase'] == 'sleep' else ''} | {g[0]['streams_seen']} | {g[0]['buffer']} "
                  f"| {sum(x['machine_correct'] for x in g)} "
                  f"| {sum(x['unseen_exact'] == 1.0 for x in g)} / {sum(x['unseen_exact_mid'] == 1.0 for x in g)} "
                  f"| {min(x['recall_keys'] for x in g):.3f} | {np.mean([x['recall_without_key'] for x in g]):.3f} "
                  f"| {np.mean([x['hidden_neurons'] for x in g]):.1f} | {np.mean([x['fixed_neurons'] for x in g]):.1f} "
                  f"| {np.mean([x['deleted_fixed'] for x in g]):.1f} / {np.mean([x['deleted_hidden'] for x in g]):.1f} |")
    print("\n(* = a sleep round)\n")
    print("At the end (round 8), runs exact on every unseen stream (midpoint), by task\n")
    print("| task | " + " | ".join(STRATEGIES) + " |")
    print("|" + "---|" * (len(STRATEGIES) + 1))
    last = STREAMS // PER_ROUND
    for task in TASKS:
        print(f"| {task} | " + " | ".join(str(sum(x['unseen_exact_mid'] == 1.0 for x in _sel(rows, task=task, strategy=s, round=last)))
                                        for s in STRATEGIES) + " |")
    pr = [x for x in rows if x["phase"] in ("build", "sleep") and x["strategy"] != "fixed"]
    better = sum(x["unseen_exact_mid"] > x["unseen_exact_mid_before_delete"] for x in pr)
    worse = sum(x["unseen_exact_mid"] < x["unseen_exact_mid_before_delete"] for x in pr)
    print(f"\nsafe deletion: {len(pr)} sleeps, {sum(x['deleted_fixed'] for x in pr)} fixed and "
          f"{sum(x['deleted_hidden'] for x in pr)} hidden neurons deleted; unseen score (midpoint) "
          f"better in {better}, worse in {worse}, same in {len(pr) - better - worse}")
    for strategy in STRATEGIES:
        g = [x for x in rows if x["strategy"] == strategy and x["phase"] == "wake"]
        print(f"{strategy}, after wake rounds: deletable neurons (fixed / hidden, mean) "
              f"{np.mean([x['deletable_fixed'] for x in g]):.2f} / {np.mean([x['deletable_hidden'] for x in g]):.2f}, "
              f"hidden neurons (mean) {np.mean([x['hidden_neurons'] for x in g]):.1f}")
    print(f"recall from keys, min over every row: {min(x['recall_keys'] for x in rows):.3f}")


def main(tasks=TASKS, seeds=SEEDS):
    jobs = [(t, s, seed) for t in tasks for s in STRATEGIES for seed in seeds]
    with Pool() as pool:
        rows = [r for group in pool.map(one_run, jobs) for r in group]
    H.write_csv("exp15_wake_sleep.csv", rows)
    summarise(rows)


if __name__ == "__main__":
    main()
