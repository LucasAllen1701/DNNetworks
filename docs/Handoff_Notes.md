# Discrete Neural Networks — handoff notes (20 November 2026)

State of the work after experiments 1–32, for whoever continues it (Claude
Code on Lucas's machine from now on; see `CLAUDE.md` for how batches run).
Detailed results are in `experiments/RESULTS.md`: read only the section you
need and its Summary entry.

## 1 State in brief

1. The baseline algorithm (`dnn.py`) recalls every trained sequence exactly and never forgets (proved, true in every test), but generalises only when no hidden neurons are needed.
2. Fixed neurons installed before round 1 generalise and keep the no-forgetting proof. Everything since builds them from the data: helpers (exp14, 16, 21), state neurons by merging (exp13), letters (exp17, 20, 23), sleep and buffers (exp15, 18, 23, 25).
3. **Exp20–23:**
   - exp20: growing the letter library by one step (OR/AND of two library letters, "composed") finds letters outside it (dnf2 from 8–16 streams, exact at 32; or3). A joint exact search with a free label per pattern overfits and runs out of budget: letters need a class of simple functions.
   - exp21: removing the 20-helper limit fixes upstart's collapse; a layer of neighbouring-pixel pair ANDs (h = 3) is best: with cover, lines 0.96, blocks 0.94; with upstart, 0.91 mean over six image tasks, mirror 0.94.
   - exp22: choosing the layout by held-out training streams comes close to the oracle on images (0.954 at 32 streams vs 0.886 for the best single layout). On Boolean functions it helps at 4–8 streams; fewest helpers is as good with very few.
   - exp23 (rung 3): 4×4 two-plane grid game, letters of 1–2 screen regions + state neurons: true machine in 58/60 runs, 0.995 of unseen episodes exact.
4. **Exp24–27:**
   - exp24 (gradient baseline, 64 tanh units, Adam, untuned): MLP about level with the DNN without memory (Boolean 0.84 / 0.88, images 0.95 / 0.95 at the largest size; 0.97 with exp27's choice). Elman RNN fits its training streams but only 0.21 (state) / 0.27 (game) of unseen streams exact vs DNN 0.99 / 1.00. Caveats: RNN untuned, strict whole-stream score, the DNN uses designed building blocks — "not beaten by a plain baseline on small exact tasks", not "a better learner".
   - exp25: letter_witness buffer (also keep the streams that rule out rival letters) = whole buffer at cap 16 (58/60), 55 at cap 8 (witness 47), 41 at cap 4 (witness 33, recent 21).
   - exp26: imitating a key/door expert (2 output bits): true machine in every run from 16 episodes; in closed-loop play it equals the expert with the same 3-step lag (2.74 laps per 60 steps vs 14.28 without lag, 1.45 random). **Latency, not learning, limits acting.** true_bits sometimes merges to 3 states (labelling dependence).
   - exp27: nbr3_upstart as a 4th image layout: held-out choice 0.966 at 32 (exp22 0.954, oracle 0.969, best single layout 0.913); fewest helpers drops to 0.920.
5. **Exp28–32** (batch `acting-noise`):
   - exp28: a game that **waits for the move** (each screen shown 3 network steps, move read after 3) removes the lag: from 16 episodes every run plays like the expert, 14.28 laps. Outputs reading the letter directly (h = 2) cannot be single neurons; TRAIN memorises, play collapses.
   - exp29: 5–20% wrong labels. Images: every model fits every wrong label; discrete networks lose about as much as the MLP (nbr3_upstart 0.913 → 0.760 at p = 0.1, 32 streams; MLP 0.948 → 0.840). Game: exact merging finds no machine at all at p = 0.05 from 24 episodes.
   - exp30: game solved for tiny machines: letter and machine by **search** (fewest disagreements, majority outputs, states chosen on held-out episodes; `dnn_noise`), then relabel and run the exact LPs: expert play in every run up to p = 0.2. Images: a 4-fold disagreement filter helps at 16–32 streams (0.808 vs 0.760) but hurts with few streams and on clean data.
   - exp31: idea 3, first form (buffer successes, redraw weights inside boxes on failure): **does not learn** (0.17 laps after 50 rounds vs random 1.45). The buffer fixes every (state, letter) cell, so redraws never change play.
   - exp32: tuned RNN and tuned GRU: whole unseen streams 0.24 (state) / 0.45 (game) vs DNN 0.99 / 1.00; per step 0.75 / 0.87. The gap stays; DNN still uses designed letters.
6. **Honest assessment:** good in a narrow sense (exact recall, no forgetting, real generalisation on small exact tasks), but tasks are tiny, much success comes from building blocks (letter libraries, helper layouts, screen regions) Claude designed, and merging is fragile.

## 2 Where things live

| Files | What |
|---|---|
| `docs/DiscreteNeuralNetworkNotes151126.md` | Lucas's notes: the original idea. |
| `dnn.py`, `tests/test_dnn.py` | Baseline. Keep unchanged unless Lucas asks. |
| `dnn_fixed.py`, `dnn_box.py`, `dnn_buffer.py`, `dnn_merge.py`, `dnn_sleep.py`, `docs/*.md` | Fixed neurons, box rules, buffer, merging, sleep. Propositions 1–8. |
| `dnn_upstart.py` | Helpers: `build` (upstart), `build_cover`, `make_layout_h3`, `pair_rules`, `neighbour_pair_rules`. |
| `dnn_letters.py` | Letters: library (exp17); `compose_letter`, `joint_search`, `learn_letter`, `cover_letter` (exp20); `choose_bits`, `make_layout_bits` (multi-bit letters, exp23; `n_out` outputs, `learn_machine_multi`, exp26). |
| `baseline_gd.py` | exp24: MLP and Elman RNN by Adam, numpy only; GRU (exp32). |
| `dnn_noise.py` | exp29–31: label noise (`noisy`), machine search (`fit_machine`, `learn_machine_heldout`, `choose_bits_search`), `relabel`. |
| `experiments/exp1…exp32_*.py`, `experiments/harness.py` | Experiments; CSVs in `experiments/results/`; `RESULTS.md` (sections 1–32 and Summary). |
| `dashboard.py` | Local web page for running experiments one at a time. |

## 3 Running experiments

- **Machines:** exp1–27 ran on Arch Linux, Python 3.14.10, numpy 2.5.4, scipy 1.18.1, 12 CPUs; results were bit-for-bit reproducible. Next batches run on Lucas's test machine (i7-10510U, 4 cores / 8 threads at 1.8 GHz, 7.6 GiB RAM): record its versions and a reproduction check here in its first session (see `CLAUDE.md`).
- **Dashboard conventions** (so an experiment also appears in `dashboard.py`): `experiments/exp<N>_<name>.py`, module docstring whose first paragraph is its title, CSVs via `harness.write_csv`, worker functions at module level, `main()` under `if __name__ == "__main__"`. Python 3.14 uses forkserver: module-level patches of other modules must happen at import of the experiment script (they are re-applied in workers).
- **Two pitfalls:** (1) the dashboard sets `DNN_RESULTS_DIR`, so `harness.RESULTS` is its run folder: read committed CSVs from `experiments/results/` (exp24 printed '?'). (2) Default arguments are fixed at import: a script that replaces another module's tables must also pass them explicitly (exp27 failed with KeyError 'boolean'). Small test runs should call `main()` with no arguments and `DNN_RESULTS_DIR` set.

## 4 Goals and the path to Doom

Programs that run locally, built from discrete neural networks: (1) play Doom; (2) Lean 4 proofs; (3) a chat bot. All far off; Doom nearest.

**Path to Doom:** (1) baseline comparison with gradient descent [exp24, done]; (2) acting, multi-bit outputs, letter witnesses [exp25, exp26, done; lag fixed by the wait game, exp28]; (3) noisy data [game with tiny machines done by search + relabel, exp30; images partly; search does not scale]; (4) bigger screens (16×16, 32×32), helpers shared across positions, more layers; (5) closed-loop play, from demonstrations [done, exp28] then from success/failure (Lucas's idea 3; first form fails, exp31); (6) long memory (maze with keys); (7) ViZDoom simple scenarios with small binarised frames; (8) full levels.

## 5 Lucas's design rationale

1. **Train an ordinary network by linear programming**; if an LP is infeasible, add a neuron. Boxes instead of weights.
2. **Run the network repeatedly** for short-term memory.

**h is depth** (h = 2: one hidden layer; the output answers the input h steps earlier). Core question: when an LP is infeasible, what should the added neuron compute? Neurons computable from the past generalise; copying the answer memorises.

## 6 Decisions (Lucas's; ask before changing)

1. Weight regions are boxes. 2. Baseline: predictor/anti-predictor chains, condition (D), keys. 3. Fixed neurons before round 1. 4. **h = 2 by default**; h = 3 for second hidden layers and state neurons reading helpers — state h when reporting. 5. A buffer of early data, for now. 6. Parked: Lean formalisation; VS Code setup. 7. Continue along the path to Doom (19 November). 8. Experiments move to Claude Code on Lucas's machine, in large batches, one PR per batch (20 November).

## 7 Lucas's three ideas (15 November)

1. One neuron per time step: memorises (exp13). 2. Expansion and compression cycles: compression by rebuild works (exp15); deletion rule safe but finds nothing. 3. Success and failure (buffer successes; on failure randomise inside the boxes): first form **fails** (exp31): redraws inside the boxes never change play once the buffer covers every cell. Variants below.

## 8 Next steps (candidates, not yet chosen)

(a) Idea 3 variants: explore by flipping the machine's output in one (state, letter) cell instead of redrawing weights; store only the steps leading to each lap; or start from a few imitation episodes and improve by success/failure. (b) Scale the machine search (better than n^(n·L): e.g. greedy splitting, or the search seeded by merging) so noisy data work for bigger machines. (c) Image noise: a stronger filter or a noise-aware helper search. (d) Step 4, bigger screens. Exp28's wait game is the default for acting.

## 9 Open problems

- Letters / helpers come from libraries Claude designed; finding building blocks without one (exp20: free labels overfit).
- Data that are not exact: solved for tiny machines only (exp30 search costs n^(n·L)); image filter weak with few streams and drops many correct images.
- Bounded buffers: on the grid game, letter_witness at cap 8 is nearly as good as keeping everything (exp25); cap 4 too small.
- Merging errors even with the true letter; they depend on letter labelling (exp23, exp26 true_bits).
- Acting latency: solved by waiting 3 network steps per move (exp28); h = 2 outputs cannot be single neurons.
- Learning without an expert (idea 3): first form does not explore (exp31).

## 10 Working conventions

- Branch and open a pull request; never push to `main`. Lucas merges.
- Fixed seeds; CSVs in `experiments/results/`; negative results stated plainly; experiments that reuse earlier layouts must reproduce the earlier rows exactly; CSVs checked by script (rows, sha256, reproduced rows).
- Python: `.venv/bin/python` (system python has no numpy). The machine has no git identity: commit with per-commit env vars as Claude (`GIT_AUTHOR_NAME`/`GIT_COMMITTER_NAME` and the matching `_EMAIL` vars), git config untouched.
- Merged `claude/*` branches on GitHub can be deleted, except the branch of an open PR.
