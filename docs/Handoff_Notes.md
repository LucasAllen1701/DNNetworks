# Discrete Neural Networks — handoff notes (20 November 2026)

State of the work after experiments 1–27, for whoever continues it (Claude
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
5. **Honest assessment:** good in a narrow sense (exact recall, no forgetting, real generalisation on small exact tasks), but tasks are tiny, much success comes from building blocks (letter libraries, helper layouts, screen regions) Claude designed, and merging is fragile.

## 2 Where things live

| Files | What |
|---|---|
| `docs/DiscreteNeuralNetworkNotes151126.md` | Lucas's notes: the original idea. |
| `dnn.py`, `tests/test_dnn.py` | Baseline. Keep unchanged unless Lucas asks. |
| `dnn_fixed.py`, `dnn_box.py`, `dnn_buffer.py`, `dnn_merge.py`, `dnn_sleep.py`, `docs/*.md` | Fixed neurons, box rules, buffer, merging, sleep. Propositions 1–8. |
| `dnn_upstart.py` | Helpers: `build` (upstart), `build_cover`, `make_layout_h3`, `pair_rules`, `neighbour_pair_rules`. |
| `dnn_letters.py` | Letters: library (exp17); `compose_letter`, `joint_search`, `learn_letter`, `cover_letter` (exp20); `choose_bits`, `make_layout_bits` (multi-bit letters, exp23; `n_out` outputs, `learn_machine_multi`, exp26). |
| `baseline_gd.py` | exp24: MLP and Elman RNN by Adam, numpy only. |
| `experiments/exp1…exp27_*.py`, `experiments/harness.py` | Experiments; CSVs in `experiments/results/`; `RESULTS.md` (sections 1–27 and Summary). |
| `dashboard.py` | Local web page for running experiments one at a time. |

## 3 Running experiments

- **Machines:** exp1–27 ran on Arch Linux, Python 3.14.10, numpy 2.5.4, scipy 1.18.1, 12 CPUs; results were bit-for-bit reproducible. From exp28, runs are on Lucas's test machine (i7-10510U, 4 cores / 8 threads at 1.8 GHz, 7.6 GiB RAM): versions and a reproduction check to be recorded here in its first session (see `CLAUDE.md`).
- **Dashboard conventions** (so an experiment also appears in `dashboard.py`): `experiments/exp<N>_<name>.py`, module docstring whose first paragraph is its title, CSVs via `harness.write_csv`, worker functions at module level, `main()` under `if __name__ == "__main__"`. Python 3.14 uses forkserver: module-level patches of other modules must happen at import of the experiment script (they are re-applied in workers).
- **Two pitfalls:** (1) the dashboard sets `DNN_RESULTS_DIR`, so `harness.RESULTS` is its run folder: read committed CSVs from `experiments/results/` (exp24 printed '?'). (2) Default arguments are fixed at import: a script that replaces another module's tables must also pass them explicitly (exp27 failed with KeyError 'boolean'). Small test runs should call `main()` with no arguments and `DNN_RESULTS_DIR` set.

## 4 Goals and the path to Doom

Programs that run locally, built from discrete neural networks: (1) play Doom; (2) Lean 4 proofs; (3) a chat bot. All far off; Doom nearest.

**Path to Doom:** (1) baseline comparison with gradient descent [exp24, done]; (2) acting, multi-bit outputs, letter witnesses [exp25, exp26, done]; (3) noisy data — biggest open problem, likely to need a real algorithm change; (4) bigger screens (16×16, 32×32), helpers shared across positions, more layers; (5) closed-loop play, from demonstrations then from success/failure (Lucas's idea 3); (6) long memory (maze with keys); (7) ViZDoom simple scenarios with small binarised frames; (8) full levels.

## 5 Lucas's design rationale

1. **Train an ordinary network by linear programming**; if an LP is infeasible, add a neuron. Boxes instead of weights.
2. **Run the network repeatedly** for short-term memory.

**h is depth** (h = 2: one hidden layer; the output answers the input h steps earlier). Core question: when an LP is infeasible, what should the added neuron compute? Neurons computable from the past generalise; copying the answer memorises.

## 6 Decisions (Lucas's; ask before changing)

1. Weight regions are boxes. 2. Baseline: predictor/anti-predictor chains, condition (D), keys. 3. Fixed neurons before round 1. 4. **h = 2 by default**; h = 3 for second hidden layers and state neurons reading helpers — state h when reporting. 5. A buffer of early data, for now. 6. Parked: Lean formalisation; VS Code setup. 7. Continue along the path to Doom (19 November). 8. Experiments move to Claude Code on Lucas's machine, in large batches, one PR per batch (20 November).

## 7 Lucas's three ideas (15 November)

1. One neuron per time step: memorises (exp13). 2. Expansion and compression cycles: compression by rebuild works (exp15); deletion rule safe but finds nothing. 3. Success and failure (buffer successes; on failure randomise inside the boxes): **not tested yet**; belongs to step 5 of the path.

## 8 Next steps (candidates, not yet chosen)

(a) Fix acting latency (h = 2 outputs, outputs read from letter neurons, or a game that waits for the move) and re-run exp26's play; (b) path step 3, noisy data; (c) Lucas's idea 3 (success/failure) in closed-loop play; (d) a fairer exp24 (tuned RNN, LSTM/GRU, per-step score) if the comparison is to be shown to others; then step 4, bigger screens.

## 9 Open problems

- Letters / helpers come from libraries Claude designed; finding building blocks without one (exp20: free labels overfit).
- Data that are not exact (untested; step 3 of the path).
- Bounded buffers: on the grid game, letter_witness at cap 8 is nearly as good as keeping everything (exp25); cap 4 too small.
- Merging errors even with the true letter; they depend on letter labelling (exp23, exp26 true_bits).
- Acting latency h (exp26): the network plays like a 3-step-lagged expert.

## 10 Working conventions

- Branch and open a pull request; never push to `main`. Lucas merges.
- Fixed seeds; CSVs in `experiments/results/`; negative results stated plainly; experiments that reuse earlier layouts must reproduce the earlier rows exactly; CSVs checked by script (rows, sha256, reproduced rows).
- Merged `claude/*` branches on GitHub can be deleted, except the branch of an open PR.
