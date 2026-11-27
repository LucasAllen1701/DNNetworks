# Discrete Neural Networks — handoff notes (26 November 2026)

State of the work after experiments 1–36, for whoever continues it (Claude
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
6. **Exp33–36** (batch `bigger-screens`, desktop):
   - exp33: the wait game on 8×8 and 16×16 screens (512 inputs): letter search never fails (up to 7,260 candidates, also from a library of straight cuts), play equals the expert; data needed grow (true machine in all runs from 16 / 32 / 64 episodes at side 4 / 8 / 16). The task still needs only half-screen regions.
   - exp34: cover's helpers fitted in small windows (2×2, 3×3, rows, columns; Claude's shapes) and **shared** (copied to every position where they stay off on all negatives): hline/vline/block exact at side 4 from 32 streams, 0.955 at side 8 (cover 0.678, upstart 0.603); hurts global tasks (top 0.533 vs upstart 0.912).
   - exp35: idea 3, second form (**flip one machine cell**, keep it if laps on the same 8 games do not fall): from a random machine 14.18 laps after 300 rounds (expert 14.28, exp31 0.17). But the key/door game is solvable without memory (circling the quadrants), so memory learning is untested; flips never fix a wrong letter.
   - exp36: machine search by **splitting states** (`dnn_noise.learn_machine_split`): 4-state three-key game from 64 noisy episodes: true machine in 8 / 7 / 5 runs at p = 0 / 0.1 / 0.2, vs exp30's local search 1 / 1 / 0 and exact merging (p = 0) 3. Letter given.
7. **Honest assessment:** good in a narrow sense (exact recall, no forgetting, real generalisation on small exact tasks), but tasks are tiny, much success comes from building blocks (letter libraries, helper layouts, screen regions) Claude designed, and merging is fragile.

## 2 Where things live

| Files | What |
|---|---|
| `docs/DiscreteNeuralNetworkNotes151126.md` | Lucas's notes: the original idea. |
| `dnn.py`, `tests/test_dnn.py` | Baseline. Keep unchanged unless Lucas asks. |
| `dnn_fixed.py`, `dnn_box.py`, `dnn_buffer.py`, `dnn_merge.py`, `dnn_sleep.py`, `docs/*.md` | Fixed neurons, box rules, buffer, merging, sleep. Propositions 1–8. |
| `dnn_upstart.py` | Helpers: `build` (upstart), `build_cover`, `make_layout_h3`, `pair_rules`, `neighbour_pair_rules`; shared helpers `translates`, `windows`, `build_cover_local` (exp34). |
| `dnn_letters.py` | Letters: library (exp17); `compose_letter`, `joint_search`, `learn_letter`, `cover_letter` (exp20); `choose_bits`, `make_layout_bits` (multi-bit letters, exp23; `n_out` outputs, `learn_machine_multi`, exp26); `choose_bits_fast` (same result, ~20× faster, exp33). |
| `baseline_gd.py` | exp24: MLP and Elman RNN by Adam, numpy only; GRU (exp32). |
| `dnn_noise.py` | exp29–31: label noise (`noisy`), machine search (`fit_machine`, `learn_machine_heldout`, `choose_bits_search`), `relabel`; split search `split_tables`, `learn_machine_split` (exp36). |
| `experiments/exp1…exp36_*.py`, `experiments/harness.py` | Experiments; CSVs in `experiments/results/`; `RESULTS.md` (sections 1–36 and Summary). |
| `dashboard.py` | Local web page for running experiments one at a time. |

## 3 Running experiments

- **Machines:** exp1–32 ran on Lucas's laptop: Arch Linux, Intel i7-10510U (4 cores / 8 threads at 1.8 GHz), 7.6 GiB RAM, Python 3.14.10, numpy 2.5.4, scipy 1.18.1; results were bit-for-bit reproducible. From exp33, runs are on his desktop (i7-3930K, 6 cores / 12 threads, 15 GiB RAM): first session 24 November 2026: Python 3.14.10, numpy 2.5.4, scipy 1.18.1 (same as the laptop); `pytest -q` 91 passed (4.5 min); exp26 (shortest of exp23–32) re-run: 80/80 rows identical to the committed CSV, timing columns ignored. Results are reproducible on the desktop. **Watch for hardware errors:** the first full run of exp33 crashed inside scipy ("number of non-zero array elements changed during function execution", which points to a flipped bit; the RAM is non-ECC) and the identical re-run passed and reproduced exp28. CPU package reached 78 °C under full load (high mark 81 °C). The first full run of exp35 also crashed once ("neuron 32 still infeasible after adding helpers") and the identical code then passed three times, twice as full runs with identical CSVs. Two failures that do not repeat suggested unreliable hardware; **a memory test on 25 November confirmed bad RAM** (see section 9). Keep the reproduction checks. Also: in a shell, `cd dir && (loop) & cmd` puts the `cd` into the background subshell; put `cd` on its own line.
- **Dashboard conventions** (so an experiment also appears in `dashboard.py`): `experiments/exp<N>_<name>.py`, module docstring whose first paragraph is its title, CSVs via `harness.write_csv`, worker functions at module level, `main()` under `if __name__ == "__main__"`. Python 3.14 uses forkserver: module-level patches of other modules must happen at import of the experiment script (they are re-applied in workers).
- **Two pitfalls:** (1) the dashboard sets `DNN_RESULTS_DIR`, so `harness.RESULTS` is its run folder: read committed CSVs from `experiments/results/` (exp24 printed '?'). (2) Default arguments are fixed at import: a script that replaces another module's tables must also pass them explicitly (exp27 failed with KeyError 'boolean'). Small test runs should call `main()` with no arguments and `DNN_RESULTS_DIR` set.

## 4 Goals and the path to Doom

Programs that run locally, built from discrete neural networks: (1) play Doom; (2) Lean 4 proofs; (3) a chat bot. All far off; Doom nearest.

**Path to Doom:** (1) baseline comparison with gradient descent [exp24, done]; (2) acting, multi-bit outputs, letter witnesses [exp25, exp26, done; lag fixed by the wait game, exp28]; (3) noisy data [game done by search + relabel, exp30; up to 4 states by split search, exp36; images partly]; (4) bigger screens (16×16, 32×32), helpers shared across positions, more layers [screens to 16×16 with half-screen letters, exp33; shared window helpers on 8×8 images, exp34]; (5) closed-loop play, from demonstrations [done, exp28] then from success/failure (Lucas's idea 3; first form fails, exp31; cell flips learn a reactive policy, exp35, memory untested); (6) long memory (maze with keys); (7) ViZDoom simple scenarios with small binarised frames; (8) full levels.

## 5 Lucas's design rationale

1. **Train an ordinary network by linear programming**; if an LP is infeasible, add a neuron. Boxes instead of weights.
2. **Run the network repeatedly** for short-term memory.

**h is depth** (h = 2: one hidden layer; the output answers the input h steps earlier). Core question: when an LP is infeasible, what should the added neuron compute? Neurons computable from the past generalise; copying the answer memorises.

## 6 Decisions (Lucas's; ask before changing)

1. Weight regions are boxes. 2. Baseline: predictor/anti-predictor chains, condition (D), keys. 3. Fixed neurons before round 1. 4. **h = 2 by default**; h = 3 for second hidden layers and state neurons reading helpers — state h when reporting. 5. A buffer of early data, for now. 6. Parked: Lean formalisation; VS Code setup. 7. Continue along the path to Doom (19 November). 8. Experiments move to Claude Code on Lucas's machine, in large batches, one PR per batch (20 November). 9. **Margin ε = 10⁻⁴** for new experiments; exp1–36 used 1/2 (26 November). 10. **Training has two phases**, wake and sleep (`docs/DiscreteNeuralNetworkNotes261126.md`, section 4); the cohort label is no longer part of the network (26 November). 11. **Claude merges its own PRs** (26 November; rules in `CLAUDE.md`). 12. **Wake adds clock neurons** (one per vector of each sequence, an identity matrix appended to the data) instead of predictor chains (26 November; notes section 4.3). Code not changed yet: Lucas wants the design settled first.

## 7 Lucas's three ideas (15 November)

1. One neuron per time step: memorises (exp13). 2. Expansion and compression cycles: compression by rebuild works (exp15); deletion rule safe but finds nothing. 3. Success and failure (buffer successes; on failure randomise inside the boxes): first form **fails** (exp31): redraws inside the boxes never change play once the buffer covers every cell. Second form (flip one machine cell, keep it if laps do not fall; exp35) learns to play the key/door game from a random machine, but that game needs no memory.

## 8 Next steps (candidates, not yet chosen)

**27 November: experiment 37** (wake with clock neurons, sleep by Lucas's merge; run in the claude.ai cloud workspace, `dnn_clock.py`): exact merging merges nothing, any stretch breaks recall, and merging cannot change behaviour without keys, so this sleep does not generalise (RESULTS.md section 37). The exp37–40 proposal below is renumbered from 38 on when revived.

**26 November: on hold.** Lucas wants the training algorithm, the choice of training sequences (reward, dislodging) and the experiment list agreed first: `docs/Plan_Before_Experiments.md`. The items below predate that and may change.

(a) Idea 3 on a game that needs memory: exp36's three-key order defeats circling; flips from scratch, and imitation (split search) plus flips. (b) Small objects on big screens: key and door as small squares at fixed or random places on 16×16, letters from window regions at every position (exp34's windows as a letter library). (c) Shared helpers as a fifth image layout in exp27's held-out choice (sharing wins local tasks, loses global ones). (d) Letter and machine both searched for the 3- and 4-state games with noise (choose_bits_search + split). (e) Image noise: a stronger filter or a noise-aware helper search. Exp28's wait game is the default for acting.

**Proposed next batch (exp37–40), suggested to Lucas on 24 November, not yet approved:**
1. exp37: idea 3 on exp36's three-key game (circling cannot solve it): cell flips from a random machine (4–5 states), and split-search imitation from a few episodes plus flips. Does success/failure learn memory? Steps 5, 6.
2. exp38: small objects on a 16×16 screen: key and door as small squares, letters from window regions at every position (exp34's `windows` as a letter library). Step 4.
3. exp39: letter and machine both searched (letter search + `learn_machine_split`) for the 3- and 4-state games with 0–20% noise, no given letter. Steps 3, 6.
4. exp40: shared window helpers (cover_shared) as a fifth image layout in exp27's held-out layout choice. Step 4.

## 9 Open problems

- Letters / helpers come from libraries Claude designed; finding building blocks without one (exp20: free labels overfit).
- Data that are not exact: machines up to 4 states by split search (exp36, letter given, 64 episodes); image filter weak with few streams and drops many correct images.
- Bounded buffers: on the grid game, letter_witness at cap 8 is nearly as good as keeping everything (exp25); cap 4 too small.
- Merging errors even with the true letter; they depend on letter labelling (exp23, exp26 true_bits).
- Acting latency: solved by waiting 3 network steps per move (exp28); h = 2 outputs cannot be single neurons.
- Learning without an expert (idea 3): cell flips learn a reactive policy (exp35); learning memory and the letter by success/failure is untested.
- Big screens: shared window helpers solve local image tasks (exp34) but hurt global ones; window shapes are Claude's. Games on big screens still use half-screen letters (exp33).
- Hardware: two crashes on the desktop that did not repeat (exp33, exp35); see section 3. **25 November: a memory test confirmed bad RAM.** No new experiments until Lucas says the RAM is fixed; then first re-run exp33–36 and check every CSV matches the committed one (timing columns ignored), since a bit flip need not crash.

## 10 Working conventions

- Branch and open a pull request; never push to `main`. Claude merges its own PRs (from 26 November; see `CLAUDE.md`).
- Fixed seeds; CSVs in `experiments/results/`; negative results stated plainly; experiments that reuse earlier layouts must reproduce the earlier rows exactly; CSVs checked by script (rows, sha256, reproduced rows).
- Python: `.venv/bin/python` (system python has no numpy). The machine has no git identity: commit with per-commit env vars as Claude (`GIT_AUTHOR_NAME`/`GIT_COMMITTER_NAME` and the matching `_EMAIL` vars), git config untouched.
- Merged `claude/*` branches on GitHub can be deleted, except the branch of an open PR.
