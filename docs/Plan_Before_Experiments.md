# Before more experiments (26 November 2026)

Lucas wants three things understood and agreed before any new experiment is run. This note records them, what is already known about each, and the questions still open. The algorithms themselves are in `docs/DiscreteNeuralNetworkNotes261126.md` (the notes); results are in `experiments/RESULTS.md`.

Status: **no new experiments** until this plan is agreed (and until the RAM is fixed and exp33–36 have been re-checked; see `CLAUDE.md`). Batch exp37–40, proposed on 24 November, is on hold: it was designed before the training algorithm was reconsidered and may change.

## 1 The training algorithm: wake and sleep

**What there is** (notes, section 4):

- **Wake** (memorisation, expansion): LP training round by round; when an LP is infeasible, one clock neuron per vector of the round's sequences (decided 26 November; `dnn.train` still uses predictor chains). It is exact and forgets nothing (the Theorem), but it only memorises: a hidden neuron needs a key and, under (D), cannot be switched on by the inputs.
- **Sleep** (generalisation, compression): rebuild from a buffer of the streams seen. Find a finite machine by state merging, install it as fixed state neurons, train a fresh network on the buffer in one wake round, then delete neurons that nothing needs. Tested once (experiment 15): it fixes wrong early machines (59/60 true machines against 26), but deletion found nothing to delete.

**To settle:**

1. **What sleep should do in general.** Today's sleep only builds finite machines over a few input patterns, and its method (state merging) was designed by Claude. Screens need letters first, and these come from a separate search. What is the general rule? For example: find neurons computed from the past that explain as many buffered streams as possible without keys, with fewest neurons.
2. **What sleep may use.** The boxes and keys alone, or a buffer of raw streams (Decision 5: "a buffer of early data, for now")? If a buffer, how large, and which streams may leave it (experiments 18 and 25: keep the witnesses)?
3. **Condition (D).** It gives no forgetting, but it is why wake only memorises. Keep it and leave all generalisation to sleep, or relax it so that hidden neurons can read the inputs?
4. **When to sleep.** After a fixed number of rounds (experiment 15: every second round), or when memory grows (number of keyed streams or hidden neurons)?
5. **What success means for sleep.** Proposed: after a sleep, every buffered stream replays exactly (guaranteed), as many as possible replay *without* keys, unseen streams come out right, and the network has fewer neurons than before the sleep.
6. **ε = 10⁻⁴** (decided 26 November). Experiments 1–36 used 1/2. Do we re-run a few key experiments at the new value to set a new baseline?

## 2 Choosing what to train (reward) and dislodging unwanted behaviour

The original notes left this open: "how we determine what V is ... is a different problem". For Doom it is the central problem.

**What there is:**

- **Imitation**: V is a set of expert episodes (experiments 26 and 28: expert play from 16 episodes). This needs an expert. For Doom that could be recordings of Lucas playing.
- **Success and failure** (Lucas's idea 3). First form: buffer successes and, on failure, redraw weights inside the boxes. It does not learn (experiment 31): once the buffer fixes every cell, a redraw never changes play. Second form: flip one machine cell and keep the flip if play does not get worse. It learns a game that needs no memory (experiment 35); with memory it is untested.
- **Noisy labels**: search plus relabelling solves tiny machines up to 20% noise (experiments 30 and 36).

**Why dislodging is hard here.** Wake is exact and forgets nothing. A bad sequence that has been trained is replayed for ever from its key, and its constraints stay in the boxes of the visible neurons. Ways to remove it:

- (a) never train it: filter V, for example train only successful episodes or segments;
- (b) drop its key: memorised behaviour then never replays (Lemma 1), but the boxes still carry its constraints;
- (c) sleep without it: rebuild from a buffer that no longer holds it. This is the only full removal we have;
- (d) train a contradicting sequence: wake memorises both, each with its own key, so this does not change unkeyed behaviour by itself.

**To settle:**

1. What counts as success in a game, and which steps of an episode get the credit (credit over time)?
2. Where do new episodes come from: the current network's own play, random changes (flips, redraws), or an expert?
3. Is (c), sleeping on a filtered buffer, the mechanism for unlearning? If so, the buffer policy and the reward rule are one design question.
4. How does this extend from small machines to policies over screens?

## 3 Experiments: what to run, what success means, how each moves toward Doom

Path to Doom (repo handoff notes, section 4): (1) gradient baseline ✓; (2) acting ✓; (3) noisy data (partly); (4) bigger screens (partly); (5) closed-loop play from success and failure (reactive only); (6) long memory; (7) ViZDoom with small binarised frames; (8) full levels.

A **draft** list to discuss once sections 1 and 2 are settled; none is approved.

| # | Experiment | Success means | Doom step |
|---|---|---|---|
| A | Re-run exp33–36 after the RAM fix | every CSV identical to the committed one (timing ignored) | trust in 4–5 |
| B | Re-run a few key experiments (exp23, 15, 28) at ε = 10⁻⁴ | numbers recorded as the new baseline; same or better than at 1/2 | all |
| C | Wake–sleep cycle on the three-key game (needs memory), episodes arriving in rounds | after each sleep: true machine, unseen episodes exact without keys, play equal to the expert | 3, 6 |
| D | Dislodging: train some bad episodes, then sleep on a filtered buffer | after the sleep, play equal to the expert; count the good episodes needed | 5 |
| E | Success and failure with memory (was exp37): flips or sleep-based, on the three-key game | laps close to the expert's from a random start | 5, 6 |
| F | Small objects on 16×16 with letters at every position (was exp38) | letters found and play equal to the expert | 4 |
| G | ViZDoom set-up only: install it on the desktop, binarise 16×16 or 32×32 frames of one basic scenario, record episodes, no learning | the pipeline runs locally at a usable speed; a recorded dataset; a list of what the learner must cope with | 7 |

G does not depend on sections 1 and 2 and would show early what step 7 actually needs.
