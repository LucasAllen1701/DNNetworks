# The causal variant (`dnn_causal.py`)

This note describes the variant as implemented. It changes the training
algorithm of `docs/DiscreteNeuralNetworkNotes151126.md` (the *notes*)
so that hidden neurons can compute from the past and therefore work on data that
was never trained. The price is the no-forgetting guarantee. `dnn.py` is
unchanged and remains the baseline; the variant imports its RUN, FEAS, SUBBOX,
step function and weight sampling.

## 1 Why

In the baseline, every hidden neuron belongs to a cohort that is silent unless
the run starts from a key (Lemma 1 of the notes). Its activations are copies of
future targets (predictors) or their complements (anti-predictors). So a
network started on an unseen input with its hidden neurons at 0 is just its
visible-to-visible weights: it generalises only on tasks that are threshold
functions of the visible state, and otherwise plays back the answer stored in
the key (`experiments/RESULTS.md`, experiments 1 and 4).

## 2 Neuron kinds

| kind | box | reads | at time 0 | needs a key |
|---|---|---|---|---|
| visible (cohort 0) | as in the notes | everything | from the sequence | – |
| **feature** (cohort label −1) | a single point (fixed weights) | visible neurons, feature neurons, itself | 0 | no |
| chain (cohort $r$) | as in the notes, condition (D) | as in the notes | from the key | yes |

A feature neuron reads the previous step, like every neuron:
$f(t+1) = s(w \cdot v(t))$ with $w_j = 0$ for every chain neuron $j$. It is not
an input, so `dnn.run` computes it from the weight matrix like any other
neuron; nothing in RUN changes.

## 3 Candidate pool

For neurons $i \ne j$ among the visible neurons (inputs and an always-on input
included) and the existing feature neurons, with $s$ the step function and
$f$ the candidate itself:

| name | rule |
|---|---|
| copy | $f(t+1) = s(v_i(t))$ |
| AND | $s(v_i/2 + v_j/2)$ |
| OR | $s(v_i + v_j)$ |
| AND-NOT | $s(v_i - v_j)$ (ordered pairs) |
| latch | $s(v_i + f)$ |
| set-reset | $s(v_i + f - 2 v_j)$ (ordered pairs) |

Each candidate's activations on a training sequence are obtained by
simulation: it starts at 0 and reads the sequence's visible values and the
already simulated feature values. A candidate is skipped if it is silent on
every training sequence, or if its activations equal those of an existing
neuron or of an earlier candidate in the same step. The skipped candidates
would leave the LP below unchanged, so skipping only saves time.

## 4 One training round

Inputs as in the notes: a network and a set $V$ of sequences on the visible
neurons. Margin $\varepsilon$ (0.5 in all experiments), `max_features`
(default 4).

0. Extend each sequence to all neurons: visible values from the sequence,
   feature neurons simulated from 0, chain neurons 0. The worklist and the refit
   list are the visible non-input neurons.
1. Take $g$ from the worklist. If FEAS($g$) holds, continue.
2. **Feature search** (only for visible $g$). Let the *slack* of $g$ be the
   optimum of
   $$\min \sum_c \xi_c \quad\text{s.t.}\quad a \in \tilde F(g),\ \xi \ge 0,\
   a\cdot x_c + \xi_c \ge 1 \ (y_c = 1),\ a \cdot x_c - \xi_c \le 1-\varepsilon \ (y_c = 0),$$
   one slack per constraint $c$ of section 4.1 of the notes. Identical
   constraints are merged and their slack weighted by their count, which
   gives the same optimum. For every candidate, append its column with an
   unrestricted coordinate in $g$'s box and solve the slack LP. Add the
   candidate with the lowest slack if that is lower than the current slack;
   ties go to the first candidate in a fixed order. Repeat until the slack is
   0 (tolerance $10^{-9}$), no candidate lowers it, or `max_features`
   features have been added for $g$ in this round. Features are kept even if
   $g$ stays infeasible.
3. **Fallback.** If FEAS($g$) still fails, add a predictor, else an
   anti-predictor, else both, exactly as in step 1 of TRAIN in the notes.
   These chain neurons, and their own predictors, go into the worklist and
   the refit list. Hidden $g$ (chain neurons) go straight to this step, so
   the chain termination argument of Lemma 4 is unchanged.
4. When the worklist is empty, SUBBOX every neuron in the refit list (visible
   non-input neurons and this round's chain neurons). Feature boxes stay
   points; earlier chain neurons keep their boxes.

**Lifting.** When a neuron $m$ is added, every existing box gets a coordinate
for the weight from $m$:

| existing neuron $j$ | $m$ is a feature | $m$ is a chain neuron of round $r$ |
|---|---|---|
| visible | $\mathbb{R}$ | $\mathbb{R}$ |
| feature | $\{0\}$ | $\{0\}$ |
| chain of round $r$ | $(-\infty, 0]$ | $\mathbb{R}$ |
| chain of an earlier round | $(-\infty, 0]$ | $(-\infty, 0]$ |

The new neuron's own box is the candidate's weight point for a feature, and
for a chain neuron $(-\infty, 0]$ from every neuron outside its cohort and
$\mathbb{R}$ inside it, as in ADD of the notes. Feature neurons count as
outside every chain cohort.

## 5 What still holds and what is given up

- **Learning holds.** Every sequence of the latest round replays exactly from
  its key, on all neurons, for every weight matrix in the boxes. Feature
  neurons are deterministic and read nothing that differs from training (no
  chain neurons). Visible and current chain neurons are refitted inside
  $P_g$. Earlier chain cohorts are 0 throughout and stay 0 by Lemma 1, which
  still applies because their weights from features and visible neurons are
  $\le 0$. Tested in `tests/test_dnn_causal.py`.
- **Termination and size.** At most `max_features` $\cdot (n_0 - |I|)$
  features per round, plus the Lemma 4 bound for chain neurons.
- **No forgetting is given up.** A feature added in round $r$ fires on the
  sequences of earlier rounds, and the visible neurons' weights on it are
  fitted only to round $r$. Earlier sequences may therefore stop replaying,
  and in experiment 7 almost all of them did. A round that adds no features
  keeps the old guarantee: the visible boxes only shrink and new chain cohorts
  stay silent.
- **Space.** Still $2n^2 + n$ numbers (a point box stores its point twice),
  plus a description of each feature for reporting.

## 6 Deviations from the specification

- *Search order.* The feature search runs for each visible neuron when the
  worklist reaches it, so features added for one neuron are available to the
  neurons after it, but neurons already passed are not revisited. This follows
  the specification, which applies the search "when FEAS($g$) fails".
- *Ties and duplicates.* Ties keep the first candidate in the order copy, latch,
  AND/OR over unordered pairs, AND-NOT/set-reset over ordered pairs.
  Duplicate and silent candidates are skipped (section 3). This makes the
  search deterministic without changing which slack values are reachable.
- *A hidden $g$ never gets features*, as the specification says ("for a visible
  neuron $g$").
- *`hand_built`* is a helper that builds a network with given point boxes. It
  is used to show that two-step parity is realisable (experiment 6) and is not
  part of training.

## 7 Results in brief

See `experiments/RESULTS.md`, sections 6 and 7. Solved on unseen data:
$y(t+1) = x(t-1)$, $y(t+1) = x(t) \wedge x(t-1)$ (tests), Tomita 2, and
two-step rule 90 (0.85 exact and 0.98 cell accuracy on unseen states with 32
training states, with no chain neurons). Not solved: two-step parity (realisable
with two features, but not by this greedy search), single-step rule 90,
parity, the adder, and Tomita 3–7. Forgetting is near-total in multi-round
training. Training is slower than the baseline: about the same on tasks where no
candidate helps, and up to about 300× slower after many rounds.
