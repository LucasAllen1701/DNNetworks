# Experiment results

Each script runs from the repository root (`python experiments/<script>.py`),
uses fixed random seeds, writes its raw results to `experiments/results/*.csv`,
and prints the summary tables below. `dnn.py` is used unchanged. The margin is
$\varepsilon = 0.5$ throughout. "Replay" uses `replays` from
`tests/test_dnn.py`. Unless stated otherwise, scores on runs from a state are
averaged over five weight matrices sampled from the boxes (lower corner, upper
corner, three random points).

**A fact that shapes experiments 1 and 4.** By Lemma 1, a cohort whose
neurons are all 0 stays at 0 for every weight choice. So running "from an
unseen state with every hidden neuron at 0" means the hidden neurons never fire,
and the output is computed by the visible-to-visible weights alone, which are
threshold units without a bias. The scripts assert that no hidden neuron fires
in these runs. Any generalisation must therefore come from the boxes of the
visible neurons; the hidden neurons only ever help when the run starts from a
key.

## 1 Cellular automata (`exp1_cellular_automata.py`)

Ring of 8 cells, one visible neuron per cell, no inputs. Trajectories of 10
steps from random distinct starting states, all trained in one round. Unseen
test set: 100 random starting states not among the training starts. Means over
5 seeds.

| rule | training trajectories | hidden added | train s | training replay from key | training starts, hidden at 0: exact | unseen: exact trajectory | unseen: first step exact | unseen: cell accuracy | all-zeros baseline |
|---|---|---|---|---|---|---|---|---|---|
| 232 | 1 | 0 | 0.019 | 1 | 1 | 0.047 | 0.049 | 0.507 | 0.505 |
| 232 | 2 | 0 | 0.019 | 1 | 1 | 0.046 | 0.047 | 0.534 | 0.493 |
| 232 | 4 | 0 | 0.021 | 1 | 1 | 0.054 | 0.056 | 0.568 | 0.514 |
| 232 | 8 | 0 | 0.026 | 1 | 1 | 0.086 | 0.110 | 0.660 | 0.506 |
| 232 | 16 | 0 | 0.032 | 1 | 1 | 0.303 | 0.354 | 0.788 | 0.497 |
| 232 | 32 | 0 | 0.041 | 1 | 1 | **1.000** | **1.000** | **1.000** | 0.497 |
| 90 | 1 | 0 | 0.021 | 1 | 1 | 0.015 | 0.017 | 0.777 | 0.850 |
| 90 | 2 | 0 | 0.021 | 1 | 1 | 0.029 | 0.035 | 0.824 | 0.847 |
| 90 | 4 | 3.6 | 0.038 | 1 | 0.100 | 0.020 | 0.024 | 0.807 | 0.850 |
| 90 | 8 | 7.4 | 0.065 | 1 | 0.025 | 0.014 | 0.014 | 0.819 | 0.850 |
| 90 | 16 | 11.8 | 0.114 | 1 | 0.013 | 0.014 | 0.014 | 0.850 | 0.852 |
| 90 | 32 | 14.0 | 0.158 | 1 | 0.013 | 0.016 | 0.016 | 0.849 | 0.849 |

(The Lemma 4 bound for one round is $2 \cdot 10 \cdot 8 = 160$ neurons.)

- **Rule 232** never needs a hidden neuron: majority is a threshold function
  with no bias (weights $\tfrac12$ on the three neighbours fit with
  $\varepsilon = \tfrac12$). It generalises only once the training data pins
  down the box. Majority trajectories fall into fixed points within a few
  steps, so a few trajectories leave most weights unconstrained, and the
  sampled corners then get unseen states wrong. With 32 trajectories every
  sampled weight matrix reproduced every unseen trajectory exactly.
- **Rule 90** needs hidden neurons from 4 trajectories on, and all training
  trajectories still replay exactly from their keys. But the hidden neurons
  carry the memory: started with hidden neurons at 0, even the *training*
  starting states fail (1–10% exact), and unseen trajectories are almost never
  right (1–3%). Cell accuracy on unseen states matches the all-zeros baseline,
  so the visible weights have in effect learned "predict 0". This is the
  memorisation described in section 8.1 of the notes.
- The baseline is high for rule 90 because 8 is a power of two: on this ring,
  rule 90 sends every state to all zeros within 8 steps. Short transients are
  also why the number of hidden neurons levels off at about 14, far below the
  bound.

## 2 Continual learning (`exp2_continual_learning.py`)

4 visible neurons, output neuron 3; either no inputs or neuron 0 as an input.
10 rounds, each with 3 new random sequences of 6 activation vectors. After every
round, every sequence trained so far is replayed from its key with 20 weight
matrices (both corners and 18 random points); a sequence counts as recalled only
if all 20 replays are exact. Means over 5 seeds.

| inputs | round | neurons added | total neurons | train s | recall, all so far | recall, this round | recall, earlier rounds | stored numbers $2n^2+n$ |
|---|---|---|---|---|---|---|---|---|
| none | 1 | 7.8 | 11.8 | 0.055 | 1 | 1 | – | 291 |
| none | 2 | 7.6 | 19.4 | 0.059 | 1 | 1 | 1 | 774 |
| none | 3 | 7.4 | 26.8 | 0.055 | 1 | 1 | 1 | 1 470 |
| none | 4 | 7.0 | 33.8 | 0.051 | 1 | 1 | 1 | 2 330 |
| none | 5 | 8.0 | 41.8 | 0.059 | 1 | 1 | 1 | 3 550 |
| none | 6 | 8.0 | 49.8 | 0.062 | 1 | 1 | 1 | 5 030 |
| none | 7 | 7.4 | 57.2 | 0.059 | 1 | 1 | 1 | 6 630 |
| none | 8 | 6.8 | 64.0 | 0.054 | 1 | 1 | 1 | 8 290 |
| none | 9 | 7.2 | 71.2 | 0.056 | 1 | 1 | 1 | 10 200 |
| none | 10 | 7.4 | 78.6 | 0.062 | 1 | 1 | 1 | 12 500 |
| neuron 0 | 1 | 7.0 | 11.0 | 0.045 | 1 | 1 | – | 253 |
| neuron 0 | 2 | 6.6 | 17.6 | 0.039 | 1 | 1 | 1 | 638 |
| neuron 0 | 3 | 6.6 | 24.2 | 0.040 | 1 | 1 | 1 | 1 200 |
| neuron 0 | 4 | 6.2 | 30.4 | 0.045 | 1 | 1 | 1 | 1 880 |
| neuron 0 | 5 | 7.2 | 37.6 | 0.046 | 1 | 1 | 1 | 2 870 |
| neuron 0 | 6 | 7.0 | 44.6 | 0.045 | 1 | 1 | 1 | 4 030 |
| neuron 0 | 7 | 7.0 | 51.6 | 0.046 | 1 | 1 | 1 | 5 390 |
| neuron 0 | 8 | 7.6 | 59.2 | 0.054 | 1 | 1 | 1 | 7 080 |
| neuron 0 | 9 | 6.8 | 66.0 | 0.051 | 1 | 1 | 1 | 8 790 |
| neuron 0 | 10 | 6.4 | 72.4 | 0.044 | 1 | 1 | 1 | 10 600 |

(Lemma 4 bound per round: $2 \cdot 5 \cdot 4 = 40$ neurons without inputs, 30
with one input.)

- **No forgetting holds exactly.** In all 100 rounds (2 configurations × 5
  seeds × 10 rounds), every sequence from every round replayed exactly for all
  20 weight matrices. This matches part 2 of the Theorem.
- **Growth is linear and does not slow down.** Every round adds about 7
  neurons, with no downward trend, because earlier cohorts are never reused
  (section 8.4 of the notes): each round's sequences get their own new helper
  neurons. The neuron count grows linearly in the number of rounds, so the
  stored size $2n^2 + n$ grows quadratically: about 12 500 numbers after
  30 sequences of 6 vectors each.
- **Training time per round is flat** (0.04–0.06 s) at this scale. The LPs grow
  with the total neuron count but are still tiny at 80 neurons, so the cost is
  dominated by per-call overhead.

## 3 Capacity (`exp3_capacity.py`)

4 visible neurons ($n_0 = 4$, no inputs), random sequences of $k+1$ vectors
($k$ transitions). The same $N$ sequences are trained either in one round or in
$N$ rounds of one sequence each. The Lemma 4 bound is $2k n_0 = 8k$ per round,
so $8k$ for one round and $8kN$ summed over $N$ rounds. Means over 3 seeds.
Every sequence replayed from its key in every run (5 weight matrices each), and
the bound held in every run.

| $k$ | $N$ | $Nk$ | one round: added | bound | per sequence: added | bound (sum) | one round: added per transition | per sequence: added per transition | one round s | per sequence s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 0 | 16 | 0 | 16 | 0 | 0 | 0.010 | 0.009 |
| 2 | 2 | 4 | 1.33 | 16 | 1.67 | 32 | 0.33 | 0.42 | 0.015 | 0.027 |
| 2 | 4 | 8 | 3.00 | 16 | 3.67 | 64 | 0.38 | 0.46 | 0.022 | 0.053 |
| 2 | 8 | 16 | 6.67 | 16 | 12.0 | 128 | 0.42 | 0.75 | 0.039 | 0.145 |
| 2 | 16 | 32 | 7.67 | 16 | 24.0 | 256 | 0.24 | 0.75 | 0.051 | 0.352 |
| 4 | 1 | 4 | 2.67 | 32 | 2.67 | 32 | 0.67 | 0.67 | 0.030 | 0.029 |
| 4 | 2 | 8 | 4.33 | 32 | 4.33 | 64 | 0.54 | 0.54 | 0.042 | 0.056 |
| 4 | 4 | 16 | 6.67 | 32 | 10.0 | 128 | 0.42 | 0.63 | 0.056 | 0.109 |
| 4 | 8 | 32 | 11.0 | 32 | 24.7 | 256 | 0.34 | 0.77 | 0.069 | 0.230 |
| 4 | 16 | 64 | 13.3 | 32 | 48.3 | 512 | 0.21 | 0.76 | 0.086 | 0.450 |
| 8 | 1 | 8 | 5.00 | 64 | 5.00 | 64 | 0.63 | 0.63 | 0.034 | 0.034 |
| 8 | 2 | 16 | 8.67 | 64 | 10.7 | 128 | 0.54 | 0.67 | 0.053 | 0.073 |
| 8 | 4 | 32 | 13.0 | 64 | 20.0 | 256 | 0.41 | 0.63 | 0.077 | 0.149 |
| 8 | 8 | 64 | 19.7 | 64 | 39.7 | 512 | 0.31 | 0.62 | 0.120 | 0.300 |
| 8 | 16 | 128 | 25.7 | 64 | 80.0 | 1024 | 0.20 | 0.63 | 0.170 | 0.561 |
| 16 | 1 | 16 | 8.67 | 128 | 8.67 | 128 | 0.54 | 0.54 | 0.044 | 0.047 |
| 16 | 2 | 32 | 14.7 | 128 | 18.7 | 256 | 0.46 | 0.58 | 0.071 | 0.095 |
| 16 | 4 | 64 | 26.3 | 128 | 33.7 | 512 | 0.41 | 0.53 | 0.137 | 0.182 |
| 16 | 8 | 128 | 38.7 | 128 | 69.7 | 1024 | 0.30 | 0.54 | 0.249 | 0.463 |
| 16 | 16 | 256 | 49.7 | 128 | 141 | 2048 | 0.19 | 0.55 | 0.551 | 1.15 |

- **Per-sequence rounds grow linearly in total length**: about 0.55–0.75 new
  neurons per transition, steady across $N$. This is the growth seen in
  experiment 2: no cohort is shared between rounds.
- **One round grows sublinearly** in $N$ for fixed $k$: the neurons added per
  transition fall from about 0.6 at $N = 1$ to about 0.2 at $N = 16$. Sequences
  trained together share helper chains, and the per-round bound $8k$ does not
  depend on $N$. With $N = 16$, one round used 2.8× ($k = 16$) to 3.6× ($k = 4$) fewer
  neurons than one round per sequence. It was also faster.
- **The Lemma 4 bound is loose.** The largest fraction of the bound used in
  any run was 50% for one round ($k = 2$, $N = 16$: 8 of 16) and 12.5% for one
  round per sequence. At $k = 2$, $N = 16$ one round is close to its cap, which
  is the only place the bound, rather than the data, limits growth.
- Caveat: with 4 visible neurons there are only 16 visible states, so long
  random sequences repeat states with conflicting successors. Most helpers
  resolve those conflicts. With more visible neurons, conflicts would be rarer
  and fewer helpers needed.

## 4 Input-driven tasks (`exp4_input_tasks.py`)

Visible neurons: one input (two for the adder) and one output. A stream of
$T + 1 = 13$ random input vectors gives the training sequence
$v(t) = (x_t, y_t)$ with $y_t = f(x_0 \dots x_{t-1})$, so the output answers
for each input one step after reading it. All training streams are trained in
one round. Accuracy is the fraction of $y_1 \dots y_T$ correct, averaged over 5
weight matrices; "exact" is the fraction of streams with every output correct.
Unseen streams (100, none in the training set) start from their true visible
state with every hidden neuron at 0. Means over 3 seeds.

Diagnostic columns:
- *function / separable*: whether $y_{t+1}$ is determined by the visible state
  $v(t)$ at all, and whether it is a threshold function of it without a bias at
  margin 0.5 (FEAS on the visible neurons, unbounded box), over all training and
  unseen transitions. By Lemma 1, a run with every hidden neuron at 0 can be
  right on every stream only if both hold.
- *key ignores inputs*: each training key is run with the inputs of a
  different training stream; the fraction of runs whose output is still
  exactly the trained output.

| task | train streams | hidden added | train s | train acc (from key) | train exact (from key) | train acc (hidden at 0) | key ignores inputs | unseen acc | unseen exact | majority baseline | function | separable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prev_input | 8 | 0 | 0.004 | 1 | 1 | 1 | 0 | **1** | **1** | 0.506 | yes | yes |
| prev_input | 32 | 0 | 0.004 | 1 | 1 | 1 | 0 | **1** | **1** | 0.509 | yes | yes |
| prev_input | 128 | 0 | 0.008 | 1 | 1 | 1 | 0 | **1** | **1** | 0.510 | yes | yes |
| parity | 8 | 11.7 | 0.060 | 1 | 1 | 0.510 | 0.842 | 0.509 | 0 | 0.512 | yes | no |
| parity | 32 | 12 | 0.092 | 1 | 1 | 0.484 | 0.979 | 0.517 | 0 | 0.517 | yes | no |
| parity | 128 | 12 | 0.190 | 1 | 1 | 0.501 | 0.997 | 0.515 | 0 | 0.515 | yes | no |
| adder | 8 | 11 | 0.057 | 1 | 1 | 0.469 | 0.758 | 0.505 | 0 | 0.513 | no | no |
| adder | 32 | 12 | 0.089 | 1 | 1 | 0.495 | 0.973 | 0.513 | 0 | 0.523 | no | no |
| adder | 128 | 12 | 0.207 | 1 | 1 | 0.501 | 0.983 | 0.491 | 0 | 0.510 | no | no |
| tomita1 | 8 | 0 | 0.003 | 1 | 1 | 1 | 0.417 | **1** | **1** | 0.917 | yes | yes |
| tomita1 | 32 | 0 | 0.004 | 1 | 1 | 1 | 0.281 | **1** | **1** | 0.920 | yes | yes |
| tomita1 | 128 | 0 | 0.007 | 1 | 1 | 1 | 0.346 | **1** | **1** | 0.925 | yes | yes |
| tomita2 | 8 | 4 | 0.024 | 1 | 1 | 0.958 | 0.942 | 0.975 | 0.757 | 0.975 | no | no |
| tomita2 | 32 | 7.33 | 0.050 | 1 | 1 | 0.964 | 0.906 | 0.975 | 0.777 | 0.975 | no | no |
| tomita2 | 128 | 6 | 0.076 | 1 | 1 | 0.971 | 0.916 | 0.975 | 0.787 | 0.975 | no | no |
| tomita3 | 8 | 9 | 0.045 | 1 | 1 | 0.638 | 0.800 | 0.506 | 0.037 | 0.529 | no | no |
| tomita3 | 32 | 11.7 | 0.080 | 1 | 1 | 0.554 | 0.869 | 0.517 | 0.031 | 0.508 | no | no |
| tomita3 | 128 | 12 | 0.187 | 1 | 1 | 0.510 | 0.891 | 0.486 | 0.030 | 0.529 | no | no |
| tomita4 | 8 | 12 | 0.062 | 1 | 1 | 0.392 | 0.900 | 0.305 | 0 | 0.712 | no | no |
| tomita4 | 32 | 12 | 0.092 | 1 | 1 | 0.318 | 0.938 | 0.307 | 0 | 0.709 | no | no |
| tomita4 | 128 | 12 | 0.196 | 1 | 1 | 0.314 | 0.946 | 0.329 | 0 | 0.686 | no | no |
| tomita5 | 8 | 11.7 | 0.058 | 1 | 1 | 0.753 | 0.925 | 0.743 | 0.020 | 0.743 | no | no |
| tomita5 | 32 | 12 | 0.079 | 1 | 1 | 0.747 | 0.979 | 0.741 | 0.007 | 0.741 | no | no |
| tomita5 | 128 | 12 | 0.171 | 1 | 1 | 0.761 | 0.993 | 0.750 | 0.017 | 0.750 | no | no |
| tomita6 | 8 | 12 | 0.063 | 1 | 1 | 0.663 | 0.867 | 0.682 | 0 | 0.682 | no | no |
| tomita6 | 32 | 12 | 0.086 | 1 | 1 | 0.695 | 0.971 | 0.683 | 0 | 0.683 | no | no |
| tomita6 | 128 | 12 | 0.177 | 1 | 1 | 0.687 | 0.997 | 0.689 | 0 | 0.689 | no | no |
| tomita7 | 8 | 9.33 | 0.046 | 1 | 1 | 0.537 | 0.900 | 0.435 | 0 | 0.582 | no | no |
| tomita7 | 32 | 12 | 0.082 | 1 | 1 | 0.496 | 0.929 | 0.458 | 0 | 0.558 | no | no |
| tomita7 | 128 | 12 | 0.178 | 1 | 1 | 0.424 | 0.910 | 0.420 | 0 | 0.595 | no | no |

- **The expected result is confirmed.** On every task, every training stream is
  reproduced exactly from its key. On unseen streams, the two tasks whose next
  output is a bias-free threshold function of the visible state
  (`prev_input`, and Tomita 1, where $y_{t+1} = x_t \wedge y_t$) are solved
  perfectly with no hidden neurons. Every other task — parity, the adder and
  Tomita 2–7 — is at or below the majority-class baseline on unseen streams,
  and almost no unseen stream is fully correct. More training streams do not
  help.
- **Sometimes worse than chance.** On Tomita 4 and 7 the unseen accuracy
  (0.31–0.33 and 0.42–0.46) is well below the majority baseline (0.69–0.71 and
  0.56–0.60). The visible weights fitted alongside the hidden helpers are not a
  reasonable fallback once the helpers are silent.
- **The keys are tapes.** For every non-separable task the hidden count levels
  off at 12, exactly $T$: one helper chain of full length, the delay line of
  section 8.1. Its key holds the stream's whole future output, and the inputs
  hardly matter. With 128 training streams, replaying a key with *another
  stream's inputs* still produced the trained output exactly in 91–99.7% of
  runs (parity 99.7%, Tomita 6 99.7%, adder 98.3%). For the separable control
  the figure is 0, as it should be. (For Tomita 1 the 0.3–0.4 is coincidence:
  many outputs are all 0 after the first 0 input.) So for these tasks the
  network does not compute the task from its inputs; it plays back the answer
  stored in the key.
- Training with hidden neurons at 0 scores the same as unseen streams, which
  confirms that nothing learned about the task lives outside the keys.
- Tomita 2's 0.975 accuracy is just the baseline: almost every prefix is
  rejected and the network predicts 0.

## 5 Baseline with an always-on input neuron (`exp5_bias.py`)

Experiments 1 and 4 rerun with `dnn.py` on exactly the same data, with and
without an extra input neuron that is 1 at every step. That neuron gives
every threshold unit a bias weight. Hidden neurons are chain neurons; the
CA "all-zero baseline" and the task "majority baseline" do not depend on the
bias. Means over 5 seeds (CA) and 3 seeds (tasks). The shared code is in
`experiments/harness.py`.

| task | training sequences | hidden: no bias | hidden: bias | unseen exact: no bias | unseen exact: bias | unseen accuracy: no bias | unseen accuracy: bias | baseline |
|---|---|---|---|---|---|---|---|---|
| rule 232 | 1 | 0 | 0 | 0.048 | 0.042 | 0.508 | 0.503 | 0.505 |
| rule 232 | 8 | 0 | 0 | 0.086 | 0.087 | 0.661 | 0.661 | 0.506 |
| rule 232 | 16 | 0 | 0 | 0.301 | 0.298 | 0.786 | 0.783 | 0.497 |
| rule 232 | 32 | 0 | 0 | **1.000** | **0.878** | **1.000** | **0.976** | 0.497 |
| rule 90 | 1 | 0 | 0 | 0.015 | 0.014 | 0.779 | 0.777 | 0.850 |
| rule 90 | 8 | 7.4 | 7.4 | 0.013 | 0.014 | 0.819 | 0.792 | 0.850 |
| rule 90 | 32 | 14 | 14 | 0.016 | 0.016 | 0.849 | 0.849 | 0.849 |
| prev_input | 8–128 | 0 | 0 | 1 | 1 | 1 | 1 | 0.51 |
| parity | 128 | 12 | 12 | 0 | 0 | 0.515 | 0.515 | 0.515 |
| adder | 128 | 12 | 12 | 0 | 0 | 0.491 | 0.491 | 0.510 |
| tomita1 | 8–128 | 0 | 0 | 1 | 1 | 1 | 1 | 0.92 |
| tomita2 | 128 | 6 | 6 | 0.787 | 0.787 | 0.975 | 0.975 | 0.975 |
| tomita3 | 128 | 12 | 12 | 0.030 | 0 | 0.486 | 0.471 | 0.529 |
| tomita4 | 128 | 12 | 12 | 0 | 0 | 0.329 | 0.314 | 0.686 |
| tomita5 | 128 | 12 | 12 | 0.017 | 0.017 | 0.750 | 0.750 | 0.750 |
| tomita6 | 128 | 12 | 12 | 0 | 0 | 0.689 | 0.689 | 0.689 |
| tomita7 | 128 | 12 | 12 | 0 | 0 | 0.420 | 0.405 | 0.595 |

(Abridged; every training size of both experiments is in
`experiments/results/exp5_bias.csv`, and `python experiments/exp5_bias.py`
prints the full tables. The rows left out show the same pattern.)

- **Nothing that failed is fixed, and nothing that worked gets better.** No
  task changes from failing to passing or from needing hidden neurons to not
  needing them. Every training sequence still replays from its key.
- That is expected. The failing tasks fail because the next output is either
  not a function of the visible state at all (adder, Tomita 2–7) or an XOR of
  it (rule 90, parity). A bias weight does not make XOR linearly separable, and
  it adds no memory.
- **The bias makes things slightly worse.** Rule 232 with 32 trajectories drops
  from 1.0 to 0.88 exact on unseen states. The bias weight is one more loosely
  constrained coordinate in each box, so the sampled corners disagree with
  majority on states not seen in training. Tomita 3, 4 and 7 lose about 1.5
  points of unseen accuracy for the same reason.

## 6 Tasks given one extra network step (`exp6_two_step.py`)

**(a) Two-step parity**: input $x$, output $y$,
$y(t+2) = x(0) \oplus \dots \oplus x(t)$, $y(0) = y(1) = 0$; streams of 13
vectors. **(b) Two-step rule 90** on a ring of 6: $c(0), 0, c(1), 0, \dots,
c(5)$, so each automaton step takes two network steps and the cells are 0 on
odd steps. There are only 64 states; the unseen set is every state not used for
training. Unseen data starts from its true visible state with every hidden
neuron at 0. Means over 3 seeds.

**The task is realisable.** The hand-built network
$h_1 = s(x - h_1 - h_2)$, $h_2 = s(h_1 + h_2 - x)$, $y = s(h_1 + h_2)$, with
$h_1, h_2$ as feature neurons, was exact on 1000 of 1000 random streams (it is
also a unit test). The two hidden neurons hold the running parity, encoded as
"$h_1$ or $h_2$".

| task | trainer | training sequences | features | chain neurons | train s | training exact from key | unseen exact | unseen accuracy | baseline |
|---|---|---|---|---|---|---|---|---|---|
| parity, 2-step | hand-built | – | 2 | 0 | – | – | **1.000** | **1.000** | – |
| parity, 2-step | baseline | 8 | 0 | 10.7 | 0.05 | 1 | 0 | 0.544 | 0.544 |
| parity, 2-step | baseline | 32 | 0 | 12 | 0.08 | 1 | 0 | 0.553 | 0.553 |
| parity, 2-step | baseline | 128 | 0 | 12 | 0.17 | 1 | 0 | 0.552 | 0.552 |
| parity, 2-step | variant | 8 | 3.33 | 10.7 | 0.24 | 1 | 0 | 0.544 | 0.544 |
| parity, 2-step | variant | 32 | 3.67 | 12 | 0.38 | 1 | 0 | 0.553 | 0.553 |
| parity, 2-step | variant | 128 | 4 | 12 | 0.82 | 1 | 0 | 0.552 | 0.552 |
| rule 90, 2-step | baseline | 2 | 0 | 13.3 | 0.07 | 1 | 0.061 | 0.730 | 0.749 |
| rule 90, 2-step | baseline | 8 | 0 | 26 | 0.15 | 1 | 0.054 | 0.749 | 0.749 |
| rule 90, 2-step | baseline | 32 | 0 | 44 | 0.35 | 1 | 0.052 | 0.739 | 0.740 |
| rule 90, 2-step | variant | 2 | 2.67 | 0 | 0.22 | 1 | 0.065 | 0.752 | 0.749 |
| rule 90, 2-step | variant | 4 | 6.67 | 0 | 2.1 | 1 | 0.057 | 0.757 | 0.749 |
| rule 90, 2-step | variant | 8 | 9.67 | 0 | 5.2 | 1 | 0.193 | 0.835 | 0.749 |
| rule 90, 2-step | variant | 16 | 10.7 | 0 | 7.4 | 1 | 0.428 | 0.889 | 0.752 |
| rule 90, 2-step | variant | 32 | 11.7 | 0 | 11.2 | 1 | **0.854** | **0.976** | 0.740 |

(Baseline rows for 4 and 16 training states are in the CSV and show the same
pattern.)

- **Two-step parity: not found, as expected.** The greedy search spends its 4
  features (latch, set-reset, copy and an AND/OR) without reaching zero slack,
  then falls back to a full chain of 12. Unseen accuracy is exactly the
  baseline's. The hand-built $h_1, h_2$ each have three inputs, so neither is
  in the candidate pool, and no single pool candidate lowers the slack in a way
  that leads towards them. So the task is realisable but not by this search.
- **Two-step rule 90: solved structurally, mostly solved on unseen data.** The
  variant uses no chain neurons at all for any training size. With 32 training
  states it finds, in 2 of 3 seeds, all 12 features a correct solution needs:
  for each cell, OR and AND of its two neighbours (XOR = OR AND NOT AND). Every
  sampled weight matrix reproduces the training data from its key, and on
  unseen states it reaches 0.85 exact and 0.98 cell accuracy, against
  0.05 / 0.74 for the baseline.
- **Why not 1.0 — loose boxes, not missing features.** A diagnostic run on the
  same data (not in the CSV) separated the two causes. For the two seeds with
  all 12 features, the midpoint of the boxes was exact on every unseen state,
  and 20 random interior samples were exact on 97–100%. The failures come from
  the box corners: lower and upper corners were exact on 72–94%. The cell
  weights on irrelevant coordinates are constrained only by the training
  states, and `sample_weights` cuts unbounded sides at an arbitrary ±5, so the
  corners are the worst case. The third seed found 11 of 12 features and is
  wrong on about 30% of unseen states for every weight choice. This is the same
  effect as rule 232 in experiment 1: correct structure, with generalisation
  limited by how much training data constrains the box.
- The variant is much slower here (11 s against 0.35 s at 32 states), because
  each greedy step solves one slack LP per candidate.

## 7 The causal variant next to the baseline (`exp7_causal.py`)

Experiments 1–4 rerun with `dnn_causal.py` (described in
`docs/Causal_Variant.md`; `max_features = 4`, margin 0.5) and with `dnn.py`,
each with and without the always-on input neuron. "Features" are the variant's
fixed-weight neurons; "chain" counts predictor and anti-predictor neurons.
Unseen data is run with every hidden neuron at 0. Features start at 0 too, but
unlike chain neurons they then compute.

**Problem sizes were reduced** because the candidate search is slow. Every
greedy step solves one slack LP per candidate, the pool grows with the square
of the number of visible and feature neurons, and features accumulate over
rounds: a 10-round run of experiment 2 took about 80 s per seed, against 0.5 s
for the baseline. Changes from experiments 1–4:
- exp 1: 4, 16 and 32 trajectories, 3 seeds;
- exp 2: only the configuration without an input neuron, 3 seeds;
- exp 3: $k \in \{2, 4, 8\}$, $N \in \{1, 2, 4, 8\}$, 2 seeds;
- exp 4: unchanged.

Configurations run in 4 worker processes and the whole script takes about
6 minutes. The data is the same as in the original experiments wherever the
sizes overlap. The full tables, with every bias setting, are printed by the
script and stored in `experiments/results/exp7_causal_exp{1,2,3,4}.csv`; the
tables below are condensed.

### 7.1 Cellular automata (8 cells, one automaton step per network step)

| rule | trajectories | bias | trainer | features | chain | training starts, hidden at 0: exact | unseen exact | unseen cell accuracy | all-zero baseline |
|---|---|---|---|---|---|---|---|---|---|
| 232 | 32 | no | both | 0 | 0 | 1 | 1.000 | 1.000 | 0.491 |
| 232 | 32 | yes | both | 0 | 0 | 1 | 0.880 | 0.976 | 0.491 |
| 90 | 4 | no | baseline | 0 | 3.3 | 0.083 | 0.023 | 0.768 | 0.850 |
| 90 | 4 | no | variant | 2 | 0 | **1** | 0.033 | 0.782 | 0.850 |
| 90 | 16 | no | baseline | 0 | 11.3 | 0.021 | 0.017 | 0.853 | 0.853 |
| 90 | 16 | no | variant | 9.7 | 7.7 | 0.021 | 0.007 | 0.732 | 0.853 |
| 90 | 32 | no | baseline | 0 | 14 | 0.010 | 0.007 | 0.850 | 0.850 |
| 90 | 32 | no | variant | 10.7 | 13.3 | 0.010 | 0.003 | 0.843 | 0.850 |
| 90 | 32 | yes | variant | 9 | 13 | 0.010 | 0.003 | 0.820 | 0.850 |

- Rule 232 needs no hidden neurons, so the two trainers give identical networks.
- **Single-step rule 90 is not solved, and the variant is worse.** Its next
  state is an XOR of the *current* state, but a feature reads the state one step
  earlier. A feature therefore cannot supply that XOR in time for the next step,
  which is why experiment 6 gives the network an extra step. With 4
  trajectories, two features are enough to replay the training trajectories
  from hidden 0, but they fit those particular trajectories rather than the
  rule: unseen accuracy is unchanged. With 16–32 trajectories the variant adds
  about 10 features and still needs chains. Its unseen cell accuracy
  (0.73–0.84) is below the all-zero baseline the original trainer reaches,
  because the features fire on unseen states and push the cells into wrong
  values.

### 7.2 Continual learning (4 visible neurons, 10 rounds of 3 sequences)

| round | variant: features added | variant: chain added | variant: total neurons | variant: train s | variant: recall, this round | variant: recall, earlier rounds (no bias / bias) | baseline: recall, earlier rounds |
|---|---|---|---|---|---|---|---|
| 1 | 5.0 | 2.3 | 11.3 | 0.8 | 1 | – | – |
| 2 | 5.7 | 4.7 | 21.7 | 3.2 | 1 | **0 / 0** | 1 |
| 3 | 5.3 | 4.0 | 31.0 | 4.2 | 1 | 0 / 0 | 1 |
| 4 | 4.7 | 5.0 | 40.7 | 7.2 | 1 | 0 / 0 | 1 |
| 5 | 4.7 | 5.0 | 50.3 | 6.6 | 1 | 0 / 0 | 1 |
| 6 | 4.7 | 5.3 | 60.3 | 10.0 | 1 | 0 / 0 | 1 |
| 7 | 5.3 | 4.7 | 70.3 | 11.5 | 1 | 0 / 0 | 1 |
| 8 | 4.7 | 5.7 | 80.7 | 9.6 | 1 | 0.016 / 0 | 1 |
| 9 | 5.0 | 4.3 | 90.0 | 14.4 | 1 | 0 / 0 | 1 |
| 10 | 5.0 | 3.7 | 98.7 | 17.1 | 1 | 0 / 0 | 1 |

(Variant columns other than recall are without bias; with bias they are within
about 1 neuron and up to 1.9× slower. Recall counts a sequence only if its
visible values replay exactly, from its zero-padded key, for 5 weight
matrices.)

- **Forgetting appears, as expected, and it is near-total.** After every
  round from round 2 on, essentially no sequence from an earlier round replays
  (1 of 63 old sequences in one round of one seed). The current round always
  replays, as the variant guarantees.
- **Why it is total.** Every round adds about 5 features, and new features
  fire on old sequences. When the visible neurons are refitted, their weights
  on the new feature coordinates are fitted only to the new sequences. A
  control run confirmed the cause: after a round that adds no neurons (an
  all-zero sequence), every round-1 sequence still replayed. So forgetting is
  caused by new features, not by box shrinkage or the dormant chain neurons.
- **The variant is bigger and slower.** It ends with about 99 neurons against
  79 for the baseline, because it still adds 4–6 chain neurons per round. Time
  per round grows from 0.8 s to 17 s as the pool of feature candidates grows;
  the baseline stays at about 0.05 s.

### 7.3 Capacity (4 visible neurons, no bias)

| $k$ | $N$ | baseline one round: chain | variant one round: features + chain | baseline per sequence: chain | variant per sequence: features + chain | variant per sequence: recall of all $N$ |
|---|---|---|---|---|---|---|
| 2 | 2 | 1 | 0.5 + 0 | 1.5 | 1 + 0 | 0.75 |
| 2 | 4 | 3 | 3 + 0 | 3.5 | 1.5 + 2 | 1.00 |
| 2 | 8 | 6.5 | 4.5 + 4 | 12 | 4 + 6.5 | 0.31 |
| 4 | 1 | 2 | 0.5 + 0 | 2 | 0.5 + 0 | 1 |
| 4 | 2 | 3.5 | 3 + 0 | 3.5 | 2 + 1.5 | 0.50 |
| 4 | 4 | 6.5 | 5.5 + 2 | 9.5 | 5.5 + 4 | 0.25 |
| 4 | 8 | 10.5 | 11.5 + 7.5 | 25.5 | 14.5 + 11.5 | 0.125 |
| 8 | 1 | 5 | 2 + 0 | 5 | 2 + 0 | 1 |
| 8 | 2 | 9 | 5 + 4 | 10.5 | 5.5 + 4.5 | 0.50 |
| 8 | 4 | 13 | 9 + 6 | 19 | 13 + 10 | 0.25 |
| 8 | 8 | 20.5 | 15.5 + 16.5 | 41 | 27.5 + 23.5 | 0.125 |

(Rows with $N = 1$ for $k = 2$ add nothing. The baseline recalled every sequence
in every run, and so did the variant in one-round mode. With bias the numbers
are similar.)

- On short random data the variant uses fewer chain neurons, but **more
  neurons in total** from about $N k = 16$ on (equal at $k = 8$, $N = 2$; more in
  every larger configuration). Random sequences have no structure to
  find, so features only fit coincidences.
- **One round per sequence leaves only the last sequence**, in most runs: recall
  of all $N$ is typically $1/N$. This is the forgetting of 7.2 again.
- The bounds hold. Chain neurons used at most 30% of the Lemma 4 bound. Features
  stay within the variant's own cap of $4(n_0 - |I|)$ per round, and reached it
  in one run.
- Training in one round is up to 15 s for the variant, against 0.1 s for the
  baseline.

### 7.4 Input-driven tasks (128 training streams; other sizes in the CSV)

| task | baseline: chain | baseline: unseen acc | variant: features + chain | variant: unseen acc (no bias / bias) | variant: unseen exact | majority baseline |
|---|---|---|---|---|---|---|
| prev_input | 0 | **1** | 0 + 0 | **1 / 1** | 1 | 0.51 |
| parity | 12 | 0.515 | 2.3 + 12 | 0.515 / 0.515 | 0 | 0.515 |
| adder | 12 | 0.491 | 4 + 12 | 0.491 / 0.491 | 0 | 0.510 |
| tomita1 | 0 | **1** | 0 + 0 | **1 / 1** | 1 | 0.925 |
| tomita2 | 6 | 0.975 | **1 + 0** | **1 / 1** | **1** | 0.975 |
| tomita3 | 12 | 0.486 | 4 + 12 | 0.480 / 0.479 | 0.02 | 0.529 |
| tomita4 | 12 | 0.329 | 0 + 12 | 0.329 / 0.314 | 0 | 0.686 |
| tomita5 | 12 | 0.750 | 2 + 12 | 0.750 / 0.750 | 0.02 | 0.750 |
| tomita6 | 12 | 0.689 | 3 + 12 | 0.691 / 0.691 | 0.01 | 0.689 |
| tomita7 | 12 | 0.420 | 0 + 12 | 0.420 / 0.405 | 0 | 0.595 |

- **Tomita 2 is solved by the variant**: one feature and no chain neurons,
  exact on every unseen stream at every training size (8, 32 and 128 streams),
  with or without bias. The feature is $f = \text{AND}(x, y)$, and the output
  learns $y(t+1) = f(t) \wedge \neg x(t)$, which is exactly "the string so far
  is $(10)^*$". The baseline needed 4–7 chain neurons and only matched the
  majority class.
- **Everything else is unchanged.** Parity, the adder, and Tomita 3, 5 and 6 get
  2–4 features that lower the slack but do not reach 0. The search then falls
  back to full chains, and unseen accuracy is the baseline's to within 0.01.
  For Tomita 4 and 7, no single candidate lowers the slack at all, so the
  greedy search stops immediately. Both need two-level features (for example,
  "the last two inputs were 0"), and a one-step greedy search cannot see
  through the first level.
- With bias the variant sometimes finds fewer features (the always-on neuron
  makes some candidates duplicates). It changes no outcome.

## 8 Past shifts of the inputs (`exp8_past_shift_inputs.py`)

`dnn.py` is used unchanged. For a depth $d$, every input neuron gets $d$ extra
input neurons carrying its earlier values $x(t-1), \dots, x(t-d)$, with 0
before the stream starts. They exist before round 1 and are ordinary input
neurons, clamped by RUN, so the Theorem of the notes applies as written and
the caller supplies the sliding window. A copy neuron inside the network, with
weight 1 from its predecessor, would have exactly the same activations.

*Start clock.* In the "clock" configuration the always-on input neuron of
experiment 5 is added and shifted like any other input. Its shifts are 0 for
the first steps and 1 afterwards, so they tell the network how long ago the
stream started.

Tasks: the ten of experiment 4 on the same data, plus five that depend on the
last few inputs: `delay2` ($y(t+1) = x(t-1)$), `and2` ($x(t) \wedge x(t-1)$),
`maj3` (majority of the last three inputs), `xor2` ($x(t) \oplus x(t-1)$), and
`no000` (0 exactly when the last three inputs read were all 0; the stateless
core of Tomita 4). Scores are as in experiment 4, and unseen streams start
with every hidden neuron at 0. With depth 0 and no clock the script reproduces
the baseline rows of experiment 5 exactly (90 of 90 runs).

### 8.1 One round (128 training streams, means over 3 seeds)

Each cell lists depths 0 / 1 / 2 / 3. "Threshold from depth" is the smallest
depth at which the next output is a threshold function of the visible state
(inputs, shifts and current output) over all training and unseen transitions,
checked with FEAS on an unbounded box.

| task | chain neurons, no clock | unseen exact, no clock | chain neurons, clock | unseen exact, clock | threshold from depth (no clock / clock) |
|---|---|---|---|---|---|
| prev_input | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 |
| tomita1 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 1 | 0 / 0 |
| delay2 | 12 / 0 / 0 / 0 | 0 / **1 / 1 / 1** | 12 / 0 / 0 / 0 | 0 / **1 / 1 / 1** | 1 / 1 |
| and2 | 12 / 0 / 0 / 0 | 0.073 / **1 / 1 / 1** | 12 / 0 / 0 / 0 | 0.073 / **1 / 1 / 1** | 1 / 1 |
| maj3 | 12 / 12 / 0 / 0 | 0.023 / 0.023 / **1 / 1** | 12 / 12 / 0 / 0 | 0.023 / 0.023 / **1 / 1** | 2 / 2 |
| no000 | 12 / 12 / 12 / 12 | 0 / 0.013 / 0.163 / 0.193 | 12 / 12 / 0 / 0 | 0 / 0.019 / **1 / 1** | never / 2 |
| tomita4 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 0 / 0 | 0 / 0 / **1 / 1** | never / 2 |
| xor2 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | never |
| parity | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | never |
| adder | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | never |
| tomita2 | 6 / 6 / 6 / 6 | 0.787 at every depth | 6 / 6 / 6 / 6 | 0.787 at every depth | never |
| tomita3 | 12 / 12 / 12 / 12 | 0.030 at every depth | 12 / 12 / 12 / 12 | 0 at every depth | never |
| tomita5 | 12 / 12 / 12 / 12 | 0.017 at every depth | 12 / 12 / 12 / 12 | 0.017 at every depth | never |
| tomita6 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | never |
| tomita7 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | 12 / 12 / 12 / 12 | 0 / 0 / 0 / 0 | never |

(Unseen accuracy, the 8- and 32-stream runs and the midpoint-weight scores are
in `experiments/results/exp8_shift_one_round.csv`. Every training stream
replayed exactly from its key in all 1080 runs.)

- **The outcome is decided by one test.** Over all 1080 runs, whenever the
  next output is a threshold function of the visible state (324 runs),
  training added no hidden neuron. With 128 training streams all 108 of those
  runs were exact on every unseen stream for every sampled weight matrix; with
  32 streams 107 of 108; with 8 streams 73 of 108. Whenever it is not a
  threshold function (756 runs), no run was exact on every unseen stream.
- **Shifts widen the class as expected.** `delay2` and `and2` are solved from
  depth 1 and `maj3` from depth 2, with no hidden neurons and no key. At depth
  0 each needs a full chain of 12 and scores at the majority baseline.
- **Tomita 4 is solved, with the clock and depth 2**: no hidden neurons and
  exact on every unseen stream, where both trainers were below the majority
  baseline before (0.33 against 0.69). For seed 0 the midpoint weights into
  the output are 16 from itself, 5 from each of $x(t), x(t-1), x(t-2)$, 0.5
  from the clock and its first shift, and −18.5 from the clock's second shift.
  So once two steps have passed, the output stays on only if it was on and one
  of the last three inputs was 1.
- **Without the clock Tomita 4 and `no000` fail at every depth.** The shifts
  are 0 before the stream starts, which looks the same as having read 0s, so
  the next output is not a function of the visible state.
- **What shifts do not fix.** `xor2` becomes a function of the visible state
  at depth 1 and the adder does too, but neither is a threshold function, so
  both keep a full chain. Parity depends on the whole history. Tomita 2, 3, 5,
  6 and 7 are not functions of the last four inputs and the current output.
  For all of these, the chain count and the unseen scores are the same at
  every depth.
- **More inputs need more data.** At depth 2 with the clock and only 8
  training streams, `and2` is exact on 0.85 of unseen streams and Tomita 4 on
  0.95; both reach 1 at 32 streams. Midpoint weights were not reliably better
  here: over the 324 threshold-function runs they were exact in 302 against 288
  for the sampled weights, but worse on `and2` with 8 streams (0.81 against
  0.85).
- Training never took more than 0.3 s.

### 8.2 Several rounds (depth 2, the same 32 training streams, 10 seeds)

The streams are fed 4 per round for 8 rounds, or 16 per round for 2 rounds.
After every round, every stream trained so far is replayed from its zero-padded
key with 20 weight matrices. A run "adds chains late" if any round after the
first adds a hidden neuron.

| task | clock | rounds × streams | chain neurons, round 1 | runs that add chains late | neurons at the end | recall of earlier rounds (min) | unseen exact after round 1 | unseen exact at the end | runs exact on every unseen stream |
|---|---|---|---|---|---|---|---|---|---|
| and2 | no | 8 × 4 | 0 | 0 of 10 | 4 | 1 | 1.000 | 1.000 | 10 of 10 |
| and2 | no | 2 × 16 | 0 | 0 of 10 | 4 | 1 | 1.000 | 1.000 | 10 of 10 |
| and2 | yes | 8 × 4 | 0 | 1 of 10 | 11.8 | 1 | 0.946 | 0.955 | 9 of 10 |
| and2 | yes | 2 × 16 | 0 | 0 of 10 | 7 | 1 | 1.000 | 1.000 | 10 of 10 |
| maj3 | no | 8 × 4 | 0 | 0 of 10 | 4 | 1 | 1.000 | 1.000 | 10 of 10 |
| maj3 | no | 2 × 16 | 0 | 0 of 10 | 4 | 1 | 1.000 | 1.000 | 10 of 10 |
| maj3 | yes | 8 × 4 | 0 | 0 of 10 | 7 | 1 | 0.983 | 1.000 | 10 of 10 |
| maj3 | yes | 2 × 16 | 0 | 0 of 10 | 7 | 1 | 1.000 | 1.000 | 10 of 10 |
| no000 | yes | 8 × 4 | 0 | **4 of 10** | 31.7 | 1 | 0.858 | 0.858 | 6 of 10 |
| no000 | yes | 2 × 16 | 0 | 0 of 10 | 7 | 1 | 1.000 | 1.000 | 10 of 10 |
| tomita4 | yes | 8 × 4 | 0 | **4 of 10** | 30.7 | 1 | 0.829 | 0.880 | 6 of 10 |
| tomita4 | yes | 2 × 16 | 0 | 0 of 10 | 7 | 1 | 1.000 | 1.000 | 10 of 10 |
| parity | no | 8 × 4 | 9.4 | 10 of 10 | 79.8 | 1 | 0.001 | 0.001 | 0 of 10 |
| parity | no | 2 × 16 | 12 | 10 of 10 | 27.9 | 1 | 0.001 | 0.001 | 0 of 10 |

(`no000` and Tomita 4 without the clock, and parity with it, are in
`experiments/results/exp8_shift_rounds.csv`. They are not learnable, and every
run adds chain neurons after round 1.)

- **No forgetting held in every check.** After every round of every run, every
  stream from every earlier round replayed exactly for all 20 weight matrices
  (800 round checks), and so did the current round's streams (1000 checks).
  This is the Theorem; the shifts are input neurons and change nothing in it.
- **Learnable tasks in rounds of 16: learning continues with no growth.** In
  all 10 runs of every learnable task, neither round added a hidden neuron,
  the network stayed at its 4 or 7 visible neurons, and every unseen stream
  was exact.
- **A small first round can lock in a wrong box.** With 4 streams per round,
  Tomita 4 and `no000` went wrong in 4 of 10 runs (the same four seeds, which
  share input streams), and `and2` with the clock in 1 of 10. In those runs
  round 1 fits its 4 streams with no hidden neuron, then most later rounds add
  6–12 chain neurons, and the fraction of unseen streams that are exact stays
  at its round-1 value (0.47–0.82) through round 8. The other 6 runs never add
  a hidden neuron and are exact on every unseen stream by round 4 at the
  latest.
- **Why.** A chain is added only when FEAS fails, which means the box kept
  from earlier rounds contains no weight vector that fits the new streams.
  SUBBOX keeps one wide box out of the many that fit the first 4 streams, and
  it can be one that contains no correct weight vector. Boxes only shrink, so
  that run never recovers: every later stream that disagrees is memorised
  through its key. This is the cost of the inner approximation described in
  section 5 of the notes, showing up as a dependence on the size and order of
  the rounds.
- Parity is unchanged: about 9 chain neurons per round and no unseen stream
  right.

## 9 Fixed neurons inside the network (`exp9_fixed_neurons.py`)

The shifts are now computed by the network. `dnn_fixed.py` installs them
before round 1 as visible neurons with fixed weights (a copy of a neuron, then
a copy of the copy), and `dnn.py` trains, unchanged, on sequences extended with
their values. `docs/Fixed_Neurons.md` proves that the Theorem still holds
(Proposition 1) and that the fixed neurons compute their rule on any run that
starts with the hidden neurons at 0 (Proposition 2). Unlike experiment 8, the
shifts can follow the output neuron as well as the inputs.

Configurations:
- *sources*: `inputs` (shifts of the input neurons only, the in-network
  version of experiment 8) or `all` (shifts of the output neuron too);
- *extra*: `none`; `clock` (the always-on input and its shifts); or `twins`
  (the always-on input and, for every shift neuron, a twin that is 1 exactly
  when the shifted value is 0 and the stream has started).

Unseen streams start from their first task state with every fixed and hidden
neuron at 0.

### 9.1 One round (tasks and data of experiment 8)

Smallest depth at which a task is solved with 128 training streams: no hidden
neuron, and every unseen stream exact for every sampled weight matrix in all
3 seeds, at that depth and every larger one tested (up to 3).

| task | inputs, none | inputs, clock | all, none | all, clock | all, twins |
|---|---|---|---|---|---|
| prev_input, tomita1 | 0 | 0 | 0 | 0 | 0 |
| delay2, and2 | 1 | 1 | 1 | 1 | 1 |
| maj3 | 2 | 2 | 2 | 2 | 2 |
| tomita2 | never | never | **1** | **1** | **1** |
| tomita4 | never | 2 | never | 2 | 2 |
| no000 | never | 2 | **2** | 2 | 2 |
| parity, adder, xor2, tomita3, tomita5, tomita6, tomita7 | never | never | never | never | never |

(Chain counts and scores for every depth, size and seed are in
`experiments/results/exp9_fixed_one_round.csv`; the script prints the full
tables.)

- **The guarantees held.** In all 2295 runs every training stream replayed
  exactly from its key, and every fixed neuron's weights from the visible
  neurons were unchanged after training.
- **The same test decides the outcome.** Whenever the next output is a
  threshold function of the visible state (873 runs), no hidden neuron was
  added, and every unseen stream was exact in 291 of 291 such runs with 128
  training streams (287 of 291 with 32, 157 of 291 with 8). Whenever it is not
  (1422 runs), no run was exact on every unseen stream.
- **Computing the shifts inside the network changes nothing.** The `inputs`
  configurations match experiment 8 run for run (810 runs): the same chain
  counts, the same threshold test and the same scores with midpoint weights.
  Scores with sampled weights are identical in 775 runs and differ by at most
  0.16 in the rest, because the random matrices are drawn in a different
  column order.
- **Tomita 2 is solved by a shift of the output**, at depth 1 and without a
  clock: no hidden neuron, exact on every unseen stream. The baseline needed 6
  chain neurons and only matched the majority class. For seed 0 the midpoint
  weights into the output are −3 from the input, −2 from itself, and 0.5 each
  from the input and the output one step earlier. So the output turns on
  exactly when the current input is 0, the previous input was 1 and the output
  was on two steps before, which is "the string so far is $(10)^*$".
- **`no000` no longer needs the clock** once the output is shifted (depth 2).
  Its output is 1 during the first steps, so its own shifts probably mark the
  start of the stream; this explanation was not checked further. Tomita 4
  still needs the clock or the twins.
- **Twins and the clock solve the same tasks**, as Proposition 3 says. Over the
  405 paired runs the threshold test agreed in every one, and one layout added
  no hidden neuron exactly when the other added none. With 128 training
  streams the outcomes are the same.
- **With little data the twins did better than the clock.** With 8 training
  streams, of the 63 runs where the task is a threshold function, the twins
  were exact on every unseen stream in 33 and the clock in 20 (15 runs twins
  only, 2 clock only). On the 57 of those runs that need neither, the count
  with no extra neurons was 31, against 20 with the clock and 32 with the
  twins. So the clock loosens the boxes, as the bias did in experiment 5, and
  the twins did not. With 32 streams the difference is gone (61 against 62).
- **Nothing else moves.** Parity, the adder, `xor2` and Tomita 3, 5, 6 and 7
  keep their full chains and their scores in every configuration and at every
  depth.
- Training never took more than 0.8 s.

### 9.2 Several rounds (shifts of all task neurons, 32 training streams, 10 seeds)

As in 8.2: 4 streams per round for 8 rounds, or 16 per round for 2 rounds, with
every stream trained so far replayed from its zero-padded key with 20 weight
matrices after every round.

| task | depth | extra | rounds × streams | chain neurons, round 1 | runs that add chains late | neurons at the end | recall of earlier rounds (min) | unseen exact after round 1 | unseen exact at the end | runs exact on every unseen stream |
|---|---|---|---|---|---|---|---|---|---|---|
| tomita2 | 1 | none | 8 × 4 | 0 | 1 of 10 | 11.5 | 1 | 0.892 | 0.963 | 9 of 10 |
| tomita2 | 1 | none | 2 × 16 | 0 | 0 of 10 | 4 | 1 | 1.000 | 1.000 | 10 of 10 |
| tomita4 | 2 | clock | 8 × 4 | 0 | **8 of 10** | 54 | 1 | 0.724 | 0.724 | 2 of 10 |
| tomita4 | 2 | clock | 2 × 16 | 0 | 1 of 10 | 10.2 | 1 | 0.973 | 0.973 | 9 of 10 |
| tomita4 | 2 | twins | 8 × 4 | 0 | 3 of 10 | 27.8 | 1 | 0.834 | 0.900 | 7 of 10 |
| tomita4 | 2 | twins | 2 × 16 | 0 | 0 of 10 | 11 | 1 | 0.978 | 1.000 | 10 of 10 |
| maj3 | 2 | none | 8 × 4 | 0 | 0 of 10 | 6 | 1 | 1.000 | 1.000 | 10 of 10 |
| maj3 | 2 | none | 2 × 16 | 0 | 0 of 10 | 6 | 1 | 1.000 | 1.000 | 10 of 10 |
| parity | 2 | none | 8 × 4 | 9.2 | 10 of 10 | 80.1 | 1 | 0.001 | 0.001 | 0 of 10 |
| parity | 2 | none | 2 × 16 | 12 | 10 of 10 | 29.9 | 1 | 0.001 | 0.001 | 0 of 10 |

- **No forgetting held in every check**: 400 round checks of earlier rounds
  and 500 of the current round, with shifts of the output neuron in the
  network.
- **The wrong-box problem is worse with more fixed neurons.** With the clock,
  Tomita 4 in rounds of 4 streams went wrong in 8 of 10 runs, against 4 of 10
  in 8.2, where only the input was shifted. One run went wrong even in rounds
  of 16. Every such run stayed below 1 on unseen streams to the end.
- **The twins went wrong less often than the clock**: 3 of 10 runs in rounds
  of 4 and none in rounds of 16. This fits 9.1, but 10 seeds are too few to be
  sure of the size of the difference.
- Tomita 2 went wrong in 1 of 10 runs in rounds of 4 and in none in rounds
  of 16.

### 9.3 Random sequences (design of experiment 2, means over 5 seeds)

4 task neurons, with no input or with neuron 0 as an input, and shifts of all
4 at depths 0–3. 10 rounds of 3 random sequences of 6 vectors. "Recall without
a key" starts each trained sequence from its first task state with every fixed
and hidden neuron at 0 (5 weight matrices).

| inputs | depth | fixed neurons | chain neurons added per round | chain neurons after 10 rounds | total neurons after 10 rounds | stored numbers | recall from keys, every round (min) | recall without a key after 10 rounds |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 0 | 7.48 | 74.8 | 78.8 | 12 514 | 1 | 0.000 |
| none | 1 | 4 | 6.34 | 63.4 | 71.4 | 10 285 | 1 | 0.000 |
| none | 2 | 8 | 5.56 | 55.6 | 67.6 | 9 228 | 1 | 0.040 |
| none | 3 | 12 | 5.24 | 52.4 | 68.4 | 9 437 | 1 | 0.060 |
| neuron 0 | 0 | 0 | 6.94 | 69.4 | 73.4 | 10 881 | 1 | 0.000 |
| neuron 0 | 1 | 4 | 5.82 | 58.2 | 66.2 | 8 845 | 1 | 0.000 |
| neuron 0 | 2 | 8 | 4.96 | 49.6 | 61.6 | 7 683 | 1 | 0.040 |
| neuron 0 | 3 | 12 | 4.64 | 46.4 | 62.4 | 7 893 | 1 | 0.060 |

- **No forgetting held on unstructured data too.** After every round of every
  run, every sequence trained so far replayed exactly from its key for all 20
  weight matrices (360 round checks of earlier rounds), and the fixed neurons'
  weights were unchanged.
- **Shifts save some helper neurons, but not many.** At depth 3 a round adds
  about 5 chain neurons where the baseline adds about 7. The saving is largest
  in round 1 (1.2 against 7.8 without inputs); later rounds add 5–6.5 against
  6.4–8. After
  10 rounds the network is about 15% smaller at depth 2, and depth 3 is no
  better because its 4 extra fixed neurons cost as much as they save.
- **Random sequences are still memorised.** Almost none of them replay
  without a key (0–6%). A window of earlier states tells conflicting states
  apart, but the successor is rarely a threshold function of it, so the chains
  still do the work.

## 10 A feature library and state neurons (`exp10_library_and_states.py`)

The plan was to install the causal variant's features before round 1, where
Proposition 1 of `docs/Fixed_Neurons.md` makes them safe from forgetting.
Working out what they could reach gave a result first
(`docs/Latency_And_State.md`, Proposition 4): **six of the stream tasks cannot
be computed at the experiments' timing by any network without a key**,
whatever hidden neurons it has. The tasks ask for $y(t+1)$ as a function of
the inputs up to $x(t)$, so the output must react to the current input
through one fixed weight. Parity needs that weight to be positive when the
parity so far is even and negative when it is odd.

So every task is run twice: at *latency 0*, as before, and at *latency 1*,
with the same outputs one step later.

Layouts, all with the always-on input and all made of fixed neurons:
- **shifts**: past shifts of depth 2 of every task neuron (4 fixed neurons, 6
  for the adder). The reference.
- **library**: the same, plus every feature template of the variant (copy,
  latch, AND, OR, AND-NOT, set-reset) over all those neurons and pairs of
  them: 105 fixed neurons (238 for the adder). It is the same for every
  task; duplicates are removed on random probe sequences, not on training
  data.
- **states**: the state neurons of the task's own finite machine (3 to 11
  fixed neurons), which is Minsky's construction. This layout is built from
  the task's definition, so it shows what can be represented, not what is
  found. Only the output neuron is trained.

### 10.1 One round (128 training streams, 3 seeds)

"yes" means no hidden neuron in any seed. A number in brackets is the
fraction of unseen streams that are exact for the sampled weight matrices
when that is below 1; the midpoint weights were exact on every unseen stream
in all those cases.

| task | machine states | possible at latency 0 | latency 0: shifts | latency 0: library | latency 0: states | latency 1: shifts | latency 1: library | latency 1: states |
|---|---|---|---|---|---|---|---|---|
| prev_input | 1 | yes | yes | yes | yes | yes | yes | yes |
| tomita1 | 2 | yes | yes | yes | yes | yes | yes | yes |
| delay2 | 2 | yes | yes | yes | yes | yes | yes | yes |
| and2 | 2 | yes | yes | yes | yes | yes | yes | yes |
| maj3 | 4 | yes | yes | yes (0.988) | yes | – | yes (0.999) | yes |
| no000 | 3 | yes | yes | yes (0.997) | yes | – | yes (0.998) | yes |
| tomita2 | 3 | yes | yes | yes | yes | – | yes (0.999) | yes |
| tomita4 | 4 | yes | yes | yes | yes | – | yes | yes |
| tomita7 | 5 | yes | – | – | **yes** | – | – | **yes** |
| parity | 2 | **no** | – | – | – | – | **yes** (0.996) | **yes** |
| xor2 | 2 | **no** | – | – | – | – | **yes** (0.997) | **yes** |
| adder | 2 | **no** | – | – | – | – | – | **yes** |
| tomita3 | 5 | **no** | – | – | – | – | – | **yes** |
| tomita5 | 4 | **no** | – | – | – | – | – | **yes** |
| tomita6 | 3 | **no** | – | – | – | – | – | **yes** |

(Chain counts, scores and the 8- and 32-stream runs are in
`experiments/results/exp10_one_round.csv`.)

- **The theory predicted every run of the states layout** (270 of 270). At
  latency 1 no hidden neuron was added for any of the 15 tasks. At latency 0
  none was added for the 9 tasks that pass the test of Proposition 4, and a
  full chain was added for the 6 that fail it, in all 54 runs. Wherever no
  hidden neuron was added, every unseen stream was exact with 32 or 128
  training streams (72 of 72 runs each; 57 of 72 with 8).
- **Every task is now solved somewhere.** Parity, the adder and Tomita 3, 5
  and 6 are exact on unseen streams at latency 1, and Tomita 7 at latency 0,
  once the right state neurons exist.
- **The library finds parity.** With the same 105 neurons it is given for
  every task, the output learns parity at latency 1 with no hidden neuron.
  It also finds xor2, and at latency 1 it solves Tomita 2, Tomita 4, maj3 and
  no000, which shifts of depth 2 alone do not.
- **The library does not find** the adder or Tomita 3, 5, 6 and 7. At latency
  0 it solves exactly the tasks the shifts solve.
- **The library's boxes are loose.** Of its 54 runs with no hidden neuron at
  128 streams, the sampled weight matrices were exact on every unseen stream
  in 41 and the midpoint weights in all 54. With 8 training streams parity is
  exact on about half of the unseen streams.
- **The same test decides the outcome** in all 810 runs: no hidden neuron when
  the next output is a threshold function of the visible state (486 runs),
  never exact on unseen streams when it is not (324).
- Every training stream replayed from its key in every run. Training took at
  most 2.4 s with the library and 0.3 s otherwise.

### 10.2 Longer streams

Networks trained on 128 streams of 13 vectors, tested on 200 streams of 101
vectors (means over 3 seeds).

| task | latency | layout | long streams exact, sampled weights | long streams exact, midpoint weights |
|---|---|---|---|---|
| parity | 1 | library | 0.925 | 1.000 |
| xor2 | 1 | library | 0.961 | 1.000 |
| tomita2 | 1 | library | 0.999 | 1.000 |
| tomita4 | 1 | library | 1.000 | 1.000 |
| maj3 | 1 | library | 0.896 | 1.000 |
| no000 | 1 | library | 0.953 | 0.998 |
| parity, adder, tomita3, tomita5, tomita6 | 1 | states | 1.000 | 1.000 |
| tomita7 | 0 | states | 1.000 | 1.000 |

- **What was learned is the machine**, not a trick for short streams. With
  midpoint weights the library's parity network is exact on streams eight
  times longer than any it was trained on. The sampled weights lose a little
  more on long streams, which is the loose boxes again.

### 10.3 Two-step rule 90 (the task and data of experiment 6, 3 seeds)

Ring of 6 cells, two network steps per automaton step, library over the 6
cells (102 fixed neurons). "none" is the baseline with the always-on input.

| layout | training states | chain neurons | train s | unseen exact, sampled | unseen exact, midpoint | unseen cell accuracy | baseline |
|---|---|---|---|---|---|---|---|
| none | 8 | 26 | 0.15 | 0.054 | 0.054 | 0.749 | 0.749 |
| none | 32 | 44 | 0.38 | 0.052 | 0.052 | 0.740 | 0.740 |
| library | 2 | 0 | 0.03 | 0.044 | 0.048 | 0.686 | 0.749 |
| library | 4 | 0 | 0.03 | 0.090 | 0.094 | 0.747 | 0.749 |
| library | 8 | 0 | 0.04 | 0.287 | 0.357 | 0.838 | 0.749 |
| library | 16 | 0 | 0.06 | 0.622 | 0.778 | 0.922 | 0.752 |
| library | 32 | 0 | 0.10 | 0.869 | **0.990** | 0.977 | 0.740 |

- **The library matches the variant's greedy search and is about 100 times
  faster.** With 32 training states it is exact on 0.87 of the unseen states
  (0.99 with midpoint weights), against 0.85 for the variant in section 6,
  and trains in 0.1 s against 11 s, because nothing is searched. No hidden
  neuron is added at any size.
- It needs the data: with 8 training states it is exact on 0.29 of the unseen
  states.

### 10.4 Several rounds (32 training streams, 10 seeds)

As in 8.2 and 9.2. The last column counts runs that are exact on every unseen
stream at the end, with sampled weights and with midpoint weights.

| task | latency | layout | rounds × streams | runs that add chains late | neurons at the end | recall of earlier rounds (min) | unseen exact after round 1 | unseen exact at the end | runs exact at the end (sampled / midpoint) |
|---|---|---|---|---|---|---|---|---|---|
| parity | 1 | states | 8 × 4 | 0 of 10 | 8 | 1 | 0.973 | 1.000 | 10 / 10 |
| parity | 1 | states | 2 × 16 | 0 of 10 | 8 | 1 | 1.000 | 1.000 | 10 / 10 |
| tomita7 | 0 | states | 8 × 4 | 1 of 10 | 16.4 | 1 | 0.753 | 0.958 | 9 / 9 |
| tomita7 | 0 | states | 2 × 16 | 0 of 10 | 14 | 1 | 1.000 | 1.000 | 10 / 10 |
| xor2 | 1 | library | 8 × 4 | 0 of 10 | 108 | 1 | 0.746 | 0.987 | 2 / 10 |
| xor2 | 1 | library | 2 × 16 | 0 of 10 | 108 | 1 | 0.933 | 0.991 | 4 / 10 |
| parity | 1 | library | 8 × 4 | **10 of 10** | 169.8 | 1 | 0.224 | 0.266 | 0 / 0 |
| parity | 1 | library | 2 × 16 | 1 of 10 | 109.7 | 1 | 0.866 | 0.938 | 2 / 9 |
| tomita4 | 0 | library | 8 × 4 | 5 of 10 | 124.6 | 1 | 0.649 | 0.847 | 1 / 4 |
| tomita4 | 0 | library | 2 × 16 | 1 of 10 | 109.2 | 1 | 0.959 | 0.969 | 5 / 9 |

- **No forgetting held in every check** (400 round checks of earlier rounds,
  500 of the current round), with up to 105 fixed neurons in the network.
- **With state neurons, learning continues across small rounds.** Parity
  trained 4 streams per round never added a hidden neuron and was exact on
  every unseen stream in all 10 runs. Tomita 7 went wrong in 1 of 10.
- **With the library, small rounds mostly fail.** Parity went wrong in all 10
  runs in rounds of 4 and Tomita 4 in 5 of 10. In rounds of 16 each went
  wrong once. This is the wrong-box problem of 8.2 and 9.2 with 105
  coordinates in place of a handful.
- xor2 with the library never added a hidden neuron, and its score on unseen
  streams rose from 0.75 after the first round to 0.99 after the last.

## 11 The wrong-box problem: box rules and a constraint buffer (`exp11_box_rule.py`)

Sections 8.2, 9.2 and 10.4 found that a learnable task trained in small rounds
can keep a box that contains no correct weight vector. `docs/Box_Rule.md`
finds the cause, proves that no rule for choosing the box avoids it
(Proposition 6), and proposes a *constraint buffer*: the trained neuron keeps
its first $K$ distinct constraints exactly and only turns them into a box when
there are more (Proposition 7). `dnn.py` is unchanged; `dnn_box.py` and
`dnn_buffer.py` swap the rule in for the duration of a training call.

Nine cases from experiments 8–10, all of them threshold functions of the
visible state, so that no hidden neuron is ever needed. A run *goes wrong* if
any round after the first adds one. 32 training streams, 10 seeds.

### 11.1 Other box rules, no buffer

Runs that go wrong, of 10, with 4 streams per round (8 rounds). "cap 16" is
the notes' rule with widths capped at 16; "bounded" and "volume" are described
in `docs/Box_Rule.md`.

| case | visible neurons | notes' rule | cap 16 | bounded 8 | volume 8 | volume 32 |
|---|---|---|---|---|---|---|
| tomita4, input shifts + clock | 7 | 4 | 4 | 2 | 1 | 3 |
| tomita4, all shifts + clock | 9 | 8 | 8 | 4 | 4 | 3 |
| tomita4, all shifts + twins | 11 | 3 | 3 | 4 | 1 | 1 |
| tomita2, all shifts | 4 | 1 | 1 | 2 | 3 | 3 |
| tomita7, states | 14 | 1 | 1 | 2 | 0 | 0 |
| parity (latency 1), states | 8 | 0 | 0 | 0 | 0 | 0 |
| parity (latency 1), library | 108 | 10 | 10 | 10 | 10 | 10 |
| tomita4, library | 108 | 5 | 5 | 7 | 8 | 8 |
| xor2 (latency 1), library | 108 | 0 | 0 | 9 | 9 | 9 |
| **total, of 90** | | **32** | 32 | 40 | 36 | 37 |
| total with 16 streams per round, of 90 | | **3** | 3 | 13 | 5 | 5 |

- **No rule is better overall.** The largest-volume rule roughly halves the
  failures on the six small layouts (9 or 10 of 60 against 17), but with the
  library it nearly doubles them (27 of 30 against 15). The notes' rule has
  the fewest failures in total at both round sizes.
- **The cap does not matter.** Cap 16 went wrong in exactly the same runs as
  cap 1.
- This is what Proposition 6 leads one to expect: a rule moves the failures
  to other tasks.
- Every rule keeps the Theorem: all trained streams replayed from their keys
  at the end of all 900 runs.

### 11.2 The constraint buffer

The notes' rule is used when a buffer overflows. Capacity 0 is the algorithm
of the notes. Runs that go wrong:

| streams per round | runs | capacity 0 | capacity 16 | capacity 64 | capacity 256 |
|---|---|---|---|---|---|
| 1 (32 rounds; the six cases without the library) | 60 | 42 | 16 | **0** | **0** |
| 4 (8 rounds) | 90 | 32 | 29 | 1 | **0** |
| 16 (2 rounds) | 90 | 3 | 3 | 1 | **0** |
| runs exact on every unseen stream at the end, midpoint weights | 240 | 162 | 191 | 238 | **240** |

Largest number of constraints stored in any run, at capacity 256 (no
overflow):

| case | visible neurons | constraints stored |
|---|---|---|
| tomita4, input shifts + clock | 7 | 22 |
| tomita4, all shifts + clock | 9 | 28 |
| tomita4, all shifts + twins | 11 | 28 |
| tomita2, all shifts | 4 | 10 |
| tomita7, states | 14 | 22 |
| parity (latency 1), states | 8 | 10 |
| parity (latency 1), library | 108 | 110 |
| tomita4, library | 108 | 91 |
| xor2 (latency 1), library | 108 | 156 |

- **With a buffer that does not overflow, no run goes wrong** (0 of 240), and
  every run is exact on every unseen stream at the end with midpoint weights.
  This is Proposition 7(3).
- **This includes training one stream at a time.** Without a buffer 42 of 60
  such runs go wrong and end with 68 hidden neurons on average. With it, none
  goes wrong and no hidden neuron is added.
- **The buffer stays small.** It holds one entry per distinct pair of visible
  state and next output, however many streams are trained: 10 to 28 for the
  small layouts and 91 to 156 with the library. The largest was 3.1 times the
  number of visible neurons.
- **A buffer that overflows early helps less.** At capacity 16, 48 of the 240
  runs still go wrong (77 without a buffer). At capacity 64 the two failures
  are library runs that overflowed.
- **No forgetting held throughout**: after every round of all 960 runs, every
  stream trained so far replayed from its key for 5 weight matrices taken from
  the regions.
- **Which weight vector is used matters.** With a buffer the region is not a
  box. Taking the point nearest the box's midpoint was exact on every unseen
  stream in all 240 runs at capacity 256; points nearest the corner and random
  samples were exact in 186.

## 12 How many steps to give the output (`exp12_latency.py`)

A training sequence has to say when the output for an input prefix appears:
$h$ steps after the last input it depends on. Experiments 4–9 used $h = 1$
and experiment 10 added $h = 2$ (its "latency 1"). Here $h = 1, \dots, 4$,
for the library and the state-neuron layouts of experiment 10, 128 training
streams, 3 seeds. The table counts the tasks, of 15, for which no hidden
neuron was added in any seed.

| layout | $h = 1$ | $h = 2$ | $h = 3$ | $h = 4$ |
|---|---|---|---|---|
| library | 8 | 10 | 7 | 3 |
| states | 9 | **15** | 8 | 1 |

(The table for each task is printed by the script; the runs are in
`experiments/results/exp12_latency.csv`.)

- **$h$ has to be chosen, and both too little and too much fail.** With
  $h = 1$ six tasks are impossible (Proposition 4). With $h = 2$ the state
  neurons solve all 15, as Proposition 5 says. With $h = 3$ or $4$ fewer tasks
  are solved, because the network must then hold the answer for the extra
  steps and has no neurons to hold it with. The library loses tasks more
  slowly, probably because it includes shifts of depth 2.
- **$h = 2$ was the best value for both layouts**, and by Proposition 5 it is
  enough in principle for every finite-state task.
- In all 183 runs with no hidden neuron the midpoint weights were exact on
  every unseen stream; in the 177 runs with hidden neurons none was.

## 13 Finding the state neurons from the data (`exp13_state_merging.py`)

Experiment 10 solved every task with the state neurons of the task's machine,
but those were built from the task's definition. `dnn_merge.py` finds a
machine from the training streams alone (`docs/State_Merging.md`): it builds
the prefix tree of the streams, which is a machine that has memorised them,
and merges its nodes into states wherever the data do not contradict it.
This is the RPNI algorithm of Oncina and García (1992). The state neurons of
the merged machine are installed before round 1 and the output is trained by
`dnn.py`, unchanged. Nothing about the task is given. Output timing is
$h = 2$ (experiment 10's "latency 1").

### 13.1 One round (15 tasks, 10 seeds)

Runs, of 10, that are exact on every unseen stream (midpoint weights), by
number of training streams:

| task | states of the task's machine | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| prev_input | 1 | 10 | 10 | 10 | 10 | 10 | 10 |
| parity | 2 | 10 | 10 | 10 | 10 | 10 | 10 |
| xor2 | 2 | 10 | 10 | 10 | 10 | 10 | 10 |
| delay2 | 2 | 10 | 10 | 10 | 10 | 10 | 10 |
| adder | 2 | 9 | 10 | 10 | 10 | 10 | 10 |
| and2 | 2 | 9 | 9 | 10 | 10 | 10 | 10 |
| tomita1 | 2 | 7 | 10 | 10 | 10 | 10 | 10 |
| tomita6 | 3 | 8 | 9 | 10 | 10 | 10 | 10 |
| tomita5 | 4 | 3 | 8 | 10 | 10 | 10 | 10 |
| maj3 | 4 | 1 | 7 | 10 | 10 | 10 | 10 |
| tomita7 | 5 | 0 | 5 | 10 | 10 | 10 | 10 |
| tomita3 | 5 | 1 | 2 | 8 | 10 | 10 | 10 |
| no000 | 3 | 5 | 5 | 9 | 9 | 10 | 10 |
| tomita4 | 4 | 1 | 4 | 7 | 9 | 10 | 10 |
| tomita2 | 3 | 0 | 4 | 5 | 9 | 9 | 10 |
| **all 15, of 150** | | 84 | 113 | 139 | 147 | 149 | **150** |

Totals over the 15 tasks. "tree" is the prefix tree with no merging, run up
to 32 streams.

| layout | training streams | runs | task's machine found | exact on every unseen stream, sampled weights | midpoint weights | exact on every stream of 101 vectors, midpoint weights | training streams exact without a key | hidden neurons added | states (mean) | fixed neurons (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| merged | 4 | 150 | 83 | 67 | 84 | 84 | 150 | 0 | 3.2 | 7.7 |
| merged | 8 | 150 | 113 | 100 | 113 | 113 | 150 | 0 | 3.3 | 7.8 |
| merged | 16 | 150 | 139 | 139 | 139 | 139 | 150 | 0 | 3.0 | 7.3 |
| merged | 32 | 150 | 147 | 147 | 147 | 147 | 150 | 0 | 2.9 | 7.2 |
| merged | 64 | 150 | 149 | 149 | 149 | 149 | 150 | 0 | 2.9 | 7.2 |
| merged | 128 | 150 | **150** | **150** | **150** | **150** | 150 | 0 | 2.9 | 7.1 |
| tree | 4 | 150 | 0 | 0 | 0 | – | 150 | 0 | 40.1 | 87.1 |
| tree | 8 | 150 | 0 | 0 | 0 | – | 150 | 0 | 72.0 | 156.1 |
| tree | 16 | 150 | 0 | 0 | 0 | – | 150 | 0 | 130.8 | 283.4 |
| tree | 32 | 150 | 0 | 0 | 0 | – | 150 | 0 | 232.9 | 506.3 |

- **Every task is learned from the data.** With 128 training streams the
  merged machine is the task's machine in all 150 runs, and the network is
  exact on every unseen stream, including streams eight times longer than
  the training streams. This includes the adder and Tomita 3, 5, 6 and 7,
  which the feature library of experiment 10 did not find, and it uses about
  7 fixed neurons where the library used 105.
- **It needs little data.** 32 streams are enough in 147 of 150 runs, 16 in
  139 and 8 in 113. Tasks with more states, or with states that are rarely
  told apart (Tomita 2, 3, 4), need the most.
- **Right machine, right answers.** In all 781 runs where the task's machine
  was found, every unseen stream and every long stream was exact. In the
  other 119 runs one was exact on the long streams. The machine found had
  more states than the task's in 62 runs and fewer in 23.
- **The training streams never need a key** (Proposition 8). In all 1500
  runs, with the merged machine or with the tree, no hidden neuron was added
  and every training stream was reproduced from its first state alone.
- **The tree memorises and does not generalise.** It is exact on its training
  streams and on 11 to 14% of unseen ones, and it grows with the data (233
  states for 32 streams). Merging is what turns it into the task's machine.
- Merging took at most 0.05 s and training the merged layout 0.06 s.

### 13.2 Several rounds (32 training streams, 10 seeds)

The first $S$ streams are trained together as round 1 and give the machine.
The rest follow 4 per round. After every round every stream trained so far is
replayed from its key with 20 weight matrices.

| task | $S$ | runs with the task's machine | runs that add hidden neurons after round 1 | hidden neurons at the end (mean) | runs exact on every unseen stream at the end |
|---|---|---|---|---|---|
| parity | 4 | 10 | 0 | 0 | 10 |
| adder | 4 | 8 | 1 | 7.1 | 9 |
| tomita3 | 4 | 2 | 8 | 44 | 2 |
| tomita4 | 4 | 1 | 8 | 49.8 | 2 |
| tomita5 | 4 | 3 | 7 | 59.9 | 3 |
| tomita7 | 4 | 2 | 8 | 53.1 | 2 |
| **all six, of 60** | 4 | 26 | 32 | | 28 |
| all six, of 60 | 8 | 42 | 15 | | 44 |
| all six, of 60 | 16 | 56 | 4 | | 56 |
| all six, of 60 | 32 | 59 | 0 | | 59 |

(The rows for each task at $S$ = 8, 16 and 32 are printed by the script and
are in `experiments/results/exp13_rounds.csv`.)

- **No forgetting held in every check** (1020 round checks of earlier rounds,
  1260 of the current round).
- **The machine is settled by the first streams.** Found from 4 streams, it is
  the task's machine in 26 of 60 runs. In 32 of the other 34, later streams
  contradict it and are memorised through helper chains (67 hidden neurons on
  average by the end), and unseen streams stay wrong. From 16 streams it is
  right in 56 of 60 runs and from 32 in 59.
- **This is the wrong-box problem one level up.** A merge, like a box, is a
  commitment made from the data seen so far. The remedy is the same: keep the
  first streams until there are enough to settle the machine. Here 16 to 32
  streams of 13 vectors were enough.

## 14 Boolean functions of six inputs at depth 2 (`exp14_boolean_functions.py`)

Ladder rung 1. The output at time $t + 2$ is $f(x(t))$ for a fixed Boolean
function $f$ of six inputs, so a hidden layer is needed between them. The
network is layered: the output's weights from the inputs and from itself are
fixed at 0. Three layouts, all trained by `dnn.py` unchanged; each has the
always-on neuron and a "started" fixed neuron (0 at time 0, then 1), which
gives the output a bias from time 2 on.

- **baseline**: nothing else; TRAIN adds predictor chains.
- **lookup**: one fixed helper per positive input pattern seen in training.
- **upstart**: fixed helpers built from the training data by `dnn_upstart.py`
  in the style of the upstart algorithm (Frean 1990): the best perceptron of
  $f$, then helpers for the patterns that are wrongly on or wrongly off,
  each itself a best-fit perceptron of the inputs (fewest violations, then
  smallest sum of $|w_j|$). After 20 helpers the rest are indicators.

The helpers read only the inputs, so they are fixed neurons installed before
round 1 and `docs/Fixed_Neurons.md` applies: nothing is forgotten. Training: 1
to 16 streams of 13 vectors (11 labelled patterns each, of 64 possible), 10
seeds. Testing: 200 random streams, run with no key; each output from $t = 2$
on is scored, split by whether its pattern was seen in training. Run on
Lucas's machine (Python 3.14.10, scipy 1.18.1), commit `9a7f228`.

### 14.1 Results

Mean accuracy on unseen patterns (midpoint weights), upstart (lookup in
brackets; the baseline equals lookup to within 0.02, except and3 at 1 stream).

| task | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| majority | 0.703 (0.364) | 0.887 (0.343) | 0.965 (0.375) | 1.000 (0.427) | 1.000 (0.331) |
| and3 | 0.866 (0.879) | 0.946 (0.887) | 1.000 (0.905) | 1.000 (0.933) | 1.000 (0.953) |
| xor2 | 0.647 (0.520) | 1.000 (0.506) | 1.000 (0.484) | 1.000 (0.485) | 1.000 (0.552) |
| dnf3 | 0.642 (0.414) | 0.701 (0.406) | 0.787 (0.423) | 0.773 (0.452) | 0.889 (0.364) |
| mux | 0.569 (0.484) | 0.619 (0.469) | 0.643 (0.468) | 0.730 (0.435) | 0.667 (0.337) |
| equal3 | 0.718 (0.887) | 0.721 (0.882) | 0.787 (0.873) | 0.765 (0.865) | 0.768 (0.878) |
| parity6 | 0.440 (0.501) | 0.434 (0.507) | 0.345 (0.518) | 0.518 (0.513) | 0.531 (0.531) |
| random | 0.482 (0.452) | 0.467 (0.451) | 0.455 (0.455) | 0.506 (0.466) | 0.387 (0.527) |

Totals over the 8 tasks:

| layout | streams | patterns seen | helpers | hidden neurons added | seen acc | unseen acc (mid / sampled) | training exact without key |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 31.7 | 0 | 708 | 0.558 | 0.563 / 0.563 | 0 of 80 |
| baseline | 16 | 59.9 | 0 | 891 | 0.561 | 0.559 / 0.559 | 0 of 80 |
| lookup | 4 | 31.7 | 14.0 | 0 | 1.000 | 0.563 / 0.563 | 80 of 80 |
| lookup | 16 | 59.9 | 26.3 | 0 | 1.000 | 0.559 / 0.559 | 80 of 80 |
| upstart | 4 | 31.7 | 4.6 | 0 | 1.000 | 0.748 / 0.746 | 80 of 80 |
| upstart | 16 | 59.9 | 13.4 | 0 | 1.000 | 0.780 / 0.780 | 80 of 80 |

Upstart needed no hidden neuron from TRAIN in any of 400 runs.

### 14.2 Findings

- **Helpers computed from the inputs generalise across input vectors**,
  which state merging cannot do. xor2 is exact on unseen patterns from 2
  streams (19 of 64 patterns seen), and3 from 4, majority from 8;
  lookup and the baseline stay at the constant answer.
- dnf3 (an OR of three ANDs) improves steadily, to 0.89; mux stops near 0.7.
  Both need several helpers that must be found together, and the
  one-error-set-at-a-time search finds them only partly.
- **equal3 is worse than the constant answer** (0.72–0.79 against about
  0.88): its positives are 8 isolated patterns, and helpers fitted to them
  fire on neighbours. Parity6 and the random function gain nothing, as they
  should not.
- Upstart uses about half as many helpers as lookup and recalls the training
  streams exactly without keys. The baseline needs keys for every stream.
- Sampled and midpoint weights agree to within 0.004: the output's box is
  tight once the helpers are fixed.
- An earlier version (first run, commit `5665b6d`) let the output read its
  own last value and required it to be 0 when every helper was 0. Seen
  accuracy was 0.994–0.998 and majority reached only 0.72. Both are fixed.

## 15 Wake-and-sleep cycles (`exp15_wake_sleep.py`)

Lucas's idea 2 (expansion and compression), one version. `dnn_sleep.py`
gives the sleep step: state merging on a buffer of streams, a fresh network
trained on the buffer, then safe deletion of every neuron whose outgoing
weight ranges all contain 0 (safe by the Theorem; the argument is in the
module docstring). The six tasks of experiment 13 part B, 32 streams arriving
4 per round (8 rounds), 10 seeds. Round 1 builds the network from its 4
streams by state merging. Strategies:

- **fixed**: wake only; the machine from the first 4 streams is kept
  (experiment 13 part B with S = 4).
- **sleep**: sleep after every even round, buffer of every stream so far.
- **sleep16**: the same, with a buffer of the 16 most recent streams.

Run on Lucas's machine, commit `138d06d`. The "fixed" rows reproduce
experiment 13 part B, S = 4 (26 true machines, 28 runs exact at the end).

### 15.1 Results (60 runs per row; * = a sleep round)

| strategy | round | streams seen | buffer | true machine | exact on every unseen stream (midpoint) | recall from keys (min) | recall of all streams seen, no key (mean) | hidden neurons (mean) |
|---|---|---|---|---|---|---|---|---|
| fixed | 2 | 8 | – | 26 | 27 | 1.000 | 0.848 | 5.2 |
| fixed | 4 | 16 | – | 26 | 27 | 1.000 | 0.773 | 14.3 |
| fixed | 8 | 32 | – | 26 | 28 | 1.000 | 0.734 | 35.6 |
| sleep | 2* | 8 | 8 | 42 | 42 | 1.000 | 1.000 | 0.0 |
| sleep | 3 | 12 | 12 | 42 | 42 | 1.000 | 0.972 | 1.6 |
| sleep | 4* | 16 | 16 | 56 | 56 | 1.000 | 1.000 | 0.0 |
| sleep | 8* | 32 | 32 | 59 | 59 | 1.000 | 1.000 | 0.0 |
| sleep16 | 6* | 24 | 16 | 57 | 57 | 1.000 | 0.995 | 0.0 |
| sleep16 | 8* | 32 | 16 | 57 | 57 | 1.000 | 0.996 | 0.0 |

Runs exact on every unseen stream at the end, by task (fixed / sleep /
sleep16): parity 10 / 10 / 10, adder 9 / 10 / 10, tomita3 2 / 10 / 10,
tomita4 2 / 9 / 7, tomita5 3 / 10 / 10, tomita7 2 / 10 / 10.

Safe deletion: 0 neurons deleted in 600 sleeps, and 0 deletable after any
wake round in any strategy (the "fixed" networks then had 20 hidden neurons
on average).

### 15.2 Findings

- **Sleep undoes a wrong early commitment.** Rebuilding from the buffer
  finds the true machine in 59 of 60 runs by round 6, against 26 for the
  machine kept from round 1, and the rebuilt network needs no hidden neurons.
  This is experiment 13 part A reached incrementally.
- **Nothing in the buffer is forgotten**: recall from keys is 1.000 in every
  row. Between sleeps, streams the machine does not fit are memorised with
  keys (the dip to 0.972 without keys at round 3), and the next sleep absorbs
  them.
- **Without sleep, memory grows and unkeyed recall falls**: 36 hidden
  neurons by round 8, and only 73% of the streams seen replay without keys.
- **A bounded buffer forgets a little.** With the 16 most recent streams, 3
  runs end with a wrong machine (all tomita4), built from a buffer that no
  longer holds the streams that ruled it out; about 0.4% of the streams seen
  are then wrong without keys. What may safely leave the buffer is still
  open.
- **Safe deletion found nothing to delete.** Every memorised neuron is read
  by the neuron it was added for, and after a sleep there are no silent
  neurons. The rule removes only neurons nothing uses; the compression in
  this experiment is entirely the rebuild.

## 16 Helpers for OR-of-AND functions (`exp16_or_of_ands.py`)

Next step 1. The tasks, data and scoring of experiment 14 (six inputs, 1 to
16 training streams, 10 seeds, 200 random test streams), three layouts, all
trained by `dnn.py` unchanged:

- **upstart**: experiment 14's helpers (h = 2).
- **cover**: a covering search (h = 2, `dnn_upstart.build_cover`). Each
  helper is off on every negative pattern seen and on for one uncovered
  positive (the seed), with the smallest sum of $|w_j|$; the output is then
  the OR of the helpers. On the full truth table it finds the 3 terms of dnf3
  and the 4 terms of mux.
- **pairs3**: a second hidden layer (h = 3, `dnn_upstart.make_layout_h3`).
  Layer 1 is every AND of two literals (60 fixed neurons, the same for every
  task); layer 2 is built by upstart over the inputs and layer 1. The output
  at $t + 3$ is $f(x(t))$, so a training stream labels 10 patterns, not 11.

All helpers are fixed neurons, so nothing is forgotten. Run on Lucas's
machine, commit `372dd49`. The upstart rows equal experiment 14's in every
column except the timings.

### 16.1 Results

Mean accuracy on unseen patterns (midpoint weights). Unseen patterns per run
(mean, h = 2 / h = 3): 54 / 55 at 1 stream, 32 / 35 at 4, 4 / 6 at 16, so
the 16-stream column rests on a few patterns per run.

| task | layout | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| majority | upstart | 0.703 | 0.887 | 0.965 | 1.000 | 1.000 |
| majority | cover | 0.694 | 0.714 | 0.713 | 0.603 | 0.539 |
| majority | pairs3 | 0.685 | 0.724 | 0.755 | 0.838 | 1.000 |
| and3 | upstart | 0.866 | 0.946 | 1.000 | 1.000 | 1.000 |
| and3 | cover | 0.822 | 0.900 | 1.000 | 1.000 | 1.000 |
| and3 | pairs3 | 0.820 | 0.948 | 1.000 | 1.000 | 1.000 |
| xor2 | upstart | 0.647 | 1.000 | 1.000 | 1.000 | 1.000 |
| xor2 | cover | 0.688 | 0.982 | 1.000 | 1.000 | 1.000 |
| xor2 | pairs3 | 0.798 | 1.000 | 1.000 | 1.000 | 1.000 |
| dnf3 | upstart | 0.642 | 0.701 | 0.787 | 0.773 | 0.889 |
| dnf3 | cover | 0.673 | 0.838 | 0.983 | 1.000 | 1.000 |
| dnf3 | pairs3 | 0.656 | 0.743 | 1.000 | 1.000 | 1.000 |
| mux | upstart | 0.569 | 0.619 | 0.643 | 0.730 | 0.667 |
| mux | cover | 0.569 | 0.648 | 0.798 | 0.931 | 1.000 |
| mux | pairs3 | 0.583 | 0.666 | 0.764 | 0.724 | 0.847 |
| equal3 | upstart | 0.718 | 0.721 | 0.787 | 0.765 | 0.768 |
| equal3 | cover | 0.669 | 0.676 | 0.636 | 0.510 | 0.597 |
| equal3 | pairs3 | 0.713 | 0.765 | 0.824 | 0.903 | 1.000 |
| parity6 | upstart | 0.440 | 0.434 | 0.345 | 0.518 | 0.531 |
| parity6 | cover | 0.406 | 0.312 | 0.207 | 0.091 | 0.000 |
| parity6 | pairs3 | 0.417 | 0.414 | 0.404 | 0.302 | 0.733 |
| random | upstart | 0.482 | 0.467 | 0.455 | 0.506 | 0.387 |
| random | cover | 0.490 | 0.476 | 0.458 | 0.425 | 0.424 |
| random | pairs3 | 0.468 | 0.466 | 0.525 | 0.482 | 0.519 |

Runs exact on every unseen pattern at 16 streams (of 10; upstart / cover /
pairs3): majority 10 / 1 / 10, dnf3 6 / 10 / 10, mux 2 / 10 / 4, equal3
4 / 3 / 10.

Totals over the 8 tasks (mean unseen accuracy, midpoint):

| layout | 1 | 2 | 4 | 8 | 16 | helpers at 16 (mean) |
|---|---|---|---|---|---|---|
| upstart | 0.633 | 0.722 | 0.748 | 0.787 | 0.780 | 13.4 |
| cover | 0.626 | 0.693 | 0.724 | 0.695 | 0.695 | 10.4 |
| pairs3 | 0.642 | 0.716 | 0.784 | 0.781 | 0.887 | 1.7 |

Every layout recalls all training streams exactly without keys, and no run
needed a hidden neuron from TRAIN. Mean accuracies with sampled and midpoint
weights agree to within 0.003 (single runs differ by up to 0.19).

### 16.2 Findings

- **The covering search solves the OR-of-AND functions**: dnf3 is exact
  from 8 streams and mux from 16 (all 10 runs), where upstart stops at 0.89
  and 0.67. Each helper is a term.
- **It is the wrong bias elsewhere.** On majority it gets worse with more
  data (0.54 at 16 streams), and on parity6 it falls to 0: each pure helper
  covers a small patch around its seed, so unseen patterns next to a positive
  are called positive. Cover helps exactly when the function is a short OR of
  ANDs.
- **The second hidden layer is the best all-round layout** (0.887 mean at 16
  streams, against 0.780), with 1 or 2 layer-2 helpers. It fixes equal3
  (exact in all 10 runs at 16 streams, against 0.77): equal3 is a threshold
  of the pair ANDs, since each "x_i = x_{i+3}" is the sum of two of them.
  It is exact on dnf3 from 4 streams but only 0.85 on mux, whose terms have
  three literals.
- The pairs3 figure of 0.733 on parity6 at 16 streams rests on about 6
  unseen patterns per run and is noise; parity6 and the random function gain
  nothing from any layout, as they should not.
- The two new layouts are complementary: cover for short ORs of ANDs,
  pairs3 when the answer is a threshold of pairwise features. Neither alone
  beats upstart on every task.

## 17 State neurons that read helpers (`exp17_helpers_and_state.py`)

Next step 2: hidden neurons that carry state **and** generalise across input
vectors. Each task is a machine of experiment 13 (parity or a Tomita
language) run on a binary letter $l(x(t))$ of six input bits; the other
inputs are distractors. Training: 2 to 32 streams of 13 vectors, 10 seeds.
Testing: 200 random streams of 13 vectors, run with no key; a stream counts
if every output is right (midpoint weights). Layouts (`dnn_letters.py`):

- **merged**: experiment 13 (h = 2), state merging with the 64 raw input
  patterns as letters.
- **letters** (h = 3): the letter is chosen from a fixed library of 101
  candidates (each input, each AND of two literals, each XOR of two inputs,
  each majority of three) as the consistent one whose merged machine has the
  fewest states. Helpers compute it, and each state neuron is the AND of "in
  state q" and "the letter is l", read from the helpers, in one neuron.
- **true_letter** (h = 3): the same with the true letter given, so only the
  machine comes from the data.

The last task's letter, $(x_0 \wedge x_1) \vee (x_2 \wedge x_3)$, is not in
the library. Run on Lucas's machine, commit `6b3f02f`.

### 17.1 Results

Runs that found the true machine (of 10) / mean fraction of unseen streams
exact, by number of training streams.

| task | layout | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| parity/maj012 | merged | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.00 | 0 / 0.01 |
| parity/maj012 | letters | 8 / 0.80 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| parity/maj012 | true_letter | 8 / 0.85 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| tomita3/xor01 | merged | 0 / 0.03 | 0 / 0.02 | 0 / 0.01 | 0 / 0.01 | 0 / 0.01 |
| tomita3/xor01 | letters | 0 / 0.09 | 0 / 0.54 | 5 / 0.81 | 8 / 0.96 | 10 / 1.00 |
| tomita3/xor01 | true_letter | 0 / 0.34 | 0 / 0.59 | 5 / 0.81 | 8 / 0.96 | 10 / 1.00 |
| tomita4/and01 | merged | 0 / 0.01 | 0 / 0.01 | 0 / 0.01 | 0 / 0.00 | 0 / 0.01 |
| tomita4/and01 | letters | 0 / 0.14 | 0 / 0.53 | 5 / 0.85 | 9 / 0.98 | 10 / 1.00 |
| tomita4/and01 | true_letter | 0 / 0.45 | 0 / 0.66 | 5 / 0.85 | 9 / 0.98 | 10 / 1.00 |
| tomita5/x0 | merged | 0 / 0.01 | 0 / 0.01 | 0 / 0.01 | 0 / 0.01 | 0 / 0.01 |
| tomita5/x0 | letters | 0 / 0.02 | 3 / 0.41 | 6 / 0.73 | 10 / 1.00 | 10 / 1.00 |
| tomita5/x0 | true_letter | 0 / 0.08 | 3 / 0.44 | 6 / 0.73 | 10 / 1.00 | 10 / 1.00 |
| tomita7/maj345 | merged | 0 / 0.02 | 0 / 0.04 | 0 / 0.02 | 0 / 0.01 | 0 / 0.02 |
| tomita7/maj345 | letters | 0 / 0.12 | 0 / 0.15 | 4 / 0.77 | 10 / 1.00 | 10 / 1.00 |
| tomita7/maj345 | true_letter | 0 / 0.30 | 0 / 0.45 | 4 / 0.81 | 10 / 1.00 | 10 / 1.00 |
| tomita4/dnf2 | merged | 0 / 0.11 | 0 / 0.10 | 0 / 0.05 | 0 / 0.05 | 0 / 0.04 |
| tomita4/dnf2 | letters | 0 / 0.18 | 0 / 0.19 | 0 / 0.27 | 0 / 0.28 | – |
| tomita4/dnf2 | true_letter | 0 / 0.30 | 4 / 0.68 | 5 / 0.81 | 7 / 0.92 | 10 / 1.00 |

"letters": runs whose chosen letter is the true one (2 / 4 / 8 / 16 / 32
streams): parity/maj012 8 / 10 / 10 / 10 / 10, tomita3/xor01 1 / 9 / 10 /
10 / 10, tomita4/and01 0 / 7 / 10 / 10 / 10, tomita5/x0 1 / 8 / 10 / 10 /
10, tomita7/maj345 0 / 0 / 9 / 10 / 10, tomita4/dnf2 never.

Means over all tasks (unseen streams exact): merged 0.03 / 0.03 / 0.02 /
0.01 / 0.01, letters 0.23 / 0.47 / 0.74 / 0.93 / 1.00, true_letter 0.39 /
0.64 / 0.83 / 0.98 / 1.00 (the letters mean leaves out the 15 runs with no
consistent letter). No run needed a hidden neuron from TRAIN; recall from
keys is 1.000 in every run.

### 17.2 Findings

- **State neurons that read helpers generalise across input vectors.**
  Patterns never seen in training (mean of 64): 42 at 2 streams, 29 at 4,
  12 at 8, 3 at 16, 0 at 32; so about 94% of test streams contain an unseen
  pattern at 8 streams and 42% at 16. At 16 streams letters is exact on
  every unseen stream for parity, tomita5 and tomita7 (all 10 runs) and
  0.96–0.98 on tomita3 and tomita4. This answers open problem 1 for this
  class of tasks; at 32 streams every pattern has been seen, so that column
  tests only the machine.
- **Merging on raw patterns fails completely**, even at 32 streams when
  every pattern has been seen (at most 4% of unseen streams, 55% of single
  outputs): with 64 letters most (state, pattern) transitions are never
  observed.
- **Once the letter is right, nothing else is lost.** In all 193 runs where
  the search chose the true letter, the result is identical to
  true_letter. The remaining errors at 8 and 16 streams are the merging's,
  as in experiment 13. With 2 to 4 streams a wrong letter is often
  consistent and gives a smaller machine.
- **A letter outside the library is not found, and the search says so when
  it has enough data**: on tomita4/dnf2 a wrong but consistent letter is
  chosen up to 8 streams; at 16 streams 5 runs and at 32 all 10 find no
  consistent letter (the 15 runs scored "–"). With the true letter given,
  the same task is exact from 32 streams.
- Limits: the letter is one bit and comes from a fixed library; the helpers
  are built on its full truth table, not from data. Learning the letter from
  data (for example the cover or pairs3 helpers of experiment 16) is the
  next step. Depth is h = 3.

## 18 Which streams may leave a bounded buffer (`exp18_buffer_policy.py`)

Next step 3. Experiment 15 rebuilt the network from a buffer at every sleep;
with the 16 most recent streams, 3 of 60 runs (all tomita4) ended with a
wrong machine. Here the buffer is cut to its cap just before every sleep by
one of four policies (`dnn_sleep.evict`):

- **all**: no cap (experiment 15's "sleep").
- **recent**: the most recent streams (experiment 15's "sleep16" for cap 16).
- **random**: a random subset.
- **witness**: streams are dropped oldest first, but only while the machine
  merged from the rest equals the machine merged from the whole buffer. The
  streams that cannot go are that machine's witnesses; if there are more
  witnesses than the cap, the oldest go.

The six tasks of experiment 15, 32 streams arriving 4 per round, a sleep
after every second round (4 sleeps), caps 16 and 8, 20 seeds (seeds 0–9 are
experiment 15's data). Wake rounds are left out, since a sleep rebuilds from
the buffer alone. Run on Lucas's machine, commit `6b3f02f`. Check: recent 16
on seeds 0–9 finds the true machine at the end in 57 of 60 runs, as
experiment 15's sleep16.

### 18.1 Results

Runs with the true machine after each sleep (of 120: 6 tasks × 20 seeds),
and means at the end.

| policy | cap | round 2 | round 4 | round 6 | round 8 | unseen streams exact | recall of all streams seen, no key |
|---|---|---|---|---|---|---|---|
| all | – | 86 | 115 | 118 | 119 | 0.999 | 1.000 |
| recent | 16 | 86 | 115 | 115 | 112 | 0.983 | 0.993 |
| random | 16 | 86 | 115 | 113 | 112 | 0.984 | 0.994 |
| witness | 16 | 86 | 115 | 118 | 119 | 0.999 | 0.999 |
| recent | 8 | 86 | 84 | 93 | 79 | 0.875 | 0.901 |
| random | 8 | 86 | 94 | 89 | 89 | 0.894 | 0.914 |
| witness | 8 | 86 | 115 | 118 | 119 | 0.999 | 0.995 |

At the end, runs with the true machine by task (of 20):

| task | all | recent 16 | random 16 | witness 16 | recent 8 | random 8 | witness 8 |
|---|---|---|---|---|---|---|---|
| parity | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| adder | 20 | 20 | 20 | 20 | 19 | 20 | 20 |
| tomita3 | 20 | 20 | 18 | 20 | 10 | 14 | 20 |
| tomita4 | 19 | 14 | 14 | 19 | 6 | 9 | 19 |
| tomita5 | 20 | 19 | 20 | 20 | 12 | 13 | 20 |
| tomita7 | 20 | 19 | 20 | 20 | 12 | 13 | 20 |

No run needed hidden neurons. No witness run ever had more witnesses than
its cap.

### 18.2 Findings

- **Keeping the witnesses loses nothing.** At both caps, in every one of the
  960 sleeps, the witness buffer gives the same machine and the same
  unseen-stream score as the unbounded buffer. The one failure (tomita4,
  seed 2) fails with every policy, including no cap.
- **Eight streams are enough** when they are the right eight: at cap 8 the
  witness buffer keeps 119 of 120 runs right, against 79 for recent and 89
  for random. A machine with 2 to 5 states is fixed by a few streams.
- **Recent and random lose the hard Tomita tasks**, worst on tomita4 (6 and
  9 of 20 at cap 8), and the number right can fall from one sleep to the
  next (recent 8: 93 at round 6, 79 at round 8).
- Recall without key dips slightly at cap 8 even for witness (0.995): in
  those runs (adder, tomita3, tomita4) the machine is right and every unseen
  stream is exact with midpoint weights; the dip comes from sampled weights,
  since an output trained on 8 streams has wider boxes.
- Cost: the witness check merges the buffer once per candidate stream,
  up to 24 merges per sleep here (cap + 8 streams). Cheap at these sizes; it grows
  with the buffer.
- Open: the witness test compares machines, so it applies only when the
  sleep is state merging. A policy for helpers (experiments 14, 16) would
  keep the patterns whose removal changes a helper.

## 19 Small black-and-white images (`exp19_small_images.py`)

Next step 4, ladder rung 2. The input $x(t)$ is a 4 × 4 image (16 inputs)
and the output at $t + 2$ is a fixed property of it: hline (some row all
black), vline (some column), block (some 2 × 2 square), ink8 (at least 8
black pixels), top (more black in the top half than in the bottom half),
mirror (left-right symmetric). Pixels are black with a random density per
image; for hline, vline, block and mirror half the images get the shape
planted, so that both answers are common. Layouts as in experiments 14 and
16 (h = 2, the output reads only its helpers): lookup (one indicator per
positive image seen: the constant-answer reference), upstart and cover; both
of the last two switch to indicators after 20 helpers. Training: 2 to 32
streams of 13 images (22 to 333 distinct images), 10 seeds. Testing: 200
streams of 13 images. "Base rate" is the accuracy of the better constant
answer. Run on Lucas's machine, commit `6b3f02f`.

### 19.1 Results

Mean accuracy on unseen images (midpoint weights).

| task | base rate | layout | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| hline | 0.536 | lookup | 0.464 | 0.466 | 0.465 | 0.467 | 0.470 |
| hline | 0.536 | upstart | 0.640 | 0.648 | 0.693 | 0.565 | 0.470 |
| hline | 0.536 | cover | 0.672 | 0.737 | 0.784 | 0.865 | 0.875 |
| vline | 0.536 | lookup | 0.464 | 0.465 | 0.466 | 0.466 | 0.470 |
| vline | 0.536 | upstart | 0.648 | 0.658 | 0.708 | 0.607 | 0.470 |
| vline | 0.536 | cover | 0.660 | 0.749 | 0.804 | 0.865 | 0.861 |
| block | 0.551 | lookup | 0.448 | 0.448 | 0.447 | 0.448 | 0.444 |
| block | 0.551 | upstart | 0.682 | 0.680 | 0.736 | 0.664 | 0.447 |
| block | 0.551 | cover | 0.704 | 0.741 | 0.805 | 0.824 | 0.791 |
| ink8 | 0.542 | lookup | 0.458 | 0.458 | 0.457 | 0.457 | 0.457 |
| ink8 | 0.542 | upstart | 0.817 | 0.855 | 0.927 | 0.987 | 1.000 |
| ink8 | 0.542 | cover | 0.775 | 0.826 | 0.857 | 0.897 | 0.883 |
| top | 0.605 | lookup | 0.604 | 0.605 | 0.604 | 0.605 | 0.604 |
| top | 0.605 | upstart | 0.737 | 0.825 | 0.945 | 1.000 | 1.000 |
| top | 0.605 | cover | 0.661 | 0.745 | 0.784 | 0.805 | 0.785 |
| mirror | 0.512 | lookup | 0.520 | 0.551 | 0.590 | 0.645 | 0.716 |
| mirror | 0.512 | upstart | 0.521 | 0.653 | 0.792 | 0.811 | 0.856 |
| mirror | 0.512 | cover | 0.541 | 0.563 | 0.599 | 0.689 | 0.740 |

Means over the 6 tasks (unseen accuracy, midpoint), with mean helpers:

| layout | 2 | 4 | 8 | 16 | 32 | helpers at 8 / 16 / 32 |
|---|---|---|---|---|---|---|
| lookup | 0.493 | 0.499 | 0.505 | 0.515 | 0.527 | 44 / 87 / 167 |
| upstart | 0.674 | 0.720 | 0.800 | 0.772 | 0.707 | 3.5 / 41.5 / 109 |
| cover | 0.669 | 0.727 | 0.772 | 0.824 | 0.822 | 12.9 / 24.4 / 52.4 |

Every run recalls all training streams exactly without keys; no run needed a
hidden neuron from TRAIN. Upstart passed 20 helpers (and switched to
indicators) in 22 of 60 runs at 16 streams and 35 at 32; cover in 44 and
all 60.

### 19.2 Findings

- **Threshold properties are learned exactly.**
  ink8 and top need one upstart helper (the best perceptron) and are exact on
  unseen images in all 10 runs at 32 streams (top already at 16).
- **Shape properties are ORs of ANDs, and cover is the method for them**:
  hline and vline reach 0.87, block 0.82, against base rates 0.54–0.55.
  Each helper is a row, a column or a square, or part of one.
- **Upstart collapses on shapes as data grows**: 0.69–0.74 at 8 streams, the
  constant answer's level at 32. Its best perceptron is not exact, it passes
  the 20-helper limit, and the remaining errors become indicators (about 200
  of them at 32 streams), which is lookup.
- **Cover stops improving at 16 streams** for the same reason: it also
  reaches the limit, and its later helpers are indicators. A larger limit,
  or the second hidden layer of experiment 16, is the next thing to try.
- Mirror (a test of equality between pixel pairs, like equal3) is hard for
  all. Lookup's rise to 0.72 is not learning: mirrored images repeat, so the
  unseen ones are increasingly negative and the constant answer 0 gains.
- No single layout wins every task. Rung 2 is reached for threshold
  properties; shapes need helpers for groups of pixels beyond 20 at a time.

## 20 Learning the letter from the data (`exp20_learned_letter.py`)

Next step 1 (follow-up to experiment 17). Experiment 17 chose the letter
of a finite-state task from a fixed library of 101 functions of the 6 input
bits; its tomita4/dnf2 letter, (x0 ∧ x1) ∨ (x2 ∧ x3), is not in the library
and was never found. Two ways to learn the letter, both h = 3:

- **composed**: the library grown by one step, the OR and the AND of every
  pair of library letters (about 10,000 letters). The consistent letter of
  either level whose merged machine has the fewest states wins (ties: the
  library letter). Helpers as in experiment 17.
- **search**: no library. The smallest machine and a letter label for every
  training pattern are found together by backtracking, so that every
  training output is reproduced (at most 6 states, 300,000 search nodes per
  size; the library only orders the free label choices). The labels the
  machine forces are fitted by the covering search of experiment 16, so
  unseen patterns get the helpers' answer.

"letters" and "true_letter" are experiment 17's layouts. Tasks: experiment
17's six, plus tomita7/or3 (x0 ∨ x1 ∨ x2, also outside the library).
Training: 2 to 32 streams of 13 vectors, 10 seeds; testing: 200 streams, no
key, midpoint weights. Run on Lucas's machine, commit `64f3838` (with the
fix of PR #23). The 600 letters and true_letter rows equal experiment 17's.

### 20.1 Results

Runs with the true machine (of 10) / mean fraction of unseen streams exact,
by number of training streams ("–": no letter found in any run).

| task | layout | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| parity/maj012 | letters | 8 / 0.80 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| parity/maj012 | true_letter | 8 / 0.85 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| parity/maj012 | composed | 8 / 0.83 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| parity/maj012 | search | 0 / 0.04 | 9 / 0.90 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| tomita3/xor01 | letters | 0 / 0.09 | 0 / 0.54 | 5 / 0.81 | 8 / 0.96 | 10 / 1.00 |
| tomita3/xor01 | true_letter | 0 / 0.34 | 0 / 0.59 | 5 / 0.81 | 8 / 0.96 | 10 / 1.00 |
| tomita3/xor01 | composed | 0 / 0.09 | 0 / 0.50 | 5 / 0.81 | 8 / 0.96 | 10 / 1.00 |
| tomita3/xor01 | search | 0 / 0.04 | 0 / 0.03 | 0 / 0.32 | 2 / 1.00 | – |
| tomita4/and01 | letters | 0 / 0.14 | 0 / 0.53 | 5 / 0.85 | 9 / 0.98 | 10 / 1.00 |
| tomita4/and01 | true_letter | 0 / 0.45 | 0 / 0.66 | 5 / 0.85 | 9 / 0.98 | 10 / 1.00 |
| tomita4/and01 | composed | 0 / 0.12 | 0 / 0.35 | 5 / 0.84 | 9 / 0.98 | 10 / 1.00 |
| tomita4/and01 | search | 0 / 0.02 | 0 / 0.05 | 1 / 0.27 | – | – |
| tomita5/x0 | letters | 0 / 0.02 | 3 / 0.41 | 6 / 0.73 | 10 / 1.00 | 10 / 1.00 |
| tomita5/x0 | true_letter | 0 / 0.08 | 3 / 0.44 | 6 / 0.73 | 10 / 1.00 | 10 / 1.00 |
| tomita5/x0 | composed | 0 / 0.01 | 4 / 0.48 | 8 / 0.87 | 10 / 1.00 | 10 / 1.00 |
| tomita5/x0 | search | 0 / 0.01 | 0 / 0.02 | 1 / 0.51 | – | – |
| tomita7/maj345 | letters | 0 / 0.12 | 0 / 0.15 | 4 / 0.77 | 10 / 1.00 | 10 / 1.00 |
| tomita7/maj345 | true_letter | 0 / 0.30 | 0 / 0.45 | 4 / 0.81 | 10 / 1.00 | 10 / 1.00 |
| tomita7/maj345 | composed | 0 / 0.11 | 0 / 0.15 | 3 / 0.68 | 10 / 1.00 | 10 / 1.00 |
| tomita7/maj345 | search | 0 / 0.04 | 0 / 0.09 | – | – | – |
| tomita4/dnf2 | letters | 0 / 0.18 | 0 / 0.19 | 0 / 0.27 | 0 / 0.28 | – |
| tomita4/dnf2 | true_letter | 0 / 0.30 | 4 / 0.68 | 5 / 0.81 | 7 / 0.92 | 10 / 1.00 |
| tomita4/dnf2 | composed | 0 / 0.18 | 0 / 0.21 | 3 / 0.64 | 7 / 0.92 | 10 / 1.00 |
| tomita4/dnf2 | search | 0 / 0.16 | 0 / 0.14 | 0 / 0.12 | – | – |
| tomita7/or3 | letters | 0 / 0.52 | 0 / 0.48 | 0 / 0.53 | 0 / 0.61 | 0 / 0.57 |
| tomita7/or3 | true_letter | 0 / 0.75 | 0 / 0.79 | 0 / 0.75 | 0 / 0.88 | 4 / 0.93 |
| tomita7/or3 | composed | 0 / 0.53 | 0 / 0.45 | 0 / 0.64 | 0 / 0.79 | 3 / 0.94 |
| tomita7/or3 | search | 0 / 0.55 | 0 / 0.55 | 0 / 0.77 | – | – |

Runs whose letter is the true one on all 64 patterns: letters / composed /
search.

| task | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| parity/maj012 | 8 / 8 / 0 | 10 / 10 / 9 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| tomita3/xor01 | 1 / 1 / 0 | 9 / 8 / 0 | 10 / 10 / 1 | 10 / 10 / 2 | 10 / 10 / 0 |
| tomita4/and01 | 0 / 0 / 0 | 7 / 2 / 0 | 10 / 9 / 1 | 10 / 10 / 0 | 10 / 10 / 0 |
| tomita5/x0 | 1 / 1 / 0 | 8 / 7 / 0 | 10 / 10 / 1 | 10 / 10 / 0 | 10 / 10 / 0 |
| tomita7/maj345 | 0 / 0 / 0 | 0 / 0 / 0 | 9 / 7 / 0 | 10 / 10 / 0 | 10 / 10 / 0 |
| tomita4/dnf2 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 6 / 0 | 0 / 10 / 0 | 0 / 10 / 0 |
| tomita7/or3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 3 / 0 | 0 / 7 / 0 | 0 / 10 / 0 |

No letter found: letters 15 runs, composed 0, search 159 (by streams: 0,
7, 34, 58, 60). Composed chose a two-letter combination in 115 of 350 runs;
on the five library tasks only at 2 to 8 streams (51 runs). Where search
found a machine, it had fewer states than the true machine in 131 of 191
runs; its letter agreed with the true one on 0.71 of the patterns on
average. Every run recalls its training streams from keys; no run needed a
hidden neuron from TRAIN. Longest build: 8.5 s (composed).

### 20.2 Findings

- **The composed library finds both letters outside the library.** dnf2's
  letter is found in 6 of 10 runs at 8 streams and all 10 from 16, and
  composed then matches true_letter exactly (7 / 0.92 at 16, 10 / 1.00 at
  32) where experiment 17's library gets 0.28 or nothing. For or3 the letter
  is found in 3, 7 and 10 runs at 8, 16 and 32 streams, and composed matches
  true_letter at 32 (0.94 against 0.93). What still limits or3 is merging:
  its letter is 0 only one time in 8, so even the true letter gives the true
  machine in only 4 of 10 runs.
- **The bigger library costs a little with few streams.** On the five
  library tasks composed sometimes prefers a two-letter combination with a
  smaller machine at 2 to 8 streams (and01 at 4: 0.35 against 0.53; maj345
  at 8: 0.68 against 0.77; x0 at 8 gains: 0.87 against 0.73). From 16
  streams it equals experiment 17 on all five.
- **Learning the letter with no prior on its form fails.** The joint search
  gives each training pattern its own free label, so a small wrong machine
  plus a letter that copies the outputs fits the data: its machines are
  usually smaller than the true one, and only parity/maj012 (2 states) is
  learned. With more data a consistent labelling exists only at the true
  size, and the backtracking search runs out of budget there (no letter in
  58 of 70 runs at 16 streams, all 70 at 32). A SAT solver might reach the
  true size, but the overfitting at small sizes would remain.
- So the letter must come from a class of simple functions (the library,
  grown by combination) chosen by the data, not from free labels. This is
  the same lesson as experiments 9 to 15: generalisation comes from neurons
  computed from the inputs, not from labels copied from the outputs.

## 21 More helpers for shapes (`exp21_more_helpers_images.py`)

Next step 2 (follow-up to experiment 19). In experiment 19 the helper
searches stopped at 20 helpers and then added one indicator per remaining
positive image; upstart fell to the constant answer on lines and blocks, and
cover stopped improving at 16 streams. Same tasks, images, seeds and scoring
as experiment 19. Layouts:

- **upstart, cover**: experiment 19's (limit 20).
- **upstart200, cover200**: the same searches with a limit of 200 (never
  reached: at most 97 and 57 helpers), so in effect no limit.
- **nbr3_upstart, nbr3_cover**: a second hidden layer (h = 3, as
  experiment 16's pairs3). Layer 1 is the AND of two literals for every pair
  of neighbouring pixels (24 pairs × 4 = 96 fixed neurons); layer 2 is built
  over the inputs and layer 1 by upstart or cover.
- **pairs3_cover**: the same with every pair of pixels in layer 1 (120 pairs
  × 4 = 480 fixed neurons), layer 2 by cover.

Run on Lucas's machine, commit `64f3838`, 35 minutes. The 600 upstart and
cover rows equal experiment 19's.

### 21.1 Results

Mean accuracy on unseen images (midpoint weights).

| task | base rate | layout | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| hline | 0.536 | upstart | 0.640 | 0.648 | 0.693 | 0.565 | 0.470 |
| hline | 0.536 | cover | 0.672 | 0.737 | 0.784 | 0.865 | 0.875 |
| hline | 0.536 | upstart200 | 0.640 | 0.648 | 0.693 | 0.739 | 0.763 |
| hline | 0.536 | cover200 | 0.672 | 0.737 | 0.784 | 0.873 | 0.917 |
| hline | 0.536 | nbr3_upstart | 0.641 | 0.728 | 0.821 | 0.846 | 0.860 |
| hline | 0.536 | nbr3_cover | 0.691 | 0.765 | 0.832 | 0.901 | 0.958 |
| hline | 0.536 | pairs3_cover | 0.645 | 0.731 | 0.786 | 0.877 | 0.940 |
| vline | 0.536 | upstart | 0.648 | 0.658 | 0.708 | 0.607 | 0.470 |
| vline | 0.536 | cover | 0.660 | 0.749 | 0.804 | 0.865 | 0.861 |
| vline | 0.536 | upstart200 | 0.648 | 0.658 | 0.708 | 0.739 | 0.764 |
| vline | 0.536 | cover200 | 0.660 | 0.749 | 0.804 | 0.876 | 0.904 |
| vline | 0.536 | nbr3_upstart | 0.674 | 0.768 | 0.809 | 0.843 | 0.838 |
| vline | 0.536 | nbr3_cover | 0.703 | 0.797 | 0.817 | 0.906 | 0.957 |
| vline | 0.536 | pairs3_cover | 0.639 | 0.739 | 0.817 | 0.901 | 0.927 |
| block | 0.551 | upstart | 0.682 | 0.680 | 0.736 | 0.664 | 0.447 |
| block | 0.551 | cover | 0.704 | 0.741 | 0.805 | 0.824 | 0.791 |
| block | 0.551 | upstart200 | 0.682 | 0.680 | 0.736 | 0.761 | 0.794 |
| block | 0.551 | cover200 | 0.704 | 0.741 | 0.805 | 0.848 | 0.882 |
| block | 0.551 | nbr3_upstart | 0.702 | 0.776 | 0.811 | 0.854 | 0.879 |
| block | 0.551 | nbr3_cover | 0.709 | 0.802 | 0.842 | 0.901 | 0.943 |
| block | 0.551 | pairs3_cover | 0.687 | 0.754 | 0.791 | 0.859 | 0.886 |
| ink8 | 0.542 | upstart | 0.817 | 0.855 | 0.927 | 0.987 | 1.000 |
| ink8 | 0.542 | cover | 0.775 | 0.826 | 0.857 | 0.897 | 0.883 |
| ink8 | 0.542 | upstart200 | 0.817 | 0.855 | 0.927 | 0.987 | 1.000 |
| ink8 | 0.542 | cover200 | 0.775 | 0.826 | 0.857 | 0.898 | 0.909 |
| ink8 | 0.542 | nbr3_upstart | 0.786 | 0.836 | 0.881 | 0.921 | 0.958 |
| ink8 | 0.542 | nbr3_cover | 0.721 | 0.780 | 0.832 | 0.868 | 0.888 |
| ink8 | 0.542 | pairs3_cover | 0.717 | 0.780 | 0.813 | 0.880 | 0.902 |
| top | 0.605 | upstart | 0.737 | 0.825 | 0.945 | 1.000 | 1.000 |
| top | 0.605 | cover | 0.661 | 0.745 | 0.784 | 0.805 | 0.785 |
| top | 0.605 | upstart200 | 0.737 | 0.825 | 0.945 | 1.000 | 1.000 |
| top | 0.605 | cover200 | 0.661 | 0.745 | 0.784 | 0.820 | 0.845 |
| top | 0.605 | nbr3_upstart | 0.680 | 0.759 | 0.839 | 0.910 | 1.000 |
| top | 0.605 | nbr3_cover | 0.617 | 0.693 | 0.739 | 0.771 | 0.805 |
| top | 0.605 | pairs3_cover | 0.648 | 0.709 | 0.772 | 0.806 | 0.836 |
| mirror | 0.513 | upstart | 0.521 | 0.653 | 0.792 | 0.811 | 0.856 |
| mirror | 0.513 | cover | 0.541 | 0.563 | 0.599 | 0.689 | 0.740 |
| mirror | 0.513 | upstart200 | 0.521 | 0.653 | 0.792 | 0.824 | 0.870 |
| mirror | 0.513 | cover200 | 0.541 | 0.563 | 0.599 | 0.685 | 0.719 |
| mirror | 0.513 | nbr3_upstart | 0.644 | 0.757 | 0.822 | 0.906 | 0.941 |
| mirror | 0.513 | nbr3_cover | 0.592 | 0.605 | 0.669 | 0.714 | 0.767 |
| mirror | 0.513 | pairs3_cover | 0.566 | 0.564 | 0.629 | 0.697 | 0.748 |

Means over the 6 tasks (unseen accuracy, midpoint), with mean helpers at 32
streams (layer 2 at h = 3):

| layout | 2 | 4 | 8 | 16 | 32 | helpers at 32 |
|---|---|---|---|---|---|---|
| upstart | 0.674 | 0.720 | 0.800 | 0.772 | 0.707 | 109 |
| cover | 0.669 | 0.727 | 0.772 | 0.824 | 0.822 | 52.4 |
| upstart200 | 0.674 | 0.720 | 0.800 | 0.842 | 0.865 | 38.5 |
| cover200 | 0.669 | 0.727 | 0.772 | 0.833 | 0.863 | 38.7 |
| nbr3_upstart | 0.688 | 0.771 | 0.831 | 0.880 | **0.913** | 1.9 |
| nbr3_cover | 0.672 | 0.740 | 0.788 | 0.843 | 0.886 | 35.0 |
| pairs3_cover | 0.651 | 0.713 | 0.768 | 0.837 | 0.873 | 34.2 |

Every run recalls all training streams exactly without keys; no run needed a
hidden neuron from TRAIN. Longest build: 43 s (pairs3_cover); the other
layouts under 4 s.

### 21.2 Findings

- **The limit was the problem.** Without it, upstart no longer collapses
  (lines and blocks 0.76–0.79 at 32 streams, against 0.45–0.47) and cover
  keeps improving (lines 0.90–0.92, blocks 0.88). Both now use fewer helpers
  than with the limit, since no indicators are added.
- **Neighbouring-pixel pairs plus cover is best on shapes**: lines 0.96,
  blocks 0.94 at 32 streams, the best (or tied best) on these three tasks
  at every size.
  All pairs (480 neurons) is a little worse and ten times slower: the extra
  pairs are mostly noise for local shapes.
- **Neighbour pairs plus upstart is the best layout overall** (0.913 mean
  at 32 streams) with 1 to 5 helpers. It is exact on top, 0.96 on ink8, and
  0.94 on mirror, the task that was hard for all before (0.86 at best). Its
  helpers are best perceptrons over pixels and pair ANDs: equal neighbouring
  pixels are pair ANDs, so a single threshold over them goes a long way.
- **Still no layout wins every task**: plain upstart is exact on the
  threshold tasks; nbr3_cover wins the shapes; nbr3_upstart wins mirror.
  Experiment 22 tests choosing between them from the training data.
- With the right layout for each task, every task of rung 2 is now at 0.94
  or better at 32 streams (ink8 and top exact).

## 22 Choosing the layout from the training data (`exp22_choose_layout.py`)

Next step 3. Different layouts win different tasks (experiments 16, 19,
21). Two rules choose the layout from the training streams alone:

- **heldout**: the streams are split into k = min(streams, 4) folds; for each
  fold every layout is built and trained on the other streams and scored on
  the held-out ones (outputs whose input pattern the other streams lack,
  midpoint weights). Best pooled score wins (ties: fewer helpers). With 1
  stream it falls back to the next rule.
- **fewest**: the layout with the fewest helpers built from the data (at
  h = 3, layer 2 only).

The chosen layout, built on all the training streams, is scored on the test
streams and compared with each fixed layout and with **oracle** (the best
layout of each run on the test streams, which a learner cannot know).
Tasks: experiment 16's eight Boolean functions of 6 inputs (1 to 16 streams;
upstart, cover, pairs3) and experiment 19's six image tasks (2 to 32
streams; upstart, cover and experiment 21's nbr3_cover). 10 seeds. Run on
Lucas's machine, commit `64f3838`, 11 minutes. Every row's test columns equal
experiment 16's (1200 rows), 19's (600) or 21's (900).

### 22.1 Results

Mean accuracy on unseen patterns (midpoint), over all tasks and seeds.

Boolean functions:

| layout or rule | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| upstart | 0.633 | 0.722 | 0.748 | 0.787 | 0.780 |
| cover | 0.626 | 0.693 | 0.724 | 0.695 | 0.695 |
| pairs3 | 0.642 | 0.716 | 0.784 | 0.781 | 0.887 |
| heldout | 0.642 | 0.720 | 0.791 | 0.824 | 0.884 |
| fewest | 0.642 | 0.734 | 0.806 | 0.810 | 0.887 |
| oracle | 0.685 | 0.766 | 0.824 | 0.868 | 0.929 |

Images:

| layout or rule | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| upstart | 0.674 | 0.720 | 0.800 | 0.772 | 0.707 |
| cover | 0.669 | 0.727 | 0.772 | 0.824 | 0.822 |
| nbr3_cover | 0.672 | 0.740 | 0.788 | 0.843 | 0.886 |
| heldout | 0.681 | 0.754 | 0.854 | 0.918 | 0.954 |
| fewest | 0.674 | 0.720 | 0.802 | 0.893 | 0.955 |
| oracle | 0.717 | 0.790 | 0.864 | 0.921 | 0.956 |

Layout chosen by the held-out rule (runs per task over all sizes, of 50) and
runs where it is the oracle's layout:

| task | upstart | cover | pairs3 / nbr3_cover | same as oracle |
|---|---|---|---|---|
| majority | 43 | 3 | 4 | 37 |
| and3 | 44 | 2 | 4 | 42 |
| xor2 | 8 | 1 | 41 | 46 |
| dnf3 | 11 | 11 | 28 | 34 |
| mux | 14 | 24 | 12 | 28 |
| equal3 | 20 | 2 | 28 | 22 |
| parity6 | 14 | 7 | 29 | 21 |
| random | 19 | 11 | 20 | 18 |
| hline | 1 | 8 | 41 | 38 |
| vline | 6 | 6 | 38 | 34 |
| block | 4 | 6 | 40 | 41 |
| ink8 | 34 | 7 | 9 | 34 |
| top | 43 | 4 | 3 | 43 |
| mirror | 33 | 4 | 13 | 33 |

Head to head (runs where held-out's choice scores higher minus runs where it
scores lower than fewest's): Boolean −11, −6, +2, −1 at 2, 4, 8, 16 streams;
images +15, +16, +28, +9, 0 at 2 to 32 streams. Longest held-out
evaluation: 9.3 s.

### 22.2 Findings

- **On images, choosing by held-out streams is nearly as good as knowing
  the answer**: 0.918 and 0.954 at 16 and 32 streams against the oracle's
  0.921 and 0.956, and 0.05–0.08 above the best fixed layout from 8 streams.
  At 32 streams it matches the best layout on every task (lines and blocks
  nbr3_cover, ink8 and top upstart) and beats it on mirror (0.87).
- **On the Boolean functions it is less clear-cut**: held-out beats every
  fixed layout at 4 and 8 streams (0.82 against 0.79 at 8) but is about
  equal to pairs3 at 16. Fewest helpers does as well or better with 2 to 4
  streams, where folds of one or two streams are too small to score.
- **Each rule has a blind spot at 16 Boolean streams**: fewest helpers
  picks wrongly on mux (0.85 against 1.00); held-out picks wrongly on equal3
  (0.96 against 1.00) and on the random function (0.38 against 0.52), where
  there is nothing to learn and only a few unseen patterns are scored.
- **Fewest helpers is a good cheap rule once data are plentiful** (images
  0.955 at 32 streams) but poor with few streams (images 0.72 at 4, the
  upstart level, since upstart starts with one helper).
- The gap to the oracle is largest with few streams (0.03–0.05 at 1 to 4
  streams), where the held-out folds are smallest.
- Not tried: experiment 21's best overall image layout, nbr3_upstart, was
  not among the choices (experiment 22 was written before experiment 21's
  results).

## 23 A tiny grid game (`exp23_grid_game.py`)

Next step 4, ladder rung 3. The screen is a 4 × 4 grid in two planes (32
inputs): a player and a monster, one black pixel each, both walking at
random (up, down, left, right; a move into a wall stays put). An episode is
25 screens. The output at $t + 3$ is a game signal that needs memory:
key_door (on the door, bottom-left quadrant, after the key, top-right),
lap (enters the right half after the left), count2 (every second entry into
the top row), monster_key (monster in the left half after the player took
the key), parity_top (odd number of steps in the top half), both_bottom
(game still on: player and monster never in the bottom row together). Each
is a machine of 2 to 4 states reading a letter of 1 or 2 bits, each bit a
region of one plane.

Layout (h = 3): each letter bit is one fixed neuron (the OR of the region's
pixels); state neurons read the bits. **bits** chooses the letter from 32
regions (rows, columns, halves, quadrants of each plane): every set of 1 or
2 regions (528 letters) with consistent letter streams is merged, fewest
states win. **true_bits** is given the true regions. Raw state merging is
not possible (2^32 letters). Data arrive as 32 episodes, 4 per round; after
every second round the network is rebuilt from the buffer (a sleep). Buffer
policies: all, recent (16 or 8), witness (16 or 8: choose the letter on the
whole buffer, keep the streams its machine needs, rebuild from those). 10
seeds; 200 test episodes, no key, midpoint weights. Run on Lucas's machine,
commit `64f3838`, 3 minutes.

### 23.1 Results

After each sleep: runs with the true machine (of 10) / mean fraction of
unseen episodes exact. "True machine" is judged on every sequence of
screens, so a different letter can give it.

| task | layout | buffer | 8 streams | 16 streams | 24 streams | 32 streams |
|---|---|---|---|---|---|---|
| key_door | bits | all | 3 / 0.78 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| key_door | true_bits | all | 8 / 0.97 | 9 / 0.98 | 10 / 1.00 | 10 / 1.00 |
| key_door | bits | recent 16 | 3 / 0.78 | 10 / 1.00 | 8 / 0.95 | 8 / 0.96 |
| key_door | bits | witness 16 | 3 / 0.78 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| key_door | bits | recent 8 | 3 / 0.78 | 5 / 0.83 | 2 / 0.79 | 7 / 0.89 |
| key_door | bits | witness 8 | 3 / 0.78 | 9 / 0.98 | 6 / 0.91 | 8 / 0.95 |
| lap | all layouts and buffers | | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| count2 | bits | all | 2 / 0.76 | 7 / 0.93 | 8 / 0.97 | 8 / 0.97 |
| count2 | true_bits | all | 6 / 0.90 | 8 / 0.98 | 9 / 0.99 | 9 / 0.99 |
| count2 | bits | recent 16 | 2 / 0.76 | 7 / 0.93 | 6 / 0.93 | 4 / 0.86 |
| count2 | bits | witness 16 | 2 / 0.76 | 7 / 0.93 | 7 / 0.95 | 5 / 0.89 |
| count2 | bits | recent 8 | 2 / 0.76 | 2 / 0.77 | 1 / 0.76 | 2 / 0.77 |
| count2 | bits | witness 8 | 2 / 0.76 | 3 / 0.81 | 2 / 0.73 | 2 / 0.77 |
| monster_key | bits | all | 3 / 0.82 | 5 / 0.94 | 9 / 0.99 | 10 / 1.00 |
| monster_key | true_bits | all | 3 / 0.90 | 4 / 0.93 | 7 / 0.96 | 8 / 0.98 |
| monster_key | bits | recent 16 | 3 / 0.82 | 5 / 0.94 | 6 / 0.95 | 6 / 0.95 |
| monster_key | bits | witness 16 | 3 / 0.82 | 5 / 0.94 | 9 / 0.99 | 10 / 1.00 |
| monster_key | bits | recent 8 | 3 / 0.82 | 1 / 0.67 | 4 / 0.83 | 1 / 0.76 |
| monster_key | bits | witness 8 | 3 / 0.82 | 5 / 0.90 | 9 / 0.99 | 7 / 0.94 |
| parity_top | all layouts and buffers | | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| both_bottom | bits | all | 5 / 0.77 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | true_bits | all | 5 / 0.88 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | bits | recent 16 | 5 / 0.77 | 8 / 0.98 | 7 / 0.93 | 9 / 0.97 |
| both_bottom | bits | witness 16 | 5 / 0.77 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | bits | recent 8 | 5 / 0.77 | 4 / 0.86 | 5 / 0.85 | 6 / 0.85 |
| both_bottom | bits | witness 8 | 5 / 0.77 | 7 / 0.91 | 9 / 0.98 | 10 / 1.00 |

At the end (of 60 runs): letter is the true regions / true machine / mean
unseen exact / mean recall without key of every episode seen:

| layout | buffer | true regions | true machine | unseen exact | recall |
|---|---|---|---|---|---|
| bits | all | 37 | 58 | 0.995 | 1.000 |
| true_bits | all | 60 | 57 | 0.994 | 1.000 |
| bits | recent 16 | 35 | 47 | 0.959 | 0.976 |
| bits | witness 16 | 33 | 55 | 0.981 | 0.992 |
| bits | recent 8 | 35 | 36 | 0.879 | 0.907 |
| bits | witness 8 | 30 | 47 | 0.943 | 0.966 |

No sleep needed a hidden neuron from TRAIN; every sleep found a consistent
letter (71 of 528 on average at the end). The witnesses always fitted in the
cap. Longest build: 3.8 s.

### 23.2 Findings

- **Rung 3 works for game signals that need memory.** With the whole
  buffer, the learned letter gives the true machine in 58 of 60 runs and
  0.995 of unseen episodes exact; screen helpers, state neurons and merging
  together, with nothing forgotten.
- **The learned letter is often a different but equivalent one.** For
  key_door it is always "top half + left half", whose 4 values include the
  key and door quadrants exactly. Only 37 of 60 letters are the true
  regions, but 58 machines are right.
- **Merging, not the letter, limits the hard cases**, as in experiment 13:
  given the true regions, merging still errs in 3 runs. In monster_key the
  two failures with the true regions are the seeds where "bits" chose the
  opposite half (monster right, not left) and merged correctly: merging
  depends on how the letter is labelled.
- **Witnesses help, but less than in experiment 18.** With 16 streams they
  nearly match the whole buffer (55 against 47 for recent, of 60); with 8,
  47 against 36. They protect the machine of the current letter but not the
  streams that ruled out other letters: in 17 of the 18 capped witness runs
  that end wrong, the letter chosen from the kept streams differs from the
  one chosen from the whole buffer. A witness rule for the letter choice is
  the missing piece.
- count2 (4 states, rare outputs) is the hardest task; lap and parity_top
  are exact in every run and every buffer.

## 24 An ordinary network trained by gradient descent (`exp24_gradient_baseline.py`)

Path to Doom, step 1: the first comparison with ordinary networks. Same
training and test data, seeds and scores as the earlier experiments.
boolean: experiment 16's eight functions of 6 inputs, 1 to 16 streams;
images: experiment 19's six 4 × 4 image tasks, 2 to 32 streams; for both an
MLP is trained on the labelled patterns and scored by accuracy on test
patterns not seen in training. state: experiment 20's seven finite-state
tasks, 2 to 32 streams; game: experiment 23's six grid-game tasks, the first
8 to 32 of its episodes; for both an Elman RNN reads x(t) and is trained to
give the output that answers x(0..t), scored by the fraction of unseen test
streams with every output right. Networks (`baseline_gd.py`, numpy only):
64 tanh hidden units, sigmoid output, cross-entropy, full-batch Adam (MLP
2000 steps at 0.01, RNN 3000 steps at 0.005), not tuned per task. 10 seeds.

The discrete networks' numbers come from the committed CSVs: experiment 22's
held-out layout choice (boolean, images; h = 2 or 3 as chosen), experiment
20's composed letters (state, h = 3) and experiment 23's learned bits with
the whole buffer (game, h = 3). Run on Lucas's machine, commit `f18d7b2`,
6 minutes; 1,290 rows, sha256 `8612c1ee088afe62`. Every gradient network
fitted its training data exactly (mean 1.000).

The dashboard run printed "?" for the discrete column: the script looked
for the committed CSVs in the dashboard's run folder. Fixed in this PR; the
discrete column below was filled in from the committed CSVs by the fixed
summary, without rerunning. The MLP and RNN rows are unaffected.

### 24.1 Results

Mean over 10 seeds: gradient network / discrete network.

**boolean** (accuracy on unseen patterns)

| task | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| majority | 0.71 / 0.70 | 0.81 / 0.85 | 0.94 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 |
| and3 | 0.80 / 0.87 | 0.91 / 0.95 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| xor2 | 0.59 / 0.78 | 0.76 / 1.00 | 0.97 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| dnf3 | 0.61 / 0.64 | 0.70 / 0.74 | 0.86 / 0.98 | 0.97 / 1.00 | 0.96 / 1.00 |
| mux | 0.57 / 0.55 | 0.61 / 0.65 | 0.73 / 0.76 | 0.83 / 0.89 | 0.91 / 1.00 |
| equal3 | 0.71 / 0.68 | 0.75 / 0.75 | 0.83 / 0.76 | 0.92 / 0.88 | 0.96 / 0.96 |
| parity6 | 0.42 / 0.43 | 0.38 / 0.35 | 0.35 / 0.40 | 0.39 / 0.35 | 0.36 / 0.73 |
| random | 0.48 / 0.49 | 0.47 / 0.47 | 0.43 / 0.50 | 0.48 / 0.47 | 0.53 / 0.38 |
| mean | 0.61 / 0.64 | 0.67 / 0.72 | 0.76 / 0.79 | 0.82 / 0.82 | 0.84 / 0.88 |

**images** (accuracy on unseen patterns)

| task | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| hline | 0.61 / 0.68 | 0.71 / 0.76 | 0.79 / 0.83 | 0.89 / 0.90 | 0.92 / 0.96 |
| vline | 0.63 / 0.69 | 0.71 / 0.72 | 0.81 / 0.81 | 0.89 / 0.91 | 0.93 / 0.96 |
| block | 0.67 / 0.70 | 0.72 / 0.78 | 0.80 / 0.82 | 0.86 / 0.90 | 0.90 / 0.94 |
| ink8 | 0.70 / 0.78 | 0.80 / 0.80 | 0.89 / 0.90 | 0.94 / 0.99 | 0.96 / 1.00 |
| top | 0.79 / 0.68 | 0.89 / 0.82 | 0.96 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 |
| mirror | 0.52 / 0.55 | 0.62 / 0.65 | 0.75 / 0.81 | 0.91 / 0.81 | 0.98 / 0.87 |
| mean | 0.65 / 0.68 | 0.74 / 0.75 | 0.83 / 0.85 | 0.91 / 0.92 | 0.95 / 0.95 |

**state** (fraction of unseen streams exact)

| task | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| parity/maj012 | 0.00 / 0.83 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| tomita3/xor01 | 0.08 / 0.09 | 0.04 / 0.50 | 0.05 / 0.81 | 0.04 / 0.96 | 0.05 / 1.00 |
| tomita4/and01 | 0.21 / 0.12 | 0.27 / 0.35 | 0.26 / 0.84 | 0.29 / 0.98 | 0.40 / 1.00 |
| tomita5/x0 | 0.02 / 0.01 | 0.02 / 0.48 | 0.02 / 0.87 | 0.02 / 1.00 | 0.03 / 1.00 |
| tomita7/maj345 | 0.12 / 0.11 | 0.15 / 0.15 | 0.16 / 0.68 | 0.18 / 1.00 | 0.19 / 1.00 |
| tomita4/dnf2 | 0.21 / 0.18 | 0.19 / 0.21 | 0.19 / 0.64 | 0.22 / 0.92 | 0.22 / 1.00 |
| tomita7/or3 | 0.60 / 0.53 | 0.54 / 0.45 | 0.58 / 0.64 | 0.57 / 0.79 | 0.57 / 0.94 |
| mean | 0.18 / 0.27 | 0.17 / 0.45 | 0.18 / 0.78 | 0.19 / 0.95 | 0.21 / 0.99 |

**game** (fraction of unseen episodes exact)

| task | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| key_door | 0.31 / 0.78 | 0.36 / 1.00 | 0.40 / 1.00 | 0.42 / 1.00 |
| lap | 0.14 / 1.00 | 0.14 / 1.00 | 0.14 / 1.00 | 0.15 / 1.00 |
| count2 | 0.44 / 0.76 | 0.43 / 0.93 | 0.43 / 0.97 | 0.41 / 0.97 |
| monster_key | 0.10 / 0.82 | 0.13 / 0.94 | 0.14 / 0.99 | 0.17 / 1.00 |
| parity_top | 0.01 / 1.00 | 0.01 / 1.00 | 0.01 / 1.00 | 0.01 / 1.00 |
| both_bottom | 0.40 / 0.77 | 0.42 / 0.98 | 0.44 / 0.98 | 0.42 / 1.00 |
| mean | 0.23 / 0.85 | 0.25 / 0.97 | 0.26 / 0.99 | 0.27 / 1.00 |

### 24.2 Findings

- **Without memory the two are close.** On Boolean functions and images the
  MLP and the discrete network are within a few points at every size (means
  0.84 / 0.88 and 0.95 / 0.95 at the largest). The discrete network is
  ahead on xor2, dnf3 and mux at small sizes and on the line and block
  images; the MLP is ahead on top at few streams and on mirror at 16–32
  (0.98 / 0.87). Neither learns parity6 or the random function, except the
  discrete network's parity6 at 16 streams (0.73).
- **With memory the discrete network is far ahead.** The RNN fits every
  training stream but gets 0.21 of unseen state streams and 0.27 of unseen
  game episodes exact at the largest size, and barely improves with more
  data; the discrete network reaches 0.99 and 1.00. On parity tasks the RNN
  scores 0.00–0.01, the discrete network 1.00.
- **Caveats.** The RNN is one untuned setting (64 units, 3000 steps), and
  "every output of a 25-step stream right" is a strict score for a network
  with soft outputs; a tuned or larger RNN, an LSTM or GRU, or more data
  would likely do better. The discrete networks are helped by building
  blocks Claude designed for these tasks (helper layouts, letter libraries,
  screen regions); the gradient networks get none. So this shows that the
  discrete networks are not beaten by a plain baseline on these small exact
  tasks, not that they are better learners in general.
- **Fixed in this PR:** the summary now reads the committed CSVs from
  `experiments/results/` even when the dashboard sets `DNN_RESULTS_DIR`.

## 25 Witnesses for the letter choice (`exp25_letter_witnesses.py`)

Follow-up to experiment 23, path to Doom step 2. There, a bounded buffer that
kept the merged machine's witnesses still ended wrong when the letter chosen
from the kept streams differed from the one chosen from the whole buffer: the
streams that ruled out other letters had been dropped. The new policy
**letter_witness** chooses the letter on the whole buffer, keeps the
machine's witnesses (the newest, if they exceed the cap), fills up to the cap
with recent streams, then, while the letter chosen from the kept streams is
a different one (the rival), swaps the oldest filler for the stream that
makes the rival inconsistent, or else gives it the most states. Same game,
tasks, seeds and schedule as experiment 23 (32 episodes, 4 per round, a
sleep every 2 rounds; h = 3). Buffers: all; witness and letter_witness with
caps 16, 8, 4; recent with cap 4. Run on Lucas's machine, commit `f18d7b2`,
6 minutes; 1,920 rows, sha256 `52133f7cb3db0fba`. The 720 rows for all,
witness 16 and witness 8 equal experiment 23's exactly (timing columns
ignored).

### 25.1 Results

After each sleep: runs with the true machine (of 10) / mean fraction of
unseen episodes exact. lap and parity_top are 10 / 1.00 at every size for
every buffer except recent 4 (lap 7–10, parity_top 9–10).

| task | buffer | 8 streams | 16 streams | 24 streams | 32 streams |
|---|---|---|---|---|---|
| key_door | all | 3 / 0.78 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| key_door | witness 8 | 3 / 0.78 | 9 / 0.98 | 6 / 0.91 | 8 / 0.95 |
| key_door | letter_witness 8 | 3 / 0.78 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| key_door | recent 4 | 1 / 0.55 | 1 / 0.59 | 0 / 0.59 | 0 / 0.50 |
| key_door | witness 4 | 1 / 0.67 | 2 / 0.73 | 0 / 0.59 | 3 / 0.74 |
| key_door | letter_witness 4 | 2 / 0.73 | 3 / 0.76 | 1 / 0.62 | 4 / 0.85 |
| count2 | all | 2 / 0.76 | 7 / 0.93 | 8 / 0.97 | 8 / 0.97 |
| count2 | witness 16 | 2 / 0.76 | 7 / 0.93 | 7 / 0.95 | 5 / 0.89 |
| count2 | letter_witness 16 | 2 / 0.76 | 7 / 0.93 | 8 / 0.97 | 8 / 0.97 |
| count2 | witness 8 | 2 / 0.76 | 3 / 0.81 | 2 / 0.73 | 2 / 0.77 |
| count2 | letter_witness 8 | 2 / 0.76 | 7 / 0.93 | 8 / 0.97 | 7 / 0.94 |
| count2 | recent 4 | 0 / 0.58 | 0 / 0.53 | 0 / 0.58 | 0 / 0.53 |
| count2 | witness 4 | 0 / 0.59 | 1 / 0.62 | 0 / 0.57 | 1 / 0.67 |
| count2 | letter_witness 4 | 1 / 0.71 | 2 / 0.69 | 0 / 0.49 | 3 / 0.71 |
| monster_key | all | 3 / 0.82 | 5 / 0.94 | 9 / 0.99 | 10 / 1.00 |
| monster_key | witness 8 | 3 / 0.82 | 5 / 0.90 | 9 / 0.99 | 7 / 0.94 |
| monster_key | letter_witness 8 | 3 / 0.82 | 5 / 0.94 | 9 / 0.99 | 8 / 0.98 |
| monster_key | recent 4 | 0 / 0.60 | 0 / 0.35 | 1 / 0.60 | 1 / 0.51 |
| monster_key | witness 4 | 2 / 0.64 | 0 / 0.49 | 5 / 0.73 | 3 / 0.71 |
| monster_key | letter_witness 4 | 2 / 0.62 | 1 / 0.56 | 7 / 0.85 | 4 / 0.78 |
| both_bottom | all | 5 / 0.77 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | witness 8 | 5 / 0.77 | 7 / 0.91 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | letter_witness 8 | 5 / 0.77 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |
| both_bottom | recent 4 | 1 / 0.65 | 2 / 0.62 | 2 / 0.60 | 2 / 0.58 |
| both_bottom | witness 4 | 3 / 0.63 | 5 / 0.81 | 6 / 0.75 | 6 / 0.74 |
| both_bottom | letter_witness 4 | 4 / 0.72 | 8 / 0.98 | 9 / 0.98 | 10 / 1.00 |

Rows not shown (key_door, monster_key and both_bottom with cap 16) are the
same for all three policies.

At the end (of 60 runs): true machine, mean unseen exact, mean recall of all
streams seen (no key), runs that dropped a witness.

| buffer | true machine | unseen exact | recall | dropped a witness |
|---|---|---|---|---|
| all | 58 | 0.995 | 1.000 | – |
| witness 16 | 55 | 0.981 | 0.992 | 0 |
| letter_witness 16 | 58 | 0.995 | 1.000 | 0 |
| witness 8 | 47 | 0.943 | 0.966 | 0 |
| letter_witness 8 | 55 | 0.987 | 0.996 | 0 |
| recent 4 | 21 | 0.672 | 0.710 | – |
| witness 4 | 33 | 0.811 | 0.842 | 6 |
| letter_witness 4 | 41 | 0.890 | 0.908 | 5 |

### 25.2 Findings

- **Keeping the streams that rule out rival letters fixes most of
  experiment 23's gap.** With 16 streams letter_witness equals the whole
  buffer (58 of 60, 0.995); with 8 it gets 55 (witness: 47). The 3 runs
  short of the whole buffer are count2 (1) and monster_key (2).
- **A buffer of 4 is too small for every policy**, but letter_witness is
  still best (41 of 60, against 33 for witness and 21 for recent). It
  solves both_bottom fully; key_door and count2 fail in most runs. At this
  cap the machine's witnesses alone overflow it in 5–6 runs.
- So a buffer of about 8 streams, chosen by letter_witness, holds what the
  sleep needs on these tasks; the old buffer of everything is no longer
  required here.

## 26 Imitating an expert, then playing (`exp26_imitate_expert.py`)

Path to Doom, step 2: acting with multi-bit outputs. On experiment 23's
4 × 4 two-plane grid, an expert fetches the key (top-right quadrant) and
takes it to the door (bottom-left quadrant), again and again. Key and door
are not drawn, so the expert must remember whether it holds the key. Its
move is 2 output bits. This is a 2-state machine reading the letter (player
in top half, player in right half). Layout (h = 3, `make_layout_bits` with 2
output neurons reading only the state neurons): **bits** chooses the letter
from experiment 23's 32 regions; **true_bits** is given (P:top, P:right).
Trained on 8 to 32 expert episodes; scored on unseen episodes (every move
right) and in closed-loop play: the network's moves drive the player for 60
steps, counted in laps. Because the output comes h = 3 steps after the
input, the same lag is given to an "expert with lag" reference. 10 seeds.
Run on Lucas's machine, commit `f18d7b2`, 25 seconds; 80 rows, sha256
`03005b04077f3e8b`. No earlier rows to reproduce.

### 26.1 Results

Runs with the true machine (of 10) / mean fraction of unseen episodes with
every move right:

| layout | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| bits | 8 / 0.89 | 10 / 1.00 | 10 / 1.00 | 10 / 1.00 |
| true_bits | 7 / 0.86 | 8 / 0.99 | 8 / 0.99 | 9 / 0.99 |

Mean laps per game of 60 steps (network / expert with lag 3 / expert /
random):

| layout | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| bits | 2.49 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 |
| true_bits | 2.49 / 2.74 / 14.28 / 1.45 | 2.77 / 2.74 / 14.28 / 1.45 | 2.77 / 2.74 / 14.28 / 1.45 | 2.76 / 2.74 / 14.28 / 1.45 |

bits always chose (P:top, P:left): P:left is the complement of P:right, so
the letter is equivalent to the true one. No hidden neurons were added by
training; recall from keys 1.000.

### 26.2 Findings

- **Imitation works.** From 16 episodes the network finds the expert's
  machine in every run and gets every move of every unseen episode right.
  At 8 episodes 2 runs are wrong; one (seed 4) found 3 states and plays
  badly (0.23 laps).
- **Given the true regions it does slightly worse** (true machine in 9 of
  10 at 32 episodes): seeds 1 and 6 merge into 3 states instead of 2, with
  every training episode still right. This is the labelling dependence of
  merging seen in experiment 23. Those wrong machines happen to play a
  little better than the lagged expert (2.85–2.88 laps against 2.72–2.76).
- **The lag is the bottleneck for acting.** The network plays exactly as
  well as the expert does with a 3-step lag (2.74 laps), which is a fifth
  of the expert without lag (14.28) and not far above random (1.45). The
  network copies the expert's decisions correctly, but by the time it acts
  on a screen the player has moved on. Next step: a shorter latency
  (h = 2, or outputs read directly from letter neurons) or a game that
  waits for the move.

## 27 Choosing the image layout with nbr3_upstart among the choices (`exp27_choose_layout_nbr3.py`)

Closes a gap in experiment 22: its image choices were upstart, cover and
nbr3_cover, but experiment 21's best layout overall, nbr3_upstart
(neighbouring-pixel pair ANDs in layer 1, upstart in layer 2, h = 3), was
not among them. Same method as experiment 22 (held-out training streams, or
fewest helpers, against each fixed layout and the oracle), image tasks only,
with four layouts. 10 seeds. Run on Lucas's machine, commit `01f1d34`
(after the fix in PR #32: the first run stopped at once with a KeyError),
12 minutes; 1,200 rows, sha256 `24c1a3df66b59d7c`. The 900 rows of upstart,
cover and nbr3_cover equal experiment 22's image rows, and all 1,200 equal
experiment 21's on the columns they share (timing columns ignored).

### 27.1 Results

Mean accuracy on unseen patterns (midpoint), over all tasks and seeds:

| layout or rule | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| upstart | 0.674 | 0.720 | 0.800 | 0.772 | 0.707 |
| cover | 0.669 | 0.727 | 0.772 | 0.824 | 0.822 |
| nbr3_cover | 0.672 | 0.740 | 0.788 | 0.843 | 0.886 |
| nbr3_upstart | 0.688 | 0.771 | 0.831 | 0.880 | 0.913 |
| heldout | 0.677 | 0.765 | 0.858 | 0.929 | 0.966 |
| fewest | 0.674 | 0.760 | 0.857 | 0.906 | 0.920 |
| oracle | 0.729 | 0.810 | 0.876 | 0.936 | 0.969 |
| heldout, experiment 22 (3 layouts) | 0.681 | 0.754 | 0.854 | 0.918 | 0.954 |

Layout chosen by the held-out rule, runs per task (all sizes), and runs
where it is the oracle's layout:

| task | upstart | cover | nbr3_cover | nbr3_upstart | same as oracle | of |
|---|---|---|---|---|---|---|
| hline | 1 | 5 | 33 | 11 | 31 | 50 |
| vline | 6 | 5 | 30 | 9 | 28 | 50 |
| block | 3 | 2 | 34 | 11 | 32 | 50 |
| ink8 | 30 | 5 | 6 | 9 | 31 | 50 |
| top | 40 | 4 | 1 | 5 | 40 | 50 |
| mirror | 9 | 4 | 3 | 34 | 29 | 50 |

### 27.2 Findings

- **Adding nbr3_upstart helps the held-out choice** by about 0.01 from 4
  streams on (0.966 at 32, against 0.954 with three layouts). It is within
  0.02 of the oracle from 8 streams on (0.003 at 32), 0.05 below at 2–4. nbr3_upstart is the best single
  layout at every size (0.913 at 32), but choosing per task is still better
  by 0.05.
- The held-out rule picks nbr3_upstart mostly for mirror (34 of 50),
  nbr3_cover for lines and blocks, and upstart for ink8 and top.
- **Fewest helpers gets worse with the fourth layout** at 32 streams (0.920,
  against 0.955 in experiment 22): fewest helpers is not a reliable rule
  across layouts of different depth.
- For the comparison of experiment 24, the discrete networks' image score at
  32 streams becomes 0.97 against the MLP's 0.95.

## 28 Acting without the lag (`exp28_acting_latency.py`)

Path to Doom, step 2 (fix of experiment 26). Experiment 26's network
imitates the key/door expert exactly but plays like an expert with a 3-step
lag, because with h = 3 its move answers the screen of 3 steps earlier.
Three variants, each with layouts **bits** (letter chosen from 32 regions)
and **true_bits** (given P:top, P:right), 8 to 32 training episodes, 10
seeds:
- **lag3**: experiment 26 unchanged.
- **direct**: the output neurons also read the letter-bit neurons and are
  trained to answer 2 steps after the input (h = 2 for the outputs; letter
  and state neurons as before).
- **wait**: the game waits for the move. Each screen is shown for 3 network
  steps and the move is read 3 steps after the screen first appears, so it
  answers the current screen. The training episodes show each screen 3
  times too (h = 3).

Run on Lucas's machine, commit `b7822fc`, 1 minute 40
seconds; 240 rows, sha256 `1d32e4dab1763014`. The 80 lag3 rows equal
experiment 26's rows (17 columns, timing ignored).

### 28.1 Results

Runs with the true machine (of 10) / mean fraction of unseen episodes with
every move right / runs in which TRAIN added hidden neurons:

| variant | layout | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| lag3 | bits | 8 / 0.89 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 |
| lag3 | true_bits | 7 / 0.86 / 0 | 8 / 0.99 / 0 | 8 / 0.99 / 0 | 9 / 0.99 / 0 |
| direct | bits | 8 / 0.00 / 10 | 10 / 0.00 / 10 | 10 / 0.00 / 10 | 10 / 0.00 / 10 |
| direct | true_bits | 7 / 0.00 / 10 | 8 / 0.00 / 10 | 8 / 0.00 / 10 | 9 / 0.00 / 10 |
| wait | bits | 8 / 0.90 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 |
| wait | true_bits | 8 / 0.90 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 | 10 / 1.00 / 0 |

Mean laps per game of 60 game steps (network / expert with the same lag /
expert without lag / random):

| variant | lag | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| lag3, bits | 3 | 2.49 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 | 2.74 / 2.74 / 14.28 / 1.45 |
| direct, bits | 2 | 0.37 / 3.36 / 14.28 / 1.45 | 0.06 / 3.36 / 14.28 / 1.45 | 0.00 / 3.36 / 14.28 / 1.45 | 0.00 / 3.36 / 14.28 / 1.45 |
| wait, bits | 0 | 12.90 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 |
| wait, true_bits | 0 | 12.90 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 | 14.28 / 14.28 / 14.28 / 1.45 |

true_bits for lag3 and direct (not shown): 2.49 to 2.77 and 0.00 to 0.04
laps.

### 28.2 Findings

- **Waiting for the move removes the lag completely.** From 16 episodes,
  every run plays exactly like the expert without lag (14.28 laps, 5 times
  experiment 26's 2.74). No hidden neurons are added. The game takes 3
  network steps per move, which is the price.
- **Reading the letter directly does not work.** With the output 2 steps
  after the input, one output neuron would need the move as a threshold
  function of the previous (state, letter) neuron and the current letter
  bits. For the vertical move bit the sign of the letter weights depends on
  the state, so no single neuron can do it. TRAIN then adds hidden neurons
  in every run (up to 40) that memorise the training episodes: no unseen
  episode is right and play is worse than random. Even a perfect lag-2
  player would gain little (expert with lag 2: 3.36 laps).
- With repeated screens, true_bits finds the true machine in all 10 runs
  from 16 episodes (8 of 10 without repeats). Repeating each letter seems
  to remove the labelling dependence of merging seen in experiments 23 and
  26, but this was not tested further.

## 29 Training labels with noise (`exp29_noisy_labels.py`)

Path to Doom, step 3: the first experiments with labels that are not exact.
A fraction p = 0, 0.05, 0.1, 0.2 of the training labels is wrong, and the
test labels are exact. The algorithms are unchanged; this only measures the
damage.
- **images**: experiment 19's six 4 × 4 image tasks, 2 to 32 streams. A
  training image's label is flipped when a hash of (image, seed) falls
  below p (`dnn_noise.noisy`), so an image always gets the same label and
  the images flipped at 0.05 are among those flipped at 0.1. Layouts:
  **cover** (experiment 19, h = 2), **nbr3_upstart** (experiment 21, h = 3,
  at most 200 helpers), and experiment 24's **MLP** (64 tanh units, Adam,
  untuned) on the same noisy images.
- **game**: experiment 28's wait game, layouts bits and true_bits, 8 to 32
  episodes. With probability p the expert's move for a screen is replaced
  by one of the 3 other moves (on all 3 network steps of that screen).

10 seeds. Run on Lucas's machine, commit `b7822fc`, about
20 minutes; 3,920 rows, sha256 `5c3d05aff2f0cf33`. The p = 0 rows equal
experiment 19's cover rows, experiment 27's nbr3_upstart rows, experiment
24's image MLP rows and experiment 28's wait rows (300, 300, 300 and 80
rows; timing ignored).

### 29.1 Results

Images, mean accuracy on unseen test images (midpoint weights), all tasks
and seeds:

| layout | p | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| cover | 0 | 0.669 | 0.727 | 0.772 | 0.824 | 0.822 |
| cover | 0.05 | 0.658 | 0.691 | 0.728 | 0.756 | 0.736 |
| cover | 0.1 | 0.645 | 0.664 | 0.695 | 0.719 | 0.689 |
| cover | 0.2 | 0.592 | 0.608 | 0.634 | 0.644 | 0.629 |
| nbr3_upstart | 0 | 0.688 | 0.771 | 0.831 | 0.880 | 0.913 |
| nbr3_upstart | 0.05 | 0.656 | 0.724 | 0.765 | 0.779 | 0.822 |
| nbr3_upstart | 0.1 | 0.637 | 0.679 | 0.715 | 0.728 | 0.760 |
| nbr3_upstart | 0.2 | 0.591 | 0.612 | 0.648 | 0.645 | 0.692 |
| MLP | 0 | 0.653 | 0.742 | 0.833 | 0.914 | 0.948 |
| MLP | 0.05 | 0.639 | 0.711 | 0.774 | 0.841 | 0.886 |
| MLP | 0.1 | 0.624 | 0.684 | 0.738 | 0.800 | 0.840 |
| MLP | 0.2 | 0.582 | 0.627 | 0.673 | 0.713 | 0.745 |

At 32 streams, mean helpers: cover 52 / 82 / 97 / 118 and nbr3_upstart
1.9 / 5.0 / 9.8 / 28.8 for p = 0 / 0.05 / 0.1 / 0.2. No hidden neurons were
added in any run. Accuracy on seen images against their true labels is
about 1 − p for all three models (0.94, 0.88, 0.77 at 32 streams): every model
reproduces every wrong label it was given.

Game: runs (of 10) in which a consistent letter was found / runs with the
true machine / mean laps of the trained runs:

| layout | p | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| bits | 0 | 10 / 8 / 12.90 | 10 / 10 / 14.28 | 10 / 10 / 14.28 | 10 / 10 / 14.28 |
| bits | 0.05 | 7 / 0 / 2.86 | 1 / 0 / 8.04 | 0 / 0 / – | 0 / 0 / – |
| bits | 0.1 | 4 / 0 / 1.71 | 0 / 0 / – | 0 / 0 / – | 0 / 0 / – |
| bits | 0.2 | 2 / 0 / 1.05 | 0 / 0 / – | 0 / 0 / – | 0 / 0 / – |
| true_bits | 0 | 10 / 8 / 12.90 | 10 / 10 / 14.28 | 10 / 10 / 14.28 | 10 / 10 / 14.28 |
| true_bits | 0.05 | 1 / 0 / 11.81 | 1 / 0 / 12.96 | 0 / 0 / – | 0 / 0 / – |
| true_bits | 0.1 | 1 / 0 / 4.33 | 0 / 0 / – | 0 / 0 / – | 0 / 0 / – |
| true_bits | 0.2 | 0 / 0 / – | 0 / 0 / – | 0 / 0 / – | 0 / 0 / – |

### 29.2 Findings

- **Images: noise costs the discrete networks about as much as the MLP, and
  more at the larger sizes.** At 32 streams and p = 0.1, nbr3_upstart drops
  from 0.913 to 0.760 (−0.15), cover from 0.822 to 0.689 (−0.13) and the
  MLP from 0.948 to 0.840 (−0.11). The helper search fits every wrong label
  with more helpers (nbr3_upstart: 2 to 29 at p = 0.2), so the noise is
  memorised by the helper layer rather than by hidden neurons. The MLP,
  untuned and trained to fit, also memorises every wrong label, so this is
  not yet a fair test of a noise-robust learner on either side.
- **Game: exact merging breaks down at once.** At p = 0.05 and 24 or more
  episodes no letter gives a consistent machine in any run, because no
  machine explains every wrong move. The few runs that pass with fewer
  episodes found wrong machines (0 of 10 true). This is the failure that
  experiment 30 addresses.

## 30 A first noise-tolerant training (`exp30_noise_tolerant.py`)

Path to Doom, step 3. The same data, noise levels, sizes and seeds as
experiment 29, with one change per family made before the linear programs.
The linear programs and layouts are unchanged.
- **game: the letter and machine by search instead of exact merging**
  (`dnn_noise`). For each letter (at most 2 of the 32 regions), find the
  machine of at most 2 states that disagrees least with the training moves:
  every transition table is tried, and each (state, letter) cell outputs
  the majority of the moves seen there. Keep the letter with the fewest
  disagreements. Then choose its number of states (1 to 3; 3 by local
  search from 8 random starts) by disagreements on held-out episodes
  (every 4th) and refit on all episodes. The training episodes are then
  relabelled with the machine's moves (`dnn_noise.relabel`) and trained as
  in experiment 28's wait game. Layouts bits and true_bits (letter given,
  number of states still chosen).
- **images: a disagreement filter.** The distinct training images are split
  into 4 folds. A network built on 3 folds predicts the 4th, and the images
  whose (noisy) label disagrees with the prediction are dropped. The final
  network is built on the rest, with the kept images shown as one stream.
  Layouts: nbr3_upstart (h = 3) and the MLP, each filtered by its own kind
  of network.

The first attempt for the game was greedy merging with a vote: counts in the
prefix tree, and a merge allowed when the pooled minority was at most 30%,
or when ALERGIA's Hoeffding test passed. In a check on true_bits it found
the true machine in few runs, even without noise: key and no-key states
differ on only some letters, and a wrong early merge cannot be undone. It is
not in the CSV and was replaced by the search.

Run on Lucas's machine, commit `b7822fc`, 28 minutes;
2,720 rows, sha256 `c4c74ae5ffbfdc7f`. No earlier rows to reproduce (every
row uses the new method); experiment 29's rows are the comparison.

### 30.1 Results

Game: runs with the true machine (of 10) / mean fraction of unseen episodes
with every move right / mean laps (experiment 29, exact merging: runs with
any machine at all):

| layout | p | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| bits | 0 | 10 / 1.00 / 14.28 (10) | 10 / 1.00 / 14.28 (10) | 10 / 1.00 / 14.28 (10) | 10 / 1.00 / 14.28 (10) |
| bits | 0.05 | 10 / 1.00 / 14.28 (7) | 10 / 1.00 / 14.28 (1) | 10 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) |
| bits | 0.1 | 10 / 1.00 / 14.28 (4) | 10 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) |
| bits | 0.2 | 10 / 1.00 / 14.28 (2) | 10 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) |
| true_bits | 0.2 | 10 / 1.00 / 14.28 (0) | 9 / 1.00 / 14.28 (0) | 10 / 1.00 / 14.28 (0) | 9 / 1.00 / 14.28 (0) |

(true_bits at p ≤ 0.1: 10 / 1.00 / 14.28 everywhere.) bits always chose
P:top + P:left (equivalent to the true letter). In the two true_bits runs
counted as wrong, held-out choice picked 3 states. The extra state never
changes a move on the test episodes or in play. After relabelling, every
training move equals the clean expert's in all 320 runs. No hidden neurons
were added.

Images, mean accuracy on unseen test images, filtered (experiment 29
unfiltered):

| layout | p | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| nbr3_upstart | 0 | 0.609 (0.688) | 0.707 (0.771) | 0.804 (0.831) | 0.873 (0.880) | 0.919 (0.913) |
| nbr3_upstart | 0.05 | 0.606 (0.656) | 0.676 (0.724) | 0.769 (0.765) | 0.827 (0.779) | 0.864 (0.822) |
| nbr3_upstart | 0.1 | 0.606 (0.637) | 0.665 (0.679) | 0.737 (0.715) | 0.789 (0.728) | 0.808 (0.760) |
| nbr3_upstart | 0.2 | 0.574 (0.591) | 0.616 (0.612) | 0.664 (0.648) | 0.702 (0.645) | 0.731 (0.692) |
| MLP | 0 | 0.608 (0.653) | 0.666 (0.742) | 0.742 (0.833) | 0.857 (0.914) | 0.924 (0.948) |
| MLP | 0.05 | 0.598 (0.639) | 0.651 (0.711) | 0.719 (0.774) | 0.820 (0.841) | 0.908 (0.886) |
| MLP | 0.1 | 0.594 (0.624) | 0.640 (0.684) | 0.702 (0.738) | 0.791 (0.800) | 0.883 (0.840) |
| MLP | 0.2 | 0.570 (0.582) | 0.608 (0.627) | 0.650 (0.673) | 0.731 (0.713) | 0.807 (0.745) |

At 32 streams (about 300 distinct images for nbr3_upstart), the filter
dropped 29 / 67 / 89 / 123 images for p = 0 / 0.05 / 0.1 / 0.2. Of the
30 wrong labels at p = 0.1 it dropped 22 and kept 8.

### 30.2 Findings

- **Game: noise is no longer a problem.** With the search, every run at
  every noise level up to 20% wrong moves, from 8 episodes, plays exactly
  like the expert (14.28 laps) and gets every unseen episode right. Exact
  merging found any machine at all in only 0–7 of 10 runs, and never the
  true one. Relabelling turns the noisy episodes back into exact ones, so
  the linear programs and the no-forgetting proof are untouched.
- **This depends on the machine being tiny.** The search tries every
  transition table for 2 states over 4 letters (256) and uses local search
  for 3 states. Its cost grows as n^(n·L), so it will not scale to the
  large machines of later steps without a better search. Letters still come
  from Claude's 32 regions.
- **Images: the filter helps only with enough data.** At 16–32 streams it
  recovers a fifth to a half of the loss (nbr3_upstart at p = 0.1, 32
  streams: 0.808 against 0.760 unfiltered and 0.913 clean). The MLP gains
  the same way, and with the filter it stays ahead (0.883). With 2–8
  streams the filter hurts. On clean data (p = 0) it hurts everywhere
  except nbr3_upstart at 32 streams. The fold networks are too weak to tell noise from rare patterns,
  so the filter drops many correct images (about 67 of 89 dropped at
  p = 0.1). Noise on images is not solved.

## 31 Learning to play from success and failure (`exp31_success_failure.py`)

Path to Doom, step 5: Lucas's idea 3 (buffer successes; on failure
randomise inside the boxes), first test. The game is experiment 28's wait
game (key/door, 60 game steps). No expert move is ever seen. Each run plays
50 rounds of one game each:
- The player is a random mover until the first success, then the network.
- A game is a **success** if it scores at least *target* laps; the target
  is the most laps of any success so far (at least 1).
- **Success:** the game (its screens and the moves the player made) joins
  the buffer. The network is rebuilt on the whole buffer with experiment
  30's pipeline (letter and machine of at most 3 states by search, then
  relabelling and the linear programs) and plays with the midpoint of its
  boxes.
- **Failure:** every weight is redrawn uniformly inside its box, so the
  network still replays its buffer but may act differently elsewhere.

After every round the current network (midpoint weights) is scored on
experiment 26's 100 test games. References: experiment 28's imitation of
the expert, 14.28 laps; random player, 1.45. 10 seeds. Run on Lucas's
machine, commit `b7822fc`, about 1 minute; 500 rows,
sha256 `ec4b9e32444eb136`. No earlier rows to reproduce.

### 31.1 Results

Mean laps of the current network on the test games / mean successes so
far, by round (before the first success the network scores 0):

| round | 1 | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| mean of 10 seeds | 0.17 / 0.9 | 0.17 / 1.1 | 0.17 / 1.2 | 0.17 / 1.5 | 0.17 / 1.6 | 0.17 / 1.6 | 0.17 / 1.6 |

In 9 of 10 runs the buffer holds a single game, the random first success
(round 1 or 2), for all 50 rounds. Every network found is a 1-state
(memoryless) machine on a letter that is not the key/door letter. Test laps:
seed 0 1.42 (7 successes, target 5), seed 6 0.24, the others 0.00. In 8
runs every exploration game after round 2 scored 0 laps.

### 31.2 Findings

- **Negative result: in this form the idea does not learn.** After one
  random success the network never improves, and it plays worse than the
  random player (0.17 against 1.45 laps).
- **The cause is checked: randomising inside the boxes does not explore.**
  The network is a 1-state machine over a 2-bit letter. Its one buffered
  game already shows every letter, so the boxes fix the move for every
  (state, letter) cell, and every weight drawn inside them gives the same
  moves. For seed 1, 20 redraws played the identical game. This is the
  no-forgetting property working against exploration: the boxes guarantee
  the buffer is replayed, and when the buffer covers every cell there is
  nothing left to vary.
- **A random game is a poor teacher.** Its moves are not a function of any
  short history, so the search finds a 1-state machine fitting few of them,
  and relabelling turns the game into "the same move per letter". Such
  policies get stuck against walls (0 laps).
- What might work, untested: explore by flipping the machine's output in
  one (state, letter) cell at a time instead of redrawing weights; store
  only the steps leading up to each lap rather than whole games; or start
  from a few imitation episodes and use success and failure to improve
  on them.

## 32 A fairer gradient baseline (`exp32_fair_gradient.py`)

Experiment 24 compared the discrete networks on the tasks with memory to an
untuned Elman RNN, scored only on whole streams. This experiment tunes it,
adds a GRU and reports per-step accuracy too. Same data, tasks and seeds as
experiment 24's **state** family (experiment 20's seven finite-state tasks,
2 to 32 streams) and **game** family (experiment 23's six grid-game tasks, 8
to 32 episodes). Models (numpy, full batch, Adam, 3000 steps, BPTT with
gradient clipping at 5):
- **rnn**: experiment 24's RNN (64 tanh units, learning rate 0.005).
- **rnn_tuned**: the RNN with (hidden units, learning rate) chosen from
  (64, 0.005), (32, 0.005), (128, 0.005), (64, 0.02) by per-step accuracy on
  held-out training streams (every 4th; with 2 streams, the second), then
  refitted on all training streams.
- **gru_tuned**: a GRU (`baseline_gd.GRU`, new, gradient-checked in the
  tests) chosen the same way.

The discrete networks are taken from the committed CSVs: experiment 20's
composed letters (h = 3) for state, experiment 23's learned bits (all
regions, h = 3) for the game. Experiment 23 recorded only whole-stream
exactness. Both use letters built from Claude's building blocks (experiment
20's composed letters, experiment 23's 32 regions); the gradient models get
the raw inputs.

Run on Lucas's machine, branch `claude/batch-acting-noise`, 2 hours 30
minutes (longest job 349 s); 1,770 rows, sha256 `d54e4394f073c5c3`. The 590
rnn rows equal experiment 24's rows (train_exact, unseen_exact,
unseen_step_acc).

### 32.1 Results

Unseen test streams with every output right / per-step accuracy, mean of
all tasks and 10 seeds:

| family | model | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| state | rnn | 0.18 / 0.664 | 0.17 / 0.688 | 0.18 / 0.699 | 0.19 / 0.718 | | 0.21 / 0.733 |
| state | rnn_tuned | 0.18 / 0.666 | 0.17 / 0.690 | 0.19 / 0.703 | 0.20 / 0.720 | | 0.22 / 0.734 |
| state | gru_tuned | 0.18 / 0.674 | 0.18 / 0.696 | 0.19 / 0.705 | 0.20 / 0.727 | | 0.24 / 0.747 |
| state | discrete | 0.27 / 0.689 | 0.45 / 0.794 | 0.78 / 0.919 | 0.95 / 0.981 | | 0.99 / 0.997 |
| game | rnn | | | 0.23 / 0.770 | 0.25 / 0.791 | 0.26 / 0.800 | 0.27 / 0.808 |
| game | rnn_tuned | | | 0.24 / 0.776 | 0.26 / 0.798 | 0.27 / 0.811 | 0.29 / 0.824 |
| game | gru_tuned | | | 0.25 / 0.790 | 0.31 / 0.835 | 0.37 / 0.847 | 0.45 / 0.872 |
| game | discrete | | | 0.85 / – | 0.97 / – | 0.99 / – | 1.00 / – |

Every gradient model fits every training stream exactly. The held-out
choice is spread over all four settings (state rnn_tuned: (64, 0.005) in
139 of 350 runs, the others 43 to 96), so tuning changes little.

At the largest size, whole streams right for gru_tuned by task: state from
0.01 (parity/maj012) and 0.03 (tomita5/x0) to 0.61 (tomita7/or3); game from
0.11 (parity_top) to 0.60 (lap). Tuning helps most on lap (rnn 0.15, GRU 0.60).

### 32.2 Findings

- **The gap stays large after tuning.** On whole unseen streams the best
  gradient model (GRU, tuned) reaches 0.24 on state and 0.45 on the game at
  the largest size, against 0.99 and 1.00 for the discrete networks. Tuning
  the RNN adds at most 0.02; the GRU adds more on the game (0.45 against
  0.27), little on state.
- **Per step the gap is smaller but still clear.** With few streams (2) the
  discrete networks are only slightly ahead per step (0.689 against 0.674);
  from 8 streams they are far ahead (0.919 against 0.705 on state at 8;
  0.997 against 0.747 at 32). The gradient models get most single steps
  right but rarely a whole stream.
- **Not fully fair yet.** The discrete networks get letters made from
  Claude's building blocks while the gradient models see raw inputs; the
  grid is small (4 settings), there is no weight decay, early stopping or
  LSTM, and 3000 steps is a fixed budget. The comparison is fairer than
  experiment 24's but not final.

## Summary

**Baseline (`dnn.py`, experiments 1–5).**
- No bugs were found in `dnn.py`. Every learning and no-forgetting check passed,
  the Lemma 4 bound always held, and hidden cohorts started at 0 stayed silent
  (Lemma 1).
- The algorithm does what the Theorem promises — exact recall without
  forgetting — and no more. It generalises only when the task is a threshold
  function of the visible state with no bias (rule 232, previous input,
  Tomita 1), and then only once the training data constrains the visible boxes
  enough.
- When hidden neurons are needed they store the answer in the key rather than
  computing it. Started from an unseen state they stay at 0, so accuracy falls
  to the majority baseline or below.
- An always-on input neuron (experiment 5) changes no outcome. It cannot make
  XOR separable or add memory, and it slightly loosens the boxes.

**Causal variant (`dnn_causal.py`, experiments 6–7, `docs/Causal_Variant.md`).**
- The expected outcomes were confirmed:
  - Both test tasks ($y(t+1) = x(t-1)$ and $x(t) \wedge x(t-1)$) are learned
    with features alone and are exact on unseen streams.
  - Two-step rule 90 is learned with features alone (no chain neurons): 0.85
    exact and 0.98 cell accuracy on unseen states with 32 training states,
    against 0.05 and 0.74 for the baseline. The remaining errors come from
    loose box corners, plus one seed that missed one of the 12 needed features.
  - Two-step parity is realisable with two hand-built features, but the greedy
    search does not find them, so it falls back to chains and stays at the
    baseline.
  - Forgetting appears, and it is near-total: after a later round almost no
    earlier sequence replays.
- One result was not predicted: **Tomita 2 is solved exactly on unseen
  streams** with a single AND feature.
- Nothing else improves:
  - Single-step rule 90 cannot be solved, because a feature reads one step too
    late; the variant does worse than the baseline here.
  - Parity, the adder and Tomita 3, 5 and 6 get features that lower the slack
    but do not remove it, and the variant falls back to full chains.
  - For Tomita 4 and 7, no single candidate helps at all, so the search stops
    at once.
- The variant pays for features in size and time. It still adds chain neurons
  on random data, and its candidate search makes training up to about 300×
  slower over many rounds. That is why experiment 7 runs at reduced sizes.
- Taken together: features computed from the past do generalise when the
  needed computation is one candidate deep (one AND, or an OR/AND pair). The
  greedy, one-step slack search is the bottleneck for anything deeper. Because
  nothing protects old sequences from new features, the variant is not usable
  for continual learning as it stands.

**Past shifts of the inputs (experiment 8, `dnn.py` unchanged).**
- Input neurons that carry the inputs of the last $d$ steps, present before
  round 1, keep both guarantees: the Theorem applies as written, no stream was
  forgotten in any check, and the visible size is fixed.
- They extend what the baseline generalises on from threshold functions of
  the current visible state to threshold functions of the last $d + 1$ inputs
  and the current output. In 1080 runs that single test decided the outcome:
  no hidden neurons and (given 128 training streams) every unseen stream exact
  when it holds, never exact when it does not.
- With a start clock (the always-on input, shifted too) this includes
  **Tomita 4**, which is solved exactly with no hidden neurons.
- They add memory, not new kinds of function. Parity, the adder, `xor2` and
  Tomita 2, 3, 5, 6 and 7 are unchanged at every depth.
- New limitation: when a learnable task is trained in several small rounds,
  the box kept after the first round can exclude every correct weight vector
  (4 of 10 runs for Tomita 4 with 4 streams per round). The run then
  memorises later streams and never improves. Rounds of 16 streams avoided
  this in every run.
- Shifts of the output and other non-input neurons need fixed-weight neurons
  inside the network; these are tested in experiment 9.

**Fixed neurons inside the network (experiment 9, `dnn_fixed.py`,
`docs/Fixed_Neurons.md`).**
- Fixed-weight neurons that exist before round 1 keep the Theorem
  (Proposition 1), and they compute their rule on unseen runs
  (Proposition 2). Both were confirmed in every run: no stream was forgotten,
  and the in-network input shifts reproduced experiment 8 run for run.
- Shifting the output as well adds **Tomita 2**, solved exactly with one shift
  and no hidden neuron. Parity, the adder, `xor2` and Tomita 3, 5, 6 and 7 are
  still unchanged.
- **Twins** (a neuron that is on exactly when its shift neuron is off) solve
  the same tasks as the start clock, which is what Proposition 3 says. They
  cost more neurons, but with little data, and in small rounds, they went
  wrong less often than the clock.
- The wrong-box problem gets worse as fixed neurons are added: Tomita 4 in
  rounds of 4 streams went wrong in 8 of 10 runs with the clock (4 of 10 with
  input shifts only) and in 3 of 10 with twins.
- On random sequences shifts save a quarter to a third of the helper neurons
  per round, and almost every sequence still needs its key.

**Latency, a feature library and state neurons (experiment 10,
`docs/Latency_And_State.md`).**
- Parity, the adder, `xor2` and Tomita 3, 5 and 6 cannot be computed at the
  experiments' timing by any network without a key (Proposition 4). The output
  at the next step would have to react to the current input through one fixed
  weight, and these tasks need that weight to change sign. This is why nothing
  moved them in experiments 4 and 7–9. One step later they are all possible.
- With the state neurons of the task's own machine installed before round 1,
  every one of the 15 tasks is solved exactly where the theory says it can be:
  all of them at latency 1, and at latency 0 the 9 that pass the test,
  Tomita 7 included (270 of 270 runs as predicted). These neurons are built
  from the task's definition; only the output is trained.
- A task-independent library of the variant's features, installed before
  round 1, finds parity and `xor2` at latency 1 and matches the variant's
  greedy search on two-step rule 90 at about 1% of the training time, with no
  forgetting. It does not find the adder or Tomita 3, 5, 6 and 7.
- What is learned is the machine: networks trained on streams of 13 vectors
  are exact on streams of 101 with midpoint weights.
- The library's 105 coordinates make the boxes loose and the wrong-box
  problem severe: parity trained 4 streams per round went wrong in 10 of 10
  runs with the library and in none with the 5 state neurons.

**The wrong-box problem (experiment 11, `docs/Box_Rule.md`).**
- A box cannot keep an open trade-off between weights, such as "these weights
  add up to at least 1"; it has to settle it. After a small round the choice
  is arbitrary and can exclude every correct weight vector.
- No rule for choosing the box avoids this (Proposition 6), and none of the
  four tried was better overall than the notes' rule.
- A constraint buffer does: the trained neuron keeps its first constraints
  exactly and turns them into a box only when there are too many. With a
  buffer that does not overflow, no run went wrong (0 of 240, against 77
  without), including when streams were trained one at a time, and no stream
  was forgotten. It needed at most 156 stored constraints, about 3 per visible
  neuron at most.
- The buffer adds storage that is still polynomial in the number of neurons.
  Whether to adopt it is a decision for Lucas.

**Output timing (experiment 12).**
- The number of steps $h$ between an input and the output that depends on it
  is a parameter of the training data, and it matters: $h = 2$ was best,
  $h = 1$ makes six tasks impossible, and larger $h$ fails unless the network
  has neurons to hold the answer.

**Finding the state neurons (experiment 13, `docs/State_Merging.md`).**
- A state is a future: two input histories are in the same state when every
  continuation gives the same outputs. Merging the nodes of the prefix tree
  of the training streams, wherever the data do not contradict it, finds the
  task's machine from the data alone.
- With 128 training streams all 15 tasks are learned in every run (150 of
  150), exactly, on unseen streams and on streams eight times longer. 32
  streams are enough in 147 of 150 runs.
- The training streams are reproduced without keys (Proposition 8), and no
  stream was forgotten in later rounds.
- The unmerged tree is the same network with nothing generalised: it
  memorises its streams without keys and is right on 11 to 14% of unseen
  ones.
- The machine has to be settled by the first streams. Found from 4 streams it
  was wrong in 34 of 60 runs; from 16, in 4. A bounded buffer of early streams
  would postpone the merge in the way the constraint buffer postpones the box.

**Boolean functions at depth 2 (experiment 14, `dnn_upstart.py`).**
- Upstart-style helpers, built from the data and installed as fixed neurons,
  generalise across input vectors: majority, and3 and xor2 are exact on unseen
  patterns from 8 streams, dnf3 reaches 0.89, with no hidden neurons from
  TRAIN and nothing forgotten. Mean unseen accuracy 0.78 against 0.56.
- They lose to the constant answer on equal3 (isolated positives) and gain
  nothing on parity6, as expected. OR-of-AND functions (dnf3, mux) need a
  better search for groups of helpers, or a second hidden layer.

**Wake-and-sleep cycles (experiment 15, `dnn_sleep.py`).**
- Sleeping (state merging on a buffer, then a rebuild) corrects the wrong
  machine of an early commitment: 59 of 60 runs end with the true machine,
  against 26 without sleep, with no hidden neurons and nothing in the buffer
  forgotten.
- A buffer of the 16 most recent streams loses 2 of these runs (tomita4).
  Safe deletion never fired: it removes only unused neurons.

**Helpers for OR-of-AND functions (experiment 16).**
- A covering search (each helper off on every negative, on for one seed,
  smallest weights) makes dnf3 and mux exact on unseen patterns (from 8 and
  16 streams), but is worse than upstart on majority, equal3 and parity6.
- A second hidden layer over every AND of two literals (h = 3) has the best
  mean (0.89 against 0.78 at 16 streams) and makes equal3 exact; mux reaches
  0.85.

**State neurons that read helpers (experiment 17).**
- A finite-state task on a binary letter of six input bits: choosing the
  letter from a library (fewest merged states) and letting state neurons
  read its helpers makes the five library tasks exact or nearly exact on
  unseen streams from 16 streams, when 42% of test streams still hold an
  unseen pattern; merging on the raw patterns gets at most 4%. A letter
  outside the library is not found. Depth h = 3.

**Bounded buffers (experiment 18).**
- Keeping the witnesses (drop a stream only if the merged machine stays the
  same) matches the unbounded buffer in every sleep, even with 8 streams:
  119 of 120 runs end right, against 79 (most recent 8) and 89 (random 8).

**Small images (experiment 19, ladder rung 2).**
- On 4 × 4 images, threshold properties (ink count, top heavier than
  bottom) are exact on unseen images with one upstart helper. Shapes (lines,
  2 × 2 blocks) reach 0.82–0.87 with the cover search; upstart falls to the
  constant answer when its helpers run out. Mirror symmetry is hard for all.

**Learning the letter (experiment 20).**
- Growing the library by one step (OR and AND of two library letters) finds
  both letters outside it, (x0 ∧ x1) ∨ (x2 ∧ x3) and x0 ∨ x1 ∨ x2, and
  then matches the true letter's results (dnf2 exact from 32 streams). A
  joint search with a free label per pattern overfits (small wrong machines)
  and runs out of budget with more data: the letter needs a class of simple
  functions.

**More helpers for shapes (experiment 21).**
- Removing the 20-helper limit stops upstart's collapse and lets cover keep
  improving. A layer of neighbouring-pixel pair ANDs (h = 3) is better
  still: with cover, lines 0.96 and blocks 0.94; with upstart, 0.91 mean
  over all six tasks from 1 to 5 helpers, and mirror 0.94.

**Choosing the layout (experiment 22).**
- Choosing between layouts by scoring them on held-out training streams
  comes within 0.003 of the best possible choice on images at 16 and 32
  streams (0.954 at 32, against 0.886 for the best single layout). On the
  Boolean functions it helps at 4 and 8 streams; fewest helpers is as good
  with very few streams.

**A tiny grid game (experiment 23, ladder rung 3).**
- On a 4 × 4 two-plane game with random walkers, letters chosen from screen
  regions plus state neurons give the true machine in 58 of 60 runs (0.995
  of unseen episodes exact). Witnesses keep a bounded buffer close to that
  with 16 streams (55), less so with 8 (47): they do not protect the
  streams that ruled out other letters.

**Gradient-descent baseline (experiment 24, path to Doom step 1).**
- On the same data, an ordinary MLP (64 tanh units, Adam) is close to the
  discrete networks on Boolean functions and images (0.84 / 0.88 and
  0.95 / 0.95 at the largest sizes). An Elman RNN fits its training streams
  but gets 0.21 of unseen state streams and 0.27 of unseen game episodes
  exact, against 0.99 and 1.00. The RNN is untuned and the discrete networks
  use designed building blocks, so this is a sanity check, not proof of a
  better learner.

**Witnesses for the letter choice (experiment 25).**
- Keeping, besides the machine's witnesses, the streams that rule out rival
  letters (letter_witness) makes a buffer of 16 as good as keeping
  everything (58 of 60 true machines) and one of 8 nearly so (55, against 47
  for witnesses alone). With 4 it gets 41, against 33 and 21.

**Imitating an expert (experiment 26).**
- On a key-and-door game with 2-bit moves, the network learns the expert's
  2-state machine from 16 episodes in every run (every unseen move right).
  In closed-loop play it equals the expert with the same 3-step lag (2.74
  laps per 60 steps), a fifth of the expert without lag (14.28): latency,
  not learning, limits acting.

**Choosing the image layout with nbr3_upstart (experiment 27).**
- With nbr3_upstart as a fourth choice, choosing the layout by held-out
  training streams reaches 0.966 on images at 32 streams (0.954 with three
  layouts; oracle 0.969; best single layout 0.913). Fewest helpers drops to
  0.920.

**Acting without the lag (experiment 28).**
- A game that waits for the move (each screen shown 3 network steps) removes
  the lag: from 16 episodes every run plays exactly like the expert (14.28
  laps, against 2.74 with the lag). Outputs that read the letter directly
  (h = 2) cannot be one neuron; TRAIN memorises and play collapses.

**Noisy labels (experiment 29).**
- With 5–20% wrong training labels, every model fits every wrong label. On
  images the discrete networks lose about as much as an MLP (nbr3_upstart
  0.913 → 0.760 at p = 0.1, 32 streams; MLP 0.948 → 0.840), absorbing the
  noise in extra helpers. In the game, exact merging finds no machine at
  all from 24 episodes at p = 0.05.

**Noise-tolerant training (experiment 30).**
- Game: finding the letter and machine by search (fewest disagreements,
  majority outputs, number of states by held-out episodes), then
  relabelling, recovers the expert in every run up to 20% wrong moves (true
  machine 78 of 80 at p = 0.2, all play at 14.28 laps). Only for tiny
  machines: the search is exhaustive.
- Images: a 4-fold disagreement filter recovers a fifth to a half of the
  loss at 16–32 streams (nbr3_upstart 0.808 against 0.760 at p = 0.1) but
  hurts with few streams and on clean data. The filtered MLP stays ahead.

**Success and failure without an expert (experiment 31).**
- Lucas's idea 3, first form: buffer successful games, redraw weights
  inside the boxes on failure. It does not learn (0.17 laps after 50 rounds,
  random 1.45): the buffered game fixes every move, so redrawing inside the
  boxes never changes the play.

**A fairer gradient baseline (experiment 32).**
- Tuning the RNN on held-out streams and adding a GRU barely closes the
  gap: whole unseen streams right at the largest size 0.24 (state) and 0.45
  (game) for the tuned GRU, against 0.99 and 1.00 for the discrete networks.
  Per step the gradient models do better (0.75 and 0.87) but stay behind
  from 4 streams on. The discrete networks still get designed letters.

**Deviations from the specification.**
- Experiment 7 runs at reduced sizes (section 7).
- The feature search runs for each visible neuron when the worklist reaches
  it, and neurons already passed are not revisited.
- Duplicate and silent candidates are skipped, and ties go to the first
  candidate in a fixed order. This does not change which slack values are
  reachable.
- For two-step rule 90 the unseen set is every one of the 64 states not used
  for training, rather than 100 random states.
