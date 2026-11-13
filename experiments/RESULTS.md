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

**Deviations from the specification.**
- Experiment 7 runs at reduced sizes (section 7).
- The feature search runs for each visible neuron when the worklist reaches
  it, and neurons already passed are not revisited.
- Duplicate and silent candidates are skipped, and ties go to the first
  candidate in a fixed order. This does not change which slack values are
  reachable.
- For two-step rule 90 the unseen set is every one of the 64 states not used
  for training, rather than 100 random states.
