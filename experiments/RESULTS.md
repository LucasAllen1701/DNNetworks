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

## Summary

- No bugs were found in `dnn.py`. Every learning and no-forgetting check
  passed in all four experiments, the Lemma 4 bound always held, and hidden
  cohorts started at 0 always stayed silent, as Lemma 1 says.
- The algorithm does what the Theorem promises — exact recall without
  forgetting — and no more. It generalises only when the task is a bias-free
  threshold function of the visible state (rule 232, previous input, Tomita 1),
  and then only once the training data constrains the visible boxes enough.
- When hidden neurons are needed, they store the answer in the key rather than
  computing it. Started from an unseen state, the hidden neurons stay at 0 (by
  Lemma 1), so they cannot help, and accuracy falls to the majority baseline or
  below.
