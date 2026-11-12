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
  depend on $N$. With $N = 16$, one round used 2.8× (k=16) to 3.6× (k=8) fewer
  neurons than one round per sequence. It was also faster.
- **The Lemma 4 bound is loose.** The largest fraction of the bound used in
  any run was 50% for one round ($k = 2$, $N = 16$: 8 of 16) and 12.5% for one
  round per sequence. At $k = 2$, $N = 16$ one round is close to its cap, which
  is the only place the bound, rather than the data, limits growth.
- Caveat: with 4 visible neurons there are only 16 visible states, so long
  random sequences repeat states with conflicting successors. Most helpers
  resolve those conflicts. With more visible neurons, conflicts would be rarer
  and fewer helpers needed.
