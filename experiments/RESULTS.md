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
