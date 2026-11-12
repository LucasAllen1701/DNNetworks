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
