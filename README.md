# DNNetworks

Discrete neural networks: networks of step-activation neurons that learn
sequences by solving linear feasibility problems and adding neurons when a
sequence does not fit.

## Contents

- `docs/Discrete_Neural_Network_Notes_Corrected.md` — definitions, the RUN and
  TRAIN algorithms, and proofs that training learns new sequences without
  forgetting old ones.
- `dnn.py` — reference implementation of both algorithms in Python.
- `tests/test_dnn.py` — tests: the worked examples from the notes, random
  multi-round training with replay, the neuron-count bound, and the dormancy
  condition.
- `dnn_causal.py`, `docs/Causal_Variant.md`, `tests/test_dnn_causal.py` — a
  variant with fixed-weight *feature* neurons computed from the past. They
  work on unseen data, but the variant gives up the no-forgetting guarantee.
- `dnn_fixed.py`, `docs/Fixed_Neurons.md`, `tests/test_dnn_fixed.py` —
  fixed-weight neurons installed before the first round: past shifts of the
  visible neurons, a start clock and twins. They work on unseen data and keep
  the no-forgetting guarantee; `dnn.py` is used unchanged.
- `docs/Latency_And_State.md` — which stream tasks no network can compute at
  the experiments' timing, and how fixed state neurons compute every
  finite-state task one step later.
- `dnn_box.py`, `dnn_buffer.py`, `docs/Box_Rule.md`, `tests/test_dnn_box.py` —
  why a small training round can lock a neuron into a wrong box, a proof that
  no rule for choosing the box avoids it, and a bounded buffer of exact
  constraints that does.
- `dnn_merge.py`, `docs/State_Merging.md`, `tests/test_dnn_merge.py` — finds
  the state neurons of a task from its training streams, by merging the nodes
  of their prefix tree.
- `experiments/` — experiment scripts; findings are in
  `experiments/RESULTS.md`.

## Quick start

```
pip install -r requirements.txt
python -m pytest
```

```python
import numpy as np
from dnn import Net, train, run, sample_weights

net = Net(n_visible=1, outputs={0})
(seq,) = train(net, [np.array([[1], [1], [1], [0]])])   # learn 1,1,1,0
A = sample_weights(net)                                 # any weights from the boxes
states = run(A, net.I, seq[0], [set()] * 4, steps=3)    # replay from the key seq[0]
print([int(v[0]) for v in states])                      # [1, 1, 1, 0]
```

## Status

The weight region of each neuron is stored as a box, so a network with `n`
neurons takes `2n^2 + n` numbers regardless of how many sequences it has been
trained on. Known limitations are listed in section 8 of the notes; the main
one is that, in the worst case, a sequence is memorised in the initial state
of the hidden neurons rather than generalised.

## License

Apache License 2.0. See `LICENSE`.
