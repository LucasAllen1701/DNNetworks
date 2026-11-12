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
