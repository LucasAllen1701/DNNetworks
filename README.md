# DNNetworks

Discrete neural networks: networks of step-activation neurons that learn
sequences by solving linear feasibility problems and adding neurons when a
sequence does not fit.

## Contents

- `docs/DiscreteNeuralNetworkNotes151126.md` — definitions, the RUN and
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
- `dnn_upstart.py`, `tests/test_dnn_upstart.py` — helper neurons built from
  the data for Boolean functions of the inputs: upstart-style helpers, a
  covering search, and a second hidden layer.
- `dnn_sleep.py`, `tests/test_dnn_sleep.py` — sleep (rebuild from a buffer
  of streams), safe deletion, and policies for which streams leave a bounded
  buffer.
- `dnn_letters.py`, `tests/test_dnn_letters.py` — state neurons that read
  helpers, so that a finite-state task on several input bits generalises
  across input patterns.
- `experiments/` — experiment scripts; findings are in
  `experiments/RESULTS.md`.
- `dashboard.py` — a local web page for running the experiments (see below).

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

## Running the experiments on your own computer

`dashboard.py` is a small local web page for running the experiments. It needs
Python 3 and git, and nothing beyond what the experiments already use.

On Arch Linux:

```
git clone https://github.com/LucasAllen1701/DNNetworks.git
cd DNNetworks
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dashboard.py
```

(Arch does not let `pip` install into the system Python, which is why the
commands use a virtual environment in `.venv`. The alternative is
`sudo pacman -S python-numpy python-scipy python-pytest` and then
`python dashboard.py`.)

The page opens at http://127.0.0.1:8765. Press Ctrl+C in the terminal to stop
it.

1. Click **Check the setup** and **Run**. This runs the test suite and takes
   about a minute. If it finishes with a tick, the installation works.
2. Click an experiment and **Run**. Experiments take from a few seconds to
   several minutes; after the first run the page says how long the last one
   took. Most print nothing until they finish.
3. The **Summary** tab compares the results with the files committed in
   `experiments/results/` and gives a short report to copy. **Output** shows
   what the script printed, **Tables** the result files, and **Chart** the
   mean of any numeric column.

Each run is kept in `runs/<time>_<experiment>/` with its log, its result files
and its report. The committed results are never overwritten, and `runs/` is
ignored by git. To get new experiments, run `git pull` in the folder.

The dashboard listens on this computer only and can start nothing but the
experiment scripts and the test suite.

## Status

The weight region of each neuron is stored as a box, so a network with `n`
neurons takes `2n^2 + n` numbers regardless of how many sequences it has been
trained on. Known limitations are listed in section 8 of the notes; the main
one is that, in the worst case, a sequence is memorised in the initial state
of the hidden neurons rather than generalised.

## License

Apache License 2.0. See `LICENSE`.
