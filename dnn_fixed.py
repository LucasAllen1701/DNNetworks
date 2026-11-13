"""Fixed neurons: visible neurons with fixed weights, present before round 1.

A fixed neuron is a visible, non-input neuron whose weight box is a single
point with integer entries on the visible neurons. Its value at time t + 1 is
the step function of its weighted sum at time t, like any other neuron, so
RUN needs no change. Training is `dnn.train`, unchanged, on sequences that
have been extended with the fixed neurons' values. docs/Fixed_Neurons.md
proves that the Theorem of the notes still holds and that the fixed neurons
compute their rule on any run started with the hidden neurons at 0.

The fixed neurons built here are:
- past shifts: neuron (i, s) has the value of neuron i from s steps earlier,
  and 0 during the first s steps;
- a start clock: past shifts of an always-on input neuron;
- twins: neuron (i, s) is 1 exactly when neuron i was 0 s steps earlier, and 0
  during the first s steps. They need the always-on input neuron.
"""
import numpy as np

import dnn


class Layout:
    """The visible neurons of a network with fixed neurons.

    Indices 0 ... n_task - 1 are the task's own neurons, unchanged. Then comes
    the always-on input neuron (if any), then the fixed neurons.
    `weights[m]` is the fixed weight vector of fixed neuron m over the visible
    neurons, and `names[m]` describes every visible neuron.
    """

    def __init__(self, n_task, inputs=(), outputs=()):
        self.n_task = n_task
        self.inputs = set(inputs)
        self.outputs = set(outputs)
        self.names = [f"v{i}" for i in range(n_task)]
        self.on = None
        self._rows = {}

    @property
    def n_visible(self):
        return len(self.names)

    @property
    def fixed(self):
        return sorted(self._rows)

    @property
    def weights(self):
        """{fixed neuron: weight vector over all visible neurons}."""
        out = {}
        for m, row in self._rows.items():
            w = np.zeros(self.n_visible)
            for j, value in row.items():
                w[j] = value
            out[m] = w
        return out

    def add_on(self):
        """Add the always-on input neuron. Must come before any fixed neuron."""
        assert not self._rows and self.on is None
        self.on = self.n_visible
        self.names.append("on")
        self.inputs.add(self.on)
        return self.on

    def add_fixed(self, row, name):
        """Add a fixed neuron with integer weights `row` = {source: weight}."""
        assert all(float(w).is_integer() for w in row.values())
        m = self.n_visible
        self.names.append(name)
        self._rows[m] = dict(row)
        return m


def make_layout(n_task, inputs=(), outputs=(), depth=0, sources=None, extra="none"):
    """Layout with past shifts of depth `depth` of the neurons in `sources`
    (default: every task neuron).

    extra: "none"; "clock" adds the always-on input and its shifts; "twins"
    adds the always-on input and a twin for every shift neuron.
    """
    assert extra in ("none", "clock", "twins")
    lay = Layout(n_task, inputs, outputs)
    sources = list(range(n_task)) if sources is None else list(sources)
    if extra != "none":
        lay.add_on()
    prev = {i: i for i in sources}
    prev_twin = {}
    prev_on = lay.on
    for s in range(1, depth + 1):
        for i in sources:
            prev[i] = lay.add_fixed({prev[i]: 1}, f"v{i}-{s}")
        if extra == "clock":
            prev_on = lay.add_fixed({prev_on: 1}, f"on-{s}")
        if extra == "twins":
            for i in sources:
                row = {lay.on: 1, i: -1} if s == 1 else {prev_twin[i]: 1}
                prev_twin[i] = lay.add_fixed(row, f"not v{i}-{s}")
    return lay


def make_net(layout):
    """An untrained network in which every fixed neuron's box is its point."""
    net = dnn.Net(layout.n_visible, layout.inputs, layout.outputs)
    for m, w in layout.weights.items():
        net.L[m] = w
        net.U[m] = w
    return net


def extend(layout, seq):
    """Extend a sequence on the task neurons to all visible neurons.

    The always-on input is 1 throughout. Every fixed neuron is 0 at time 0 and
    then follows its rule, reading the sequence's own values.
    """
    seq = np.asarray(seq, int)
    assert seq.shape[1] == layout.n_task
    v = np.zeros((len(seq), layout.n_visible), int)
    v[:, :layout.n_task] = seq
    if layout.on is not None:
        v[:, layout.on] = 1
    weights = layout.weights
    for t in range(len(seq) - 1):
        for m, w in weights.items():
            v[t + 1, m] = int(w @ v[t] >= 1)
    return v


def train(net, layout, V, eps=0.5):
    """One round of dnn.train on the extended sequences. Returns the extended
    sequences on all neurons; row 0 of each is its key."""
    return dnn.train(net, [extend(layout, s) for s in V], eps)


def start(layout, net, task_state):
    """Initial vector for a run without a key: the given state on the task
    neurons, every fixed and hidden neuron at 0."""
    v0 = np.zeros(net.n, int)
    v0[:layout.n_task] = task_state
    return v0


def fixed_boxes_intact(net, layout):
    """True if every fixed neuron's box is still its point on the visible
    neurons (its weights from hidden neurons may be intervals)."""
    nv = layout.n_visible
    return all(np.array_equal(net.L[m, :nv], w) and np.array_equal(net.U[m, :nv], w)
               for m, w in layout.weights.items())
