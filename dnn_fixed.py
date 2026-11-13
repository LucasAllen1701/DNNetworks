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
  during the first s steps. They need the always-on input neuron;
- a library: every feature template of the causal variant over a set of
  source neurons (`add_library`);
- state neurons of a finite machine (`add_automaton`), see
  docs/Latency_And_State.md.
"""
from itertools import combinations, permutations, product

import numpy as np
from scipy.optimize import linprog

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


def train(net, layout, V, eps=0.5, refit_fixed=False):
    """One round of dnn.train on the extended sequences. Returns the extended
    sequences on all neurons; row 0 of each is its key.

    By default the fixed neurons are left out of TRAIN's worklist and their
    weights from hidden neurons are set to 0. That row always satisfies the
    neuron's constraints (docs/Fixed_Neurons.md, Proposition 1), and every
    other neuron gets the same box as with `refit_fixed=True`, which runs
    TRAIN on the fixed neurons too. Skipping them saves two linear programs
    per fixed neuron per round.
    """
    ext = [extend(layout, s) for s in V]
    if refit_fixed:
        return dnn.train(net, ext, eps)
    fixed = set(layout.fixed)
    net.I |= fixed                      # dnn.train uses I only to choose its worklist
    try:
        out = dnn.train(net, ext, eps)
    finally:
        net.I -= fixed
    nv = layout.n_visible
    for m in fixed:
        net.L[m, nv:] = 0.0
        net.U[m, nv:] = 0.0
    return out


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


# ---- a library of feature templates (those of docs/Causal_Variant.md)

TEMPLATES = ("copy", "latch", "and", "or", "and_not", "set_reset")


def template_row(kind, i, j, me, on):
    """Integer weights of one feature. `me` is the feature's own index."""
    if kind == "copy":
        return {i: 1}
    if kind == "latch":                 # on from the step after i is first on
        return {i: 1, me: 1}
    if kind == "and":
        return {i: 1, j: 1, on: -1}
    if kind == "or":
        return {i: 1, j: 1}
    if kind == "and_not":
        return {i: 1, j: -1}
    if kind == "set_reset":             # set by i, cleared and blocked by j
        return {i: 1, me: 1, j: -2}
    raise ValueError(kind)


def add_library(layout, sources=None, kinds=TEMPLATES, probe_seed=0):
    """Add one feature per template and per source neuron or pair of source
    neurons (default sources: every visible neuron so far). Needs the
    always-on input.

    Features that never fire, or that always equal an existing neuron or an
    earlier feature, are left out. This is decided on random probe sequences,
    not on training data, so the library does not depend on the task.
    Returns the indices of the features added.
    """
    assert layout.on is not None
    sources = list(range(layout.n_visible)) if sources is None else list(sources)
    cands = []
    for kind in kinds:
        if kind in ("copy", "latch"):
            cands += [(kind, i, None) for i in sources]
        elif kind in ("and", "or"):
            cands += [(kind, i, j) for i, j in combinations(sources, 2)]
        else:
            cands += [(kind, i, j) for i, j in permutations(sources, 2)]

    def with_candidates(chosen):
        trial = Layout(layout.n_task, layout.inputs, layout.outputs)
        trial.names, trial.on, trial._rows = list(layout.names), layout.on, dict(layout._rows)
        for kind, i, j in chosen:
            me = trial.n_visible
            name = f"{kind}({layout.names[i]}" + (f", {layout.names[j]})" if j is not None else ")")
            trial.add_fixed(template_row(kind, i, j, me, layout.on), name)
        return trial

    trial = with_candidates(cands)
    rng = np.random.default_rng(probe_seed)
    probes = np.vstack([extend(trial, rng.integers(0, 2, (24, layout.n_task))) for _ in range(40)])
    seen = {probes[:, m].tobytes() for m in range(layout.n_visible)}
    seen.add(np.zeros(len(probes), int).tobytes())
    keep = []
    for k, cand in enumerate(cands):
        key = probes[:, layout.n_visible + k].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(cand)
    final = with_candidates(keep)
    first = layout.n_visible
    layout.names, layout._rows = final.names, final._rows
    return list(range(first, layout.n_visible))


# ---- finite machines (docs/Latency_And_State.md)

def machine_from_task(f, n_in, horizon=None, max_states=64):
    """The smallest machine for a stream task y(t+1) = out(x(0), ..., x(t)).

    f maps an input array of shape (T + 1, n_in) to the outputs y(0..T), where
    y(t) depends on x(0..t-1) only. Two input prefixes are the same state if
    they give the same outputs on every continuation of `horizon` steps.
    Returns (delta, out, y0): delta[q][a] is the next state and out[q][a] the
    output after reading input pattern a (a tuple of bits) in state q. State 0
    is the start state.
    """
    horizon = horizon or (6 if n_in == 1 else 4)
    alphabet = list(product((0, 1), repeat=n_in))
    tails = list(product(alphabet, repeat=horizon))

    def signature(prefix):
        k = len(prefix)
        sig = []
        for tail in tails:
            x = np.array(list(prefix) + list(tail) + [alphabet[0]], int).reshape(-1, n_in)
            sig.append(tuple(int(v) for v in f(x)[k + 1: k + horizon + 1]))
        return tuple(sig)

    states, prefixes, delta, out = {signature(()): 0}, [()], [], []
    q = 0
    while q < len(prefixes):
        delta.append({})
        out.append({})
        for a in alphabet:
            p = prefixes[q] + (a,)
            x = np.array(list(p) + [alphabet[0]], int).reshape(-1, n_in)
            out[q][a] = int(f(x)[len(p)])
            sig = signature(p)
            if sig not in states:
                assert len(prefixes) < max_states, "too many states; is the task finite-state?"
                states[sig] = len(prefixes)
                prefixes.append(p)
            delta[q][a] = states[sig]
        q += 1
    y0 = int(f(np.zeros((1, n_in), int))[0])
    return delta, out, y0


def reachable_states(delta):
    """States reachable from state 0."""
    seen, todo = {0}, [0]
    while todo:
        for nxt in delta[todo.pop()].values():
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def realisable_without_latency(out, delta=None, margin=0.5):
    """True if some weights w on the inputs and offsets c_q satisfy, for every
    state q and input pattern a,  w.a + c_q >= 1 when out[q][a] is 1 and
    w.a + c_q <= 1 - margin when it is 0. This is necessary for any network
    to compute the task without a key at the task's own timing, and
    sufficient given state neurons (docs/Latency_And_State.md). If `delta` is
    given, only the states reachable from state 0 are considered."""
    n_in = len(next(iter(out[0])))
    states = range(len(out)) if delta is None else reachable_states(delta)
    A, b = [], []
    for q, row in enumerate(out):
        if q not in states:
            continue
        for a, y in row.items():
            coef = np.zeros(n_in + len(out))
            coef[:n_in] = a
            coef[n_in + q] = 1
            if y:
                A.append(-coef)
                b.append(-1.0)
            else:
                A.append(coef)
                b.append(1.0 - margin)
    res = linprog(np.zeros(n_in + len(out)), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None)] * (n_in + len(out)))
    return res.status == 0


def add_automaton(layout, delta, input_neurons=None):
    """Add state neurons for the machine `delta` (start state 0). Needs the
    always-on input.

    Neuron (q, a) is 1 at time t + 1 exactly when the machine was in state q
    before reading x(t) and x(t) = a. A one-step clock neuron marks time 0, so
    that the start state needs no key. Returns {(q, a): neuron index}.
    """
    assert layout.on is not None
    inputs = sorted(layout.inputs - {layout.on}) if input_neurons is None else list(input_neurons)
    clock = layout.add_fixed({layout.on: 1}, "started")
    index = {}
    for q, row in enumerate(delta):
        for a in row:
            index[q, a] = layout.add_fixed({}, f"state {q}, read {''.join(map(str, a))}")
    for (q, a), m in index.items():
        row = {index[p, b]: 1 for p, drow in enumerate(delta) for b, nxt in drow.items() if nxt == q}
        for i, bit in zip(inputs, a):
            row[i] = 1 if bit else -1
        row[layout.on] = (1 if q == 0 else 0) - sum(a)
        if q == 0:
            row[clock] = -1
        layout._rows[m] = {j: w for j, w in row.items() if w != 0}
    return index
