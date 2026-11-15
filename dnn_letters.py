"""State neurons that read helpers: finite-state tasks on several input bits.

Experiment 13 merges the prefix tree of the training streams into a machine
whose letters are the raw input patterns. With 6 input bits there are 64
letters, so most transitions are never observed and the machine cannot
generalise across input vectors. Here the machine reads a binary *letter*
l(x(t)) computed from the inputs by helper neurons (experiment 14), and its
state neurons read the helpers in place of the inputs.

Choosing the letter (`choose_letter`). Every candidate in a fixed library
(`library`: each input, each AND of two literals, each XOR of two inputs, each
majority of three inputs) turns the training streams into letter streams.
A candidate is rejected if the outputs are not a function of the letter
history (the prefix tree is inconsistent); otherwise its letter streams are
merged by `dnn_merge`. The candidate whose machine has the fewest states wins
(ties: the earlier candidate, simplest first). The letter functions are fixed
and known, so their helpers are built by `dnn_upstart.build` on the full
truth table; the training data only choose between them.

The network (`make_layout`), at depth h = 3. Inputs, output, the always-on
neuron, "started" (0 at time 0) and "started2" (0 at times 0 and 1); the
helpers, which read the inputs (value at t + 1 a function of x(t)); then one
state neuron per (state q, letter l). Neuron (q, l) is 1 at time t + 2 when
the machine was in q before reading x(t) and l(x(t)) = l. Its rule is the AND
of two conditions that are each one threshold:
  S_q = (sum of the state neurons (p, l') with delta(p, l') = q) plus, for the
        start state, started - started2 (1 only at time 1); S_q is 0 or 1;
  lin = v.H + c >= 1 for letter 1 (the letter's rule over the helpers H), or
        1 - v.H - c >= 1 for letter 0;
as K S_q + lin - K >= 1, with K at least the largest value of lin. Exactly one
state neuron is 1 at every time from 2 on, for every input stream. The output
at t + 3 reads only the state neurons. All of these are fixed neurons, so
docs/Fixed_Neurons.md applies and nothing is forgotten.
"""
from itertools import combinations, product

import numpy as np

import dnn_fixed as F
import dnn_merge as M
import dnn_upstart as U

H = 3


def library(n_in):
    """[(name, f)] with f mapping an array of patterns (k, n_in) to 0/1."""
    lib = [(f"x{i}", lambda X, i=i: X[:, i]) for i in range(n_in)]
    for i, j in combinations(range(n_in), 2):
        for si, sj in product((1, 0), repeat=2):
            name = f"{'' if si else '~'}x{i}&{'' if sj else '~'}x{j}"
            lib.append((name, lambda X, i=i, j=j, si=si, sj=sj: ((X[:, i] == si) & (X[:, j] == sj)).astype(int)))
    for i, j in combinations(range(n_in), 2):
        lib.append((f"x{i}^x{j}", lambda X, i=i, j=j: X[:, i] ^ X[:, j]))
    for c in combinations(range(n_in), 3):
        lib.append((f"maj(x{c[0]},x{c[1]},x{c[2]})", lambda X, c=c: (X[:, list(c)].sum(1) >= 2).astype(int)))
    return lib


def letter_streams(V, n_in, f):
    """Training streams with the inputs replaced by the letter column."""
    return [np.c_[f(np.asarray(s)[:, :n_in]), np.asarray(s)[:, n_in]] for s in V]


def choose_letter(V, n_in, h=H, lib=None):
    """(name, f, delta, out, candidates consistent) for the library letter whose
    merged machine has the fewest states; None if no candidate is consistent.
    delta and out are keyed by 1-tuples (letter,)."""
    best, consistent = None, 0
    for name, f in lib or library(n_in):
        try:
            delta, out = M.learn_machine(letter_streams(V, n_in, f), 1, h)
        except ValueError:
            continue
        consistent += 1
        if best is None or len(delta) < len(best[2]):
            best = (name, f, delta, out)
    return None if best is None else best + (consistent,)


def letter_helpers(f, n_in):
    """Helper rules for the letter f (built on its full truth table) and the
    letter's rule (v, c) over the helper values."""
    X = np.array(list(product((0, 1), repeat=n_in)))
    y = np.asarray(f(X), int)
    if y.min() == y.max():
        return [], (np.zeros(0, int), int(y[0]))
    rules = U.build(X, y)
    Hv = np.array([U.fire(r, X) for r in rules]).T
    v, c = U.integer_rule(*U.best_fit(Hv, y))
    assert (U.fire((v, c), Hv) == y).all()
    return rules, (v, c)


def make_layout(n_in, helper_rules, letter_rule, delta):
    """Layout of the module docstring. delta: a complete machine over the
    letters (0,) and (1,). Returns (layout, state neurons {(q, l): index})."""
    lay = F.Layout(n_in + 1, range(n_in), {n_in})
    lay.add_on()
    started = lay.add_fixed({lay.on: 1}, "started")
    started2 = lay.add_fixed({started: 1}, "started2")
    helpers = []
    for k, (wi, bi) in enumerate(helper_rules):
        row = {j: int(wi[j]) for j in range(n_in) if wi[j]}
        if bi:
            row[lay.on] = int(bi)
        helpers.append(lay.add_fixed(row, f"helper{k}"))
    v, c = letter_rule
    lins = {1: (np.asarray(v, int), int(c)), 0: (-np.asarray(v, int), 1 - int(c))}
    index = {(q, l): lay.add_fixed({}, f"state {q}, letter {l}") for q in range(len(delta)) for l in (0, 1)}
    for (q, l), m in index.items():
        w, b = lins[l]
        K = max(U.gate((w, b)), 1)
        row = {index[p, a[0]]: K for p, drow in enumerate(delta) for a, nxt in drow.items() if nxt == q}
        if q == 0:
            row[started] = row.get(started, 0) + K
            row[started2] = row.get(started2, 0) - K
        for j, wj in zip(helpers, w):
            row[j] = row.get(j, 0) + int(wj)
        row[lay.on] = row.get(lay.on, 0) + b - K
        lay._rows[m] = {j: x for j, x in row.items() if x != 0}
    return lay, index


def same_composite(delta, out, key, true_delta, true_out, true_key, patterns):
    """True if the found machine, reading key(x), gives the true machine's
    output (reading true_key(x)) on every input stream over `patterns`. A
    missing transition counts as a difference."""
    seen, todo = {(0, 0)}, [(0, 0)]
    while todo:
        q, p = todo.pop()
        for x in patterns:
            a, b = key(x), true_key(x)
            if a not in delta[q] or out[q][a] != true_out[p][b]:
                return False
            pair = (delta[q][a], true_delta[p][b])
            if pair not in seen:
                seen.add(pair)
                todo.append(pair)
    return True
