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

Added for experiment 20 (learning the letter from data):
- `compose_letter`: the library grown by one step, the OR and the AND of
  every pair of library letters; the consistent letter with the fewest
  merged states wins.
- `joint_search`: the smallest machine and a letter label for every
  training pattern, found together by backtracking so that every training
  output is reproduced (exact identification; no library).
- `learn_letter`: the labels that machine forces are fitted by
  `dnn_upstart.build_cover` (the letter is the OR of its helpers), so the
  letter on unseen patterns comes from helpers built from the data, not
  from a truth table. A library letter only orders the free label choices.

Added for experiment 23 (a tiny grid game):
- letters of several bits, each bit one fixed neuron over the inputs
  (`choose_bits`, `make_layout_bits`). State neuron (q, a) is the AND of
  "in state q" and "the bits equal a", again one threshold, so h = 3.
  Experiment 26 adds several output bits (`n_out`, `learn_machine_multi`).
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


# ---- experiment 20: the letter learned from the data

def _indexed(V, n_in, h=H):
    """Distinct patterns that are answered in the streams (x(t), t < len - h),
    and each stream as (pattern indices, outputs)."""
    index, seqs = {}, []
    for s in V:
        s = np.asarray(s, int)
        word, outs = M.observations(s, n_in, h)
        seqs.append(([index.setdefault(w, len(index)) for w in word], outs))
    return np.array(list(index), int).reshape(-1, n_in), seqs


def conflicts(z, seqs):
    """Number of outputs in the prefix tree of the letter streams (letters
    z[pattern]) that disagree with the majority output on their edge."""
    counts = {}
    for word, outs in seqs:
        node = ()
        for p, o in zip(word, outs):
            node = node + (int(z[p]),)
            counts.setdefault(node, [0, 0])[o] += 1
    return sum(min(c) for c in counts.values())


class Budget(Exception):
    pass


def joint_search(seqs, n_pat, n_states, prefer, budget=200_000):
    """A machine of at most n_states states over the letters 0, 1 together
    with a letter label for every pattern, such that every stream gives its
    observed outputs (exact identification by backtracking). Labels are
    tried in the order `prefer`, then the other; states in order, with at
    most one new state per step (this removes renamings). The steps are taken
    in time order across the streams (step t of every stream before t + 1). Returns (z, delta,
    out) with -1 for anything not used, None if there is no such machine,
    and raises Budget after `budget` search nodes."""
    T = max(len(w) for w, _ in seqs)
    steps = [(j, w[t], outs[t]) for t in range(T) for j, (w, outs) in enumerate(seqs) if t < len(w)]
    z = [-1] * n_pat
    delta = [[-1, -1] for _ in range(n_states)]
    out = [[-1, -1] for _ in range(n_states)]
    cur = [0] * len(seqs)
    count = [0]

    def solve(i, used):
        trail = []
        ok = None
        while i < len(steps):                    # forced steps: no branching
            j, p, o = steps[i]
            q = cur[j]
            a = z[p]
            if a < 0 or delta[q][a] < 0:
                break
            if out[q][a] != o:
                ok = False
                break
            trail.append((j, q))
            cur[j] = delta[q][a]
            i += 1
        if ok is None and i == len(steps):
            return True
        if ok is None:
            count[0] += 1
            if count[0] > budget:
                raise Budget
            j, p, o = steps[i]
            q = cur[j]
            for a in ([z[p]] if z[p] >= 0 else [int(prefer[p]), 1 - int(prefer[p])]):
                if out[q][a] >= 0 and out[q][a] != o:
                    continue
                new_z, new_t = z[p] < 0, delta[q][a] < 0
                z[p] = a
                targets = range(min(used + 1, n_states)) if new_t else [delta[q][a]]
                for r in targets:
                    if new_t:
                        delta[q][a], out[q][a] = r, o
                    cur[j] = r
                    if solve(i + 1, max(used, r + 1)):
                        return True
                    cur[j] = q
                if new_t:
                    delta[q][a] = out[q][a] = -1
                if new_z:
                    z[p] = -1
        for j, q in reversed(trail):
            cur[j] = q
        return False

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * len(steps) + 100))
    if not solve(0, 1):
        return None
    return np.array(z), delta, out


def forced(z, delta, out, seqs):
    """Patterns whose label cannot be flipped alone: with the flip, some
    stream would use a transition the machine does not have or give a wrong
    output."""
    res = np.zeros(len(z), bool)
    for p in range(len(z)):
        if z[p] < 0:
            continue
        zf = z.copy()
        zf[p] ^= 1
        for w, outs in seqs:
            q, bad = 0, False
            for pp, o in zip(w, outs):
                a = zf[pp]
                if delta[q][a] < 0 or out[q][a] != o:
                    bad = True
                    break
                q = delta[q][a]
            if bad:
                res[p] = True
                break
    return res


def _machine(z, seqs):
    """Merged machine of the letter streams (letters keyed by 1-tuples)."""
    return M.merge(*M.prefix_tree([([(int(z[p]),) for p in w], o) for w, o in seqs]))


def cover_letter(X, z):
    """Helpers for labels z on the patterns X by the covering search, with
    the letter's rule over them: the OR of the helpers, or the NOR if the
    complement needs fewer helpers. Returns (rules, (v, c))."""
    X, z = np.asarray(X, int), np.asarray(z, int)
    if len(z) == 0 or z.min() == z.max():          # no labels, or one value: a constant letter
        return [], (np.zeros(0, int), int(z[0]) if len(z) else 0)
    pos, neg = U.build_cover(X, z), U.build_cover(X, 1 - z)
    if len(neg) < len(pos):
        return neg, (-np.ones(len(neg), int), 1)
    return pos, (np.ones(len(pos), int), 0)


def letter_function(rules, rule):
    """The letter computed by helper rules and the letter's rule over them."""
    v, c = rule

    def f(X):
        X = np.asarray(X, int)
        Hv = np.array([U.fire(r, X) for r in rules]).T if rules else np.zeros((len(X), 0), int)
        return U.fire((np.asarray(v, int), int(c)), Hv)
    return f


def learn_letter(V, n_in, h=H, lib=None, max_states=6, budget=300_000):
    """The letter learned from the training streams V.

    1. For n = 1, 2, ... states, search for an n-state machine and labels of
       the training patterns that reproduce every output (`joint_search`);
       free label choices follow the library letter with the fewest
       conflicts. The first n that succeeds wins. A size whose search runs
       out of budget is passed over, so the machine may not be the smallest.
    2. Fit the labels that the machine forces (`forced`) by the covering
       search (`cover_letter`); the letter is the OR of those helpers.
    3. Merge the letter streams of the fitted letter (`dnn_merge`); if they
       are inconsistent, fit every label in place of the forced ones.
    Returns None if no machine of at most max_states states was found within
    the budget, else a dict with name, rules, rule, f, delta, out, states
    searched, forced labels, library letter used for the preferences."""
    X, seqs = _indexed(V, n_in, h)
    lib = lib or library(n_in)
    scores = [(conflicts(np.asarray(f(X), int), seqs), k) for k, (_, f) in enumerate(lib)]
    k0 = min(scores)[1]
    prefer = np.asarray(lib[k0][1](X), int)
    sol, n, budget_hits = None, 0, 0
    for n in range(1, max_states + 1):
        try:
            sol = joint_search(seqs, len(X), n, prefer, budget)
        except Budget:                       # undecided at this size: try the next
            budget_hits += 1
            continue
        if sol is not None:
            break
    if sol is None:
        return None
    z, delta, out = sol
    must = forced(z, delta, out, seqs)
    for fit_on in (must & (z >= 0), z >= 0):
        if not fit_on.any():                # nothing forced: fit every label
            continue
        rules, rule = cover_letter(X[fit_on], z[fit_on])
        f = letter_function(rules, rule)
        try:
            delta2, out2 = _machine(f(X), seqs)
            break
        except ValueError:
            continue
    else:
        return None
    return dict(name=f"search({lib[k0][0]})", rules=rules, rule=rule, f=f, delta=delta2, out=out2,
                states_search=n, budget_hits=budget_hits, labels_forced=int(must.sum()), patterns=len(X))


def compose_letter(V, n_in, h=H, lib=None, beam=None):
    """Letters made of two library letters. Level 1 is the library; level 2
    is the OR and the AND of every pair of library letters (or, with `beam`,
    of each of the `beam` level-1 letters with the fewest conflicts with
    every library letter). Among the consistent
    letters of both levels, the one whose merged machine has the fewest
    states wins (ties: level 1, then the earlier letter). Returns None if
    none is consistent, else (name, f, delta, out, level, consistent)."""
    X, seqs = _indexed(V, n_in, h)
    lib = lib or library(n_in)
    found, ranked, seen = [], [], set()
    for k, (name, f) in enumerate(lib):
        z = np.asarray(f(X), int)
        seen.add(z.tobytes())
        c = conflicts(z, seqs)
        ranked.append((c, k))
        if c == 0:
            found.append((name, f, z, 1))
    for _, k in (sorted(ranked)[:beam] if beam else ((0, k) for k in range(len(lib)))):
        n1, f1 = lib[k]
        for name, g in (lib if beam else lib[k + 1:]):
            for op, fn in (("|", lambda X, a=f1, b=g: a(X) | b(X)), ("&", lambda X, a=f1, b=g: a(X) & b(X))):
                z = np.asarray(fn(X), int)
                if z.tobytes() in seen:
                    continue
                seen.add(z.tobytes())
                if conflicts(z, seqs) == 0:
                    found.append((f"({n1}){op}({name})", fn, z, 2))
    if not found:
        return None
    best = None
    for name, f, z, level in found:
        delta, out = _machine(z, seqs)
        if best is None or len(delta) < len(best[2]):
            best = (name, f, delta, out, level)
    return best + (len(found),)


# ---- experiment 23: letters of several bits

def learn_machine_multi(seqs, n_in, n_out, h=H):
    """dnn_merge.learn_machine for streams with n_out output columns: the
    outputs are tuples."""
    obs = []
    for s in seqs:
        s = np.asarray(s, int)
        steps = len(s) - h
        obs.append(([tuple(int(v) for v in s[t, :n_in]) for t in range(steps)],
                    [tuple(int(v) for v in s[t + h, n_in:n_in + n_out]) for t in range(steps)]))
    return M.merge(*M.prefix_tree(obs))


def bit_streams(V, n_in, bits, n_out=1):
    """Training streams with the inputs replaced by the letter bits (each a
    function of an array of patterns)."""
    return [np.c_[np.array([b(np.asarray(s)[:, :n_in]) for b in bits]).T, np.asarray(s)[:, n_in:n_in + n_out]]
            for s in V]


def choose_bits(V, n_in, features, h=H, max_bits=2, n_out=1):
    """(indices, delta, out, candidates consistent) for the set of at most
    `max_bits` features (name, rule over the inputs) whose letter streams are
    consistent and merge into the fewest states (ties: fewer bits, then the
    earlier set). None if no set is consistent. With n_out > 1 the streams
    have n_out output columns and the machine's outputs are tuples."""
    from itertools import combinations
    X = [np.asarray(s, int)[:, :n_in] for s in V]
    cols = [[U.fire(r, x) for x in X] for _, r in features]
    best, consistent = None, 0
    for k in range(1, max_bits + 1):
        for idx in combinations(range(len(features)), k):
            seqs = [np.c_[np.array([cols[i][j] for i in idx]).T, np.asarray(s)[:, n_in:n_in + n_out]]
                    for j, s in enumerate(V)]
            try:
                delta, out = M.learn_machine(seqs, k, h) if n_out == 1 else learn_machine_multi(seqs, k, n_out, h)
            except ValueError:
                continue
            consistent += 1
            if best is None or len(delta) < len(best[1]):
                best = (idx, delta, out)
    return None if best is None else best + (consistent,)


def choose_bits_fast(V, n_in, features, h=H, max_bits=2, n_out=1):
    """choose_bits with the same result (experiment 33): the feature columns
    and output tuples are computed once per stream, and each candidate's
    letter tuples come from a lookup table instead of array rows."""
    from itertools import combinations
    V = [np.asarray(s, int) for s in V]
    steps = [len(s) - h for s in V]
    cols = [[U.fire(r, s[:, :n_in])[:n] for s, n in zip(V, steps)] for _, r in features]
    if n_out == 1:
        outs = [[int(v) for v in s[h:, n_in]] for s in V]
    else:
        outs = [[tuple(int(v) for v in row) for row in s[h:, n_in:n_in + n_out]] for s in V]
    best, consistent = None, 0
    for k in range(1, max_bits + 1):
        lut = list(product((0, 1), repeat=k))          # code -> tuple, first bit most significant
        for idx in combinations(range(len(features)), k):
            obs = []
            for j in range(len(V)):
                code = cols[idx[0]][j].copy()
                for i in idx[1:]:
                    code = 2 * code + cols[i][j]
                obs.append(([lut[c] for c in code.tolist()], outs[j]))
            try:
                delta, out = M.merge(*M.prefix_tree(obs))
            except ValueError:
                continue
            consistent += 1
            if best is None or len(delta) < len(best[1]):
                best = (idx, delta, out)
    return None if best is None else best + (consistent,)


def make_layout_bits(n_in, bit_rules, delta, n_out=1):
    """Layout for a letter of k bits, each bit one fixed neuron (a rule over
    the inputs). delta: a complete machine over the k-bit tuples. State
    neuron (q, a) = S_q + lin_a - 1 >= 1, where lin_a = sum of (b_i if a_i
    else -b_i) + 1 - |a| is 1 exactly when the bits equal a and at most 0
    otherwise. The n_out output neurons follow the inputs. Returns (layout,
    state neurons {(q, a): index})."""
    lay = F.Layout(n_in + n_out, range(n_in), set(range(n_in, n_in + n_out)))
    lay.add_on()
    started = lay.add_fixed({lay.on: 1}, "started")
    started2 = lay.add_fixed({started: 1}, "started2")
    bits = []
    for k, (wi, bi) in enumerate(bit_rules):
        row = {j: int(wi[j]) for j in range(n_in) if wi[j]}
        if bi:
            row[lay.on] = int(bi)
        bits.append(lay.add_fixed(row, f"bit{k}"))
    letters = sorted(delta[0])
    index = {(q, a): lay.add_fixed({}, f"state {q}, letter {a}") for q in range(len(delta)) for a in letters}
    for (q, a), m in index.items():
        row = {index[p, b]: 1 for p, drow in enumerate(delta) for b, nxt in drow.items() if nxt == q}
        if q == 0:
            row[started] = row.get(started, 0) + 1
            row[started2] = row.get(started2, 0) - 1
        for j, ai in zip(bits, a):
            row[j] = row.get(j, 0) + (1 if ai else -1)
        row[lay.on] = row.get(lay.on, 0) + (1 - sum(a)) - 1
        lay._rows[m] = {j: x for j, x in row.items() if x != 0}
    return lay, index
