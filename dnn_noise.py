"""Noisy labels (experiments 29-31).

- `flip_u`, `noisy`: per-pattern label noise. A pattern's label is flipped
  when a hash of (pattern, seed) falls below p, so the same pattern always
  gets the same label and the flipped sets are nested in p.
- `fit_machine`: the machine with n states over k-bit letters that
  disagrees least with the training outputs. Exact merging (dnn_merge)
  fails on noisy streams because no machine fits them all; here every
  transition table is tried (when there are at most 4096) or improved by
  local search from random starts, each (state, letter) cell outputs the
  majority of the outputs seen there, and the cost is the number of
  outputs that disagree. Start state 0.
- `learn_machine_heldout`: the number of states (1 to max_states) chosen by
  the disagreements on held-out streams (every 4th; the second of 2 or 3),
  then refitted on all streams.
- `choose_bits_search`: dnn_letters.choose_bits for noisy streams: the
  letter (at most 2 of the features) whose best machine of at most 2 states
  disagrees least (ties: fewer bits, earlier set); then its machine by
  `learn_machine_heldout`.
- `relabel`: training streams with each output replaced by the machine's.

Greedy merging with a vote (counts in the prefix tree, a tolerance or
ALERGIA's Hoeffding test for conflicts) was tried first and recovered the
true machine of experiment 29's game in few runs even without noise: a
wrong early merge cannot be undone.
"""
import hashlib
from itertools import combinations, product

import numpy as np

import dnn_upstart as U


def flip_u(x, seed):
    """A number in [0, 1) fixed by the pattern x and the seed."""
    d = hashlib.sha256(np.asarray(x, np.int64).tobytes() + int(seed).to_bytes(8, "little")).digest()
    return int.from_bytes(d[:8], "little") / 2 ** 64


def noisy(f, p, seed):
    """f with the label of each pattern flipped when flip_u < p."""
    if p == 0:
        return f
    return lambda x: int(f(x)) ^ int(flip_u(x, seed) < p)


def _encode(seqs, k, n_out, h):
    """(letters, outputs) as int arrays per stream from bit streams of k
    letter columns and n_out output columns; output t + h answers letter t."""
    S, Y = [], []
    for s in seqs:
        s = np.asarray(s, int)
        S.append(s[:len(s) - h, :k] @ (1 << np.arange(k)[::-1]))
        Y.append(s[h:, k:k + n_out] @ (1 << np.arange(n_out)[::-1]))
    return S, Y


def _costs(D, S, Y, n_lab):
    """Disagreements of each transition table in D (B x n x L) with majority
    outputs, and the counts (B x n*L x n_lab)."""
    B, n, L = D.shape
    counts = np.zeros(B * n * L * n_lab, int)
    rows = np.arange(B)[:, None]
    for s, y in zip(S, Y):
        if len(s) == 0:
            continue
        q = np.zeros((B, 1), int)
        cells = []
        for a in s:
            cells.append(q * L + a)
            q = D[rows, q, a]
        c = np.concatenate(cells, 1)                                    # B x steps
        flat = (rows * (n * L) + c) * n_lab + y[None, :]
        counts += np.bincount(flat.ravel(), minlength=len(counts))
    counts = counts.reshape(B, n * L, n_lab)
    return counts.sum((1, 2)) - counts.max(2).sum(1), counts


def _local_search(S, Y, n, L, n_lab, rng, starts=8):
    best = None
    for _ in range(starts):
        D = rng.integers(0, n, (n, L))
        cost = _costs(D[None], S, Y, n_lab)[0][0]
        improved = True
        while improved:
            improved = False
            cands = []
            for i, a, v in product(range(n), range(L), range(n)):
                if D[i, a] != v:
                    E = D.copy()
                    E[i, a] = v
                    cands.append(E)
            c = _costs(np.array(cands), S, Y, n_lab)[0]
            j = int(np.argmin(c))
            if c[j] < cost:
                D, cost, improved = cands[j], c[j], True
        if best is None or cost < best[1]:
            best = (D, cost)
    return best


def fit_machine(seqs, k, n_out, h, n, seed=0):
    """(delta, out, disagreements) of the n-state machine over k-bit letters
    that disagrees least with the outputs of the bit streams `seqs`. delta
    and out are lists of dicts over letter tuples, as dnn_merge's."""
    S, Y = _encode(seqs, k, n_out, h)
    L, n_lab = 1 << k, 1 << n_out
    if n ** (n * L) <= 4096:
        tables = np.array(list(product(range(n), repeat=n * L))).reshape(-1, n, L)
        cost = _costs(tables, S, Y, n_lab)[0]
        j = int(np.argmin(cost))
        D, err = tables[j], int(cost[j])
    else:
        D, err = _local_search(S, Y, n, L, n_lab, np.random.default_rng(seed))
        err = int(err)
    counts = _costs(D[None], S, Y, n_lab)[1][0]
    letters = list(product((0, 1), repeat=k))
    bits = lambda v: tuple(int(b) for b in format(v, f"0{n_out}b"))
    delta = [{a: int(D[q, i]) for i, a in enumerate(letters)} for q in range(n)]
    out = [{a: bits(int(np.argmax(counts[q * L + i]))) for i, a in enumerate(letters)} for q in range(n)]
    return delta, out, err


def machine_errors(seqs, k, n_out, h, delta, out):
    """Outputs of the bit streams `seqs` that the machine gets wrong."""
    err = 0
    for s in seqs:
        s = np.asarray(s, int)
        q = 0
        for t in range(len(s) - h):
            a = tuple(int(v) for v in s[t, :k])
            err += out[q][a] != tuple(int(v) for v in s[t + h, k:k + n_out])
            q = delta[q][a]
    return err


def learn_machine_heldout(seqs, k, n_out, h, max_states=3, seed=0):
    """(delta, out, disagreements on all streams, states): the number of
    states with the fewest held-out disagreements (ties: fewer), refitted on
    all streams."""
    m = len(seqs)
    held = [i for i in range(m) if (i % 4 == 3 if m >= 4 else i == 1)]
    fit = [s for i, s in enumerate(seqs) if i not in held]
    best = None
    for n in range(1, max_states + 1):
        delta, out, _ = fit_machine(fit, k, n_out, h, n, seed)
        e = machine_errors([seqs[i] for i in held], k, n_out, h, delta, out)
        if best is None or e < best[1]:
            best = (n, e)
    delta, out, err = fit_machine(seqs, k, n_out, h, best[0], seed)
    return delta, out, err, best[0]


def choose_bits_search(V, n_in, features, h, max_bits=2, n_out=1, max_states=3, seed=0):
    """(indices, delta, out, disagreements, states) for noisy streams."""
    X = [np.asarray(s, int)[:, :n_in] for s in V]
    cols = [[U.fire(r, x) for x in X] for _, r in features]
    best = None
    for k in range(1, max_bits + 1):
        for idx in combinations(range(len(features)), k):
            seqs = [np.c_[np.array([cols[i][j] for i in idx]).T, np.asarray(s)[:, n_in:n_in + n_out]]
                    for j, s in enumerate(V)]
            err = min(fit_machine(seqs, k, n_out, h, n, seed)[2] for n in (1, 2))
            if best is None or err < best[0]:
                best = (err, idx, seqs)
    _, idx, seqs = best
    delta, out, err, n = learn_machine_heldout(seqs, len(idx), n_out, h, max_states, seed)
    return idx, delta, out, err, n


def relabel(V, n_in, n_out, h, bits, delta, out):
    """Streams with output t + h replaced by the machine's answer to the
    letters 0..t (bits: functions of an array of patterns). Outputs before h
    are kept."""
    new = []
    for s in V:
        s = np.array(s, int)
        letters = list(zip(*[b(s[:, :n_in]) for b in bits]))
        q = 0
        for t in range(len(s) - h):
            a = tuple(int(v) for v in letters[t])
            if a not in delta[q]:
                break
            s[t + h, n_in:n_in + n_out] = out[q][a]
            q = delta[q][a]
        new.append(s)
    return new


# ---- experiment 36: growing the machine by splitting states

def _to_machine(D, S, Y, k, n_out):
    """(delta, out) for transition table D with majority outputs."""
    n, L = D.shape
    counts = _costs(D[None], S, Y, 1 << n_out)[1][0]
    letters = list(product((0, 1), repeat=k))
    bits = lambda v: tuple(int(b) for b in format(v, f"0{n_out}b"))
    delta = [{a: int(D[q, i]) for i, a in enumerate(letters)} for q in range(n)]
    out = [{a: bits(int(np.argmax(counts[q * L + i]))) for i, a in enumerate(letters)} for q in range(n)]
    return delta, out


def _hill(D, cost, S, Y, n_lab):
    """Best single-cell change of the transition table while it lowers the
    disagreements (ties: the first change in (state, letter, target) order)."""
    n, L = D.shape
    while True:
        cands = []
        for i, a, v in product(range(n), range(L), range(n)):
            if D[i, a] != v:
                E = D.copy()
                E[i, a] = v
                cands.append(E)
        if not cands:
            return D, cost
        c = _costs(np.array(cands), S, Y, n_lab)[0]
        j = int(np.argmin(c))
        if c[j] >= cost:
            return D, cost
        D, cost = cands[j], c[j]


def split_tables(S, Y, L, n_lab, max_states):
    """Transition tables of 1 .. max_states states grown by splitting: from
    the n-state table, every (state, letter) transition is in turn sent to a
    new state whose row either copies the row of the old target or stays in
    the new state; each such split is improved by _hill and the one with the
    fewest disagreements is kept (ties: the first). Returns [(D, cost)]."""
    D = np.zeros((1, L), int)
    out = [(D, _costs(D[None], S, Y, n_lab)[0][0])]
    for n in range(2, max_states + 1):
        cands = []
        for q, a in product(range(n - 1), range(L)):
            for copy in (True, False):
                E = np.zeros((n, L), int)
                E[:n - 1] = D
                E[n - 1] = D[D[q, a]] if copy else n - 1
                E[q, a] = n - 1
                cands.append(E)
        c = _costs(np.array(cands), S, Y, n_lab)[0]
        best = None
        for E, ce in zip(cands, c):
            F, cf = _hill(E, ce, S, Y, n_lab)
            if best is None or cf < best[1]:
                best = (F, cf)
        D = best[0]
        out.append(best)
    return out


def learn_machine_split(seqs, k, n_out, h, max_states=5):
    """learn_machine_heldout with split_tables instead of the exhaustive or
    local search: (delta, out, disagreements on all streams, states)."""
    m = len(seqs)
    held = [i for i in range(m) if (i % 4 == 3 if m >= 4 else i == 1)]
    fit = [s for i, s in enumerate(seqs) if i not in held]
    L, n_lab = 1 << k, 1 << n_out
    Sf, Yf = _encode(fit, k, n_out, h)
    best = None
    for n, (D, _) in enumerate(split_tables(Sf, Yf, L, n_lab, max_states), 1):
        delta, out = _to_machine(D, Sf, Yf, k, n_out)
        e = machine_errors([seqs[i] for i in held], k, n_out, h, delta, out)
        if best is None or e < best[1]:
            best = (n, e)
    S, Y = _encode(seqs, k, n_out, h)
    D, err = split_tables(S, Y, L, n_lab, best[0])[-1]
    delta, out = _to_machine(D, S, Y, k, n_out)
    return delta, out, int(err), best[0]
