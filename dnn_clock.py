"""Clock-neuron wake round and merge-based sleep (notes of 26 November 2026).

wake_clock: one wake round (docs/DiscreteNeuralNetworkNotes261126.md, section 4.4).
If some visible non-input neuron does not fit, add one clock neuron per vector
of every sequence of the round except its last vector; clock neuron e_{j,t} is
on at time t of sequence j only. Boxes follow ADD (section 4.3), so condition
(D) holds. Returns the extended sequences; row 0 of each is its key.

sleep_merge: Lucas's merge (27 November). Repeatedly merge the pair of hidden
neurons whose boxes need the least stretching to meet, while that stretch is
at most delta. The new neuron's incoming box is the meeting of the two
(stretched) incoming boxes, with the i and j axes combined by adding them
(the two fire together); every other neuron's weight from it is the sum of
its weights from i and j. Two neurons can only merge if they have the same
value in every key. No linear programs are solved.
"""
import numpy as np

from dnn import INF, feas, subbox


def _add(L, U, c, r, seqs, cols, free_from=()):
    """ADD (notes, section 4.3) for one new neuron with the given column in
    each sequence. Weights into it from neurons in free_from are also
    unrestricted (input-driven neurons); everything else follows (D)."""
    m = len(c)
    nL = np.full((m + 1, m + 1), -INF)
    nU = np.full((m + 1, m + 1), INF)
    nL[:m, :m], nU[:m, :m] = L, U
    for i in range(m):
        if c[i] != 0 and c[i] != r:
            nU[i, m] = 0.0              # older hidden neurons: only inhibited
        if c[i] != r and i not in free_from:
            nU[m, i] = 0.0              # new neuron: excited by its own round only
    c.append(r)
    seqs = [np.hstack([q, np.asarray(col, int)[:, None]]) for q, col in zip(seqs, cols)]
    return nL, nU, seqs


def _add_clocks(L, U, c, r, seqs, V, refit):
    for j, s in enumerate(V):
        for t in range(len(s) - 1):
            cols = [((np.arange(len(q)) == t) & (k == j)).astype(int) for k, q in enumerate(seqs)]
            refit.append(len(c))
            L, U, seqs = _add(L, U, c, r, seqs, cols)
    return L, U, seqs


def wake_clock(net, V, eps):
    """One wake round with clock neurons. Modifies net; returns the
    extended sequences (row 0 of each is its key)."""
    net.round += 1
    r = net.round
    nh = net.n - net.n_visible
    seqs = [np.hstack([np.asarray(s, int), np.zeros((len(s), nh), int)]) for s in V]
    L, U, c = net.L, net.U, net.c
    refit = [g for g in range(net.n_visible) if g not in net.I]
    if not all(feas(seqs, g, L, U, eps) for g in refit):
        L, U, seqs = _add_clocks(L, U, c, r, seqs, V, refit)
    for g in refit:
        assert feas(seqs, g, L, U, eps), f"neuron {g} infeasible after clocks"
    for g in refit:
        L[g], U[g] = subbox(seqs, g, L, U, eps)
    net.L, net.U, net.n = L, U, len(c)
    return seqs


def wake_tree(net, V, eps):
    """One wake round with input-driven clock neurons (a tree of input
    histories). If some visible non-input neuron does not fit, add one neuron
    for each distinct input history x(0..t) of the round's sequences, on at
    time t+1 in exactly the sequences with that history (t + 1 before the
    last vector). Outputs must answer the inputs up to t at time t + 2
    (depth 2), and a start input that is on only at time 0 gives the first
    neurons something to start from.
    These neurons may be excited by the input neurons, so they are not
    dormant: condition (D) does not hold for them, and they fire without a
    key. If a visible neuron still does not fit (two sequences with the same
    inputs need different outputs), keyed clock neurons are added as well.
    Returns (extended sequences, number of tree neurons, keyed clocks added?)."""
    net.round += 1
    r = net.round
    nh = net.n - net.n_visible
    I = sorted(net.I)
    seqs = [np.hstack([np.asarray(s, int), np.zeros((len(s), nh), int)]) for s in V]
    L, U, c = net.L, net.U, net.c
    refit = [g for g in range(net.n_visible) if g not in net.I]
    nodes, fallback = 0, False
    if not all(feas(seqs, g, L, U, eps) for g in refit):
        hist = {}                                   # history -> times on, per sequence
        for j, s in enumerate(V):
            s = np.asarray(s)
            for t in range(len(s) - 2):             # on at t + 1, which must not be the last vector
                h = s[:t + 1][:, I].tobytes() + bytes([t])
                hist.setdefault(h, []).append((j, t + 1))
        order = sorted(hist, key=lambda h: (h[-1], h))   # shorter histories first
        for h in order:
            on = set(hist[h])
            cols = [np.array([(k, u) in on for u in range(len(q))], int) for k, q in enumerate(seqs)]
            refit.append(len(c))
            L, U, seqs = _add(L, U, c, r, seqs, cols, free_from=set(I))
            nodes += 1
        if not all(feas(seqs, g, L, U, eps) for g in refit[:len(refit) - nodes]):
            fallback = True
            L, U, seqs = _add_clocks(L, U, c, r, seqs, V, refit)
    for g in refit:
        assert feas(seqs, g, L, U, eps), f"neuron {g} infeasible"
    for g in refit:
        L[g], U[g] = subbox(seqs, g, L, U, eps)
    net.L, net.U, net.n = L, U, len(c)
    return seqs, nodes, fallback


def _gap(a, b, c, d):
    """Distance between intervals [a,b] and [c,d] (0 if they meet)."""
    with np.errstate(invalid="ignore"):
        g = np.maximum(0.0, np.maximum(c - b, a - d))
    return np.nan_to_num(g, nan=0.0, posinf=INF)


def _meet(a, b, c, d):
    """Interval where [a,b] and [c,d] meet, after stretching each just enough."""
    lo, hi = max(a, c), min(b, d)
    return (lo, hi) if lo <= hi else (hi, lo)


def merge_costs(L, U, keys, n_visible, combine="sum"):
    """Matrix of merge costs for every pair of hidden neurons (INF on the
    diagonal and for pairs whose key values differ).

    combine="sum": the two neurons fire together, so weights from the merged
    neuron are sums and give no condition. combine="meet": they never fire
    together, so every other neuron's weights from i and from j must meet."""
    n = L.shape[0]
    H = list(range(n_visible, n))
    h = len(H)
    cost = np.full((h, h), INF)
    if h < 2:
        return cost
    Hs = np.array(H)
    idx = np.arange(h)
    Lh, Uh = L[H], U[H]                                     # rows: incoming boxes
    G = _gap(Lh[:, None, :], Uh[:, None, :], Lh[None, :, :], Uh[None, :, :])
    G[idx[:, None], idx[None, :], Hs[:, None]] = 0.0       # skip axes i and j
    G[idx[:, None], idx[None, :], Hs[None, :]] = 0.0
    cost = G.max(axis=2)
    Ls, Us = L[np.ix_(Hs, Hs)], U[np.ix_(Hs, Hs)]
    dL, dU = np.diag(Ls), np.diag(Us)
    if combine == "sum":
        # combined axis: [L_ii + L_ij, U_ii + U_ij] against [L_ji + L_jj, U_ji + U_jj]
        ci_lo, ci_hi = dL[:, None] + Ls, dU[:, None] + Us
        comb = _gap(ci_lo, ci_hi, ci_lo.T, ci_hi.T)
    else:
        # combined axis: all four intervals L_ii, L_ij, L_ji, L_jj must meet
        lo = np.maximum(np.maximum(dL[:, None], Ls), np.maximum(Ls.T, dL[None, :]))
        hi = np.minimum(np.minimum(dU[:, None], Us), np.minimum(Us.T, dU[None, :]))
        with np.errstate(invalid="ignore"):
            comb = np.nan_to_num(np.maximum(0.0, lo - hi), nan=0.0, posinf=INF)
        Lc, Uc = L[:, H], U[:, H]                          # columns: outgoing weights
        G2 = _gap(Lc[:, :, None], Uc[:, :, None], Lc[:, None, :], Uc[:, None, :])
        G2[Hs[:, None], idx[:, None], idx[None, :]] = 0.0  # skip rows i and j
        G2[Hs[None, :], idx[:, None], idx[None, :]] = 0.0
        cost = np.maximum(cost, G2.max(axis=0))
    cost = np.maximum(cost, comb)
    K = np.asarray(keys)[:, H]
    differ = (K[:, :, None] != K[:, None, :]).any(axis=0)
    cost[differ] = INF
    np.fill_diagonal(cost, INF)
    return cost


def merge(L, U, keys, i, j, combine="sum"):
    """Merge neurons i and j (indices into L, U); the new neuron is last."""
    n = L.shape[0]
    rest = [m for m in range(n) if m not in (i, j)]
    nL = np.full((n - 1, n - 1), -INF)
    nU = np.full((n - 1, n - 1), INF)
    nL[:-1, :-1] = L[np.ix_(rest, rest)]
    nU[:-1, :-1] = U[np.ix_(rest, rest)]
    if combine == "sum":                                     # weights from l: sums
        nL[:-1, -1] = L[rest, i] + L[rest, j]
        nU[:-1, -1] = U[rest, i] + U[rest, j]
        nL[-1, -1], nU[-1, -1] = _meet(L[i, i] + L[i, j], U[i, i] + U[i, j],
                                       L[j, i] + L[j, j], U[j, i] + U[j, j])
    else:                                                    # weights from l: meet
        for a, k in enumerate(rest):
            nL[a, -1], nU[a, -1] = _meet(L[k, i], U[k, i], L[k, j], U[k, j])
        lo = max(L[i, i], L[i, j], L[j, i], L[j, j])
        hi = min(U[i, i], U[i, j], U[j, i], U[j, j])
        nL[-1, -1], nU[-1, -1] = (lo, hi) if lo <= hi else (hi, lo)
    for a, m in enumerate(rest):                             # weights into l: meet
        nL[-1, a], nU[-1, a] = _meet(L[i, m], U[i, m], L[j, m], U[j, m])
    K = np.asarray(keys)
    nK = np.hstack([K[:, rest], K[:, [i]]])
    return nL, nU, nK


def sleep_merge(net, keys, delta, combine="sum"):
    """Merge hidden neurons greedily while the cheapest merge costs <= delta.
    Modifies net (c of a merged neuron: that of i); returns (new keys, merges)."""
    L, U, K = net.L.copy(), net.U.copy(), np.asarray(keys).copy()
    c = list(net.c)
    merges = 0
    while True:
        cost = merge_costs(L, U, K, net.n_visible, combine)
        if cost.size == 0 or not np.isfinite(cost.min()) or cost.min() > delta:
            break
        a, b = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = net.n_visible + a, net.n_visible + b
        L, U, K = merge(L, U, K, i, j, combine)
        c = [c[m] for m in range(len(c)) if m not in (i, j)] + [c[i]]
        merges += 1
    net.L, net.U, net.c, net.n = L, U, c, L.shape[0]
    return K, merges
