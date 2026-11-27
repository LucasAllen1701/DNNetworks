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
        for j, s in enumerate(V):
            for t in range(len(s) - 1):
                m = len(c)
                nL = np.full((m + 1, m + 1), -INF)
                nU = np.full((m + 1, m + 1), INF)
                nL[:m, :m], nU[:m, :m] = L, U
                for i in range(m):
                    if c[i] != 0 and c[i] != r:
                        nU[i, m] = 0.0          # older hidden neurons: only inhibited
                    if c[i] != r:
                        nU[m, i] = 0.0          # new neuron: excited by its own round only
                L, U = nL, nU
                c.append(r)
                col = lambda k, q: ((np.arange(len(q)) == t) & (k == j)).astype(int)[:, None]
                seqs = [np.hstack([q, col(k, q)]) for k, q in enumerate(seqs)]
                refit.append(m)
    for g in refit:
        assert feas(seqs, g, L, U, eps), f"neuron {g} infeasible after clocks"
    for g in refit:
        L[g], U[g] = subbox(seqs, g, L, U, eps)
    net.L, net.U, net.n = L, U, len(c)
    return seqs


def _gap(a, b, c, d):
    """Distance between intervals [a,b] and [c,d] (0 if they meet)."""
    with np.errstate(invalid="ignore"):
        g = np.maximum(0.0, np.maximum(c - b, a - d))
    return np.nan_to_num(g, nan=0.0, posinf=INF)


def _meet(a, b, c, d):
    """Interval where [a,b] and [c,d] meet, after stretching each just enough."""
    lo, hi = max(a, c), min(b, d)
    return (lo, hi) if lo <= hi else (hi, lo)


def merge_costs(L, U, keys, n_visible):
    """Matrix of merge costs for every pair of hidden neurons (INF on the
    diagonal and for pairs whose key values differ)."""
    n = L.shape[0]
    H = list(range(n_visible, n))
    h = len(H)
    cost = np.full((h, h), INF)
    if h < 2:
        return cost
    Lh, Uh = L[H], U[H]                                     # rows: incoming boxes
    G = _gap(Lh[:, None, :], Uh[:, None, :], Lh[None, :, :], Uh[None, :, :])
    idx = np.arange(h)
    G[idx[:, None], idx[None, :], np.array(H)[:, None]] = 0.0   # skip axes i and j
    G[idx[:, None], idx[None, :], np.array(H)[None, :]] = 0.0
    inc = G.max(axis=2)
    # combined axis: [L_ii + L_ij, U_ii + U_ij] against [L_ji + L_jj, U_ji + U_jj]
    Hs = np.array(H)
    Ls, Us = L[np.ix_(Hs, Hs)], U[np.ix_(Hs, Hs)]
    dL, dU = np.diag(Ls), np.diag(Us)
    ci_lo, ci_hi = dL[:, None] + Ls, dU[:, None] + Us          # row i: own axis + axis j
    cj_lo, cj_hi = ci_lo.T, ci_hi.T
    with np.errstate(invalid="ignore"):
        comb = _gap(ci_lo, ci_hi, cj_lo, cj_hi)
    cost = np.maximum(inc, comb)
    K = np.asarray(keys)[:, H]
    differ = (K[:, :, None] != K[:, None, :]).any(axis=0)
    cost[differ] = INF
    np.fill_diagonal(cost, INF)
    return cost


def merge(L, U, keys, i, j):
    """Merge neurons i and j (indices into L, U); the new neuron is last."""
    n = L.shape[0]
    rest = [m for m in range(n) if m not in (i, j)]
    nL = np.full((n - 1, n - 1), -INF)
    nU = np.full((n - 1, n - 1), INF)
    nL[:-1, :-1] = L[np.ix_(rest, rest)]
    nU[:-1, :-1] = U[np.ix_(rest, rest)]
    nL[:-1, -1] = L[rest, i] + L[rest, j]                    # weights from l: sums
    nU[:-1, -1] = U[rest, i] + U[rest, j]
    for a, m in enumerate(rest):                             # weights into l: meet
        nL[-1, a], nU[-1, a] = _meet(L[i, m], U[i, m], L[j, m], U[j, m])
    nL[-1, -1], nU[-1, -1] = _meet(L[i, i] + L[i, j], U[i, i] + U[i, j],
                                   L[j, i] + L[j, j], U[j, i] + U[j, j])
    K = np.asarray(keys)
    nK = np.hstack([K[:, rest], K[:, [i]]])
    return nL, nU, nK


def sleep_merge(net, keys, delta):
    """Merge hidden neurons greedily while the cheapest merge costs <= delta.
    Modifies net (c of a merged neuron: that of i); returns (new keys, merges)."""
    L, U, K = net.L.copy(), net.U.copy(), np.asarray(keys).copy()
    c = list(net.c)
    merges = 0
    while True:
        cost = merge_costs(L, U, K, net.n_visible)
        if cost.size == 0 or not np.isfinite(cost.min()) or cost.min() > delta:
            break
        a, b = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = net.n_visible + a, net.n_visible + b
        L, U, K = merge(L, U, K, i, j)
        c = [c[m] for m in range(len(c)) if m not in (i, j)] + [c[i]]
        merges += 1
    net.L, net.U, net.c, net.n = L, U, c, L.shape[0]
    return K, merges
