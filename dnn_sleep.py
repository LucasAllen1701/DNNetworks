"""Sleep: rebuild the network from a buffer of streams, then delete neurons
that nothing needs.

One wake-and-sleep cycle (experiment 15):
- wake: new streams are trained into the current network by `dnn_fixed.train`
  (state neurons fixed, `dnn.py` unchanged); streams the machine does not fit
  are memorised with predictor chains and keys. Streams go into a buffer.
- sleep (`sleep`): state merging (`dnn_merge.make_layout`) on the buffer gives
  new state neurons; a fresh network is trained on the buffer in one round;
  then `prune` deletes every neuron it can.

Safe deletion (`prune`). Neuron j may be deleted when every weight range
leaving it contains 0: L[i, j] <= 0 <= U[i, j] for every other neuron i. The
boxes are products of intervals, so setting all those weights to 0 gives a
point of every box. By the Theorem every trained sequence replays from its key
for every point of the boxes, and with those weights at 0 nothing depends on
j's values, so the sequences with column j removed replay on the network with
j removed. Deleting several such neurons at once is the same argument. Rows of
deleted neurons go too, which can make more neurons deletable, so `prune`
repeats until nothing changes. Task neurons and the always-on neuron are never
deleted. A fixed neuron is deletable only if no kept fixed neuron reads it
(their weights are points), so the kept fixed neurons still compute their
rules.

Behaviour on streams not trained can change: a deleted neuron's weights were
somewhere in their ranges, and are now exactly 0.
"""
import numpy as np

import dnn
import dnn_fixed as F
import dnn_merge as M


def deletable(net, protect, alive):
    """Neurons j (alive, not protected) with 0 in every range leaving j."""
    rows = [i for i in alive]
    out = []
    for j in alive:
        if j in protect:
            continue
        others = [i for i in rows if i != j]
        if all(net.L[i, j] <= 0 <= net.U[i, j] for i in others):
            out.append(j)
    return out


def prune(net, lay, seqs):
    """Delete neurons by the safe-deletion rule. Returns (net, layout, seqs,
    deleted fixed, deleted hidden); the inputs are left unchanged."""
    protect = set(range(lay.n_task)) | ({lay.on} if lay.on is not None else set())
    alive = list(range(net.n))
    while True:
        gone = deletable(net, protect, alive)
        if not gone:
            break
        alive = [j for j in alive if j not in gone]
    keep = alive
    if len(keep) == net.n:
        return net, lay, seqs, 0, 0
    new = {old: k for k, old in enumerate(keep)}
    nv_old = lay.n_visible
    kept_visible = [j for j in keep if j < nv_old]
    lay2 = F.Layout(lay.n_task, {new[i] for i in lay.inputs}, {new[i] for i in lay.outputs})
    lay2.names = [lay.names[j] for j in kept_visible]
    lay2.on = new[lay.on] if lay.on is not None else None
    lay2.inputs = {new[i] for i in lay.inputs if i in new}
    lay2._rows = {new[m]: {new[j]: w for j, w in row.items()} for m, row in lay._rows.items() if m in new}
    net2 = dnn.Net(len(kept_visible), {new[i] for i in net.I if i in new}, {new[i] for i in net.O if i in new})
    net2.L = net.L[np.ix_(keep, keep)].copy()
    net2.U = net.U[np.ix_(keep, keep)].copy()
    net2.c = [net.c[j] for j in keep]
    net2.n = len(keep)
    net2.round = net.round
    seqs2 = [np.asarray(s)[:, [j for j in keep if j < s.shape[1]]] for s in seqs]
    d_fixed = sum(1 for j in range(nv_old) if j not in new)
    return net2, lay2, seqs2, d_fixed, net.n - len(keep) - d_fixed


def sleep(buffer, out, eps=0.5, pruning=True):
    """Rebuild from the buffer. Returns (net, layout, trained, delta, m_out,
    deleted fixed, deleted hidden, unpruned (net, layout))."""
    lay, delta, m_out = M.make_layout(buffer, out)
    net = F.make_net(lay)
    trained = F.train(net, lay, buffer, eps)
    before = (net, lay)
    df = dh = 0
    if pruning:
        net, lay, trained, df, dh = prune(net, lay, trained)
    return net, lay, trained, delta, m_out, df, dh, before


# ---- experiment 18: which streams may leave a bounded buffer

def _same(m1, m2):
    return M.same_machine(*m1, *m2) and M.same_machine(*m2, *m1)


def evict(buffer, cap, n_in, policy, rng=None):
    """The streams kept when `buffer` (oldest first) must shrink to `cap`.

    "recent":  the cap most recent streams (experiment 15's sleep16).
    "random":  cap streams chosen at random, kept in order.
    "witness": streams are removed oldest first, but only while the machine
               merged from what is left equals the machine merged from the
               whole buffer (both partial machines give the same outputs
               wherever either is defined). Streams whose removal would
               change it are the witnesses of that machine: they ruled out a
               merge or showed a transition. If the buffer is still too big,
               the oldest witnesses go. Returns (kept, witnesses dropped);
               the second is None for the other policies."""
    if len(buffer) <= cap:
        return list(buffer), None
    if policy == "recent":
        return list(buffer[-cap:]), None
    if policy == "random":
        keep = sorted(rng.choice(len(buffer), cap, replace=False))
        return [buffer[i] for i in keep], None
    assert policy == "witness"
    ref = M.learn_machine(buffer, n_in)
    keep = list(range(len(buffer)))
    for i in range(len(buffer)):
        if len(keep) <= cap:
            break
        trial = [j for j in keep if j != i]
        if _same(M.learn_machine([buffer[j] for j in trial], n_in), ref):
            keep = trial
    dropped = max(len(keep) - cap, 0)
    return [buffer[j] for j in keep[-cap:]], dropped
