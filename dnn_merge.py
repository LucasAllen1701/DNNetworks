"""Finding state neurons from the training streams, by state merging.

docs/Latency_And_State.md shows that a stream task is solved once the state
neurons of its machine are installed. This module finds a machine from the
training streams alone, with no knowledge of the task:

1. build the prefix tree of the streams: one node per input prefix seen, with
   the observed output on every edge. This is a machine that has memorised the
   streams;
2. merge nodes into states. Two nodes may be the same state unless the data
   contradict it, that is, unless some continuation seen from both gives
   different outputs. Nodes are taken in order of prefix length and merged
   into the first earlier state that fits (the RPNI order of Oncina and
   Garcia, 1992);
3. install the state neurons of the merged machine with
   `dnn_fixed.add_automaton` and train the output with `dnn.py`, unchanged.

See docs/State_Merging.md.
"""
from itertools import product

import numpy as np

import dnn_fixed as F


def observations(seq, n_in, h=2):
    """(input patterns, outputs) of one training sequence whose columns are
    the n_in inputs and then the output. outputs[t] is the output that
    answers inputs[0..t]; it appears h steps after inputs[t]."""
    s = np.asarray(seq, int)
    steps = len(s) - h
    return ([tuple(int(v) for v in s[t, :n_in]) for t in range(steps)],
            [int(s[t + h, n_in]) for t in range(steps)])


def prefix_tree(streams):
    """Prefix tree of (input patterns, outputs) pairs. Node 0 is the empty
    prefix; child[p][a] is the node reached from p by reading a and out[p][a]
    the output observed there. Raises ValueError if the same input prefix was
    seen with two different outputs."""
    child, out = [{}], [{}]
    for word, outputs in streams:
        node = 0
        for a, o in zip(word, outputs):
            if a not in child[node]:
                child[node][a] = len(child)
                out[node][a] = o
                child.append({})
                out.append({})
            elif out[node][a] != o:
                raise ValueError("the outputs are not a function of the input history")
            node = child[node][a]
    return child, out


def merge(child, out):
    """Merge the nodes of a prefix tree into states. Returns (delta, out) of
    the merged machine, as lists of dicts indexed by state; state 0 is the
    start state. Transitions that were never observed are absent."""
    trans, outs = [dict(c) for c in child], [dict(o) for o in out]
    rep = list(range(len(child)))
    order, queue = {0: 0}, [0]              # nodes by prefix length, then by pattern
    for node in queue:
        for a in sorted(child[node]):
            order[child[node][a]] = len(order)
            queue.append(child[node][a])
    states, is_state = [0], {0}

    def find(u):
        while rep[u] != u:
            u = rep[u]
        return u

    def try_merge(r, b):
        """Make r and b one state, and with them every pair of nodes that the
        same inputs lead to. Returns (False if two outputs disagree, undo log)."""
        log, work = [], [(r, b)]
        while work:
            u, v = work.pop()
            u, v = find(u), find(v)
            if u == v:
                continue
            if (v in is_state, -order[v]) > (u in is_state, -order[u]):
                u, v = v, u
            rep[v] = u
            log.append((v, None))
            for a, c in trans[v].items():
                if a in trans[u]:
                    if outs[u][a] != outs[v][a]:
                        return False, log
                    work.append((trans[u][a], c))
                else:
                    trans[u][a], outs[u][a] = c, outs[v][a]
                    log.append((u, a))
        return True, log

    def undo(log):
        for u, a in reversed(log):
            if a is None:
                rep[u] = u
            else:
                del trans[u][a], outs[u][a]

    while True:
        fringe = {find(c) for r in states for c in trans[r].values()} - is_state
        if not fringe:
            break
        b = min(fringe, key=order.get)
        for r in states:
            ok, log = try_merge(r, b)
            if ok:
                break
            undo(log)
        else:
            states.append(b)
            is_state.add(b)
    index = {r: i for i, r in enumerate(states)}
    return ([{a: index[find(c)] for a, c in trans[r].items()} for r in states],
            [dict(outs[r]) for r in states])


def learn_machine(seqs, n_in, h=2, merging=True):
    """The machine found from training sequences: (delta, out), partial. With
    merging=False the prefix tree itself is returned, one state per prefix."""
    child, out = prefix_tree([observations(s, n_in, h) for s in seqs])
    return merge(child, out) if merging else (child, out)


def complete(delta, n_in):
    """Give every state a transition for every input pattern. A transition
    that was never observed stays in its state; nothing is known about it."""
    return [{a: row.get(a, q) for a in product((0, 1), repeat=n_in)}
            for q, row in enumerate(delta)]


def make_layout(seqs, n_in, h=2, merging=True):
    """Layout with the always-on input and the state neurons of the machine
    found from `seqs`. Returns (layout, delta, out)."""
    delta, out = learn_machine(seqs, n_in, h, merging)
    lay = F.Layout(n_in + 1, range(n_in), {n_in})
    lay.add_on()
    F.add_automaton(lay, complete(delta, n_in))
    return lay, delta, out


def same_machine(delta, out, true_delta, true_out):
    """True if the found machine gives the true machine's output on every
    input stream. A missing transition counts as a difference."""
    seen, todo = {(0, 0)}, [(0, 0)]
    while todo:
        q, p = todo.pop()
        for a, nxt in true_delta[p].items():
            if a not in delta[q] or out[q][a] != true_out[p][a]:
                return False
            pair = (delta[q][a], nxt)
            if pair not in seen:
                seen.add(pair)
                todo.append(pair)
    return True
