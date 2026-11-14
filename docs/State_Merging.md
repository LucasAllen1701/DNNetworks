# Finding the state neurons: state merging

`docs/Latency_And_State.md` shows that a stream task is solved once the state
neurons of its machine are installed (Proposition 5), but it builds them from
the task's definition. This note finds them from the training streams alone.
Output timing is $h = 2$ throughout: the output that answers
$x(0), \dots, x(t)$ appears at time $t + 2$.

Lemma, Theorem and section numbers without a prefix refer to
`docs/Discrete_Neural_Network_Notes_Corrected.md` (the *notes*). Propositions
1–7 are in `docs/Fixed_Neurons.md`, `docs/Latency_And_State.md` and
`docs/Box_Rule.md`.

## 1 A state is its future

Two input histories put a machine in the same state exactly when every
continuation of the inputs gives the same outputs after both. So a state is
not something extra to be guessed: it is the function from future inputs to
future outputs.

A helper chain of the notes already stores a piece of that function. The
chain neurons of a stream hold its future outputs, $g(t+1), g(t+2), \dots$,
for the one continuation that the stream itself took. What the chain lacks is
any link between streams: two moments with the same future get two separate
chains, and each needs its key.

State merging supplies the link. It looks for moments whose observed futures
never disagree and makes them one state.

## 2 The algorithm

The data are the training streams. Each gives a word of input patterns
$a_0 a_1 \dots$ and the outputs $o_0 o_1 \dots$, where $o_t$ answers
$a_0 \dots a_t$. They must be a function of the input history: no input prefix
may appear with two different outputs.

**Prefix tree.** One node per input prefix that occurs, the empty prefix
being the root. Reading $a$ at node $p$ leads to node $pa$, and the edge
carries the output observed there. The tree is a machine with one state per
prefix. It gives the right output on every training stream and knows nothing
else: this is memorising.

**Merging.** A set of *states* starts with the root. Repeatedly take the
earliest node (shortest prefix first) that some state leads to and that is
not yet a state. Try to merge it into each state in turn. Merging two nodes
forces more merges: if both have an edge for the same input, the two edges
must carry the same output, and the nodes they lead to must be merged too.
If two outputs disagree, the merge is undone and the next state is tried. If
no state fits, the node becomes a new state. The result is a machine whose
states are the nodes that could not be merged into anything earlier.

This is the RPNI algorithm of Oncina and García (1992), in the form used for
machines that give an output at every step (Giantamidis and Tripakis, 2016).
`dnn_merge.merge` implements it.

**Network.** A transition that never occurred in the data is made to stay in
its state; nothing is known about it. The state neurons of the resulting
machine are installed before round 1 (Definition L3), and the output neuron
is trained by TRAIN, unchanged.

## 3 What is guaranteed

**Proposition 8.** Suppose the training streams are a function of the input
history. Let $M$ be their prefix tree, or the machine obtained from it by
merging.

1. $M$ gives the observed output at every step of every training stream.
2. With the state neurons of $M$ installed before round 1, the first round
   adds no hidden neuron. So every training stream is reproduced from its
   first state alone, without a key, for every weight matrix in the boxes.
3. In later rounds the Theorem holds: no stream is forgotten.

*Proof.* (1) For the tree this holds by construction. During merging, keep
the map $\varphi$ from tree nodes to the classes they have been merged into.
Claim: for every tree edge from $p$ to $p'$ with input $a$ and output $o$,
the class $\varphi(p)$ has an $a$-edge to $\varphi(p')$ with output $o$. This
is true at the start. A merge puts two classes together. If both have an
$a$-edge, either the outputs differ, and the whole merge is undone, or they
agree, the two target classes are put together as well, and one edge is
kept; otherwise the single edge is kept. In each case the claim still holds.
So the tree path of a training stream is carried to a path of $M$ with the
same outputs. Completing $M$ only adds edges that no training stream uses.

(2) Lemma S holds for any machine, so on a training stream the state neuron
that is on at time $t \ge 1$ is $n_{q,a}$ with $q$ the state of $M$ before
$x(t-1)$ and $a = x(t-1)$. Give the output neuron the weight 1 from every
$n_{q,a}$ whose edge carries the output 1, $y_0$ from `on`, $-y_0$ from
`started`, and 0 otherwise. By (1) the sum at time $t \ge 1$ is the observed
output $y(t+1)$, and at $t = 0$ it is $y_0 = y(1)$. The sums are integers, so
the margin is met and $P_y$ is non-empty. The output neuron's box is
unbounded in round 1, so FEAS holds and no helper is added. With no hidden
neuron the key of a stream is its first state (Corollary 2 of
`docs/Fixed_Neurons.md`).

(3) The state neurons are fixed neurons, so this is Proposition 1. $\square$

Proposition 8 holds however little data there is and however the merging
goes. With the tree, or with a wrong machine, the network has still learned
its training streams exactly; it then generalises badly or not at all.

**Getting the right machine.** For RPNI on automata that accept or reject, it
is known that the algorithm runs in polynomial time, and that every regular
language has a *characteristic sample* of polynomial size: any sample that
contains it and agrees with the language gives the smallest automaton of
that language (Oncina and García, 1992). The same has been shown for machines
with outputs (Giantamidis and Tripakis, 2016). We have not re-proved this for
the exact variant here. `tests/test_dnn_merge.py` checks it on random
machines with plenty of data, and experiment 13 measures how much data the 15
tasks need.

If the data fall short of a characteristic sample, the machine found can be
wrong in either direction: too few states (two real states were never seen to
disagree) or too many (a node was merged into the wrong state early, and its
relatives then fit nowhere).

## 4 Memorising and generalising

The prefix tree and the merged machine are the two ends of one construction.

| | prefix tree | merged machine |
|---|---|---|
| states | one per input prefix seen | one per distinguishable future |
| training streams | exact, without a key | exact, without a key |
| unseen streams | right only while they follow a seen prefix | right if the machine is the task's |
| size | grows with the data | fixed once the task's machine is found |

Merging is the step from one to the other, and the data decide how far it
goes: two moments stay one state until some stream shows that they differ.

Compared with the helper chains of the notes, the tree memorises from the
*past* (which inputs have arrived) where a chain memorises from the *future*
(which outputs are still to come). The past is available during a run, so no
key is needed.

## 5 Several rounds

The state neurons are installed before round 1, so later rounds are covered
by the Theorem. But the machine itself cannot be changed afterwards, for the
reason given in the remark after Proposition 3: a neuron added later fires on
old streams.

So the machine is settled by the streams available at the start. If they
are too few, the machine is wrong, later streams that contradict it are
memorised through helper chains, and unseen streams stay wrong. This is the
wrong-box problem of `docs/Box_Rule.md` again, one level up: a merge, like a
box, is a commitment made from the data seen so far, and it cannot be undone
once those data are gone. The same remedy applies. Keeping a bounded number
of early streams, and settling the machine only when that buffer is full,
postpones the commitment until there is enough to base it on. Experiment 13
reports how many streams that takes.

**What could be changed safely.** Two kinds of input history are certain not
to belong to any old stream: one that uses a transition never observed
before, and one that continues past a step where the current machine has just
been found wrong. New states reached only by such histories are silent on
every old stream, so adding them would not cause forgetting. This suggests a
way to let the machine grow in later rounds. It is not implemented.

## 6 Limits

- **Finite-state, input-driven tasks.** The outputs must be a function of the
  input history, and the task must have a finite machine.
- **Exact data.** One wrong output in the training streams is a disagreement
  like any other and produces extra states.
- **Enough data at the start** (section 5).
- **The number of state neurons** is $|Q|\,2^k + 1$ for $k$ input neurons,
  which grows quickly with $k$.
- **Worst case.** Finding the smallest machine that agrees with an arbitrary
  sample is NP-hard (Gold, 1978). Merging does not solve that problem: it
  finds some machine that agrees with the sample, and the smallest one when
  the sample is characteristic.

## 7 Implementation and tests

`dnn_merge.py`: `prefix_tree`, `merge`, `learn_machine`, `make_layout`
(which installs the state neurons with `dnn_fixed.add_automaton`) and
`same_machine`. `tests/test_dnn_merge.py` checks Proposition 8(1) and 8(2) on
random machines, for the tree and for the merged machine; that plenty of data
gives the smallest machine of the task; parity from 32 streams, tested on
longer streams; and that inconsistent data are rejected. Results are in
`experiments/RESULTS.md`, section 13.

## References

- E. M. Gold (1978). Complexity of automaton identification from given data.
  *Information and Control* 37.
- J. Oncina and P. García (1992). Inferring regular languages in polynomial
  updated time. In *Pattern Recognition and Image Analysis*, World Scientific.
- G. Giantamidis and S. Tripakis (2016). Learning Moore machines from
  input-output traces. *FM 2016*. arXiv:1605.07805.
