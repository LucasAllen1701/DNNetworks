# The box rule and the constraint buffer

TRAIN keeps one box per neuron. After a small round, that box can contain no
weight vector that fits the next round, and the neuron then stops learning:
later sequences are memorised through helper chains
(`experiments/RESULTS.md`, sections 8.2, 9.2 and 10.4). This note explains
why, shows that no rule for choosing the box can prevent it, and gives a
bounded amount of extra storage that does.

Lemma, Theorem and section numbers without a prefix refer to
`docs/Discrete_Neural_Network_Notes_Corrected.md` (the *notes*).

## 1 What goes wrong

For a neuron $g$, the weight vectors that fit a round form a polyhedron $P_g$
(section 4.1): one inequality $a \cdot x \ge 1$ for every state $x$ after
which $g$ is 1, and one inequality $a \cdot x \le 1 - \varepsilon$ for every
state after which it is 0. SUBBOX keeps a box inside $P_g$ and inside the box
kept before (section 4.2). The Theorem needs exactly that and nothing more,
so the choice among such boxes is free.

In the next round, FEAS asks whether the kept box contains a weight vector
that fits the new sequences. If the task is a threshold function of the
visible state, a correct weight vector exists and fits every round. But the
kept box need not contain one. If it contains none, FEAS fails, a helper
chain is added, and since boxes only shrink the neuron never recovers.

Two things make this likely after a small round. Both were found in the runs
of experiment 8.2 that went wrong (Tomita 4 with the start clock, 4 streams
per round).

1. **A round with only one kind of target.** If $g$ is 1 after every state of
   the round, every inequality is a lower bound. The widths are capped at 1,
   and the cap is reached wherever the lower endpoints are, because the upper
   endpoints are unbounded. So the lower endpoints are whatever the solver
   returns. In seed 1 no stream of round 1 is rejected, and the box has every
   weight $\ge 0$. The task needs a weight $\le -0.5$ from the oldest clock
   neuron, so every correct weight vector is excluded.
2. **Neurons the data cannot yet tell apart.** If two neurons have been on
   together in every state so far, the inequalities only constrain the sum of
   their weights. A box cannot keep a sum; it has to split it between the two,
   and the split is arbitrary. In seeds 3, 5 and 7 the task's large negative
   weight was put on the clock neuron one step back (about $-18$) and none on
   the one two steps back. The first state in which only one of the two is on
   then contradicts the whole box.

Both are the same difficulty: $P_g$ leaves a trade-off between weights open,
and a box must settle it.

## 2 No rule for choosing the box is safe

**Proposition 6.** Let $k > 1/\varepsilon$ (so $k = 3$ for the margin
$\varepsilon = 1/2$ used in the experiments). There are one round-1 sequence
and $k$ tasks, each a threshold function of the visible state and each
agreeing with that sequence, such that every box inside $P_g$ after round 1
excludes every correct weight vector of at least one of the tasks.

*Proof.* Take $k$ input neurons and one output neuron $g$. Task $i$ is
"$g$ copies input $i$": $g(t+1) = x_i(t)$. Round 1 shows the single state in
which all $k$ inputs are 1, followed by $g = 1$; every task agrees. So $P_g$
is $a_1 + \dots + a_k \ge 1$, and a box with lower endpoints $l_j$ lies inside
it only if

$$l_1 + \dots + l_k \ge 1. \tag{1}$$

Under task $i$, the state in which every input except $i$ is 1 is followed by
$g = 0$, so a correct weight vector has $\sum_{j \ne i} a_j \le 1 -
\varepsilon$. The box contains such a vector only if

$$\sum_{j \ne i} l_j \le 1 - \varepsilon. \tag{2}$$

If the box contained a correct vector for every task, adding (2) over
$i = 1, \dots, k$ would give $(k-1) \sum_j l_j \le k(1 - \varepsilon)$, and with
(1), $k - 1 \le k(1 - \varepsilon)$, that is $k \le 1/\varepsilon$. $\square$

So the problem is not the rule. It comes from keeping a box, which cannot
represent a set like $a_1 + a_2 + a_3 \ge 1$ without committing to how the 1
is shared out. A better rule can make the failure rarer, and
`dnn_box.py` has two:

- **bounded**: the widest box whose endpoints lie between $-M$ and $M$, with
  no cap on the widths.
- **volume**: the box of largest volume with endpoints between $-M$ and $M$
  (maximise the sum of the logarithms of the widths, by the Frank–Wolfe
  method). A volume is 0 as soon as one width is 0, so the room is spread
  over all the coordinates and an open trade-off is settled near the middle
  instead of at an extreme.

Both return a box inside $P_g$ and inside the previous box, so the Theorem
holds with either. Section 4 reports what they achieve.

## 3 The constraint buffer

The inequalities of $P_g$ are few when the data are few, and that is exactly
when choosing a box is dangerous. So keep them.

**Definition B1 (buffer).** A neuron $g$ with a buffer of capacity $K$ has a
box and a list $C_g$ of at most $K$ *stored constraints* $(x, y)$: an
activation pattern $x$ over the neurons that existed when it was stored, and
a target $y \in \{0, 1\}$. Its weight region is

$$F(g) = \text{box}(g) \cap \{a : a \cdot x \ge 1 \text{ if } y = 1,\ a \cdot x \le 1 - \varepsilon \text{ if } y = 0, \text{ for all } (x, y) \in C_g\},$$

where a pattern is extended by 0 for neurons added after it was stored.

**TRAIN with buffers.** FEAS($g$) tests $F(g)$, that is, the box together
with the stored constraints. In step 2, let $N_g$ be the distinct constraints
of $g$ in this round.

- If $|C_g \cup N_g| \le K$: set $C_g := C_g \cup N_g$ and leave the box as it
  is.
- Otherwise (*overflow*): choose a box inside $P_g$, inside the old box and
  inside every stored constraint, by any SUBBOX rule, and set
  $C_g := \emptyset$.

With $K = 0$ this is TRAIN of the notes. Hidden neurons need no buffer: their
boxes are fitted in their own round and never again.

**Proposition 7.** With buffers of any capacity:

1. The Theorem holds: learning, no forgetting and condition (D). The storage
   is $2n^2 + n$ numbers and, per buffered neuron, at most $K$ patterns of at
   most $n$ bits each and $K$ target bits.
2. If the buffer of $g$ has never overflowed, then at the start of every
   round $F(g)$ is exactly the set of weight vectors, in $g$'s initial box,
   that fit $g$ on every sequence trained so far. So a helper is added for
   $g$ only when no weight vector at all fits the old and the new sequences
   together.
3. In particular, if $g$'s next value is a threshold function of the visible
   state at margin $\varepsilon$, and the buffer never overflows, no helper
   is ever added for $g$, whatever the size and order of the rounds.

*Proof.* (1) The proof of the Theorem uses the new region $F'(g)$ only
through $F'(g) \subset P_g$ and $F'(g) \subset \tilde F(g)$, the old region
lifted to the new neurons. Both hold in both branches: in the first, $F'(g)$
is the old region cut by the round's inequalities, which are those of $P_g$;
in the second, the box is chosen inside all three sets. Extending a stored
pattern by 0 is correct because a hidden neuron added later is 0 on every
earlier sequence (Lemma 1, by condition (D)), so the inequality it stands for
is the one the old sequence imposes in the larger network. Hidden neurons are
handled as in the notes.

(2) By induction over rounds. Without overflow the box is never changed
except by lifting, and $C_g$ holds every distinct constraint of every round,
so $F(g)$ is the initial box cut by all of them. FEAS($g$) fails exactly when
this set, cut by the new round's inequalities, is empty.

(3) A weight vector that computes the threshold function fits every sequence
of the task, so it lies in $F(g)$ at every round by (2) and fits the new
round, and FEAS($g$) holds. $\square$

**How large must $K$ be?** $C_g$ holds one entry per distinct pair (visible
state, next value of $g$), however many sequences are trained. For a task
given by a machine with state neurons (`docs/Latency_And_State.md`) the
visible state is determined by the state neuron that is on and the current
input, so there are at most $(|Q|\,2^k + 1)\,2^k$ of them: 10 for parity. With
a library of 105 features there are more, because more combinations of
features occur. The experiment reports the numbers.

**Choosing weights.** A weight matrix must have row $g$ in $F(g)$, which is no
longer a box. `dnn_buffer.sample_weights` samples the box as before and then
moves row $g$ to the nearest point of $F(g)$ (smallest sum of absolute
differences, one linear program).

**What it costs.** Storage as in Proposition 7; one linear program per
buffered neuron to pick a weight vector; and FEAS has $K$ more rows. When the
buffer overflows, a box is chosen from more data than before, but
Proposition 6 still applies from then on.

## 4 What was measured

See `experiments/RESULTS.md`, section 11. In short: the alternative rules
reduce the number of runs that go wrong but do not remove them; a buffer large
enough not to overflow removes all of them, including when the streams are
trained one at a time.

## 5 Implementation and tests

`dnn_box.py` has the rules and `using(rule)`, which swaps a rule into
`dnn.train` for the duration of a `with` block. `dnn_buffer.py` has `Buffer`
and its `active()` block, which does the same for FEAS and SUBBOX, and
`sample_weights`. `dnn.py` is unchanged. `tests/test_dnn_box.py` checks:

- every rule keeps the Theorem on random networks;
- Proposition 6: the three inequalities have no solution, every rule goes
  wrong on one of the three tasks, and a buffer on none;
- a buffer of capacity 0 gives the boxes of `dnn.train`;
- Proposition 7(1) on random networks with small capacities that overflow;
- Proposition 7(3) on random threshold tasks trained one stream per round.
